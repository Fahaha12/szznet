import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.SwinTransformer import SwinTransformer
from model.birads_classification_models import FCLayer, CrossModalAttentionBlock


class ModelSwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ModelSwinTransformer, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x_image, x_phenotype):
        x = self.backbone.patch_embed(x_image)

        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        # Dropout层
        x = self.backbone.pos_drop(x)

        # 开始逐层前向传播
        for layer in self.backbone.layers:
            x = layer(x)

        x = self.backbone.norm(x)  # B L C
        x = self.backbone.avgpool(x.transpose(1, 2))  # B C 1

        x = torch.flatten(x, 1)
        x = self.head(x)
        x = torch.log(F.softmax(x, 1))
        return x


class OnlyPhenotypicModel(nn.Module):
    def __init__(self, phenotypic_dim, num_classes=5, drop_rate=0.):
        super(OnlyPhenotypicModel, self).__init__()

        self.phenotypic_dim = phenotypic_dim
        self.fc_phenotypic_n_0 = FCLayer(phenotypic_dim, phenotypic_dim * 96, drop_rate=drop_rate)
        self.fc_phenotypic_n_1 = FCLayer(phenotypic_dim, phenotypic_dim * 2, drop_rate=drop_rate)
        self.fc_phenotypic_c_1 = FCLayer(96, 192, 'c', drop_rate=drop_rate)
        self.fc_phenotypic_n_2 = FCLayer(phenotypic_dim * 2, phenotypic_dim * 4, drop_rate=drop_rate)
        self.fc_phenotypic_c_2 = FCLayer(192, 384, 'c', drop_rate=drop_rate)
        self.fc_phenotypic_n_3 = FCLayer(phenotypic_dim * 4, phenotypic_dim * 8, drop_rate=drop_rate)
        self.fc_phenotypic_c_3 = FCLayer(384, 768, 'c', drop_rate=drop_rate)

        # 输出层
        self.norm_phenotypic = nn.LayerNorm(768)
        self.avgpool_phenotypic = nn.AdaptiveAvgPool1d(1)
        self.fc_out_phenotypic = nn.Linear(768, num_classes)

    def forward(self, x_image, x_phenotypic):
        x_phenotypic = self.fc_phenotypic_n_0(x_phenotypic)
        x_phenotypic = rearrange(x_phenotypic, 'b (n k) -> b k n', k=96)
        x_phenotypic = self.fc_phenotypic_n_1(x_phenotypic)
        x_phenotypic = self.fc_phenotypic_c_1(x_phenotypic)
        x_phenotypic = self.fc_phenotypic_n_2(x_phenotypic)
        x_phenotypic = self.fc_phenotypic_c_2(x_phenotypic)
        x_phenotypic = self.fc_phenotypic_n_3(x_phenotypic)
        x_phenotypic = self.fc_phenotypic_c_3(x_phenotypic)
        x_phenotypic = rearrange(x_phenotypic, 'b c n -> b n c')

        # out
        x_phenotypic = self.norm_phenotypic(x_phenotypic)
        x_phenotypic = self.avgpool_phenotypic(x_phenotypic.transpose(1, 2))
        x_phenotypic = torch.flatten(x_phenotypic, 1)
        phenotypic_out = self.fc_out_phenotypic(x_phenotypic)
        phenotypic_out = F.softmax(phenotypic_out, 1)
        out = torch.log(phenotypic_out)
        return out


class MultiModalModelSwinTBackbone(nn.Module):
    # image B, 49, 768   pyenotypic B, 5, 768   进行Attention时，dim = 768
    def __init__(self, phenotypic_dim, device='cpu', img_size=224, patch_size=4, class_num=4, window_size=7,
                 pretrain=True,
                 pretrain_path='', drop_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MultiModalModelSwinTBackbone, self).__init__()
        # image
        self.image_backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location=device)
            self.image_backbone.load_state_dict(checkpoint['model'])
            # del checkpoint
            # torch.cuda.empty_cache()

        # phenotypic
        self.phenotypic_dim = phenotypic_dim
        self.fc_phenotypic_n_0 = FCLayer(phenotypic_dim, phenotypic_dim * 96, act_layer=act_layer,
                                         norm_layer=nn.LayerNorm, drop_rate=drop_rate)
        self.fc_phenotypic_n_1 = FCLayer(phenotypic_dim, phenotypic_dim * 2, act_layer=act_layer,
                                         norm_layer=nn.LayerNorm, drop_rate=drop_rate)
        self.fc_phenotypic_c_1 = FCLayer(96, 192, 'c', act_layer=act_layer, norm_layer=nn.LayerNorm,
                                         drop_rate=drop_rate)
        self.fc_phenotypic_n_2 = FCLayer(phenotypic_dim * 2, phenotypic_dim * 4, act_layer=act_layer,
                                         norm_layer=nn.LayerNorm, drop_rate=drop_rate)
        self.fc_phenotypic_c_2 = FCLayer(192, 384, 'c', act_layer=act_layer, norm_layer=nn.LayerNorm,
                                         drop_rate=drop_rate)
        self.fc_phenotypic_n_3 = FCLayer(phenotypic_dim * 4, phenotypic_dim * 8, act_layer=act_layer,
                                         norm_layer=nn.LayerNorm, drop_rate=drop_rate)
        self.fc_phenotypic_c_3 = FCLayer(384, 768, 'c', act_layer=act_layer, norm_layer=nn.LayerNorm,
                                         drop_rate=drop_rate)

        # attention
        self.cross_modal_attention_block = CrossModalAttentionBlock(dim=768, num_heads=24, a_n=49,
                                                                    b_n=phenotypic_dim * 8, drop=drop_rate,
                                                                    attn_drop=drop_rate, drop_path=0.1,
                                                                    act_layer=act_layer, norm_layer=nn.LayerNorm)

        # out
        self.fc_out_image = nn.Linear(768, class_num)

        self.norm_phenotypic = norm_layer(768)
        self.avgpool_phenotypic = nn.AdaptiveAvgPool1d(1)
        self.fc_out_phenotypic = nn.Linear(768, class_num)

    def _forward_image_feature(self, x_image):
        x_image = self.image_backbone.patch_embed(x_image)
        if self.image_backbone.ape:
            x_image = x_image + self.image_backbone.absolute_pos_embed
        x_image = self.image_backbone.pos_drop(x_image)
        for i, layer in enumerate(self.image_backbone.layers):
            for blk in layer.blocks:
                if layer.use_checkpoint:
                    x_image = layer.checkpoint.checkpoint(blk, x_image)
                else:
                    x_image = blk(x_image)
            if layer.downsample is not None:
                x_image = layer.downsample(x_image)
        return x_image

    def forward(self, x_image, x_phenotype):
        # backbone
        x_image = self._forward_image_feature(x_image)

        x_phenotype = self.fc_phenotypic_n_0(x_phenotype)
        x_phenotype = rearrange(x_phenotype, 'b (n k) -> b k n', k=96)
        x_phenotype = self.fc_phenotypic_n_1(x_phenotype)
        x_phenotype = self.fc_phenotypic_c_1(x_phenotype)
        x_phenotype = self.fc_phenotypic_n_2(x_phenotype)
        x_phenotype = self.fc_phenotypic_c_2(x_phenotype)
        x_phenotype = self.fc_phenotypic_n_3(x_phenotype)
        x_phenotype = self.fc_phenotypic_c_3(x_phenotype)
        x_phenotype = rearrange(x_phenotype, 'b c n -> b n c')

        # attention
        x = [x_image, x_phenotype]
        x = self.cross_modal_attention_block(x)
        x_image, x_phenotype = x[0], x[1]

        # out
        x_image = self.image_backbone.norm(x_image)  # B L C
        x_image = self.image_backbone.avgpool(x_image.transpose(1, 2))  # B C 1
        x_image = torch.flatten(x_image, 1)
        image_out = self.fc_out_image(x_image)
        image_out = F.softmax(image_out, 1)

        x_phenotype = self.norm_phenotypic(x_phenotype)
        x_phenotype = self.avgpool_phenotypic(x_phenotype.transpose(1, 2))
        x_phenotype = torch.flatten(x_phenotype, 1)
        phenotypic_out = self.fc_out_phenotypic(x_phenotype)
        phenotypic_out = F.softmax(phenotypic_out, 1)

        out = torch.log((image_out + phenotypic_out) / 2)
        return out
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from torch.utils import model_zoo

from model.SwinTransformer import SwinTransformer, Mlp
from model.classification_models import FourViewAveragePool2d, FourViewResnet18BackboneAllShared, \
    FourViewResnet34BackboneAllShared, FourViewResnet50BackboneAllShared, OutputLayer, ViewWiseCVABlockSwinTransNoSwin, \
    FourViewOutputLayer, ViewWiseCVABlockSwinTransWithClassToken, ViewWiseCVABlockSwinTransForPhenotype

from model.resnet import Bottleneck, BasicBlock, model_urls, conv1x1
from utils.constants import VIEWS
import torch.nn.functional as F


# Image Based Models
class ResNetBackbone(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNetBackbone, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # if pretrained:
        #     self.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        # 第一个layer没有下采样过程，这个下采样是在每个layer第一个block对跳跃连接中的x用的
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)

        return x


class SVModelResnet(nn.Module):
    def __init__(self, pretrained=False, backbone='resnet50', class_num=2):
        super(SVModelResnet, self).__init__()
        if backbone == 'resnet50':
            self.block = Bottleneck
            self.resnet_backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
            if pretrained:
                self.resnet_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        elif backbone == 'resnet34':
            self.block = BasicBlock
            self.resnet_backbone = ResNetBackbone(BasicBlock, [3, 4, 6, 3])
            if pretrained:
                self.resnet_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        elif backbone == 'resnet18':
            self.block = BasicBlock
            self.resnet_backbone = ResNetBackbone(BasicBlock, [2, 2, 2, 2])
            if pretrained:
                self.resnet_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_out = nn.Linear(512 * self.block.expansion, class_num)

    def forward(self, x_image, x_phenotype, x_phenotype_case):
        x = self.resnet_backbone(x_image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        x = self.head(x)
        out = torch.log(F.softmax(x, 1))
        return out


class SVModelSwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(SVModelSwinTransformer, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x_image, x_phenotype, x_phenotype_case):
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


class ViewWiseResNet50LastStageConcat(nn.Module):
    def __init__(self, pretrained=False, backbone='resnet50', class_num=2):
        super(ViewWiseResNet50LastStageConcat, self).__init__()
        if backbone == 'resnet50':
            self.block = Bottleneck
            self.four_view_backbone = FourViewResnet50BackboneAllShared(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.block = BasicBlock
            self.four_view_backbone = FourViewResnet34BackboneAllShared(pretrained=pretrained)
        elif backbone == 'resnet18':
            self.block = BasicBlock
            self.four_view_backbone = FourViewResnet18BackboneAllShared(pretrained=pretrained)

        self.four_view_avg_pool = FourViewAveragePool2d()

        self.cc_fc_0 = nn.Linear(512 * self.block.expansion * 2, 512)
        self.cc_bn = nn.BatchNorm1d(512)
        self.cc_output_layer = OutputLayer(512, (2, class_num))

        self.mlo_fc_0 = nn.Linear(512 * self.block.expansion * 2, 512)
        self.mlo_bn = nn.BatchNorm1d(512)
        self.mlo_output_layer = OutputLayer(512, (2, class_num))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_image, x_phenotype, x_phenotype_case):
        x = self.four_view_backbone(x_image)
        x = self.four_view_avg_pool(x)
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_cc = torch.cat([x_l_cc, x_r_cc], dim=1)
        x_mlo = torch.cat([x_l_mlo, x_r_mlo], dim=1)
        x_cc = torch.flatten(x_cc, 1)
        x_mlo = torch.flatten(x_mlo, 1)

        x_cc = self.cc_fc_0(x_cc)
        x_cc = self.cc_bn(x_cc)
        x_cc = self.relu(x_cc)
        cc_output = self.cc_output_layer(x_cc)

        x_mlo = self.mlo_fc_0(x_mlo)
        x_mlo = self.mlo_bn(x_mlo)
        x_mlo = self.relu(x_mlo)
        mlo_output = self.mlo_output_layer(x_mlo)

        cc_output = F.softmax(cc_output, 2)
        mlo_output = F.softmax(mlo_output, 2)

        output = (cc_output + mlo_output) / 2
        output = torch.log(output)
        output_list = output.chunk(2, 1)
        m_l_output, m_r_output = output_list[0].squeeze(1), output_list[1].squeeze(1)
        return m_l_output, m_r_output


class ViewWiseSwinTransLastStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, device='cpu', img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True,
                 pretrain_path=''):
        super(ViewWiseSwinTransLastStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location=device)
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.four_stage_cva_blocks = ViewWiseCVABlockSwinTransNoSwin(dim=768, input_resolution=(7, 7), num_heads=24,
                                                                     window_size=window_size,
                                                                     drop_path=0.1)

        self.four_view_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)

    def _forward_pe_and_ape(self, x):
        for view in VIEWS.LIST:
            x[view] = self.backbone.patch_embed(x[view])
            if self.backbone.ape:
                x[view] = x[view] + self.backbone.absolute_pos_embed
            # Dropout层
            x[view] = self.backbone.pos_drop(x[view])
        return x

    def _forward_feature(self, x):
        for i, layer in enumerate(self.backbone.layers):
            for view in VIEWS.LIST:
                for blk in layer.blocks:
                    if layer.use_checkpoint:
                        x[view] = layer.checkpoint.checkpoint(blk, x[view])
                    else:
                        x[view] = blk(x[view])
                if layer.downsample is not None:
                    x[view] = layer.downsample(x[view])

        x = self.four_stage_cva_blocks(x)

        return x

    def forward(self, x, x_phenotype, x_phenotype_case):
        x = self._forward_pe_and_ape(x)
        # 开始逐层前向传播
        x = self._forward_feature(x)

        for view in VIEWS.LIST:
            x[view] = self.backbone.norm(x[view])  # B L C
            x[view] = self.backbone.avgpool(x[view].transpose(1, 2))  # B C 1
            x[view] = torch.flatten(x[view], 1)

        x = self.four_view_output_layer(x)

        l_cc_out, r_cc_out, l_mlo_out, r_mlo_out = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]
        l_cc_out = F.softmax(l_cc_out, 1)
        r_cc_out = F.softmax(r_cc_out, 1)
        l_mlo_out = F.softmax(l_mlo_out, 1)
        r_mlo_out = F.softmax(r_mlo_out, 1)

        l_output = torch.log((l_cc_out + l_mlo_out) / 2)
        r_output = torch.log((r_cc_out + r_mlo_out) / 2)

        return l_output, r_output


# Phenotypic Data Based Models
class OnlyPhenotypicModel0(nn.Module):
    def __init__(self, phenotypic_dim, num_classes=5, hidden_layer_dims=None, drop_rate=0.):
        super(OnlyPhenotypicModel0, self).__init__()
        if hidden_layer_dims is None:
            hidden_layer_dims = []
        self.dim = phenotypic_dim
        self.layers = []
        # 隐层
        for hidden_layer_dim in hidden_layer_dims:
            fc = nn.Linear(self.dim, hidden_layer_dim)
            self.dim = hidden_layer_dim
            drop_out = nn.Dropout(drop_rate)
            relu = nn.ReLU(inplace=True)
            bn = nn.LayerNorm(self.dim)

            self.layers.append(fc)
            self.layers.append(drop_out)
            self.layers.append(bn)
            self.layers.append(relu)

        # 输出层
        self.layers.append(nn.Linear(self.dim, num_classes))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x_image, x_phenotypic, x_phenotype_case):
        x = torch.cat([x_phenotypic, x_phenotype_case], 1)
        x = self.layers(x)
        x = torch.flatten(x, 1)
        output = F.softmax(x, 1)
        output = torch.log(output)
        return output


class OnlyPhenotypicModel1(nn.Module):
    def __init__(self, phenotypic_dim, num_classes=5, drop_rate=0.):
        super(OnlyPhenotypicModel1, self).__init__()

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

    def forward(self, x_image, x_phenotypic, x_phenotype_case):
        x_phenotypic = torch.cat([x_phenotypic, x_phenotype_case], 1)
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


class ViewWiseOnlyPhenotypicModel1LastStagesCVA(nn.Module):
    def __init__(self, phenotypic_dim, num_classes=5, window_size=7, drop_rate=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(ViewWiseOnlyPhenotypicModel1LastStagesCVA, self).__init__()

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

        # cross view attention
        self.four_stage_cva_blocks_phenotype = ViewWiseCVABlockSwinTransForPhenotype(dim=768,
                                                                                       input_resolution=phenotypic_dim * 8,
                                                                                       num_heads=24,
                                                                                       window_size=window_size,
                                                                                       drop_path=0.1)

        self.norm_phenotypic = norm_layer(768)
        self.avgpool_phenotypic = nn.AdaptiveAvgPool1d(1)
        self.four_view_phenotypic_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)

    def _forward_phenotype_feature(self, x_phenotype, x_phenotype_case):
        for view in VIEWS.LIST:
            x_phenotype[view] = torch.cat([x_phenotype[view], x_phenotype_case], 1)
            x_phenotype[view] = self.fc_phenotypic_n_0(x_phenotype[view])
            x_phenotype[view] = rearrange(x_phenotype[view], 'b (n k) -> b k n', k=96)
            x_phenotype[view] = self.fc_phenotypic_n_1(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_1(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_n_2(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_2(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_n_3(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_3(x_phenotype[view])
            x_phenotype[view] = rearrange(x_phenotype[view], 'b c n -> b n c')
        return x_phenotype

    def forward(self, x_image, x_phenotype, x_phenotype_case):
        # 开始逐层前向传播
        x_phenotype = self._forward_phenotype_feature(x_phenotype, x_phenotype_case)

        # cross view attention
        x_phenotype = self.four_stage_cva_blocks_phenotype(x_phenotype)

        for view in VIEWS.LIST:
            # norm, avg, flatten
            x_phenotype[view] = self.norm_phenotypic(x_phenotype[view])
            x_phenotype[view] = self.avgpool_phenotypic(x_phenotype[view].transpose(1, 2))
            x_phenotype[view] = torch.flatten(x_phenotype[view], 1)

        x_phenotype = self.four_view_phenotypic_output_layer(x_phenotype)

        l_cc_out_phenotype, r_cc_out_phenotype, l_mlo_out_phenotype, r_mlo_out_phenotype = \
            F.softmax(x_phenotype[VIEWS.L_CC], 1), F.softmax(x_phenotype[VIEWS.R_CC], 1), \
            F.softmax(x_phenotype[VIEWS.L_MLO], 1), F.softmax(x_phenotype[VIEWS.R_MLO], 1)
        l_output = torch.log((l_cc_out_phenotype + l_mlo_out_phenotype) / 2)
        r_output = torch.log((r_cc_out_phenotype + r_mlo_out_phenotype) / 2)

        return l_output, r_output


# MultiModal Models
### SingleView + MultiModal
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

    def forward(self, x_image, x_phenotype, x_phenotype_case):
        # backbone
        x_image = self._forward_image_feature(x_image)

        x_phenotype = torch.cat([x_phenotype, x_phenotype_case], 1)
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


### MultiView + MultiModal
class ViewWiseMultiModalSwinTransLastStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, phenotypic_dim, device='cpu', img_size=224, patch_size=4, num_classes=5, window_size=7,
                 pretrain=True,
                 pretrain_path='', drop_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(ViewWiseMultiModalSwinTransLastStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location=device)
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

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

        # cross modal attention
        self.cross_modal_attention_block = CrossModalAttentionBlock(dim=768, num_heads=24, a_n=49,
                                                                    b_n=phenotypic_dim * 8, drop=drop_rate,
                                                                    attn_drop=drop_rate, drop_path=0.1,
                                                                    act_layer=act_layer, norm_layer=nn.LayerNorm)

        # cross view attention
        self.four_stage_cva_blocks_image = ViewWiseCVABlockSwinTransNoSwin(dim=768, input_resolution=(7, 7),
                                                                           num_heads=24,
                                                                           window_size=window_size,
                                                                           drop_path=0.1)
        self.four_stage_cva_blocks_phenotype = ViewWiseCVABlockSwinTransNoSwin(dim=768,
                                                                               input_resolution=(phenotypic_dim * 8, 1),
                                                                               num_heads=24,
                                                                               window_size=window_size,
                                                                               drop_path=0.1)

        self.norm_phenotypic = norm_layer(768)
        self.avgpool_phenotypic = nn.AdaptiveAvgPool1d(1)
        self.four_view_image_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)
        self.four_view_phenotypic_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)

    def _forward_pe_and_ape(self, x):
        for view in VIEWS.LIST:
            x[view] = self.backbone.patch_embed(x[view])
            if self.backbone.ape:
                x[view] = x[view] + self.backbone.absolute_pos_embed
            # Dropout层
            x[view] = self.backbone.pos_drop(x[view])
        return x

    def _forward_image_feature(self, x):
        for i, layer in enumerate(self.backbone.layers):
            for view in VIEWS.LIST:
                for blk in layer.blocks:
                    if layer.use_checkpoint:
                        x[view] = layer.checkpoint.checkpoint(blk, x[view])
                    else:
                        x[view] = blk(x[view])
                if layer.downsample is not None:
                    x[view] = layer.downsample(x[view])

        # x = self.four_stage_cva_blocks_image(x)

        return x

    def _forward_phenotype_feature(self, x_phenotype, x_phenotype_case):
        for view in VIEWS.LIST:
            x_phenotype[view] = torch.cat([x_phenotype[view], x_phenotype_case], 1)
            x_phenotype[view] = self.fc_phenotypic_n_0(x_phenotype[view])
            x_phenotype[view] = rearrange(x_phenotype[view], 'b (n k) -> b k n', k=96)
            x_phenotype[view] = self.fc_phenotypic_n_1(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_1(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_n_2(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_2(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_n_3(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_3(x_phenotype[view])
            x_phenotype[view] = rearrange(x_phenotype[view], 'b c n -> b n c')
        return x_phenotype

    def forward(self, x_image, x_phenotype, x_phenotype_case):
        x_image = self._forward_pe_and_ape(x_image)
        # 开始逐层前向传播
        x_image = self._forward_image_feature(x_image)
        x_phenotype = self._forward_phenotype_feature(x_phenotype, x_phenotype_case)

        for view in VIEWS.LIST:
            # cross modal attention
            x = [x_image[view], x_phenotype[view]]
            x = self.cross_modal_attention_block(x)
            x_image[view], x_phenotype[view] = x[0], x[1]

        # cross view attention
        x_image = self.four_stage_cva_blocks_image(x_image)
        x_phenotype = self.four_stage_cva_blocks_phenotype(x_phenotype)

        for view in VIEWS.LIST:
            # norm, avg, flatten
            x_image[view] = self.backbone.norm(x_image[view])  # B L C
            x_image[view] = self.backbone.avgpool(x_image[view].transpose(1, 2))  # B C 1
            x_image[view] = torch.flatten(x_image[view], 1)
            x_phenotype[view] = self.norm_phenotypic(x_phenotype[view])
            x_phenotype[view] = self.avgpool_phenotypic(x_phenotype[view].transpose(1, 2))
            x_phenotype[view] = torch.flatten(x_phenotype[view], 1)

        x_image = self.four_view_image_output_layer(x_image)
        x_phenotype = self.four_view_phenotypic_output_layer(x_phenotype)

        l_cc_out_image, r_cc_out_image, l_mlo_out_image, r_mlo_out_image = \
            F.softmax(x_image[VIEWS.L_CC], 1), F.softmax(x_image[VIEWS.R_CC], 1), \
            F.softmax(x_image[VIEWS.L_MLO], 1), F.softmax(x_image[VIEWS.R_MLO], 1)
        l_output_image = torch.log((l_cc_out_image + l_mlo_out_image) / 2)
        r_output_image = torch.log((r_cc_out_image + r_mlo_out_image) / 2)

        l_cc_out_phenotype, r_cc_out_phenotype, l_mlo_out_phenotype, r_mlo_out_phenotype = \
            F.softmax(x_phenotype[VIEWS.L_CC], 1), F.softmax(x_phenotype[VIEWS.R_CC], 1), \
            F.softmax(x_phenotype[VIEWS.L_MLO], 1), F.softmax(x_phenotype[VIEWS.R_MLO], 1)
        l_output_phenotype = torch.log((l_cc_out_phenotype + l_mlo_out_phenotype) / 2)
        r_output_phenotype = torch.log((r_cc_out_phenotype + r_mlo_out_phenotype) / 2)

        l_output = (l_output_image + l_output_phenotype) / 2
        r_output = (r_output_image + r_output_phenotype) / 2

        return l_output, r_output


### MultiView + MultiModal + OnlyImage
class ViewWiseMultiModalOnlyImageInput(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, phenotypic_dim, device='cpu', img_size=224, patch_size=4, num_classes=5, window_size=7,
                 pretrain=True,
                 pretrain_path='', drop_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(ViewWiseMultiModalOnlyImageInput, self).__init__()
        self.device = device
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location=device)
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

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

        # cross modal attention
        self.cross_modal_attention_block = CrossModalAttentionBlock(dim=768, num_heads=24, a_n=49,
                                                                    b_n=phenotypic_dim * 8, drop=drop_rate,
                                                                    attn_drop=drop_rate, drop_path=0.1,
                                                                    act_layer=act_layer, norm_layer=nn.LayerNorm)

        # cross view attention
        self.four_stage_cva_blocks_image = ViewWiseCVABlockSwinTransNoSwin(dim=768, input_resolution=(7, 7),
                                                                           num_heads=24,
                                                                           window_size=window_size,
                                                                           drop_path=0.1)
        self.four_stage_cva_blocks_phenotype = ViewWiseCVABlockSwinTransNoSwin(dim=768,
                                                                               input_resolution=(phenotypic_dim * 8, 1),
                                                                               num_heads=24,
                                                                               window_size=window_size,
                                                                               drop_path=0.1)

        self.norm_phenotypic = norm_layer(768)
        self.avgpool_phenotypic = nn.AdaptiveAvgPool1d(1)
        self.four_view_shape_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=6)
        self.four_view_margin_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=6)
        self.four_view_subtlety_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=6)
        self.four_view_image_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)
        self.four_view_phenotypic_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)

    def _forward_pe_and_ape(self, x):
        for view in VIEWS.LIST:
            x[view] = self.backbone.patch_embed(x[view])
            if self.backbone.ape:
                x[view] = x[view] + self.backbone.absolute_pos_embed
            # Dropout层
            x[view] = self.backbone.pos_drop(x[view])
        return x

    def _forward_image_feature(self, x):
        for i, layer in enumerate(self.backbone.layers):
            for view in VIEWS.LIST:
                for blk in layer.blocks:
                    if layer.use_checkpoint:
                        x[view] = layer.checkpoint.checkpoint(blk, x[view])
                    else:
                        x[view] = blk(x[view])
                if layer.downsample is not None:
                    x[view] = layer.downsample(x[view])

        # x = self.four_stage_cva_blocks_image(x)

        return x

    def _forward_phenotype_feature(self, x_phenotype, x_phenotype_case):
        for view in VIEWS.LIST:
            x_phenotype[view] = torch.cat([x_phenotype[view], x_phenotype_case], 1)
            x_phenotype[view] = self.fc_phenotypic_n_0(x_phenotype[view])
            x_phenotype[view] = rearrange(x_phenotype[view], 'b (n k) -> b k n', k=96)
            x_phenotype[view] = self.fc_phenotypic_n_1(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_1(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_n_2(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_2(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_n_3(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_3(x_phenotype[view])
            x_phenotype[view] = rearrange(x_phenotype[view], 'b c n -> b n c')
        return x_phenotype

    def forward(self, x_image_ori, x_phenotype_case):
        x_image = {view: [] for view in VIEWS.LIST}
        for view in VIEWS.LIST:
            x_image[view] = x_image_ori[view]

        x_image = self._forward_pe_and_ape(x_image)
        # 开始逐层前向传播
        x_image = self._forward_image_feature(x_image)
        x_image = self.four_stage_cva_blocks_image(x_image)

        x_images_copy = {view: [] for view in VIEWS.LIST}
        for view in VIEWS.LIST:
            # norm, avg, flatten
            x_images_copy[view] = self.backbone.norm(x_image[view])  # B L C
            x_images_copy[view] = self.backbone.avgpool(x_images_copy[view].transpose(1, 2))  # B C 1
            x_images_copy[view] = torch.flatten(x_images_copy[view], 1)
        shape_out = self.four_view_shape_output_layer(x_images_copy)
        margin_out = self.four_view_margin_output_layer(x_images_copy)
        subtlety_out = self.four_view_subtlety_output_layer(x_images_copy)

        l_cc_out_shape, r_cc_out_shape, l_mlo_out_shape, r_mlo_out_shape = \
            F.softmax(shape_out[VIEWS.L_CC], 1), F.softmax(shape_out[VIEWS.R_CC], 1), \
            F.softmax(shape_out[VIEWS.L_MLO], 1), F.softmax(shape_out[VIEWS.R_MLO], 1)
        l_output_shape = torch.log((l_cc_out_shape + l_mlo_out_shape) / 2)
        r_output_shape = torch.log((r_cc_out_shape + r_mlo_out_shape) / 2)
        _, l_output_shape_max = l_output_shape.topk(1, 1)
        _, r_output_shape_max = r_output_shape.topk(1, 1)

        l_cc_out_margin, r_cc_out_margin, l_mlo_out_margin, r_mlo_out_margin = \
            F.softmax(margin_out[VIEWS.L_CC], 1), F.softmax(margin_out[VIEWS.R_CC], 1), \
            F.softmax(margin_out[VIEWS.L_MLO], 1), F.softmax(margin_out[VIEWS.R_MLO], 1)
        l_output_margin = torch.log((l_cc_out_margin + l_mlo_out_margin) / 2)
        r_output_margin = torch.log((r_cc_out_margin + r_mlo_out_margin) / 2)
        _, l_output_margin_max = l_output_margin.topk(1, 1)
        _, r_output_margin_max = r_output_margin.topk(1, 1)

        l_cc_out_subtlety, r_cc_out_subtlety, l_mlo_out_subtlety, r_mlo_out_subtlety = \
            F.softmax(subtlety_out[VIEWS.L_CC], 1), F.softmax(subtlety_out[VIEWS.R_CC], 1), \
            F.softmax(subtlety_out[VIEWS.L_MLO], 1), F.softmax(subtlety_out[VIEWS.R_MLO], 1)
        l_output_subtlety = torch.log((l_cc_out_subtlety + l_mlo_out_subtlety) / 2)
        r_output_subtlety = torch.log((r_cc_out_subtlety + r_mlo_out_subtlety) / 2)
        _, l_output_subtlety_max = l_output_subtlety.topk(1, 1)
        _, r_output_subtlety_max = r_output_subtlety.topk(1, 1)

        x_phenotype = {view: [] for view in VIEWS.LIST}
        x_phenotype[VIEWS.L_CC] = torch.cat([l_output_shape_max, l_output_margin_max, l_output_subtlety_max],
                                            1).type(torch.FloatTensor).to(self.device)
        x_phenotype[VIEWS.L_MLO] = torch.cat([l_output_shape_max, l_output_margin_max, l_output_subtlety_max],
                                             1).type(torch.FloatTensor).to(self.device)
        x_phenotype[VIEWS.R_CC] = torch.cat([r_output_shape_max, r_output_margin_max, r_output_subtlety_max],
                                            1).type(torch.FloatTensor).to(self.device)
        x_phenotype[VIEWS.R_MLO] = torch.cat([r_output_shape_max, r_output_margin_max, r_output_subtlety_max],
                                             1).type(torch.FloatTensor).to(self.device)

        x_phenotype = self._forward_phenotype_feature(x_phenotype, x_phenotype_case)

        # cross view attention
        x_phenotype = self.four_stage_cva_blocks_phenotype(x_phenotype)

        for view in VIEWS.LIST:
            # cross modal attention
            x = [x_image[view], x_phenotype[view]]
            x = self.cross_modal_attention_block(x)
            x_image[view], x_phenotype[view] = x[0], x[1]

        for view in VIEWS.LIST:
            # norm, avg, flatten
            x_image[view] = self.backbone.norm(x_image[view])  # B L C
            x_image[view] = self.backbone.avgpool(x_image[view].transpose(1, 2))  # B C 1
            x_image[view] = torch.flatten(x_image[view], 1)
            x_phenotype[view] = self.norm_phenotypic(x_phenotype[view])
            x_phenotype[view] = self.avgpool_phenotypic(x_phenotype[view].transpose(1, 2))
            x_phenotype[view] = torch.flatten(x_phenotype[view], 1)

        x_image = self.four_view_image_output_layer(x_image)
        x_phenotype = self.four_view_phenotypic_output_layer(x_phenotype)

        l_cc_out_image, r_cc_out_image, l_mlo_out_image, r_mlo_out_image = \
            F.softmax(x_image[VIEWS.L_CC], 1), F.softmax(x_image[VIEWS.R_CC], 1), \
            F.softmax(x_image[VIEWS.L_MLO], 1), F.softmax(x_image[VIEWS.R_MLO], 1)
        l_output_image = torch.log((l_cc_out_image + l_mlo_out_image) / 2)
        r_output_image = torch.log((r_cc_out_image + r_mlo_out_image) / 2)

        l_cc_out_phenotype, r_cc_out_phenotype, l_mlo_out_phenotype, r_mlo_out_phenotype = \
            F.softmax(x_phenotype[VIEWS.L_CC], 1), F.softmax(x_phenotype[VIEWS.R_CC], 1), \
            F.softmax(x_phenotype[VIEWS.L_MLO], 1), F.softmax(x_phenotype[VIEWS.R_MLO], 1)
        l_output_phenotype = torch.log((l_cc_out_phenotype + l_mlo_out_phenotype) / 2)
        r_output_phenotype = torch.log((r_cc_out_phenotype + r_mlo_out_phenotype) / 2)

        l_output = (l_output_image + l_output_phenotype) / 2
        r_output = (r_output_image + r_output_phenotype) / 2

        return l_output, r_output, l_output_shape, r_output_shape, \
               l_output_margin, r_output_margin, l_output_subtlety, r_output_subtlety


class ViewWiseMultiModalMultiViewFirst(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, phenotypic_dim, device='cpu', img_size=224, patch_size=4, num_classes=5, window_size=7,
                 pretrain=True,
                 pretrain_path='', drop_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(ViewWiseMultiModalMultiViewFirst, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location=device)
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

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

        # cross modal attention
        self.cross_modal_attention_block = CrossModalAttentionBlock(dim=768, num_heads=24, a_n=49,
                                                                    b_n=phenotypic_dim * 8, drop=drop_rate,
                                                                    attn_drop=drop_rate, drop_path=0.1,
                                                                    act_layer=act_layer, norm_layer=nn.LayerNorm)

        # cross view attention
        self.four_stage_cva_blocks_image = ViewWiseCVABlockSwinTransNoSwin(dim=768, input_resolution=(7, 7),
                                                                           num_heads=24,
                                                                           window_size=window_size,
                                                                           drop_path=0.1)
        self.four_stage_cva_blocks_phenotype = ViewWiseCVABlockSwinTransNoSwin(dim=768,
                                                                               input_resolution=(phenotypic_dim * 8, 1),
                                                                               num_heads=24,
                                                                               window_size=window_size,
                                                                               drop_path=0.1)

        self.norm_phenotypic = norm_layer(768)
        self.avgpool_phenotypic = nn.AdaptiveAvgPool1d(1)
        self.four_view_image_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)
        self.four_view_phenotypic_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)

    def _forward_pe_and_ape(self, x):
        for view in VIEWS.LIST:
            x[view] = self.backbone.patch_embed(x[view])
            if self.backbone.ape:
                x[view] = x[view] + self.backbone.absolute_pos_embed
            # Dropout层
            x[view] = self.backbone.pos_drop(x[view])
        return x

    def _forward_image_feature(self, x):
        for i, layer in enumerate(self.backbone.layers):
            for view in VIEWS.LIST:
                for blk in layer.blocks:
                    if layer.use_checkpoint:
                        x[view] = layer.checkpoint.checkpoint(blk, x[view])
                    else:
                        x[view] = blk(x[view])
                if layer.downsample is not None:
                    x[view] = layer.downsample(x[view])

        # x = self.four_stage_cva_blocks_image(x)

        return x

    def _forward_phenotype_feature(self, x_phenotype, x_phenotype_case):
        for view in VIEWS.LIST:
            x_phenotype[view] = torch.cat([x_phenotype[view], x_phenotype_case], 1)
            x_phenotype[view] = self.fc_phenotypic_n_0(x_phenotype[view])
            x_phenotype[view] = rearrange(x_phenotype[view], 'b (n k) -> b k n', k=96)
            x_phenotype[view] = self.fc_phenotypic_n_1(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_1(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_n_2(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_2(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_n_3(x_phenotype[view])
            x_phenotype[view] = self.fc_phenotypic_c_3(x_phenotype[view])
            x_phenotype[view] = rearrange(x_phenotype[view], 'b c n -> b n c')
        return x_phenotype

    def forward(self, x_image, x_phenotype, x_phenotype_case):
        x_image = self._forward_pe_and_ape(x_image)
        # 开始逐层前向传播
        x_image = self._forward_image_feature(x_image)
        x_phenotype = self._forward_phenotype_feature(x_phenotype, x_phenotype_case)

        # cross view attention
        x_image = self.four_stage_cva_blocks_image(x_image)
        x_phenotype = self.four_stage_cva_blocks_phenotype(x_phenotype)

        for view in VIEWS.LIST:
            # cross modal attention
            x = [x_image[view], x_phenotype[view]]
            x = self.cross_modal_attention_block(x)
            x_image[view], x_phenotype[view] = x[0], x[1]

        for view in VIEWS.LIST:
            # norm, avg, flatten
            x_image[view] = self.backbone.norm(x_image[view])  # B L C
            x_image[view] = self.backbone.avgpool(x_image[view].transpose(1, 2))  # B C 1
            x_image[view] = torch.flatten(x_image[view], 1)
            x_phenotype[view] = self.norm_phenotypic(x_phenotype[view])
            x_phenotype[view] = self.avgpool_phenotypic(x_phenotype[view].transpose(1, 2))
            x_phenotype[view] = torch.flatten(x_phenotype[view], 1)

        x_image = self.four_view_image_output_layer(x_image)
        x_phenotype = self.four_view_phenotypic_output_layer(x_phenotype)

        l_cc_out_image, r_cc_out_image, l_mlo_out_image, r_mlo_out_image = \
            F.softmax(x_image[VIEWS.L_CC], 1), F.softmax(x_image[VIEWS.R_CC], 1), \
            F.softmax(x_image[VIEWS.L_MLO], 1), F.softmax(x_image[VIEWS.R_MLO], 1)
        l_output_image = torch.log((l_cc_out_image + l_mlo_out_image) / 2)
        r_output_image = torch.log((r_cc_out_image + r_mlo_out_image) / 2)

        l_cc_out_phenotype, r_cc_out_phenotype, l_mlo_out_phenotype, r_mlo_out_phenotype = \
            F.softmax(x_phenotype[VIEWS.L_CC], 1), F.softmax(x_phenotype[VIEWS.R_CC], 1), \
            F.softmax(x_phenotype[VIEWS.L_MLO], 1), F.softmax(x_phenotype[VIEWS.R_MLO], 1)
        l_output_phenotype = torch.log((l_cc_out_phenotype + l_mlo_out_phenotype) / 2)
        r_output_phenotype = torch.log((r_cc_out_phenotype + r_mlo_out_phenotype) / 2)

        l_output = (l_output_image + l_output_phenotype) / 2
        r_output = (r_output_image + r_output_phenotype) / 2

        return l_output, r_output


class FCLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_or_c='n', act_layer=nn.GELU, norm_layer=nn.LayerNorm, drop_rate=0.):
        super().__init__()
        if n_or_c == 'n':
            self.t = False
        else:
            self.t = True

        self.fc = nn.Linear(in_dim, out_dim)
        self.drop_out = nn.Dropout(drop_rate)
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        if self.t:
            x = rearrange(x, 'b c n -> b n c')
        x = self.fc(x)
        x = self.drop_out(x)
        x = self.norm(x)
        x = self.act(x)
        if self.t:
            x = rearrange(x, 'b n c -> b c n')
        return x


# Cross Modal Attention
class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads, a_n, b_n, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.a_pos_embedding = nn.Parameter(torch.randn(1, a_n, dim))
        # self.b_pos_embedding = nn.Parameter(torch.randn(1, b_n, dim))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_a, x_b):
        B_a, N_a, C_a = x_a.shape
        # x_a = x_a + self.a_pos_embedding
        q = self.q(x_a).reshape(B_a, N_a, self.num_heads, C_a // self.num_heads).permute(0, 2, 1, 3)

        B_b, N_b, C_b = x_b.shape
        # x_b = x_b + self.b_pos_embedding
        kv = self.kv(x_b).reshape(B_b, N_b, 2, self.num_heads, C_b // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn

        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_a, N_a, C_a)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossModalAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, a_n, b_n, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.a_norm1 = norm_layer(dim)
        self.b_norm1 = norm_layer(dim)
        self.b2a_cva = CrossModalAttention(dim, num_heads=num_heads, a_n=a_n, b_n=b_n, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.a2b_cva = CrossModalAttention(dim, num_heads=num_heads, a_n=b_n, b_n=a_n, qkv_bias=qkv_bias,
                                           qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.a_norm2 = norm_layer(dim)
        self.b_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.a_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.b_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        a, b = x[0], x[1]

        shortcut_a = a
        shortcut_b = b
        a = self.a_norm1(a)
        b = self.b_norm1(b)

        a_windows = a
        b_windows = b

        a = self.b2a_cva(a_windows, b_windows)  # nW*B, window_size*window_size, C
        b = self.a2b_cva(b_windows, a_windows)  # nW*B, window_size*window_size, C

        # FFN
        a = shortcut_a + self.drop_path(a)
        a = a + self.drop_path(self.a_mlp(self.a_norm2(a)))
        b = shortcut_b + self.drop_path(b)
        b = b + self.drop_path(self.b_mlp(self.b_norm2(b)))

        x_out = [a, b]

        return x_out

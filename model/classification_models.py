import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from torch.utils import model_zoo

from model.SwinTransformer import SwinTransformer, CrossViewSwinTransformerBlock, CVSTBWithClassificationToken, \
    SwinTransformerBlockWithClassToken, CVSTBForPhenotype
from model.resnet import Bottleneck, BasicBlock, model_urls, conv1x1
from utils.constants import VIEWS
import numpy as np
import torch.nn.functional as F


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

    def forward(self, x_image):
        x = self.resnet_backbone(x_image)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc_out(x)
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

    def forward(self, x_image):
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
        return x


# ViewWise Models
### Concat
class ViewWiseResnet50LastStageConcat(nn.Module):
    def __init__(self, pretrained=False, backbone='resnet50', class_num=2):
        super(ViewWiseResnet50LastStageConcat, self).__init__()
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

    def forward(self, x_image):
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


class ViewWiseSwinTransLastStagesConcat(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ViewWiseSwinTransLastStagesConcat, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        # self.four_stage_cva_blocks = ViewWiseCVABlockSwinTransNoSwin(dim=768, input_resolution=(7, 7), num_heads=24,
        #                                                              window_size=window_size,
        #                                                              drop_path=0.1)

        # self.four_view_output_layer = FourViewOutputLayer(feature_channels=768, output_shape=num_classes)

        self.cc_fc_0 = nn.Linear(768 * 2, 768)
        self.cc_bn = nn.BatchNorm1d(768)
        self.cc_output_layer = OutputLayer(768, (2, 2))

        self.mlo_fc_0 = nn.Linear(768 * 2, 768)
        self.mlo_bn = nn.BatchNorm1d(768)
        self.mlo_output_layer = OutputLayer(768, (2, 2))

        self.relu = nn.ReLU(inplace=True)

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

        # x = self.four_stage_cva_blocks(x)

        return x

    def forward(self, x):
        x = self._forward_pe_and_ape(x)
        # 开始逐层前向传播
        x = self._forward_feature(x)

        for view in VIEWS.LIST:
            x[view] = self.backbone.norm(x[view])  # B L C
            x[view] = self.backbone.avgpool(x[view].transpose(1, 2))  # B C 1
            x[view] = torch.flatten(x[view], 1)

        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_cc = torch.cat([x_l_cc, x_r_cc], dim=1)
        x_mlo = torch.cat([x_l_mlo, x_r_mlo], dim=1)

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


### CVA
class ViewWiseResnet50LastStagesCVA(nn.Module):
    # 在resnet50的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, num_classes=2, window_size=7, pretrain=True):
        super(ViewWiseResnet50LastStagesCVA, self).__init__()
        self.resnet_backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
        if pretrain:
            self.resnet_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        self.four_stage_cva_blocks = ViewWiseCVABlockSwinTransNoSwin(dim=2048, input_resolution=(7, 7), num_heads=32,
                                                                     window_size=window_size,
                                                                     drop_path=0.1)

        self.norm = nn.LayerNorm(2048)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.four_view_output_layer = FourViewOutputLayer(feature_channels=2048, output_shape=num_classes)

    def forward(self, x):
        # 开始逐层前向传播
        for view in VIEWS.LIST:
            x[view] = self.resnet_backbone(x[view])
            x[view] = rearrange(x[view], 'b c h w -> b (h w) c', h=7, w=7)

        x = self.four_stage_cva_blocks(x)

        for view in VIEWS.LIST:
            x[view] = self.norm(x[view])  # B L C
            x[view] = self.avgpool(x[view].transpose(1, 2))  # B C 1
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


class ViewWiseSwinTransAllStagesCVA(nn.Module):
    # 在swin transformer的每个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ViewWiseSwinTransAllStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.cva_blocks = nn.ModuleList()

        self.one_stage_cva_blocks = ViewWiseCVABlockSwinTrans(dim=96, input_resolution=(56, 56), num_heads=3,
                                                              window_size=window_size,
                                                              drop_path=self.backbone.layers[0].drop_path)
        self.two_stage_cva_blocks = ViewWiseCVABlockSwinTrans(dim=192, input_resolution=(28, 28), num_heads=6,
                                                              window_size=window_size,
                                                              drop_path=self.backbone.layers[1].drop_path)
        self.three_stage_cva_blocks = ViewWiseCVABlockSwinTrans(dim=384, input_resolution=(14, 14), num_heads=12,
                                                                window_size=window_size,
                                                                drop_path=self.backbone.layers[2].drop_path)
        self.four_stage_cva_blocks = ViewWiseCVABlockSwinTrans(dim=768, input_resolution=(7, 7), num_heads=24,
                                                               window_size=window_size,
                                                               drop_path=self.backbone.layers[3].drop_path)
        self.cva_blocks.append(self.one_stage_cva_blocks)
        self.cva_blocks.append(self.two_stage_cva_blocks)
        self.cva_blocks.append(self.three_stage_cva_blocks)
        self.cva_blocks.append(self.four_stage_cva_blocks)

        # self.head = nn.Linear(self.backbone.num_features, num_classes)
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

            x = self.cva_blocks[i](x)

            # if not i < 2:
            #     x = self.cva_blocks[i - 2](x)

            if layer.downsample is not None:
                for view in VIEWS.LIST:
                    x[view] = layer.downsample(x[view])

        return x

    def forward(self, x):
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


class ViewWiseSwinTransLastStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ViewWiseSwinTransLastStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
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

    def forward(self, x):
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


class ViewWiseSwinTransThreeStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ViewWiseSwinTransThreeStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.Three_stage_cva_blocks = ViewWiseCVABlockSwinTrans(dim=384, input_resolution=(14, 14), num_heads=12,
                                                                window_size=window_size,
                                                                drop_path=self.backbone.layers[2].drop_path)
        # self.Three_stage_cva_blocks = ViewWiseCVABlockSwinTransNoSwin(dim=384, input_resolution=(14, 14), num_heads=12,
        #                                                               window_size=14,
        #                                                               drop_path=self.backbone.layers[2].drop_path)

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

            if i == 2:
                x = self.Three_stage_cva_blocks(x)

            if layer.downsample is not None:
                for view in VIEWS.LIST:
                    x[view] = layer.downsample(x[view])
        return x

    def forward(self, x):
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


class ViewWiseSwinTransTwoStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ViewWiseSwinTransTwoStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.Two_stage_cva_blocks = ViewWiseCVABlockSwinTrans(dim=192, input_resolution=(28, 28), num_heads=6,
                                                              window_size=window_size,
                                                              drop_path=self.backbone.layers[1].drop_path)
        # self.Two_stage_cva_blocks = ViewWiseCVABlockSwinTransNoSwin(dim=192, input_resolution=(28, 28), num_heads=6,
        #                                                             window_size=28,
        #                                                             drop_path=self.backbone.layers[1].drop_path)

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

            if i == 1:
                x = self.Two_stage_cva_blocks(x)

            if layer.downsample is not None:
                for view in VIEWS.LIST:
                    x[view] = layer.downsample(x[view])
        return x

    def forward(self, x):
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


class ViewWiseSwinTransOneStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ViewWiseSwinTransOneStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.One_stage_cva_blocks = ViewWiseCVABlockSwinTrans(dim=96, input_resolution=(56, 56), num_heads=3,
                                                              window_size=window_size,
                                                              drop_path=self.backbone.layers[0].drop_path)

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

            if i == 0:
                x = self.One_stage_cva_blocks(x)

            if layer.downsample is not None:
                for view in VIEWS.LIST:
                    x[view] = layer.downsample(x[view])
        return x

    def forward(self, x):
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


class ViewWiseSwinTransBeforeStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ViewWiseSwinTransBeforeStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.Before_stage_cva_blocks = ViewWiseCVABlockSwinTrans(dim=96, input_resolution=(56, 56), num_heads=3,
                                                                 window_size=window_size,
                                                                 drop_path=self.backbone.layers[0].drop_path)

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

        x = self.Before_stage_cva_blocks(x)

        for i, layer in enumerate(self.backbone.layers):
            for view in VIEWS.LIST:
                for blk in layer.blocks:
                    if layer.use_checkpoint:
                        x[view] = layer.checkpoint.checkpoint(blk, x[view])
                    else:
                        x[view] = blk(x[view])

                if layer.downsample is not None:
                    x[view] = layer.downsample(x[view])
        return x

    def forward(self, x):
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


class ViewWiseSwinTransLastStagesCVAWithClassToken(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(ViewWiseSwinTransLastStagesCVAWithClassToken, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.class_token = {view: nn.Parameter(torch.randn(1, 1, 768)) for view in VIEWS.LIST}

        self.self_attention_block0 = SwinTransformerBlockWithClassToken(dim=768, input_resolution=(7, 7),
                                                                        num_heads=24,
                                                                        window_size=window_size,
                                                                        drop_path=0.1)
        # self.self_attention_block1 = SwinTransformerBlockWithClassToken(dim=768, input_resolution=(7, 7),
        #                                                                 num_heads=24,
        #                                                                 window_size=window_size,
        #                                                                 drop_path=0.1)

        self.four_stage_cva_blocks = ViewWiseCVABlockSwinTransWithClassToken(dim=768, input_resolution=(7, 7),
                                                                             num_heads=24,
                                                                             window_size=window_size,
                                                                             drop_path=0.1)
        self.norm = nn.LayerNorm(768)

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

                if i == 3:
                    B, L, C = x[view].shape
                    x[view] = torch.cat((self.class_token[view].repeat(B, 1, 1), x[view]), 1)
                    x[view] = self.self_attention_block0(x[view])
                    # x[view] = self.self_attention_block1(x[view])

        x = self.four_stage_cva_blocks(x)

        return x

    def forward(self, x):
        for view in VIEWS.LIST:
            self.class_token[view] = self.class_token[view].to(x[view].device)
        x = self._forward_pe_and_ape(x)
        # 开始逐层前向传播
        x = self._forward_feature(x)

        for view in VIEWS.LIST:
            # x[view] = self.backbone.norm(x[view])  # B L C
            # x[view] = self.backbone.avgpool(x[view].transpose(1, 2))  # B C 1

            x[view] = x[view].transpose(1, 2)[:, :, 0]  # B C 1
            x[view] = torch.flatten(x[view], 1)
            x[view] = self.norm(x[view])

        x = self.four_view_output_layer(x)

        l_cc_out, r_cc_out, l_mlo_out, r_mlo_out = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]
        l_cc_out = F.softmax(l_cc_out, 1)
        r_cc_out = F.softmax(r_cc_out, 1)
        l_mlo_out = F.softmax(l_mlo_out, 1)
        r_mlo_out = F.softmax(r_mlo_out, 1)

        l_output = torch.log((l_cc_out + l_mlo_out) / 2)
        r_output = torch.log((r_cc_out + r_mlo_out) / 2)

        return l_output, r_output


# BreastWise Models
### Concat
class BreastWiseResnet50LastStageConcat(nn.Module):
    def __init__(self, pretrained=False, backbone='resnet50', split=False):
        super(BreastWiseResnet50LastStageConcat, self).__init__()
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

        self.l_fc_0 = nn.Linear(512 * self.block.expansion * 2, 512)
        self.l_bn = nn.BatchNorm1d(512)
        self.l_output_layer = OutputLayer(512, 2)

        self.r_fc_0 = nn.Linear(512 * self.block.expansion * 2, 512)
        self.r_bn = nn.BatchNorm1d(512)
        self.r_output_layer = OutputLayer(512, 2)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_image):
        x = self.four_view_backbone(x_image)
        x = self.four_view_avg_pool(x)
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_l = torch.cat([x_l_cc, x_l_mlo], dim=1)
        x_r = torch.cat([x_r_cc, x_r_mlo], dim=1)
        x_l = torch.flatten(x_l, 1)
        x_r = torch.flatten(x_r, 1)

        x_l = self.l_fc_0(x_l)
        x_l = self.l_bn(x_l)
        x_l = self.relu(x_l)
        l_output = self.l_output_layer(x_l)

        x_r = self.r_fc_0(x_r)
        x_r = self.r_bn(x_r)
        x_r = self.relu(x_r)
        r_output = self.r_output_layer(x_r)

        l_output = torch.log(F.softmax(l_output, 1))
        r_output = torch.log(F.softmax(r_output, 1))

        return l_output, r_output


class BreastWiseSwinTransLastStagesConcat(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(BreastWiseSwinTransLastStagesConcat, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.l_fc_0 = nn.Linear(768 * 2, 768)
        self.l_bn = nn.BatchNorm1d(768)
        self.l_output_layer = OutputLayer(768, 2)

        self.r_fc_0 = nn.Linear(768 * 2, 768)
        self.r_bn = nn.BatchNorm1d(768)
        self.r_output_layer = OutputLayer(768, 2)

        self.relu = nn.ReLU(inplace=True)

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

        # x = self.four_stage_cva_blocks(x)

        return x

    def forward(self, x):
        x = self._forward_pe_and_ape(x)
        # 开始逐层前向传播
        x = self._forward_feature(x)

        for view in VIEWS.LIST:
            x[view] = self.backbone.norm(x[view])  # B L C
            x[view] = self.backbone.avgpool(x[view].transpose(1, 2))  # B C 1
            x[view] = torch.flatten(x[view], 1)

        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_l = torch.cat([x_l_cc, x_l_mlo], dim=1)
        x_r = torch.cat([x_r_cc, x_r_mlo], dim=1)
        x_l = torch.flatten(x_l, 1)
        x_r = torch.flatten(x_r, 1)

        x_l = self.l_fc_0(x_l)
        x_l = self.l_bn(x_l)
        x_l = self.relu(x_l)
        l_output = self.l_output_layer(x_l)

        x_r = self.r_fc_0(x_r)
        x_r = self.r_bn(x_r)
        x_r = self.relu(x_r)
        r_output = self.r_output_layer(x_r)

        l_output = torch.log(F.softmax(l_output, 1))
        r_output = torch.log(F.softmax(r_output, 1))

        return l_output, r_output


### CVA
class BreastWiseResnet50LastStagesCVA(nn.Module):
    # 在resnet50的最后一个stage结束后添加cross view attention block
    def __init__(self, num_classes=2, window_size=7, pretrain=True):
        super(BreastWiseResnet50LastStagesCVA, self).__init__()
        self.resnet_backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
        if pretrain:
            self.resnet_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        self.four_stage_cva_blocks = BreastWiseCVABlockSwinTransNoSwin(dim=2048, input_resolution=(7, 7), num_heads=32,
                                                                       window_size=window_size,
                                                                       drop_path=0.1)

        self.norm = nn.LayerNorm(2048)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.four_view_output_layer = FourViewOutputLayer(feature_channels=2048, output_shape=num_classes)

    def forward(self, x):
        # 开始逐层前向传播
        for view in VIEWS.LIST:
            x[view] = self.resnet_backbone(x[view])
            x[view] = rearrange(x[view], 'b c h w -> b (h w) c', h=7, w=7)

        x = self.four_stage_cva_blocks(x)

        for view in VIEWS.LIST:
            x[view] = self.norm(x[view])  # B L C
            x[view] = self.avgpool(x[view].transpose(1, 2))  # B C 1
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


class BreastWiseSwinTransLastStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(BreastWiseSwinTransLastStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.four_stage_cva_blocks = BreastWiseCVABlockSwinTransNoSwin(dim=768, input_resolution=(7, 7), num_heads=24,
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

    def forward(self, x):
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


class BreastWiseSwinTransLastStagesCVAWithClassToken(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(BreastWiseSwinTransLastStagesCVAWithClassToken, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.class_token = {view: nn.Parameter(torch.randn(1, 1, 768)) for view in VIEWS.LIST}

        self.self_attention_block0 = SwinTransformerBlockWithClassToken(dim=768, input_resolution=(7, 7),
                                                                        num_heads=24,
                                                                        window_size=window_size,
                                                                        drop_path=0.1)

        self.four_stage_cva_blocks = BreastWiseCVABlockSwinTransWithClassToken(dim=768, input_resolution=(7, 7),
                                                                               num_heads=24,
                                                                               window_size=window_size,
                                                                               drop_path=0.1)
        self.norm = nn.LayerNorm(768)

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

                if i == 3:
                    B, L, C = x[view].shape
                    x[view] = torch.cat((self.class_token[view].repeat(B, 1, 1), x[view]), 1)
                    x[view] = self.self_attention_block0(x[view])

        x = self.four_stage_cva_blocks(x)

        return x

    def forward(self, x):
        for view in VIEWS.LIST:
            self.class_token[view] = self.class_token[view].to(x[view].device)
        x = self._forward_pe_and_ape(x)
        # 开始逐层前向传播
        x = self._forward_feature(x)

        for view in VIEWS.LIST:
            # x[view] = self.backbone.norm(x[view])  # B L C
            # x[view] = self.backbone.avgpool(x[view].transpose(1, 2))  # B C 1

            x[view] = x[view].transpose(1, 2)[:, :, 0]  # B C 1
            x[view] = torch.flatten(x[view], 1)
            x[view] = self.norm(x[view])

        x = self.four_view_output_layer(x)

        l_cc_out, r_cc_out, l_mlo_out, r_mlo_out = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]
        l_cc_out = F.softmax(l_cc_out, 1)
        r_cc_out = F.softmax(r_cc_out, 1)
        l_mlo_out = F.softmax(l_mlo_out, 1)
        r_mlo_out = F.softmax(r_mlo_out, 1)

        l_output = torch.log((l_cc_out + l_mlo_out) / 2)
        r_output = torch.log((r_cc_out + r_mlo_out) / 2)

        return l_output, r_output


# Joint Models
### Concat
class JointResnet50LastStageConcat(nn.Module):
    def __init__(self, pretrained=False, backbone='resnet50'):
        super(JointResnet50LastStageConcat, self).__init__()

        if backbone == 'resnet50':
            self.res_block = Bottleneck
            self.four_view_backbone = FourViewResnet50BackboneAllShared(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.res_block = BasicBlock
            self.four_view_backbone = FourViewResnet34BackboneAllShared(pretrained=pretrained)
        elif backbone == 'resnet18':
            self.res_block = BasicBlock
            self.four_view_backbone = FourViewResnet18BackboneAllShared(pretrained=pretrained)

        self.four_view_avg_pool = FourViewAveragePool2d()

        self.fc_0 = nn.Linear(512 * self.res_block.expansion * 4, 512)
        self.bn = nn.BatchNorm1d(512)
        self.output_layer = OutputLayer(512, (2, 2))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_image):
        x = self.four_view_backbone(x_image)
        x = self.four_view_avg_pool(x)
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x = torch.cat([x_l_cc, x_l_mlo, x_r_cc, x_r_mlo], dim=1)
        x = torch.flatten(x, 1)

        x = self.fc_0(x)
        x = self.bn(x)
        x = self.relu(x)
        output = self.output_layer(x)
        output = torch.log(F.softmax(output, 2))

        output_list = output.chunk(2, 1)
        l_output, r_output = output_list[0].squeeze(1), output_list[1].squeeze(1)

        return l_output, r_output


class JointSwinTransLastStagesConcat(nn.Module):
    # 在swin transformer的最后一个stage结束后添加bw cross view attention block和vwcvab
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(JointSwinTransLastStagesConcat, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.fc_0 = nn.Linear(768 * 4, 768)
        self.bn = nn.BatchNorm1d(768)
        self.output_layer = OutputLayer(768, (2, 2))

        self.relu = nn.ReLU(inplace=True)

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

        return x

    def forward(self, x):
        x = self._forward_pe_and_ape(x)
        # 开始逐层前向传播
        x = self._forward_feature(x)

        for view in VIEWS.LIST:
            x[view] = self.backbone.norm(x[view])  # B L C
            x[view] = self.backbone.avgpool(x[view].transpose(1, 2))  # B C 1
            x[view] = torch.flatten(x[view], 1)

        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x = torch.cat([x_l_cc, x_l_mlo, x_r_cc, x_r_mlo], dim=1)
        x = torch.flatten(x, 1)

        x = self.fc_0(x)
        x = self.bn(x)
        x = self.relu(x)
        output = self.output_layer(x)
        output = torch.log(F.softmax(output, 2))

        output_list = output.chunk(2, 1)
        l_output, r_output = output_list[0].squeeze(1), output_list[1].squeeze(1)

        return l_output, r_output


### CVA
class JointResnet50LastStagesCVA(nn.Module):
    # 在resnet50的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, num_classes=2, window_size=7, pretrain=True):
        super(JointResnet50LastStagesCVA, self).__init__()
        self.resnet_backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
        if pretrain:
            self.resnet_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

        self.four_stage_bw_cva_blocks = BreastWiseCVABlockSwinTransNoSwin(dim=2048, input_resolution=(7, 7),
                                                                          num_heads=32,
                                                                          window_size=window_size,
                                                                          drop_path=0.1)
        self.four_stage_vw_cva_blocks = ViewWiseCVABlockSwinTransNoSwin(dim=2048, input_resolution=(7, 7),
                                                                        num_heads=32,
                                                                        window_size=window_size,
                                                                        drop_path=0.1)

        # self.four_stage_cva_blocks = ViewWiseCVABlockSwinTransNoSwin(dim=2048, input_resolution=(7, 7), num_heads=32,
        #                                                              window_size=window_size,
        #                                                              drop_path=0.1)

        self.norm = nn.LayerNorm(2048)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.four_view_output_layer = FourViewOutputLayer(feature_channels=2048, output_shape=num_classes)

    def forward(self, x):
        # 开始逐层前向传播
        for view in VIEWS.LIST:
            x[view] = self.resnet_backbone(x[view])
            x[view] = rearrange(x[view], 'b c h w -> b (h w) c', h=7, w=7)

        x = self.four_stage_bw_cva_blocks(x)
        x = self.four_stage_vw_cva_blocks(x)

        for view in VIEWS.LIST:
            x[view] = self.norm(x[view])  # B L C
            x[view] = self.avgpool(x[view].transpose(1, 2))  # B C 1
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


class JointSwinTransLastStagesCVA(nn.Module):
    # 在swin transformer的最后一个stage结束后添加bw cross view attention block和vwcvab
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(JointSwinTransLastStagesCVA, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.four_stage_bw_cva_blocks = BreastWiseCVABlockSwinTransNoSwin(dim=768, input_resolution=(7, 7),
                                                                          num_heads=24,
                                                                          window_size=window_size,
                                                                          drop_path=0.1)
        self.four_stage_vw_cva_blocks = ViewWiseCVABlockSwinTransNoSwin(dim=768, input_resolution=(7, 7),
                                                                        num_heads=24,
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

        x = self.four_stage_bw_cva_blocks(x)
        x = self.four_stage_vw_cva_blocks(x)

        return x

    def forward(self, x):
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


class JointSwinTransLastStagesCVAWithClassToken(nn.Module):
    # 在swin transformer的最后一个stage结束后添加cross view attention block以及swin cross view attention block
    def __init__(self, img_size=224, patch_size=4, num_classes=2, window_size=7, pretrain=True, pretrain_path=''):
        super(JointSwinTransLastStagesCVAWithClassToken, self).__init__()
        self.backbone = SwinTransformer(img_size=img_size, patch_size=patch_size, window_size=window_size)
        if pretrain:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            self.backbone.load_state_dict(checkpoint['model'])
            del checkpoint
            torch.cuda.empty_cache()

        self.class_token = {view: nn.Parameter(torch.randn(1, 1, 768)) for view in VIEWS.LIST}

        self.self_attention_block0 = SwinTransformerBlockWithClassToken(dim=768, input_resolution=(7, 7),
                                                                        num_heads=24,
                                                                        window_size=window_size,
                                                                        drop_path=0.1)
        # self.self_attention_block1 = SwinTransformerBlockWithClassToken(dim=768, input_resolution=(7, 7),
        #                                                                 num_heads=24,
        #                                                                 window_size=window_size,
        #                                                                 drop_path=0.1)

        self.four_stage_vw_cva_blocks = ViewWiseCVABlockSwinTransWithClassToken(dim=768, input_resolution=(7, 7),
                                                                                num_heads=24,
                                                                                window_size=window_size,
                                                                                drop_path=0.1)
        self.four_stage_bw_cva_blocks = BreastWiseCVABlockSwinTransWithClassToken(dim=768, input_resolution=(7, 7),
                                                                                  num_heads=24,
                                                                                  window_size=window_size,
                                                                                  drop_path=0.1)
        self.norm = nn.LayerNorm(768)

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

                if i == 3:
                    B, L, C = x[view].shape
                    x[view] = torch.cat((self.class_token[view].repeat(B, 1, 1), x[view]), 1)
                    x[view] = self.self_attention_block0(x[view])
                    # x[view] = self.self_attention_block1(x[view])

        x = self.four_stage_vw_cva_blocks(x)
        x = self.four_stage_bw_cva_blocks(x)

        return x

    def forward(self, x):
        for view in VIEWS.LIST:
            self.class_token[view] = self.class_token[view].to(x[view].device)
        x = self._forward_pe_and_ape(x)
        # 开始逐层前向传播
        x = self._forward_feature(x)

        for view in VIEWS.LIST:
            # x[view] = self.backbone.norm(x[view])  # B L C
            # x[view] = self.backbone.avgpool(x[view].transpose(1, 2))  # B C 1

            x[view] = x[view].transpose(1, 2)[:, :, 0]  # B C 1
            x[view] = torch.flatten(x[view], 1)
            x[view] = self.norm(x[view])

        x = self.four_view_output_layer(x)

        l_cc_out, r_cc_out, l_mlo_out, r_mlo_out = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]
        l_cc_out = F.softmax(l_cc_out, 1)
        r_cc_out = F.softmax(r_cc_out, 1)
        l_mlo_out = F.softmax(l_mlo_out, 1)
        r_mlo_out = F.softmax(r_mlo_out, 1)

        l_output = torch.log((l_cc_out + l_mlo_out) / 2)
        r_output = torch.log((r_cc_out + r_mlo_out) / 2)

        return l_output, r_output


# 输出层
class OutputLayer(nn.Module):
    def __init__(self, in_features, output_shape):
        super(OutputLayer, self).__init__()
        if not isinstance(output_shape, (list, tuple)):
            output_shape = [output_shape]
        self.output_shape = output_shape
        self.flattened_output_shape = int(np.prod(output_shape))
        self.fc_layer = nn.Linear(in_features, self.flattened_output_shape)

    def forward(self, x):
        h = self.fc_layer(x)
        if len(self.output_shape) > 1:
            h = h.view(h.shape[0], *self.output_shape)
        # h = F.log_softmax(h, dim=-1)
        return h


# 一些针对4视图的层
class FourViewAveragePool2d(nn.Module):
    def __init__(self):
        super(FourViewAveragePool2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        h_dict = {
            view: self.avg_pool(x[view])
            for view in VIEWS.LIST
        }
        return h_dict


class FourViewAveragePool1d(nn.Module):
    def __init__(self):
        super(FourViewAveragePool1d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        h_dict = {
            view: self.avg_pool(x[view])
            for view in VIEWS.LIST
        }
        return h_dict


class FourViewFC(nn.Module):
    def __init__(self, feature_channels, out_channels):
        super(FourViewFC, self).__init__()
        self.l_cc_fc = nn.Linear(feature_channels, out_channels)

        self.r_cc_fc = nn.Linear(feature_channels, out_channels)

        self.l_mlo_fc = nn.Linear(feature_channels, out_channels)

        self.r_mlo_fc = nn.Linear(feature_channels, out_channels)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.fc_layer_dict = {VIEWS.L_CC: self.l_cc_fc, VIEWS.L_MLO: self.l_mlo_fc,
                              VIEWS.R_CC: self.r_cc_fc, VIEWS.R_MLO: self.r_mlo_fc}

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.relu(self.bn(self.fc_layer_dict[view](single_x)))


class FourViewOutputLayer(nn.Module):
    def __init__(self, feature_channels, output_shape):
        super(FourViewOutputLayer, self).__init__()
        # self.l_cc_fc = nn.Linear(feature_channels, 512)
        self.l_cc_output_layer = OutputLayer(feature_channels, output_shape)

        # self.r_cc_fc = nn.Linear(feature_channels, 512)
        self.r_cc_output_layer = OutputLayer(feature_channels, output_shape)

        # self.l_mlo_fc = nn.Linear(feature_channels, 512)
        self.l_mlo_output_layer = OutputLayer(feature_channels, output_shape)

        # self.r_mlo_fc = nn.Linear(feature_channels, 512)
        self.r_mlo_output_layer = OutputLayer(feature_channels, output_shape)

        # self.fc_dict = {VIEWS.L_CC: self.l_cc_fc, VIEWS.L_MLO: self.l_mlo_fc, VIEWS.R_CC: self.r_cc_fc,
        #                 VIEWS.R_MLO: self.r_mlo_fc}
        self.output_layer_dict = {VIEWS.L_CC: self.l_cc_output_layer, VIEWS.L_MLO: self.l_mlo_output_layer,
                                  VIEWS.R_CC: self.r_cc_output_layer, VIEWS.R_MLO: self.r_mlo_output_layer}

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.output_layer_dict[view](single_x)


class FourViewResnet50BackboneAllShared(nn.Module):
    def __init__(self, pretrained=False):
        super(FourViewResnet50BackboneAllShared, self).__init__()
        self.resnet50_backbone = ResNetBackbone(Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.resnet50_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet50']))

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.resnet50_backbone(single_x)


class FourViewResnet34BackboneAllShared(nn.Module):
    def __init__(self, pretrained=False):
        super(FourViewResnet34BackboneAllShared, self).__init__()
        self.resnet34_backbone = ResNetBackbone(BasicBlock, [3, 4, 6, 3])
        if pretrained:
            self.resnet34_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet34']))

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.resnet34_backbone(single_x)


class FourViewResnet18BackboneAllShared(nn.Module):
    def __init__(self, pretrained=False):
        super(FourViewResnet18BackboneAllShared, self).__init__()
        self.resnet18_backbone = ResNetBackbone(BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.resnet18_backbone.load_state_dict(model_zoo.load_url(model_urls['resnet18']))

    def forward(self, x):
        h_dict = {
            view: self.single_forward(x[view], view)
            for view in VIEWS.LIST
        }
        return h_dict

    def single_forward(self, single_x, view):
        return self.resnet18_backbone(single_x)


# CVA block
class ViewWiseCVABlockSwinTrans(nn.Module):
    # Cross View Attention Block 包含一个基本window based cross view attention block 和一个 swin cvab
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(ViewWiseCVABlockSwinTrans, self).__init__()

        self.cc_cva_block = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                          num_heads=num_heads, window_size=window_size,
                                                          shift_size=0,
                                                          mlp_ratio=mlp_ratio,
                                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                          drop=drop, attn_drop=attn_drop,
                                                          drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                list) else drop_path,
                                                          norm_layer=norm_layer)
        self.cc_cva_block_shifted = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                                  num_heads=num_heads, window_size=window_size,
                                                                  shift_size=window_size // 2,
                                                                  mlp_ratio=mlp_ratio,
                                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                  drop=drop, attn_drop=attn_drop,
                                                                  drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                        list) else drop_path,
                                                                  norm_layer=norm_layer)

        self.mlo_cva_block = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                           num_heads=num_heads, window_size=window_size,
                                                           shift_size=0,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                           drop=drop, attn_drop=attn_drop,
                                                           drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                 list) else drop_path,
                                                           norm_layer=norm_layer)
        self.mlo_cva_block_shifted = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                                   num_heads=num_heads, window_size=window_size,
                                                                   shift_size=window_size // 2,
                                                                   mlp_ratio=mlp_ratio,
                                                                   qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                   drop=drop, attn_drop=attn_drop,
                                                                   drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                         list) else drop_path,
                                                                   norm_layer=norm_layer)

    def forward(self, x):
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_cc = [x_l_cc, x_r_cc]
        x_mlo = [x_l_mlo, x_r_mlo]

        x_cc = self.cc_cva_block(x_cc)
        x_cc = self.cc_cva_block_shifted(x_cc)
        x_mlo = self.mlo_cva_block(x_mlo)
        x_mlo = self.mlo_cva_block_shifted(x_mlo)

        x_l_cc, x_r_cc = x_cc[0], x_cc[1]
        x_l_mlo, x_r_mlo = x_mlo[0], x_mlo[1]

        x_out = {view: [] for view in VIEWS.LIST}
        x_out[VIEWS.L_CC], x_out[VIEWS.R_CC], x_out[VIEWS.L_MLO], x_out[VIEWS.R_MLO] = x_l_cc, x_r_cc, x_l_mlo, x_r_mlo

        return x_out


class ViewWiseCVABlockSwinTransNoSwin(nn.Module):
    # Cross View Attention Block 只包含一个基本window based cross view attention block
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(ViewWiseCVABlockSwinTransNoSwin, self).__init__()

        self.cc_cva_block = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                          num_heads=num_heads, window_size=window_size,
                                                          shift_size=0,
                                                          mlp_ratio=mlp_ratio,
                                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                          drop=drop, attn_drop=attn_drop,
                                                          drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                list) else drop_path,
                                                          norm_layer=norm_layer)

        self.mlo_cva_block = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                           num_heads=num_heads, window_size=window_size,
                                                           shift_size=0,
                                                           mlp_ratio=mlp_ratio,
                                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                           drop=drop, attn_drop=attn_drop,
                                                           drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                 list) else drop_path,
                                                           norm_layer=norm_layer)

    def forward(self, x):
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_cc = [x_l_cc, x_r_cc]
        x_mlo = [x_l_mlo, x_r_mlo]

        x_cc = self.cc_cva_block(x_cc)
        x_mlo = self.mlo_cva_block(x_mlo)

        x_l_cc, x_r_cc = x_cc[0], x_cc[1]
        x_l_mlo, x_r_mlo = x_mlo[0], x_mlo[1]

        x_out = {view: [] for view in VIEWS.LIST}
        x_out[VIEWS.L_CC], x_out[VIEWS.R_CC], x_out[VIEWS.L_MLO], x_out[VIEWS.R_MLO] = x_l_cc, x_r_cc, x_l_mlo, x_r_mlo

        return x_out


class ViewWiseCVABlockSwinTransWithClassToken(nn.Module):
    # 使用ClassToken的
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(ViewWiseCVABlockSwinTransWithClassToken, self).__init__()

        self.cc_cva_block = CVSTBWithClassificationToken(dim=dim, input_resolution=input_resolution,
                                                         num_heads=num_heads, window_size=window_size,
                                                         shift_size=0,
                                                         mlp_ratio=mlp_ratio,
                                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                               list) else drop_path,
                                                         norm_layer=norm_layer)

        self.mlo_cva_block = CVSTBWithClassificationToken(dim=dim, input_resolution=input_resolution,
                                                          num_heads=num_heads, window_size=window_size,
                                                          shift_size=0,
                                                          mlp_ratio=mlp_ratio,
                                                          qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                          drop=drop, attn_drop=attn_drop,
                                                          drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                list) else drop_path,
                                                          norm_layer=norm_layer)

    def forward(self, x):
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_cc = [x_l_cc, x_r_cc]
        x_mlo = [x_l_mlo, x_r_mlo]

        x_cc = self.cc_cva_block(x_cc)
        x_mlo = self.mlo_cva_block(x_mlo)

        x_l_cc, x_r_cc = x_cc[0], x_cc[1]
        x_l_mlo, x_r_mlo = x_mlo[0], x_mlo[1]

        x_out = {view: [] for view in VIEWS.LIST}
        x_out[VIEWS.L_CC], x_out[VIEWS.R_CC], x_out[VIEWS.L_MLO], x_out[VIEWS.R_MLO] = x_l_cc, x_r_cc, x_l_mlo, x_r_mlo

        return x_out


class BreastWiseCVABlockSwinTrans(nn.Module):
    # Cross View Attention Block 包含一个基本window based cross view attention block 和一个 swin cvab
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(BreastWiseCVABlockSwinTrans, self).__init__()

        self.l_cva_block = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                         num_heads=num_heads, window_size=window_size,
                                                         shift_size=0,
                                                         mlp_ratio=mlp_ratio,
                                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                               list) else drop_path,
                                                         norm_layer=norm_layer)
        self.l_cva_block_shifted = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                                 num_heads=num_heads, window_size=window_size,
                                                                 shift_size=window_size // 2,
                                                                 mlp_ratio=mlp_ratio,
                                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                 drop=drop, attn_drop=attn_drop,
                                                                 drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                       list) else drop_path,
                                                                 norm_layer=norm_layer)

        self.r_cva_block = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                         num_heads=num_heads, window_size=window_size,
                                                         shift_size=0,
                                                         mlp_ratio=mlp_ratio,
                                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                               list) else drop_path,
                                                         norm_layer=norm_layer)
        self.r_cva_block_shifted = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                                 num_heads=num_heads, window_size=window_size,
                                                                 shift_size=window_size // 2,
                                                                 mlp_ratio=mlp_ratio,
                                                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                                 drop=drop, attn_drop=attn_drop,
                                                                 drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                                       list) else drop_path,
                                                                 norm_layer=norm_layer)

    def forward(self, x):
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_l = [x_l_cc, x_l_mlo]
        x_r = [x_r_cc, x_r_mlo]

        x_l = self.l_cva_block(x_l)
        x_l = self.l_cva_block_shifted(x_l)
        x_r = self.r_cva_block(x_r)
        x_r = self.r_cva_block_shifted(x_r)

        x_l_cc, x_l_mlo = x_l[0], x_l[1]
        x_r_cc, x_r_mlo = x_r[0], x_r[1]

        x_out = {view: [] for view in VIEWS.LIST}
        x_out[VIEWS.L_CC], x_out[VIEWS.R_CC], x_out[VIEWS.L_MLO], x_out[VIEWS.R_MLO] = x_l_cc, x_r_cc, x_l_mlo, x_r_mlo

        return x_out


class BreastWiseCVABlockSwinTransNoSwin(nn.Module):
    # Cross View Attention Block 只包含一个基本window based cross view attention block
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(BreastWiseCVABlockSwinTransNoSwin, self).__init__()

        self.l_cva_block = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                         num_heads=num_heads, window_size=window_size,
                                                         shift_size=0,
                                                         mlp_ratio=mlp_ratio,
                                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                               list) else drop_path,
                                                         norm_layer=norm_layer)

        self.r_cva_block = CrossViewSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                                         num_heads=num_heads, window_size=window_size,
                                                         shift_size=0,
                                                         mlp_ratio=mlp_ratio,
                                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                         drop=drop, attn_drop=attn_drop,
                                                         drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                               list) else drop_path,
                                                         norm_layer=norm_layer)

    def forward(self, x):
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_l = [x_l_cc, x_l_mlo]
        x_r = [x_r_cc, x_r_mlo]

        x_l = self.l_cva_block(x_l)
        x_r = self.r_cva_block(x_r)

        x_l_cc, x_l_mlo = x_l[0], x_l[1]
        x_r_cc, x_r_mlo = x_r[0], x_r[1]

        x_out = {view: [] for view in VIEWS.LIST}
        x_out[VIEWS.L_CC], x_out[VIEWS.R_CC], x_out[VIEWS.L_MLO], x_out[VIEWS.R_MLO] = x_l_cc, x_r_cc, x_l_mlo, x_r_mlo

        return x_out


class BreastWiseCVABlockSwinTransWithClassToken(nn.Module):
    # Cross View Attention Block 只包含一个基本window based cross view attention block
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(BreastWiseCVABlockSwinTransWithClassToken, self).__init__()

        self.l_cva_block = CVSTBWithClassificationToken(dim=dim, input_resolution=input_resolution,
                                                        num_heads=num_heads, window_size=window_size,
                                                        shift_size=0,
                                                        mlp_ratio=mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop, attn_drop=attn_drop,
                                                        drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                              list) else drop_path,
                                                        norm_layer=norm_layer)

        self.r_cva_block = CVSTBWithClassificationToken(dim=dim, input_resolution=input_resolution,
                                                        num_heads=num_heads, window_size=window_size,
                                                        shift_size=0,
                                                        mlp_ratio=mlp_ratio,
                                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                        drop=drop, attn_drop=attn_drop,
                                                        drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                              list) else drop_path,
                                                        norm_layer=norm_layer)

    def forward(self, x):
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_l = [x_l_cc, x_l_mlo]
        x_r = [x_r_cc, x_r_mlo]

        x_l = self.l_cva_block(x_l)
        x_r = self.r_cva_block(x_r)

        x_l_cc, x_l_mlo = x_l[0], x_l[1]
        x_r_cc, x_r_mlo = x_r[0], x_r[1]

        x_out = {view: [] for view in VIEWS.LIST}
        x_out[VIEWS.L_CC], x_out[VIEWS.R_CC], x_out[VIEWS.L_MLO], x_out[VIEWS.R_MLO] = x_l_cc, x_r_cc, x_l_mlo, x_r_mlo

        return x_out


class ViewWiseCVABlockSwinTransForPhenotype(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(ViewWiseCVABlockSwinTransForPhenotype, self).__init__()

        self.cc_cva_block = CVSTBForPhenotype(dim=dim, input_resolution=input_resolution,
                                              num_heads=num_heads,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                                              drop=drop, attn_drop=attn_drop,
                                              drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                    list) else drop_path,
                                              norm_layer=norm_layer)

        self.mlo_cva_block = CVSTBForPhenotype(dim=dim, input_resolution=input_resolution,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               drop=drop, attn_drop=attn_drop,
                                               drop_path=drop_path[-1] if isinstance(drop_path,
                                                                                     list) else drop_path,
                                               norm_layer=norm_layer)

    def forward(self, x):
        x_l_cc, x_r_cc, x_l_mlo, x_r_mlo = x[VIEWS.L_CC], x[VIEWS.R_CC], x[VIEWS.L_MLO], x[VIEWS.R_MLO]

        x_cc = [x_l_cc, x_r_cc]
        x_mlo = [x_l_mlo, x_r_mlo]

        x_cc = self.cc_cva_block(x_cc)
        x_mlo = self.mlo_cva_block(x_mlo)

        x_l_cc, x_r_cc = x_cc[0], x_cc[1]
        x_l_mlo, x_r_mlo = x_mlo[0], x_mlo[1]

        x_out = {view: [] for view in VIEWS.LIST}
        x_out[VIEWS.L_CC], x_out[VIEWS.R_CC], x_out[VIEWS.L_MLO], x_out[VIEWS.R_MLO] = x_l_cc, x_r_cc, x_l_mlo, x_r_mlo

        return x_out

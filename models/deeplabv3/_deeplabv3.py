from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from models.utils import CoordConv, LearnablePE

__all__ = ["DeepLabV3"]


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
        coord_conv: Type of coordinate convolution to apply ("cartesian", "radial", or False)
        input_size: Input image size for initializing coordinate grids
    """

    def __init__(
        self,
        backbone,
        classifier,
        coord_conv: Literal["cartesian", "radial", False] = False,
        input_size: tuple[int, int] = (256, 256),
    ):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.coord_conv = coord_conv
        if coord_conv:
            self.coord_conv_input = CoordConv(coord_conv, input_size[0], input_size[1])

    def forward(self, x):
        input_shape = x.shape[-2:]
        if self.coord_conv:
            x = self.coord_conv_input(x)
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        return x


class DeepLabHeadV3Plus(nn.Module):
    def __init__(
        self,
        in_channels,
        low_level_channels,
        num_classes,
        aspp_dilate=[12, 24, 36],
        coord_conv: Literal["cartesian", "radial", False] = False,
        learnable_pe: Literal["add", "concat", False] = False,
        feature_size: tuple[int, int] = (32, 32),
        low_level_size: tuple[int, int] = (64, 64),
    ):
        super(DeepLabHeadV3Plus, self).__init__()
        self.coord_conv = coord_conv
        self.learnable_pe = learnable_pe

        if self.learnable_pe == "add":
            self.learnable_pe_low = LearnablePE(
                "add", low_level_size[0], low_level_size[1], low_level_channels
            )
        elif self.learnable_pe == "concat":
            pe_low_channels = 16
            self.learnable_pe_low = LearnablePE(
                "concat", low_level_size[0], low_level_size[1], pe_low_channels
            )
            low_level_channels += pe_low_channels
        if self.coord_conv:
            self.coord_conv_low = CoordConv(
                self.coord_conv, low_level_size[0], low_level_size[1]
            )
            low_level_channels += 2

        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        if self.learnable_pe == "add":
            self.learnable_pe_feat = LearnablePE(
                "add", feature_size[0], feature_size[1], in_channels
            )
        elif self.learnable_pe == "concat":
            pe_feat_channels = 16
            self.learnable_pe_feat = LearnablePE(
                "concat", feature_size[0], feature_size[1], pe_feat_channels
            )
            in_channels += pe_feat_channels
        if self.coord_conv:
            self.coord_conv_feat = CoordConv(
                self.coord_conv, feature_size[0], feature_size[1]
            )
            in_channels += 2

        self.aspp = ASPP(in_channels, aspp_dilate)

        self.classifier = nn.Sequential(
            nn.Conv2d(304, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weight()

    def forward(self, feature):
        low_level_feature = feature["low_level"]

        if self.learnable_pe:
            low_level_feature = self.learnable_pe_low(low_level_feature)

        if self.coord_conv:
            low_level_feature = self.coord_conv_low(low_level_feature)

        low_level_feature = self.project(low_level_feature)

        output_feature = feature["out"]

        if self.learnable_pe:
            output_feature = self.learnable_pe_feat(output_feature)

        if self.coord_conv:
            output_feature = self.coord_conv_feat(output_feature)

        output_feature = self.aspp(output_feature)
        output_feature = F.interpolate(
            output_feature,
            size=low_level_feature.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.classifier(torch.cat([low_level_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        aspp_dilate=[12, 24, 36],
        coord_conv: Literal["cartesian", "radial", False] = False,
        learnable_pe: Literal["add", "concat", False] = False,
        feature_size: tuple[int, int] = (32, 32),
    ):
        super(DeepLabHead, self).__init__()
        self.coord_conv = coord_conv
        self.learnable_pe = learnable_pe

        if self.learnable_pe == "add":
            self.learnable_pe_feat = LearnablePE(
                "add", feature_size[0], feature_size[1], in_channels
            )
        elif self.learnable_pe == "concat":
            pe_feat_channels = 16
            self.learnable_pe_feat = LearnablePE(
                "concat", feature_size[0], feature_size[1], pe_feat_channels
            )
            in_channels += pe_feat_channels
        if self.coord_conv:
            self.coord_conv_feat = CoordConv(
                self.coord_conv, feature_size[0], feature_size[1]
            )
            in_channels += 2

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1),
        )
        self._init_weight()

    def forward(self, feature):
        output_feature = feature["out"]

        if self.learnable_pe:
            output_feature = self.learnable_pe_feat(output_feature)

        if self.coord_conv:
            output_feature = self.coord_conv_feat(output_feature)

        return self.classifier(output_feature)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(min(32, out_channels), out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

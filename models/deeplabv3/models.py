from torch import nn

from ._deeplabv3 import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from .backbones import hrnetv2, mobilenetv2, resnet
from .utils import IntermediateLayerGetter


def _segm_hrnet(name, backbone_name, num_classes, **kwargs):
    if backbone_name not in hrnetv2.__dict__:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    backbone = hrnetv2.__dict__[backbone_name](**kwargs)
    # HRNetV2 config:
    # the final output channels is dependent on highest resolution channel config (c).
    # output of backbone will be the inplanes to assp:
    _, hrnet_channels = backbone_name.split("_")
    inplanes = sum([int(hrnet_channels) * 2**i for i in range(4)])
    low_level_planes = (
        256  # all hrnet version channel output from bottleneck is the same
    )
    aspp_dilate = [12, 24, 36]  # If follow paper trend, can put [24, 48, 72].

    if name == "deeplabv3plus":
        return_layers = {"stage4": "out", "layer1": "low_level"}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"stage4": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)

    backbone = IntermediateLayerGetter(
        backbone, return_layers=return_layers, hrnet_flag=True
    )
    model = DeepLabV3(backbone, classifier)
    return model


def _segm_resnet(name, backbone_name, num_classes, output_stride, **kwargs):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    if backbone_name not in resnet.__dict__:
        raise ValueError(f"Unknown backbone: {backbone_name}")
    backbone = resnet.__dict__[backbone_name](
        replace_stride_with_dilation=replace_stride_with_dilation,
        **kwargs,
    )

    inplanes = 2048
    low_level_planes = 256

    if name == "deeplabv3plus":
        return_layers = {"layer4": "out", "layer1": "low_level"}
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"layer4": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, num_classes, output_stride, **kwargs):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenetv2(output_stride=output_stride, **kwargs)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None  # type: ignore
    backbone.classifier = None  # type: ignore

    inplanes = 320
    low_level_planes = 24

    if name == "deeplabv3plus":
        return_layers = {
            "high_level_features": "out",
            "low_level_features": "low_level",
        }
        classifier = DeepLabHeadV3Plus(
            inplanes, low_level_planes, num_classes, aspp_dilate
        )
    elif name == "deeplabv3":
        return_layers = {"high_level_features": "out"}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def load_deeplabv3(
    plus: bool, backbone: str, output_channels: int, **kwargs
) -> nn.Module:
    """
    Load a DeepLabV3 model with the specified architecture and backbone.

    Args:
        plus (bool): Whether to load DeepLabV3+ (True) or DeepLabV3 (False).
        backbone (str): Backbone model to use (e.g., "resnet50", "mobilenetv2", "hrnetv2_48").
        output_channels (int): Number of output channels for the model.
        **kwargs: Additional keyword arguments for initilizing the backbone.

    Returns:
        nn.Module: The loaded DeepLabV3 model.
    """
    arch_type = "deeplabv3plus" if plus else "deeplabv3"
    if "output_stride" not in kwargs and backbone in ["mobilenetv2", "resnet50"]:
        kwargs["output_stride"] = 8

    if backbone == "mobilenetv2":
        model = _segm_mobilenet(arch_type, output_channels, **kwargs)
    elif backbone.startswith("resnet"):
        model = _segm_resnet(arch_type, backbone, output_channels, **kwargs)
    elif backbone.startswith("hrnetv2"):
        model = _segm_hrnet(arch_type, backbone, output_channels, **kwargs)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return model

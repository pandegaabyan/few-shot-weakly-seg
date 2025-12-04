from typing import Any

from torch import nn

from config.config_type import ModelConfig
from models.deeplabv3.models import load_deeplabv3
from models.unetmini import UNetMini


def make_segmentation_model(
    config: ModelConfig,
    input_size: tuple[int, int],
    input_channels: int,
    output_channels: int,
) -> nn.Module:
    arch = config.get("arch")
    if arch is None:
        raise ValueError("Model architecture must be specified in the config.")
    backbone = config.get("backbone", None)
    kwargs: dict[str, Any] = {
        k: v for k, v in config.items() if k not in ["arch", "backbone"]
    }

    if arch == "unetmini":
        if backbone is not None:
            raise ValueError(
                "Backbone should not be specified for UNetMini architecture."
            )
        return UNetMini(
            input_channels, output_channels, input_size=input_size, **kwargs
        )
    if arch in ["deeplabv3", "deeplabv3plus"]:
        kwargs["in_channels"] = input_channels
        if backbone is None:
            raise ValueError("Backbone must be specified for DeepLabV3 architecture.")
        return load_deeplabv3(False, backbone, output_channels, **kwargs)
    raise ValueError(
        f"Unsupported architecture: {arch}. Supported architectures are 'unetmini', 'deeplabv3', and 'deeplabv3plus'"
    )

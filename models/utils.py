from collections import OrderedDict
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


class CoordConv(nn.Module):
    """
    Adds coordinate channels to input features.

    This module appends 2 additional channels containing coordinate information
    to the input tensor. The coordinates can be either cartesian (x, y) or
    radial (r, θ), normalized to the range [-1, 1].

    Args:
        coord_type: Type of coordinates - "cartesian" or "radial"
        height: Height of the coordinate grid
        width: Width of the coordinate grid
    """

    def __init__(
        self, coord_type: Literal["cartesian", "radial"], height: int, width: int
    ):
        super(CoordConv, self).__init__()
        self.coord_type = coord_type

        if self.coord_type not in ("cartesian", "radial"):
            raise ValueError(f"Unknown coord_type: {self.coord_type}")

        coord_1, coord_2 = self.create_coord_grid(height, width, self.coord_type)
        self.register_buffer("coord_1", coord_1)
        self.register_buffer("coord_2", coord_2)

    def forward(self, x):
        batch_size = x.size(0)
        coord_1 = self.coord_1.unsqueeze(0).expand(batch_size, 1, -1, -1)
        coord_2 = self.coord_2.unsqueeze(0).expand(batch_size, 1, -1, -1)

        return torch.cat([x, coord_1, coord_2], dim=1)

    @staticmethod
    def create_coord_grid(
        height: int, width: int, coord_type: Literal["cartesian", "radial"]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create coordinate grids for CoordConv.

        Args:
            height: Height of the coordinate grid
            width: Width of the coordinate grid
            coord_type: Type of coordinates - "cartesian" (x, y) or "radial" (r, θ)

        Returns:
            Tuple of two tensors representing the coordinate channels
        """
        y_coords = torch.linspace(-1, 1, height).view(height, 1).expand(height, width)
        x_coords = torch.linspace(-1, 1, width).view(1, width).expand(height, width)

        if coord_type == "cartesian":
            return x_coords, y_coords
        elif coord_type == "radial":
            r_coords = torch.sqrt(x_coords**2 + y_coords**2) / (2**0.5)
            theta_coords = torch.atan2(y_coords, x_coords) / torch.pi
            return r_coords, theta_coords
        else:
            raise ValueError(f"Unknown coord_type: {coord_type}")


class LearnablePE(nn.Module):
    """
    Learnable Positional Encoding module.

    This module provides learnable positional encodings to the input features.
    The positional encodings can be added or concatenated, based on the specified mode.

    Args:
        mode: "add" to add positional encodings, "concat" to concatenate them
        height: Height of the positional encoding grid
        width: Width of the positional encoding grid
        channels: Number of channels for the positional encoding
    """

    def __init__(
        self,
        mode: Literal["add", "concat"],
        height: int,
        width: int,
        channels: int,
    ):
        super(LearnablePE, self).__init__()
        self.mode = mode

        if self.mode == "add":
            pe_h, pe_w = self.create_pe_decompose(height, width, channels)
            self.pe_h = nn.Parameter(pe_h)
            self.pe_w = nn.Parameter(pe_w)
            self.pe_alpha = nn.Parameter(torch.ones(channels, 1, 1) * -6.0)
        elif self.mode == "concat":
            pe_2d = self.create_pe_2d(height, width, channels)
            self.pe_2d = nn.Parameter(pe_2d)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        self.pe_dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        if self.mode == "add":
            positional_encoding = self.pe_dropout(self.pe_h + self.pe_w)
            x = x + torch.sigmoid(self.pe_alpha) * positional_encoding
        elif self.mode == "concat":
            positional_encoding = self.pe_dropout(self.pe_2d)
            x = torch.cat([x, positional_encoding.expand(x.size(0), -1, -1, -1)], dim=1)
        return x

    @staticmethod
    def create_pe_decompose(
        height: int, width: int, channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create decomposed learnable positional encoding parameters (separate H and W).

        Uses sinusoidal encoding similar to Transformer positional encodings,
        but decomposed into separate height and width components for efficiency.

        Args:
            height: Height of the positional encoding grid
            width: Width of the positional encoding grid
            channels: Number of channels for the positional encoding

        Returns:
            Tuple of (pe_h, pe_w) tensors with shapes (1, channels, height, 1) and (1, channels, 1, width)
        """
        pe_h = torch.zeros(1, channels, height, 1)
        pe_w = torch.zeros(1, channels, 1, width)

        h_pos = (
            torch.arange(height).float().unsqueeze(0).unsqueeze(0).unsqueeze(-1)
            / height
        )
        w_pos = (
            torch.arange(width).float().unsqueeze(0).unsqueeze(0).unsqueeze(0) / width
        )
        div_term = torch.exp(
            torch.arange(0, channels, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / channels)
        )

        pe_h[:, 0::2, :, :] = torch.sin(h_pos * div_term.view(1, -1, 1, 1))
        pe_h[:, 1::2, :, :] = torch.cos(h_pos * div_term.view(1, -1, 1, 1))
        pe_w[:, 0::2, :, :] = torch.sin(w_pos * div_term.view(1, -1, 1, 1))
        pe_w[:, 1::2, :, :] = torch.cos(w_pos * div_term.view(1, -1, 1, 1))

        return pe_h, pe_w

    @staticmethod
    def create_pe_2d(height: int, width: int, channels: int) -> torch.Tensor:
        """
        Create 2D learnable positional encoding parameters.

        Uses sinusoidal encoding based on the sum of normalized height and width positions.

        Args:
            height: Height of the positional encoding grid
            width: Width of the positional encoding grid
            channels: Number of channels for the positional encoding

        Returns:
            Tensor with shape (1, channels, height, width)
        """
        pe_2d = torch.zeros(1, channels, height, width)

        h_pos = torch.arange(height).float().view(1, 1, height, 1) / height
        w_pos = torch.arange(width).float().view(1, 1, 1, width) / width
        grid = h_pos + w_pos

        div_term = torch.exp(
            torch.arange(0, channels, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / channels)
        )

        pe_2d[:, 0::2, :, :] = torch.sin(grid * div_term.view(-1, 1, 1))
        pe_2d[:, 1::2, :, :] = torch.cos(grid * div_term.view(-1, 1, 1))

        return pe_2d


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """

    def __init__(self, model, return_layers, hrnet_flag=False):
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            raise ValueError("return_layers are not present in model")

        self.hrnet_flag = hrnet_flag

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            if self.hrnet_flag and name.startswith(
                "transition"
            ):  # if using hrnet, you need to take care of transition
                if (
                    name == "transition1"
                ):  # in transition1, you need to split the module to two streams first
                    x = [trans(x) for trans in module]  # type: ignore
                else:  # all other transition is just an extra one stream split
                    x.append(module(x[-1]))  # type: ignore
            else:  # other models (ex:resnet,mobilenet) are convolutions in series.
                x = module(x)

            if name in self.return_layers:
                out_name = self.return_layers[name]
                if (
                    name == "stage4" and self.hrnet_flag
                ):  # In HRNetV2, we upsample and concat all outputs streams together
                    output_h, output_w = (
                        x[0].size(2),
                        x[0].size(3),
                    )  # Upsample to size of highest resolution stream
                    x1 = F.interpolate(
                        x[1],
                        size=(output_h, output_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    x2 = F.interpolate(
                        x[2],
                        size=(output_h, output_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    x3 = F.interpolate(
                        x[3],
                        size=(output_h, output_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    x = torch.cat([x[0], x1, x2, x3], dim=1)
                    out[out_name] = x
                else:
                    out[out_name] = x
        return out

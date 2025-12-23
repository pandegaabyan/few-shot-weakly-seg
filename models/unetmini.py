from collections import OrderedDict
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from torchmeta import modules


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if (
                isinstance(module, nn.Conv2d)
                or isinstance(module, nn.Linear)
                or isinstance(module, modules.MetaConv2d)
                or isinstance(module, modules.MetaLinear)
            ):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d) or isinstance(
                module, modules.MetaBatchNorm2d
            ):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class MetaConvTranspose2d(nn.ConvTranspose2d, modules.MetaModule):
    __doc__ = nn.ConvTranspose2d.__doc__

    def forward(self, input, output_size=None, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        weights = params.get("weight", None)
        bias = params.get("bias", None)

        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )
        if isinstance(self.padding, str):
            raise ValueError("Only integers padding is supported")
        if weights is None:
            raise ValueError("Weights should not be None")

        output_padding = self._output_padding(
            input,
            output_size,
            list(self.stride),
            list(self.padding),
            list(self.kernel_size),
            2,
            list(self.dilation),
        )

        return F.conv_transpose2d(
            input,
            weights,
            bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )


class _MetaEncoderBlock(modules.MetaModule):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_MetaEncoderBlock, self).__init__()

        layers = [
            modules.MetaConv2d(in_channels, out_channels, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            modules.MetaConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout:
            layers.append(nn.Dropout())

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.encode = modules.MetaSequential(*layers)

    def forward(self, x, params=None):
        return self.encode(x, self.get_subdict(params, "encode"))


class _MetaDecoderBlock(modules.MetaModule):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_MetaDecoderBlock, self).__init__()

        self.decode = modules.MetaSequential(
            nn.Dropout2d(),
            modules.MetaConv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            modules.MetaConv2d(
                middle_channels, middle_channels, kernel_size=3, padding=1
            ),
            modules.MetaBatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            MetaConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                padding=0,
                output_padding=0,
            ),
        )

    def forward(self, x, params=None):
        return self.decode(x, self.get_subdict(params, "decode"))


class UNetMini(modules.MetaModule):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        prototype: bool = False,
        coord_conv: Literal["cartesian", "radial", False] = False,
        learnable_pe: Literal["add", "concat", False] = False,
        input_size: tuple[int, int] = (256, 256),
        **kwargs,
    ):
        super(UNetMini, self).__init__()

        self.prototype = prototype
        self.coord_conv = coord_conv
        self.learnable_pe = learnable_pe
        height, width = input_size

        effective_input_channels = input_channels
        if self.coord_conv:
            self.create_coords(height, width)
            effective_input_channels += 2

        self.enc1 = _MetaEncoderBlock(effective_input_channels, 32)
        self.enc2 = _MetaEncoderBlock(32, 64)
        self.enc3 = _MetaEncoderBlock(64, 128, dropout=True)

        effective_center_channels = 128
        if self.learnable_pe == "add":
            self.create_pe_decompose(height // 8, width // 8, effective_center_channels)
            self.pe_alpha = nn.Parameter(
                torch.ones(effective_center_channels, 1, 1) * -6.0
            )
            self.pe_dropout = nn.Dropout2d(0.2)
        elif self.learnable_pe == "concat":
            pe_channels = 16
            effective_center_channels += pe_channels
            self.create_pe_2d(height // 8, width // 8, pe_channels)
            self.pe_dropout = nn.Dropout2d(0.2)

        self.center = _MetaDecoderBlock(effective_center_channels, 256, 128)

        self.dec3 = _MetaDecoderBlock(128 + effective_center_channels, 128, 64)
        self.dec2 = _MetaDecoderBlock(128, 64, 32)

        dec1_out_channels = output_channels if self.prototype else 32
        self.dec1 = modules.MetaSequential(
            nn.Dropout2d(),
            modules.MetaConv2d(64, dec1_out_channels, kernel_size=3, padding=1),
            modules.MetaBatchNorm2d(dec1_out_channels),
            nn.ReLU(inplace=True),
            modules.MetaConv2d(
                dec1_out_channels, dec1_out_channels, kernel_size=3, padding=1
            ),
            modules.MetaBatchNorm2d(dec1_out_channels),
            nn.ReLU(inplace=True),
        )

        if not self.prototype:
            self.final = modules.MetaConv2d(32, output_channels, kernel_size=1)

        initialize_weights(self)

    def create_coords(self, height, width):
        y_coords = torch.linspace(-1, 1, height).view(height, 1).expand(height, width)
        x_coords = torch.linspace(-1, 1, width).view(1, width).expand(height, width)

        if self.coord_conv == "cartesian":
            self.coord_1, self.coord_2 = x_coords, y_coords
        if self.coord_conv == "radial":
            r_coords = torch.sqrt(x_coords**2 + y_coords**2) / (2**0.5)
            theta_coords = torch.atan2(y_coords, x_coords) / torch.pi
            self.coord_1, self.coord_2 = r_coords, theta_coords

    def create_pe_decompose(self, height, width, channels):
        self.pe_h = nn.Parameter(torch.zeros(1, channels, height, 1))
        self.pe_w = nn.Parameter(torch.zeros(1, channels, 1, width))

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

        with torch.no_grad():
            self.pe_h[:, 0::2, :, :] = torch.sin(h_pos * div_term.view(1, -1, 1, 1))
            self.pe_h[:, 1::2, :, :] = torch.cos(h_pos * div_term.view(1, -1, 1, 1))
            self.pe_w[:, 0::2, :, :] = torch.sin(w_pos * div_term.view(1, -1, 1, 1))
            self.pe_w[:, 1::2, :, :] = torch.cos(w_pos * div_term.view(1, -1, 1, 1))

    def create_pe_2d(self, height, width, channels):
        self.pe_2d = nn.Parameter(torch.zeros(1, channels, height, width))

        h_pos = torch.arange(height).float().view(1, 1, height, 1) / height
        w_pos = torch.arange(width).float().view(1, 1, 1, width) / width
        grid = h_pos + w_pos

        div_term = torch.exp(
            torch.arange(0, channels, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / channels)
        )

        with torch.no_grad():
            self.pe_2d[:, 0::2, :, :] = torch.sin(grid * div_term.view(-1, 1, 1))
            self.pe_2d[:, 1::2, :, :] = torch.cos(grid * div_term.view(-1, 1, 1))

    def forward(self, x, params=None):
        if self.coord_conv:
            batch_size = x.size(0)
            coord_1 = self.coord_1.expand(batch_size, 1, -1, -1).to(x.device)
            coord_2 = self.coord_2.expand(batch_size, 1, -1, -1).to(x.device)
            x = torch.cat([x, coord_1, coord_2], dim=1)

        enc1 = self.enc1(x, self.get_subdict(params, "enc1"))
        enc2 = self.enc2(enc1, self.get_subdict(params, "enc2"))
        enc3 = self.enc3(enc2, self.get_subdict(params, "enc3"))

        if self.learnable_pe == "add":
            positional_encoding = self.pe_dropout(self.pe_h + self.pe_w)
            enc3 = enc3 + torch.sigmoid(self.pe_alpha) * positional_encoding
        elif self.learnable_pe == "concat":
            positional_encoding = self.pe_dropout(self.pe_2d)
            enc3 = torch.cat(
                [enc3, positional_encoding.expand(enc3.size(0), -1, -1, -1)], dim=1
            )

        center = self.center(enc3, self.get_subdict(params, "center"))

        dec3 = self.dec3(
            torch.cat(
                [
                    center,
                    F.interpolate(enc3, center.size()[2:], mode="bilinear"),
                ],
                1,
            ),
            self.get_subdict(params, "dec3"),
        )
        dec2 = self.dec2(
            torch.cat(
                [dec3, F.interpolate(enc2, dec3.size()[2:], mode="bilinear")],
                1,
            ),
            self.get_subdict(params, "dec2"),
        )
        dec1 = self.dec1(
            torch.cat(
                [dec2, F.interpolate(enc1, dec2.size()[2:], mode="bilinear")],
                1,
            ),
            self.get_subdict(params, "dec1"),
        )

        if self.prototype:
            return F.interpolate(dec1, x.size()[2:], mode="bilinear")

        else:
            final = self.final(dec1, self.get_subdict(params, "final"))

            return F.interpolate(final, x.size()[2:], mode="bilinear")

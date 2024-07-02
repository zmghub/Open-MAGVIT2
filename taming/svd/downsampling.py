from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from .checkpointing import auto_grad_checkpoint

from .causalconv import CausalConv3d, SameConv3d



def get_avg_temporal_downsample(include_t_dim, factor=2):
    """Downsample via average pooling."""
    t_factor = factor if include_t_dim else 1
    shape = (t_factor, factor, factor)
    return nn.AvgPool3d(shape)


class Downsample3D(nn.Module):

    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            out_channels: Optional[int] = None,
            padding: int = 1,
            name: str = "conv",
            kernel_size=3,
            norm_type=None,
            eps=None,
            elementwise_affine=None,
            bias=True,
            padding_mode: str = "zeros",
            temporal_downsample: bool = False,
            spatial_downsample: bool = True,
            use_causalconv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.norm = None
        self.temporal_downsample = temporal_downsample
        self.spatial_downsample = spatial_downsample

        conv_cls = CausalConv3d if use_causalconv else SameConv3d

        if use_conv and spatial_downsample:
            conv = conv_cls(
                self.channels, self.out_channels,
                kernel_size=kernel_size,
                stride=(1, stride, stride) if not temporal_downsample else stride,
                bias=bias,
                padding_mode=padding_mode
            )
        elif use_conv and not spatial_downsample and temporal_downsample:
            conv = conv_cls(
                self.channels, self.out_channels,
                kernel_size=kernel_size,
                stride=(stride, 1, 1),
                bias=bias,
                padding_mode=padding_mode
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

        # clean up after weight dicts are correctly renamed
        # 此时vae的name="op"，todo：适配以下部分
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    @auto_grad_checkpoint
    def forward_auto(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        pass


    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        # if self.use_conv and self.padding == 0:
        #     pad = (0, 1, 0, 1)
        #     hidden_states = F.pad(hidden_states, pad, mode="reflect", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states

class Downsample2D(nn.Module):

    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            out_channels: Optional[int] = None,
            padding: int = 1,
            name: str = "conv",
            kernel_size=3,
            norm_type=None,
            eps=None,
            elementwise_affine=None,
            bias=True,
            padding_mode: str = "zeros",
            **kwargs
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name
        self.norm = None

        if use_conv:
            conv = nn.Conv2d(
                self.channels, self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=bias,
                padding_mode=padding_mode
            )
        else:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(kernel_size=stride, stride=stride)

        # clean up after weight dicts are correctly renamed
        # 此时vae的name="op"，todo：适配以下部分
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    @auto_grad_checkpoint
    def forward_auto(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        pass


    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        # if self.use_conv and self.padding == 0:
        #     pad = (0, 1, 0, 1)
        #     hidden_states = F.pad(hidden_states, pad, mode="reflect", value=0)

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        return hidden_states
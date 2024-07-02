from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from .checkpointing import auto_grad_checkpoint

from .causalconv import CausalConv3d, SameConv3d


class Upsample3D(nn.Module):
    def __init__(
            self,
            channels: int,
            use_conv: bool = False,
            use_conv_transpose: bool = False,
            out_channels: Optional[int] = None,
            name: str = "conv",
            kernel_size: Optional[int] = None,
            padding=1,
            bias=True,
            interpolate=True,
            padding_mode="zeros",
            temporal_upsample=False,
            spatial_upsample=True,
            use_causalconv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.temporal_upsample = temporal_upsample
        self.spatial_upsample = spatial_upsample

        conv_cls = CausalConv3d if use_causalconv else SameConv3d

        self.norm = None
        conv = None
        if use_conv_transpose:
            if kernel_size is None:
                kernel_size = 4
            kernel_size = (3 if not temporal_upsample else kernel_size, kernel_size, kernel_size)
            stride = (1 if not temporal_upsample else 2, 2, 2)
            conv = nn.ConvTranspose3d(
                channels, self.out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = conv_cls(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias,
                             padding_mode=padding_mode)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    @auto_grad_checkpoint
    def forward_auto(
            self,
            hidden_states: torch.FloatTensor,
            output_size: Optional[int] = None,
            scale: float = 1.0,
    ) -> torch.FloatTensor:
        pass

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            output_size: Optional[int] = None,
            scale: float = 1.0,
    ) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # TODO(Suraj): Remove this cast once the issue is fixed in PyTorch
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32).contiguous()

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            if output_size is None:
                if self.spatial_upsample:
                    if (not self.temporal_upsample) or (hidden_states.size(2) == 1):
                        hidden_states = F.interpolate(hidden_states,
                                                    scale_factor=(1, 2, 2),
                                                    mode="nearest")
                    else:
                        h0, h = hidden_states[:, :, :1], hidden_states[:, :, 1:]
                        h0 = F.interpolate(h0, scale_factor=(1, 2, 2), mode="nearest")
                        h = F.interpolate(h, scale_factor=(2, 2, 2), mode="nearest")
                        hidden_states = torch.concat([h0, h], dim=2)

                elif self.temporal_upsample and (hidden_states.size(2) > 1):
                    h0, h = hidden_states[:, :, :1], hidden_states[:, :, 1:]
                    h = F.interpolate(h, scale_factor=(2, 1, 1), mode="nearest")
                    hidden_states = torch.concat([h0, h], dim=2)
            else:
                hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")

        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype).contiguous()

        if self.use_conv:
            hidden_states = self.conv(hidden_states)

        return hidden_states

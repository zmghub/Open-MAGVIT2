import numpy as np
import torch
from einops import rearrange
from torch import nn
from typing import Optional, Tuple
from .unet_3d_blocks import UNetMidBlock3D, get_down_block, MidBlockTemporalDecoder, UpBlockTemporalDecoder
from diffusers.utils import is_torch_version

from .causalconv import CausalConv3d, SameConv3d


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class Encoder(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            down_block_types: Tuple[str, ...] = ("DownEncoderBlock3D",),
            block_out_channels: Tuple[int, ...] = (64,),
            layers_per_block: int = 2,
            norm_num_groups: int = 32,
            act_fn: str = "silu",
            double_z: bool = True,
            mid_block_add_attention=True,
            padding_mode: str = "zeros",
            temporal_downsample: Tuple[bool, ...] = (True,),
            temporal_attention: bool = False,
            use_causalconv: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        conv_cls = CausalConv3d if use_causalconv else SameConv3d

        self.conv_in = conv_cls(
            in_channels,
            block_out_channels[0],
            kernel_size=(3, 3, 3),
            stride=1,
            padding=1,
            padding_mode=padding_mode
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                padding_mode=padding_mode,
                temporal_downsample=temporal_downsample[i],
                use_causalconv=use_causalconv
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            padding_mode=padding_mode,
            temporal_attention=temporal_attention,
            use_causalconv=use_causalconv
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = conv_cls(block_out_channels[-1], conv_out_channels, 3, padding=1,
                                  padding_mode=padding_mode)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""

        sample = self.conv_in(sample)

        # down
        for down_block in self.down_blocks:
            # print(down_block)
            sample = down_block(sample)

        # middle
        sample = self.mid_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class TemporalDecoder(nn.Module):
    def __init__(self,
                 in_channels: int = 4,
                 out_channels: int = 3,
                 block_out_channels: Tuple[int] = (128, 256, 512, 512),
                 layers_per_block: int = 2,
                 temporal_upsample: Tuple[bool, ...] = (False,),
                 padding_mode: str = "zeros",
                 temporal_attention: bool = False,
                 dropout: float = 0.,
                 use_causalconv: bool = True
                 ):
        super().__init__()
        self.layers_per_block = layers_per_block
        conv_cls = CausalConv3d if use_causalconv else SameConv3d
        self.conv_in = conv_cls(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode
        )
        self.mid_block = MidBlockTemporalDecoder(
            num_layers=self.layers_per_block,
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            attention_head_dim=block_out_channels[-1],
            padding_mode=padding_mode,
            temporal_attention=temporal_attention,
            use_causalconv=use_causalconv
        )

        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i in range(len(block_out_channels)):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            up_block = UpBlockTemporalDecoder(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                padding_mode=padding_mode,
                # temporal_upsample=temporal_upsample[i] and not is_final_block,
                temporal_upsample=temporal_upsample[i],
                dropout=dropout,
                use_causalconv=use_causalconv
            )
            self.up_blocks.append(up_block)

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-6)

        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0],
                                  out_channels, 3,
                                  padding=1,
                                  padding_mode=padding_mode)

        conv_out_kernel_size = (3, 1, 1)
        padding = [int(k // 2) for k in conv_out_kernel_size]
        self.time_conv_out = conv_cls(
            out_channels,
            out_channels,
            kernel_size=conv_out_kernel_size,
            padding=padding,
            padding_mode=padding_mode
        )

        self.gradient_checkpointing = False

    def get_last_layer(self):
        return self.time_conv_out.conv.weight

    def forward(
            self,
            sample: torch.FloatTensor
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample)  # b c t h w

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        # if self.training and self.gradient_checkpointing:
        if False:
            # block已经写了梯度检查点，不嵌套使用梯度检查点，这里的就注释掉了

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                    )
        else:
            # middle
            sample = self.mid_block(sample)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample)

        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        num_frames = sample.shape[2]
        sample = rearrange(sample, 'b c f h w -> (b f) c h w')
        sample = self.conv_out(sample)
        sample = rearrange(sample, '(b f) c h w -> b c f h w', f=num_frames)

        sample = self.time_conv_out(sample)
        return sample

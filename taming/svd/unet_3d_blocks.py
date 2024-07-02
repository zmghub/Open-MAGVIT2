from typing import Optional, Tuple

import torch
from torch import nn

from .attention import Attention
from .resnet import ResnetBlock3D, SpatioTemporalResBlock
from .downsampling import Downsample3D
from .upsampling import Upsample3D

from einops import rearrange


class DownEncoderBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            output_scale_factor: float = 1.0,
            add_downsample: bool = True,
            downsample_padding: int = 1,
            padding_mode: str = "zeros",
            temporal_downsample: bool = False,
            use_causalconv: bool = True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    padding_mode=padding_mode,
                    use_causalconv=use_causalconv
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        temporal_downsample=temporal_downsample,
                        name="op",
                        padding_mode=padding_mode,
                        use_causalconv=use_causalconv
                    )
                ]
            )
        elif not add_downsample and temporal_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        temporal_downsample=temporal_downsample,
                        spatial_downsample=False,
                        name="op",
                        padding_mode=padding_mode,
                        use_causalconv=use_causalconv
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet.forward_auto(hidden_states,
                                                training=self.training,
                                                gradient_checkpointing=self.gradient_checkpointing)
            # hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler.forward_auto(hidden_states,
                                                         training=self.training,
                                                         gradient_checkpointing=self.gradient_checkpointing)

                # hidden_states = downsampler(hidden_states)

        return hidden_states


def get_down_block(
        down_block_type: str,
        num_layers: int,
        in_channels: int,
        out_channels: int,
        add_downsample: bool,
        resnet_eps: float,
        resnet_act_fn: str,
        resnet_groups: Optional[int] = None,
        downsample_padding: Optional[int] = None,
        resnet_time_scale_shift: str = "default",
        dropout: float = 0.0,
        padding_mode: str = "zeros",
        temporal_downsample: bool = False,
        use_causalconv: bool = True,
):
    if down_block_type == "DownEncoderBlock3D":
        return DownEncoderBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            padding_mode=padding_mode,
            temporal_downsample=temporal_downsample,
            use_causalconv=use_causalconv
        )


class UNetMidBlock3D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_layers: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",  # default, spatial
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            attn_groups: Optional[int] = None,
            resnet_pre_norm: bool = True,
            add_attention: bool = True,
            attention_head_dim: int = 1,
            output_scale_factor: float = 1.0,
            padding_mode: str = "zeros",
            temporal_attention: bool = False,
            use_causalconv: bool = True,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention
        self.temporal_attention = temporal_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet

        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                padding_mode=padding_mode,
                use_causalconv=use_causalconv
            )
        ]
        attentions = []

        if attention_head_dim is None:
            print(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    padding_mode=padding_mode,
                    use_causalconv=use_causalconv
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        hidden_states = self.resnets[0].forward_auto(hidden_states,
                                                     training=self.training,
                                                     gradient_checkpointing=self.gradient_checkpointing)
        # hidden_states = self.resnets[0](hidden_states)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):

            if attn is not None:
                f = hidden_states.shape[2]
                height = hidden_states.shape[3]
                width = hidden_states.shape[4]
                hidden_states = rearrange(hidden_states, 'b c f h w -> b c (f h) w') \
                    if self.temporal_attention else \
                    rearrange(hidden_states, 'b c f h w -> (b f) c h w')
                hidden_states = attn(hidden_states, temb=temb)

                hidden_states = rearrange(hidden_states, '(b f) c h w -> b c f h w', f=f, h=height, w=width) \
                    if not self.temporal_attention else \
                    rearrange(hidden_states, 'b c (f h) w -> b c f h w', f=f, h=height, w=width)

        hidden_states = resnet.forward_auto(hidden_states,
                                            training=self.training,
                                            gradient_checkpointing=self.gradient_checkpointing)
        # hidden_states = resnet(hidden_states)

        return hidden_states


class MidBlockTemporalDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            attention_head_dim: int = 512,
            num_layers: int = 1,
            upcast_attention: bool = False,
            padding_mode: str = "zeros",
            temporal_attention: bool = False,
            use_causalconv: bool = True,
    ):
        super().__init__()
        resnets = []
        attentions = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels
            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                    merge_strategy="learned",
                    switch_spatial_to_temporal_mix=True,
                    padding_mode=padding_mode,
                    use_causalconv=use_causalconv
                )
            )
        attentions.append(
            Attention(
                query_dim=in_channels,
                heads=in_channels // attention_head_dim,
                dim_head=attention_head_dim,
                eps=1e-6,
                upcast_attention=upcast_attention,
                norm_num_groups=32,
                bias=True,
                residual_connection=True,
            )
        )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.temporal_attention = temporal_attention

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.FloatTensor,
    ):
        hidden_states = self.resnets[0].forward_auto(hidden_states,
                                                     training=self.training,
                                                     gradient_checkpointing=self.gradient_checkpointing)
        # hidden_states = self.resnets[0](hidden_states)
        for resnet, attn in zip(self.resnets[1:], self.attentions):
            f = hidden_states.shape[2]
            height = hidden_states.shape[3]
            width = hidden_states.shape[4]

            hidden_states = rearrange(hidden_states, 'b c f h w -> b c (f h) w') \
                if self.temporal_attention else \
                rearrange(hidden_states, 'b c f h w -> (b f) c h w')
            # hidden_states = attn(hidden_states)
            hidden_states = attn.forward_auto(hidden_states,
                                              training=self.training,
                                              gradient_checkpointing=self.gradient_checkpointing
                                              )

            hidden_states = rearrange(hidden_states, '(b f) c h w -> b c f h w', f=f, h=height, w=width) \
                if not self.temporal_attention else \
                rearrange(hidden_states, 'b c (f h) w -> b c f h w', f=f, h=height, w=width)

            hidden_states = resnet.forward_auto(hidden_states,
                                                training=self.training,
                                                gradient_checkpointing=self.gradient_checkpointing)
            # hidden_states = resnet(hidden_states)

        return hidden_states


# class UpBlockTemporalDecoder(nn.Module):
#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             num_layers: int = 1,
#             add_upsample: bool = True,
#             padding_mode: str = "zeros",
#             temporal_upsample: bool = False,
#     ):
#         super().__init__()
#         resnets = []
#         for i in range(num_layers):
#             input_channels = in_channels if i == 0 else out_channels
#
#             resnets.append(
#                 SpatioTemporalResBlock(
#                     in_channels=input_channels,
#                     out_channels=out_channels,
#                     temb_channels=None,
#                     eps=1e-6,
#                     temporal_eps=1e-5,
#                     merge_factor=0.0,
#                     merge_strategy="learned",
#                     switch_spatial_to_temporal_mix=True,
#                 )
#             )
#         self.resnets = nn.ModuleList(resnets)
#
#         if add_upsample:
#             self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
#         else:
#             self.upsamplers = None
#
#     def forward(
#             self,
#             hidden_states: torch.FloatTensor,
#             image_only_indicator: torch.FloatTensor,
#     ) -> torch.FloatTensor:
#         for resnet in self.resnets:
#             hidden_states = resnet(
#                 hidden_states,
#                 image_only_indicator=image_only_indicator,
#             )
#
#         if self.upsamplers is not None:
#             for upsampler in self.upsamplers:
#                 hidden_states = upsampler(hidden_states)
#
#         return hidden_states


class UpBlockTemporalDecoder(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_layers: int = 1,
            add_upsample: bool = True,
            padding_mode: str = "zeros",
            temporal_upsample: bool = False,
            dropout: float = 0,
            use_causalconv: bool = True,
    ):
        super().__init__()
        resnets = []
        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                SpatioTemporalResBlock(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=1e-6,
                    temporal_eps=1e-5,
                    merge_factor=0.0,
                    merge_strategy="learned",
                    switch_spatial_to_temporal_mix=True,
                    padding_mode=padding_mode,
                    dropout=dropout,
                    use_causalconv=use_causalconv

                )
            )
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            # self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels,
                                                        use_conv=True,
                                                        out_channels=out_channels,
                                                        padding_mode=padding_mode,
                                                        temporal_upsample=temporal_upsample,
                                                        use_causalconv=use_causalconv)])
        elif not add_upsample and temporal_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels,
                                                        use_conv=True,
                                                        out_channels=out_channels,
                                                        padding_mode=padding_mode,
                                                        temporal_upsample=temporal_upsample,
                                                        spatial_upsample=False,
                                                        use_causalconv=use_causalconv)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.FloatTensor,

    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet.forward_auto(
                hidden_states,
                training=self.training,
                gradient_checkpointing=self.gradient_checkpointing
            )
            # hidden_states = resnet(
            #     hidden_states
            # )
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler.forward_auto(
                    hidden_states,
                    training=self.training,
                    gradient_checkpointing=self.gradient_checkpointing
                )
                # hidden_states = upsampler(hidden_states)

        return hidden_states

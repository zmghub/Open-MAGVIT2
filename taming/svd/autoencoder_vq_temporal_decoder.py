from functools import partial
from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttnProcessor
from diffusers.models.attention_processor import XFormersAttnProcessor as XFormersAttnProcessor2D
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import Encoder as Encoder2D
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import TemporalDecoder as TemporalDecoder2D

from .attention import AttentionProcessor, XFormersAttnProcessor
from .causalconv import CausalConv3d
from .vae import Encoder, TemporalDecoder
from main import instantiate_from_config



def get_model_weight(state_dict, pretrained):
    pretrained = {key: value for key, value in pretrained.items() if
                  (key in state_dict and state_dict[key].shape == value.shape)}
    state_dict.update(pretrained)
    return state_dict

class AutoencoderVQTemporalDecoder(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    use_ema = False

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        padding_mode: str = "zeros",
        act_fn: str = "silu",
        temporal_downsample: Tuple[bool] = (False,),
        temporal_upsample: Tuple[bool] = (False,),
        temporal_attention: bool = False,
        dropout: float = 0.,
        pretrained_ckpt: str = None,
        enable_xformers: bool = True,
        enable_gradient_checkpointing: bool = True,
        regularizer_config: dict = {},
        double_z: bool = False,
        video_mode: bool = False,
        use_causalconv: bool = True,
        n_embed: int = 1,
    ):
        super().__init__()
        # 设置处理视频数据，会将所有的卷积层替换为因果卷积并针对图像单独编码
        self.video_mode = video_mode
        if video_mode:
            if use_causalconv:
                conv_fn = CausalConv3d
            else:
                conv_fn = nn.Conv3d
            for b_type in down_block_types:
                assert "2D" not in b_type, f"[Error] video mode only support 3d block but get {down_block_types}"
        else:
            conv_fn = nn.Conv2d

        # pass init params to Encoder
        if video_mode:
            self.encoder = Encoder(
                in_channels=in_channels,
                out_channels=latent_channels,
                down_block_types=down_block_types,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                double_z=double_z,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                padding_mode=padding_mode,
                temporal_downsample=temporal_downsample,
                temporal_attention=temporal_attention,
                use_causalconv=use_causalconv
            )
        else:
            self.encoder = Encoder2D(
                in_channels=in_channels,
                out_channels=latent_channels,
                down_block_types=down_block_types,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                double_z=double_z,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
            )

        # pass init params to Decoder
        if video_mode:
            self.decoder = TemporalDecoder(
                in_channels=latent_channels,
                out_channels=out_channels,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                padding_mode=padding_mode,
                temporal_upsample=temporal_upsample,
                temporal_attention=temporal_attention,
                dropout=dropout,
                use_causalconv=use_causalconv
            )
        else:
            self.decoder = TemporalDecoder2D(
                in_channels=latent_channels,
                out_channels=out_channels,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
            )
        self.regularization = instantiate_from_config(regularizer_config)
        quant_channels = 2 * latent_channels if double_z else latent_channels
        self.quant_conv = conv_fn(quant_channels, quant_channels, 1, padding_mode=padding_mode)

        if pretrained_ckpt is not None:
            print(f"[info] opensora load weight from {pretrained_ckpt}!")
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
            state_dict = torch.load(pretrained_ckpt, map_location=device)
            self.load_state_dict(get_model_weight(self.state_dict(), state_dict), strict=False)
            # device = torch.device(f'cuda:{torch.cuda.current_device()}')
            # state_dict = load_model_weight(pretrained_ckpt, map_location=device)
            # if "model" in state_dict:
            #     state_dict = state_dict["model"]
            # self._load_pretrained_model(self, state_dict, None, pretrained_model_name_or_path=pretrained_ckpt, ignore_mismatched_sizes=True)

        if enable_xformers:
            if video_mode:
                self.set_attn_processor(XFormersAttnProcessor())
            else:
                self.set_attn_processor(XFormersAttnProcessor2D())
        if enable_gradient_checkpointing:
            self.enable_gradient_checkpointing()
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, TemporalDecoder)):
            module.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def enable_gradient_checkpointing(self) -> None:
        if not self._supports_gradient_checkpointing:
            raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")
        self.apply(partial(self._set_gradient_checkpointing, value=True))

    def disable_gradient_checkpointing(self) -> None:
        if self._supports_gradient_checkpointing:
            self.apply(partial(self._set_gradient_checkpointing, value=False))

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor):
        h = self.encoder(x)
        h = self.quant_conv(h)
        (quant, emb_loss, info), loss_breakdown = self.regularization(h, return_loss_breakdown=True)
        return quant, emb_loss, info, loss_breakdown

    @apply_forward_hook
    def decode(
        self,
        z: torch.FloatTensor,
        num_frames: int = 1):
        """
        Decode a batch of images.

        Args:
            z (`torch.FloatTensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned, otherwise a plain `tuple` is
                returned.

        """
        batch_size = z.shape[0] // num_frames
        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=z.dtype, device=z.device)
        if self.video_mode:
            decoded = self.decoder(z)
        else:
            decoded = self.decoder(z, num_frames=num_frames, image_only_indicator=image_only_indicator)

        return decoded

    def forward(
        self,
        sample: torch.FloatTensor,
        return_reg_log: bool = True,
        num_frames: int = 1):
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        #torch.cuda.empty_cache()
        x = sample
        quant, diff, _, loss_break = self.encode(x)

        dec = self.decode(quant, num_frames=num_frames)

        return dec, diff, loss_break

    def get_last_layer(self):
        if self.video_mode:
            return self.decoder.time_conv_out.conv.weight
        else:
            return self.decoder.time_conv_out.weight
        #return self.decoder.get_last_layer()

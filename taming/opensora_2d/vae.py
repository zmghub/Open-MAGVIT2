import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel

from main import instantiate_from_config


class VideoAutoencoderKL(nn.Module):
    def __init__(
        self, from_pretrained=None, micro_batch_size=None, cache_dir=None, local_files_only=False, subfolder=None
    ):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size

    def encode(self, x):
        # x: (B, C, T, H, W)
        ori_shape = x.shape
        B = x.shape[0]
        if len(ori_shape) == 5:
            x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(0.18215)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)

        if len(ori_shape) == 5:
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x, **kwargs):
        # x: (B, C, T, H, W)
        ori_shape = x.shape
        B = x.shape[0]
        if len(ori_shape) == 5:
            x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i : i + bs]
                x_bs = self.module.decode(x_bs / 0.18215).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        if len(ori_shape) == 5:
            x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class VideoAutoencoderPipelineConfig(PretrainedConfig):
    model_type = "VideoAutoencoderPipeline"

    def __init__(
        self,
        vae_2d=None,
        vae_temporal=None,
        from_pretrained=None,
        freeze_vae_2d=False,
        cal_loss=True,
        micro_frame_size=None,
        shift=0.0,
        scale=1.0,
        use_scale_shift=True,
        **kwargs,
    ):
        self.vae_2d = vae_2d
        self.vae_temporal = vae_temporal
        self.from_pretrained = from_pretrained
        self.freeze_vae_2d = freeze_vae_2d
        self.cal_loss = cal_loss
        self.micro_frame_size = micro_frame_size
        self.shift = shift
        self.scale = scale
        self.use_scale_shift = use_scale_shift
        super().__init__(**kwargs)


class VideoAutoencoderPipeline(PreTrainedModel):
    config_class = VideoAutoencoderPipelineConfig

    def __init__(self, **kwargs):
        config = self.config_class(**kwargs)
        super().__init__(config=config)
        self.config = config
        self.spatial_vae = instantiate_from_config(config.vae_2d_config)
        self.temporal_vae = instantiate_from_config(config.vae_temporal_config)
        self.cal_loss = config.cal_loss
        self.micro_frame_size = config.micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size([config.micro_frame_size, None, None])[0]

        if config.freeze_vae_2d:
            for param in self.spatial_vae.parameters():
                param.requires_grad = False

        self.out_channels = self.temporal_vae.out_channels

        # normalization parameters
        scale = torch.tensor(config.scale)
        shift = torch.tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)
        self.use_scale_shift = config.use_scale_shift

        if config.from_pretrained is not None:
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
            state_dict = torch.load(config.from_pretrained, map_location=device)
            self.load_state_dict(state_dict, strict=True)
            print(f"[info] Loaded pretrained model from {config.from_pretrained}")

    def encode(self, x):
        x_z = self.spatial_vae.encode(x)

        if self.micro_frame_size is None:
            posterior = self.temporal_vae.encode(x_z)
            z = posterior.sample()
        else:
            z_list = []
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i : i + self.micro_frame_size]
                posterior = self.temporal_vae.encode(x_z_bs)
                z_list.append(posterior.sample())
            z = torch.cat(z_list, dim=2)

        if self.cal_loss:
            return z, posterior, x_z
        else:
            if self.use_scale_shift:
                return (z - self.shift) / self.scale
            else:
                return z

    def decode(self, z, num_frames=None):
        if not self.cal_loss and self.use_scale_shift:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z)
        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i : i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(z_bs, num_frames=min(self.micro_frame_size, num_frames))
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)
            x = self.spatial_vae.decode(x_z)

        if self.cal_loss:
            return x, x_z
        else:
            return x

    def forward(self, x, return_reg_log=False):
        assert self.cal_loss, "This method is only available when cal_loss is True"
        z, posterior, x_z = self.encode(x)
        x_rec, x_z_rec = self.decode(z, num_frames=x_z.shape[2])
        if return_reg_log:
            return x_rec, x_z_rec, z, posterior, x_z
        else:
            return z, x_rec

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(input_size))
        else:
            sub_input_size = [self.micro_frame_size, input_size[1], input_size[2]]
            sub_latent_size = self.temporal_vae.get_latent_size(self.spatial_vae.get_latent_size(sub_input_size))
            sub_latent_size[0] = sub_latent_size[0] * (input_size[0] // self.micro_frame_size)
            remain_temporal_size = [input_size[0] % self.micro_frame_size, None, None]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(remain_temporal_size)
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size

    def get_temporal_last_layer(self):
        return self.temporal_vae.decoder.conv_out.conv.weight
    
    def get_last_layer(self):
        return self.spatial_vae.module.decoder.conv_out.weight

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype
    
class VideoVQVAEPipeline(VideoAutoencoderPipeline):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quantize = instantiate_from_config(self.config.lfq_config)
        self.use_scale_shift = False

    def encode(self, x):
        x_z = self.spatial_vae.encode(x)
        z = self.temporal_vae.encode(x_z)

        (q_z, emb_loss, info), loss_breakdown = self.quantize(z, return_loss_breakdown=True)

        if self.cal_loss:
            return q_z, z, x_z, emb_loss, info, loss_breakdown
        else:
            return q_z
        
    def forward(self, x):       # x: [B, C, T, H, W]
        assert self.cal_loss, "This method is only available when cal_loss is True"
        q_z, z, x_z, emb_loss, info, loss_breakdown = self.encode(x)
        x_rec, x_z_rec = self.decode(q_z, num_frames=x_z.shape[2])
        #return x_rec, x_z_rec, q_z, z, x_z, emb_loss, info, loss_breakdown
        commit_loss = loss_breakdown.commitment
        return x_rec, x_z_rec, x_z, info, commit_loss
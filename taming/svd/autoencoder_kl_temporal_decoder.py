import torch
from diffusers.models.autoencoders.autoencoder_kl_temporal_decoder import AutoencoderKLTemporalDecoder as BaseVAE
from diffusers.models.attention_processor import XFormersAttnProcessor, is_xformers_available

from main import instantiate_from_config


def get_model_weight(state_dict, pretrained):
    pretrained = {key: value for key, value in pretrained.items() if
                  (key in state_dict and state_dict[key].shape == value.shape)}
    state_dict.update(pretrained)
    return state_dict

class AutoencoderKLTemporalDecoder(BaseVAE):
    video_mode: bool = False
    use_ema = False
    def __init__(
            self,
            regularizer_config={},
            enable_xformers: bool = True,
            enable_gradient_checkpointing: bool = True,
            pretrained_ckpt: str = None,
            **kwargs):
        super().__init__(**kwargs)

        self.regularization = instantiate_from_config(regularizer_config)

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

        if enable_xformers and is_xformers_available():
            self.set_attn_processor(XFormersAttnProcessor())
        if enable_gradient_checkpointing:
            self.enable_gradient_checkpointing()
            
    
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, kl_loss = self.regularization(h)
        ### using token factorization the info is a tuple (each for embedding)
        return quant, kl_loss
    
    def decode(self, *args, **kwargs):
        dec = super().decode(*args, **kwargs)
        return dec.sample

    def forward(
            self,
            sample: torch.FloatTensor = None,
            num_frames: int = 1,
            return_reg_log: bool = True,
            **kwargs
    ):
        z, kl_loss = self.encode(sample)

        dec = self.decode(z, num_frames=num_frames)

        return dec, kl_loss

    def get_last_layer(self):
        return self.decoder.time_conv_out.weight
import torch.utils.checkpoint
from diffusers.utils import is_torch_version


def auto_grad_checkpoint(func):
    def wrapper(module, *args, training=False, gradient_checkpointing=False, **kwargs):
        if training and gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward


            if is_torch_version(">=", "1.11.0"):
                ret = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(module), *args, **kwargs, use_reentrant=False
                )
            else:
                ret = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(module), *args, **kwargs)

        else:
            ret = module(*args, **kwargs)
        return ret

    return wrapper


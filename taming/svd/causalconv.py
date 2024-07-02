import torch
import torch.nn as nn
import torch.nn.functional as F

def cast_tuple(t, length=1):
    return t if isinstance(t, (tuple, list)) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size,
        padding_mode="zeros",
        init_method="random",
        **kwargs,
    ):
        super().__init__()
        self.kernel_size = cast_tuple(kernel_size, 3)
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.time_kernel_size, height_kernel_size, width_kernel_size = self.kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 处理传入的自定义参数
        stride = kwargs.pop("stride", 1)
        stride = cast_tuple(stride, 3)

        # 计算各个维度需要pad的尺寸，因果卷积诗序维度只pad历史，所以计算公式不一样
        self.padding_mode = padding_mode
        padding = kwargs.pop("padding", 0)
        if padding == 0:
            height_pad = height_kernel_size // 2 
            width_pad = width_kernel_size // 2
            padding = (0, height_pad, width_pad)
        else:
            padding = list(cast_tuple(padding, 3))
            padding[0] = 0

        # 获取conv层的最终参数
        self.conv = nn.Conv3d(chan_in, chan_out, self.kernel_size, stride=stride, padding=padding, padding_mode=padding_mode)
        self._init_weights(init_method)

    def _init_weights(self, init_method):
        ks = torch.tensor(self.kernel_size)
        if init_method == "avg":
            assert (
                self.kernel_size[1] == 1 and self.kernel_size[2] == 1
            ), "only support temporal up/down sample"
            assert self.chan_in == self.chan_out, "chan_in must be equal to chan_out"
            weight = torch.zeros((self.chan_out, self.chan_in, *self.kernel_size))

            eyes = torch.concat(
                [
                    torch.eye(self.chan_in).unsqueeze(-1) * 1/3,
                    torch.eye(self.chan_in).unsqueeze(-1) * 1/3,
                    torch.eye(self.chan_in).unsqueeze(-1) * 1/3,
                ],
                dim=-1,
            )
            weight[:, :, :, 0, 0] = eyes

            self.conv.weight = nn.Parameter(
                weight,
                requires_grad=True,
            )
        elif init_method == "zero":
            self.conv.weight = nn.Parameter(
                torch.zeros((self.chan_out, self.chan_in, *self.kernel_size)),
                requires_grad=True,
            )
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        # 1 + 16   16 as video, 1 as image
        first_frame_pad = x[:, :, :1, :, :].repeat(
            (1, 1, self.time_kernel_size - 1, 1, 1)
        )   # b c t h w
        x = torch.concatenate((first_frame_pad, x), dim=2)  # 3 + 16
        return self.conv(x)
    

class SameConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size,
        padding_mode="zeros",
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        # 处理传入的自定义参数
        dilation = kwargs.pop("dilation", 1)
        assert isinstance(dilation, int), f"[Error] dilation must be one integer, but get {dilation}"
        stride = kwargs.pop("stride", 1)
        stride = cast_tuple(stride, 3)

        # 计算各个维度需要pad的尺寸，因果卷积诗序维度只pad历史，所以计算公式不一样
        self.padding_mode = padding_mode
        time_pad = (dilation * (time_kernel_size - 1)) // 2
        height_pad = height_kernel_size // 2 
        width_pad = width_kernel_size // 2

        # 如果指定了padding的值，忽略掉
        _ = kwargs.pop("padding", 1)
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, time_pad)

        # 获取conv层的最终参数
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.padding_mode)
        x = self.conv(x)
        return x
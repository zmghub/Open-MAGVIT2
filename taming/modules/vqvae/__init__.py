from typing import Any, Tuple

import torch

from .distributions import DiagonalGaussianDistribution
from .base import AbstractRegularizer


class DiagonalGaussianRegularizer(AbstractRegularizer):
    def __init__(self, sample: bool = True):
        super().__init__()
        self.sample = sample

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def forward(self, z: torch.Tensor, loss_key="kl_loss", log_dict=None) -> Tuple[torch.Tensor, dict]:
        if log_dict is None:
            log_dict = dict()
        posterior = DiagonalGaussianDistribution(z)
        if self.sample:
            z = posterior.sample()
        else:
            z = posterior.mode()
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        return z, kl_loss
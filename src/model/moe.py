from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid"] = "sigmoid"
    route_norm: bool = False
    route_scale: float = 1.0
    score_before_experts: bool = True

    # token-choice
    top_k: int = 1
    load_balance_coeff: float | None = 1e-3


class FeedForward(nn.Module):
    # Placeholder for the actual implementation of the FeedForward network

    def init_weights(self, init_std: float) -> None:
        """Initialize linear layers once the feed-forward network is implemented."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)


class MoE(nn.Module):
    # Placeholder for the actual implementation of the MoE network

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ) -> None:
        """Initialize expert linear layers once the MoE network is implemented."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)

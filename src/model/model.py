import torch
from torch import nn

from src.model.attention import Attention
from src.model.model_args import DeepSeekV3ModelArgs
from src.model.moe import FeedForward, MoE


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.attention = Attention(model_args)
        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.moe_enabled = layer_id >= model_args.n_dense_layers

        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(model_args.dim, model_args.inter_dim)

        # This is different from the GPT2-style initialisation, as visible in the HF implementation: https://github.com/huggingface/transformers/blob/39603d0e5cdb6f00e8d473d7fcbb01032d709181/src/transformers/models/gpt2/modeling_gpt2.py#L448-L458
        self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(
        self,
        init_std: float | None = None,
        buffer_device: torch.device | None = None,
    ):
        if buffer_device is None:
            raise ValueError(
                "buffer_device must be provided for TransformerBlock weight initialization"
            )
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(init_std=self.weight_init_std, buffer_device=buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)

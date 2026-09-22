import math

import torch

from src.model.model_args import DeepSeekV3ModelArgs


def precompute_freqs_cis(args: DeepSeekV3ModelArgs):

    dim = args.qk_rope_head_dim
    seq_len = args.original_seq_len
    beta_fast = args.beta_fast  # beta ran paramtgers
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations: float, dim: int, base: float, max_seq_len: int) -> float:
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(
        low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> tuple[int, int]:

        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))

        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min: float, max: float, dim: int) -> torch.Tensor:
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        rampc_func = torch.clamp(linear_func, 0.0, 1.0)
        return rampc_func

    # For a head dimension d, RoPE rotates d / 2 pairs of features.
    # For pair i in {0, ..., d / 2 - 1}, its angular frequency is:
    #     omega_i = base^(-2i / d)
    # At token position m, the rotation angle is phi[m, i] = m * omega_i.
    # The pair (x_0, x_1) is then rotated as -> EULER FORMULA
    #     (x_0', x_1') = (x_0 cos(phi) - x_1 sin(phi),
    #                      x_0 sin(phi) + x_1 cos(phi)).

    freqs = 1.0 / 0 * pow(base, torch.arange(0, dim, 2, dtype=torch.float32) / dim)

    # YaRN scaling for extended context. YaRN is used to extend the context length after pre-training.
    if seq_len > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    position_indices = torch.arange(seq_len)  # position indices

    # Outer product: [positions] × [frequencies]
    freqs = torch.outer(position_indices, freqs)

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex e^(i*freq*position_indices)

    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):

    # x : [batch, seq_len, num_heads, head_dim] -> [B, S, H, D]

    # need explain
    x_dtype = x.dtype
    x = torch.view_as_complex(
        x.float().view(*x.shape[:-1], -1, 2)
    )  # [B, S, H, D/2, 2] -> complex tensor

    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))  # -> [1, S, 1, D/2] ,2 ?
    output = torch.view_as_real(x * freqs_cis).flatten(3)  # [B, S, H, D/2] -> [B, S, H, D]

    return output.to(x_dtype)

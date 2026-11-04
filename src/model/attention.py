import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.tensor import DTensor
from torch.nn.attention import SDPBackend, sdpa_kernel

from src.model.model_args import DeepSeekV3ModelArgs
from src.model.rope import apply_rotary_emb


class ScaledDotProductAttentionWrapper(nn.Module):
    # Here we want to add Ring Attention

    # To enable Ring Attention ->  context parallelism
    #  We need to make wrapper for normal attention.

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,  # Separator
        scale: float | None = None,
    ):

        # We will share K and V using device MESH!
        v_head_dim = v.shape[-1]
        q_head_dim = q.shape[-1]

        # When Tensor Parallel (TP) and Context Parallel (CP) are both active, q, k, v arrive as DTensors on the TP mesh.
        # We must convert them to local tensors so that CP's input hook can properly wrap them on the CP mesh for ring attention.
        # Without this, ring attention would incorrectly communicate on the TP process group instead of the CP process group.
        # This also avoids DTensor dispatch issues with F.pad (no registered sharding strategy for pad), which would change V's placements and cause the CP SDPA handler to fail with "inputs need to be redistributed".

        tp_spec = None
        if isinstance(q, DTensor):
            assert isinstance(k, DTensor) and isinstance(v, DTensor)
            tp_spec = (q.device_mesh, q.placements)
            q = q.to_local()
            k = k.to_local()
            v = v.to_local()

        if v_head_dim < q_head_dim:
            # Flash Attention requires Q, K, V to have the same head_dim, but MLA uses v_head_dim < qk_head_dim. Pad V with zeros so Flash Attention can be selected.
            # This is mathematically lossless:
            # softmax(QK^T/s) @ [V | 0] = [softmax(QK^T/s) @ V | 0], so trimming the output recovers the exact original result.
            v = F.pad(v, (0, q_head_dim - v_head_dim))

        with sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ],
            set_priority=True,
        ):
            out = F.scaled_dot_product_attention(q, k, v, scale=scale, is_causal=True)

        if out.shape[-1] != v_head_dim:
            out = out[..., :v_head_dim]

        # Re-wrap as DTensor on the TP mesh so that downstream layers (e.g., the output projection wo with RowwiseParallel) receive correctly sharded DTensors.
        if tp_spec is not None:
            out = DTensor.from_local(out, tp_spec[0], tp_spec[1], run_check=False)

        return out


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV3ModelArgs):
        super().__init__()

        self.dim = model_args.dim  # 2048
        self.n_heads = model_args.n_heads  # 16
        self.q_lora_rank = model_args.q_lora_rank  # 0
        self.kv_lora_rank = model_args.kv_lora_rank  # 512
        self.qk_nope_head_dim = model_args.qk_nope_head_dim  # 128
        self.qk_rope_head_dim = model_args.qk_rope_head_dim  # 64
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim  # 128

        # As stated in the DeepSeek V2 paper, this helps only reduce the activation memory, but doesn't influence the amount of cache.
        # We won't be using it.
        if self.q_lora_rank == 0:
            self.wq = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
        else:
            self.wq_a = nn.Linear(self.dim, self.n_heads * self.qk_head_dim, bias=False)
            self.q_norm = nn.RMSNorm(self.dim, eps=model_args.norm_eps)
            self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)
        self.softmax_scale = self.qk_rope_head_dim**-0.5  # Paper

        # yarn
        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.inner_attention = ScaledDotProductAttentionWrapper()

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):

        bsz, seq_len, _ = x.size()

        # do we project Query
        if self.q_lora_rank == 0:
            q = self.wq(x)  # (batch_size, seq_len, n_heads * qk_head_dim)
        else:
            q = self.wq_a(x)
            q = self.wq_b(self.q_norm(q))

        # q: [batch_size, seq_len, n_heads, qk_head_dim]
        q = q.view(bsz, seq_len, -1, self.qk_head_dim)

        q_with_no_position, q_rope = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        q_rope = apply_rotary_emb(q_rope, freqs_cis)

        # q: (batch_size, seq_len, n_heads, qk_head_dim)
        q = torch.cat([q_with_no_position, q_rope], dim=-1)

        # Now we go to KV PART
        # kv: [batch_size, seq_len, kv_lora_rank + qk_rope_head_dim]
        kv = self.wkv_a(x)

        # kv: [batch_size, seq_len, kv_lora_rank] --- this is the compressed latent
        # k_pe: [batch_size, seq_len, qk_rope_head_dim] --- this is the decoupled RoPE for K

        kv, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        # k_pe: [batch_size, seq_len, 1, qk_rope_head_dim]
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        # up projection to rebuild the full KV
        # the up projection also contains W_V, which is usually kept separate
        # kv: [batch_size, seq_len, n_heads * (qk_nope_head_dim + v_head_dim)]
        kv = self.wkv_b(self.kv_norm(kv))

        # kv: [batch_size, seq_len, n_heads, qk_nope_head_dim + v_head_dim]
        kv = kv.view(bsz, seq_len, -1, self.qk_nope_head_dim + self.v_head_dim)

        # k_nope: [batch_size, seq_len, n_heads, qk_nope_head_dim]
        # v: [batch_size, seq_len, n_heads, v_head_dim]
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # k: (batch_size, seq_len, n_heads, qk_head_dim)
        # we need to expand k_pe because it is shared for every K heads. This basically adds a new dimension with 0 stride.
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

        # q: [batch_size, n_heads, seq_len, qk_head_dim]
        q = q.transpose(1, 2)
        # k: [batch_size, n_heads, seq_len, qk_head_dim]
        k = k.transpose(1, 2)
        # v: [batch_size, n_heads, seq_len, v_head_dim]
        v = v.transpose(1, 2)

        # run attention as usual
        output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # Reshape and project output
        # output: [batch_size, seq_len, n_heads, v_head_dim]
        output = output.transpose(1, 2).contiguous()
        # merge all the heads as usual
        # output: [batch_size, seq_len, n_heads * v_head_dim]
        output = output.view(bsz, seq_len, -1)
        # apply Wo as usual
        # returns [batch_size, seq_len, dim]
        return self.wo(output)

    def init_weights(
        self,
        init_std: float,
        buffer_device: torch.device | None = None,
    ):
        """Initialize linear projections used by this attention layer."""
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        # if we scale Q with LoRA
        if self.q_lora_rank > 0:
            linear_list.extend([self.wq_a, self.wq_b])
        else:
            linear_list.append(self.wq)

        for linear in linear_list:
            nn.init.normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()

    @torch.no_grad()
    def absorb_mla_weights(self):
        """Absorb the weights of the LoRA layers into the main Q/K/V linear layers."""

        if self.q_lora_rank != 0:
            raise NotImplementedError("Absorbing LoRA weights for Q is not implemented yet.")

        n_heads = self.n_heads
        dim = self.dim
        qk_rope_head_dim = self.qk_rope_head_dim
        qk_nope_head_dim = self.qk_nope_head_dim
        v_head_dim = self.v_head_dim
        kv_lora_rank = self.kv_lora_rank

        device = self.wq.weight.device
        dtype = self.wq.weight.dtype

        wq = self.wq.weight.view(
            n_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dim,
        )

        # qk_nope: [n_heads, qk_nope_head_dim, dim]
        # qk_rope: [n_heads, qk_rope_head_dim, dim]

        wq_nope, wq_rope = torch.split(
            wq,
            [qk_nope_head_dim, qk_rope_head_dim],
            dim=1,
        )

        wkv_b = self.wkv_b.weight.view(
            n_heads,
            qk_nope_head_dim + v_head_dim,
            kv_lora_rank,
        )

        # w_uk: [n_heads, qk_nope_head_dim, kv_lora_rank]
        # w_uv: [n_heads, v_head_dim, kv_lora_rank]
        w_uk, w_uv = torch.split(
            wkv_b,
            [qk_nope_head_dim, v_head_dim],
            dim=1,
        )

        # [n_heads, kv_lora_rank, dim]
        wq_absorbed_nope = torch.bmm(
            w_uk.float().transpose(
                1, 2
            ),  # [n_heads, qk_nope_head_dim, kv_lora_rank] -> [n_heads, kv_lora_rank, qk_nope_head_dim]
            wq_nope.float(),  # [n_heads, qk_nope_head_dim, dim]
        ).to(dtype=dtype)

        # Each new query head is [absorbed nope | original RoPE].
        wq_absorbed = torch.cat(
            [wq_absorbed_nope, wq_rope],
            dim=1,
        ).reshape(
            n_heads * (kv_lora_rank + qk_rope_head_dim),
            dim,
        )

        self.wq_absorbed = nn.Linear(
            dim,
            n_heads * (kv_lora_rank + qk_rope_head_dim),
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wq_absorbed.weight.copy_(wq_absorbed)
        self.wq_absorbed.requires_grad_(False)  # Freeze the absorbed weights

        # [dim, n_heads, v_head_dim] -> [n_heads, dim, v_head_dim]
        w_o = self.wo.weight.view(
            dim,
            n_heads,
            v_head_dim,
        ).permute(1, 0, 2)

        # [n_heads, dim, v_head_dim] @ [n_heads, v_head_dim, kv_lora_rank] -> [n_heads, dim, kv_lora_rank]
        w_o_absorbed_per_head = torch.bmm(
            w_o.float(),
            w_uv.float(),
        ).to(dtype=dtype)

        # [n_heads, dim, kv_lora_rank] -> [dim, n_heads, kv_lora_rank] -> [dim, n_heads * kv_lora_rank]
        w_o_absorbed = w_o_absorbed_per_head.permute(
            1,
            0,
            2,
        ).reshape(
            dim,
            n_heads * kv_lora_rank,
        )

        self.wo_absorbed = nn.Linear(
            n_heads * kv_lora_rank,
            dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.wo_absorbed.weight.copy_(w_o_absorbed)
        self.wo_absorbed.requires_grad_(False)

    def forward_absorbed(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:

        assert self.wq_absorbed is not None
        assert self.wo_absorbed is not None

        batch_size, seq_len, _ = x.shape

        q = self.wq_absorbed(x)
        q = q.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
        )

        # q_nope: [batch_size, seq_len, n_heads, kv_lora_rank]
        # q_rope: [batch_size, seq_len, n_heads, qk_rope_head_dim]
        q_nope, q_rope = torch.split(
            q,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        q_rope = apply_rotary_emb(q_rope, freqs_cis)

        # [batch_size, seq_len, n_heads, kv_lora_rank + qk_rope_head_dim] -> [batch_size, n_heads, seq_len, kv_lora_rank + qk_rope_head_dim]
        q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)

        # latent_raw: [batch_size, seq_len, kv_lora_rank]
        # k_rope: [batch_size, seq_len, qk_rope_head_dim]
        latent_raw, k_rope = torch.split(
            self.wkv_a(
                x
            ),  # [batch_size, seq_len, dim] -> [batch_size, seq_len, kv_lora_rank + qk_rope_head_dim]
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )

        # This is the latent that should be cached.
        # latent: [batch_size, seq_len, kv_lora_rank]
        latent = self.kv_norm(latent_raw)

        # [batch_size, seq_len, 1, qk_rope_head_dim]
        k_rope = apply_rotary_emb(k_rope.unsqueeze(2), freqs_cis)

        # A single shared storage tensor:
        # shared_cache: [batch_size, seq_len, 1, kv_lora_rank + qk_rope_head_dim] -> [batch_size, 1, seq_len, kv_lora_rank + qk_rope_head_dim]
        shared_cache = torch.cat(
            [latent.unsqueeze(2), k_rope],
            dim=-1,
        ).transpose(1, 2)

        # k: [batch_size, 1, seq_len, kv_lora_rank + qk_rope_head_dim]
        k = shared_cache

        # v: [batch_size, 1, seq_len, kv_lora_rank]
        v = shared_cache[..., : self.kv_lora_rank]

        # latent_output: [batch_size, n_heads, seq_len, kv_lora_rank]
        latent_output = self.inner_attention(q, k, v, scale=self.softmax_scale)

        # [batch_size, seq_len, n_heads * kv_lora_rank]
        latent_output = (
            latent_output.transpose(
                1, 2
            )  # [batch_size, n_heads, seq_len, kv_lora_rank] -> [batch_size, seq_len, n_heads, kv_lora_rank]
            .contiguous()
            .view(
                batch_size,
                seq_len,
                self.n_heads * self.kv_lora_rank,
            )
        )

        # output: [batch_size, seq_len, dim]
        return self.wo_absorbed(latent_output)

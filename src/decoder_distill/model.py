from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from flash_attn import flash_attn_func


@dataclass
class StudentConfig:
    vocab_size: int = 8
    max_seq_len: int = 1024
    d_model: int = 512
    n_layers: int = 18
    n_heads: int = 8
    head_dim: int = 64
    ffn_hidden_dim: int = 2048
    kv_latent_dim: int = 256
    dropout: float = 0.0
    rope_base: float = 10000.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model != self.n_heads * self.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(rms + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq)
        cos = freqs.cos().to(dtype=dtype)
        sin = freqs.sin().to(dtype=dtype)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    rotary_dim = cos.size(-1) * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos = cos.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2).view(1, cos.size(0), 1, rotary_dim)
    sin = sin.unsqueeze(0).unsqueeze(2).repeat(1, 1, 1, 2).view(1, sin.size(0), 1, rotary_dim)
    rotated = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat((rotated, x_pass), dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LatentKVAttention(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.dropout_p = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.kv_latent_proj = nn.Linear(config.d_model, config.kv_latent_dim, bias=False)
        self.k_proj = nn.Linear(config.kv_latent_dim, config.n_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.kv_latent_dim, config.n_heads * config.head_dim, bias=False)
        self.out_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        kv_latent = self.kv_latent_proj(x)
        k = self.k_proj(kv_latent).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(kv_latent).view(batch_size, seq_len, self.n_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if q.dtype not in (torch.float16, torch.bfloat16):
            attn_dtype = torch.bfloat16 if q.is_cuda and torch.cuda.is_bf16_supported() else torch.float16
            q = q.to(attn_dtype)
            k = k.to(attn_dtype)
            v = v.to(attn_dtype)
        out = flash_attn_func(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            dropout_p=self.dropout_p if self.training else 0.0,
            causal=True,
        )
        out = out.to(x.dtype).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(out)


class DecoderBlock(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = LatentKVAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden_dim)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class StudentDecoder(nn.Module):
    def __init__(self, config: StudentConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.rope = RotaryEmbedding(config.head_dim, base=config.rope_base)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len={self.config.max_seq_len}")

        x = self.embed_tokens(input_ids)
        cos, sin = self.rope(seq_len=seq_len, device=input_ids.device, dtype=x.dtype)

        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.final_norm(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

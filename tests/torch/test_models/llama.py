# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBED_DIM = 64
N_HEADS = 4
HEAD_DIM = EMBED_DIM // N_HEADS
# Same as Llama 3.2 config
ROPE_THETA = 500000.0
MAX_SEQ = 128
BIAS = False


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Copied from src/transformers/models/llama/modeling_llama.py
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x):
    """
    Copied from src/transformers/models/llama/modeling_llama.py
    Rotates half the hidden dims of the input.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Rotary(nn.Module):
    """
    Precompute cos/sin for RoPE and apply to q,k.
    Copied from src/transformers/models/llama/modeling_llama.py
    Initialize the cos and sin value once in init method
    """

    # Llama applies rotary to q,k before attention; see modeling_llama
    def __init__(self, head_dim: int, max_seq_len: int = MAX_SEQ, theta: float = ROPE_THETA, device=None):
        super().__init__()
        dtype = torch.float32
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device) / head_dim))
        t = torch.arange(max_seq_len, dtype=dtype, device=device)
        freqs = torch.einsum("t,f->tf", t, inv_freq)  # (T, Hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (T, Hd)
        self.register_buffer("cos", emb.cos()[None, None, ...], persistent=False)  # (1,1,T,Hd)
        self.register_buffer("sin", emb.sin()[None, None, ...], persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos: torch.Tensor):
        cos = self.cos[..., pos, :]
        sin = self.sin[..., pos, :]
        q_embed = (q * cos) + (_rotate_half(q) * sin)
        k_embed = (k * cos) + (_rotate_half(k) * sin)
        return q_embed, k_embed


class LlamaMLP(nn.Module):
    """
    Copied from src/transformers/models/llama/modeling_llama.py
    """

    def __init__(self, dim: int, mult: int = 2):
        super().__init__()
        # mult is used as a scaling factor of sorts. This is to define the hidden/intermediate layer size
        hidden = mult * dim
        self.gate_proj = nn.Linear(dim, hidden, bias=BIAS)
        self.up_proj = nn.Linear(dim, hidden, bias=BIAS)
        self.down_proj = nn.Linear(hidden, dim, bias=BIAS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaDecoderOnly(nn.Module):
    """
    One Llama-style transformer block (pre-norm attn + MLP) with RoPE and KV cache.
    Forward takes embeddings only.
    """

    # KV caching + past_key_values flow mirrors HF implementations. :contentReference[oaicite:4]{index=4}
    def __init__(self, dim: int = EMBED_DIM, n_heads: int = N_HEADS):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.attn_norm = LlamaRMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=BIAS)
        self.k_proj = nn.Linear(dim, dim, bias=BIAS)
        self.v_proj = nn.Linear(dim, dim, bias=BIAS)
        self.o_proj = nn.Linear(dim, dim, bias=BIAS)
        self.rope = Rotary(self.head_dim, MAX_SEQ, theta=ROPE_THETA)

        self.mlp_norm = LlamaRMSNorm(dim)
        self.mlp = LlamaMLP(dim)

    def _attn(self, x: torch.Tensor, pos: torch.Tensor, past_kv: Optional[tuple[torch.Tensor, torch.Tensor]]):
        """
        Code from LlamaAttention forward method. SDPA implementation similar to model.config._attn_implementation="SDPA"
        """
        B, T, C = x.shape
        H, Hd = self.n_heads, self.head_dim

        # QKV projections from hidden state x
        q = self.q_proj(x).view(B, T, H, Hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, Hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Hd).transpose(1, 2)

        # RoPE
        q, k = self.rope(q, k, pos)

        # KV cache
        if past_kv is not None:
            pk, pv = past_kv  # (B,H,Tpast,Hd)
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True, dropout_p=0.0)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.o_proj(y)
        return y, (k, v)

    def forward(
        self,
        x_embed: torch.Tensor,  # (B, T_new, C) embeddings only
        past_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # (B,H,Tpast,Hd)
    ):
        # positions for the *new* tokens only
        past_len = 0 if past_kv is None else past_kv[0].size(2)
        T_new = x_embed.size(1)
        pos = torch.arange(past_len, past_len + T_new, device=x_embed.device)

        # pre-norm attention + residual
        y, _kv = self._attn(self.attn_norm(x_embed), pos, past_kv)
        x = x_embed + y

        # pre-norm MLP + residual
        x = x + self.mlp(self.mlp_norm(x))
        return x

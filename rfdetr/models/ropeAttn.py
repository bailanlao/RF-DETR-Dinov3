import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch import Tensor, nn

def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], 
    # sin: [..., D], 
    # cos: [..., D], 
    return (x * cos) + (rope_rotate_half(x) * sin)

class RopeAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        device=None,
        dtype=None
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        assert self.head_dim * num_heads == self.embed_dim, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.scale = self.head_dim **-0.5
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_bias, device=device, dtype=dtype)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=proj_bias, device=device, dtype=dtype)
        
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    def apply_rope(self, q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> Tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        
        # [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)  
        cos = cos.unsqueeze(1)
        
        rope_dtype = sin.dtype
        q_rope = q.to(dtype=rope_dtype)
        k_rope = k.to(dtype=rope_dtype)
        
        q_rope = rope_apply(q_rope, sin, cos)
        k_rope = rope_apply(k_rope, sin, cos)
        
        return q_rope.to(dtype=q_dtype), k_rope.to(dtype=k_dtype)

    def forward(
        self,
        query: Tensor,
        sin: Tensor,
        cos: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.batch_first:
            query = query.transpose(0, 1)
        
        batch_size, seq_len, _ = query.shape
        
        qkv = self.qkv_proj(query)  # [batch, seq_len, 3*embed_dim]
        q, k, v = torch.split(qkv, self.embed_dim, dim=-1)  # [batch, seq_len, embed_dim]
        
        # [batch, seq_len, embed_dim] -> [batch, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        q, k = self.apply_rope(q, k, sin, cos)
        
        if key_padding_mask is not None:
            # [batch, 1, 1, seq_len]
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # 合并掩码（如有attn_mask则叠加）
            attn_mask = key_padding_mask if attn_mask is None else attn_mask + key_padding_mask
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            is_causal=False
        )  # [batch, num_heads, seq_len, head_dim]
        
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.embed_dim)  # [batch, seq_len, embed_dim]
        
        output = self.out_proj(attn_output)
        output = self.out_drop(output)
        
        attn_weights = None
        if need_weights:
            q_scaled = q * self.scale
            attn_scores = torch.matmul(q_scaled, k.transpose(-2, -1))  # [batch, num_heads, seq_len, seq_len]
            if attn_mask is not None:
                attn_scores = attn_scores.masked_fill(attn_mask, -math.inf)
            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        return (output, attn_weights)
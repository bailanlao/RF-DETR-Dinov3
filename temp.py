import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------
# Register model decorator: Try to import timm's version; if unavailable, use a dummy.
try:
    from timm.models.registry import register_model
except ImportError:
    def register_model(fn):
        return fn

# Import timm's VisionTransformer and common layers.
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import DropPath, Mlp

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.weight is not None}'

class DiffAttention(nn.Module):
    r"""
    Differential Attention Module.

    Given an input tensor X ∈ ℝ^(B×N×d_model), we first compute the linear projections:

        Q = X Wᵠ, K = X Wᵏ, V = X Wᵛ

    The queries and keys are then reshaped and split into two parts:
        Q → [Q₁; Q₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
        K → [K₁; K₂] ∈ ℝ^(B, N, 2·h_effective, d_head)
    with h_effective = num_heads // 2 and d_head = d_model / num_heads.

    The value projection is reshaped to:
        V ∈ ℝ^(B, N, h_effective, 2·d_head)

    We then compute two attention maps:
        A₁ = softmax((Q₁ K₁ᵀ) / √d_head)
        A₂ = softmax((Q₂ K₂ᵀ) / √d_head)

    A learnable scalar λ is computed via:
        λ = exp(λ_{q1} ⋅ λ_{k1}) − exp(λ_{q2} ⋅ λ_{k2}) + λ_init

    Finally, the differential attention output is:
        DiffAttn(X) = (A₁ − λ · A₂) · V

    The per-head outputs are then normalized headwise with RMSNorm and projected back to d_model.

    Args:
        dim (int): Embedding dimension (d_model).
        num_heads (int): Number of heads in the original transformer (must be even).
        qkv_bias (bool): If True, add a bias term to the Q, K, V projections.
        attn_drop (float): Dropout probability after softmax.
        proj_drop (float): Dropout probability after the output projection.
        lambda_init (float): Initial constant for lambda re-parameterization.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.dim = dim
        self.num_heads = num_heads # original number of heads
        self.effective_heads = num_heads // 2  # differential attention operates on half as many heads
        self.head_dim = dim // num_heads # per-head dimension
        self.scaling = self.head_dim ** -0.5

        # Linear projections for Q, K, V: mapping from dim → dim.
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True) # final output projection

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # RMSNorm for headwise normalization on outputs (each head's output has dimension 2·head_dim)
        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        # Learnable lambda parameters (shared across all heads)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, d_model).

        Returns:
            Tensor of shape (B, N, d_model) after applying differential attention.
        """
        B, N, _ = x.shape

        # Compute Q, K, V projections.
        q = self.q_proj(x) # shape: (B, N, d_model)
        k = self.k_proj(x) # shape: (B, N, d_model)
        v = self.v_proj(x) # shape: (B, N, d_model)

        # Reshape Q and K into (B, N, 2 * h_effective, head_dim)
        q = q.view(B, N, 2 * self.effective_heads, self.head_dim)
        k = k.view(B, N, 2 * self.effective_heads, self.head_dim)
        # Reshape V into (B, N, h_effective, 2 * head_dim)
        v = v.view(B, N, self.effective_heads, 2 * self.head_dim)

        # Transpose to bring head dimension forward.
        # q, k: (B, 2 * h_effective, N, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # v: (B, h_effective, N, 2 * head_dim)
        v = v.transpose(1, 2)

        # Scale Q.
        q = q * self.scaling

        # Compute raw attention scores: (B, 2 * h_effective, N, N)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))

        # Compute attention probabilities.
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        # Reshape to separate the two halves: (B, h_effective, 2, N, N)
        attn_probs = attn_probs.view(B, self.effective_heads, 2, N, N)

        # Compute lambda via re-parameterization.
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # Differential attention: subtract the second attention map scaled by lambda_full.
        diff_attn = attn_probs[:, :, 0, :, :] - lambda_full * attn_probs[:, :, 1, :, :] # shape: (B, h_effective, N, N)

        # Multiply the differential attention weights with V.
        attn_output = torch.matmul(diff_attn, v) # shape: (B, h_effective, N, 2 * head_dim)

        # Apply RMSNorm (headwise normalization) and scale by (1 - lambda_init)
        attn_output = self.diff_norm(attn_output) * (1 - self.lambda_init)

        # Concatenate heads: reshape from (B, h_effective, N, 2 * head_dim) → (B, N, 2 * h_effective * head_dim)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, 2 * self.effective_heads * self.head_dim)

        # Final linear projection.
        x_out = self.out_proj(attn_output)
        x_out = self.proj_drop(x_out)
        return x_out
    


if __name__ == "__main__":

    # 输入参数
    batch_size = 1
    seq_len = 16*16
    dim = 32
    num_heads = 8

    # 构造输入张量：形状 [B, N, C]
    x = torch.randn(batch_size, seq_len, dim).cuda()

    # 实例化 DiffAttention 模块
    model = DiffAttention(dim=dim, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1, lambda_init=0.8).cuda()
    print(model)

    # 前向传播
    output = model(x)

    # 输出形状
    print("输入形状:", x.shape) # [B, N, C]
    print("输出形状:", output.shape) # [B, N, C]
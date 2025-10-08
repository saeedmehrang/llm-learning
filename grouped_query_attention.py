"""
Grouped-Query Attention (GQA) Implementation in PyTorch

This module provides PyTorch implementations of Multi-Head Attention (MHA) and
Grouped-Query Attention (GQA), demonstrating the memory efficiency improvements
of GQA for large language models.

Classes:
    MultiHeadAttention: Standard multi-head attention mechanism where each
        attention head has its own query, key, and value projections.

    GroupedQueryAttention: Memory-efficient attention mechanism where multiple
        query heads share the same key-value heads, reducing KV cache size
        during inference.

Key Concept:
    In GQA, instead of having separate K/V projections for each query head,
    multiple query heads share the same K/V heads. For example, with 32 query
    heads and 8 KV heads, every 4 query heads share one KV head, resulting in
    a 4x reduction in KV cache memory.

Usage Example:
    ```python
    from grouped_query_attention import MultiHeadAttention, GroupedQueryAttention

    # Create input tensor
    x = torch.randn(batch_size=2, seq_len=10, d_model=512)

    # Standard Multi-Head Attention
    mha = MultiHeadAttention(d_model=512, num_heads=32)
    mha_output = mha(x)

    # Grouped-Query Attention (4x memory reduction)
    gqa = GroupedQueryAttention(d_model=512, num_query_heads=32, num_kv_heads=8)
    gqa_output = gqa(x)
    ```

When run as a script:
    python grouped-query-attention.py

    Demonstrates the memory savings and functional equivalence of GQA compared
    to MHA with detailed shape information and performance metrics.

References:
    - GQA: Training Generalized Multi-Query Transformer Models from Multi-Head
      Checkpoints (Ainslie et al., 2023)
    - Fast Transformer Decoding: One Write-Head is All You Need
      (Shazeer, 2019)
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        """
        Args:
            d_model: Dimension of the model (e.g., 512, 768, etc.)
            num_heads: Number of attention heads (e.g., 8, 12, etc.)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        # Each projects from d_model to d_model (which will be split into heads)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Scaling factor for dot products
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: None | torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Apply linear projections
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, d_model)
        V = self.v_proj(x)  # (batch_size, seq_len, d_model)

        # Reshape and transpose for multi-head attention
        # Split d_model into num_heads * head_dim
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # Shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        attn_output = torch.matmul(attn_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Apply output projection
        output = self.out_proj(attn_output)

        return output


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_query_heads: int, num_kv_heads: int):
        """
        Args:
            d_model: Dimension of the model (e.g., 512, 768, etc.)
            num_query_heads: Number of query heads (e.g., 32)
            num_kv_heads: Number of key-value heads (e.g., 8)
                         Must divide num_query_heads evenly
        """
        super().__init__()
        assert (
            d_model % num_query_heads == 0
        ), "d_model must be divisible by num_query_heads"
        assert (
            num_query_heads % num_kv_heads == 0
        ), "num_query_heads must be divisible by num_kv_heads"

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_query_heads // num_kv_heads
        self.head_dim = d_model // num_query_heads

        # Query projection: still projects to full d_model
        self.q_proj = nn.Linear(d_model, d_model)

        # Key and Value projections: project to fewer dimensions
        # Only num_kv_heads worth of dimensions instead of num_query_heads
        self.kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(d_model, self.kv_dim)
        self.v_proj = nn.Linear(d_model, self.kv_dim)

        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)

        # Scaling factor
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self, x: torch.Tensor, mask: None | torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Apply linear projections
        Q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        K = self.k_proj(x)  # (batch_size, seq_len, kv_dim) <- Smaller!
        V = self.v_proj(x)  # (batch_size, seq_len, kv_dim) <- Smaller!

        # Reshape queries for multiple heads
        Q = Q.view(batch_size, seq_len, self.num_query_heads, self.head_dim).transpose(
            1, 2
        )
        # Shape: (batch_size, num_query_heads, seq_len, head_dim)

        # Reshape keys and values for fewer heads
        K = K.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        V = V.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(
            1, 2
        )
        # Shape: (batch_size, num_kv_heads, seq_len, head_dim)

        # Repeat K and V to match the number of query heads
        # Each KV head is shared by num_queries_per_kv query heads
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        # Shape: (batch_size, num_query_heads, seq_len, head_dim)

        # Now the rest is identical to Multi-Head Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)

        # Apply output projection
        output = self.out_proj(attn_output)

        return output


if __name__ == "__main__":
    # ========================================================================
    # Configuration for memory comparison
    # ========================================================================
    d_model = 4096
    num_query_heads = 32
    num_kv_heads = 8
    seq_len = 2048
    num_layers = 32

    # KV cache size per layer (in elements, for both K and V)
    # Using PyTorch tensor operations for clarity
    mha_cache = torch.tensor(
        num_query_heads * (d_model // num_query_heads) * seq_len * 2
    )
    gqa_cache = torch.tensor(num_kv_heads * (d_model // num_query_heads) * seq_len * 2)

    # Total cache size for all layers
    mha_total = mha_cache * num_layers
    gqa_total = gqa_cache * num_layers

    print(f"MHA KV cache: {mha_total.item() / 1e6:.1f}M elements")
    print(f"GQA KV cache: {gqa_total.item() / 1e6:.1f}M elements")
    print(f"Memory reduction: {(mha_total / gqa_total).item():.1f}x")
    print()

    # ========================================================================
    # Practical example with actual PyTorch tensors
    # ========================================================================

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    batch_size = 2
    seq_len = 10
    d_model = 512
    num_query_heads = 32
    num_kv_heads = 8

    # Create random input tensor
    x = torch.randn(batch_size, seq_len, d_model)

    print("=" * 70)
    print("INPUT CONFIGURATION")
    print("=" * 70)
    print(f"Input shape: {x.shape}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Sequence length: {seq_len}")
    print(f"  - Model dimension (d_model): {d_model}")
    print()

    # ========================================================================
    # Multi-Head Attention Example
    # ========================================================================
    print("=" * 70)
    print("MULTI-HEAD ATTENTION")
    print("=" * 70)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_query_heads)
    mha_output = mha(x)

    print(f"Number of heads: {mha.num_heads}")
    print(f"Head dimension: {mha.head_dim}")
    print()
    print("Weight matrices:")
    print(f"  Q projection: {d_model} → {d_model}")
    print(f"  K projection: {d_model} → {d_model}")
    print(f"  V projection: {d_model} → {d_model}")
    print()
    print("After projection and reshaping:")
    print(
        f"  Q shape: (batch={batch_size}, heads={num_query_heads}, seq={seq_len}, head_dim={mha.head_dim})"
    )
    print(
        f"  K shape: (batch={batch_size}, heads={num_query_heads}, seq={seq_len}, head_dim={mha.head_dim})"
    )
    print(
        f"  V shape: (batch={batch_size}, heads={num_query_heads}, seq={seq_len}, head_dim={mha.head_dim})"
    )
    print()
    print(f"Output shape: {mha_output.shape}")
    print()

    # Calculate KV cache size for MHA using tensor operations
    kv_cache_mha = torch.tensor(
        num_query_heads * mha.head_dim * seq_len * 2
    )  # *2 for K and V
    print(f"KV cache size per sample: {kv_cache_mha.item():,} elements")
    print()

    # ========================================================================
    # Grouped-Query Attention Example
    # ========================================================================
    print("=" * 70)
    print("GROUPED-QUERY ATTENTION")
    print("=" * 70)

    gqa = GroupedQueryAttention(
        d_model=d_model, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads
    )
    gqa_output = gqa(x)

    print(f"Number of query heads: {gqa.num_query_heads}")
    print(f"Number of KV heads: {gqa.num_kv_heads}")
    print(f"Queries per KV head: {gqa.num_queries_per_kv}")
    print(f"Head dimension: {gqa.head_dim}")
    print()
    print("Weight matrices:")
    print(f"  Q projection: {d_model} → {d_model}")
    print(f"  K projection: {d_model} → {gqa.kv_dim} (smaller!)")
    print(f"  V projection: {d_model} → {gqa.kv_dim} (smaller!)")
    print()
    print("After projection and reshaping:")
    print(
        f"  Q shape: (batch={batch_size}, heads={num_query_heads}, seq={seq_len}, head_dim={gqa.head_dim})"
    )
    print(
        f"  K shape: (batch={batch_size}, heads={num_kv_heads}, seq={seq_len}, head_dim={gqa.head_dim}) [before repeat]"
    )
    print(
        f"  V shape: (batch={batch_size}, heads={num_kv_heads}, seq={seq_len}, head_dim={gqa.head_dim}) [before repeat]"
    )
    print()
    print("After repeat_interleave (to match Q heads):")
    print(
        f"  K shape: (batch={batch_size}, heads={num_query_heads}, seq={seq_len}, head_dim={gqa.head_dim})"
    )
    print(
        f"  V shape: (batch={batch_size}, heads={num_query_heads}, seq={seq_len}, head_dim={gqa.head_dim})"
    )
    print()
    print(f"Output shape: {gqa_output.shape}")
    print()

    # Calculate KV cache size for GQA using tensor operations
    kv_cache_gqa = torch.tensor(
        num_kv_heads * gqa.head_dim * seq_len * 2
    )  # *2 for K and V
    print(f"KV cache size per sample: {kv_cache_gqa.item():,} elements")
    print()

    # ========================================================================
    # Comparison
    # ========================================================================
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"MHA KV cache: {kv_cache_mha.item():,} elements")
    print(f"GQA KV cache: {kv_cache_gqa.item():,} elements")
    print(f"Memory reduction: {(kv_cache_mha / kv_cache_gqa).item():.1f}x")
    print()
    print(f"MHA output shape: {mha_output.shape}")
    print(f"GQA output shape: {gqa_output.shape}")
    print()
    print("✓ Both produce the same output shape!")
    print()

    # ========================================================================
    # Visualizing the grouping
    # ========================================================================
    print("=" * 70)
    print("QUERY HEAD GROUPING IN GQA")
    print("=" * 70)
    print(f"With {num_query_heads} query heads and {num_kv_heads} KV heads:")
    print()
    for kv_head in range(num_kv_heads):
        start_q = kv_head * gqa.num_queries_per_kv
        end_q = start_q + gqa.num_queries_per_kv - 1
        print(f"  KV head {kv_head} is shared by query heads {start_q}-{end_q}")
    print()
    print("=" * 70)

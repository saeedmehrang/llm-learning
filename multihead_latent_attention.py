"""
Multi-Head Latent Attention (MLA)
===================================

A minimal, educational implementation of Multi-Head Latent Attention, an efficient
attention mechanism that reduces KV-cache memory through latent compression.

What is MLA?
------------
MLA compresses Key-Value pairs into a low-dimensional latent space before caching,
significantly reducing memory requirements during autoregressive generation:

- Standard MHA: Stores K, V separately for each head → 2 × d_model per token
- MLA: Stores compressed latent representation → d_latent per token (4-8x smaller)

Key Concepts:
-------------
1. **Compression**: Input → low-dimensional latent (d_latent << d_model)
2. **Decompression**: Latent → full K and V (on-the-fly, not cached)
3. **Memory Savings**: Only cache latent, not full K/V tensors

Example Usage:
--------------
```python
import torch
from mla import MultiHeadLatentAttention

# Initialize MLA
mla = MultiHeadLatentAttention(
    d_model=512,      # Model dimension
    num_heads=8,      # Number of attention heads
    d_latent=128      # Latent compression dimension
)

# Single forward pass
x = torch.randn(2, 10, 512)  # [batch, seq_len, d_model]
output, cache = mla(x, use_cache=True)

# Autoregressive generation with caching
x_new = torch.randn(2, 1, 512)  # Next token
output_new, cache = mla(x_new, cache=cache, use_cache=True)

print(f"Cache stores only {cache['kv_latent'].shape[-1]} dims (not {512*2}!)")
```

Memory Comparison:
------------------
For a sequence of length L with d_model=512, d_latent=128:
- Standard MHA cache: 2 × 512 × L = 1024L parameters
- MLA cache: 128 × L parameters
- Savings: 8x memory reduction!

References:
-----------
Multi-Head Latent Attention is used in models like DeepSeek-V2 for efficient
long-context processing.

Author: LLM Learning Repository
Date: 2025-01-15
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) implementation with low-rank factorization 
    for all Q, K, and V projections, and shared compression for KV.
    """
    def __init__(self, d_model, num_heads, d_latent, dropout=0.1):
        super().__init__()
	"""
	Initializes the MultiHeadLatentAttention module.

	Args:
	    d_model (int): Model dimension (i.e., the hidden size of the input and output, 
		           e.g., 512, 768). This must be divisible by `num_heads`.
	    num_heads (int): Number of attention heads (e.g., 8, 12).
	    d_latent (int): Latent dimension ($r$) used for low-rank compression of 
		            the Query (Q), Key (K), and Value (V) projections (e.g., 128, 256). 
		            This is significantly smaller than `d_model` and enables 
		            efficient KV cache storage.
	    dropout (float, optional): Dropout probability applied to the attention weights. 
		                       Defaults to 0.1.
	"""
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_latent = d_latent
        self.scale = math.sqrt(self.d_head)
        
        # === Q Compression and Decompression ===
        # Q Compression (W^D_Q): project to latent space
        self.W_q_compress = nn.Linear(d_model, d_latent)
        # Q Decompression (W^U_Q): expand from latent space to full Q dimension
        self.W_q_decompress = nn.Linear(d_latent, d_model)
        
        # === KV Compression and Decompression ===
        # KV Compression (W^D_KV): shared projection to low-dimensional latent space
        self.W_kv_compress = nn.Linear(d_model, d_latent)
        
        # K Decompression (W^U_K): expand from latent space to full K dimension
        self.W_k_decompress = nn.Linear(d_latent, d_model)
        # V Decompression (W^U_V): expand from latent space to full V dimension
        self.W_v_decompress = nn.Linear(d_latent, d_model)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None, cache=None, use_cache=False):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            cache: Dictionary containing cached 'kv_latent' from previous steps
            use_cache: Whether to return cache for next step
            
        Returns:
            output: [batch_size, seq_len, d_model]
            cache: (optional) Dictionary with cached latent representations
        """
        batch_size, seq_len, _ = x.shape
        
        # ========== Query Compression & Decompression (Factorized Q) ==========
        # 1. Compression: C_Q = X W^D_Q
        q_latent = self.W_q_compress(x)  # [batch, seq_len, d_latent]
        
        # 2. Decompression: Q = C_Q W^U_Q
        Q = self.W_q_decompress(q_latent) # [batch, seq_len, d_model]
        
        # Split Q into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_head)
        Q = Q.transpose(1, 2)  # [batch, num_heads, seq_len, d_head]
        
        # ========== KV Compression & Caching ==========
        # 1. Compression: C_KV = X W^D_KV
        kv_latent = self.W_kv_compress(x)  # [batch, seq_len, d_latent]
        
        # Handle caching for autoregressive generation
        if cache is not None and 'kv_latent' in cache:
            # Concatenate with previous cached latent representations
            kv_latent = torch.cat([cache['kv_latent'], kv_latent], dim=1)
        
        # ========== KV Decompression (Factorized K and V) ==========
        # 2. Decompression: K = C_KV W^U_K and V = C_KV W^U_V
        
        # The total sequence length in the cache
        cached_seq_len = kv_latent.shape[1]
        
        K = self.W_k_decompress(kv_latent)  # [batch, cached_seq_len, d_model]
        V = self.W_v_decompress(kv_latent)  # [batch, cached_seq_len, d_model]
        
        # Split K and V into heads
        K = K.view(batch_size, cached_seq_len, self.num_heads, self.d_head)
        K = K.transpose(1, 2)  # [batch, num_heads, cached_seq_len, d_head]
        
        V = V.view(batch_size, cached_seq_len, self.num_heads, self.d_head)
        V = V.transpose(1, 2)  # [batch, num_heads, cached_seq_len, d_head]
        
        # ========== Attention Computation ==========
        # Compute attention scores: (Q K^T) / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch, num_heads, seq_len, cached_seq_len]
        
        # Apply mask if provided
        if mask is not None:
            # We assume the mask is compatible with the Q and KV sequence lengths
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: Attention(Q, K, V)
        attn_output = torch.matmul(attn_weights, V)
        # [batch, num_heads, seq_len, d_head]
        
        # ========== Output Projection ==========
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        output = self.W_o(attn_output)
        
        # Prepare cache for next step if needed
        new_cache = None
        if use_cache:
            # NOTE: We cache the shared latent representation C_KV
            new_cache = {'kv_latent': kv_latent}
        
        return output, new_cache
        



# ============================================================================
# DEMONSTRATION & EXAMPLES
# ============================================================================

def example_basic_usage():
    """Example 1: Basic forward pass without caching."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Forward Pass (No Caching)")
    print("=" * 70)

    # Configuration
    d_model = 512
    num_heads = 8
    d_latent = 128
    batch_size = 2
    seq_len = 10

    # Initialize MLA layer
    mla = MultiHeadLatentAttention(d_model, num_heads, d_latent)

    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)

    # Forward pass without caching
    output, _ = mla(x, use_cache=False)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\n✅ Basic attention computation complete!")


def example_autoregressive_generation():
    """Example 2: Autoregressive generation with caching."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Autoregressive Generation with Caching")
    print("=" * 70)

    d_model = 512
    num_heads = 8
    d_latent = 128
    batch_size = 2

    mla = MultiHeadLatentAttention(d_model, num_heads, d_latent)

    # Step 1: Process first token
    x_first = torch.randn(batch_size, 1, d_model)
    output_first, cache = mla(x_first, use_cache=True)

    print(f"\nStep 1:")
    print(f"  Input:  {x_first.shape}")
    print(f"  Output: {output_first.shape}")
    print(f"  Cache:  {cache['kv_latent'].shape}")

    # Step 2: Process second token (using cache)
    x_second = torch.randn(batch_size, 1, d_model)
    output_second, cache = mla(x_second, cache=cache, use_cache=True)

    print(f"\nStep 2 (with cache):")
    print(f"  Input:  {x_second.shape}")
    print(f"  Output: {output_second.shape}")
    print(f"  Cache:  {cache['kv_latent'].shape}  ← Cache grows!")

    # Step 3: Process third token
    x_third = torch.randn(batch_size, 1, d_model)
    output_third, cache = mla(x_third, cache=cache, use_cache=True)

    print(f"\nStep 3 (with cache):")
    print(f"  Input:  {x_third.shape}")
    print(f"  Output: {output_third.shape}")
    print(f"  Cache:  {cache['kv_latent'].shape}  ← Cache continues to grow!")

    print(f"\n✅ Autoregressive generation complete!")


def example_memory_comparison():
    """Example 3: Memory savings analysis."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Memory Savings Analysis")
    print("=" * 70)

    d_model = 512
    num_heads = 8
    d_latent = 128

    # Calculate cache sizes per token
    standard_cache_size = 2 * d_model  # K and V, full dimension
    mla_cache_size = d_latent  # Compressed latent only
    savings_ratio = standard_cache_size / mla_cache_size

    print(f"\nConfiguration:")
    print(f"  d_model:  {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  d_latent: {d_latent}")

    print(f"\nCache size per token:")
    print(f"  Standard MHA: {standard_cache_size} parameters")
    print(f"  MLA:          {mla_cache_size} parameters")
    print(f"  Savings:      {savings_ratio:.1f}x reduction!")

    # Example for longer sequences
    seq_length = 1000
    standard_total = standard_cache_size * seq_length
    mla_total = mla_cache_size * seq_length

    # Convert to MB (assuming float32 = 4 bytes)
    standard_mb = (standard_total * 4) / (1024 ** 2)
    mla_mb = (mla_total * 4) / (1024 ** 2)

    print(f"\nFor sequence length = {seq_length}:")
    print(f"  Standard MHA cache: {standard_mb:.2f} MB")
    print(f"  MLA cache:          {mla_mb:.2f} MB")
    print(f"  Memory saved:       {standard_mb - mla_mb:.2f} MB ({(1-mla_mb/standard_mb)*100:.1f}%)")

    print(f"\n✅ MLA saves significant memory for long sequences!")


def example_with_masking():
    """Example 4: Using attention masks."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Attention with Causal Masking")
    print("=" * 70)

    d_model = 256
    num_heads = 4
    d_latent = 64
    batch_size = 1
    seq_len = 5

    mla = MultiHeadLatentAttention(d_model, num_heads, d_latent)
    x = torch.randn(batch_size, seq_len, d_model)

    # Create causal mask (prevent attending to future tokens)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0)
    # Shape: [1, seq_len, seq_len]

    print(f"\nCausal mask (5x5):")
    print(causal_mask[0].int())
    print("(1 = can attend, 0 = masked)")

    # Forward pass with mask
    output, _ = mla(x, mask=causal_mask, use_cache=False)

    print(f"\nInput shape:  {x.shape}")
    print(f"Mask shape:   {causal_mask.shape}")
    print(f"Output shape: {output.shape}")

    print(f"\n✅ Causal masking ensures autoregressive property!")


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "MULTI-HEAD LATENT ATTENTION (MLA)" + " " * 20 + "║")
    print("║" + " " * 23 + "Educational Examples" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Run all examples
    example_basic_usage()
    example_autoregressive_generation()
    example_memory_comparison()
    example_with_masking()

    # Final summary
    print("\n" + "=" * 70)
    print("🎉 ALL EXAMPLES COMPLETE!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("  • MLA compresses KV cache into low-dimensional latent space")
    print("  • Typical memory savings: 4-8x reduction")
    print("  • Perfect for long-context generation (1000+ tokens)")
    print("  • Used in production models like DeepSeek-V2")
    print("\nFor more comprehensive experiments and comparisons, see kv_caching.py")
    print("=" * 70 + "\n")

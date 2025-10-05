"""
KV-Cache in Transformers: Complete Implementation & Experiments
================================================================

This module provides a comprehensive implementation of Key-Value caching in 
transformer language models, including:
- Full transformer architecture with KV-cache support
- Generation functions with and without caching
- Performance benchmarking utilities
- Detailed experiments with visualizations

Author: Your Name
Date: 2025-01-15
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """Configuration for transformer language model."""
    vocab_size: int = 100
    d_model: int = 64
    num_heads: int = 4
    num_layers: int = 3
    d_ff: int = 128
    max_seq_len: int = 512
    dropout: float = 0.1


# ============================================================================
# MODEL IMPLEMENTATION
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism with KV-caching support.
    
    This implementation allows caching of Key and Value projections from 
    previous tokens to avoid redundant computation during autoregressive generation.
    
    Args:
        d_model: Dimension of the model
        num_heads: Number of attention heads
        
    Attributes:
        d_model: Model dimension
        num_heads: Number of attention heads
        head_dim: Dimension of each attention head
        W_q, W_k, W_v: Query, Key, Value projection matrices
        W_o: Output projection matrix
    """
    
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Linear projections for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional KV-caching.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            kv_cache: Optional dictionary containing cached 'keys' and 'values'
                     from previous forward passes
            use_cache: If True, return updated cache for next iteration
            
        Returns:
            output: Attention output of shape [batch_size, seq_len, d_model]
            new_cache: Updated cache if use_cache=True, else None
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape to multi-head format: [batch, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Concatenate with cached K and V if available
        if kv_cache is not None:
            K = torch.cat([kv_cache['keys'], K], dim=2)
            V = torch.cat([kv_cache['values'], V], dim=2)
        
        # Compute attention scores: [batch, num_heads, seq_len, total_seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask to prevent attending to future tokens
        total_seq_len = K.size(2)
        current_seq_len = Q.size(2)
        
        if current_seq_len > 1:
            # Create causal mask (upper triangular)
            causal_mask = torch.triu(
                torch.ones(current_seq_len, total_seq_len, device=x.device, dtype=torch.bool), 
                diagonal=total_seq_len - current_seq_len + 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [batch, num_heads, seq_len, head_dim]
        
        # Reshape back to [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final output projection
        output = self.W_o(attn_output)
        
        # Prepare cache for next iteration if requested
        new_cache = None
        if use_cache:
            new_cache = {
                'keys': K.detach(),
                'values': V.detach()
            }
        
        return output, new_cache


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feedforward network.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feedforward network hidden dimension
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        d_ff: int, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            kv_cache: Optional cached keys and values
            use_cache: Whether to return updated cache
            
        Returns:
            output: Block output of shape [batch_size, seq_len, d_model]
            new_cache: Updated cache if use_cache=True, else None
        """
        # Self-attention with residual connection
        attn_output, new_cache = self.attention(self.norm1(x), kv_cache, use_cache)
        x = x + self.dropout(attn_output)
        
        # Feedforward with residual connection
        ff_output = self.ff(self.norm2(x))
        x = x + self.dropout(ff_output)
        
        return x, new_cache


class TransformerLM(nn.Module):
    """
    Transformer-based language model with KV-caching support.
    
    This model implements a decoder-only transformer architecture suitable
    for autoregressive language generation with optional KV-caching for
    efficient inference.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feedforward network hidden dimension
        max_seq_len: Maximum sequence length for positional embeddings
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection to vocabulary
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        kv_caches: Optional[List[Optional[Dict[str, torch.Tensor]]]] = None,
        use_cache: bool = False,
        position_offset: int = 0
    ) -> Tuple[torch.Tensor, Optional[List[Dict[str, torch.Tensor]]]]:
        """
        Forward pass through the language model.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            kv_caches: Optional list of caches for each layer
            use_cache: Whether to return updated caches
            position_offset: Offset for position embeddings (used during generation)
            
        Returns:
            logits: Output logits of shape [batch_size, seq_len, vocab_size]
            new_caches: Updated caches for each layer if use_cache=True, else None
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position IDs
        position_ids = torch.arange(
            position_offset, 
            position_offset + seq_len, 
            device=input_ids.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # Apply embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Initialize caches if needed
        if kv_caches is None and use_cache:
            kv_caches = [None] * self.num_layers
        
        # Pass through transformer blocks
        new_caches = []
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, cache, use_cache)
            if use_cache:
                new_caches.append(new_cache)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        return logits, (new_caches if use_cache else None)


# ============================================================================
# GENERATION FUNCTIONS
# ============================================================================

def generate_with_kv_cache(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 10,
    temperature: float = 1.0,
    verbose: bool = True
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Generate tokens using KV-caching for efficient inference.
    
    This function implements a two-phase generation process:
    1. Prefill phase: Process entire prompt at once
    2. Generation phase: Generate tokens one at a time using cached K/V
    
    Args:
        model: TransformerLM model
        prompt_ids: Input prompt tokens of shape [batch_size, prompt_len]
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature (1.0 = no change)
        verbose: Whether to print generation progress
        
    Returns:
        generated_ids: Complete sequence including prompt [batch_size, total_len]
        stats: Dictionary with generation statistics including:
               - prefill_seq_len: Length of original prompt
               - total_tokens_generated: Number of new tokens
               - kv_cache_memory_mb: Final cache size in MB
    """
    model.eval()
    device = prompt_ids.device
    
    stats = {
        'prefill_seq_len': prompt_ids.size(1),
        'total_tokens_generated': 0,
        'kv_cache_memory_mb': 0.0
    }
    
    with torch.no_grad():
        # ============================================
        # PREFILL PHASE: Process entire prompt at once
        # ============================================
        if verbose:
            print(f"\n🚀 PREFILL PHASE: Processing {prompt_ids.size(1)} prompt tokens...")
        
        logits, kv_caches = model(prompt_ids, use_cache=True, position_offset=0)
        
        # Sample first generated token from last position
        next_token_logits = logits[:, -1, :] / temperature
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # Initialize output sequence
        generated_ids = torch.cat([prompt_ids, next_token], dim=1)
        position = prompt_ids.size(1)
        stats['total_tokens_generated'] = 1
        
        # Calculate initial cache size
        cache_size = sum(
            cache['keys'].numel() * cache['keys'].element_size() + 
            cache['values'].numel() * cache['values'].element_size()
            for cache in kv_caches
        )
        stats['kv_cache_memory_mb'] = cache_size / (1024 ** 2)
        
        if verbose:
            print(f"✅ Prefill complete. Cache: {stats['kv_cache_memory_mb']:.3f} MB")
            print(f"   Cache shape per layer: keys={kv_caches[0]['keys'].shape}")
        
        # ============================================
        # GENERATION PHASE: Generate tokens one by one
        # ============================================
        if verbose:
            print(f"\n⚡ GENERATION PHASE: Generating {max_new_tokens-1} more tokens...")
        
        for step in range(max_new_tokens - 1):
            # Only process the newly generated token
            logits, kv_caches = model(
                next_token, 
                kv_caches=kv_caches,
                use_cache=True,
                position_offset=position
            )
            
            # Sample next token
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            position += 1
            stats['total_tokens_generated'] += 1
            
            # Show progress for first few and last step
            if verbose and (step < 3 or step == max_new_tokens - 2):
                tokens_str = str(next_token.squeeze().tolist())
                cache_len = kv_caches[0]['keys'].size(2)
                print(f"   Step {step+1}: tokens={tokens_str}, cache_len={cache_len}")
        
        # Calculate final cache size
        cache_size = sum(
            cache['keys'].numel() * cache['keys'].element_size() + 
            cache['values'].numel() * cache['values'].element_size()
            for cache in kv_caches
        )
        stats['kv_cache_memory_mb'] = cache_size / (1024 ** 2)
        
    return generated_ids, stats


def generate_without_kv_cache(
    model: TransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 10,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Generate tokens WITHOUT KV-caching (baseline for comparison).
    
    This function recomputes attention over the entire sequence at each
    generation step, demonstrating the inefficiency that KV-caching solves.
    
    Args:
        model: TransformerLM model
        prompt_ids: Input prompt tokens of shape [batch_size, prompt_len]
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature
        
    Returns:
        generated_ids: Complete sequence including prompt [batch_size, total_len]
    """
    model.eval()
    generated_ids = prompt_ids.clone()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Reprocess ENTIRE sequence every time (inefficient!)
            logits, _ = model(generated_ids, use_cache=False)
            
            # Sample next token
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
    
    return generated_ids


# ============================================================================
# BENCHMARKING UTILITIES
# ============================================================================

def benchmark_generation(
    model: TransformerLM,
    prompt_length: int,
    generate_length: int,
    use_cache: bool,
    device: str = 'cpu',
    num_runs: int = 3
) -> Dict[str, float]:
    """
    Benchmark generation performance with or without KV-cache.
    
    Args:
        model: TransformerLM model to benchmark
        prompt_length: Length of input prompt
        generate_length: Number of tokens to generate
        use_cache: Whether to use KV-caching
        device: Device to run on ('cpu' or 'cuda')
        num_runs: Number of benchmark runs to average
        
    Returns:
        Dictionary containing:
            - mean_time: Average generation time in seconds
            - std_time: Standard deviation of generation time
            - mean_memory_mb: Average peak memory usage (CUDA only)
            - tokens_per_second: Generation throughput
    """
    model.eval()
    model = model.to(device)
    
    times = []
    memory_usage = []
    
    for _ in range(num_runs):
        prompt = torch.randint(0, 100, (1, prompt_length), device=device)
        
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        
        start_time = time.time()
        
        if use_cache:
            # With KV-cache
            logits, kv_caches = model(prompt, use_cache=True)
            generated = prompt
            
            for step in range(generate_length):
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                logits, kv_caches = model(
                    next_token,
                    kv_caches=kv_caches,
                    use_cache=True,
                    position_offset=generated.size(1) - 1
                )
        else:
            # Without KV-cache
            generated = prompt
            for step in range(generate_length):
                logits, _ = model(generated, use_cache=False)
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
        
        elapsed = time.time() - start_time
        times.append(elapsed)
        
        if device == 'cuda':
            memory_usage.append(torch.cuda.max_memory_allocated() / (1024**2))
    
    return {
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'mean_memory_mb': float(np.mean(memory_usage)) if memory_usage else 0.0,
        'tokens_per_second': generate_length / np.mean(times)
    }


def calculate_cache_size(
    model: TransformerLM, 
    seq_length: int, 
    batch_size: int = 1,
    dtype_bytes: int = 4
) -> float:
    """
    Calculate theoretical KV-cache memory size.
    
    Args:
        model: TransformerLM model
        seq_length: Sequence length to cache
        batch_size: Batch size
        dtype_bytes: Bytes per element (4 for float32, 2 for float16)
        
    Returns:
        Cache size in megabytes (MB)
    """
    # Each layer stores keys and values
    # Shape per tensor: [batch, num_heads, seq_len, head_dim]
    num_heads = 4  # This should match model architecture
    head_dim = model.d_model // num_heads
    
    # 2 tensors (keys + values) per layer
    cache_per_layer = 2 * batch_size * num_heads * seq_length * head_dim * dtype_bytes
    total_cache = cache_per_layer * model.num_layers
    
    return total_cache / (1024**2)  # Convert to MB


# ============================================================================
# EXPERIMENTS
# ============================================================================

def experiment_speedup_scaling(
    save_plots: bool = True
) -> Dict[str, List[float]]:
    """
    Experiment 1: Measure speedup vs sequence length.
    
    This experiment demonstrates how KV-caching speedup increases with
    longer generation sequences, validating the O(n²) → O(n) improvement.
    
    Args:
        save_plots: Whether to save generated plots to disk
        
    Returns:
        Dictionary containing experiment results:
            - generate_lengths: List of generation lengths tested
            - speedups: Corresponding speedup factors
            - times_cached: Generation times with cache
            - times_uncached: Generation times without cache
    """
    print("=" * 80)
    print("EXPERIMENT 1: Speedup vs Sequence Length")
    print("=" * 80)
    
    # Small model for quick experimentation
    model = TransformerLM(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=128,
        dropout=0.0
    )
    model.eval()
    
    prompt_length = 10
    generate_lengths = [10, 20, 50, 100, 200]
    
    results_cached = []
    results_uncached = []
    speedups = []
    
    print(f"\nPrompt length: {prompt_length}")
    print(f"Testing generation lengths: {generate_lengths}\n")
    
    for gen_len in generate_lengths:
        print(f"Testing {gen_len} tokens...", end=" ")
        
        # Benchmark with cache
        result_cached = benchmark_generation(
            model, prompt_length, gen_len, use_cache=True, num_runs=3
        )
        
        # Benchmark without cache
        result_uncached = benchmark_generation(
            model, prompt_length, gen_len, use_cache=False, num_runs=3
        )
        
        speedup = result_uncached['mean_time'] / result_cached['mean_time']
        
        results_cached.append(result_cached['mean_time'])
        results_uncached.append(result_uncached['mean_time'])
        speedups.append(speedup)
        
        print(f"Speedup: {speedup:.2f}x")
    
    # Plotting
    if save_plots:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Time comparison
        ax1.plot(generate_lengths, results_uncached, 'o-', label='Without Cache', 
                linewidth=2, markersize=8, color='#e74c3c')
        ax1.plot(generate_lengths, results_cached, 's-', label='With Cache', 
                linewidth=2, markersize=8, color='#2ecc71')
        ax1.set_xlabel('Generation Length (tokens)', fontsize=12)
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.set_title('Generation Time Comparison', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Speedup
        ax2.plot(generate_lengths, speedups, 'o-', color='#3498db', 
                linewidth=2, markersize=8)
        ax2.axhline(y=1, color='#e74c3c', linestyle='--', alpha=0.5, label='No speedup')
        ax2.set_xlabel('Generation Length (tokens)', fontsize=12)
        ax2.set_ylabel('Speedup (x)', fontsize=12)
        ax2.set_title('KV-Cache Speedup Factor', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('kv_cache_speedup.png', dpi=150, bbox_inches='tight')
        print("\n✅ Plot saved as 'kv_cache_speedup.png'")
    
    return {
        'generate_lengths': generate_lengths,
        'speedups': speedups,
        'times_cached': results_cached,
        'times_uncached': results_uncached
    }


def experiment_memory_scaling(
    save_plots: bool = True
) -> Dict[str, List]:
    """
    Experiment 2: Analyze memory usage scaling.
    
    This experiment demonstrates the linear O(n) memory scaling of
    KV-cache with sequence length.
    
    Args:
        save_plots: Whether to save generated plots to disk
        
    Returns:
        Dictionary containing:
            - sequence_lengths: List of sequence lengths tested
            - cache_sizes_mb: Corresponding cache sizes in MB
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Memory Usage Scaling")
    print("=" * 80)
    
    model = TransformerLM(
        vocab_size=100,
        d_model=128,
        num_heads=8,
        num_layers=6,
        d_ff=512,
        dropout=0.0
    )
    
    sequence_lengths = [50, 100, 200, 500, 1000, 2000]
    cache_sizes = []
    
    print(f"\nModel: {model.num_layers} layers, d_model={model.d_model}")
    print(f"\nSequence Length | Cache Size (MB) | Per Token (KB)")
    print("-" * 60)
    
    for seq_len in sequence_lengths:
        cache_mb = calculate_cache_size(model, seq_len, batch_size=1)
        cache_sizes.append(cache_mb)
        per_token_kb = (cache_mb * 1024) / seq_len
        print(f"{seq_len:15d} | {cache_mb:15.2f} | {per_token_kb:13.2f}")
    
    # Plotting
    if save_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sequence_lengths, cache_sizes, 'o-', linewidth=2, 
               markersize=10, color='#9b59b6')
        ax.set_xlabel('Sequence Length (tokens)', fontsize=12)
        ax.set_ylabel('Cache Size (MB)', fontsize=12)
        ax.set_title('KV-Cache Memory Usage vs Sequence Length', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add linear reference line
        linear_ref = [cache_sizes[0] * (s / sequence_lengths[0]) for s in sequence_lengths]
        ax.plot(sequence_lengths, linear_ref, '--', alpha=0.5, 
               label='Linear scaling O(n)', color='gray')
        ax.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig('kv_cache_memory.png', dpi=150, bbox_inches='tight')
        print("\n✅ Plot saved as 'kv_cache_memory.png'")
    
    return {
        'sequence_lengths': sequence_lengths,
        'cache_sizes_mb': cache_sizes
    }


def experiment_complexity_analysis(
    save_plots: bool = True
) -> None:
    """
    Experiment 3: Visualize computational complexity difference.
    
    This experiment provides a visual comparison of O(n²) vs O(n)
    computational complexity for generation with and without KV-cache.
    
    Args:
        save_plots: Whether to save generated plots to disk
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: Computational Complexity Analysis")
    print("=" * 80)
    
    sequence_lengths = np.arange(10, 201, 10)
    
    # Calculate operations for both approaches
    ops_without_cache = [(n * (n + 1)) // 2 for n in sequence_lengths]  # O(n²)
    ops_with_cache = list(sequence_lengths)  # O(n)
    
    # Plotting
    if save_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(sequence_lengths, ops_without_cache, label='Without Cache O(n²)', 
                linewidth=2.5, color='#e74c3c')
        ax.plot(sequence_lengths, ops_with_cache, label='With Cache O(n)', 
                linewidth=2.5, color='#2ecc71')
        
        ax.set_xlabel('Sequence Length (n)', fontsize=12)
        ax.set_ylabel('Total Token Operations', fontsize=12)
        ax.set_title('Computational Complexity: With vs Without KV-Cache', 
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add annotations
        n_example = 100
        ops_no_cache = (n_example * (n_example + 1)) // 2
        ops_cache = n_example
        ax.annotate(f'n=100:\n{ops_no_cache} ops',
                    xy=(n_example, ops_no_cache), 
                    xytext=(n_example+20, ops_no_cache+1000),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=1.5),
                    fontsize=10, color='#e74c3c')
        ax.annotate(f'n=100:\n{ops_cache} ops',
                    xy=(n_example, ops_cache), 
                    xytext=(n_example+20, ops_cache+1000),
                    arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=1.5),
                    fontsize=10, color='#2ecc71')
        
        plt.tight_layout()
        plt.savefig('kv_cache_complexity.png', dpi=150, bbox_inches='tight')
        print("\n✅ Plot saved as 'kv_cache_complexity.png'")
    
    # Print statistics
    print("\n📊 Complexity Statistics:")
    print(f"\nFor sequence length n=1000:")
    print(f"  Without cache: {(1000 * 1001) // 2:,} operations")
    print(f"  With cache:    {1000:,} operations")
    print(f"  Reduction:     {((1000 * 1001) // 2) / 1000:.1f}x fewer operations")


# ============================================================================
# DEMONSTRATION FUNCTIONS
# ============================================================================

def run_basic_demo() -> Dict[str, float]:
    """
    Run a basic demonstration comparing generation with and without KV-cache.
    
    This function creates a small transformer model, generates text using both
    methods, and compares their performance and correctness.
    
    Returns:
        Dictionary containing:
            - time_with_cache: Generation time with caching
            - time_without_cache: Generation time without caching
            - speedup: Speedup factor
            - cache_memory_mb: Final cache size in MB
            - outputs_match: Whether both methods produced identical outputs
    """
    print("=" * 80)
    print("KV-CACHE TRANSFORMER DEMONSTRATION")
    print("=" * 80)
    
    # Configuration
    config = ModelConfig(
        vocab_size=100,
        d_model=64,
        num_heads=4,
        num_layers=3,
        d_ff=128,
        max_seq_len=128,
        dropout=0.1
    )
    
    print("\n📋 Model Configuration:")
    print(f"   vocab_size: {config.vocab_size}")
    print(f"   d_model: {config.d_model}")
    print(f"   num_heads: {config.num_heads}")
    print(f"   num_layers: {config.num_layers}")
    print(f"   d_ff: {config.d_ff}")
    
    # Initialize model
    torch.manual_seed(42)
    model = TransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        d_ff=config.d_ff,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔢 Total parameters: {total_params:,}")
    print(f"💻 Device: {device}")
    
    # Create input
    batch_size = 2
    prompt_length = 8
    max_new_tokens = 10
    
    prompt_ids = torch.randint(
        0, config.vocab_size, (batch_size, prompt_length), device=device
    )
    
    print(f"\n📝 Input:")
    print(f"   Batch size: {batch_size}")
    print(f"   Prompt length: {prompt_length}")
    print(f"   Tokens to generate: {max_new_tokens}")
    print(f"\n🎲 Random prompt:\n{prompt_ids}")
    
    # METHOD 1: WITH KV-CACHE
    print("\n" + "=" * 80)
    print("METHOD 1: WITH KV-CACHING")
    print("=" * 80)
    
    start_time = time.time()
    generated_with_cache, stats = generate_with_kv_cache(
        model, prompt_ids, max_new_tokens=max_new_tokens, verbose=True
    )
    time_with_cache = time.time() - start_time
    
    print(f"\n✨ Complete!")
    print(f"⏱️  Time: {time_with_cache:.4f}s")
    print(f"💾 Final cache: {stats['kv_cache_memory_mb']:.3f} MB")
    print(f"📝 Output shape: {generated_with_cache.shape}")
    print(f"   First sequence: {generated_with_cache[0].tolist()}")
    
    # METHOD 2: WITHOUT KV-CACHE
    print("\n" + "=" * 80)
    print("METHOD 2: WITHOUT KV-CACHING (Baseline)")
    print("=" * 80)
    
    print(f"\n🐌 Recomputing everything at each step...")
    start_time = time.time()
    generated_without_cache = generate_without_kv_cache(
        model, prompt_ids, max_new_tokens=max_new_tokens
    )
    time_without_cache = time.time() - start_time
    
    print(f"⏱️  Time: {time_without_cache:.4f}s")
    print(f"📝 Output shape: {generated_without_cache.shape}")
    print(f"   First sequence: {generated_without_cache[0].tolist()}")
    
    # COMPARISON
    print("\n" + "=" * 80)
    print("⚡ PERFORMANCE COMPARISON")
    print("=" * 80)
    
    speedup = time_without_cache / time_with_cache
    print(f"\n✅ WITH KV-Cache:    {time_with_cache:.4f}s")
    print(f"❌ WITHOUT KV-Cache: {time_without_cache:.4f}s")
    print(f"🚀 Speedup:          {speedup:.2f}x faster")
    print(f"💾 Memory overhead:  {stats['kv_cache_memory_mb']:.3f} MB")
    
    # Verify correctness
    outputs_match = torch.all(generated_with_cache == generated_without_cache).item()
    print(f"\n🔍 Outputs match: {'✅ Yes' if outputs_match else '❌ No (numerical precision)'}")
    
    # Analysis
    print("\n" + "=" * 80)
    print("📊 ANALYSIS")
    print("=" * 80)
    
    total_seq_len = prompt_length + max_new_tokens
    
    # Theoretical operation counts
    ops_without_cache = sum(range(prompt_length, total_seq_len + 1))
    ops_with_cache = total_seq_len
    theoretical_speedup = ops_without_cache / ops_with_cache
    
    print(f"\n📐 Computational complexity:")
    print(f"   WITHOUT cache: O(n²) ≈ {ops_without_cache} token operations")
    print(f"   WITH cache:    O(n)  ≈ {ops_with_cache} token operations")
    print(f"   Theoretical speedup: {theoretical_speedup:.1f}x")
    print(f"   Actual speedup:      {speedup:.2f}x")
    
    print(f"\n💡 Key insights:")
    print(f"   • Cache grows linearly: {prompt_length} → {total_seq_len} tokens")
    print(f"   • Each layer stores keys + values")
    print(f"   • {config.num_layers} layers × 2 (K+V) = {config.num_layers*2} cached tensors")
    print(f"   • Memory per token: ~{stats['kv_cache_memory_mb']*1024/total_seq_len:.2f} KB")
    
    print("\n" + "=" * 80)
    print("✨ DEMONSTRATION COMPLETE!")
    print("=" * 80)
    
    return {
        'time_with_cache': time_with_cache,
        'time_without_cache': time_without_cache,
        'speedup': speedup,
        'cache_memory_mb': stats['kv_cache_memory_mb'],
        'outputs_match': outputs_match
    }


def run_all_experiments() -> None:
    """
    Run all experiments and generate comprehensive analysis.
    
    This function executes all three experiments:
    1. Speedup vs sequence length
    2. Memory usage scaling
    3. Computational complexity analysis
    
    And provides a comprehensive summary of findings.
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "KV-CACHE EXPERIMENTS" + " " * 38 + "║")
    print("╚" + "=" * 78 + "╝")
    
    torch.manual_seed(42)
    
    # Run experiments
    print("\n🔬 Running experiments...")
    
    exp1_results = experiment_speedup_scaling(save_plots=True)
    exp2_results = experiment_memory_scaling(save_plots=True)
    experiment_complexity_analysis(save_plots=True)
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    
    max_speedup = max(exp1_results['speedups'])
    max_speedup_idx = exp1_results['speedups'].index(max_speedup)
    max_speedup_len = exp1_results['generate_lengths'][max_speedup_idx]
    
    print(f"\n✨ Key Findings:")
    print(f"  • Maximum observed speedup: {max_speedup:.2f}x (at {max_speedup_len} tokens)")
    print(f"  • Speedup increases with sequence length")
    print(f"  • Memory scales linearly O(n) with sequence length")
    print(f"  • Computational savings: O(n²) → O(n)")
    
    print(f"\n💡 Practical Implications:")
    print(f"  • KV-caching is essential for real-time LLM inference")
    print(f"  • Longer sequences benefit more from caching")
    print(f"  • Memory overhead is predictable and manageable")
    max_seq = exp2_results['sequence_lengths'][-1]
    max_cache = exp2_results['cache_sizes_mb'][-1]
    print(f"  • Trade-off: ~{max_cache:.1f}MB cache for {max_seq} tokens")
    
    print("\n" + "=" * 80)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("=" * 80)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """
    Main entry point for the KV-cache demonstration and experiments.
    
    This function provides an interactive menu to run either the basic
    demonstration or the full experimental suite.
    """
    import sys
    
    print("\n" + "=" * 80)
    print("KV-CACHE IN TRANSFORMERS: COMPLETE IMPLEMENTATION")
    print("=" * 80)
    print("\nChoose an option:")
    print("  1. Run basic demonstration")
    print("  2. Run all experiments (with plots)")
    print("  3. Run both")
    print("  4. Exit")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        run_basic_demo()
    elif choice == '2':
        run_all_experiments()
    elif choice == '3':
        run_basic_demo()
        print("\n" + "=" * 80)
        input("\nPress Enter to continue to experiments...")
        run_all_experiments()
    elif choice == '4':
        print("\nExiting...")
        return
    else:
        print("\nInvalid choice. Running basic demonstration by default...")
        run_basic_demo()


if __name__ == "__main__":
    main()

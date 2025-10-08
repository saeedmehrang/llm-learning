# LLM Learning

A collection of code and tools for learning the fundamental principles of training and fine-tuning Large Language Models (LLMs).

## Overview

This repository serves as a practical learning resource for understanding the core concepts and techniques involved in building, training, and fine-tuning language models from scratch.

## Environment Setup

**Install uv if you haven't**

`curl -LsSf https://astral.sh/uv/install.sh | sh`

**Create virtual environment and install dependencies**

´´´
uv venv
source .venv/bin/activate  # or `.venv/Scripts/activate` on Windows
uv pip install -e .
´´´

**Or use uv sync for more advanced dependency management**


`uv sync`


## Contents

This repository contains educational implementations of state-of-the-art (SOTA) techniques used in modern large language models.

### SOTA Implementations

#### 1. **KV-Cache** ([kv_caching.py](kv_caching.py))
Essential optimization for efficient autoregressive generation.
- Full transformer architecture with KV-cache support
- Performance benchmarking showing O(n²) → O(n) improvement
- Memory usage analysis and visualizations
- Used in: All modern LLMs (GPT, LLaMA, Claude, etc.)

#### 2. **Positional Embeddings** ([positional_embeddings.py](positional_embeddings.py))
Methods for encoding position information in sequences.
- **Sinusoidal Positional Embeddings** - Original Transformer approach
- **RoPE (Rotary Position Embeddings)** - Relative position encoding
- Comparative visualizations and attention pattern analysis
- Used in: RoPE in LLaMA, GPT-NeoX, PaLM

#### 3. **SwiGLU Activation** ([swiglu.py](swiglu.py))
Advanced activation function combining Swish with gating mechanism.
- SwiGLU feed-forward network implementation
- Comparison with traditional ReLU/GELU activations
- Performance benchmarks and gradient flow analysis
- Used in: LLaMA, PaLM, GPT-4 (rumored)

#### 4. **Grouped-Query Attention (GQA)** ([grouped_query_attention.py](grouped_query_attention.py))
Memory-efficient attention mechanism reducing KV cache size.
- Multi-Head Attention (MHA) baseline implementation
- Grouped-Query Attention with shared KV heads
- Memory efficiency comparisons (up to 8x reduction)
- Used in: LLaMA 2, Mistral, Mixtral

#### 5. **Multi-Head Latent Attention (MLA)** ([multihead_latent_attention.py](multihead_latent_attention.py))
Advanced attention with latent KV compression for extreme memory efficiency.
- Low-dimensional latent space compression
- 4-8x memory reduction compared to standard attention
- Autoregressive generation with compressed caching
- Used in: DeepSeek-V2, DeepSeek-V3

## Learning Goals

This repository aims to help you understand:

- **Transformer architecture fundamentals** - attention mechanisms, multi-head attention, feedforward networks
- **SOTA optimization techniques** - KV-caching, GQA, MLA for efficient inference
- **Modern architectural components** - RoPE, SwiGLU, advanced attention mechanisms
- **Performance analysis** - benchmarking, complexity analysis, memory profiling
- **Implementation details** - from-scratch implementations in PyTorch

## Requirements

```bash
pip install torch matplotlib numpy
```

## Future Topics

Planned additions to this learning repository:

- [x] Positional encodings (Sinusoidal, RoPE)
- [x] Advanced attention mechanisms (GQA, MLA)
- [x] Modern activation functions (SwiGLU)
- [ ] Training loops and optimization strategies
- [ ] Fine-tuning techniques (LoRA, QLoRA, prefix tuning)
- [ ] Tokenization and vocabulary construction
- [ ] Flash Attention and other attention optimizations
- [ ] Model parallelism and distributed training
- [ ] Quantization and compression techniques
- [ ] Evaluation metrics and benchmarking

## Contributing

This is a personal learning repository. Feel free to fork and adapt for your own learning journey.

## License

MIT License - Free to use for educational purposes.

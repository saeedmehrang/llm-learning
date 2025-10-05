# LLM Learning

A collection of code and tools for learning the fundamental principles of training and fine-tuning Large Language Models (LLMs).

## Overview

This repository serves as a practical learning resource for understanding the core concepts and techniques involved in building, training, and fine-tuning language models from scratch.

## Contents

### KV-Cache Implementation (`kv_cache_consolidated.py`)

A comprehensive implementation demonstrating Key-Value caching in transformer language models:

- **Full transformer architecture** with multi-head attention and KV-cache support
- **Generation functions** comparing performance with and without caching
- **Performance benchmarking** utilities for measuring speedup and memory usage
- **Detailed experiments** with visualizations showing:
  - Speedup vs sequence length (O(n²) → O(n) improvement)
  - Memory usage scaling (linear growth with sequence length)
  - Computational complexity analysis

**Key Features:**
- Demonstrates why KV-caching is essential for efficient LLM inference
- Shows practical speedup improvements (increasing with longer sequences)
- Includes memory overhead analysis
- Provides both educational demonstrations and experimental validation

**Usage:**
```bash
python kv_cache_consolidated.py
```

Choose from:
1. Basic demonstration (quick comparison)
2. Full experiments with plots
3. Both

## Learning Goals

This repository aims to help you understand:

- **Transformer architecture fundamentals** - attention mechanisms, multi-head attention, feedforward networks
- **Optimization techniques** - KV-caching for efficient autoregressive generation
- **Performance analysis** - benchmarking, complexity analysis, memory profiling
- **Implementation details** - from-scratch transformer implementation in PyTorch

## Requirements

```bash
pip install torch matplotlib numpy
```

## Future Topics

Planned additions to this learning repository:

- [ ] Positional encodings (absolute, relative, RoPE)
- [ ] Training loops and optimization strategies
- [ ] Fine-tuning techniques (LoRA, QLoRA, prefix tuning)
- [ ] Tokenization and vocabulary construction
- [ ] Attention variants (flash attention, sparse attention)
- [ ] Model parallelism and distributed training
- [ ] Quantization and compression techniques
- [ ] Evaluation metrics and benchmarking

## Contributing

This is a personal learning repository. Feel free to fork and adapt for your own learning journey.

## License

MIT License - Free to use for educational purposes.

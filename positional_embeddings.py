"""
Positional Embeddings Module - PyTorch Implementation
======================================================

A PyTorch-based educational module for understanding and using positional embeddings
in transformer models.

Overview
--------
This module provides reusable PyTorch implementations of:
- **Sinusoidal Positional Embeddings** (original Transformer)
- **Rotary Position Embeddings (RoPE)** (used in LLaMA, GPT-NeoX)

Quick Start
-----------
Import the module::

    from positional_embeddings import (
        SinusoidalPositionalEmbedding,
        RotaryPositionalEmbedding,
        RoPEAttentionAnalyzer,
        visualize_embeddings_comparison
    )

Sinusoidal Positional Embeddings::

    # Create the embedding module
    sin_embed = SinusoidalPositionalEmbedding(d_model=512, max_len=1000)

    # Get positional embeddings for a sequence
    pos_embeddings = sin_embed(seq_len=100)  # Shape: (100, 512)

    # Add to token embeddings
    embeddings_with_pos = token_embeddings + pos_embeddings

Rotary Position Embeddings (RoPE)::

    # Create RoPE module
    rope = RotaryPositionalEmbedding(d_model=64, max_len=2048)

    # Apply to single vector at specific position
    query = torch.randn(64)
    query_rotated = rope(query, position=5)

    # Apply to entire sequence
    sequence = torch.randn(batch_size, seq_len, d_model)
    rotated_sequence = rope.apply_to_sequence(sequence)

Usage in Transformer Models
---------------------------
Sinusoidal Embeddings (Vanilla Transformer)::

    class TransformerWithSinusoidal(torch.nn.Module):
        def __init__(self, vocab_size, d_model, max_len):
            super().__init__()
            self.token_embed = torch.nn.Embedding(vocab_size, d_model)
            self.pos_embed = SinusoidalPositionalEmbedding(d_model, max_len)

        def forward(self, x):
            seq_len = x.shape[1]
            embeddings = self.token_embed(x) + self.pos_embed(seq_len)
            return embeddings

RoPE (Modern LLMs)::

    class AttentionWithRoPE(torch.nn.Module):
        def __init__(self, d_model, num_heads):
            super().__init__()
            self.d_head = d_model // num_heads
            self.rope = RotaryPositionalEmbedding(self.d_head)

        def forward(self, x):
            # Apply RoPE to queries and keys
            Q_rotated = self.rope.apply_to_sequence(Q)
            K_rotated = self.rope.apply_to_sequence(K)

            # Compute attention
            attention = torch.softmax(Q_rotated @ K_rotated.transpose(-2, -1), dim=-1)
            return attention @ V

Key Concepts
------------
Sinusoidal Embeddings:
    - **Absolute positioning**: Each position has a unique embedding
    - **Fixed patterns**: Uses sine and cosine of different frequencies
    - **Additive**: Simply added to token embeddings
    - **Used in**: Original Transformer, BERT

Rotary Position Embeddings (RoPE):
    - **Relative positioning**: Encodes position through rotation
    - **Multiplicative**: Rotates the embedding vectors
    - **Distance-aware**: Dot product naturally decays with distance
    - **Used in**: LLaMA, GPT-NeoX, PaLM

Why RoPE?
    1. Better for long sequences - handles longer sequences than seen during training
    2. Relative positions - attention depends on relative distance
    3. Efficient - no extra parameters
    4. Natural decay - attention naturally decreases with distance

Mathematical Details
-------------------
Sinusoidal Formula::

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

RoPE Rotation::

    For dimension pair (i):
    θ_i = 10000^(-2i/d_model)

    Rotation at position m:
    [x'_{2i}  ]   [cos(m·θ_i)  -sin(m·θ_i)] [x_{2i}  ]
    [x'_{2i+1}] = [sin(m·θ_i)   cos(m·θ_i)] [x_{2i+1}]

Running Examples
---------------
Run the complete demo::

    python positional_embeddings.py

This will:
    1. Generate comparison visualizations
    2. Analyze attention patterns
    3. Create animated rotation demonstration
    4. Show example usage patterns

References
----------
- Attention Is All You Need (https://arxiv.org/abs/1706.03762) - Original Transformer
- RoFormer (https://arxiv.org/abs/2104.09864) - RoPE paper
- LLaMA (https://arxiv.org/abs/2302.13971) - Uses RoPE in production
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.patches as mpatches
import os


class SinusoidalPositionalEmbedding(torch.nn.Module):
    """
    Sinusoidal Positional Embeddings as used in the original Transformer paper.

    These embeddings add absolute position information to token embeddings.
    Uses fixed sine and cosine functions of different frequencies.

    Args:
        d_model: Dimension of the embeddings
        max_len: Maximum sequence length
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Precompute positional embeddings
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer("pe", pe)

    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Returns positional embeddings for the given sequence length.

        Args:
            seq_len: Length of the sequence

        Returns:
            Tensor of shape (seq_len, d_model)
        """
        return self.pe[:seq_len]

    def get_embeddings(self) -> torch.Tensor:
        """Returns all precomputed embeddings."""
        return self.pe


class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Rotary Position Embeddings (RoPE) as used in models like LLaMA and GPT-NeoX.

    Instead of adding position information, RoPE rotates the embeddings in a way
    that naturally encodes relative positions through dot products.

    Key insight: When you rotate two vectors by different amounts and take their
    dot product, the result depends on the angle difference (relative position).

    Args:
        d_model: Dimension of the embeddings (must be even)
        max_len: Maximum sequence length
        base: Base for the geometric progression of frequencies (default: 10000)
    """

    def __init__(self, d_model: int, max_len: int = 5000, base: float = 10000.0):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"

        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Compute rotation frequencies (theta values)
        # Lower dimensions rotate faster, higher dimensions rotate slower
        theta = base ** (-2 * torch.arange(0, d_model // 2) / d_model)
        self.register_buffer("theta", theta)

    def forward(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """
        Apply rotary position embedding to vector x at given position.

        Args:
            x: Input tensor of shape (d_model,) or (batch_size, d_model)
            position: Position index

        Returns:
            Rotated tensor of same shape as input
        """
        # Handle batch dimension
        if x.dim() == 1:
            return self._apply_rope_single(x, position)
        else:
            # Apply to each item in batch
            return torch.stack(
                [self._apply_rope_single(x[i], position) for i in range(x.shape[0])]
            )

    def _apply_rope_single(self, x: torch.Tensor, position: int) -> torch.Tensor:
        """Apply RoPE to a single vector."""
        # Compute angles for this position
        m_theta = position * self.theta
        cos_vals = torch.cos(m_theta)
        sin_vals = torch.sin(m_theta)

        # Apply rotation to each dimension pair
        x_rotated = x.clone()
        for i in range(self.d_model // 2):
            x1 = x[2 * i]
            x2 = x[2 * i + 1]
            x_rotated[2 * i] = x1 * cos_vals[i] - x2 * sin_vals[i]
            x_rotated[2 * i + 1] = x1 * sin_vals[i] + x2 * cos_vals[i]

        return x_rotated

    def apply_to_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to a sequence of vectors.

        Args:
            x: Tensor of shape (seq_len, d_model) or (batch_size, seq_len, d_model)

        Returns:
            Rotated tensor of same shape
        """
        if x.dim() == 2:
            seq_len, _ = x.shape
            return torch.stack([self.forward(x[pos], pos) for pos in range(seq_len)])
        else:
            batch_size, seq_len, _ = x.shape
            result = []
            for b in range(batch_size):
                result.append(
                    torch.stack(
                        [self.forward(x[b, pos], pos) for pos in range(seq_len)]
                    )
                )
            return torch.stack(result)


class RoPEAttentionAnalyzer:
    """
    Utility class for analyzing attention patterns with RoPE.

    Demonstrates how RoPE naturally creates attention decay based on
    relative distance between positions.
    """

    def __init__(self, rope: RotaryPositionalEmbedding):
        self.rope = rope

    def compute_attention_pattern(
        self, query_vector: torch.Tensor, max_len: int
    ) -> torch.Tensor:
        """
        Compute attention scores between a query vector and keys at all positions.

        This shows how attention score changes with relative distance.

        Args:
            query_vector: Query vector of shape (d_model,)
            max_len: Number of positions to compute

        Returns:
            Tensor of shape (max_len,) containing attention scores
        """
        scores = []
        for pos in range(max_len):
            # Apply RoPE to query at this position
            q_rotated = self.rope.forward(query_vector, pos)
            # Apply RoPE to key at position 0
            k_rotated = self.rope.forward(query_vector, 0)
            # Compute dot product (attention score)
            score = torch.dot(q_rotated, k_rotated)
            scores.append(score.item())
        return torch.tensor(scores)

    def compute_relative_attention(
        self,
        query_vector: torch.Tensor,
        key_vector: torch.Tensor,
        query_pos: int,
        key_pos: int,
    ) -> float:
        """
        Compute attention score between query at query_pos and key at key_pos.

        Args:
            query_vector: Query vector
            key_vector: Key vector
            query_pos: Query position
            key_pos: Key position

        Returns:
            Attention score (dot product)
        """
        q_rotated = self.rope.forward(query_vector, query_pos)
        k_rotated = self.rope.forward(key_vector, key_pos)
        return torch.dot(q_rotated, k_rotated).item()


# ============================================================================
# VISUALIZATION FUNCTIONS (for educational purposes)
# ============================================================================


def visualize_embeddings_comparison(
    sin_embed: SinusoidalPositionalEmbedding,
    rope: RotaryPositionalEmbedding,
    seq_len: int = 20,
    filename: str = "rope_vs_sinusoid.png",
    save_dir: str = 'artifacts/images',
):
    """
    Visualize and compare Sinusoidal and RoPE embeddings.

    Args:
        sin_embed: Sinusoidal embedding module
        rope: RoPE embedding module
        seq_len: Sequence length to visualize
        filename: Filename to save the figure
        save_dir: Directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    # Get sinusoidal embeddings
    sin_embeddings = sin_embed.forward(seq_len).numpy()

    # Generate RoPE embeddings (rotate a sample vector at each position)
    sample_query = torch.randn(rope.d_model)
    sample_query = sample_query / torch.norm(sample_query)
    rope_embeddings = rope.apply_to_sequence(
        sample_query.unsqueeze(0).repeat(seq_len, 1)
    ).numpy()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sinusoidal
    im1 = axes[0].imshow(sin_embeddings.T, aspect="auto", cmap="RdBu")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Dimension")
    axes[0].set_title("Sinusoidal Positional Embeddings")
    plt.colorbar(im1, ax=axes[0])

    # RoPE
    im2 = axes[1].imshow(rope_embeddings.T, aspect="auto", cmap="RdBu")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Dimension")
    axes[1].set_title("RoPE: Same Vector Rotated")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def visualize_attention_decay(
    rope: RotaryPositionalEmbedding,
    seq_len: int = 20,
    filename: str = "rope_attention_pattern.png",
    save_dir: str = 'artifacts/images',
):
    """
    Visualize how attention decays with distance using RoPE.

    Args:
        rope: RoPE embedding module
        seq_len: Sequence length to analyze
        filename: Filename to save the figure
        save_dir: Directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    # Create analyzer
    analyzer = RoPEAttentionAnalyzer(rope)

    # Generate random query vector
    sample_query = torch.randn(rope.d_model)
    sample_query = sample_query / torch.norm(sample_query)

    # Calculate attention pattern
    attention_pattern = analyzer.compute_attention_pattern(sample_query, seq_len)

    # Visualize
    plt.figure(figsize=(10, 5))
    plt.plot(attention_pattern.numpy(), linewidth=2.5, color="darkgreen")
    plt.axhline(y=attention_pattern[0].item(), color="red", linestyle="--", alpha=0.5)
    plt.xlabel("Query Position")
    plt.ylabel("Attention Score with Key at Position 0")
    plt.title("RoPE: Attention Score vs Relative Distance")
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path)
    plt.show()

    print(f"Score at position 0 (distance=0): {attention_pattern[0]:.3f}")
    print(f"Score at position 5 (distance=5): {attention_pattern[5]:.3f}")
    print(f"Score at position 10 (distance=10): {attention_pattern[10]:.3f}")


def create_rope_animation(
    rope: RotaryPositionalEmbedding,
    max_positions: int = 50,
    filename: str = "rope_animation.gif",
    save_dir: str = 'artifacts/images',
):
    """
    Create an animated visualization showing how RoPE rotates at different speeds.

    Args:
        rope: RoPE embedding module
        max_positions: Number of positions to animate
        filename: Filename to save the figure
        save_dir: Directory to save the figure
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    # configure
    d_model = rope.d_model
    num_pairs = d_model // 2
    theta = rope.theta.numpy()

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)

    # Axes for each dimension pair (circles)
    circle_axes = [fig.add_subplot(gs[0, i]) for i in range(min(4, num_pairs))]
    # Axes for showing all rotations together
    combined_ax = fig.add_subplot(gs[1, :])
    # Axes for attention decay visualization
    attention_ax = fig.add_subplot(gs[2, :])

    # Colors for each dimension pair
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    # Setup circle plots
    arrows = []
    position_texts = []

    for idx, ax in enumerate(circle_axes):
        if idx >= num_pairs:
            break

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="k", linewidth=0.5)
        ax.axvline(x=0, color="k", linewidth=0.5)

        # Draw circle
        circle = Circle((0, 0), 1, fill=False, color=colors[idx], linewidth=2)
        ax.add_patch(circle)

        # Initial arrow
        arrow = FancyArrowPatch(
            (0, 0),
            (1, 0),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=3,
            color=colors[idx],
        )
        ax.add_patch(arrow)
        arrows.append(arrow)

        # Labels
        speed_text = f"Speed: {theta[idx]:.4f} rad/pos"
        ax.set_title(
            f"Dimension Pair {idx}\n{speed_text}",
            fontsize=11,
            fontweight="bold",
            color=colors[idx],
        )
        ax.set_xlabel("Dimension 2i", fontsize=9)
        ax.set_ylabel(f"Dimension 2i+1", fontsize=9)

        # Position text
        pos_text = ax.text(
            0, -1.35, "Position: 0", ha="center", fontsize=10, fontweight="bold"
        )
        position_texts.append(pos_text)

    # Setup combined view
    combined_ax.set_xlim(-1.5, 1.5)
    combined_ax.set_ylim(-1.5, 1.5)
    combined_ax.set_aspect("equal")
    combined_ax.grid(True, alpha=0.3)
    combined_ax.axhline(y=0, color="k", linewidth=0.5)
    combined_ax.axvline(x=0, color="k", linewidth=0.5)
    combined_ax.set_title(
        "All Dimension Pairs Together (Clock Hands at Different Speeds)",
        fontsize=13,
        fontweight="bold",
    )
    combined_ax.set_xlabel("Real Component", fontsize=10)
    combined_ax.set_ylabel("Imaginary Component", fontsize=10)

    # Draw reference circle
    ref_circle = Circle(
        (0, 0), 1, fill=False, color="gray", linewidth=1, linestyle="--", alpha=0.5
    )
    combined_ax.add_patch(ref_circle)

    # Arrows for combined view
    combined_arrows = []
    for idx in range(min(num_pairs, 4)):
        arrow = FancyArrowPatch(
            (0, 0),
            (1, 0),
            arrowstyle="->",
            mutation_scale=15,
            linewidth=2.5,
            color=colors[idx],
            alpha=0.8,
        )
        combined_ax.add_patch(arrow)
        combined_arrows.append(arrow)

    # Legend for combined view
    legend_elements = [
        mpatches.Patch(color=colors[i], label=f"Pair {i} (θ={theta[i]:.4f})")
        for i in range(min(num_pairs, 4))
    ]
    combined_ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1), fontsize=9
    )

    combined_pos_text = combined_ax.text(
        0, 1.3, "Position: 0", ha="center", fontsize=12, fontweight="bold"
    )

    # Setup attention decay plot
    positions = np.arange(max_positions)

    # Initialize plot lines
    attention_lines = []
    for idx in range(min(num_pairs, 4)):
        (line,) = attention_ax.plot(
            [], [], color=colors[idx], linewidth=2.5, label=f"Pair {idx}", alpha=0.8
        )
        attention_lines.append(line)

    attention_ax.set_xlim(0, max_positions)
    attention_ax.set_ylim(-1.1, 1.1)
    attention_ax.grid(True, alpha=0.3)
    attention_ax.set_xlabel("Relative Position Distance", fontsize=11)
    attention_ax.set_ylabel("Attention Score (Cosine Similarity)", fontsize=11)
    attention_ax.set_title(
        "How Attention Decays with Distance (for each rotation speed)",
        fontsize=13,
        fontweight="bold",
    )
    attention_ax.legend(loc="upper right", fontsize=9)
    attention_ax.axhline(y=0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)

    # Animation function
    def animate(frame):
        position = frame % max_positions

        # Update individual circle plots
        for idx in range(len(arrows)):
            angle = theta[idx] * position
            x = np.cos(angle)
            y = np.sin(angle)
            arrows[idx].set_positions((0, 0), (x, y))
            position_texts[idx].set_text(f"Position: {position}")

        # Update combined view
        for idx in range(len(combined_arrows)):
            angle = theta[idx] * position
            x = np.cos(angle)
            y = np.sin(angle)
            combined_arrows[idx].set_positions((0, 0), (x, y))

        combined_pos_text.set_text(f"Position: {position}")

        # Update attention decay plot
        for idx in range(len(attention_lines)):
            # Compute cosine similarity between position 0 and all other positions
            angles_diff = theta[idx] * positions
            attention_scores = np.cos(angles_diff)
            attention_lines[idx].set_data(positions, attention_scores)

        return (
            arrows
            + combined_arrows
            + attention_lines
            + position_texts
            + [combined_pos_text]
        )

    # Create animation
    anim = FuncAnimation(
        fig, animate, frames=max_positions, interval=100, blit=False, repeat=True
    )
    anim.save(save_path, writer="pillow", fps=10)

    plt.suptitle(
        "RoPE: Rotary Position Embeddings Visualization\n"
        + "Watch how different dimension pairs rotate at different speeds (like clock hands)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )

    # Print explanation
    print("\n" + "=" * 70)
    print("WHAT YOU'RE SEEING:")
    print("=" * 70)
    print("\n📍 TOP ROW (Individual Circles):")
    print("   Each circle shows one dimension pair rotating at its own speed")
    print("   - Pair 0 (RED): Fastest rotation - completes many circles")
    print("   - Pair 3 (ORANGE): Slowest rotation - barely moves")
    print("\n🎯 MIDDLE (Combined View):")
    print("   All dimension pairs shown together - like multiple clock hands")
    print("   - Short-range: Fast rotations distinguish nearby positions")
    print("   - Long-range: Slow rotations distinguish distant positions")
    print("\n📉 BOTTOM (Attention Decay):")
    print("   Shows how attention score decreases with distance")
    print("   - Fast rotations (RED): Oscillate quickly, capture short-range")
    print("   - Slow rotations (ORANGE): Decay slowly, capture long-range")
    print("\n💡 KEY INSIGHT:")
    print("   The dot product between rotated vectors equals cos(angle_difference)")
    print("   As positions get farther apart, the angle difference increases,")
    print("   and cos(Δangle) decreases → attention naturally decays!")
    print("=" * 70)

    plt.show()

    print(f"\nAnimation saved to: {save_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Setup
    d_model = 8
    seq_len = 20

    # Create embedding modules
    sin_embed = SinusoidalPositionalEmbedding(d_model=d_model, max_len=seq_len)
    rope = RotaryPositionalEmbedding(d_model=d_model, max_len=seq_len)

    print("=" * 70)
    print("POSITIONAL EMBEDDINGS COMPARISON")
    print("=" * 70)
    print(f"\nModel dimension: {d_model}")
    print(f"Sequence length: {seq_len}")

    # save path
    save_dir = "artifacts/images/positional_embeddings"
    os.makedirs(save_dir, exist_ok=True)

    # Visualize embeddings comparison
    print("\n1. Generating embeddings comparison visualization...")
    visualize_embeddings_comparison(sin_embed, rope, seq_len, save_dir=save_dir)

    # Visualize attention decay
    print("\n2. Analyzing attention decay pattern...")
    visualize_attention_decay(rope, seq_len, save_dir=save_dir)

    # Create animation
    print("\n3. Creating RoPE rotation animation...")
    print("   (This may take a minute...)")
    create_rope_animation(rope, max_positions=50, save_dir=save_dir)

    # Demonstrate usage as reusable module
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Using as Reusable Modules")
    print("=" * 70)

    # Example 1: Apply sinusoidal embeddings to a sequence
    print("\nExample 1: Sinusoidal Embeddings")
    sequence_length = 10
    sin_pos_embed = sin_embed.forward(sequence_length)
    print(f"  Generated embeddings shape: {sin_pos_embed.shape}")
    print(f"  Device: {sin_pos_embed.device}")

    # Example 2: Apply RoPE to query/key vectors
    print("\nExample 2: RoPE for Attention")
    query = torch.randn(d_model)
    key = torch.randn(d_model)

    # Rotate at different positions
    query_rotated = rope.forward(query, position=5)
    key_rotated = rope.forward(key, position=3)

    print(f"  Query at position 5: {query_rotated[:4]}...")
    print(f"  Key at position 3: {key_rotated[:4]}...")

    # Compute attention score
    attention_score = torch.dot(query_rotated, key_rotated)
    print(f"  Attention score (relative distance=2): {attention_score:.4f}")

    # Example 3: Apply to full sequence
    print("\nExample 3: Apply RoPE to Sequence")
    sequence = torch.randn(seq_len, d_model)
    rotated_sequence = rope.apply_to_sequence(sequence)
    print(f"  Input sequence shape: {sequence.shape}")
    print(f"  Rotated sequence shape: {rotated_sequence.shape}")

    print("\n" + "=" * 70)
    print("All visualizations saved!")
    print("  - rope_vs_sinusoid.png")
    print("  - rope_attention_pattern.png")
    print("  - rope_animation.gif")
    print("=" * 70)

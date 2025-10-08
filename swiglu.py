"""
SwiGLU Feed-Forward Network Implementation

This module provides a PyTorch implementation of SwiGLU (Swish-Gated Linear Unit),
a gated activation function variant used in modern transformer architectures.

SwiGLU combines the Swish activation function with a gating mechanism, providing
improved performance over traditional ReLU-based feed-forward networks. This
implementation is suitable for both training language models and educational purposes.

Key Components:
    - Traditional FFN with ReLU (baseline)
    - GLU (Gated Linear Unit)
    - Swish/SiLU activation
    - SwiGLU activation
    - SwiGLU_FFN: Complete feed-forward network module

References:
    - GLU Variants Improve Transformer (Shazeer, 2020)
    - Swish: A Self-Gated Activation Function (Ramachandran et al., 2017)

Author: Generated for educational and practical use
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os


# ============================================================================
# Activation Functions
# ============================================================================

def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Swish activation function (also known as SiLU when beta=1).

    Formula: Swish(x) = x * σ(βx) where σ is the sigmoid function.

    Swish is a smooth, non-monotonic activation function that has been shown
    to outperform ReLU in deep networks. Unlike ReLU, it allows small negative
    values to pass through, which can help with gradient flow.

    Args:
        x (torch.Tensor): Input tensor of any shape. Each element is processed
            independently.
        beta (float, optional): Scaling parameter for the sigmoid. Default is 1.
            - Higher beta makes Swish behave more like ReLU
            - Lower beta makes it more linear
            - beta=1 gives SiLU (Sigmoid Linear Unit)

    Returns:
        torch.Tensor: Output tensor of same shape as input, with Swish
            activation applied element-wise.

    Examples:
        >>> x = torch.randn(2, 3)
        >>> output = swish(x)
        >>> output.shape
        torch.Size([2, 3])
    """
    return x * torch.sigmoid(beta * x)


def glu(x: torch.Tensor, W: torch.Tensor, V: torch.Tensor,
        b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    GLU (Gated Linear Unit) activation function.

    Formula: GLU(x) = (xW + b) ⊗ σ(xV + c)
    where ⊗ is element-wise multiplication and σ is the sigmoid function.

    GLU uses a gating mechanism where one linear transformation is gated by
    the sigmoid of another linear transformation, allowing the network to
    control information flow dynamically.

    Args:
        x (torch.Tensor): Input tensor of shape (..., d_model).
        W (torch.Tensor): First weight matrix of shape (d_model, d_ff) for
            the value component.
        V (torch.Tensor): Second weight matrix of shape (d_model, d_ff) for
            the gate component.
        b (torch.Tensor): Bias vector for value component of shape (d_ff,).
        c (torch.Tensor): Bias vector for gate component of shape (d_ff,).

    Returns:
        torch.Tensor: Output tensor of shape (..., d_ff) where the value
            component is element-wise multiplied by the sigmoid-gated component.

    Examples:
        >>> x = torch.randn(4, 10, 512)
        >>> W = torch.randn(512, 2048)
        >>> V = torch.randn(512, 2048)
        >>> b = torch.zeros(2048)
        >>> c = torch.zeros(2048)
        >>> output = glu(x, W, V, b, c)
        >>> output.shape
        torch.Size([4, 10, 2048])
    """
    # Value component (linear transformation)
    value = x @ W + b

    # Gate component (sigmoid activation)
    gate = torch.sigmoid(x @ V + c)

    # Element-wise multiplication: value gated by sigmoid
    return value * gate


def swiglu(x: torch.Tensor, W: torch.Tensor, V: torch.Tensor,
           b: torch.Tensor, c: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    SwiGLU (Swish-Gated Linear Unit) activation function.

    Formula: SwiGLU(x) = Swish(xW + b) ⊗ (xV + c)
    where ⊗ is element-wise multiplication.

    SwiGLU replaces the sigmoid gating in GLU with Swish activation, providing
    better gradient flow and improved performance. This is used in models like
    PaLM and LLaMA.

    Args:
        x (torch.Tensor): Input tensor of shape (..., d_model).
        W (torch.Tensor): First weight matrix of shape (d_model, d_ff) for
            the gate component.
        V (torch.Tensor): Second weight matrix of shape (d_model, d_ff) for
            the value component.
        b (torch.Tensor): Bias vector for gate component of shape (d_ff,).
        c (torch.Tensor): Bias vector for value component of shape (d_ff,).
        beta (float, optional): Swish activation parameter. Default is 1.

    Returns:
        torch.Tensor: Output tensor of shape (..., d_ff) where the
            Swish-activated gate component is element-wise multiplied by
            the linear value component.

    Examples:
        >>> x = torch.randn(4, 10, 512)
        >>> W = torch.randn(512, 2048)
        >>> V = torch.randn(512, 2048)
        >>> b = torch.zeros(2048)
        >>> c = torch.zeros(2048)
        >>> output = swiglu(x, W, V, b, c)
        >>> output.shape
        torch.Size([4, 10, 2048])
    """
    # Gate component (Swish activation) - controls information flow
    gate = swish(x @ W + b, beta)

    # Value component (linear transformation) - the actual information
    value = x @ V + c

    # Element-wise multiplication: value gated by Swish
    return gate * value


# ============================================================================
# Feed-Forward Network Modules
# ============================================================================

class TraditionalFFN(nn.Module):
    """
    Traditional Feed-Forward Network with ReLU activation.

    This is the standard FFN used in the original Transformer architecture.
    Formula: FFN(x) = ReLU(xW1 + b1)W2 + b2

    Attributes:
        d_model (int): Model dimension (input and output dimension).
        d_ff (int): Hidden dimension (typically 4 * d_model).
        fc1 (nn.Linear): First linear transformation.
        fc2 (nn.Linear): Second linear transformation.

    Args:
        d_model (int): Model dimension (input and output dimension).
        d_ff (int, optional): Hidden dimension. Defaults to 4 * d_model.

    Examples:
        >>> ffn = TraditionalFFN(d_model=512, d_ff=2048)
        >>> x = torch.randn(4, 10, 512)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([4, 10, 512])
    """

    def __init__(self, d_model: int, d_ff: int = None):
        """Initialize Traditional FFN with ReLU activation."""
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model

        self.d_model = d_model
        self.d_ff = d_ff

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Traditional FFN.

        Args:
            x (torch.Tensor): Input tensor of shape (..., d_model).

        Returns:
            torch.Tensor: Output tensor of shape (..., d_model).
        """
        hidden = F.relu(self.fc1(x))
        output = self.fc2(hidden)
        return output


class SwiGLU_FFN(nn.Module):
    """
    SwiGLU Feed-Forward Network module.

    This module implements the SwiGLU variant of feed-forward networks, which
    has been shown to outperform traditional ReLU-based FFNs in transformer
    models. Used in models like PaLM, LLaMA, and others.

    Architecture:
        1. Split projection into gate and value paths
        2. Apply Swish activation to gate path
        3. Element-wise multiply gate and value
        4. Project back to model dimension

    Parameter Efficiency:
        For similar parameter count to traditional FFN with expansion factor 4:
        - Traditional FFN: d_model * d_ff + d_ff * d_model = 2 * d_model * (4 * d_model)
        - SwiGLU: d_model * d_ff + d_model * d_ff + d_ff * d_model = 3 * d_model * d_ff
        - To match: d_ff ≈ (8/3) * d_model ≈ 2.67 * d_model

    Attributes:
        d_model (int): Model dimension (input and output dimension).
        d_ff (int): Hidden dimension for the feed-forward layer.
        beta (float): Parameter for Swish activation function.
        gate_proj (nn.Linear): Linear projection for gate component.
        value_proj (nn.Linear): Linear projection for value component.
        output_proj (nn.Linear): Output projection back to model dimension.

    Args:
        d_model (int): Model dimension (input and output dimension).
        d_ff (int, optional): Hidden dimension. If None, uses (8/3) * d_model
            for parameter efficiency matching traditional 4x expansion.
        beta (float, optional): Parameter for Swish activation. Default is 1.
        bias (bool, optional): Whether to use bias in linear layers. Default is True.

    Examples:
        >>> ffn = SwiGLU_FFN(d_model=512, d_ff=1365)
        >>> x = torch.randn(4, 10, 512)
        >>> output = ffn(x)
        >>> output.shape
        torch.Size([4, 10, 512])
        >>> # Check parameter count
        >>> param_count = sum(p.numel() for p in ffn.parameters())
    """

    def __init__(self, d_model: int, d_ff: int = None, beta: float = 1.0,
                 bias: bool = True):
        """Initialize SwiGLU FFN module."""
        super().__init__()

        # Default to (8/3) * d_model for parameter efficiency
        if d_ff is None:
            d_ff = int(8 / 3 * d_model)

        self.d_model = d_model
        self.d_ff = d_ff
        self.beta = beta

        # Three linear projections for SwiGLU
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.value_proj = nn.Linear(d_model, d_ff, bias=bias)
        self.output_proj = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SwiGLU FFN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
                or (..., d_model) for any number of leading dimensions.

        Returns:
            torch.Tensor: Output tensor of same shape as input (..., d_model).
        """
        # Gate component: Swish(xW + b) - controls what information passes through
        gate = swish(self.gate_proj(x), self.beta)

        # Value component: xV + c - the actual information to be gated
        value = self.value_proj(x)

        # Gated hidden representation: element-wise multiplication
        hidden = gate * value

        # Project back to model dimension
        output = self.output_proj(hidden)

        return output

    def count_parameters(self) -> int:
        """
        Count the number of trainable parameters in the module.

        Returns:
            int: Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Visualization and Analysis Functions
# ============================================================================

def plot_activation_comparison(save_dir: str = "artifacts/activations"):
    """
    Plot and compare ReLU and Swish activation functions and their gradients.

    This function creates a side-by-side comparison of:
    1. Activation functions: ReLU vs Swish
    2. Gradients: Shows the smoothness advantage of Swish

    Args:
        save_dir (str): Directory to save the plot. Default is "artifacts/activations".

    Returns:
        None. Saves plot to '{save_dir}/activation_comparison.png'.

    Notes:
        - Highlights key differences in gradient behavior
        - Shows Swish's smooth, continuous gradients vs ReLU's discontinuity
        - Demonstrates negative value handling
    """
    os.makedirs(save_dir, exist_ok=True)

    # Create input range
    x = torch.linspace(-5, 5, 1000)

    # Compute activations
    relu_vals = F.relu(x)
    swish_vals = swish(x)

    # Compute gradients
    x_grad = x.clone().requires_grad_(True)
    relu_out = F.relu(x_grad)
    relu_out.sum().backward()
    relu_grad = x_grad.grad.clone()

    x_grad = x.clone().requires_grad_(True)
    swish_out = swish(x_grad)
    swish_out.sum().backward()
    swish_grad = x_grad.grad.clone()

    # Convert to numpy for plotting
    x_np = x.detach().numpy()
    relu_vals_np = relu_vals.detach().numpy()
    swish_vals_np = swish_vals.detach().numpy()
    relu_grad_np = relu_grad.detach().numpy()
    swish_grad_np = swish_grad.detach().numpy()

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot activations
    ax1.plot(x_np, relu_vals_np, label='ReLU', linewidth=2, color='#ff6b6b')
    ax1.plot(x_np, swish_vals_np, label='Swish', linewidth=2, color='#48dbfb')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('Activation(x)', fontsize=12)
    ax1.set_title('Activation Functions', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot gradients
    ax2.plot(x_np, relu_grad_np, label='ReLU derivative', linewidth=2, color='#ff6b6b')
    ax2.plot(x_np, swish_grad_np, label='Swish derivative', linewidth=2, color='#48dbfb')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Gradient', fontsize=12)
    ax2.set_title('Activation Gradients', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'activation_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved activation comparison to {save_path}")
    plt.close()

    # Print observations
    print("\nKey observations:")
    print("- ReLU has a discontinuous gradient at x=0 (sudden jump)")
    print("- Swish has smooth, continuous gradients")
    print("- Swish allows small negative values to pass through (unlike ReLU)")


def plot_gating_mechanism(save_dir: str = "artifacts/activations"):
    """
    Visualize the gating mechanism in SwiGLU.

    This function demonstrates how SwiGLU's gating mechanism works by showing:
    1. Gate component (Swish-activated)
    2. Value component (linear)
    3. Gated output (element-wise product)
    4. Comparison with non-gated Swish

    Args:
        save_dir (str): Directory to save the plot. Default is "artifacts/activations".

    Returns:
        None. Saves plot to '{save_dir}/gating_mechanism.png'.

    Notes:
        - Uses histogram visualization to show value distributions
        - Demonstrates information modulation through gating
        - Compares gated vs non-gated outputs
    """
    os.makedirs(save_dir, exist_ok=True)

    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create sample input
    sample_input = torch.randn(100)

    # Create weight matrices for demonstration
    W_demo = torch.randn(100, 100) * 0.1
    V_demo = torch.randn(100, 100) * 0.1

    # Without gating (traditional Swish)
    swish_output = swish(sample_input @ W_demo)

    # With gating (SwiGLU)
    gate_part = swish_output  # GATE: Swish-activated component
    linear_part = sample_input @ V_demo  # VALUE: Linear component
    swiglu_output = gate_part * linear_part  # GATED OUTPUT

    # Convert to numpy for plotting
    gate_np = gate_part.detach().numpy()
    linear_np = linear_part.detach().numpy()
    swiglu_np = swiglu_output.detach().numpy()
    swish_np = swish_output.detach().numpy()

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot gate component
    axes[0, 0].hist(gate_np, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Gate Component: Swish(xW)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot linear component
    axes[0, 1].hist(linear_np, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Value Component: xV', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot gated output
    axes[1, 0].hist(swiglu_np, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_title('Gated Output: Swish(xW) ⊗ xV', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot comparison
    axes[1, 1].hist(swish_np, bins=30, alpha=0.5, color='blue',
                    label='Swish', edgecolor='black')
    axes[1, 1].hist(swiglu_np, bins=30, alpha=0.5, color='red',
                    label='SwiGLU', edgecolor='black')
    axes[1, 1].set_title('Output Comparison', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'gating_mechanism.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved gating mechanism visualization to {save_path}")
    plt.close()

    print("\nGating allows the network to:")
    print("- Modulate information flow based on context")
    print("- Learn which features to emphasize or suppress")
    print("- Create more expressive representations")


def demonstrate_usage(save_dir: str = "artifacts/activations"):
    """
    Demonstrate basic usage of SwiGLU_FFN module.

    This function shows how to:
    1. Instantiate the SwiGLU_FFN module
    2. Process input tensors
    3. Compare with traditional FFN
    4. Display parameter counts

    Args:
        save_dir (str): Directory for any output files. Default is "artifacts/activations".

    Returns:
        None. Prints usage information and parameter statistics.

    Examples:
        >>> demonstrate_usage()
        Creating SwiGLU FFN...
        Input shape: torch.Size([4, 10, 512])
        Output shape: torch.Size([4, 10, 512])
        ...
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*70)
    print("SwiGLU FFN Usage Demonstration")
    print("="*70)

    # Configuration
    d_model = 512
    d_ff = int(8 / 3 * d_model)  # ~1365, commonly used ratio for SwiGLU
    batch_size = 4
    seq_len = 10

    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\nConfiguration:")
    print(f"  d_model: {d_model}")
    print(f"  d_ff: {d_ff}")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")

    # Initialize and run SwiGLU FFN
    print(f"\nCreating SwiGLU FFN...")
    swiglu_ffn = SwiGLU_FFN(d_model, d_ff)
    output = swiglu_ffn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"SwiGLU parameters: {swiglu_ffn.count_parameters():,}")

    # Compare with Traditional FFN
    print(f"\nCreating Traditional FFN for comparison...")
    traditional_ffn = TraditionalFFN(d_model, d_ff=4*d_model)
    output_traditional = traditional_ffn(x)

    print(f"Traditional FFN output shape: {output_traditional.shape}")
    print(f"Traditional FFN parameters: {sum(p.numel() for p in traditional_ffn.parameters()):,}")

    # Parameter comparison
    print(f"\n" + "-"*70)
    print("Parameter Count Comparison:")
    print("-"*70)
    swiglu_params = swiglu_ffn.count_parameters()
    traditional_params = sum(p.numel() for p in traditional_ffn.parameters())
    print(f"SwiGLU FFN (d_ff={d_ff}):           {swiglu_params:>10,} params")
    print(f"Traditional FFN (d_ff={4*d_model}): {traditional_params:>10,} params")
    print(f"Difference:                           {abs(swiglu_params - traditional_params):>10,} params")
    print(f"Ratio (SwiGLU/Traditional):           {swiglu_params/traditional_params:>10.2f}x")

    print("\n" + "="*70 + "\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    """
    Main execution block for demonstrations and visualizations.

    This section runs when the module is executed directly (not imported).
    It performs the following:
    1. Creates output directory for artifacts
    2. Demonstrates module usage with example inputs
    3. Generates activation function comparison plots
    4. Visualizes gating mechanism

    Output:
        - Plots saved to 'artifacts/activations/' directory
        - Console output with usage examples and statistics
    """
    print("\n" + "="*70)
    print("SwiGLU Feed-Forward Network - Educational Demonstration")
    print("="*70 + "\n")

    # Set up save directory
    save_dir = "artifacts/images/activations"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}\n")

    # Run demonstrations
    print("Running usage demonstration...")
    demonstrate_usage(save_dir)

    print("Generating activation comparison plot...")
    plot_activation_comparison(save_dir)

    print("\nGenerating gating mechanism visualization...")
    plot_gating_mechanism(save_dir)

    print("\n" + "="*70)
    print("All demonstrations complete!")
    print(f"Check '{save_dir}' for generated plots.")
    print("="*70 + "\n")

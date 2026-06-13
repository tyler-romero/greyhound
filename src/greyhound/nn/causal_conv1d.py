import torch
import torch.nn as nn

from greyhound.nn.functional import causal_conv1d


class GreyhoundCausalConv1d(nn.Conv1d):
    """
    Causal 1D depthwise convolution with optional SiLU activation.

    Wraps the fused Greyhound kernel as a drop-in replacement for depthwise
    ``nn.Conv1d`` with causal (left) padding. The kernel handles boundary
    conditions internally via masked loads, avoiding the cost of explicit
    ``F.pad()``.

    Inherits from ``nn.Conv1d`` with ``groups=channels`` (depthwise),
    ``stride=1``, ``dilation=1``, and ``padding=0`` (causal padding is
    applied inside the kernel).

    Args:
        channels: Number of input (and output) channels.
        kernel_size: Width of the convolution kernel (typically 2-4).
        bias: If ``True``, adds a learnable bias. Default: ``True``.
        activation: Activation to apply after convolution. ``"silu"`` for
            SiLU activation, or ``None`` for no activation. Default: ``None``.
        device: Device for parameter initialization.
        dtype: Data type for parameters.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        bias: bool = True,
        activation: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=channels,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )
        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply causal depthwise convolution.

        Args:
            input: Input tensor of shape ``[B, D, T]``.

        Returns:
            Output tensor of shape ``[B, D, T]``.
        """
        # nn.Conv1d weight is [D, 1, W] for depthwise; kernel expects [D, W]
        return causal_conv1d(input, self.weight.squeeze(1), self.bias, activation=self.activation)

    def extra_repr(self) -> str:
        s = super().extra_repr()
        if self.activation is not None:
            s += f", activation={self.activation}"
        return s

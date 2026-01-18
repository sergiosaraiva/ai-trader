"""Base architecture class and registry for plugin-based model system.

The architecture system separates neural network structure from training logic,
allowing any architecture to be used at any timeframe with any configuration.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class ArchitectureConfig:
    """Base configuration for architectures.

    Attributes:
        input_dim: Number of input features.
        sequence_length: Input sequence length.
        output_dim: Number of output dimensions.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of layers.
        dropout: Dropout rate.
        use_batch_norm: Whether to use batch normalization.
        activation: Activation function name.
        output_type: Type of output ('regression', 'classification', 'multi').
        num_classes: Number of classes for classification.
        prediction_horizons: List of prediction horizons.
    """

    input_dim: int = 50
    sequence_length: int = 100
    output_dim: int = 1
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    use_batch_norm: bool = True
    activation: str = "relu"
    output_type: str = "regression"
    num_classes: int = 3
    prediction_horizons: List[int] = field(default_factory=lambda: [1])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation,
            "output_type": self.output_type,
            "num_classes": self.num_classes,
            "prediction_horizons": self.prediction_horizons,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArchitectureConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class BaseArchitecture(nn.Module, ABC):
    """Base class for all neural network architectures.

    This is the foundation for the plugin-based architecture system.
    Architectures define only the neural network structure, not training logic.

    Subclasses must implement:
        - forward(): Forward pass
        - get_output_info(): Information about outputs

    Example:
        ```python
        class MyArchitecture(BaseArchitecture):
            def __init__(self, config: ArchitectureConfig):
                super().__init__(config)
                self.layers = nn.Sequential(...)

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                return {"prediction": self.layers(x)}

            def get_output_info(self) -> Dict[str, Tuple[str, Tuple]]:
                return {"prediction": ("regression", (self.config.output_dim,))}
        ```
    """

    # Architecture metadata
    name: str = "base"
    description: str = "Base architecture"
    supported_output_types: List[str] = ["regression", "classification", "multi"]

    def __init__(self, config: Optional[Union[ArchitectureConfig, Dict]] = None):
        """Initialize architecture.

        Args:
            config: Architecture configuration.
        """
        super().__init__()

        if config is None:
            config = ArchitectureConfig()
        elif isinstance(config, dict):
            config = ArchitectureConfig.from_dict(config)

        self.config = config
        self._built = False

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence_length, input_dim).

        Returns:
            Dictionary of output tensors with named keys.
        """
        pass

    @abstractmethod
    def get_output_info(self) -> Dict[str, Tuple[str, Tuple]]:
        """Get information about architecture outputs.

        Returns:
            Dictionary mapping output names to (type, shape) tuples.
            Types: 'regression', 'classification', 'probability'
        """
        pass

    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """Get architecture summary string."""
        lines = [
            f"Architecture: {self.name}",
            f"Description: {self.description}",
            f"Total parameters: {self.get_num_parameters():,}",
            f"Trainable parameters: {self.get_trainable_parameters():,}",
            f"Input: (batch, {self.config.sequence_length}, {self.config.input_dim})",
            "Outputs:",
        ]

        for name, (out_type, shape) in self.get_output_info().items():
            lines.append(f"  - {name}: {out_type}, shape {shape}")

        return "\n".join(lines)

    def _get_activation(self) -> nn.Module:
        """Get activation function from config."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "elu": nn.ELU(),
            "selu": nn.SELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
            "swish": nn.SiLU(),
        }
        return activations.get(self.config.activation.lower(), nn.ReLU())


class ArchitectureRegistry:
    """Registry for managing architecture classes.

    Provides plugin-based architecture loading and creation.

    Example:
        ```python
        # Register an architecture
        ArchitectureRegistry.register("my_arch", MyArchitecture)

        # Create from registry
        model = ArchitectureRegistry.create("my_arch", input_dim=50)

        # List available
        print(ArchitectureRegistry.available())
        ```
    """

    _architectures: Dict[str, Type[BaseArchitecture]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        architecture_class: Type[BaseArchitecture],
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register an architecture class.

        Args:
            name: Primary name for the architecture.
            architecture_class: The architecture class.
            aliases: Optional list of alternative names.
        """
        name_lower = name.lower()
        cls._architectures[name_lower] = architecture_class

        # Register aliases
        if aliases:
            for alias in aliases:
                cls._architectures[alias.lower()] = architecture_class

        logger.debug(f"Registered architecture: {name}")

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister an architecture.

        Args:
            name: Architecture name to remove.
        """
        name_lower = name.lower()
        if name_lower in cls._architectures:
            del cls._architectures[name_lower]

    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[Union[ArchitectureConfig, Dict]] = None,
        **kwargs,
    ) -> BaseArchitecture:
        """Create an architecture instance.

        Args:
            name: Architecture name.
            config: Configuration object or dict.
            **kwargs: Additional config parameters.

        Returns:
            Architecture instance.

        Raises:
            ValueError: If architecture not found.
        """
        name_lower = name.lower()

        if name_lower not in cls._architectures:
            available = ", ".join(cls._architectures.keys())
            raise ValueError(
                f"Unknown architecture: {name}. Available: {available}"
            )

        architecture_class = cls._architectures[name_lower]

        # Merge config with kwargs - always pass as dict to let architecture handle
        # its own config conversion (specialized configs like CNNLSTMConfig)
        if config is None:
            config = kwargs  # Pass as dict
        elif isinstance(config, dict):
            config.update(kwargs)
            # Keep as dict
        elif isinstance(config, ArchitectureConfig):
            # Convert to dict and merge with kwargs
            config_dict = config.to_dict()
            config_dict.update(kwargs)
            config = config_dict
        else:
            config = kwargs

        return architecture_class(config)

    @classmethod
    def available(cls) -> List[str]:
        """Get list of available architecture names."""
        return list(set(cls._architectures.keys()))

    @classmethod
    def get_class(cls, name: str) -> Type[BaseArchitecture]:
        """Get architecture class by name.

        Args:
            name: Architecture name.

        Returns:
            Architecture class.
        """
        name_lower = name.lower()
        if name_lower not in cls._architectures:
            raise ValueError(f"Unknown architecture: {name}")
        return cls._architectures[name_lower]

    @classmethod
    def get_info(cls, name: str) -> Dict[str, Any]:
        """Get information about an architecture.

        Args:
            name: Architecture name.

        Returns:
            Dictionary with architecture info.
        """
        arch_class = cls.get_class(name)
        return {
            "name": arch_class.name,
            "description": arch_class.description,
            "supported_output_types": arch_class.supported_output_types,
            "class": arch_class.__name__,
        }


# Helper modules used by architectures


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer architectures."""

    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network for TFT-style architectures."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        context_dim: Optional[int] = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        # Skip connection
        if input_dim != output_dim:
            self.skip_layer = nn.Linear(input_dim, output_dim)
        else:
            self.skip_layer = None

        # Main pathway
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.elu = nn.ELU()

        if context_dim is not None:
            self.context_fc = nn.Linear(context_dim, hidden_dim, bias=False)

        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

        # GLU gate
        self.gate = nn.Linear(output_dim, output_dim * 2)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional context."""
        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x

        # Main pathway
        hidden = self.fc1(x)

        if context is not None and self.context_dim is not None:
            hidden = hidden + self.context_fc(context)

        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # GLU gate
        gate_input = self.gate(hidden)
        gate, hidden = gate_input.chunk(2, dim=-1)
        hidden = torch.sigmoid(gate) * hidden

        # Residual + LayerNorm
        output = self.layer_norm(skip + hidden)
        return output


class MultiHeadAttention(nn.Module):
    """Multi-head attention module."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (output, attention_weights).
        """
        batch_size = query.size(0)

        # Linear projections
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_linear(context)

        return output, attn_weights

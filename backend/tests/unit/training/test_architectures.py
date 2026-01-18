"""Unit tests for architecture system."""

import pytest
import torch
import torch.nn as nn

from src.training.architectures import (
    ArchitectureConfig,
    ArchitectureRegistry,
    BaseArchitecture,
    CNNLSTMAttention,
    GatedResidualNetwork,
    MultiHeadAttention,
    NBEATSTransformer,
    PositionalEncoding,
    TemporalFusionTransformer,
)


class TestArchitectureConfig:
    """Tests for ArchitectureConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ArchitectureConfig()
        assert config.input_dim == 50
        assert config.sequence_length == 100
        assert config.output_dim == 1
        assert config.hidden_dim == 256
        assert config.num_layers == 2
        assert config.dropout == 0.3
        assert config.use_batch_norm is True
        assert config.activation == "relu"
        assert config.output_type == "regression"
        assert config.num_classes == 3
        assert config.prediction_horizons == [1]

    def test_custom_values(self):
        """Test custom configuration."""
        config = ArchitectureConfig(
            input_dim=100,
            sequence_length=168,
            hidden_dim=512,
            num_layers=4,
            dropout=0.5,
            prediction_horizons=[1, 4, 12, 24],
        )
        assert config.input_dim == 100
        assert config.sequence_length == 168
        assert config.hidden_dim == 512
        assert config.num_layers == 4
        assert config.dropout == 0.5
        assert config.prediction_horizons == [1, 4, 12, 24]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ArchitectureConfig()
        d = config.to_dict()
        assert d["input_dim"] == 50
        assert d["hidden_dim"] == 256
        assert "prediction_horizons" in d

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "input_dim": 100,
            "hidden_dim": 512,
            "extra_field": "ignored",
        }
        config = ArchitectureConfig.from_dict(data)
        assert config.input_dim == 100
        assert config.hidden_dim == 512


class TestArchitectureRegistry:
    """Tests for ArchitectureRegistry."""

    def test_architectures_are_registered(self):
        """Test that built-in architectures are registered."""
        available = ArchitectureRegistry.available()
        assert "cnn_lstm_attention" in available
        assert "temporal_fusion_transformer" in available
        assert "nbeats_transformer" in available

    def test_aliases_work(self):
        """Test that aliases resolve to architectures."""
        available = ArchitectureRegistry.available()
        assert "cnn_lstm" in available
        assert "tft" in available
        assert "nbeats" in available

    def test_create_architecture(self):
        """Test creating architecture from registry."""
        model = ArchitectureRegistry.create(
            "cnn_lstm_attention",
            input_dim=50,
            sequence_length=100,
        )
        assert isinstance(model, CNNLSTMAttention)
        assert model.config.input_dim == 50

    def test_create_with_config(self):
        """Test creating with config object."""
        config = ArchitectureConfig(input_dim=100, hidden_dim=512)
        model = ArchitectureRegistry.create("tft", config=config)
        assert model.config.input_dim == 100
        assert model.config.hidden_dim == 512

    def test_create_unknown_raises(self):
        """Test that unknown architecture raises error."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            ArchitectureRegistry.create("nonexistent_arch")

    def test_get_class(self):
        """Test getting architecture class."""
        cls = ArchitectureRegistry.get_class("cnn_lstm_attention")
        assert cls == CNNLSTMAttention

    def test_get_info(self):
        """Test getting architecture info."""
        info = ArchitectureRegistry.get_info("cnn_lstm_attention")
        assert "name" in info
        assert "description" in info
        assert "supported_output_types" in info

    def test_case_insensitive(self):
        """Test case-insensitive lookup."""
        model1 = ArchitectureRegistry.create("CNN_LSTM_Attention", input_dim=50)
        model2 = ArchitectureRegistry.create("cnn_lstm_attention", input_dim=50)
        assert type(model1) == type(model2)


class TestPositionalEncoding:
    """Tests for PositionalEncoding module."""

    def test_output_shape(self):
        """Test output shape."""
        pe = PositionalEncoding(d_model=256, max_len=100)
        x = torch.randn(8, 50, 256)  # batch, seq, dim
        output = pe(x)
        assert output.shape == x.shape

    def test_positional_values_differ(self):
        """Test that different positions have different encodings."""
        pe = PositionalEncoding(d_model=256, max_len=100)
        x = torch.zeros(1, 50, 256)
        output = pe(x)
        # Different positions should have different encodings
        assert not torch.allclose(output[0, 0], output[0, 1])

    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        pe = PositionalEncoding(d_model=256, dropout=0.5)
        pe.train()
        x = torch.ones(8, 50, 256)
        outputs = [pe(x) for _ in range(5)]
        # Outputs should differ due to dropout
        assert not all(torch.allclose(outputs[0], o) for o in outputs[1:])


class TestGatedResidualNetwork:
    """Tests for GatedResidualNetwork module."""

    def test_same_dim_output(self):
        """Test output when input_dim == output_dim."""
        grn = GatedResidualNetwork(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
        )
        x = torch.randn(8, 50, 256)
        output = grn(x)
        assert output.shape == (8, 50, 256)

    def test_different_dim_output(self):
        """Test output when input_dim != output_dim."""
        grn = GatedResidualNetwork(
            input_dim=256,
            hidden_dim=512,
            output_dim=128,
        )
        x = torch.randn(8, 50, 256)
        output = grn(x)
        assert output.shape == (8, 50, 128)

    def test_with_context(self):
        """Test with context vector."""
        grn = GatedResidualNetwork(
            input_dim=256,
            hidden_dim=512,
            output_dim=256,
            context_dim=64,
        )
        x = torch.randn(8, 50, 256)
        context = torch.randn(8, 50, 64)
        output = grn(x, context)
        assert output.shape == (8, 50, 256)


class TestMultiHeadAttention:
    """Tests for MultiHeadAttention module."""

    def test_output_shape(self):
        """Test output shape."""
        mha = MultiHeadAttention(d_model=256, num_heads=8)
        q = k = v = torch.randn(8, 50, 256)
        output, weights = mha(q, k, v)
        assert output.shape == (8, 50, 256)
        assert weights.shape == (8, 8, 50, 50)

    def test_self_attention(self):
        """Test self-attention (q=k=v)."""
        mha = MultiHeadAttention(d_model=256, num_heads=4)
        x = torch.randn(8, 50, 256)
        output, weights = mha(x, x, x)
        assert output.shape == x.shape

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to approximately 1."""
        mha = MultiHeadAttention(d_model=256, num_heads=4)
        mha.eval()  # Disable dropout for deterministic results
        x = torch.randn(4, 20, 256)
        with torch.no_grad():
            _, weights = mha(x, x, x)
        # Sum over last dimension (attention targets)
        sums = weights.sum(dim=-1)
        # Attention weights should sum close to 1 (with tolerance for floating point)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


class TestCNNLSTMAttention:
    """Tests for CNNLSTMAttention architecture."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        # Use dict config to let the architecture create its specialized config
        return CNNLSTMAttention({
            "input_dim": 50,
            "sequence_length": 100,
            "hidden_dim": 64,  # Smaller for tests
            "num_layers": 1,
            "prediction_horizons": [1, 4],
        })

    def test_initialization(self, model):
        """Test model initialization."""
        assert isinstance(model, BaseArchitecture)
        assert model.config.input_dim == 50

    def test_forward_pass(self, model):
        """Test forward pass."""
        x = torch.randn(8, 100, 50)  # batch, seq, features
        outputs = model(x)
        assert isinstance(outputs, dict)
        assert "price" in outputs
        assert "direction_logits" in outputs
        assert "alpha" in outputs
        assert "beta" in outputs

    def test_output_shapes(self, model):
        """Test output shapes."""
        x = torch.randn(8, 100, 50)
        outputs = model(x)
        num_horizons = len(model.config.prediction_horizons)
        assert outputs["price"].shape == (8, num_horizons)
        assert outputs["direction_logits"].shape == (8, num_horizons, 3)
        assert outputs["alpha"].shape == (8, num_horizons)
        assert outputs["beta"].shape == (8, num_horizons)

    def test_get_output_info(self, model):
        """Test get_output_info."""
        info = model.get_output_info()
        assert "price" in info
        assert info["price"][0] == "regression"

    def test_get_num_parameters(self, model):
        """Test parameter counting."""
        num_params = model.get_num_parameters()
        assert num_params > 0
        assert isinstance(num_params, int)

    def test_summary(self, model):
        """Test summary string."""
        summary = model.summary()
        assert "Architecture" in summary
        assert "parameters" in summary.lower()


class TestTemporalFusionTransformer:
    """Tests for TemporalFusionTransformer architecture."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        # Use dict config to let the architecture create its specialized config
        return TemporalFusionTransformer({
            "input_dim": 50,
            "sequence_length": 90,
            "hidden_dim": 64,
            "num_layers": 1,
            "prediction_horizons": [1, 3, 5],
        })

    def test_initialization(self, model):
        """Test model initialization."""
        assert isinstance(model, BaseArchitecture)

    def test_forward_pass(self, model):
        """Test forward pass."""
        x = torch.randn(4, 90, 50)
        outputs = model(x)
        assert isinstance(outputs, dict)
        assert "price" in outputs
        assert "quantiles" in outputs
        assert "direction_logits" in outputs
        assert "attention_weights" in outputs

    def test_quantile_outputs(self, model):
        """Test quantile prediction outputs."""
        x = torch.randn(4, 90, 50)
        outputs = model(x)
        num_horizons = len(model.config.prediction_horizons)
        num_quantiles = len(model.config.quantiles)
        assert outputs["quantiles"].shape == (4, num_horizons, num_quantiles)

    def test_attention_weights(self, model):
        """Test attention weight output."""
        x = torch.randn(4, 90, 50)
        outputs = model(x)
        attn = outputs["attention_weights"]
        assert attn.shape[0] == 4  # batch size
        # Weights should sum approximately to 1 (with tolerance for numerical precision)
        sums = attn.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=0.1)


class TestNBEATSTransformer:
    """Tests for NBEATSTransformer architecture."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        # Use dict config to let the architecture create its specialized config
        return NBEATSTransformer({
            "input_dim": 50,
            "sequence_length": 52,
            "hidden_dim": 64,
            "num_layers": 1,
            "prediction_horizons": [1, 2, 4],
        })

    def test_initialization(self, model):
        """Test model initialization."""
        assert isinstance(model, BaseArchitecture)
        assert model.name == "nbeats_transformer"

    def test_forward_pass(self, model):
        """Test forward pass."""
        x = torch.randn(4, 52, 50)
        outputs = model(x)
        assert isinstance(outputs, dict)
        assert "price" in outputs
        assert "nbeats_forecast" in outputs
        assert "regime_logits" in outputs
        assert "trend_strength" in outputs

    def test_output_shapes(self, model):
        """Test output shapes."""
        x = torch.randn(4, 52, 50)
        outputs = model(x)
        num_horizons = len(model.config.prediction_horizons)
        assert outputs["price"].shape == (4, num_horizons)
        assert outputs["nbeats_forecast"].shape == (4, num_horizons)
        # Regime classification has 4 classes by default
        assert outputs["regime_logits"].shape[0] == 4

    def test_trend_strength_range(self, model):
        """Test trend strength is in [0, 1]."""
        x = torch.randn(8, 52, 50)
        outputs = model(x)
        trend = outputs["trend_strength"]
        assert (trend >= 0).all() and (trend <= 1).all()


class TestArchitectureIntegration:
    """Integration tests for architectures."""

    @pytest.mark.parametrize("arch_name", [
        "cnn_lstm_attention",
        "temporal_fusion_transformer",
        "nbeats_transformer",
    ])
    def test_all_architectures_work(self, arch_name):
        """Test all registered architectures can be created and run."""
        model = ArchitectureRegistry.create(
            arch_name,
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
            num_layers=1,
            prediction_horizons=[1, 2],
        )

        # Forward pass
        x = torch.randn(2, 50, 30)
        outputs = model(x)

        assert isinstance(outputs, dict)
        assert "price" in outputs
        assert "direction_logits" in outputs
        assert "alpha" in outputs
        assert "beta" in outputs

    @pytest.mark.parametrize("arch_name", [
        "cnn_lstm",  # alias
        "tft",  # alias
        "nbeats",  # alias
    ])
    def test_aliases_work(self, arch_name):
        """Test architecture aliases."""
        model = ArchitectureRegistry.create(
            arch_name,
            input_dim=30,
            sequence_length=50,
        )
        assert isinstance(model, BaseArchitecture)

    def test_gradient_flow(self):
        """Test that gradients flow through model."""
        model = ArchitectureRegistry.create(
            "cnn_lstm_attention",
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
        )

        x = torch.randn(2, 50, 30, requires_grad=True)
        outputs = model(x)
        loss = outputs["price"].mean()
        loss.backward()

        assert x.grad is not None
        # Check that gradients flow to CNN layers
        assert model.cnn[0].conv.weight.grad is not None

    def test_model_can_be_saved_and_loaded(self):
        """Test model state dict save/load."""
        model1 = ArchitectureRegistry.create(
            "cnn_lstm_attention",
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
        )

        # Get state dict
        state = model1.state_dict()

        # Create new model and load state
        model2 = ArchitectureRegistry.create(
            "cnn_lstm_attention",
            input_dim=30,
            sequence_length=50,
            hidden_dim=32,
        )
        model2.load_state_dict(state)

        # Set both to eval mode for deterministic output
        model1.eval()
        model2.eval()

        # Verify they produce same output
        x = torch.randn(2, 50, 30)
        with torch.no_grad():
            out1 = model1(x)
            out2 = model2(x)

        assert torch.allclose(out1["price"], out2["price"], rtol=1e-5)

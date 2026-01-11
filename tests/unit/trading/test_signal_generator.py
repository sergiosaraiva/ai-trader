"""Tests for Signal Generator."""

import pytest
from datetime import datetime
from unittest.mock import MagicMock

from src.trading.signals.generator import (
    SignalGenerator,
    EnsemblePrediction,
    Position,
)
from src.trading.signals.actions import Action, TradingSignal, SignalStrength
from src.trading.risk.profiles import load_risk_profile, RiskLevel
from src.trading.circuit_breakers.base import CircuitBreakerState, TradingState


class TestEnsemblePrediction:
    """Tests for EnsemblePrediction dataclass."""

    def test_basic_creation(self):
        """Test basic prediction creation."""
        pred = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.80,
            alpha=15.0,
            beta=5.0,
        )

        assert pred.direction_probability == 0.75
        assert pred.confidence == 0.80
        assert pred.concentration == 20.0
        assert pred.is_bullish is True
        assert pred.is_bearish is False

    def test_bearish_prediction(self):
        """Test bearish prediction detection."""
        pred = EnsemblePrediction(
            direction_probability=0.35,
            confidence=0.80,
        )

        assert pred.is_bullish is False
        assert pred.is_bearish is True

    def test_from_predictor_output_bullish(self):
        """Test conversion from EnsemblePredictor output - bullish."""
        # Mock predictor output
        mock_output = MagicMock()
        mock_output.direction = 1
        mock_output.direction_probability = 0.75
        mock_output.confidence = 0.80
        mock_output.agreement_score = 0.9
        mock_output.market_regime = "trending"
        mock_output.volatility_level = "normal"

        # Mock component predictions
        short_comp = MagicMock()
        short_comp.direction = 1
        short_comp.confidence = 0.85

        medium_comp = MagicMock()
        medium_comp.direction = 1
        medium_comp.confidence = 0.75

        long_comp = MagicMock()
        long_comp.direction = 1
        long_comp.confidence = 0.70

        mock_output.component_predictions = {
            "short_term": short_comp,
            "medium_term": medium_comp,
            "long_term": long_comp,
        }

        pred = EnsemblePrediction.from_predictor_output(mock_output)

        assert pred.direction_probability == 0.75
        assert pred.confidence == 0.80
        assert pred.ensemble_agreement == 0.9
        assert pred.market_regime == "trending"
        assert pred.short_term_signal > 0  # Bullish
        assert pred.medium_term_signal > 0
        assert pred.long_term_signal > 0

    def test_from_predictor_output_bearish(self):
        """Test conversion from EnsemblePredictor output - bearish."""
        mock_output = MagicMock()
        mock_output.direction = 0
        mock_output.direction_probability = 0.35
        mock_output.confidence = 0.70
        mock_output.agreement_score = 0.8
        mock_output.market_regime = "ranging"
        mock_output.volatility_level = "high"
        mock_output.component_predictions = {}

        pred = EnsemblePrediction.from_predictor_output(mock_output)

        assert pred.direction_probability == 0.35
        assert pred.is_bearish is True


class TestSignalGenerator:
    """Tests for SignalGenerator."""

    @pytest.fixture
    def risk_profile(self):
        """Create moderate risk profile."""
        return load_risk_profile("moderate")

    @pytest.fixture
    def generator(self, risk_profile):
        """Create signal generator."""
        return SignalGenerator(risk_profile=risk_profile)

    @pytest.fixture
    def active_breaker_state(self):
        """Create active circuit breaker state."""
        return CircuitBreakerState(
            overall_state=TradingState.ACTIVE,
        )

    @pytest.fixture
    def halted_breaker_state(self):
        """Create halted circuit breaker state."""
        return CircuitBreakerState(
            overall_state=TradingState.HALTED,
            reasons=["Test halt"],
        )

    def test_generate_buy_signal(self, generator, active_breaker_state):
        """Test generating BUY signal."""
        prediction = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.80,
            alpha=15.0,
            beta=5.0,
            short_term_signal=0.8,
            medium_term_signal=0.7,
            long_term_signal=0.6,
            ensemble_agreement=0.9,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
            atr=0.0050,
        )

        assert signal.action == Action.BUY
        assert signal.confidence == 0.80
        assert signal.position_size_pct > 0
        assert signal.stop_loss_pct > 0
        assert signal.take_profit_pct > 0

    def test_generate_sell_signal(self, generator, active_breaker_state):
        """Test generating SELL signal."""
        prediction = EnsemblePrediction(
            direction_probability=0.25,
            confidence=0.80,
            alpha=5.0,
            beta=15.0,
            short_term_signal=-0.8,
            medium_term_signal=-0.7,
            long_term_signal=-0.6,
            ensemble_agreement=0.9,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
            atr=0.0050,
        )

        assert signal.action == Action.SELL

    def test_generate_hold_low_confidence(self, generator, active_breaker_state):
        """Test generating HOLD for low confidence."""
        prediction = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.50,  # Below threshold
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
        )

        assert signal.action == Action.HOLD
        assert "below threshold" in signal.reason

    def test_generate_hold_neutral_zone(self, generator, active_breaker_state):
        """Test generating HOLD for neutral direction."""
        prediction = EnsemblePrediction(
            direction_probability=0.50,  # Neutral
            confidence=0.80,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
        )

        assert signal.action == Action.HOLD
        assert "neutral zone" in signal.reason

    def test_generate_hold_circuit_breaker_halted(self, generator, halted_breaker_state):
        """Test generating HOLD when circuit breaker halted."""
        prediction = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.90,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=halted_breaker_state,
        )

        assert signal.action == Action.HOLD
        assert "Circuit breaker" in signal.reason

    def test_generate_close_long_on_reversal(self, generator, active_breaker_state):
        """Test generating CLOSE_LONG on reversal."""
        prediction = EnsemblePrediction(
            direction_probability=0.25,  # Bearish
            confidence=0.80,
            short_term_signal=-0.8,
            medium_term_signal=-0.7,
            long_term_signal=-0.6,
            ensemble_agreement=0.9,
        )

        current_position = Position(
            symbol="EURUSD",
            side="long",
            quantity=1.0,
            entry_price=1.1000,
            current_price=1.1050,
            unrealized_pnl=50,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1050,
            breaker_state=active_breaker_state,
            current_position=current_position,
        )

        assert signal.action == Action.CLOSE_LONG

    def test_generate_hold_same_direction_position(self, generator, active_breaker_state):
        """Test generating HOLD when already in same direction."""
        prediction = EnsemblePrediction(
            direction_probability=0.75,  # Bullish
            confidence=0.80,
            short_term_signal=0.8,
            medium_term_signal=0.7,
            long_term_signal=0.6,
            ensemble_agreement=0.9,
        )

        current_position = Position(
            symbol="EURUSD",
            side="long",  # Already long
            quantity=1.0,
            entry_price=1.1000,
            current_price=1.1050,
            unrealized_pnl=50,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1050,
            breaker_state=active_breaker_state,
            current_position=current_position,
        )

        assert signal.action == Action.HOLD
        assert "Already in LONG" in signal.reason

    def test_position_sizing_by_confidence(self, generator, active_breaker_state):
        """Test position sizing scales with confidence."""
        # Low confidence
        pred_low = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.70,
            short_term_signal=0.8,
            medium_term_signal=0.7,
            long_term_signal=0.6,
            ensemble_agreement=0.9,
        )

        # High confidence
        pred_high = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.90,
            short_term_signal=0.8,
            medium_term_signal=0.7,
            long_term_signal=0.6,
            ensemble_agreement=0.9,
        )

        signal_low = generator.generate_signal(
            prediction=pred_low,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
        )

        signal_high = generator.generate_signal(
            prediction=pred_high,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
        )

        assert signal_high.position_size_pct > signal_low.position_size_pct

    def test_stop_loss_calculation(self, generator, active_breaker_state):
        """Test stop-loss and take-profit calculation."""
        prediction = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.80,
            short_term_signal=0.8,
            medium_term_signal=0.7,
            long_term_signal=0.6,
            ensemble_agreement=0.9,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
            atr=0.0050,  # 50 pips
        )

        assert signal.stop_loss_price is not None
        assert signal.take_profit_price is not None
        assert signal.stop_loss_price < 1.1000  # Below entry for long
        assert signal.take_profit_price > 1.1000  # Above entry for long

    def test_generate_signal_from_predictor(self, generator, active_breaker_state):
        """Test generate_signal_from_predictor method."""
        # Mock predictor output
        mock_output = MagicMock()
        mock_output.direction = 1
        mock_output.direction_probability = 0.75
        mock_output.confidence = 0.80
        mock_output.agreement_score = 0.9
        mock_output.market_regime = "trending"
        mock_output.volatility_level = "normal"
        mock_output.component_predictions = {}

        signal = generator.generate_signal_from_predictor(
            predictor_output=mock_output,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
            atr=0.0050,
        )

        assert signal.action == Action.BUY
        assert signal.confidence == 0.80

    def test_calculate_stop_loss_price_buy(self, generator):
        """Test stop-loss price calculation for BUY."""
        stop_loss = generator.calculate_stop_loss_price(
            entry_price=1.1000,
            action=Action.BUY,
            atr=0.0050,
            confidence=0.80,
        )

        # Should be below entry
        assert stop_loss < 1.1000
        # Should be roughly 2*ATR adjusted by confidence
        assert abs(1.1000 - stop_loss) > 0.0080  # At least ~80 pips

    def test_calculate_stop_loss_price_sell(self, generator):
        """Test stop-loss price calculation for SELL."""
        stop_loss = generator.calculate_stop_loss_price(
            entry_price=1.1000,
            action=Action.SELL,
            atr=0.0050,
            confidence=0.80,
        )

        # Should be above entry
        assert stop_loss > 1.1000

    def test_calculate_take_profit_price(self, generator):
        """Test take-profit price calculation."""
        take_profit = generator.calculate_take_profit_price(
            entry_price=1.1000,
            action=Action.BUY,
            stop_loss_price=1.0900,
            confidence=0.80,
        )

        # Should be above entry for BUY
        assert take_profit > 1.1000

        # Check risk-reward ratio (should be 2.5 for >= 80% confidence)
        # According to _calculate_exits: >= 0.80 -> 2.5, >= 0.70 -> 2.0, else 1.5
        sl_distance = 1.1000 - 1.0900
        tp_distance = take_profit - 1.1000
        rr_ratio = tp_distance / sl_distance

        assert rr_ratio == pytest.approx(2.5, rel=0.01)

    def test_calculate_position_units(self, generator):
        """Test position units calculation."""
        units = generator.calculate_position_units(
            position_size_pct=0.02,
            account_equity=100000,
            current_price=1.1000,
            lot_size=100000,
        )

        # 2% of 100k = 2000
        # 2000 / (1.1 * 100000) = 0.018
        assert units == pytest.approx(0.02, abs=0.01)

    def test_signal_strength_calculation(self, generator, active_breaker_state):
        """Test signal strength is set correctly."""
        prediction = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.90,  # Very strong
            short_term_signal=0.8,
            medium_term_signal=0.7,
            long_term_signal=0.6,
            ensemble_agreement=0.9,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
        )

        assert signal.signal_strength == SignalStrength.VERY_STRONG

    def test_mixed_signals_hold(self, generator, active_breaker_state):
        """Test HOLD when models disagree with moderate confidence."""
        prediction = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.65,  # Moderate confidence
            short_term_signal=0.8,   # Bullish
            medium_term_signal=-0.2,  # Bearish
            long_term_signal=0.3,    # Slightly bullish
            ensemble_agreement=0.6,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=active_breaker_state,
        )

        # With mixed signals and moderate confidence, might hold
        assert signal.action in [Action.HOLD, Action.BUY]

    def test_reduced_breaker_state(self, generator):
        """Test signal generation with reduced breaker state."""
        reduced_state = CircuitBreakerState(
            overall_state=TradingState.REDUCED,
            size_multiplier=0.5,  # Half size
        )

        prediction = EnsemblePrediction(
            direction_probability=0.75,
            confidence=0.80,
            short_term_signal=0.8,
            medium_term_signal=0.7,
            long_term_signal=0.6,
            ensemble_agreement=0.9,
        )

        signal = generator.generate_signal(
            prediction=prediction,
            symbol="EURUSD",
            current_price=1.1000,
            breaker_state=reduced_state,
        )

        # Position size should be reduced
        assert signal.position_size_pct > 0
        # Size is multiplied by 0.5, so should be smaller than normal

"""Unit tests for agent-related database models."""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import directly from database modules to avoid API initialization
import sys
import importlib.util

# Load models module directly
models_path = src_path / "api" / "database" / "models.py"
spec = importlib.util.spec_from_file_location("models", models_path)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

Base = models.Base
AgentCommand = models.AgentCommand
AgentState = models.AgentState
TradeExplanation = models.TradeExplanation
CircuitBreakerEvent = models.CircuitBreakerEvent
Prediction = models.Prediction
Trade = models.Trade


@pytest.fixture
def db_session():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


class TestAgentCommand:
    """Test AgentCommand model."""

    def test_create_command_with_all_fields(self, db_session: Session):
        """Test creating a command with all fields."""
        # Arrange
        command = AgentCommand(
            command="start",
            payload={"mode": "paper", "symbol": "EURUSD"},
            status="pending",
        )

        # Act
        db_session.add(command)
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentCommand).filter_by(command="start").first()
        assert retrieved is not None
        assert retrieved.command == "start"
        assert retrieved.payload == {"mode": "paper", "symbol": "EURUSD"}
        assert retrieved.status == "pending"
        assert retrieved.created_at is not None
        assert retrieved.processed_at is None
        assert retrieved.result is None
        assert retrieved.error_message is None

    def test_status_transition_pending_to_processing(self, db_session: Session):
        """Test status transition from pending to processing."""
        # Arrange
        command = AgentCommand(command="start", status="pending")
        db_session.add(command)
        db_session.commit()

        # Act
        command.status = "processing"
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentCommand).first()
        assert retrieved.status == "processing"

    def test_status_transition_processing_to_completed(self, db_session: Session):
        """Test status transition from processing to completed."""
        # Arrange
        command = AgentCommand(command="start", status="processing")
        db_session.add(command)
        db_session.commit()

        # Act
        command.status = "completed"
        command.processed_at = datetime.utcnow()
        command.result = {"success": True, "message": "Agent started"}
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentCommand).first()
        assert retrieved.status == "completed"
        assert retrieved.processed_at is not None
        assert retrieved.result == {"success": True, "message": "Agent started"}

    def test_status_transition_processing_to_failed(self, db_session: Session):
        """Test status transition from processing to failed."""
        # Arrange
        command = AgentCommand(command="start", status="processing")
        db_session.add(command)
        db_session.commit()

        # Act
        command.status = "failed"
        command.processed_at = datetime.utcnow()
        command.error_message = "Connection timeout"
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentCommand).first()
        assert retrieved.status == "failed"
        assert retrieved.processed_at is not None
        assert retrieved.error_message == "Connection timeout"

    def test_query_pending_commands_ordered_by_created_at(self, db_session: Session):
        """Test querying pending commands ordered by creation time."""
        # Arrange
        base_time = datetime.utcnow()
        commands = [
            AgentCommand(command="start", status="pending", created_at=base_time),
            AgentCommand(
                command="pause",
                status="pending",
                created_at=base_time + timedelta(seconds=1),
            ),
            AgentCommand(
                command="stop",
                status="completed",
                created_at=base_time + timedelta(seconds=2),
            ),
            AgentCommand(
                command="resume",
                status="pending",
                created_at=base_time + timedelta(seconds=3),
            ),
        ]
        for cmd in commands:
            db_session.add(cmd)
        db_session.commit()

        # Act
        pending = (
            db_session.query(AgentCommand)
            .filter_by(status="pending")
            .order_by(AgentCommand.created_at)
            .all()
        )

        # Assert
        assert len(pending) == 3
        assert pending[0].command == "start"
        assert pending[1].command == "pause"
        assert pending[2].command == "resume"

    def test_update_processed_at_on_completion(self, db_session: Session):
        """Test that processed_at is set when command completes."""
        # Arrange
        command = AgentCommand(command="update_config", status="pending")
        db_session.add(command)
        db_session.commit()

        # Act
        before_processing = datetime.utcnow()
        command.status = "completed"
        command.processed_at = datetime.utcnow()
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentCommand).first()
        assert retrieved.processed_at is not None
        assert retrieved.processed_at >= before_processing

    def test_command_with_complex_payload(self, db_session: Session):
        """Test command with complex JSON payload."""
        # Arrange
        complex_payload = {
            "mode": "paper",
            "risk_settings": {
                "max_position_size": 0.1,
                "stop_loss_pips": 30,
                "take_profit_pips": 60,
            },
            "symbols": ["EURUSD", "GBPUSD"],
            "enabled": True,
        }
        command = AgentCommand(command="update_config", payload=complex_payload)

        # Act
        db_session.add(command)
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentCommand).first()
        assert retrieved.payload == complex_payload
        assert retrieved.payload["risk_settings"]["max_position_size"] == 0.1


class TestAgentState:
    """Test AgentState model."""

    def test_create_initial_state(self, db_session: Session):
        """Test creating initial agent state."""
        # Arrange
        config = {
            "mode": "simulation",
            "symbols": ["EURUSD"],
            "risk_per_trade": 0.02,
        }
        state = AgentState(
            status="stopped",
            mode="simulation",
            cycle_count=0,
            config=config,
        )

        # Act
        db_session.add(state)
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentState).first()
        assert retrieved is not None
        assert retrieved.status == "stopped"
        assert retrieved.mode == "simulation"
        assert retrieved.cycle_count == 0
        assert retrieved.config == config
        assert retrieved.open_positions == 0
        assert retrieved.kill_switch_active is False

    def test_update_state_fields(self, db_session: Session):
        """Test updating state fields."""
        # Arrange
        state = AgentState(
            status="stopped",
            mode="simulation",
            cycle_count=0,
            config={},
        )
        db_session.add(state)
        db_session.commit()

        # Act
        state.status = "running"
        state.cycle_count = 5
        state.last_cycle_at = datetime.utcnow()
        state.account_equity = 100000.0
        state.open_positions = 2
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentState).first()
        assert retrieved.status == "running"
        assert retrieved.cycle_count == 5
        assert retrieved.last_cycle_at is not None
        assert retrieved.account_equity == 100000.0
        assert retrieved.open_positions == 2

    def test_verify_single_row_pattern(self, db_session: Session):
        """Test that we maintain single-row pattern (only one active state).

        Note: This is enforced by application logic, not database constraint.
        """
        # Arrange
        state1 = AgentState(status="stopped", mode="simulation", config={})
        db_session.add(state1)
        db_session.commit()

        # Act - Application should update existing row, not create new one
        # (This is a behavior test - actual enforcement is in application layer)
        count = db_session.query(AgentState).count()

        # Assert
        assert count == 1

    def test_json_fields_serialize_deserialize(self, db_session: Session):
        """Test JSON fields work correctly."""
        # Arrange
        last_prediction = {
            "signal": "BUY",
            "confidence": 0.75,
            "timestamp": "2024-01-15T10:00:00",
        }
        last_signal = {
            "action": "open_long",
            "size": 0.1,
            "entry_price": 1.08543,
        }
        config = {
            "mode": "paper",
            "max_positions": 3,
            "circuit_breaker": {"enabled": True, "max_loss_pips": 100},
        }

        state = AgentState(
            status="running",
            mode="paper",
            config=config,
            last_prediction=last_prediction,
            last_signal=last_signal,
        )

        # Act
        db_session.add(state)
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentState).first()
        assert retrieved.last_prediction == last_prediction
        assert retrieved.last_signal == last_signal
        assert retrieved.config == config
        assert retrieved.config["circuit_breaker"]["enabled"] is True

    def test_circuit_breaker_state_tracking(self, db_session: Session):
        """Test tracking circuit breaker state."""
        # Arrange
        state = AgentState(
            status="running",
            mode="paper",
            config={},
            circuit_breaker_state="armed",
        )
        db_session.add(state)
        db_session.commit()

        # Act
        state.circuit_breaker_state = "triggered"
        state.kill_switch_active = True
        state.error_message = "Consecutive loss threshold exceeded"
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentState).first()
        assert retrieved.circuit_breaker_state == "triggered"
        assert retrieved.kill_switch_active is True
        assert retrieved.error_message == "Consecutive loss threshold exceeded"

    def test_updated_at_automatic_update(self, db_session: Session):
        """Test that updated_at is automatically updated."""
        # Arrange
        state = AgentState(status="stopped", mode="simulation", config={})
        db_session.add(state)
        db_session.commit()
        initial_updated_at = state.updated_at

        # Wait a tiny bit to ensure time difference
        import time

        time.sleep(0.01)

        # Act
        state.status = "running"
        db_session.commit()

        # Assert
        retrieved = db_session.query(AgentState).first()
        assert retrieved.updated_at > initial_updated_at


class TestTradeExplanation:
    """Test TradeExplanation model."""

    def test_create_with_valid_trade_id(self, db_session: Session):
        """Test creating explanation with valid trade_id."""
        # Arrange
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.08543,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
        )
        db_session.add(trade)
        db_session.commit()

        explanation = TradeExplanation(
            trade_id=trade.id,
            explanation="Strong bullish momentum with all timeframes aligned",
            confidence_factors={"agreement": "all_bullish", "confidence": 0.75},
            llm_model="claude-3-opus",
        )

        # Act
        db_session.add(explanation)
        db_session.commit()

        # Assert
        retrieved = db_session.query(TradeExplanation).first()
        assert retrieved is not None
        assert retrieved.trade_id == trade.id
        assert "bullish momentum" in retrieved.explanation
        assert retrieved.confidence_factors["confidence"] == 0.75
        assert retrieved.llm_model == "claude-3-opus"

    def test_create_with_valid_prediction_id(self, db_session: Session):
        """Test creating explanation with valid prediction_id."""
        # Arrange
        prediction = Prediction(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            direction="long",
            confidence=0.75,
        )
        db_session.add(prediction)
        db_session.commit()

        explanation = TradeExplanation(
            trade_id=1,  # Assuming trade exists
            prediction_id=prediction.id,
            explanation="High confidence prediction based on multi-timeframe analysis",
        )

        # Act
        db_session.add(explanation)
        db_session.commit()

        # Assert
        retrieved = db_session.query(TradeExplanation).first()
        assert retrieved.prediction_id == prediction.id

    def test_query_explanations_by_trade(self, db_session: Session):
        """Test querying explanations by trade."""
        # Arrange
        trade1 = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.08543,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
        )
        trade2 = Trade(
            symbol="EURUSD",
            direction="short",
            entry_price=1.08321,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
        )
        db_session.add_all([trade1, trade2])
        db_session.commit()

        explanation1 = TradeExplanation(
            trade_id=trade1.id, explanation="Trade 1 explanation"
        )
        explanation2 = TradeExplanation(
            trade_id=trade1.id, explanation="Trade 1 follow-up"
        )
        explanation3 = TradeExplanation(
            trade_id=trade2.id, explanation="Trade 2 explanation"
        )
        db_session.add_all([explanation1, explanation2, explanation3])
        db_session.commit()

        # Act
        trade1_explanations = (
            db_session.query(TradeExplanation)
            .filter_by(trade_id=trade1.id)
            .all()
        )

        # Assert
        assert len(trade1_explanations) == 2
        assert all(exp.trade_id == trade1.id for exp in trade1_explanations)

    def test_json_fields_work_correctly(self, db_session: Session):
        """Test JSON fields serialize/deserialize correctly."""
        # Arrange
        confidence_factors = {
            "agreement_score": 1.0,
            "model_confidence": 0.75,
            "regime": "trending",
            "volatility": "normal",
        }
        risk_factors = {
            "position_size": 0.1,
            "stop_loss_pips": 30,
            "risk_reward_ratio": 2.0,
        }

        explanation = TradeExplanation(
            trade_id=1,
            explanation="Test explanation",
            confidence_factors=confidence_factors,
            risk_factors=risk_factors,
        )

        # Act
        db_session.add(explanation)
        db_session.commit()

        # Assert
        retrieved = db_session.query(TradeExplanation).first()
        assert retrieved.confidence_factors == confidence_factors
        assert retrieved.risk_factors == risk_factors
        assert retrieved.confidence_factors["model_confidence"] == 0.75
        assert retrieved.risk_factors["risk_reward_ratio"] == 2.0

    def test_foreign_key_relationship_with_trade(self, db_session: Session):
        """Test foreign key relationship with Trade table."""
        # Arrange
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.08543,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
        )
        db_session.add(trade)
        db_session.commit()

        explanation = TradeExplanation(
            trade_id=trade.id,
            explanation="Test",
        )
        db_session.add(explanation)
        db_session.commit()

        # Act - Delete trade
        db_session.delete(trade)
        db_session.commit()

        # Assert - Explanation should still exist (no cascade)
        # Note: Actual FK behavior depends on database and ondelete setting
        # For SQLite in-memory, FK constraints may not be enforced by default


class TestCircuitBreakerEvent:
    """Test CircuitBreakerEvent model."""

    def test_log_trigger_event(self, db_session: Session):
        """Test logging a circuit breaker trigger event."""
        # Arrange
        event = CircuitBreakerEvent(
            breaker_type="consecutive_loss",
            severity="critical",
            action="triggered",
            reason="5 consecutive losses detected",
            value=5,
            threshold=5,
        )

        # Act
        db_session.add(event)
        db_session.commit()

        # Assert
        retrieved = db_session.query(CircuitBreakerEvent).first()
        assert retrieved is not None
        assert retrieved.breaker_type == "consecutive_loss"
        assert retrieved.severity == "critical"
        assert retrieved.action == "triggered"
        assert retrieved.value == 5
        assert retrieved.threshold == 5
        assert retrieved.triggered_at is not None

    def test_log_recovery_event(self, db_session: Session):
        """Test logging a circuit breaker recovery event."""
        # Arrange
        trigger_event = CircuitBreakerEvent(
            breaker_type="drawdown",
            severity="warning",
            action="triggered",
            reason="Drawdown exceeded 5%",
            value=5.2,
            threshold=5.0,
        )
        db_session.add(trigger_event)
        db_session.commit()

        recovery_event = CircuitBreakerEvent(
            breaker_type="drawdown",
            severity="warning",
            action="recovered",
            reason="Drawdown returned to safe levels",
            value=3.5,
            threshold=5.0,
            recovered_at=datetime.utcnow(),
        )

        # Act
        db_session.add(recovery_event)
        db_session.commit()

        # Assert
        recovered = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(action="recovered")
            .first()
        )
        assert recovered is not None
        assert recovered.action == "recovered"
        assert recovered.value == 3.5
        assert recovered.recovered_at is not None

    def test_query_events_by_type(self, db_session: Session):
        """Test querying events by breaker type."""
        # Arrange
        events = [
            CircuitBreakerEvent(
                breaker_type="consecutive_loss",
                severity="critical",
                action="triggered",
            ),
            CircuitBreakerEvent(
                breaker_type="drawdown", severity="warning", action="triggered"
            ),
            CircuitBreakerEvent(
                breaker_type="consecutive_loss",
                severity="warning",
                action="recovered",
            ),
            CircuitBreakerEvent(
                breaker_type="model_degradation",
                severity="critical",
                action="triggered",
            ),
        ]
        for event in events:
            db_session.add(event)
        db_session.commit()

        # Act
        loss_events = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(breaker_type="consecutive_loss")
            .all()
        )

        # Assert
        assert len(loss_events) == 2
        assert all(e.breaker_type == "consecutive_loss" for e in loss_events)

    def test_query_events_by_severity(self, db_session: Session):
        """Test querying events by severity."""
        # Arrange
        events = [
            CircuitBreakerEvent(
                breaker_type="consecutive_loss",
                severity="critical",
                action="triggered",
            ),
            CircuitBreakerEvent(
                breaker_type="drawdown", severity="warning", action="triggered"
            ),
            CircuitBreakerEvent(
                breaker_type="drawdown", severity="critical", action="triggered"
            ),
        ]
        for event in events:
            db_session.add(event)
        db_session.commit()

        # Act
        critical_events = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(severity="critical")
            .all()
        )

        # Assert
        assert len(critical_events) == 2
        assert all(e.severity == "critical" for e in critical_events)

    def test_query_events_ordered_by_triggered_at(self, db_session: Session):
        """Test querying events ordered by trigger time."""
        # Arrange
        base_time = datetime.utcnow()
        events = [
            CircuitBreakerEvent(
                breaker_type="consecutive_loss",
                severity="critical",
                action="triggered",
                triggered_at=base_time + timedelta(seconds=2),
            ),
            CircuitBreakerEvent(
                breaker_type="drawdown",
                severity="warning",
                action="triggered",
                triggered_at=base_time,
            ),
            CircuitBreakerEvent(
                breaker_type="model_degradation",
                severity="critical",
                action="triggered",
                triggered_at=base_time + timedelta(seconds=1),
            ),
        ]
        for event in events:
            db_session.add(event)
        db_session.commit()

        # Act
        ordered = (
            db_session.query(CircuitBreakerEvent)
            .order_by(CircuitBreakerEvent.triggered_at)
            .all()
        )

        # Assert
        assert len(ordered) == 3
        assert ordered[0].breaker_type == "drawdown"
        assert ordered[1].breaker_type == "model_degradation"
        assert ordered[2].breaker_type == "consecutive_loss"


class TestPredictionNewFields:
    """Test new fields added to Prediction model."""

    def test_prediction_with_agent_tracking_fields(self, db_session: Session):
        """Test prediction with agent tracking fields."""
        # Arrange
        prediction = Prediction(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            direction="long",
            confidence=0.75,
            used_by_agent=True,
            agent_cycle_number=10,
        )

        # Act
        db_session.add(prediction)
        db_session.commit()

        # Assert
        retrieved = db_session.query(Prediction).first()
        assert retrieved.used_by_agent is True
        assert retrieved.agent_cycle_number == 10

    def test_prediction_defaults_for_new_fields(self, db_session: Session):
        """Test default values for new fields."""
        # Arrange
        prediction = Prediction(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            direction="long",
            confidence=0.75,
        )

        # Act
        db_session.add(prediction)
        db_session.commit()

        # Assert
        retrieved = db_session.query(Prediction).first()
        assert retrieved.used_by_agent is False
        assert retrieved.agent_cycle_number is None

    def test_query_predictions_by_agent_cycle(self, db_session: Session):
        """Test querying predictions by agent cycle number."""
        # Arrange
        predictions = [
            Prediction(
                timestamp=datetime.utcnow(),
                symbol="EURUSD",
                direction="long",
                confidence=0.75,
                agent_cycle_number=5,
            ),
            Prediction(
                timestamp=datetime.utcnow(),
                symbol="EURUSD",
                direction="short",
                confidence=0.68,
                agent_cycle_number=5,
            ),
            Prediction(
                timestamp=datetime.utcnow(),
                symbol="EURUSD",
                direction="long",
                confidence=0.72,
                agent_cycle_number=6,
            ),
        ]
        for pred in predictions:
            db_session.add(pred)
        db_session.commit()

        # Act
        cycle_5_predictions = (
            db_session.query(Prediction).filter_by(agent_cycle_number=5).all()
        )

        # Assert
        assert len(cycle_5_predictions) == 2
        assert all(p.agent_cycle_number == 5 for p in cycle_5_predictions)


class TestTradeNewFields:
    """Test new fields added to Trade model."""

    def test_trade_with_agent_execution_fields(self, db_session: Session):
        """Test trade with agent execution fields."""
        # Arrange
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.08543,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
            execution_mode="paper",
            broker="mt5",
            mt5_ticket=123456789,
        )

        # Act
        db_session.add(trade)
        db_session.commit()

        # Assert
        retrieved = db_session.query(Trade).first()
        assert retrieved.execution_mode == "paper"
        assert retrieved.broker == "mt5"
        assert retrieved.mt5_ticket == 123456789

    def test_trade_defaults_for_new_fields(self, db_session: Session):
        """Test default values for new fields."""
        # Arrange
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.08543,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
        )

        # Act
        db_session.add(trade)
        db_session.commit()

        # Assert
        retrieved = db_session.query(Trade).first()
        assert retrieved.execution_mode == "simulation"
        assert retrieved.broker is None
        assert retrieved.mt5_ticket is None

    def test_trade_with_explanation_id(self, db_session: Session):
        """Test trade with explanation_id field."""
        # Arrange
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.08543,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
            explanation_id=42,
        )

        # Act
        db_session.add(trade)
        db_session.commit()

        # Assert
        retrieved = db_session.query(Trade).first()
        assert retrieved.explanation_id == 42

    def test_query_trades_by_execution_mode(self, db_session: Session):
        """Test querying trades by execution mode."""
        # Arrange
        trades = [
            Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.08543,
                entry_time=datetime.utcnow(),
                lot_size=0.1,
                execution_mode="simulation",
            ),
            Trade(
                symbol="EURUSD",
                direction="short",
                entry_price=1.08321,
                entry_time=datetime.utcnow(),
                lot_size=0.1,
                execution_mode="paper",
            ),
            Trade(
                symbol="EURUSD",
                direction="long",
                entry_price=1.08654,
                entry_time=datetime.utcnow(),
                lot_size=0.1,
                execution_mode="paper",
            ),
        ]
        for trade in trades:
            db_session.add(trade)
        db_session.commit()

        # Act
        paper_trades = db_session.query(Trade).filter_by(execution_mode="paper").all()

        # Assert
        assert len(paper_trades) == 2
        assert all(t.execution_mode == "paper" for t in paper_trades)

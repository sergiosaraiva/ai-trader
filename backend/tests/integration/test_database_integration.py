"""Integration tests for database operations."""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import directly from database modules to avoid API initialization
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
    """Create an in-memory SQLite database for integration testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()


class TestAgentCommandWorkflow:
    """Test complete agent command workflow."""

    def test_command_lifecycle_start_to_complete(self, db_session):
        """Test full command lifecycle from creation to completion."""
        # Arrange - Backend creates a command
        command = AgentCommand(
            command="start",
            payload={"mode": "paper", "symbols": ["EURUSD"]},
            status="pending",
        )
        db_session.add(command)
        db_session.commit()

        # Act 1 - Agent picks up pending command
        pending_commands = (
            db_session.query(AgentCommand)
            .filter_by(status="pending")
            .order_by(AgentCommand.created_at)
            .all()
        )
        assert len(pending_commands) == 1
        cmd = pending_commands[0]

        # Act 2 - Agent marks command as processing
        cmd.status = "processing"
        db_session.commit()

        # Act 3 - Agent executes command and marks complete
        cmd.status = "completed"
        cmd.processed_at = datetime.utcnow()
        cmd.result = {"success": True, "agent_status": "running"}
        db_session.commit()

        # Assert - Command lifecycle complete
        final = db_session.query(AgentCommand).first()
        assert final.status == "completed"
        assert final.processed_at is not None
        assert final.result["success"] is True

    def test_command_lifecycle_with_failure(self, db_session):
        """Test command lifecycle with failure."""
        # Arrange
        command = AgentCommand(
            command="update_config",
            payload={"invalid_key": "value"},
            status="pending",
        )
        db_session.add(command)
        db_session.commit()

        # Act - Agent processes and fails
        cmd = db_session.query(AgentCommand).first()
        cmd.status = "processing"
        db_session.commit()

        cmd.status = "failed"
        cmd.processed_at = datetime.utcnow()
        cmd.error_message = "Invalid configuration key"
        db_session.commit()

        # Assert
        failed = db_session.query(AgentCommand).first()
        assert failed.status == "failed"
        assert failed.error_message == "Invalid configuration key"

    def test_multiple_commands_processing_order(self, db_session):
        """Test that multiple commands are processed in order."""
        # Arrange - Create multiple commands
        base_time = datetime.utcnow()
        commands = [
            AgentCommand(
                command="pause", status="pending", created_at=base_time
            ),
            AgentCommand(
                command="resume",
                status="pending",
                created_at=base_time + timedelta(seconds=1),
            ),
            AgentCommand(
                command="stop",
                status="pending",
                created_at=base_time + timedelta(seconds=2),
            ),
        ]
        for cmd in commands:
            db_session.add(cmd)
        db_session.commit()

        # Act - Agent processes in order
        pending = (
            db_session.query(AgentCommand)
            .filter_by(status="pending")
            .order_by(AgentCommand.created_at)
            .all()
        )

        # Assert - Order is correct
        assert len(pending) == 3
        assert pending[0].command == "pause"
        assert pending[1].command == "resume"
        assert pending[2].command == "stop"


class TestAgentStateManagement:
    """Test agent state management."""

    def test_agent_state_full_lifecycle(self, db_session):
        """Test complete agent state lifecycle."""
        # Arrange - Initialize state
        initial_config = {
            "mode": "simulation",
            "symbols": ["EURUSD"],
            "risk_per_trade": 0.02,
        }
        state = AgentState(
            status="stopped",
            mode="simulation",
            cycle_count=0,
            config=initial_config,
        )
        db_session.add(state)
        db_session.commit()

        # Act 1 - Start agent
        state.status = "starting"
        state.started_at = datetime.utcnow()
        db_session.commit()

        state.status = "running"
        db_session.commit()

        # Act 2 - Run cycles
        for i in range(5):
            state.cycle_count += 1
            state.last_cycle_at = datetime.utcnow()
            state.account_equity = 100000 + (i * 100)
            db_session.commit()

        # Act 3 - Stop agent
        state.status = "stopping"
        db_session.commit()

        state.status = "stopped"
        db_session.commit()

        # Assert - Final state
        final = db_session.query(AgentState).first()
        assert final.status == "stopped"
        assert final.cycle_count == 5
        assert final.account_equity == 100400

    def test_agent_state_with_circuit_breaker(self, db_session):
        """Test agent state with circuit breaker activation."""
        # Arrange
        state = AgentState(
            status="running",
            mode="paper",
            cycle_count=10,
            config={},
        )
        db_session.add(state)
        db_session.commit()

        # Act - Circuit breaker triggers
        state.circuit_breaker_state = "triggered"
        state.kill_switch_active = True
        state.status = "paused"
        state.error_message = "Consecutive loss threshold exceeded"
        db_session.commit()

        # Assert
        paused = db_session.query(AgentState).first()
        assert paused.status == "paused"
        assert paused.circuit_breaker_state == "triggered"
        assert paused.kill_switch_active is True

        # Act - Recover from circuit breaker
        state.circuit_breaker_state = "recovered"
        state.kill_switch_active = False
        state.status = "running"
        state.error_message = None
        db_session.commit()

        # Assert - Recovered
        recovered = db_session.query(AgentState).first()
        assert recovered.status == "running"
        assert recovered.kill_switch_active is False


class TestTradeWithExplanation:
    """Test trade with explanation relationships."""

    def test_trade_with_explanation_full_workflow(self, db_session):
        """Test complete workflow of trade with explanation."""
        # Arrange - Create prediction
        prediction = Prediction(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            direction="long",
            confidence=0.75,
            used_by_agent=True,
            agent_cycle_number=5,
        )
        db_session.add(prediction)
        db_session.commit()

        # Act 1 - Create trade
        trade = Trade(
            symbol="EURUSD",
            direction="long",
            entry_price=1.08543,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
            execution_mode="paper",
            broker="mt5",
            prediction_id=prediction.id,
        )
        db_session.add(trade)
        db_session.commit()

        # Act 2 - Create explanation
        explanation = TradeExplanation(
            trade_id=trade.id,
            prediction_id=prediction.id,
            explanation="Strong bullish momentum with all timeframes aligned. "
            "1H shows uptrend with RSI at 58, 4H confirms with MACD crossover, "
            "Daily shows support at key level.",
            confidence_factors={
                "agreement_score": 1.0,
                "all_timeframes_aligned": True,
                "confidence": 0.75,
            },
            risk_factors={
                "position_size": 0.1,
                "stop_loss_pips": 30,
                "take_profit_pips": 60,
                "risk_reward_ratio": 2.0,
            },
            llm_model="claude-3-opus",
        )
        db_session.add(explanation)
        db_session.commit()

        # Update trade with explanation_id (if needed)
        trade.explanation_id = explanation.id
        db_session.commit()

        # Assert - All relationships exist
        retrieved_trade = db_session.query(Trade).first()
        retrieved_explanation = db_session.query(TradeExplanation).first()
        retrieved_prediction = db_session.query(Prediction).first()

        assert retrieved_trade.prediction_id == prediction.id
        assert retrieved_explanation.trade_id == trade.id
        assert retrieved_explanation.prediction_id == prediction.id
        assert retrieved_trade.explanation_id == explanation.id

    def test_multiple_explanations_per_trade(self, db_session):
        """Test that a trade can have multiple explanations."""
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

        # Act - Create multiple explanations
        explanations = [
            TradeExplanation(
                trade_id=trade.id,
                explanation="Initial trade decision based on momentum",
                llm_model="claude-3-opus",
            ),
            TradeExplanation(
                trade_id=trade.id,
                explanation="Trade still valid after 1 hour, holding position",
                llm_model="claude-3-opus",
            ),
            TradeExplanation(
                trade_id=trade.id,
                explanation="Exit decision: take profit target reached",
                llm_model="claude-3-opus",
            ),
        ]
        for exp in explanations:
            db_session.add(exp)
        db_session.commit()

        # Assert
        trade_explanations = (
            db_session.query(TradeExplanation)
            .filter_by(trade_id=trade.id)
            .all()
        )
        assert len(trade_explanations) == 3


class TestCircuitBreakerEventTracking:
    """Test circuit breaker event tracking."""

    def test_circuit_breaker_trigger_and_recovery_cycle(self, db_session):
        """Test complete circuit breaker trigger and recovery cycle."""
        # Act 1 - Trigger event
        trigger = CircuitBreakerEvent(
            breaker_type="consecutive_loss",
            severity="critical",
            action="triggered",
            reason="5 consecutive losing trades",
            value=5,
            threshold=5,
        )
        db_session.add(trigger)
        db_session.commit()

        # Update agent state
        state = AgentState(
            status="paused",
            mode="paper",
            config={},
            circuit_breaker_state="triggered",
            kill_switch_active=True,
        )
        db_session.add(state)
        db_session.commit()

        # Act 2 - Recovery after manual intervention
        recovery = CircuitBreakerEvent(
            breaker_type="consecutive_loss",
            severity="critical",
            action="recovered",
            reason="Manual reset after review",
            value=0,
            threshold=5,
            recovered_at=datetime.utcnow(),
        )
        db_session.add(recovery)
        db_session.commit()

        # Update agent state
        state.circuit_breaker_state = "recovered"
        state.kill_switch_active = False
        state.status = "running"
        db_session.commit()

        # Assert
        events = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(breaker_type="consecutive_loss")
            .order_by(CircuitBreakerEvent.triggered_at)
            .all()
        )
        assert len(events) == 2
        assert events[0].action == "triggered"
        assert events[1].action == "recovered"
        assert events[1].recovered_at is not None

    def test_multiple_circuit_breaker_types(self, db_session):
        """Test tracking multiple circuit breaker types."""
        # Arrange & Act
        events = [
            CircuitBreakerEvent(
                breaker_type="consecutive_loss",
                severity="critical",
                action="triggered",
                value=5,
                threshold=5,
            ),
            CircuitBreakerEvent(
                breaker_type="drawdown",
                severity="warning",
                action="triggered",
                value=5.2,
                threshold=5.0,
            ),
            CircuitBreakerEvent(
                breaker_type="model_degradation",
                severity="warning",
                action="triggered",
                reason="Accuracy dropped below 55%",
                value=52.5,
                threshold=55.0,
            ),
        ]
        for event in events:
            db_session.add(event)
        db_session.commit()

        # Assert - Query by type
        loss_events = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(breaker_type="consecutive_loss")
            .count()
        )
        drawdown_events = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(breaker_type="drawdown")
            .count()
        )
        degradation_events = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(breaker_type="model_degradation")
            .count()
        )

        assert loss_events == 1
        assert drawdown_events == 1
        assert degradation_events == 1

        # Assert - Query by severity
        critical_events = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(severity="critical")
            .count()
        )
        warning_events = (
            db_session.query(CircuitBreakerEvent)
            .filter_by(severity="warning")
            .count()
        )

        assert critical_events == 1
        assert warning_events == 2


class TestPredictionToTradeWorkflow:
    """Test complete workflow from prediction to trade."""

    def test_agent_cycle_prediction_to_trade(self, db_session):
        """Test complete agent cycle from prediction to trade execution."""
        # Act 1 - Agent generates prediction
        prediction = Prediction(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            direction="long",
            confidence=0.75,
            prob_up=0.75,
            prob_down=0.25,
            should_trade=True,
            used_by_agent=True,
            agent_cycle_number=10,
        )
        db_session.add(prediction)
        db_session.commit()

        # Act 2 - Agent decides to trade based on prediction
        trade = Trade(
            prediction_id=prediction.id,
            symbol="EURUSD",
            direction="long",
            entry_price=1.08543,
            entry_time=datetime.utcnow(),
            lot_size=0.1,
            take_profit=1.08603,  # 60 pips
            stop_loss=1.08513,  # 30 pips
            confidence=0.75,
            execution_mode="paper",
            broker="mt5",
            mt5_ticket=123456789,
        )
        db_session.add(trade)
        db_session.commit()

        # Act 3 - Generate explanation
        explanation = TradeExplanation(
            trade_id=trade.id,
            prediction_id=prediction.id,
            explanation="High confidence long signal with all timeframes aligned",
            confidence_factors={"agreement": "unanimous", "confidence": 0.75},
            risk_factors={"risk_reward_ratio": 2.0},
        )
        db_session.add(explanation)
        db_session.commit()

        # Act 4 - Update trade with explanation reference
        trade.explanation_id = explanation.id
        db_session.commit()

        # Act 5 - Close trade
        trade.exit_price = 1.08603
        trade.exit_time = datetime.utcnow()
        trade.exit_reason = "tp"
        trade.pips = 60
        trade.pnl_usd = 60  # Simplified calculation
        trade.is_winner = True
        trade.status = "closed"
        db_session.commit()

        # Assert - Complete workflow captured
        final_prediction = db_session.query(Prediction).first()
        final_trade = db_session.query(Trade).first()
        final_explanation = db_session.query(TradeExplanation).first()

        assert final_prediction.used_by_agent is True
        assert final_prediction.agent_cycle_number == 10
        assert final_trade.prediction_id == final_prediction.id
        assert final_trade.status == "closed"
        assert final_trade.is_winner is True
        assert final_explanation.trade_id == final_trade.id
        assert final_explanation.prediction_id == final_prediction.id


class TestIndexPerformance:
    """Test that indexes are properly created."""

    def test_agent_command_indexes_exist(self, db_session):
        """Test that agent_commands indexes exist."""
        # Get engine from session
        engine = db_session.get_bind()
        from sqlalchemy import inspect

        inspector = inspect(engine)

        # Get indexes for agent_commands table
        indexes = inspector.get_indexes("agent_commands")
        index_names = {idx["name"] for idx in indexes}

        # Assert key indexes exist
        assert "idx_agent_commands_status" in index_names
        assert "idx_agent_commands_created" in index_names

    def test_agent_state_indexes_exist(self, db_session):
        """Test that agent_state indexes exist."""
        engine = db_session.get_bind()
        from sqlalchemy import inspect

        inspector = inspect(engine)
        indexes = inspector.get_indexes("agent_state")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_agent_state_status" in index_names
        assert "idx_agent_state_updated" in index_names

    def test_trade_explanations_indexes_exist(self, db_session):
        """Test that trade_explanations indexes exist."""
        engine = db_session.get_bind()
        from sqlalchemy import inspect

        inspector = inspect(engine)
        indexes = inspector.get_indexes("trade_explanations")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_trade_explanations_trade" in index_names
        assert "idx_trade_explanations_prediction" in index_names

    def test_predictions_agent_cycle_index_exists(self, db_session):
        """Test that predictions agent_cycle_number index exists."""
        engine = db_session.get_bind()
        from sqlalchemy import inspect

        inspector = inspect(engine)
        indexes = inspector.get_indexes("predictions")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_predictions_agent_cycle" in index_names

    def test_trades_execution_mode_index_exists(self, db_session):
        """Test that trades execution_mode index exists."""
        engine = db_session.get_bind()
        from sqlalchemy import inspect

        inspector = inspect(engine)
        indexes = inspector.get_indexes("trades")
        index_names = {idx["name"] for idx in indexes}

        assert "idx_trades_execution_mode" in index_names

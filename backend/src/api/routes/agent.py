"""Agent control endpoints.

This module provides REST API endpoints for controlling the autonomous trading agent
via command queue pattern. Commands are queued in the agent_commands table, and the
agent polls and processes them asynchronously.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException, Query, Depends, Body, Request
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from ..database.session import get_db
from ..database.models import AgentCommand, AgentState, Trade, CircuitBreakerEvent
from ..schemas.agent import (
    AgentStartRequest,
    AgentStopRequest,
    AgentConfigUpdateRequest,
    KillSwitchRequest,
    CommandResponse,
    AgentStatusResponse,
    AgentMetricsResponse,
    CommandStatusResponse,
    CommandListResponse,
    CommandListItem,
)
from pydantic import BaseModel
from ..utils.logging import log_exception
from ..utils.rate_limiter import limiter, RATE_LIMIT_AGENT_COMMANDS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


# Helper Functions


def _get_agent_state(db: Session) -> Optional[AgentState]:
    """Get the current agent state from database.

    Returns the most recent agent state record, or None if not initialized.
    """
    return db.query(AgentState).order_by(AgentState.updated_at.desc()).first()


def _queue_command(
    db: Session, command: str, payload: Optional[Dict[str, Any]] = None
) -> AgentCommand:
    """Queue a command for the agent to process.

    Args:
        db: Database session
        command: Command name (start, stop, pause, resume, kill, update_config)
        payload: Optional command parameters

    Returns:
        The created AgentCommand record
    """
    cmd = AgentCommand(
        command=command,
        payload=payload or {},
        status="pending",
        created_at=datetime.utcnow(),
    )
    db.add(cmd)
    db.commit()
    db.refresh(cmd)
    logger.info(f"Queued command '{command}' with ID {cmd.id}")
    return cmd


# Command Endpoints


@router.post("/start", response_model=CommandResponse)
@limiter.limit(RATE_LIMIT_AGENT_COMMANDS)
async def start_agent(
    request: Request,
    body: AgentStartRequest = Body(...),
    db: Session = Depends(get_db),
) -> CommandResponse:
    """Queue start command for the agent.

    This queues a start command with the specified configuration. The agent
    will pick up the command from the queue and begin autonomous trading.

    The agent must be in 'stopped' status to start.
    """
    try:
        # Check current agent state
        state = _get_agent_state(db)

        if state and state.status not in ("stopped", "error"):
            raise HTTPException(
                status_code=400,
                detail=f"Agent is already {state.status}. Stop it first before starting.",
            )

        # Queue start command with configuration
        payload = {
            "mode": body.mode,
            "confidence_threshold": body.confidence_threshold,
            "cycle_interval_seconds": body.cycle_interval_seconds,
            "max_position_size": body.max_position_size,
            "use_kelly_sizing": body.use_kelly_sizing,
        }

        cmd = _queue_command(db, "start", payload)

        return CommandResponse(
            status="queued",
            command_id=cmd.id,
            message=f"Start command queued successfully in {body.mode} mode. Agent will process shortly.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error queuing start command", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop", response_model=CommandResponse)
@limiter.limit(RATE_LIMIT_AGENT_COMMANDS)
async def stop_agent(
    request: Request,
    body: AgentStopRequest = Body(...),
    db: Session = Depends(get_db),
) -> CommandResponse:
    """Queue stop command for the agent.

    This queues a stop command. The agent will finish its current cycle,
    optionally close positions, and then stop.
    """
    try:
        # Check current agent state
        state = _get_agent_state(db)

        if not state:
            raise HTTPException(
                status_code=404,
                detail="Agent not initialized. Nothing to stop.",
            )

        if state.status == "stopped":
            raise HTTPException(
                status_code=400,
                detail="Agent is already stopped.",
            )

        # Queue stop command
        payload = {
            "force": body.force,
            "close_positions": body.close_positions,
        }

        cmd = _queue_command(db, "stop", payload)

        return CommandResponse(
            status="queued",
            command_id=cmd.id,
            message="Stop command queued successfully. Agent will stop after current cycle.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error queuing stop command", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pause", response_model=CommandResponse)
@limiter.limit(RATE_LIMIT_AGENT_COMMANDS)
async def pause_agent(request: Request, db: Session = Depends(get_db)) -> CommandResponse:
    """Queue pause command for the agent.

    This pauses the agent without stopping it. Trading cycles stop but
    positions remain open.
    """
    try:
        # Check current agent state
        state = _get_agent_state(db)

        if not state:
            raise HTTPException(
                status_code=404,
                detail="Agent not initialized. Nothing to pause.",
            )

        if state.status != "running":
            raise HTTPException(
                status_code=400,
                detail=f"Agent is {state.status}, not running. Cannot pause.",
            )

        # Queue pause command
        cmd = _queue_command(db, "pause")

        return CommandResponse(
            status="queued",
            command_id=cmd.id,
            message="Pause command queued successfully. Agent will pause after current cycle.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error queuing pause command", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume", response_model=CommandResponse)
@limiter.limit(RATE_LIMIT_AGENT_COMMANDS)
async def resume_agent(request: Request, db: Session = Depends(get_db)) -> CommandResponse:
    """Queue resume command for the agent.

    This resumes a paused agent, continuing trading cycles.
    """
    try:
        # Check current agent state
        state = _get_agent_state(db)

        if not state:
            raise HTTPException(
                status_code=404,
                detail="Agent not initialized. Nothing to resume.",
            )

        if state.status != "paused":
            raise HTTPException(
                status_code=400,
                detail=f"Agent is {state.status}, not paused. Cannot resume.",
            )

        # Queue resume command
        cmd = _queue_command(db, "resume")

        return CommandResponse(
            status="queued",
            command_id=cmd.id,
            message="Resume command queued successfully. Agent will resume trading.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error queuing resume command", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/config", response_model=CommandResponse)
async def update_agent_config(
    request: AgentConfigUpdateRequest = Body(...),
    db: Session = Depends(get_db),
) -> CommandResponse:
    """Queue config update command for the agent.

    This updates agent configuration while running. Only non-None fields
    will be updated.
    """
    try:
        # Check current agent state
        state = _get_agent_state(db)

        if not state:
            raise HTTPException(
                status_code=404,
                detail="Agent not initialized. Start the agent first.",
            )

        if state.status == "stopped":
            raise HTTPException(
                status_code=400,
                detail="Agent is stopped. Configuration updates only apply to running agents.",
            )

        # Build payload with only provided fields
        payload = {}
        if request.confidence_threshold is not None:
            payload["confidence_threshold"] = request.confidence_threshold
        if request.cycle_interval_seconds is not None:
            payload["cycle_interval_seconds"] = request.cycle_interval_seconds
        if request.max_position_size is not None:
            payload["max_position_size"] = request.max_position_size
        if request.use_kelly_sizing is not None:
            payload["use_kelly_sizing"] = request.use_kelly_sizing

        if not payload:
            raise HTTPException(
                status_code=400,
                detail="No configuration fields provided to update.",
            )

        # Queue update_config command
        cmd = _queue_command(db, "update_config", payload)

        return CommandResponse(
            status="queued",
            command_id=cmd.id,
            message=f"Config update queued successfully. {len(payload)} fields will be updated.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error queuing config update", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/kill-switch", response_model=CommandResponse)
@limiter.limit(RATE_LIMIT_AGENT_COMMANDS)
async def kill_switch(
    request: Request,
    body: KillSwitchRequest = Body(...),
    db: Session = Depends(get_db),
) -> CommandResponse:
    """Trigger or reset the kill switch.

    Kill switch halts all trading immediately and closes positions.
    Use this for emergency stops.
    """
    try:
        # Check current agent state
        state = _get_agent_state(db)

        if not state:
            raise HTTPException(
                status_code=404,
                detail="Agent not initialized.",
            )

        if body.action == "trigger":
            if state.kill_switch_active:
                raise HTTPException(
                    status_code=400,
                    detail="Kill switch is already active.",
                )

            # Queue kill command with high priority
            payload = {"reason": body.reason or "Manual kill switch triggered"}
            cmd = _queue_command(db, "kill", payload)

            return CommandResponse(
                status="queued",
                command_id=cmd.id,
                message="Kill switch triggered. Agent will halt immediately and close positions.",
            )

        else:  # action == "reset"
            if not state.kill_switch_active:
                raise HTTPException(
                    status_code=400,
                    detail="Kill switch is not active. Nothing to reset.",
                )

            # Queue reset command
            cmd = _queue_command(db, "reset_kill_switch")

            return CommandResponse(
                status="queued",
                command_id=cmd.id,
                message="Kill switch reset queued. Agent can be restarted after processing.",
            )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error processing kill switch", e)
        raise HTTPException(status_code=500, detail=str(e))


# Status & Metrics Endpoints


@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status(db: Session = Depends(get_db)) -> AgentStatusResponse:
    """Get current agent status.

    Returns the current state of the agent including status, configuration,
    and runtime information. This reads from the agent_state table which is
    updated by the agent on each cycle.
    """
    try:
        # Get current state
        state = _get_agent_state(db)

        if not state:
            raise HTTPException(
                status_code=404,
                detail="Agent not initialized. Start the agent first.",
            )

        # Calculate uptime if running
        uptime_seconds = None
        if state.started_at and state.status in ("running", "paused"):
            uptime_seconds = (datetime.utcnow() - state.started_at).total_seconds()

        # Format last_cycle_at
        last_cycle_at = None
        if state.last_cycle_at:
            last_cycle_at = state.last_cycle_at.isoformat()

        return AgentStatusResponse(
            status=state.status,
            mode=state.mode,
            cycle_count=state.cycle_count,
            last_cycle_at=last_cycle_at,
            account_equity=state.account_equity,
            open_positions=state.open_positions,
            circuit_breaker_state=state.circuit_breaker_state,
            kill_switch_active=state.kill_switch_active,
            error_message=state.error_message,
            uptime_seconds=uptime_seconds,
            last_prediction=state.last_prediction,
            config=state.config,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error getting agent status", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    period: str = Query(
        default="all",
        description="Time period: 'all', '24h', '7d', '30d'",
        pattern="^(all|24h|7d|30d)$",
    ),
    db: Session = Depends(get_db),
) -> AgentMetricsResponse:
    """Get agent performance metrics.

    Returns trading performance statistics for the specified time period.
    Metrics include win rate, profit factor, Sharpe ratio, and circuit breaker triggers.
    """
    try:
        # Calculate time window
        start_time = None
        if period == "24h":
            start_time = datetime.utcnow() - timedelta(hours=24)
        elif period == "7d":
            start_time = datetime.utcnow() - timedelta(days=7)
        elif period == "30d":
            start_time = datetime.utcnow() - timedelta(days=30)

        # Query trades
        query = db.query(Trade).filter(Trade.status == "closed")
        if start_time:
            query = query.filter(Trade.exit_time >= start_time)

        trades = query.all()

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.is_winner)
        losing_trades = sum(1 for t in trades if not t.is_winner)

        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        total_pips = sum(t.pips or 0 for t in trades)

        # Profit factor
        gross_profit = sum(t.pips for t in trades if t.pips and t.pips > 0)
        gross_loss = abs(sum(t.pips for t in trades if t.pips and t.pips < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Sharpe ratio (simplified - uses pip returns)
        if total_trades > 0:
            pips_list = [t.pips or 0 for t in trades]
            mean_pips = sum(pips_list) / len(pips_list)
            variance = sum((p - mean_pips) ** 2 for p in pips_list) / len(pips_list)
            std_dev = variance ** 0.5
            sharpe_ratio = (mean_pips / std_dev) if std_dev > 0 else 0.0
        else:
            sharpe_ratio = 0.0

        # Max drawdown (simplified - cumulative pip drawdown)
        cumulative_pips = 0
        peak_pips = 0
        max_drawdown = 0.0
        for trade in trades:
            cumulative_pips += trade.pips or 0
            peak_pips = max(peak_pips, cumulative_pips)
            drawdown = cumulative_pips - peak_pips
            max_drawdown = min(max_drawdown, drawdown)

        # Circuit breaker triggers
        breaker_query = db.query(CircuitBreakerEvent).filter(
            CircuitBreakerEvent.action == "triggered"
        )
        if start_time:
            breaker_query = breaker_query.filter(
                CircuitBreakerEvent.triggered_at >= start_time
            )
        circuit_breaker_triggers = breaker_query.count()

        return AgentMetricsResponse(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pips=total_pips,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            circuit_breaker_triggers=circuit_breaker_triggers,
            period=period,
        )

    except Exception as e:
        log_exception(logger, "Error getting agent metrics", e)
        raise HTTPException(status_code=500, detail=str(e))


# Command Status Endpoints


@router.get("/commands/{command_id}", response_model=CommandStatusResponse)
async def get_command_status(
    command_id: int,
    db: Session = Depends(get_db),
) -> CommandStatusResponse:
    """Get status of a specific command.

    Use this to check if a queued command has been processed and see the result.
    """
    try:
        cmd = db.query(AgentCommand).filter(AgentCommand.id == command_id).first()

        if not cmd:
            raise HTTPException(
                status_code=404,
                detail=f"Command {command_id} not found.",
            )

        return CommandStatusResponse(
            command_id=cmd.id,
            command=cmd.command,
            status=cmd.status,
            created_at=cmd.created_at.isoformat(),
            processed_at=cmd.processed_at.isoformat() if cmd.processed_at else None,
            result=cmd.result,
            error_message=cmd.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error getting command status", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/commands", response_model=CommandListResponse)
async def list_commands(
    limit: int = Query(default=20, le=100, description="Number of commands to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: pending, processing, completed, failed",
        pattern="^(pending|processing|completed|failed)$",
    ),
    db: Session = Depends(get_db),
) -> CommandListResponse:
    """List recent commands.

    Returns a paginated list of commands, optionally filtered by status.
    """
    try:
        # Build query
        query = db.query(AgentCommand)

        if status:
            query = query.filter(AgentCommand.status == status)

        # Get total count
        total = query.count()

        # Get commands
        commands = (
            query.order_by(AgentCommand.created_at.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        # Build response
        items = [
            CommandListItem(
                command_id=cmd.id,
                command=cmd.command,
                status=cmd.status,
                created_at=cmd.created_at.isoformat(),
                processed_at=cmd.processed_at.isoformat() if cmd.processed_at else None,
            )
            for cmd in commands
        ]

        return CommandListResponse(
            commands=items,
            count=len(items),
            total=total,
        )

    except Exception as e:
        log_exception(logger, "Error listing commands", e)
        raise HTTPException(status_code=500, detail=str(e))


# Health Check Endpoint


class AgentHealthResponse(BaseModel):
    """Response model for agent health check."""

    healthy: bool
    status: str
    components: Dict[str, Any]
    uptime_seconds: Optional[float]
    last_cycle_at: Optional[str]
    warnings: list[str]


@router.get("/health", response_model=AgentHealthResponse)
async def get_agent_health(db: Session = Depends(get_db)) -> AgentHealthResponse:
    """Get comprehensive agent health status.

    Checks all agent components and returns overall health status.
    Used for monitoring and alerting.

    Returns:
        - healthy: True if all critical components are healthy
        - status: Current agent status
        - components: Individual component health checks
        - uptime_seconds: How long the agent has been running
        - last_cycle_at: Timestamp of last trading cycle
        - warnings: List of non-critical issues
    """
    warnings = []
    components = {}

    try:
        # Get current agent state
        state = _get_agent_state(db)

        if not state:
            return AgentHealthResponse(
                healthy=False,
                status="not_initialized",
                components={
                    "database": {"healthy": True, "message": "Connected"},
                    "agent": {"healthy": False, "message": "Not initialized"},
                },
                uptime_seconds=None,
                last_cycle_at=None,
                warnings=["Agent has not been initialized"],
            )

        # Check database connection
        try:
            db.execute(text("SELECT 1"))
            components["database"] = {"healthy": True, "message": "Connected"}
        except Exception as e:
            components["database"] = {"healthy": False, "message": str(e)}
            warnings.append("Database connection issue")

        # Check agent status
        agent_healthy = state.status in ("running", "paused")
        components["agent"] = {
            "healthy": agent_healthy,
            "status": state.status,
            "mode": state.mode,
            "cycle_count": state.cycle_count,
        }

        if state.status == "error":
            warnings.append(f"Agent in error state: {state.error_message}")

        # Check safety systems
        safety_healthy = not state.kill_switch_active
        components["safety"] = {
            "healthy": safety_healthy,
            "kill_switch_active": state.kill_switch_active,
            "circuit_breaker_state": state.circuit_breaker_state,
        }

        if state.kill_switch_active:
            warnings.append("Kill switch is active")

        if state.circuit_breaker_state and state.circuit_breaker_state != "active":
            warnings.append(f"Circuit breaker state: {state.circuit_breaker_state}")

        # Check cycle freshness (if running, should have recent cycle)
        cycle_stale = False
        if state.status == "running" and state.last_cycle_at:
            seconds_since_cycle = (datetime.utcnow() - state.last_cycle_at).total_seconds()
            # Consider stale if no cycle in 5x the expected interval (default 60s)
            expected_interval = 60  # default
            if state.config and isinstance(state.config, dict):
                expected_interval = state.config.get("cycle_interval_seconds", 60)
            if seconds_since_cycle > expected_interval * 5:
                cycle_stale = True
                warnings.append(f"No trading cycle for {seconds_since_cycle:.0f}s")

        components["cycle_freshness"] = {
            "healthy": not cycle_stale,
            "seconds_since_last": (
                (datetime.utcnow() - state.last_cycle_at).total_seconds()
                if state.last_cycle_at else None
            ),
        }

        # Calculate uptime
        uptime_seconds = None
        if state.started_at and state.status in ("running", "paused"):
            uptime_seconds = (datetime.utcnow() - state.started_at).total_seconds()

        # Overall health is True if agent is running and no critical warnings
        overall_healthy = (
            agent_healthy
            and safety_healthy
            and not cycle_stale
            and components.get("database", {}).get("healthy", False)
        )

        return AgentHealthResponse(
            healthy=overall_healthy,
            status=state.status,
            components=components,
            uptime_seconds=uptime_seconds,
            last_cycle_at=state.last_cycle_at.isoformat() if state.last_cycle_at else None,
            warnings=warnings,
        )

    except Exception as e:
        log_exception(logger, "Error getting agent health", e)
        return AgentHealthResponse(
            healthy=False,
            status="error",
            components={"error": {"healthy": False, "message": str(e)}},
            uptime_seconds=None,
            last_cycle_at=None,
            warnings=[f"Health check failed: {str(e)}"],
        )


# Safety Endpoints


class SafetyStatusResponse(BaseModel):
    """Response model for safety status."""

    is_safe_to_trade: bool
    circuit_breakers: Dict[str, Any]
    kill_switch: Dict[str, Any]
    daily_metrics: Dict[str, Any]
    account_metrics: Dict[str, Any]


class ResetCodeResponse(BaseModel):
    """Response model for reset code generation."""

    reset_code: str
    expires_at: str
    message: str


class CircuitBreakerResetRequest(BaseModel):
    """Request model for circuit breaker reset."""

    breaker_name: str  # consecutive_loss, drawdown, model_degradation


@router.get("/safety", response_model=SafetyStatusResponse)
async def get_safety_status(db: Session = Depends(get_db)) -> SafetyStatusResponse:
    """Get detailed safety status.

    Returns the current state of all safety mechanisms including circuit breakers,
    kill switch, daily limits, and account metrics.
    """
    try:
        # Get current agent state
        state = _get_agent_state(db)

        if not state:
            raise HTTPException(
                status_code=404, detail="Agent not initialized. Start the agent first."
            )

        # Safety status is stored in agent state (updated by runner)
        # For now, return placeholder - full integration requires agent to be running
        return SafetyStatusResponse(
            is_safe_to_trade=not state.kill_switch_active,
            circuit_breakers={
                "overall_state": state.circuit_breaker_state or "active",
                "active_breakers": [],
                "can_trade": not state.kill_switch_active,
            },
            kill_switch={
                "is_active": state.kill_switch_active,
                "reason": None,
                "triggered_at": None,
            },
            daily_metrics={"trades": 0, "loss_pct": 0.0, "loss_amount": 0.0},
            account_metrics={
                "current_equity": state.account_equity or 100000.0,
                "peak_equity": 100000.0,
                "drawdown_pct": 0.0,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error getting safety status", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safety/kill-switch/reset-code", response_model=ResetCodeResponse)
async def get_kill_switch_reset_code(
    db: Session = Depends(get_db),
) -> ResetCodeResponse:
    """Generate reset authorization code for kill switch.

    Returns a code that is valid for 5 minutes. Use this code with the
    kill-switch reset endpoint to reset the kill switch after review.
    """
    try:
        # Check if agent initialized
        state = _get_agent_state(db)
        if not state:
            raise HTTPException(
                status_code=404, detail="Agent not initialized. Start the agent first."
            )

        if not state.kill_switch_active:
            raise HTTPException(
                status_code=400, detail="Kill switch is not active. Nothing to reset."
            )

        # Generate reset code (valid for 5 minutes)
        # In production, this would call safety_manager.get_reset_code()
        # For now, return placeholder
        import secrets
        from datetime import timedelta

        code = secrets.token_hex(4).upper()
        expires_at = datetime.utcnow() + timedelta(minutes=5)

        return ResetCodeResponse(
            reset_code=code,
            expires_at=expires_at.isoformat(),
            message="Use this code to reset the kill switch within 5 minutes.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error generating reset code", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/safety/circuit-breakers/reset", response_model=CommandResponse)
async def reset_circuit_breaker(
    request: CircuitBreakerResetRequest = Body(...),
    db: Session = Depends(get_db),
) -> CommandResponse:
    """Reset a specific circuit breaker.

    Use this to reset a triggered circuit breaker after reviewing the issue.
    Available breakers: consecutive_loss, drawdown, model_degradation.
    """
    try:
        # Check if agent initialized
        state = _get_agent_state(db)
        if not state:
            raise HTTPException(
                status_code=404, detail="Agent not initialized. Start the agent first."
            )

        # Validate breaker name
        valid_breakers = ["consecutive_loss", "drawdown", "model_degradation"]
        if request.breaker_name not in valid_breakers:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid breaker name. Must be one of: {', '.join(valid_breakers)}",
            )

        # Queue reset command
        payload = {"breaker_name": request.breaker_name}
        cmd = _queue_command(db, "reset_circuit_breaker", payload)

        return CommandResponse(
            status="queued",
            command_id=cmd.id,
            message=f"Circuit breaker '{request.breaker_name}' reset queued successfully.",
        )

    except HTTPException:
        raise
    except Exception as e:
        log_exception(logger, "Error resetting circuit breaker", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/safety/events")
async def get_safety_events(
    limit: int = Query(default=50, le=200, description="Number of events to return"),
    breaker_type: Optional[str] = Query(
        default=None, description="Filter by breaker type"
    ),
    severity: Optional[str] = Query(default=None, description="Filter by severity"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Get recent safety events (circuit breaker triggers, kill switch activations).

    Returns audit trail of all safety system activations.
    """
    try:
        # Build query
        query = db.query(CircuitBreakerEvent).order_by(
            CircuitBreakerEvent.triggered_at.desc()
        )

        if breaker_type:
            query = query.filter(CircuitBreakerEvent.breaker_type == breaker_type)

        if severity:
            query = query.filter(CircuitBreakerEvent.severity == severity)

        # Get events
        events = query.limit(limit).all()

        # Format response
        events_list = [
            {
                "id": event.id,
                "breaker_type": event.breaker_type,
                "severity": event.severity,
                "action": event.action,
                "reason": event.reason,
                "value": event.value,
                "threshold": event.threshold,
                "triggered_at": event.triggered_at.isoformat(),
                "recovered_at": (
                    event.recovered_at.isoformat() if event.recovered_at else None
                ),
            }
            for event in events
        ]

        return {"events": events_list, "count": len(events_list), "limit": limit}

    except Exception as e:
        log_exception(logger, "Error getting safety events", e)
        raise HTTPException(status_code=500, detail=str(e))

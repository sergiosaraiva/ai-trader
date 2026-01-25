"""Pydantic schemas for agent endpoints."""

from datetime import datetime
from typing import Dict, Any, List, Optional, Literal

from pydantic import BaseModel, Field


# Request Schemas


class AgentStartRequest(BaseModel):
    """Request to start the agent."""

    mode: Literal["simulation", "paper", "live"] = Field(
        default="simulation",
        description="Execution mode: simulation (no broker), paper (MT5 demo), or live (MT5 real)",
    )
    confidence_threshold: float = Field(
        default=0.70,
        ge=0.5,
        le=0.95,
        description="Minimum confidence to execute trades (0.5-0.95)",
    )
    cycle_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=3600,
        description="Seconds between agent cycles (10-3600)",
    )
    max_position_size: float = Field(
        default=0.1, gt=0, le=1.0, description="Maximum position size as fraction of equity (0-1)"
    )
    use_kelly_sizing: bool = Field(
        default=True, description="Use Kelly criterion for position sizing"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "mode": "simulation",
                "confidence_threshold": 0.70,
                "cycle_interval_seconds": 60,
                "max_position_size": 0.1,
                "use_kelly_sizing": True,
            }
        }
    }


class AgentStopRequest(BaseModel):
    """Request to stop the agent."""

    force: bool = Field(
        default=False, description="Force stop even if positions are open"
    )
    close_positions: bool = Field(
        default=False, description="Close open positions before stopping"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "force": False,
                "close_positions": False,
            }
        }
    }


class AgentConfigUpdateRequest(BaseModel):
    """Request to update agent configuration."""

    confidence_threshold: Optional[float] = Field(
        None, ge=0.5, le=0.95, description="Update confidence threshold"
    )
    cycle_interval_seconds: Optional[int] = Field(
        None, ge=10, le=3600, description="Update cycle interval"
    )
    max_position_size: Optional[float] = Field(
        None, gt=0, le=1.0, description="Update max position size"
    )
    use_kelly_sizing: Optional[bool] = Field(
        None, description="Update Kelly sizing setting"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "confidence_threshold": 0.75,
                "cycle_interval_seconds": 120,
            }
        }
    }


class KillSwitchRequest(BaseModel):
    """Request to trigger or reset kill switch."""

    action: Literal["trigger", "reset"] = Field(
        ..., description="Action: trigger to halt all trading, reset to restore"
    )
    reason: Optional[str] = Field(
        None, max_length=500, description="Reason for triggering kill switch"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "action": "trigger",
                "reason": "Unexpected market volatility - manual intervention required",
            }
        }
    }


# Response Schemas


class CommandResponse(BaseModel):
    """Response after queuing a command."""

    status: str = Field(..., description="Status: 'queued' or 'error'")
    command_id: int = Field(..., description="Unique command ID for tracking")
    message: str = Field(..., description="Human-readable message")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "queued",
                "command_id": 42,
                "message": "Start command queued successfully. Agent will process shortly.",
            }
        }
    }


class AgentStatusResponse(BaseModel):
    """Response for agent status query."""

    status: str = Field(
        ..., description="Agent status: stopped, starting, running, paused, stopping, error"
    )
    mode: str = Field(..., description="Execution mode: simulation, paper, live")
    cycle_count: int = Field(..., ge=0, description="Number of cycles executed")
    last_cycle_at: Optional[str] = Field(
        None, description="Timestamp of last cycle (ISO format)"
    )
    account_equity: Optional[float] = Field(
        None, ge=0, description="Current account equity"
    )
    open_positions: int = Field(..., ge=0, description="Number of open positions")
    circuit_breaker_state: Optional[str] = Field(
        None, description="Circuit breaker status (e.g., 'consecutive_loss')"
    )
    kill_switch_active: bool = Field(..., description="Whether kill switch is triggered")
    error_message: Optional[str] = Field(None, description="Error message if status is 'error'")
    uptime_seconds: Optional[float] = Field(None, ge=0, description="Seconds since agent started")
    last_prediction: Optional[Dict[str, Any]] = Field(
        None, description="Last prediction made by agent"
    )
    config: Dict[str, Any] = Field(..., description="Current agent configuration")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "running",
                "mode": "simulation",
                "cycle_count": 142,
                "last_cycle_at": "2024-01-15T14:30:00Z",
                "account_equity": 103450.00,
                "open_positions": 1,
                "circuit_breaker_state": None,
                "kill_switch_active": False,
                "error_message": None,
                "uptime_seconds": 8520.0,
                "last_prediction": {
                    "direction": "long",
                    "confidence": 0.72,
                    "should_trade": True,
                },
                "config": {
                    "confidence_threshold": 0.70,
                    "cycle_interval_seconds": 60,
                    "max_position_size": 0.1,
                    "use_kelly_sizing": True,
                },
            }
        }
    }


class AgentMetricsResponse(BaseModel):
    """Response for agent performance metrics."""

    total_trades: int = Field(..., ge=0, description="Total trades executed")
    winning_trades: int = Field(..., ge=0, description="Number of winning trades")
    losing_trades: int = Field(..., ge=0, description="Number of losing trades")
    win_rate: Optional[float] = Field(None, ge=0, le=1, description="Win rate (0-1)")
    total_pips: float = Field(..., description="Total profit/loss in pips")
    profit_factor: Optional[float] = Field(
        None, ge=0, description="Profit factor (gross profit / gross loss)"
    )
    sharpe_ratio: Optional[float] = Field(None, description="Risk-adjusted return")
    max_drawdown: Optional[float] = Field(None, le=0, description="Maximum drawdown")
    circuit_breaker_triggers: int = Field(
        ..., ge=0, description="Number of circuit breaker triggers"
    )
    period: str = Field(..., description="Metrics time period (e.g., 'all', '24h', '7d')")

    model_config = {
        "json_schema_extra": {
            "example": {
                "total_trades": 47,
                "winning_trades": 29,
                "losing_trades": 18,
                "win_rate": 0.617,
                "total_pips": 892.5,
                "profit_factor": 2.45,
                "sharpe_ratio": 3.2,
                "max_drawdown": -125.0,
                "circuit_breaker_triggers": 2,
                "period": "all",
            }
        }
    }


class CommandStatusResponse(BaseModel):
    """Response for command status query."""

    command_id: int = Field(..., description="Unique command ID")
    command: str = Field(..., description="Command name (e.g., 'start', 'stop')")
    status: str = Field(
        ..., description="Status: pending, processing, completed, failed"
    )
    created_at: str = Field(..., description="When command was created (ISO format)")
    processed_at: Optional[str] = Field(
        None, description="When command was processed (ISO format)"
    )
    result: Optional[Dict[str, Any]] = Field(None, description="Command execution result")
    error_message: Optional[str] = Field(
        None, description="Error message if command failed"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "command_id": 42,
                "command": "start",
                "status": "completed",
                "created_at": "2024-01-15T14:00:00Z",
                "processed_at": "2024-01-15T14:00:02Z",
                "result": {"agent_status": "running", "pid": 12345},
                "error_message": None,
            }
        }
    }


class CommandListItem(BaseModel):
    """Single item in command list."""

    command_id: int = Field(..., description="Unique command ID")
    command: str = Field(..., description="Command name")
    status: str = Field(..., description="Command status")
    created_at: str = Field(..., description="Creation timestamp (ISO format)")
    processed_at: Optional[str] = Field(None, description="Processing timestamp (ISO format)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "command_id": 42,
                "command": "start",
                "status": "completed",
                "created_at": "2024-01-15T14:00:00Z",
                "processed_at": "2024-01-15T14:00:02Z",
            }
        }
    }


class CommandListResponse(BaseModel):
    """Response for command list query."""

    commands: List[CommandListItem] = Field(..., description="List of commands")
    count: int = Field(..., ge=0, description="Number of commands returned")
    total: int = Field(..., ge=0, description="Total commands in database")

    model_config = {
        "json_schema_extra": {
            "example": {
                "commands": [
                    {
                        "command_id": 42,
                        "command": "start",
                        "status": "completed",
                        "created_at": "2024-01-15T14:00:00Z",
                        "processed_at": "2024-01-15T14:00:02Z",
                    }
                ],
                "count": 1,
                "total": 15,
            }
        }
    }

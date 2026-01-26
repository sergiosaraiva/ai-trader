"""Agent entry point with async main and health server.

Run with:
    python -m src.agent.main
"""

import asyncio
import logging
import signal
import sys
from typing import Optional

from aiohttp import web

from .config import AgentConfig
from .runner import AgentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

# Global agent runner instance
agent_runner: Optional[AgentRunner] = None
health_app: Optional[web.Application] = None
health_runner: Optional[web.AppRunner] = None


async def health_check(request: web.Request) -> web.Response:
    """Health check endpoint for container orchestration.

    Returns:
        200 if agent is running or paused
        503 if agent is stopped or in error state
    """
    if agent_runner is None:
        return web.Response(
            text="Agent not initialized",
            status=503,
        )

    status_dict = agent_runner.get_status()
    status = status_dict["status"]

    # Consider running and paused as healthy
    if status in ["running", "paused"]:
        return web.json_response(
            {
                "status": "healthy",
                "agent_status": status,
                "cycle_count": status_dict["cycle_count"],
                "model_loaded": status_dict["model_loaded"],
            },
            status=200,
        )
    else:
        return web.json_response(
            {
                "status": "unhealthy",
                "agent_status": status,
                "last_error": status_dict.get("last_error"),
            },
            status=503,
        )


async def status_endpoint(request: web.Request) -> web.Response:
    """Detailed status endpoint.

    Returns full agent status information.
    """
    if agent_runner is None:
        return web.json_response(
            {"error": "Agent not initialized"},
            status=503,
        )

    status_dict = agent_runner.get_status()
    return web.json_response(status_dict, status=200)


async def start_health_server(port: int) -> None:
    """Start the health check HTTP server.

    Args:
        port: Port to listen on
    """
    global health_app, health_runner

    health_app = web.Application()
    health_app.router.add_get("/health", health_check)
    health_app.router.add_get("/status", status_endpoint)

    health_runner = web.AppRunner(health_app)
    await health_runner.setup()

    site = web.TCPSite(health_runner, "0.0.0.0", port)
    await site.start()

    logger.info(f"Health server started on port {port}")


async def stop_health_server() -> None:
    """Stop the health check HTTP server."""
    global health_runner

    if health_runner:
        await health_runner.cleanup()
        logger.info("Health server stopped")


async def shutdown(sig: signal.Signals) -> None:
    """Handle shutdown signals.

    Args:
        sig: Signal that triggered shutdown
    """
    logger.info(f"Received signal {sig.name}, initiating shutdown...")

    # Stop agent
    if agent_runner:
        logger.info("Stopping agent...")
        await agent_runner.stop()

    # Stop health server
    await stop_health_server()

    # Cancel all remaining tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()

    await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("Shutdown complete")


async def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    global agent_runner

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = AgentConfig.from_env()
        logger.info(f"Configuration loaded: mode={config.mode}, threshold={config.confidence_threshold}")

        # Create agent runner
        agent_runner = AgentRunner(config)

        # Start health server
        await start_health_server(config.health_port)

        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown(s)),
            )

        # Start agent
        logger.info("Starting agent...")
        if not await agent_runner.start():
            logger.error("Failed to start agent")
            return 1

        logger.info("Agent running, press Ctrl+C to stop")

        # Keep running until stopped
        try:
            while agent_runner.status.value in ["running", "paused"]:
                await asyncio.sleep(1)

            # If we get here, agent stopped itself (error or command)
            status = agent_runner.get_status()
            if status["status"] == "error":
                logger.error(f"Agent stopped with error: {status.get('last_error')}")
                return 1

        except asyncio.CancelledError:
            logger.info("Main loop cancelled")

        return 0

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        return 0

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        if agent_runner:
            logger.info("Final cleanup...")
            await agent_runner.stop()
        await stop_health_server()


def run() -> None:
    """Entry point for running the agent.

    This function can be called from command line or other modules.
    """
    exit_code = asyncio.run(main())
    sys.exit(exit_code)


if __name__ == "__main__":
    run()

"""Quick integration test for Phase 4 agent implementation.

Tests that the trading cycle can execute without errors and integrates
properly with existing services.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent import AgentConfig, AgentRunner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def test_agent_phase4():
    """Test agent can start, run a few cycles, and stop."""
    logger.info("="*60)
    logger.info("Phase 4 Agent Integration Test")
    logger.info("="*60)

    # Create config
    config = AgentConfig(
        mode="simulation",
        confidence_threshold=0.70,
        cycle_interval_seconds=5,  # Fast cycles for testing
        initial_capital=100000.0,
    )

    logger.info(f"Config: {config.to_dict()}")

    # Create agent
    agent = AgentRunner(config)

    try:
        # Start agent
        logger.info("\n[1/4] Starting agent...")
        success = await agent.start()
        if not success:
            logger.error("Failed to start agent")
            return False

        logger.info("✓ Agent started successfully")

        # Let it run for 3 cycles (15 seconds)
        logger.info("\n[2/4] Running 3 trading cycles...")
        await asyncio.sleep(17)

        status = agent.get_status()
        logger.info(f"✓ Completed {status['cycle_count']} cycles")

        # Check status
        logger.info("\n[3/4] Checking agent status...")
        logger.info(f"Status: {status['status']}")
        logger.info(f"Cycles: {status['cycle_count']}")
        logger.info(f"Model loaded: {status['model_loaded']}")
        logger.info(f"Last error: {status['last_error']}")

        if status['cycle_count'] < 2:
            logger.warning("Expected at least 2 cycles")
            return False

        logger.info("✓ Agent running normally")

        # Stop agent
        logger.info("\n[4/4] Stopping agent...")
        success = await agent.stop()
        if not success:
            logger.error("Failed to stop agent")
            return False

        logger.info("✓ Agent stopped gracefully")

        logger.info("\n" + "="*60)
        logger.info("Phase 4 Integration Test: PASSED")
        logger.info("="*60)
        return True

    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        return False

    finally:
        # Ensure cleanup
        try:
            await agent.stop()
        except:
            pass


if __name__ == "__main__":
    result = asyncio.run(test_agent_phase4())
    sys.exit(0 if result else 1)

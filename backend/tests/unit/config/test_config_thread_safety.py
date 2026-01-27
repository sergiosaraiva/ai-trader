"""Thread safety tests for centralized trading configuration system.

Tests concurrent access patterns, lock contention, and race conditions.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

# Add backend src to path
backend_path = Path(__file__).parent.parent.parent.parent
src_path = backend_path / "src"
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import directly from module to avoid __init__.py dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "trading_config",
    src_path / "config" / "trading_config.py"
)
trading_config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trading_config_module)

TradingConfig = trading_config_module.TradingConfig


# ============================================================================
# CONCURRENT ACCESS TESTS
# ============================================================================


def test_singleton_thread_safety():
    """Test that singleton pattern is thread-safe."""
    instances = []

    def get_instance():
        """Get config instance from thread."""
        config = TradingConfig()
        instances.append(config)

    # Create 20 threads trying to get instance simultaneously
    threads = []
    for _ in range(20):
        thread = threading.Thread(target=get_instance)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # All instances should be the same object
    assert len(instances) == 20
    assert all(inst is instances[0] for inst in instances)


def test_concurrent_updates_different_categories():
    """Test concurrent updates to different categories."""
    config = TradingConfig()
    results = {"trading": [], "model": [], "risk": [], "system": []}

    def update_trading():
        """Update trading config."""
        for i in range(5):
            try:
                config.update(
                    category="trading",
                    updates={"confidence_threshold": 0.60 + i * 0.01},
                    updated_by="thread_trading",
                    db_session=None,
                )
                results["trading"].append(True)
            except Exception as e:
                results["trading"].append(False)

    def update_model():
        """Update model config."""
        for i in range(5):
            try:
                config.update(
                    category="model",
                    updates={"agreement_bonus": 0.03 + i * 0.01},
                    updated_by="thread_model",
                    db_session=None,
                )
                results["model"].append(True)
            except Exception as e:
                results["model"].append(False)

    def update_risk():
        """Update risk config."""
        for i in range(5):
            try:
                config.update(
                    category="risk",
                    updates={"max_consecutive_losses": 3 + i},
                    updated_by="thread_risk",
                    db_session=None,
                )
                results["risk"].append(True)
            except Exception as e:
                results["risk"].append(False)

    def update_system():
        """Update system config."""
        for i in range(5):
            try:
                config.update(
                    category="system",
                    updates={"cache_ttl_seconds": 50 + i * 10},
                    updated_by="thread_system",
                    db_session=None,
                )
                results["system"].append(True)
            except Exception as e:
                results["system"].append(False)

    # Run all updates concurrently
    threads = [
        threading.Thread(target=update_trading),
        threading.Thread(target=update_model),
        threading.Thread(target=update_risk),
        threading.Thread(target=update_system),
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # All updates should succeed
    assert all(results["trading"])
    assert all(results["model"])
    assert all(results["risk"])
    assert all(results["system"])


def test_concurrent_updates_same_category():
    """Test concurrent updates to same category."""
    config = TradingConfig()

    def update_config(thread_id):
        """Update config from thread."""
        try:
            for i in range(3):
                value = 0.60 + (thread_id * 0.01) + (i * 0.001)
                config.update(
                    category="trading",
                    updates={"confidence_threshold": value},
                    updated_by=f"thread_{thread_id}",
                    db_session=None,
                )
            return True
        except Exception as e:
            return False

    # Run 10 threads each doing 3 updates
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(update_config, i) for i in range(10)]
        results = [f.result() for f in as_completed(futures)]

    # All should succeed
    assert all(results)

    # Final value should be valid
    assert 0.60 <= config.trading.confidence_threshold <= 0.70


def test_concurrent_reads_and_writes():
    """Test concurrent reads don't block or get corrupted during writes."""
    config = TradingConfig()
    read_count = [0]  # Use list for mutability in closures
    write_count = [0]

    def reader():
        """Read config continuously."""
        for _ in range(100):
            val = config.trading.confidence_threshold
            # Verify value is always valid
            assert 0.0 <= val <= 1.0
            read_count[0] += 1
            time.sleep(0.001)

    def writer():
        """Write config continuously."""
        for i in range(20):
            val = 0.60 + (i % 10) * 0.01
            config.update(
                category="trading",
                updates={"confidence_threshold": val},
                updated_by="writer",
                db_session=None,
            )
            write_count[0] += 1
            time.sleep(0.005)

    # Start multiple readers and writers
    threads = []
    for _ in range(5):
        threads.append(threading.Thread(target=reader))
    for _ in range(2):
        threads.append(threading.Thread(target=writer))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # Verify operations completed
    assert read_count[0] == 500  # 5 readers * 100 reads
    assert write_count[0] == 40  # 2 writers * 20 writes


def test_concurrent_callback_execution():
    """Test that callbacks are executed safely during concurrent updates."""
    config = TradingConfig()
    callback_count = {"count": 0}
    lock = threading.Lock()

    def callback(params):
        """Callback that increments counter."""
        with lock:
            callback_count["count"] += 1
            # Simulate some work
            time.sleep(0.001)

    config.register_callback("trading", callback)

    def update_config(thread_id):
        """Update config to trigger callback."""
        try:
            config.update(
                category="trading",
                updates={"confidence_threshold": 0.60 + thread_id * 0.01},
                updated_by=f"thread_{thread_id}",
                db_session=None,
            )
            return True
        except Exception:
            return False

    # Run 10 concurrent updates
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(update_config, i) for i in range(10)]
        results = [f.result() for f in as_completed(futures)]

    # All updates should succeed
    assert all(results)

    # Callback should have been called 10 times
    assert callback_count["count"] == 10


# ============================================================================
# LOCK CONTENTION TESTS
# ============================================================================


def test_no_deadlock_on_nested_updates():
    """Test that nested updates don't cause deadlock."""
    config = TradingConfig()

    def outer_update():
        """Outer update operation."""
        config.update(
            category="trading",
            updates={"confidence_threshold": 0.75},
            updated_by="outer",
            db_session=None,
        )

    def inner_update():
        """Inner update operation."""
        config.update(
            category="model",
            updates={"agreement_bonus": 0.08},
            updated_by="inner",
            db_session=None,
        )

    # Run both concurrently
    thread1 = threading.Thread(target=outer_update)
    thread2 = threading.Thread(target=inner_update)

    thread1.start()
    thread2.start()

    # Join with timeout to detect deadlock
    thread1.join(timeout=5.0)
    thread2.join(timeout=5.0)

    # If we get here, no deadlock occurred
    assert not thread1.is_alive()
    assert not thread2.is_alive()


def test_lock_released_on_exception():
    """Test that locks are released even when update fails."""
    config = TradingConfig()

    def failing_update():
        """Update that will fail validation."""
        try:
            config.update(
                category="trading",
                updates={"confidence_threshold": 1.5},  # Invalid
                updated_by="test",
                db_session=None,
            )
        except ValueError:
            pass  # Expected

    def successful_update():
        """Update that should succeed."""
        config.update(
            category="trading",
            updates={"confidence_threshold": 0.75},
            updated_by="test",
            db_session=None,
        )

    # Run failing update
    thread1 = threading.Thread(target=failing_update)
    thread1.start()
    thread1.join()

    # Successful update should work (lock was released)
    thread2 = threading.Thread(target=successful_update)
    thread2.start()
    thread2.join(timeout=5.0)

    # Should complete without blocking
    assert not thread2.is_alive()
    assert config.trading.confidence_threshold == 0.75


def test_high_contention_stress_test():
    """Stress test with high contention on same resource."""
    config = TradingConfig()
    success_count = [0]
    failure_count = [0]
    lock = threading.Lock()

    def stress_update(thread_id):
        """Perform multiple rapid updates."""
        for i in range(10):
            try:
                value = 0.50 + ((thread_id * 10 + i) % 40) * 0.01
                config.update(
                    category="trading",
                    updates={"confidence_threshold": value},
                    updated_by=f"thread_{thread_id}",
                    db_session=None,
                )
                with lock:
                    success_count[0] += 1
            except Exception as e:
                with lock:
                    failure_count[0] += 1

    # Run 20 threads each doing 10 updates = 200 total operations
    threads = []
    for i in range(20):
        thread = threading.Thread(target=stress_update, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join(timeout=30.0)  # Generous timeout

    # All threads should complete
    assert all(not t.is_alive() for t in threads)

    # Most or all updates should succeed
    assert success_count[0] + failure_count[0] == 200
    assert success_count[0] >= 190  # Allow some failures under extreme contention


# ============================================================================
# RACE CONDITION TESTS
# ============================================================================


def test_no_race_in_validation():
    """Test that validation doesn't have race conditions."""
    config = TradingConfig()

    results = []

    def update_with_validation(valid):
        """Update with valid or invalid value."""
        try:
            value = 0.75 if valid else 1.5
            config.update(
                category="trading",
                updates={"confidence_threshold": value},
                updated_by="test",
                db_session=None,
            )
            results.append(("success", value))
        except ValueError:
            results.append(("failed", value))

    # Run mix of valid and invalid updates
    threads = []
    for i in range(20):
        valid = (i % 2 == 0)
        thread = threading.Thread(target=update_with_validation, args=(valid,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Check results
    successes = [r for r in results if r[0] == "success"]
    failures = [r for r in results if r[0] == "failed"]

    # All invalid updates should fail
    assert all(r[1] == 1.5 for r in failures)
    # All valid updates should succeed
    assert all(r[1] == 0.75 for r in successes)


def test_no_race_in_callback_registration():
    """Test that callback registration is thread-safe."""
    config = TradingConfig()
    callback_executions = []
    lock = threading.Lock()

    def create_and_register_callback(callback_id):
        """Create and register a callback."""
        def callback(params):
            with lock:
                callback_executions.append(callback_id)

        config.register_callback("trading", callback)

    # Register 10 callbacks concurrently
    threads = []
    for i in range(10):
        thread = threading.Thread(target=create_and_register_callback, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # Trigger update to execute all callbacks
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="test",
        db_session=None,
    )

    # All 10 callbacks should have been executed
    assert len(callback_executions) == 10
    assert set(callback_executions) == set(range(10))


def test_no_race_in_get_all():
    """Test that get_all() is safe during concurrent updates."""
    config = TradingConfig()
    snapshots = []

    def reader():
        """Read full config repeatedly."""
        for _ in range(20):
            snapshot = config.get_all()
            snapshots.append(snapshot)
            time.sleep(0.001)

    def writer():
        """Update config repeatedly."""
        for i in range(10):
            config.update(
                category="trading",
                updates={"confidence_threshold": 0.60 + i * 0.01},
                updated_by="writer",
                db_session=None,
            )
            time.sleep(0.002)

    # Run reader and writer concurrently
    reader_thread = threading.Thread(target=reader)
    writer_thread = threading.Thread(target=writer)

    reader_thread.start()
    writer_thread.start()

    reader_thread.join()
    writer_thread.join()

    # All snapshots should be valid (no corrupted data)
    assert len(snapshots) == 20
    for snapshot in snapshots:
        assert "trading" in snapshot
        assert "model" in snapshot
        assert "risk" in snapshot
        assert "system" in snapshot
        # Confidence should always be valid
        assert 0.0 <= snapshot["trading"]["confidence_threshold"] <= 1.0


# ============================================================================
# REENTRANT LOCK TESTS
# ============================================================================


def test_reentrant_lock_allows_nested_calls():
    """Test that reentrant lock allows nested update calls."""
    config = TradingConfig()

    nested_call_succeeded = [False]

    def callback_with_nested_update(params):
        """Callback that triggers another update."""
        # This would deadlock with a non-reentrant lock
        try:
            config.update(
                category="model",
                updates={"agreement_bonus": 0.08},
                updated_by="nested",
                db_session=None,
            )
            nested_call_succeeded[0] = True
        except Exception:
            pass

    config.register_callback("trading", callback_with_nested_update)

    # Trigger outer update which will call callback with nested update
    config.update(
        category="trading",
        updates={"confidence_threshold": 0.75},
        updated_by="outer",
        db_session=None,
    )

    # Nested call should have succeeded
    assert nested_call_succeeded[0]
    assert config.model.agreement_bonus == 0.08


# ============================================================================
# PERFORMANCE UNDER CONCURRENCY
# ============================================================================


def test_performance_degradation_under_load():
    """Test that performance doesn't degrade severely under high load."""
    config = TradingConfig()

    def timed_updates(count):
        """Perform updates and measure time."""
        start = time.time()
        for i in range(count):
            config.update(
                category="trading",
                updates={"confidence_threshold": 0.60 + (i % 30) * 0.01},
                updated_by="perf_test",
                db_session=None,
            )
        return time.time() - start

    # Sequential baseline
    sequential_time = timed_updates(50)

    # Concurrent with 5 threads (10 updates each = 50 total)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(timed_updates, 10) for _ in range(5)]
        concurrent_times = [f.result() for f in as_completed(futures)]

    max_concurrent_time = max(concurrent_times)

    # Concurrent should not be more than 3x slower than sequential
    # (allows for lock contention overhead)
    assert max_concurrent_time < sequential_time * 3

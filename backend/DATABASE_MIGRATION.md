# PostgreSQL Migration Guide

## Overview

This guide documents the migration from SQLite to PostgreSQL for the AI Trader system. PostgreSQL provides better performance, concurrency, and scalability for production deployments.

## Security Improvements Implemented

### 1. Connection Security
- ✅ Environment-based configuration with no hardcoded credentials
- ✅ Production environment detection and validation
- ✅ Warning messages for development defaults
- ✅ Connection pool configuration based on deployment environment

### 2. SQL Injection Prevention
- ✅ Parameterized queries for all data operations
- ✅ DDL abstraction for schema modifications
- ✅ No string concatenation in SQL statements

### 3. Transaction Safety
- ✅ Automatic rollback on migration failures
- ✅ Atomic migration operations with `engine.begin()`
- ✅ Proper error handling and logging

### 4. Database Integrity
- ✅ Foreign key constraints properly defined
- ✅ Cascading deletes configured (SET NULL for predictions)
- ✅ Proper indexes for query performance

## Migration Steps

### 1. Local Development Setup

```bash
# Start PostgreSQL with Docker Compose
docker-compose up -d postgres

# Wait for PostgreSQL to be ready
docker-compose exec postgres pg_isready -U aitrader

# Run migration script
cd backend
python scripts/migrate_to_postgres.py

# Or skip data migration (schema only)
python scripts/migrate_to_postgres.py --skip-data

# Validate migration
python scripts/migrate_to_postgres.py --validate-only
```

### 2. Production Deployment

#### Environment Variables (Required)

```bash
# Production database URL (use strong password!)
DATABASE_URL=postgresql://username:STRONG_PASSWORD@host:5432/dbname

# Set environment to production
ENVIRONMENT=production

# For Railway deployment
RAILWAY_ENVIRONMENT=production
```

#### Security Checklist

- [ ] Change default PostgreSQL password
- [ ] Use SSL/TLS for database connections
- [ ] Configure firewall rules for database access
- [ ] Create read-only user for reporting
- [ ] Enable query logging for audit
- [ ] Set up regular backups
- [ ] Configure connection limits

### 3. Connection Pooling

The system automatically configures connection pooling based on the environment:

| Environment | Pool Type | Configuration |
|-------------|-----------|---------------|
| Railway/Serverless | NullPool | No persistent connections |
| Local/Persistent | QueuePool | pool_size=5, max_overflow=10 |

### 4. Migration Features

The migration system includes:

- **Automatic schema creation**: Creates all tables with proper constraints
- **Data migration**: Transfers existing SQLite data to PostgreSQL
- **Column mapping**: Handles schema differences gracefully
- **Validation**: Verifies migration success with read/write tests
- **Rollback support**: Automatic rollback on migration failure

## Database Schema Updates

### New Tables for Agent System

1. **agent_commands**: Command queue from backend to agent
2. **agent_state**: Current agent status and configuration
3. **trade_explanations**: LLM-generated trade explanations
4. **circuit_breaker_events**: Safety system audit trail

### Enhanced Columns

#### Predictions Table
- `used_by_agent`: Tracks agent usage
- `agent_cycle_number`: Links to agent cycles
- `should_trade`: Pre-computed trading threshold (70% confidence)

#### Trades Table
- `execution_mode`: simulation/paper/live
- `broker`: Trading platform identifier
- `mt5_ticket`: MetaTrader 5 order number
- `explanation_id`: Link to trade explanation

## Performance Optimizations

### Indexes Created

```sql
-- Predictions
CREATE INDEX idx_predictions_timestamp_symbol ON predictions(timestamp, symbol);
CREATE INDEX idx_predictions_agent_cycle ON predictions(agent_cycle_number);

-- Trades
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);
CREATE INDEX idx_trades_execution_mode ON trades(execution_mode);

-- Agent tables
CREATE INDEX idx_agent_commands_status ON agent_commands(status);
CREATE INDEX idx_agent_state_updated ON agent_state(updated_at);
CREATE INDEX idx_circuit_breaker_triggered ON circuit_breaker_events(triggered_at);
```

## Monitoring

### Health Checks

```bash
# Check database connection
curl http://localhost:8001/health

# PostgreSQL health check (Docker)
docker-compose exec postgres pg_isready -U aitrader

# Check connection pool status
docker-compose logs backend | grep "pool"
```

### Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Connection refused | Check PostgreSQL is running: `docker-compose ps` |
| Authentication failed | Verify DATABASE_URL matches docker-compose.yml |
| Too many connections | Restart backend to reset pool: `docker-compose restart backend` |
| Migration fails | Check logs: `docker-compose logs postgres` |
| Performance slow | Run `ANALYZE` on tables: `docker-compose exec postgres psql -U aitrader -c "ANALYZE;"` |

## Rollback Plan

If issues occur, you can rollback to SQLite:

1. Stop the backend: `docker-compose stop backend`
2. Update DATABASE_URL to SQLite path
3. Restart backend: `docker-compose up -d backend`

## Security Notes

⚠️ **IMPORTANT SECURITY REMINDERS**:

1. **Never commit .env files** with real credentials
2. **Always use strong passwords** in production
3. **Enable SSL/TLS** for database connections in production
4. **Restrict database access** to application servers only
5. **Regular backups** are essential for data recovery
6. **Monitor for suspicious queries** in production logs
7. **Use read-only replicas** for reporting queries

## Testing the Migration

```bash
# Run backend tests with PostgreSQL
cd backend
pytest tests/ -v --db postgresql

# Test specific database operations
pytest tests/api/database/ -v

# Performance comparison
python scripts/benchmark_db_performance.py
```

## Next Steps

After successful migration:

1. ✅ Update production deployment configs
2. ✅ Set up automated backups
3. ✅ Configure monitoring alerts
4. ✅ Document recovery procedures
5. ✅ Train team on PostgreSQL operations

---

*Migration implemented with security-first approach following OWASP guidelines.*
# AI Assets Trader - Architecture Overview

## 1. Executive Summary

This document outlines the architecture for an AI-powered trading system capable of analyzing and trading multiple asset classes, starting with forex. The system employs a multi-layered analysis approach combining Technical, Fundamental, and Sentimental analysis, with independent models that can be combined for final predictions.

## 2. System Vision

### 2.1 Core Objectives
- Build modular, independent analysis models (Technical, Fundamental, Sentimental)
- Implement multi-timeframe predictions (Short, Medium, Long-term)
- Create a robust ensemble mechanism to combine predictions
- Provide simulation/backtesting capabilities before live trading
- Design for production scalability and reliability

### 2.2 Supported Asset Classes
- **Phase 1**: Forex (currency pairs)
- **Phase 2**: Cryptocurrencies
- **Phase 3**: Stocks, ETFs, Commodities

## 3. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AI ASSETS TRADER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        PRESENTATION LAYER                            │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │  Dashboard  │  │  API/REST   │  │   Alerts    │  │  Reports   │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        ORCHESTRATION LAYER                           │    │
│  │  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐   │    │
│  │  │ Trading Engine  │  │ Risk Management  │  │ Position Manager  │   │    │
│  │  └─────────────────┘  └──────────────────┘  └───────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     ENSEMBLE & PREDICTION LAYER                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │              Meta-Model / Ensemble Combiner                  │    │    │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │    │    │
│  │  │  │Weighted Avg  │ │Stacking Model│ │Attention-based Fusion│ │    │    │
│  │  │  └──────────────┘ └──────────────┘ └──────────────────────┘ │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │           CONFIDENCE & UNCERTAINTY ESTIMATION                │    │    │
│  │  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────────┐ │    │    │
│  │  │  │Beta Output   │ │Calibration   │ │Position Sizing       │ │    │    │
│  │  │  │(Learned)     │ │(Temperature) │ │(Confidence-based)    │ │    │    │
│  │  │  └──────────────┘ └──────────────┘ └──────────────────────┘ │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        ANALYSIS LAYER                                │    │
│  │                                                                       │    │
│  │  ┌───────────────────┐ ┌───────────────────┐ ┌───────────────────┐  │    │
│  │  │ TECHNICAL ANALYSIS│ │FUNDAMENTAL ANALYSIS│ │SENTIMENT ANALYSIS│  │    │
│  │  │                   │ │                    │ │                   │  │    │
│  │  │ ┌───────────────┐ │ │ ┌────────────────┐ │ │ ┌───────────────┐ │  │    │
│  │  │ │ Short-Term    │ │ │ │ Economic Data  │ │ │ │ News Analysis │ │  │    │
│  │  │ │ Model (1H-4H) │ │ │ │ Processing     │ │ │ │ (NLP/LLM)     │ │  │    │
│  │  │ └───────────────┘ │ │ └────────────────┘ │ │ └───────────────┘ │  │    │
│  │  │ ┌───────────────┐ │ │ ┌────────────────┐ │ │ ┌───────────────┐ │  │    │
│  │  │ │ Medium-Term   │ │ │ │ Interest Rates │ │ │ │ Social Media  │ │  │    │
│  │  │ │ Model (D-W)   │ │ │ │ Analysis       │ │ │ │ Sentiment     │ │  │    │
│  │  │ └───────────────┘ │ │ └────────────────┘ │ │ └───────────────┘ │  │    │
│  │  │ ┌───────────────┐ │ │ ┌────────────────┐ │ │ ┌───────────────┐ │  │    │
│  │  │ │ Long-Term     │ │ │ │ GDP/Inflation  │ │ │ │ Market Fear/  │ │  │    │
│  │  │ │ Model (W-M)   │ │ │ │ Forecasting    │ │ │ │ Greed Index   │ │  │    │
│  │  │ └───────────────┘ │ │ └────────────────┘ │ │ └───────────────┘ │  │    │
│  │  └───────────────────┘ └───────────────────┘ └───────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                    │                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          DATA LAYER                                  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │Market Data  │  │ Historical  │  │  News/      │  │  Economic  │  │    │
│  │  │  Feeds      │  │   Data      │  │  Social     │  │ Calendar   │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 4. Technology Stack

### 4.1 Core Technologies
| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.11+ | Industry standard for ML/AI, rich ecosystem |
| Deep Learning | PyTorch / PyTorch Lightning | Flexibility, research-friendly, production-ready |
| Time Series | Darts, PyTorch Forecasting | Specialized time series libraries |
| Technical Indicators | TA-Lib, pandas-ta | 200+ indicators, battle-tested |
| Experiment Tracking | MLflow | Model versioning, deployment, tracking |
| Data Processing | Pandas, Polars | High-performance data manipulation |
| API Framework | FastAPI | High-performance async API |
| Message Queue | Redis / RabbitMQ | Real-time data streaming |
| Database | PostgreSQL + TimescaleDB | Time-series optimized storage |
| Containerization | Docker + Kubernetes | Scalable deployment |

### 4.2 ML/AI Framework Selection
| Use Case | Primary | Alternative |
|----------|---------|-------------|
| Time Series Forecasting | Temporal Fusion Transformer | N-BEATS, DeepAR |
| Sequence Modeling | LSTM + Attention | Transformer, TCN |
| Ensemble Learning | XGBoost, LightGBM | CatBoost, Random Forest |
| NLP/Sentiment | FinBERT, GPT-4 API | RoBERTa, BERT |

## 5. Data Pipeline Architecture

### 5.1 Data Sources
```yaml
Market Data:
  - MetaTrader 5 (via Python API)
  - Alpaca Markets (stocks, crypto)
  - OANDA (forex)
  - Yahoo Finance (backup/historical)

Economic Data:
  - Federal Reserve Economic Data (FRED)
  - World Bank API
  - Trading Economics

Sentiment Data:
  - Twitter/X API
  - Reddit API
  - News APIs (NewsAPI, Benzinga)
  - Financial news feeds
```

### 5.2 Data Flow
```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw Data    │───▶│  Ingestion   │───▶│  Processing  │───▶│  Feature     │
│  Sources     │    │  Service     │    │  Pipeline    │    │  Store       │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                    ┌──────────────┐    ┌──────────────┐           │
                    │  Model       │◀───│  Training    │◀──────────┘
                    │  Registry    │    │  Pipeline    │
                    └──────────────┘    └──────────────┘
                           │
                    ┌──────────────┐    ┌──────────────┐
                    │  Inference   │───▶│  Trading     │
                    │  Service     │    │  Engine      │
                    └──────────────┘    └──────────────┘
```

## 6. Module Breakdown

### 6.1 Technical Analysis Module (Phase 1 - Current Focus)
See: `02-technical-analysis-model-design.md`

### 6.2 Fundamental Analysis Module (Phase 2)
- Economic indicator processing
- Interest rate differential analysis
- GDP/Inflation correlation models
- Central bank policy analysis

### 6.3 Sentiment Analysis Module (Phase 3)
- News sentiment extraction (FinBERT)
- Social media sentiment aggregation
- Market fear/greed indicators
- Event-driven analysis

### 6.4 Ensemble/Meta-Model (Phase 4)
- Model output aggregation
- Dynamic weight adjustment
- Confidence calibration
- Final prediction generation

### 6.5 Confidence & Uncertainty System (Critical)

The system uses **learned uncertainty** with Beta distribution outputs instead of traditional sigmoid.

**Key Components:**
- `src/models/confidence/learned_uncertainty.py` - Beta/Dirichlet/Gaussian output layers
- `src/models/confidence/calibration.py` - Temperature, Platt, Isotonic scaling
- `src/models/confidence/uncertainty.py` - MC Dropout, Ensemble disagreement
- `src/models/confidence/integration.py` - ConfidenceAwarePredictor

**Why Beta Distribution Instead of Sigmoid:**

| Aspect | Sigmoid | Beta Distribution |
|--------|---------|-------------------|
| Output | Single value 0-1 | Distribution Beta(α, β) |
| Confidence | Derived from distance to 0.5 | **Learned** via concentration (α+β) |
| "I don't know" | Cannot express | Low concentration = uncertain |
| Position Sizing | Based on guess | Based on learned confidence |

**Confidence-Based Position Sizing:**
```
Confidence ≥ 90%  →  100% position (full)
Confidence 80-90% →  75% position
Confidence 70-80% →  50% position
Confidence 60-70% →  25% position
Confidence < 60%  →  NO TRADE
```

See `docs/04-confidence-uncertainty-system.md` for full specification.

## 7. Operational Modes

### 7.1 Simulation Mode (Backtesting)
```python
class SimulationMode:
    """
    - Historical data replay
    - Paper trading execution
    - Performance metrics calculation
    - Strategy optimization
    - Walk-forward validation
    """
```

### 7.2 Paper Trading Mode
```python
class PaperTradingMode:
    """
    - Real-time market data
    - Simulated order execution
    - Real-time P&L tracking
    - Risk metric monitoring
    """
```

### 7.3 Live Trading Mode
```python
class LiveTradingMode:
    """
    - Real-time predictions
    - Actual order execution
    - Position management
    - Risk controls (stop-loss, take-profit)
    """
```

## 8. Risk Management Framework

### 8.1 Pre-Trade Risk Checks
- Position size limits
- Maximum drawdown limits
- Correlation checks
- Volatility-adjusted sizing

### 8.2 Real-Time Monitoring
- P&L tracking
- Exposure monitoring
- Margin utilization
- Slippage analysis

### 8.3 Post-Trade Analysis
- Performance attribution
- Strategy drift detection
- Model degradation alerts

## 9. Project Structure

```
ai-trader/
├── docs/                          # Documentation
│   ├── 01-architecture-overview.md
│   ├── 02-technical-analysis-model-design.md
│   └── 03-api-reference.md
├── src/
│   ├── __init__.py
│   ├── config/                    # Configuration management
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   └── model_config.py
│   ├── data/                      # Data layer
│   │   ├── __init__.py
│   │   ├── sources/              # Data source connectors
│   │   │   ├── mt5.py
│   │   │   ├── alpaca.py
│   │   │   └── yahoo.py
│   │   ├── processors/           # Data preprocessing
│   │   │   ├── ohlcv.py
│   │   │   └── features.py
│   │   └── storage/              # Data persistence
│   │       ├── database.py
│   │       └── cache.py
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   ├── technical/            # Technical indicators
│   │   │   ├── trend.py
│   │   │   ├── momentum.py
│   │   │   ├── volatility.py
│   │   │   └── volume.py
│   │   ├── fundamental/          # Fundamental features
│   │   └── sentiment/            # Sentiment features
│   ├── models/                    # ML Models
│   │   ├── __init__.py
│   │   ├── technical/            # Technical analysis models
│   │   │   ├── base.py
│   │   │   ├── short_term.py
│   │   │   ├── medium_term.py
│   │   │   └── long_term.py
│   │   ├── confidence/           # Confidence & uncertainty (CRITICAL)
│   │   │   ├── calibration.py    # Temperature, Platt, Isotonic
│   │   │   ├── uncertainty.py    # MC Dropout, Ensemble
│   │   │   ├── learned_uncertainty.py  # Beta/Dirichlet (RECOMMENDED)
│   │   │   └── integration.py    # ConfidenceAwarePredictor
│   │   ├── fundamental/
│   │   ├── sentiment/
│   │   └── ensemble/             # Ensemble models
│   │       ├── combiner.py
│   │       └── meta_model.py
│   ├── trading/                   # Trading engine
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   ├── execution.py
│   │   ├── risk.py
│   │   └── position.py
│   ├── simulation/                # Backtesting & simulation
│   │   ├── __init__.py
│   │   ├── backtester.py
│   │   ├── paper_trading.py
│   │   └── metrics.py
│   └── api/                       # REST API
│       ├── __init__.py
│       ├── main.py
│       └── routes/
├── notebooks/                     # Jupyter notebooks for research
├── tests/                         # Test suite
├── scripts/                       # Utility scripts
├── mlruns/                        # MLflow experiment tracking
├── docker/                        # Docker configurations
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 10. Development Phases

### Phase 1: Technical Analysis Foundation (Current)
- [ ] Data pipeline for forex (OHLCV data)
- [ ] Feature engineering (200+ technical indicators)
- [ ] Short-term prediction model
- [ ] Medium-term prediction model
- [ ] Long-term prediction model
- [ ] Model ensemble for technical analysis
- [ ] Backtesting framework
- [ ] Basic simulation mode

### Phase 2: Production Hardening
- [ ] Paper trading mode
- [ ] Risk management framework
- [ ] Performance monitoring
- [ ] Model versioning with MLflow
- [ ] API development
- [ ] Dashboard MVP

### Phase 3: Fundamental Analysis
- [ ] Economic data integration
- [ ] Fundamental analysis models
- [ ] Integration with ensemble

### Phase 4: Sentiment Analysis
- [ ] News sentiment pipeline
- [ ] Social media integration
- [ ] Sentiment models
- [ ] Full ensemble integration

### Phase 5: Live Trading
- [ ] Broker integration
- [ ] Live trading mode
- [ ] Advanced risk controls
- [ ] Production deployment

## 11. Success Metrics

### Model Performance
- Directional Accuracy: > 55%
- Sharpe Ratio: > 1.5
- Maximum Drawdown: < 15%
- Win Rate: > 50%

### System Performance
- Prediction Latency: < 100ms
- Data Pipeline Latency: < 1s
- System Uptime: > 99.9%

## 12. Next Steps

1. Review and approve this architecture
2. Begin implementation of Technical Analysis module
3. Set up development environment
4. Implement data pipeline for forex data
5. Start with short-term prediction model

---

*Document Version: 1.1*
*Last Updated: 2026-01-08*
*Author: AI Trader Development Team*

### Changelog
- **v1.1** (2026-01-08): Added Confidence & Uncertainty System section, updated architecture diagram with confidence layer
- **v1.0** (2025-01-06): Initial document

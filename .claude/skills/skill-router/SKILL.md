---
name: routing-to-skills
description: Routes tasks to appropriate skills by analyzing file paths, task types, and domain keywords. Returns ranked skill recommendations with confidence scores. Use when starting any implementation task to discover relevant skills, or when agents need to dynamically select skills for a workflow step.
---

# Routing to Skills

A meta-skill that enables agents to dynamically discover and invoke appropriate skills based on task context.

## Quick Reference

```
1. analyzeContext(task, files) → Extract layer, task type, keywords
2. discoverSkills()            → Scan .claude/skills/ for available skills
3. scoreSkills(context, skills)→ Rank by relevance (0-100 points)
4. invokeSkill(skill, context) → Load and apply selected skill
```

## When to Use

- Starting any implementation task (auto-route to best skill)
- Agent workflow step needs skill-based guidance
- Multiple skills might apply (need ranking)
- Task description is ambiguous about which pattern to follow
- Post-design phase (Solution Architect → Test Automator handoff)

## When NOT to Use

- Task explicitly names which skill to use
- Simple tasks requiring no specialized patterns
- Research/exploration tasks (not implementation)

---

## Context Analysis

### Layer Detection from File Paths

| Path Pattern | Layer | Candidate Skills |
|-------------|-------|------------------|
| `src/models/**` | Model | implementing-prediction-models |
| `src/api/**` | API | creating-api-endpoints |
| `src/features/technical/**` | Features | creating-technical-indicators |
| `src/data/sources/**` | Data | adding-data-sources |
| `src/data/processors/**` | Data | creating-data-processors |
| `src/trading/risk**` | Trading | implementing-risk-management |
| `src/simulation/**` | Simulation | running-backtests, analyzing-trading-performance |
| `configs/indicators/**` | Config | configuring-indicator-yaml |
| `tests/**` | Testing | validating-time-series-data |

### Task Type Detection from Keywords

| Keywords | Task Type | Skill Boost |
|----------|-----------|-------------|
| fix, bug, error, broken | bug_fix | +10 to domain skill |
| add, create, new, implement | feature | +20 to pattern skill |
| refactor, improve, clean | refactor | +15 to pattern skill |
| test, validate, verify | testing | +30 to testing skills |
| config, configure, setup | config | +25 to config skills |
| migrate, upgrade, update | migration | +10 to affected layer |

### Domain Keyword Extraction

| Domain | Keywords | Primary Skills |
|--------|----------|----------------|
| Model/ML | model, train, predict, neural, CNN, LSTM | implementing-prediction-models |
| Indicator | indicator, RSI, MACD, EMA, bollinger | creating-technical-indicators |
| Risk | risk, position, drawdown, circuit | implementing-risk-management |
| Backtest | backtest, simulate, walk-forward | running-backtests |
| Data | source, fetch, OHLCV, candle | adding-data-sources, creating-data-processors |
| API | endpoint, route, request, response | creating-api-endpoints |
| Metrics | sharpe, sortino, performance | analyzing-trading-performance |

---

## Skill Discovery

### Directory Structure

```
.claude/skills/
├── skill-router/           # This meta-skill
├── backend/                # Model, API, processor skills
├── feature-engineering/    # Indicator, config skills
├── data-layer/             # Data source, OHLCV skills
├── trading-domain/         # Backtest, risk, metrics skills
└── quality-testing/        # Dataclass, validation skills
```

### Parsing SKILL.md Metadata

Extract from each skill:
```yaml
name: skill-name              # For routing reference
description: ...              # Contains "when to use" hints
```

Key extraction patterns:
- "Use when..." → Trigger conditions
- "creating/implementing/adding" → Action verbs
- File patterns mentioned → Path matches
- Technology mentioned → Stack matches

---

## Scoring Algorithm

### Point System (0-100 scale)

```
SCORING WEIGHTS:
  file_path_match:    +50 points  # Strongest signal
  task_type_match:    +30 points  # Strong signal
  keyword_match:      +10 points  # Per keyword (max 3)
  recent_success:     +5 points   # If skill worked before in session

CONFIDENCE THRESHOLDS:
  High:   >= 80 points  # Strong recommendation
  Medium: 50-79 points  # Consider this skill
  Low:    < 50 points   # Weak match, verify fit
```

### Scoring Function (Pseudocode)

```python
def score_skill(skill, context):
    score = 0
    reasons = []

    # File path matching (+50)
    for file in context.files:
        if skill.path_patterns.match(file):
            score += 50
            reasons.append(f"File path matches {skill.layer} layer")
            break

    # Task type matching (+30)
    if context.task_type in skill.task_types:
        score += 30
        reasons.append(f"Task type '{context.task_type}' matches skill")

    # Keyword matching (+10 each, max 30)
    matched_keywords = skill.keywords & context.keywords
    keyword_score = min(len(matched_keywords) * 10, 30)
    score += keyword_score
    if matched_keywords:
        reasons.append(f"Keywords: {', '.join(matched_keywords)}")

    # Recent success bonus (+5)
    if skill.name in context.session_successes:
        score += 5
        reasons.append("Previously successful in session")

    return {
        "skill": skill.name,
        "score": score,
        "confidence": score / 100,
        "reasons": reasons
    }
```

---

## Routing Decision Tree

```
START: Receive task + files + context
│
├─ Step 1: Analyze Context
│   ├─ Extract file paths → Determine layers
│   ├─ Parse task description → Identify task type
│   └─ Extract keywords → Build keyword set
│
├─ Step 2: Discover Skills
│   ├─ Scan .claude/skills/**/SKILL.md
│   ├─ Parse YAML frontmatter (name, description)
│   └─ Build skill registry with metadata
│
├─ Step 3: Score Each Skill
│   ├─ Apply file path scoring (+50)
│   ├─ Apply task type scoring (+30)
│   ├─ Apply keyword scoring (+10 each)
│   └─ Apply session history bonus (+5)
│
├─ Step 4: Rank and Select
│   ├─ Sort by score descending
│   ├─ Return top 3 with confidence > 0.3
│   └─ Include reasons for transparency
│
└─ Step 5: Invoke Selected Skill
    ├─ Load skill content (Read SKILL.md)
    ├─ Apply skill's decision tree to task
    └─ Return skill recommendations
```

---

## Input/Output Contract

### Input Format

```json
{
  "task": "Add squeeze detection to volatility indicators",
  "files": ["src/features/technical/volatility.py"],
  "context": {
    "phase": "implementation",
    "agent": "code-engineer",
    "prior_skills": ["creating-technical-indicators"]
  }
}
```

### Output Format

```json
{
  "analysis": {
    "layer": "feature",
    "task_type": "feature",
    "keywords": ["squeeze", "detection", "volatility", "indicator"]
  },
  "recommendations": [
    {
      "skill": "creating-technical-indicators",
      "confidence": 0.95,
      "score": 95,
      "reasons": [
        "File path matches feature-engineering layer",
        "Task type 'feature' matches skill",
        "Keywords: indicator, volatility"
      ],
      "next_steps": [
        "Read skill for indicator calculator pattern",
        "Follow decision tree for new method vs new class",
        "Use calculate_all() integration pattern"
      ]
    },
    {
      "skill": "configuring-indicator-yaml",
      "confidence": 0.65,
      "score": 65,
      "reasons": [
        "Task type 'feature' matches skill",
        "Keywords: indicator"
      ],
      "next_steps": [
        "Add squeeze config to short_term_indicators.yaml",
        "Define parameters and priority level"
      ]
    }
  ],
  "selected": "creating-technical-indicators",
  "routing_log": [
    "Detected layer: feature (from src/features/technical/)",
    "Detected task type: feature (keywords: add, detection)",
    "Top skill: creating-technical-indicators (95 points)"
  ]
}
```

---

## Examples

### Example 1: Feature Implementation Routing

**Input:**
```json
{
  "task": "Add RSI divergence detection to momentum indicators",
  "files": ["src/features/technical/momentum.py"]
}
```

**Routing Analysis:**
```
Layer:     feature-engineering (src/features/technical/)
Task Type: feature (keyword "add")
Keywords:  RSI, divergence, detection, momentum, indicator

Skill Scores:
1. creating-technical-indicators: 95 pts
   - File path: +50 (src/features/technical/)
   - Task type: +30 (feature)
   - Keywords: +15 (RSI, momentum, indicator)

2. configuring-indicator-yaml: 55 pts
   - Task type: +30 (feature)
   - Keywords: +25 (indicator, RSI)

Selected: creating-technical-indicators
```

### Example 2: Bug Fix Routing

**Input:**
```json
{
  "task": "Fix NaN values in ATR calculation when volume is zero",
  "files": ["src/features/technical/volatility.py"]
}
```

**Routing Analysis:**
```
Layer:     feature-engineering
Task Type: bug_fix (keyword "fix", "NaN")
Keywords:  NaN, ATR, calculation, volume, volatility

Skill Scores:
1. creating-technical-indicators: 80 pts
   - File path: +50
   - Task type: +10 (bug_fix domain boost)
   - Keywords: +20 (ATR, volatility)

2. validating-time-series-data: 45 pts
   - Task type: +30 (validation-related)
   - Keywords: +15 (NaN handling)

Selected: creating-technical-indicators
Note: Also consider validating-time-series-data for NaN handling patterns
```

### Example 3: Post-Design Phase Routing

**Input:**
```json
{
  "task": "Generate test plan for new prediction endpoint",
  "files": ["src/api/routes/predictions.py"],
  "context": {
    "phase": "post-design",
    "agent": "solution-architect",
    "has_acceptance_criteria": true
  }
}
```

**Routing Analysis:**
```
Phase:     post-design (triggers test planning)
Layer:     API
Task Type: testing (keyword "test plan")
Keywords:  test, plan, prediction, endpoint

Skill Scores:
1. planning-test-scenarios: 90 pts  (if exists)
   - Phase: +40 (post-design trigger)
   - Task type: +30 (testing)
   - Keywords: +20 (test, plan)

2. creating-api-endpoints: 60 pts
   - File path: +50
   - Keywords: +10 (endpoint)

Selected: planning-test-scenarios (phase-based priority)
```

### Example 4: Multi-Layer Task

**Input:**
```json
{
  "task": "Implement new Polygon data source with OHLCV processing",
  "files": [
    "src/data/sources/polygon.py",
    "src/data/processors/polygon_processor.py"
  ]
}
```

**Routing Analysis:**
```
Layers:    data-layer (both source and processor)
Task Type: feature (keyword "implement", "new")
Keywords:  Polygon, data, source, OHLCV, processing

Skill Scores:
1. adding-data-sources: 90 pts
   - File path: +50 (src/data/sources/)
   - Task type: +30 (feature)
   - Keywords: +10 (source, data)

2. processing-ohlcv-data: 85 pts
   - File path: +50 (src/data/processors/)
   - Task type: +30 (feature)
   - Keywords: +5 (OHLCV)

3. creating-data-processors: 80 pts
   - File path: +50 (src/data/processors/)
   - Task type: +30 (feature)

Recommendation: Apply skills in sequence
1. adding-data-sources → For PolygonDataSource class
2. processing-ohlcv-data → For OHLCV standardization
```

---

## Conflict Resolution

When multiple skills score equally:

```
PRIORITY RULES:
1. Phase-specific skills first (post-design → test planning)
2. More specific path match wins (src/features/technical/ > src/features/)
3. Skill referenced in task description wins
4. Most recently successful skill wins (session history)
5. Ask agent/user for preference if still tied
```

### Manual Override

Agents can bypass routing:
```
"Skip routing, use skill: implementing-prediction-models"
```

The router respects explicit skill requests.

---

## Transparency & Logging

### Routing Log Format

Every routing decision includes:
```
routing_log: [
  "Input: task='...', files=[...]",
  "Detected layer: {layer} (from {evidence})",
  "Detected task type: {type} (keywords: {keywords})",
  "Scored {n} skills",
  "Top 3: {skill1}({score}), {skill2}({score}), {skill3}({score})",
  "Selected: {skill} (reason: {reason})"
]
```

This enables:
- Debugging unexpected routing
- Understanding why a skill was/wasn't selected
- Improving routing rules based on patterns

---

## Integration with Agents

### Agent-Specific Routing

| Agent | Typical Phase | Primary Skill Types |
|-------|---------------|---------------------|
| Requirements Analyst | pre-design | None (research only) |
| Solution Architect | design | Reference skills for patterns |
| Code Engineer | implementation | Implementation skills |
| Quality Guardian | review | Validation, risk skills |
| Test Automator | testing | Test planning, data generation |
| Documentation Curator | documentation | Reference skills for accuracy |

### Workflow Integration

```
Code Engineer receives task
│
├─ Invoke skill-router with task + files
│
├─ Router returns recommendations:
│   1. implementing-prediction-models (90%)
│   2. creating-dataclasses (65%)
│
├─ Engineer reads primary skill
│
├─ Engineer follows skill's decision tree
│
└─ If skill references another → Router handles chaining
```

---

## Available Skills Registry

Current skills in `.claude/skills/`:

### Backend Layer
| Skill | Triggers |
|-------|----------|
| `implementing-prediction-models` | src/models/**, model, predict, train |
| `creating-api-endpoints` | src/api/**, endpoint, route, FastAPI |
| `creating-data-processors` | src/data/processors/**, processor, transform |

### Feature Engineering
| Skill | Triggers |
|-------|----------|
| `creating-technical-indicators` | src/features/technical/**, indicator, RSI, MACD |
| `configuring-indicator-yaml` | configs/indicators/**, config, yaml |

### Data Layer
| Skill | Triggers |
|-------|----------|
| `adding-data-sources` | src/data/sources/**, source, connector, fetch |
| `creating-data-processors` | OHLCV, candle, sequence, processor, validate, clean |

### Trading Domain
| Skill | Triggers |
|-------|----------|
| `running-backtests` | backtest, simulate, walk-forward |
| `analyzing-trading-performance` | sharpe, sortino, drawdown, metrics |
| `implementing-risk-management` | src/trading/risk**, risk, position, circuit |

### Quality & Testing
| Skill | Triggers |
|-------|----------|
| `creating-dataclasses` | dataclass, DTO, @dataclass |
| `validating-time-series-data` | time series, leakage, chronological |
| `planning-test-scenarios` | test plan, acceptance criteria, test coverage |
| `generating-test-data` | fixture, mock, test data, builder |

---

## Additional Examples

### Example 5: Test Planning After Design

**Input:**
```json
{
  "task": "Create test plan for new squeeze detection feature",
  "files": ["src/features/technical/volatility.py"],
  "context": {
    "phase": "post-design",
    "acceptance_criteria": ["Detects squeeze when BB inside KC", "Handles edge cases"]
  }
}
```

**Routing Analysis:**
```
Phase:     post-design (triggers test planning priority)
Layer:     feature-engineering
Task Type: testing (keyword "test plan")
Keywords:  test, plan, squeeze, detection

Skill Scores:
1. planning-test-scenarios: 95 pts
   - Phase: +40 (post-design trigger)
   - Task type: +30 (testing)
   - Keywords: +25 (test, plan, acceptance)

2. creating-technical-indicators: 60 pts
   - File path: +50
   - Keywords: +10 (squeeze, detection)

3. generating-test-data: 55 pts
   - Task type: +30 (testing related)
   - Keywords: +25 (test)

Selected: planning-test-scenarios
Note: Will likely chain to generating-test-data for fixtures
```

### Example 6: Dataclass Creation Routing

**Input:**
```json
{
  "task": "Create TradeResult dataclass for backtest output",
  "files": ["src/simulation/backtester.py"]
}
```

**Routing Analysis:**
```
Layer:     simulation
Task Type: feature (keyword "create")
Keywords:  dataclass, trade, result, backtest

Skill Scores:
1. creating-dataclasses: 90 pts
   - Task type: +30 (feature)
   - Keywords: +40 (dataclass explicit)
   - Domain: +20 (DTO pattern)

2. running-backtests: 65 pts
   - File path: +50 (src/simulation/)
   - Keywords: +15 (backtest)

3. analyzing-trading-performance: 45 pts
   - Keywords: +25 (trade, result)
   - Related domain: +20

Selected: creating-dataclasses
Reason: Explicit "dataclass" keyword overrides file path scoring
```

---

## Quality Checklist

- [ ] Task analyzed for layer, type, and keywords
- [ ] All skills discovered and scored
- [ ] Top 3 recommendations returned with reasons
- [ ] Confidence scores are calibrated (0-1 range)
- [ ] Routing log captures decision process
- [ ] Manual override respected if provided

## Common Mistakes

- **Hardcoding skill names**: Skills may be added/renamed → Always discover dynamically
- **Ignoring context.phase**: Phase-specific routing missed → Check for post-design triggers
- **Single skill bias**: Only returning one option → Always return top 3 for transparency
- **Opaque routing**: User can't understand selection → Include routing_log in output

## Related Skills

### Backend Layer
- [implementing-prediction-models](../backend/implementing-prediction-models.md) - ML model patterns
- [creating-api-endpoints](../backend/creating-api-endpoints.md) - FastAPI endpoint patterns
- [creating-data-processors](../backend/creating-data-processors.md) - Data pipeline patterns

### Feature Engineering
- [creating-technical-indicators](../feature-engineering/creating-technical-indicators.md) - Indicator calculator patterns
- [configuring-indicator-yaml](../feature-engineering/configuring-indicator-yaml.md) - YAML configuration

### Data Layer
- [adding-data-sources](../data-layer/adding-data-sources.md) - Data source patterns
- [creating-data-processors](../backend/creating-data-processors.md) - Data processing (including OHLCV)

### Trading Domain
- [running-backtests](../trading-domain/running-backtests.md) - Backtesting patterns
- [analyzing-trading-performance](../trading-domain/analyzing-trading-performance.md) - Performance metrics
- [implementing-risk-management](../trading-domain/implementing-risk-management.md) - Risk management

### Quality & Testing
- [creating-dataclasses](../quality-testing/creating-dataclasses.md) - Dataclass patterns
- [validating-time-series-data](../quality-testing/validating-time-series-data.md) - Time series validation
- [planning-test-scenarios](../quality-testing/planning-test-scenarios.md) - Test planning
- [generating-test-data](../quality-testing/generating-test-data.md) - Test data generation

### Meta
- [improving-framework-continuously](../continuous-improvement/SKILL.md) - Framework evolution

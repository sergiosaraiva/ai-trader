# Quarterly Framework Maintenance Checklist

Run this checklist every 3 months to keep the agent-skill framework aligned with the evolving codebase.

**Schedule:** First week of January, April, July, October

---

## Pre-Maintenance

### Validation First

**CRITICAL: Run validation before ANY maintenance work.**

```bash
.claude/scripts/validate-framework.sh
```

- [ ] Validation passes with 0 errors
- [ ] Fix any YAML format issues FIRST
- [ ] All skills load correctly
- [ ] All agents load correctly

### Preparation

- [ ] Block 4-8 hours for maintenance
- [ ] Notify team of potential framework updates
- [ ] Create branch: `maintenance/YYYY-QN-framework-review`
- [ ] Export current metrics baseline

### Gather Data

- [ ] Collect all error reports from past quarter
- [ ] Export skill invocation logs (if available)
- [ ] Note any informal feedback about framework issues
- [ ] Review git history for codebase changes

---

## Part 1: Pattern Drift Analysis

### Re-discover Codebase Patterns

Run pattern discovery to capture current state:

```bash
# Search for key patterns in codebase
# Document findings in .claude/improvement/YYYY-QN-pattern-scan.md
```

**Patterns to Check:**

| Pattern | Current Location | Skill Reference | Status |
|---------|-----------------|-----------------|--------|
| BaseModel + Registry | src/models/base.py:__ | implementing-prediction-models | [ ] Match / [ ] Drift |
| Indicator Calculator | src/features/technical/*.py | creating-technical-indicators | [ ] Match / [ ] Drift |
| FastAPI Router | src/api/routes/*.py | creating-api-endpoints | [ ] Match / [ ] Drift |
| BaseDataSource | src/data/sources/base.py:__ | adding-data-sources | [ ] Match / [ ] Drift |
| RiskManager | src/trading/risk.py:__ | implementing-risk-management | [ ] Match / [ ] Drift |
| Backtester | src/simulation/backtester.py:__ | running-backtests | [ ] Match / [ ] Drift |

### Identify Changes

**Patterns That Changed:**
```
1. [Pattern name]
   - Old: [description]
   - New: [description]
   - Affected skills: [list]

2. [Pattern name]
   ...
```

**New Patterns Emerged:**
```
1. [Pattern name]
   - Location: [files]
   - Frequency: [how often used]
   - Skill needed: [ ] Yes / [ ] No
```

**Patterns No Longer Used:**
```
1. [Pattern name]
   - Was in: [files]
   - Replaced by: [new pattern or removed]
   - Skill to archive: [name]
```

---

## Part 2: Skill Health Check

### Per-Skill Review

For each skill in `.claude/skills/` (23 total):

---

#### Backend Layer (6 skills)

##### creating-fastapi-endpoints
- [ ] File references still valid (`src/api/routes/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### creating-python-services
- [ ] File references still valid (`src/api/services/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### creating-pydantic-schemas
- [ ] File references still valid (`src/api/schemas/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### implementing-prediction-models
- [ ] File references still valid (`src/models/**/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### creating-api-endpoints (legacy)
- [ ] File references still valid
- [ ] Consider merging with creating-fastapi-endpoints
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### creating-data-processors
- [ ] File references still valid (`src/data/processors/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

---

#### Frontend Layer (2 skills)

##### creating-react-components
- [ ] File references still valid (`frontend/src/components/*.jsx`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### creating-api-clients
- [ ] File references still valid (`frontend/src/api/*.js`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

---

#### Database Layer (1 skill)

##### creating-sqlalchemy-models
- [ ] File references still valid (`src/api/database/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

---

#### Feature Engineering Layer (2 skills)

##### creating-technical-indicators
- [ ] File references still valid (`src/features/technical/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### configuring-indicator-yaml
- [ ] File references still valid (`configs/indicators/*.yaml`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

---

#### Data Layer (2 skills)

##### adding-data-sources
- [ ] File references still valid (`src/data/sources/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### ~~processing-ohlcv-data~~ (DEPRECATED)
- [ ] Verify merged into creating-data-processors
- [ ] Remove from skill registry
- [ ] **Action**: [ ] Archive

---

#### Trading Domain Layer (3 skills)

##### running-backtests
- [ ] File references still valid (`src/simulation/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### analyzing-trading-performance
- [ ] File references still valid
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### implementing-risk-management
- [ ] File references still valid (`src/trading/risk.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

---

#### Testing Layer (2 skills)

##### writing-pytest-tests
- [ ] File references still valid (`tests/**/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### writing-vitest-tests
- [ ] File references still valid (`frontend/src/**/*.test.jsx`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

---

#### Quality & Testing Layer (4 skills)

##### creating-dataclasses
- [ ] File references still valid
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### validating-time-series-data
- [ ] File references still valid
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### planning-test-scenarios
- [ ] File references still valid
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### generating-test-data
- [ ] File references still valid
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

---

#### Build & Deployment Layer (1 skill)

##### creating-cli-scripts
- [ ] File references still valid (`scripts/*.py`)
- [ ] Examples still compile/run
- [ ] Invocation count this quarter: ___
- [ ] Error reports against it: ___
- [ ] Needs more examples: [ ] Yes / [ ] No
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

---

#### Meta-Skills (2 skills)

##### routing-to-skills
- [ ] YAML frontmatter valid (name matches folder)
- [ ] Skill registry up to date (all skills)
- [ ] Scoring algorithm accurate
- [ ] Path triggers current
- [ ] Keyword triggers current
- [ ] Routing accuracy this quarter: ___%
- [ ] Error reports against it: ___
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

##### improving-framework-continuously
- [ ] YAML frontmatter valid (name matches folder)
- [ ] Workflow still applicable
- [ ] Metrics tracking working
- [ ] Error template up to date
- [ ] Maintenance checklist current
- [ ] Validation script working
- [ ] Pre-commit hook documented
- [ ] Error reports against it: ___
- [ ] **Health Score**: [ ] Good / [ ] Needs Update / [ ] Archive

### Skill Health Summary

| Health | Count | Skills |
|--------|-------|--------|
| Good | ___ | |
| Needs Update | ___ | |
| Archive | ___ | |

---

## Part 3: Agent Effectiveness

### Agent Usage Review

| Agent | Invocations | Success Rate | Issues |
|-------|-------------|--------------|--------|
| Requirements Analyst | ___ | ___% | |
| Solution Architect | ___ | ___% | |
| Code Engineer | ___ | ___% | |
| Quality Guardian | ___ | ___% | |
| Test Automator | ___ | ___% | |
| Documentation Curator | ___ | ___% | |

### Workflow Bottlenecks

```
Identified bottlenecks:
1. [Agent/step]: [description of bottleneck]
2. [Agent/step]: [description of bottleneck]
```

### Developer Feedback

```
Collected feedback:
1. [Feedback item]
2. [Feedback item]
```

---

## Part 4: Routing Accuracy

### Routing Metrics

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Correct routing rate | ___% | >90% | [ ] Pass / [ ] Fail |
| High confidence accuracy | ___% | >95% | [ ] Pass / [ ] Fail |
| Low confidence rate | ___% | <10% | [ ] Pass / [ ] Fail |

### Mis-routed Tasks

```
1. Task: [description]
   Routed to: [skill]
   Should have been: [skill]
   Root cause: [explanation]

2. Task: [description]
   ...
```

### Scoring Algorithm Updates Needed

```
1. [Update description]
2. [Update description]
```

---

## Part 5: Framework Metrics

### Error Metrics

| Metric | Last Quarter | This Quarter | Trend |
|--------|--------------|--------------|-------|
| Total errors | ___ | ___ | |
| Recurrence rate | ___% | ___% | |
| Avg resolution time | ___ days | ___ days | |
| Critical errors | ___ | ___ | |
| YAML validation pass rate | ___% | ___% | (Target: 100%) |

### Error Type Distribution

| Type | Count | % of Total |
|------|-------|------------|
| Hallucination | ___ | ___% |
| Outdated Pattern | ___ | ___% |
| Missing Skill | ___ | ___% |
| Wrong Routing | ___ | ___% |
| Agent Logic | ___ | ___% |
| Incomplete Guards | ___ | ___% |

### Alignment Score

Run validation suite and calculate:

```
Alignment Score = (Correct outputs / Total scenarios) * 100

Last Quarter: ___%
This Quarter: ___%
Trend: [ ] Improving / [ ] Stable / [ ] Degrading
```

---

## Part 6: Updates

### Skills to Archive

```
1. [Skill name]: [reason for archiving]
   - Move to: .claude/skills/_archived/
```

### Skills to Consolidate

```
1. Merge [skill A] + [skill B] → [new skill]
   - Reason: [why consolidate]
```

### Skills to Split

```
1. Split [skill] → [skill A] + [skill B]
   - Reason: [why split]
```

### Version Updates

| Component | Old Version | New Version |
|-----------|-------------|-------------|
| Skills framework | ___ | ___ |
| [Skill name] | ___ | ___ |
| [Agent name] | ___ | ___ |

---

## Part 7: Changelog Generation

### Changes This Quarter

```markdown
## [YYYY-QN] - YYYY-MM-DD

### Added
- [New skill/feature]
- [New skill/feature]

### Changed
- [Updated skill/agent]
- [Updated skill/agent]

### Fixed
- [Bug fix]
- [Bug fix]

### Deprecated
- [Skill/feature being phased out]

### Removed
- [Archived skill/feature]

### Security
- [Security-related changes]
```

---

## Post-Maintenance

### Final Validation

**CRITICAL: Run validation after ALL maintenance work.**

```bash
.claude/scripts/validate-framework.sh
```

- [ ] Validation passes with 0 errors
- [ ] All skills have valid YAML frontmatter
- [ ] All agents have valid YAML frontmatter
- [ ] name fields match folder/file names
- [ ] All required fields present

### Documentation

- [ ] Save this checklist as `.claude/improvement/YYYY-QN-maintenance-report.md`
- [ ] Update README with any structural changes
- [ ] Update CLAUDE.md if project patterns changed

### Commit

- [ ] Run validation one final time
- [ ] Review all changes
- [ ] Commit with message: `maintenance(framework): YYYY-QN quarterly review`
- [ ] Pre-commit hook should pass automatically
- [ ] Create PR for team review

### Communication

- [ ] Share summary with team
- [ ] Highlight breaking changes
- [ ] Share YAML validation pass rate
- [ ] Schedule training if significant updates

---

## Quick Reference: Maintenance Commands

```bash
# Find pattern drift
grep -rn "class BaseModel" src/
grep -rn "class.*DataSource" src/
grep -rn "def calculate_all" src/

# Check skill references
grep -rn "src/.*\.py:[0-9]" .claude/skills/

# Count error reports
ls -la .claude/improvement/errors/ | wc -l

# Generate pattern scan
find src/ -name "*.py" -exec grep -l "class.*:" {} \; | head -20
```

---

## Sign-off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Reviewer | | | |
| Approver | | | |

**Next Maintenance Due:** YYYY-MM-DD (3 months from now)

# Meta Skills Validation Scenarios

## routing-to-skills

### Scenario 1: Model Layer Routing
**Task**: "Create a new transformer model for predictions"
**Files**: `src/models/technical/transformer.py`

**Expected**:
- Top recommendation: `implementing-prediction-models`
- Confidence: >0.85
- Reason: File path matches src/models/**, keyword "model"

**Validation**:
- [ ] Correct skill ranked #1
- [ ] Confidence score appropriate
- [ ] Routing log explains decision

---

### Scenario 2: Feature Layer Routing
**Task**: "Add Bollinger Band squeeze detection"
**Files**: `src/features/technical/volatility.py`

**Expected**:
- Top recommendation: `creating-technical-indicators`
- Confidence: >0.90
- Also recommends: `configuring-indicator-yaml`

**Validation**:
- [ ] Correct primary skill
- [ ] Config skill in top 3
- [ ] Keywords detected (indicator, volatility)

---

### Scenario 3: Ambiguous Task Routing
**Task**: "Add validation to the processor"
**Files**: `src/data/processors/ohlcv.py`

**Expected**:
- Top recommendation: `creating-data-processors`
- Not: `creating-api-endpoints` (despite "validation" keyword)
- Reason: File path should override keyword ambiguity

**Validation**:
- [ ] File path wins over keyword
- [ ] Data processor skill selected
- [ ] Routing log explains path priority

---

### Scenario 4: Post-Design Phase Routing
**Task**: "Generate tests for the new feature"
**Context**: `{ "phase": "post-design" }`

**Expected**:
- Top recommendation: `planning-test-scenarios`
- Phase trigger activates
- Confidence: >0.90

**Validation**:
- [ ] Phase context respected
- [ ] Test planning skill selected
- [ ] Phase mentioned in routing log

---

### Scenario 5: Multi-File Task Routing
**Task**: "Implement Polygon data source with processing"
**Files**:
  - `src/data/sources/polygon.py`
  - `src/data/processors/polygon_processor.py`

**Expected**:
- Recommendations: Both `adding-data-sources` and `creating-data-processors`
- Suggests: Apply in sequence
- Both with high confidence

**Validation**:
- [ ] Both skills recommended
- [ ] Sequence suggested
- [ ] File-specific routing

---

## improving-framework-continuously

### Scenario 6: Error Classification
**Task**: "Process this error: Skill referenced non-existent file"

**Expected**:
- Classifies as: Outdated Pattern
- Workflow: Locate file → Update reference → Verify → Test
- References: maintenance-checklist.md

**Validation**:
- [ ] Correct error type
- [ ] Resolution workflow clear
- [ ] Verification step included

---

### Scenario 7: Missing Skill Detection
**Task**: "Router returned low confidence for WebSocket task"

**Expected**:
- Classifies as: Missing Skill
- Workflow: Research → Create skill → Update router → Test
- Includes: Skill template reference

**Validation**:
- [ ] Missing skill identified
- [ ] Creation workflow clear
- [ ] Router update mentioned

---

### Scenario 8: Weekly Triage Process
**Task**: "5 error reports accumulated this week"

**Expected**:
- Provides: Prioritization framework
- Steps: Collect → Classify → Prioritize → Fix → Validate
- Metrics: Update tracking

**Validation**:
- [ ] Severity-based prioritization
- [ ] Pattern analysis mentioned
- [ ] Metrics tracking included

---

## Results Template

```markdown
## Validation Run: YYYY-MM-DD

### routing-to-skills
| Scenario | Pass/Fail | Notes |
|----------|-----------|-------|
| 1: Model Routing | | |
| 2: Feature Routing | | |
| 3: Ambiguous Routing | | |
| 4: Phase Routing | | |
| 5: Multi-File Routing | | |

### improving-framework-continuously
| Scenario | Pass/Fail | Notes |
|----------|-----------|-------|
| 6: Error Classification | | |
| 7: Missing Skill | | |
| 8: Weekly Triage | | |

**Total**: X/8 passed
**Issues Found**: [List any issues]
```

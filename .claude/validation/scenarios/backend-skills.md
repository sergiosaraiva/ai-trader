# Backend Skills Validation Scenarios

## implementing-prediction-models

### Scenario 1: New Model Creation
**Task**: "Create a new MediumTermModel for daily predictions"
**Files**: `src/models/technical/medium_term.py`

**Expected**:
- Skill selected: `implementing-prediction-models`
- Recommends: Inherit from `BaseModel` or `TechnicalBaseModel`
- References: `src/models/base.py:57` (BaseModel class)
- Includes: `DEFAULT_CONFIG` pattern
- Includes: Registry registration at module end

**Validation**:
- [ ] Correct skill selected
- [ ] BaseModel inheritance mentioned
- [ ] DEFAULT_CONFIG pattern shown
- [ ] All 4 abstract methods listed (build, train, predict, predict_batch)
- [ ] ModelRegistry.register() at end
- [ ] File references exist

---

### Scenario 2: Model Training Implementation
**Task**: "Implement the train() method for ShortTermModel with early stopping"
**Files**: `src/models/technical/short_term.py`

**Expected**:
- Skill selected: `implementing-prediction-models`
- References: Training loop example from skill
- Includes: Early stopping pattern
- Includes: `is_trained = True` at end

**Validation**:
- [ ] Training loop pattern shown
- [ ] Early stopping included
- [ ] is_trained flag mentioned
- [ ] Gradient clipping mentioned

---

## creating-api-endpoints

### Scenario 3: New Prediction Endpoint
**Task**: "Add a batch prediction endpoint /api/v1/predictions/batch"
**Files**: `src/api/routes/predictions.py`

**Expected**:
- Skill selected: `creating-api-endpoints`
- Recommends: APIRouter pattern
- Includes: Pydantic request/response models
- Includes: Field() with descriptions
- References: `src/api/routes/predictions.py`

**Validation**:
- [ ] Correct skill selected
- [ ] APIRouter usage shown
- [ ] Pydantic models mentioned
- [ ] Field(description=...) pattern
- [ ] Async handler pattern

---

### Scenario 4: Error Handling in API
**Task**: "Add proper error handling to the prediction endpoint"
**Files**: `src/api/routes/predictions.py`

**Expected**:
- Skill selected: `creating-api-endpoints`
- Includes: HTTPException usage
- Includes: Status codes (422, 500, etc.)
- References: Error handling patterns

**Validation**:
- [ ] HTTPException mentioned
- [ ] Appropriate status codes
- [ ] Error response structure

---

## creating-data-processors

### Scenario 5: New Data Processor
**Task**: "Create a processor to clean and validate incoming tick data"
**Files**: `src/data/processors/tick_processor.py`

**Expected**:
- Skill selected: `creating-data-processors`
- Recommends: Validate → Clean → Transform pipeline
- References: `src/data/processors/ohlcv.py` as example
- Includes: Input validation patterns

**Validation**:
- [ ] Correct skill selected
- [ ] Three-phase pipeline mentioned
- [ ] Validation first pattern
- [ ] Error handling for invalid data

---

## Results Template

```markdown
## Validation Run: YYYY-MM-DD

### implementing-prediction-models
| Scenario | Pass/Fail | Notes |
|----------|-----------|-------|
| 1: New Model | | |
| 2: Training | | |

### creating-api-endpoints
| Scenario | Pass/Fail | Notes |
|----------|-----------|-------|
| 3: Batch Endpoint | | |
| 4: Error Handling | | |

### creating-data-processors
| Scenario | Pass/Fail | Notes |
|----------|-----------|-------|
| 5: Tick Processor | | |

**Total**: X/5 passed
**Issues Found**: [List any issues]
```

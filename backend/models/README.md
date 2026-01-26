# Pre-trained Models

This repository includes pre-trained model weights for immediate deployment.

## Extracting the Models

The models are stored as split compressed archives due to GitHub's file size limits.

### On Linux/Mac:

```bash
# From the project root directory
cd models
cat models_trained.tar.gz.part_* | tar -xzv
cd ..

# This will extract all trained weights into the models/ directory
```

### On Windows (using Git Bash or WSL):

```bash
cd models
cat models_trained.tar.gz.part_* | tar -xzv
cd ..
```

### Using Python:

```python
import subprocess
import os

os.chdir('models')
subprocess.run("cat models_trained.tar.gz.part_* | tar -xzv", shell=True)
os.chdir('..')
```

## Included Models

After extraction, the `models/` directory contains:

| Directory | Description | Size |
|-----------|-------------|------|
| `mtf_ensemble/` | Multi-timeframe ensemble (1H, 4H, D) | ~1 MB |
| `practical_e2e/` | End-to-end trained models | ~156 MB |
| `individual_models/` | Individual timeframe models | ~144 MB |
| `trained/` | Various trained model checkpoints | ~319 MB |
| `hybrid/` | Hybrid sequence + XGBoost models | ~1 MB |
| `enhanced_hybrid/` | Enhanced hybrid models | ~3 MB |

## Usage

```python
from src.models.technical import ShortTermModel

# Load pre-trained model
model = ShortTermModel()
model.load('models/practical_e2e/short_term')

# Make predictions
predictions = model.predict(features)
```

## Re-training

If you need to re-train the models:

```bash
# Train MTF ensemble
python scripts/train_mtf_ensemble.py --pair EURUSD

# Train individual models
python scripts/train_individual_models.py --pair EURUSD
```

## Archive Files

```
models/
├── models_trained.tar.gz.part_aa  (90 MB)
├── models_trained.tar.gz.part_ab  (90 MB)
├── models_trained.tar.gz.part_ac  (90 MB)
├── models_trained.tar.gz.part_ad  (90 MB)
├── models_trained.tar.gz.part_ae  (90 MB)
├── models_trained.tar.gz.part_af  (90 MB)
├── models_trained.tar.gz.part_ag  (81 MB)
└── README.md
─────────────────────────────────────────
Total compressed:              621 MB
Total uncompressed:            677 MB
```

# Jupyter Notebooks for Analysis

This directory contains Jupyter notebooks for exploratory analysis and results visualization.

## Available Notebooks

### 01_exploratory_analysis.ipynb
- Load and explore engagement data
- Statistical summaries
- Feature distributions
- Temporal patterns

### 02_feature_visualization.ipynb
- Visualize individual features
- Feature correlations
- Feature importance analysis
- Interactive plots

### 03_results_analysis.ipynb
- Model evaluation metrics
- Confusion matrices
- ROC curves
- Performance comparisons
- Publication-ready figures

## Usage

```bash
# Install jupyter
pip install jupyterlab

# Launch jupyter
jupyter lab

# Navigate to notebooks/ directory and open desired notebook
```

## Creating New Notebooks

To create a new analysis notebook:

1. Create new notebook in this directory
2. Import required libraries:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0, '..')
from src.evaluation.metrics import compute_metrics
```

3. Load your data:
```python
df = pd.read_csv('../data/labels/engagement_data.csv')
```

4. Perform analysis and create visualizations

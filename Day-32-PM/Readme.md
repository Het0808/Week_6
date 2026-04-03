# Day 32 — Decision Trees & Random Forest Assignment

Week 6, Machine Learning & AI | Insurance Fraud Detection Case Study

---

## Overview

This assignment builds a fraud detection system using Decision Trees and Random Forests on synthetic insurance claims data (3,000 records). It covers model training, hyperparameter tuning, cost-sensitive evaluation, and deployment recommendation.

| Part | Topic | Weight |
|------|-------|--------|
| A | DT + RF models, tuning, cost analysis, deployment recommendation | 40% |
| B | Gradient Boosting vs Bagging (stretch) | 30% |
| C | Interview questions — tradeoffs, coding, debug | 20% |
| D | OOB error — AI explanation + critique | 10% |

---

## Files

```
D32_DT_RF_Assignment.ipynb   ← main notebook (all code)
D32_DT_RF_Solutions.docx     ← written report with answers & tables
README.md                    ← this file
```

---

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

Python 3.8+ required.

---

## Run

**Option 1 — Jupyter Notebook**
```bash
jupyter notebook D32_DT_RF_Assignment.ipynb
```
Then run all cells: `Kernel → Restart & Run All`

**Option 2 — JupyterLab**
```bash
jupyter lab D32_DT_RF_Assignment.ipynb
```

**Option 3 — Run as script**
```bash
jupyter nbconvert --to script D32_DT_RF_Assignment.ipynb
python D32_DT_RF_Assignment.py
```

---

## Output

Running the notebook produces these files in the same directory:

```
dt_tree.png              ← Decision Tree visualisation
confusion_matrices.png   ← DT vs RF confusion matrices
metrics_comparison.png   ← Bar chart of all metrics
cost_comparison.png      ← Business cost comparison
feature_importance.png   ← MDI vs Permutation importance
```

---

## Key Results (expected)

| Metric | Decision Tree | Random Forest (tuned) |
|--------|-------------|----------------------|
| Recall | ~0.74 | ~0.84 |
| ROC-AUC | ~0.77 | ~0.89 |
| Business Cost | ~$852 | ~$548 |

> All results are reproducible — `random_state=42` is set throughout.

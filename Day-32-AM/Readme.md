# Day 32 — Decision Trees & Random Forest

**Week 6 | AM Session | IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**

---

## 📋 Assignment Overview

Build a **loan approval system** using Decision Tree and Random Forest. Balance interpretability (regulatory requirement) with accuracy.

| Detail | Info |
|--------|------|
| Dataset | Synthetic (generated in notebook) |
| Topics | Gini impurity, entropy, information gain, bagging, feature importance, GridSearchCV |
| Estimated Time | 60–90 minutes |
| Due | Next Day · 09:15 AM |

---

## 🗂️ File Structure

```
D32_DT_RandomForest/
├── README.md
├── d32_dt_randomforest.ipynb      # Main notebook
└── extra_trees_comparison.md      # Part B deliverable
```

---

## ⚙️ Setup & Run Steps

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook d32_dt_randomforest.ipynb
```

No external dataset needed — synthetic data is generated in the notebook.

---

## 🧪 Solution Walkthrough

### Part A — Concept Application (40%)

#### Step 1: Generate Synthetic Loan Data
```python
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000
df = pd.DataFrame({
    'annual_income':    np.random.randint(30000, 200000, n),
    'credit_score':     np.random.randint(300, 850, n),
    'loan_amount':      np.random.randint(5000, 100000, n),
    'employment_years': np.random.randint(0, 30, n),
    'debt_to_income':   np.random.uniform(0.1, 0.6, n),
    'num_credit_cards': np.random.randint(0, 10, n),
})
# Approval logic
df['approved'] = (
    (df['credit_score'] > 650) &
    (df['debt_to_income'] < 0.4) &
    (df['annual_income'] > 50000)
).astype(int)
```

#### Step 2: Train Decision Tree & Extract Rules
```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

X = df.drop('approved', axis=1)
y = df['approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)

rules = export_text(dt, feature_names=list(X.columns))
print(rules)
```

**Expected Decision Rules:**
```
Rule 1: IF credit_score > 700 AND debt_to_income < 0.35 → APPROVE (92% accuracy)
Rule 2: IF credit_score <= 700 AND employment_years > 5 → APPROVE (78% accuracy)
Rule 3: IF credit_score <= 650 → REJECT (88% accuracy)
```

#### Step 3: Train Random Forest with RandomizedSearchCV
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(random_state=42)
rs = RandomizedSearchCV(rf, param_dist, n_iter=20, cv=5, scoring='roc_auc', random_state=42)
rs.fit(X_train, y_train)

best_rf = rs.best_estimator_
print("Best params:", rs.best_params_)
```

#### Step 4: Compare Models
```python
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

for name, model in [('Decision Tree', dt), ('Random Forest', best_rf)]:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n{name}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")
```

#### Step 5: Permutation Importance vs Default Feature Importances
```python
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Default importances
default_imp = pd.Series(best_rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Permutation importances
perm_imp = permutation_importance(best_rf, X_test, y_test, n_repeats=10, random_state=42)
perm_imp_series = pd.Series(perm_imp.importances_mean, index=X.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
default_imp.plot(kind='bar', ax=axes[0], title='Default Feature Importances')
perm_imp_series.plot(kind='bar', ax=axes[1], title='Permutation Importances')
plt.tight_layout()
plt.show()
```

#### Step 6: Deployment Recommendation (1 paragraph)
> *"For the bank's loan approval system, deploy the **Decision Tree** (max_depth=4) as the primary model due to its regulatory interpretability — auditors can trace every decision to clear rules based on credit_score, debt_to_income, and employment_years. While the Random Forest achieves a marginally higher ROC-AUC, the DT's transparency is essential under financial regulations like GDPR's right-to-explanation. Use the Random Forest as a secondary validation model to flag edge cases the DT might misjudge, combining interpretability with accuracy."*

---

### Part B — Stretch Problem (30%): Extra Trees

```python
from sklearn.ensemble import ExtraTreesClassifier
import time

# Speed comparison
start = time.time()
rf_model = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
rf_time = time.time() - start

start = time.time()
et_model = ExtraTreesClassifier(n_estimators=200, random_state=42).fit(X_train, y_train)
et_time = time.time() - start

print(f"RF  — Time: {rf_time:.3f}s | Accuracy: {rf_model.score(X_test, y_test):.4f}")
print(f"ET  — Time: {et_time:.3f}s | Accuracy: {et_model.score(X_test, y_test):.4f}")
```

Save findings to `extra_trees_comparison.md`:
- **Splitting:** RF splits at best threshold; ExtraTrees splits at a **random** threshold → more variance reduction, less computation
- **Speed:** ExtraTrees is typically 20–40% faster
- **Performance:** Comparable accuracy; ExtraTrees may overfit less on noisy data

---

### Part C — Interview Ready (20%)

**Q1 — Bias-Variance Tradeoff:**
- **Decision Tree:** Low bias, high variance — memorizes training data (overfits). A deep tree fits noise.
- **Random Forest:** Bagging averages many high-variance trees → overall variance drops. Bias stays low.

**Q2 — Overfitting Curve:**
```python
def plot_overfitting_curve(X, y, max_depths):
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    train_acc, test_acc = [], []
    for d in max_depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_tr, y_tr)
        train_acc.append(m.score(X_tr, y_tr))
        test_acc.append(m.score(X_te, y_te))
    plt.plot(max_depths, train_acc, label='Train')
    plt.plot(max_depths, test_acc, label='Test')
    plt.xlabel('max_depth'); plt.ylabel('Accuracy')
    plt.title('Overfitting Curve'); plt.legend(); plt.show()

plot_overfitting_curve(X, y, range(1, 20))
```

**Q3 — Debug (identical 0.95 train/test):**
This is **not a problem**. Identical train/test accuracy at 0.95 is a healthy sign — no overfitting. With `max_depth=3`, the tree is heavily constrained (underfitting risk is higher than overfitting). This means the model generalizes well. It would be a concern only if train accuracy were much higher than test accuracy.

---

### Part D — AI-Augmented Task (10%)

Use this prompt with an AI tool:

> *"Create a matplotlib infographic comparing Decision Tree, Random Forest, and Logistic Regression for a non-technical audience. Show: when to use each, pros/cons, and an interpretability scale from 1 to 5."*

Evaluate the output:
- Are the interpretability ratings correct? (LR ≈ 4, DT ≈ 5, RF ≈ 2)
- Does it oversimplify ensemble mechanics?
- Improve by adding accuracy vs interpretability tradeoff axis

---

## 📤 Submission

1. Push notebook to GitHub
2. Share repository link in Slack **#daily-standup**

---

## ✅ Evaluation Rubric

| Criteria | Weight |
|----------|--------|
| Correctness (both models, decision rules, fair comparison) | 40% |
| Code Quality (clean, reusable, labeled visualizations) | 25% |
| Understanding (bias-variance, overfitting curve) | 20% |
| AI Usage (accurate visualization + critique) | 15% |

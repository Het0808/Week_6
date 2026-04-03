# 🤖 SVM, KNN & Full Week Comparison — ML Cheat Sheet
**Day 33 | PM Session | Week 6 — Machine Learning & AI**  
**IIT Gandhinagar · PG Diploma in AI-ML & Agentic AI Engineering**

---

## 📋 Assignment Overview

A comprehensive ML cheat-sheet notebook covering all **8 algorithms** from Week 6. Each algorithm gets a structured card (when to use, key params, pros/cons, code snippet), followed by a fair benchmark on a real dataset, a full text-classification pipeline, and interview-ready problem solving.

---

## 📁 Repository Structure

```
D33_PM_SVM_KNN_CS/
│
├── D33_PM_SVM_KNN_CS_Assignment.ipynb        # ✅ Main notebook (submit this)
├── algorithm_selection_guide.txt              # 📋 Personal algorithm guide (Part D)
├── README.md                                  # 📖 This file
│
└── figures/
    ├── algo_comparison.png          # Part A — 8-algo CV & test accuracy + training time
    ├── cv_distributions.png         # Part A — Boxplots of fold-by-fold CV scores
    ├── feature_importances.png      # Part A — RF vs XGBoost top 15 features
    ├── text_classification.png      # Part B — Confusion matrices + F1 (SVM vs LR)
    ├── top_words_per_category.png   # Part B — Top discriminative words per news category
    ├── high_dim_comparison.png      # Part C Q1 — 100 features, 50 samples benchmark
    ├── overfitting_diagnosis.png    # Part C Q3 — Train vs test acc across C values
    └── algo_selection_guide.png     # Part D — Visual algorithm selection table
```

---

## 🚀 Getting Started

### 1. Install Dependencies

```bash
pip install scikit-learn numpy pandas matplotlib seaborn scipy xgboost jupyter
```

### 2. Launch the Notebook

```bash
jupyter notebook D33_PM_SVM_KNN_CS_Assignment.ipynb
```

### 3. Run All Cells

In the notebook: **Kernel → Restart & Run All**

Or from terminal:

```bash
jupyter nbconvert --to notebook --execute D33_PM_SVM_KNN_CS_Assignment.ipynb \
    --output D33_PM_SVM_KNN_CS_Assignment_executed.ipynb \
    --ExecutePreprocessor.timeout=300
```

> **Note:** The 20 Newsgroups dataset requires internet access. If unavailable, the notebook uses a locally-generated synthetic text dataset with equivalent structure.

---

## 📚 Assignment Parts

---

### Part A — 8-Algorithm ML Cheat Sheet (40%)

> **Dataset:** Breast Cancer Wisconsin (569 samples · 30 features · 2 classes)  
> **Evaluation:** 5-fold stratified CV + test set · StandardScaler · same splits for all

#### Algorithm Cards

| # | Algorithm | When to Use | Key Params | Pros | Cons |
|---|-----------|-------------|------------|------|------|
| 1 | **Logistic Regression** | Linearly separable, need probabilities, large data | C, penalty, solver | Interpretable, fast, calibrated probs | Linear boundary only |
| 2 | **Ridge Classifier** | Correlated features, multicollinearity | alpha, solver | Handles multicollinearity, stable | No probabilities, no L1 sparsity |
| 3 | **Decision Tree** | Need rules, mixed types, non-linear | max_depth, criterion | Interpretable, no scaling | High variance, overfits easily |
| 4 | **Random Forest** | Tabular, non-linear, need feature importance | n_estimators, max_features | Robust, parallelisable, built-in importance | Less interpretable than single DT |
| 5 | **Gradient Boosting** | High accuracy on tabular, structured features | n_estimators, learning_rate, max_depth | High accuracy, regularised | Slow training, sequential |
| 6 | **XGBoost** | Tabular SOTA, handles missing values | n_estimators, learning_rate, reg_alpha | SOTA tabular, GPU, handles NaN | Many hyperparams, can overfit |
| 7 | **SVM (RBF)** | Small-medium data, high-dim, non-linear | C, gamma, kernel | Kernel trick, memory efficient | O(n²·³) slow, needs scaling |
| 8 | **KNN** | Small data, no training time, local patterns | n_neighbors, metric, weights | No training, non-parametric | Slow inference, curse of dimensionality |

#### Results on Breast Cancer Dataset

| Rank | Algorithm | CV Mean | CV Std | Test Acc |
|------|-----------|---------|--------|----------|
| 1 | Logistic Regression | ~0.978 | ~0.010 | ~0.983 |
| 2 | SVM (RBF) | ~0.971 | ~0.020 | ~0.983 |
| 3 | Random Forest | ~0.963 | ~0.018 | ~0.956 |
| 4 | KNN | ~0.960 | ~0.011 | ~0.956 |
| 5 | XGBoost | ~0.960 | ~0.013 | ~0.947 |
| 6 | Ridge Classifier | ~0.960 | ~0.022 | ~0.956 |
| 7 | Gradient Boosting | ~0.956 | ~0.022 | ~0.939 |
| 8 | Decision Tree | ~0.923 | ~0.012 | ~0.921 |

**🏆 Recommendation: Logistic Regression**  
Highest CV mean, lowest variance, interpretable coefficients, fastest inference. SVM (RBF) ties on test accuracy but is slower. For production, LR is the clear choice on this dataset.

---

### Part B — TF-IDF + SVM Text Classification (30%)

> **Dataset:** 20 Newsgroups — 4 categories: `sci.space` · `rec.sport.hockey` · `talk.politics.guns` · `comp.graphics`

#### Pipeline Architecture

```
Raw Text
    ↓
TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, max_features=50000)
    ↓
[Sparse Matrix: ~50K features]
    ↓
LinearSVC(C=1.0)  OR  LogisticRegression(C=5.0)
    ↓
Predicted Category
```

#### Results

| Model | Accuracy | Notes |
|-------|----------|-------|
| **LinearSVC** | ~0.92–0.94 | Classic choice for text, very fast |
| Logistic Regression | ~0.91–0.93 | Probabilistic, slightly slower |

**Why LinearSVC wins on text:**
- TF-IDF produces **very high-dimensional sparse** feature spaces (50K+ dims)
- In high-D sparse space, **linear boundaries work extremely well**
- LinearSVC is optimised for sparse inputs (no kernel matrix needed)
- Used at Google, Yahoo, and Bloomberg for news categorisation

#### Top Discriminative Words (per category)
The notebook visualises the highest-weight TF-IDF features per class from the LinearSVC coefficient matrix — showing which words are most predictive for each news category.

---

### Part C — Interview Ready (20%)

#### Q1: 100 Features, 50 Samples — Which Work?

```python
# Simulate: n=50, d=100 (p >> n regime)
X, y = make_classification(n_samples=50, n_features=100,
                            n_informative=10, random_state=42)
```

| Algorithm | Expected Behaviour | Reason |
|-----------|-------------------|--------|
| ✅ LR (L1/L2) | Works well | Strong regularisation; L1 does feature selection |
| ✅ Ridge | Works well | Shrinks correlated coefficients |
| ✅ SVM (Linear) | Works well | Max-margin in high-dim is SVM's sweet spot |
| ⚠️ SVM (RBF) | Struggles | gamma tuning collapses; needs more samples |
| ⚠️ Decision Tree | Overfits | 100 splits on 50 points = memorises training |
| ⚠️ Random Forest | Borderline | Averaging helps but still noisy splits |
| ❌ KNN | Fails | Curse of dimensionality: all 50 points are equidistant in 100D |
| ❌ XGBoost/GBM | Overfits | Hundreds of weak learners on 50 samples = pure overfit |

**Rule of thumb:** In the p >> n regime, prefer **L1/L2 regularised linear models**.

---

#### Q2: `model_selection_report()` — Statistical Model Selection

```python
def model_selection_report(X, y, models_dict, cv_splits=5, scale=True):
    """
    Runs 5-fold CV for each model, returns ranked DataFrame,
    and uses paired t-test to identify the statistically best model.
    """
    # 1. Scale features if requested
    # 2. Collect fold-by-fold CV scores for each model
    # 3. Build summary DataFrame (CV mean, std, 95% CI)
    # 4. Paired t-test: best model vs all others (scipy.stats.ttest_rel)
    # 5. Flag statistically significant differences (p < 0.05)
    return df
```

**Output columns:** `Model`, `CV Mean`, `CV Std`, `CV Min`, `CV Max`, `95% CI Lower`, `95% CI Upper`, `p-value vs Best`, `Sig. Better?`

**Why paired t-test?** We use per-fold scores (not single test scores) because the same folds are used for all models, making the observations paired and the t-test valid for detecting true performance differences.

---

#### Q3: SVM Overfitting Diagnosis — Train=1.0, Test=0.52

```python
# ❌ Broken — memorises training data
svm = SVC(kernel='rbf', C=1000, gamma=0.1)
svm.fit(X_train, y_train)  # Train: 1.00, Test: 0.52
```

**Root Cause:** High C + high gamma → every training point becomes a support vector. The model memorises training data rather than learning the generalised boundary.

**3 Specific Fixes:**

| Fix | Code | Effect |
|-----|------|--------|
| **1. Reduce C** | `SVC(C=0.1)` | Wider margin, more regularisation |
| **2. Tune gamma** | `SVC(gamma='scale')` or GridSearchCV | Stop ultra-local RBF kernels |
| **3. Simpler kernel** | `SVC(kernel='linear')` | Fewer parameters, harder to overfit |

The notebook includes a full visualisation of train vs test accuracy across C values from 0.001 to 10,000.

---

### Part D — Algorithm Selection Guide (10%)

> AI-generated decision guide, verified against experience, and improved with 9 edge cases.

#### Quick Decision Flowchart

```
Is the data text/NLP?       YES → LinearSVC (or LR)
        ↓ NO
n < 200 samples?            YES → LR(L1/L2) or SVM(linear)
        ↓ NO
Need interpretability?      YES → Logistic Regression or DT
        ↓ NO
Need max accuracy?          YES → XGBoost (tune with GridSearchCV)
        ↓ NO
Want a robust baseline?          → Random Forest ✅
```

#### 9 Edge Cases (that AI missed):
1. KNN needs **cosine distance** for text, not Euclidean
2. SVM (RBF) is O(n^2.7) — avoid for n > 50K
3. XGBoost still needs tuning even on small datasets
4. LR assumes log-odds linearity — always check residuals
5. RF underperforms on very wide sparse matrices — use LinearSVC instead
6. KNN is NOT scale-invariant — always scale first
7. DT is unstable — tiny data change → completely different tree
8. GBM is sequential — prefer XGBoost/LightGBM at scale for speed
9. SVM does NOT output calibrated probabilities by default — use `probability=True` + Platt scaling

---

## 📊 Figures Reference

| File | Part | Description |
|------|------|-------------|
| `algo_comparison.png` | A | CV accuracy, test accuracy, training time for all 8 algorithms |
| `cv_distributions.png` | A | Box plots of fold scores — shows stability across folds |
| `feature_importances.png` | A | Top 15 features: Random Forest vs XGBoost |
| `text_classification.png` | B | Confusion matrices + per-class F1: LinearSVC vs LR |
| `top_words_per_category.png` | B | Highest-weight TF-IDF terms per news category |
| `high_dim_comparison.png` | C/Q1 | Which algorithms survive p >> n (100 features, 50 samples) |
| `overfitting_diagnosis.png` | C/Q3 | SVM train vs test accuracy sweep across C values |
| `algo_selection_guide.png` | D | Visual table: scenario → best algorithm → avoid |

---

## 🧠 Key Concepts Covered

- SVM kernel selection · C/gamma tuning · support vectors
- KNN optimal K · curse of dimensionality · scaling importance
- Algorithm comparison methodology · paired t-test for model selection
- TF-IDF pipeline · sparse text classification · LinearSVC
- Overfitting diagnosis · bias-variance tradeoff
- 8 algorithms: LR · Ridge · DT · RF · GBM · XGBoost · SVM · KNN

---

## 📬 Submission

```bash
git add .
git commit -m "D33 PM: SVM, KNN & Full Week Comparison — complete"
git push origin main
```

Share GitHub link in **Slack #daily-standup** by **09:15 AM next day**.

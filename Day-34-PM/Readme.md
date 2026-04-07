# Day 34 PM — PCA, Clustering & Week 6 Comprehensive Review

**Week 6 | PM Session | IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**

---

## 📋 Assignment Overview

Build a **complete Week 6 Review Notebook** covering all 13 algorithms/techniques, plus image compression with PCA, and a full ML pipeline question.

| Detail | Info |
|--------|------|
| Dataset | Wine dataset (sklearn), any image for Part B |
| Topics | PCA, all Week 6 supervised & unsupervised algorithms |
| Estimated Time | 90–120 minutes |
| Due | Next Day · 09:15 AM |

---

## 🗂️ File Structure

```
D34_PM_PCA/
├── README.md
├── d34_pm_pca_review.ipynb        # Main notebook
└── sample_image.jpg               # Any image for Part B (or use sklearn.datasets.load_sample_image)
```

---

## ⚙️ Setup & Run Steps

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter pillow
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook d34_pm_pca_review.ipynb
```

---

## 🧪 Solution Walkthrough

### Part A — Week 6 Algorithm Quick Reference (40%)

#### Complete Algorithm Reference (13 algorithms)

```python
# =============================================
# ALGORITHM QUICK REFERENCE - WEEK 6
# =============================================

"""
1. LOGISTIC REGRESSION (LR)
   When:  Binary/multi-class, linear boundaries, need probability output
   Code:  LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
   Params: C (regularization strength), solver
   Use:   Credit risk scoring, spam detection

2. DECISION TREE (DT)
   When:  Need interpretability, non-linear boundaries, mixed features
   Code:  DecisionTreeClassifier(max_depth=5, min_samples_split=10).fit(X, y)
   Params: max_depth, min_samples_split
   Use:   Loan approval, medical diagnosis rules

3. RANDOM FOREST (RF)
   When:  Tabular data, high accuracy, can sacrifice interpretability
   Code:  RandomForestClassifier(n_estimators=200, max_features='sqrt').fit(X, y)
   Params: n_estimators, max_features
   Use:   Fraud detection, customer churn

4. ADABOOST
   When:  Weak learners need boosting, binary classification
   Code:  AdaBoostClassifier(n_estimators=100, learning_rate=1.0).fit(X, y)
   Params: n_estimators, learning_rate
   Use:   Face detection (historical), binary classification

5. XGBOOST
   When:  Best accuracy on tabular data, handles missing values
   Code:  XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6).fit(X, y)
   Params: n_estimators, learning_rate, max_depth
   Use:   Kaggle competitions, financial modeling

6. LIGHTGBM
   When:  Large datasets, need speed comparable to XGBoost
   Code:  LGBMClassifier(n_estimators=200, num_leaves=31).fit(X, y)
   Params: num_leaves, n_estimators
   Use:   Real-time ads ranking, large-scale classification

7. VOTING ENSEMBLE
   When:  Combine diverse models for robustness
   Code:  VotingClassifier([('lr', lr), ('rf', rf)], voting='soft').fit(X, y)
   Params: estimators, voting ('hard'/'soft')
   Use:   Competition ensembles, robust production systems

8. STACKING
   When:  Maximum accuracy, heterogeneous base models
   Code:  StackingClassifier([('rf', rf), ('svm', svm)], final_estimator=lr).fit(X, y)
   Params: estimators, final_estimator
   Use:   High-stakes prediction, Kaggle grand prix

9. SVM
   When:  High-dimensional data, clear margin, small-medium datasets
   Code:  SVC(kernel='rbf', C=10, gamma='scale').fit(X, y)
   Params: C, gamma, kernel
   Use:   Text classification, bioinformatics, image recognition

10. KNN
    When:  Simple baseline, local patterns matter, small datasets
    Code:  KNeighborsClassifier(n_neighbors=5, metric='euclidean').fit(X, y)
    Params: n_neighbors, metric
    Use:   Recommendation, anomaly detection, simple classification

11. K-MEANS
    When:  Unsupervised, spherical clusters, known K
    Code:  KMeans(n_clusters=3, init='k-means++').fit(X)
    Params: n_clusters, init
    Use:   Customer segmentation, image quantization

12. DBSCAN
    When:  Unsupervised, arbitrary shapes, noise/outliers present
    Code:  DBSCAN(eps=0.5, min_samples=5).fit(X)
    Params: eps, min_samples
    Use:   Geospatial clustering, anomaly detection

13. PCA
    When:  Dimensionality reduction, visualization, noise reduction
    Code:  PCA(n_components=0.95).fit_transform(X)   # 95% variance
    Params: n_components, whiten
    Use:   Feature compression, preprocessing before slow models
"""
```

#### Algorithm Selection Flowchart (Text-Based)
```
START: What type of problem?
│
├── SUPERVISED (have labels)
│   │
│   ├── CLASSIFICATION
│   │   ├── Need interpretability?    → Decision Tree / Logistic Regression
│   │   ├── Best accuracy?            → XGBoost / LightGBM / Stacking
│   │   ├── High-dimensional text?    → SVM (linear) / Logistic Regression
│   │   ├── Small dataset?            → SVM / KNN
│   │   └── Large dataset, speed?    → LightGBM / Random Forest
│   │
│   └── REGRESSION
│       ├── Linear relationship?      → Linear Regression
│       └── Non-linear?               → XGBoost / Random Forest / SVM(RBF)
│
└── UNSUPERVISED (no labels)
    ├── Known number of clusters?    → K-Means
    ├── Arbitrary shaped clusters?   → DBSCAN
    ├── Hierarchical structure?      → AgglomerativeClustering
    └── Reduce dimensions?           → PCA
```

#### Test 3 Algorithms on Wine Dataset
```python
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

wine = load_wine()
X_w, y_w = wine.data, wine.target

algorithms = {
    'LR':  LogisticRegression(max_iter=1000),
    'RF':  RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', C=10, gamma='scale')
}

for name, model in algorithms.items():
    pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
    scores = cross_val_score(pipe, X_w, y_w, cv=5, scoring='accuracy')
    print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

### Part B — Stretch Problem (30%): Image Compression with PCA

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.decomposition import PCA

# Use sklearn's sample image (no file needed)
from sklearn.datasets import load_sample_image
china = load_sample_image('china.jpg')

# Work on one channel (grayscale)
gray = china.mean(axis=2)
print(f"Original shape: {gray.shape}")

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
axes[0].imshow(gray, cmap='gray'); axes[0].set_title('Original')

for i, n_comp in enumerate([5, 20, 50, 100]):
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(gray)
    reconstructed = pca.inverse_transform(transformed)
    
    mse = np.mean((gray - reconstructed) ** 2)
    original_size = gray.shape[0] * gray.shape[1]
    compressed_size = n_comp * (gray.shape[0] + gray.shape[1])
    ratio = original_size / compressed_size
    
    axes[i+1].imshow(reconstructed, cmap='gray')
    axes[i+1].set_title(f'n={n_comp}\nMSE={mse:.1f}\nRatio={ratio:.1f}x')

plt.tight_layout()
plt.show()
```

---

### Part C — Interview Ready (20%)

**Q1 — Complete ML Pipeline (1000 samples, 200 features):**

```
Step 1: EDA & Cleaning
  → Pandas profiling, handle missing values, outliers

Step 2: Preprocessing
  → StandardScaler (required for SVM/KNN/PCA)
  → Encode categoricals

Step 3: Dimensionality Reduction (200 features → manageable)
  → PCA(n_components=0.95): Reduce noise, speed up training
  WHY: 200 features risk curse of dimensionality

Step 4: Baseline Model
  → Logistic Regression: Fast, interpretable benchmark
  WHY: Understand if problem is linearly separable

Step 5: Advanced Models
  → Random Forest: Handle non-linearity, feature importance
  WHY: Robust to outliers, built-in feature selection
  → XGBoost: Best tabular accuracy
  WHY: Regularization + boosting handles complex patterns

Step 6: Model Selection
  → 5-fold CV, compare AUC-ROC
  → Paired t-test for statistical significance

Step 7: Deployment
  → Serialize with joblib, wrap in REST API
```

**Q2 — Weekly Model Comparison Function:**
```python
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd

def weekly_model_comparison(X, y, use_pca=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    import xgboost as xgb
    
    models = {
        'LR':       LogisticRegression(max_iter=1000),
        'RF':       RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost':  xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'SVM':      SVC(kernel='rbf', C=10, gamma='scale'),
        'KNN':      KNeighborsClassifier(n_neighbors=5),
    }
    
    results = []
    for name, model in models.items():
        steps = [('scaler', StandardScaler())]
        if use_pca:
            steps.append(('pca', PCA(n_components=0.95)))
        steps.append(('model', model))
        
        pipe = Pipeline(steps)
        scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
        results.append({'Model': name, 'Mean_CV': scores.mean(), 'Std_CV': scores.std()})
    
    return pd.DataFrame(results).sort_values('Mean_CV', ascending=False).reset_index(drop=True)

df_results = weekly_model_comparison(X_w, y_w, use_pca=True)
print(df_results)
```

**Q3 — PCA reduces 100→10 features, accuracy drops 0.92→0.85. Why?**

Three reasons:
1. **The dropped 90 features (5% variance) contain discriminative signal** — low-variance features are not necessarily low-information features (e.g., a rare but perfectly discriminative binary flag)
2. **PCA is unsupervised** — it maximizes variance without considering class separability. A supervised method (LDA) or feature selection might preserve accuracy better
3. **Model complexity mismatch** — the original model (RF/SVM) was tuned for 100 features; with 10 PCA components, it may need re-tuning of hyperparameters

---

### Part D — AI-Augmented Task (10%)

Prompt for Saturday assessment study guide:

> *"Create a structured Week 6 ML study guide covering these 13 topics: Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost, LightGBM, Voting, Stacking, SVM, KNN, K-Means, DBSCAN, PCA. For each: 2-sentence description, 2 common interview questions with answers, and 1 code pattern."*

After receiving the output, verify each section by cross-checking against your notebooks. Flag any that:
- Get hyperparameter defaults wrong
- Confuse algorithm mechanics (e.g., DBSCAN eps vs K-Means K)
- Miss key interview angles (e.g., curse of dimensionality for KNN)

---

## 📤 Submission

1. Push notebook to GitHub
2. Share repository link in Slack **#daily-standup**

---

## ✅ Evaluation Rubric

| Criteria | Weight |
|----------|--------|
| Correctness (all 13 algorithms, working code snippets, accurate flowchart) | 40% |
| Code Quality (clean, organized, reusable as reference) | 25% |
| Understanding (thorough pipeline description, PCA accuracy drop analysis) | 20% |
| AI Usage (comprehensive study guide, verified and improved) | 15% |

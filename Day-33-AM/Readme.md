# Day 33 AM — SVM & KNN: Handwritten Digit Classifier

**Week 6 | AM Session | IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**

---

## 📋 Assignment Overview

Build a **handwritten digit classifier** using SVM and KNN on the sklearn digits dataset — one of the most classic ML problems.

| Detail | Info |
|--------|------|
| Dataset | `sklearn.datasets.load_digits` (built-in) |
| Topics | Max-margin classifier, C/gamma tuning, kernel trick, KNN, distance metrics, curse of dimensionality |
| Estimated Time | 60–90 minutes |
| Due | Next Day · 09:15 AM |

---

## 🗂️ File Structure

```
D33_AM_SVM_KNN/
├── README.md
├── d33_am_svm_knn.ipynb           # Main notebook
└── faiss_comparison.md            # Part B findings
```

---

## ⚙️ Setup & Run Steps

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
# For Part B (stretch):
pip install faiss-cpu
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook d33_am_svm_knn.ipynb
```

No external dataset needed — `load_digits` is built into sklearn.

---

## 🧪 Solution Walkthrough

### Part A — Concept Application (40%)

#### Step 1: Load & Scale Dataset
```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

digits = load_digits()
X, y = digits.data, digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### Step 2: SVM with GridSearchCV (RBF kernel)
```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

svm = SVC(kernel='rbf', probability=True)
grid = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

best_svm = grid.best_estimator_
print("Best params:", grid.best_params_)
print("SVM Accuracy:", best_svm.score(X_test, y_test))
# Expected: SVM(RBF, C=10, gamma=0.001): Accuracy ≈ 0.98
```

#### Step 3: KNN with Optimal K
```python
from sklearn.neighbors import KNeighborsClassifier

# Find best K
k_scores = []
for k in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    k_scores.append(knn.score(X_test, y_test))

best_k = k_scores.index(max(k_scores)) + 1
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
print(f"KNN (K={best_k}) Accuracy:", knn.score(X_test, y_test))
# Expected: KNN(K=3): Accuracy ≈ 0.97
```

#### Step 4: Compare — Confusion Matrix & Per-Class F1
```python
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

for name, model in [('SVM', best_svm), ('KNN', knn)]:
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=digits.target_names)
    disp.plot(cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.show()
```

#### Step 5: Identify Most Confused Digit Pairs
```python
import numpy as np

y_pred_svm = best_svm.predict(X_test)
cm = confusion_matrix(y_test, y_pred_svm)
np.fill_diagonal(cm, 0)  # zero out correct predictions

most_confused = np.unravel_index(np.argsort(cm, axis=None)[-5:], cm.shape)
for i, j in zip(*most_confused):
    print(f"Digit {i} confused with Digit {j}: {cm[i,j]} times")
# Expected: (3,8), (4,9), (1,7)
```

---

### Part B — Stretch Problem (30%): FAISS vs sklearn KNN

```python
import faiss
import numpy as np
import time

# Prepare data (FAISS needs float32)
X_train_f = X_train.astype('float32')
X_test_f  = X_test.astype('float32')

# Build FAISS index
index = faiss.IndexFlatL2(X_train_f.shape[1])
index.add(X_train_f)

# FAISS search (K=3)
start = time.time()
D, I = index.search(X_test_f, 3)
faiss_time = time.time() - start

# sklearn KNN timing
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(X_train, y_train)
start = time.time()
knn3.predict(X_test)
sklearn_time = time.time() - start

print(f"sklearn KNN: {sklearn_time*1000:.2f} ms")
print(f"FAISS:       {faiss_time*1000:.2f} ms")
```

Save findings to `faiss_comparison.md`:
- FAISS is 5–50x faster at scale (billions of vectors)
- Used in Instagram, Spotify, Pinterest for recommendation systems
- Approximate nearest neighbors — trades tiny accuracy loss for massive speed gain

---

### Part C — Interview Ready (20%)

**Q1 — SVM vs Logistic Regression:**
Both find linear decision boundaries, but:
- **LR** minimizes log-loss using all data points
- **SVM** maximizes margin using only **support vectors** (nearest points to boundary)
- Prefer SVM when data has clear margin, prefer LR when you need probability outputs or faster training on large data

**Q2 — KNN from Scratch:**
```python
import numpy as np
from collections import Counter

def knn_from_scratch(X_train, y_train, X_test, k):
    predictions = []
    for test_point in X_test:
        # Euclidean distances
        distances = np.sqrt(np.sum((X_train - test_point) ** 2, axis=1))
        # K nearest indices
        k_indices = np.argsort(distances)[:k]
        # Majority vote
        k_labels = y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)[0][0]
        predictions.append(most_common)
    return np.array(predictions)
```

**Q3 — Debug (SVM 0.50 accuracy):**
The features have **very different scales** (salary: 50K–200K vs age: 20–60). SVM with RBF kernel is extremely sensitive to feature scales — the salary dimension dominates the distance calculation, making the kernel useless.

**Fix:** Apply `StandardScaler` before fitting the SVM:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
svm.fit(X_train_scaled, y_train)
```

---

### Part D — AI-Augmented Task (10%)

Use this prompt:

> *"Write matplotlib code showing how the SVM decision boundary changes as C varies from 0.01 to 100 on a 2D dataset. Show plots side by side for C = 0.01, 0.1, 1, 10, 100."*

Then use this prompt for kernel trick analogy:

> *"Explain the SVM kernel trick using a simple analogy — for example, how projecting 1D data into 2D can make it linearly separable."*

Evaluate:
- Does the visualization clearly show the margin widening as C decreases?
- Is the analogy accurate and easy to grasp?

---

## 📤 Submission

1. Push notebook to GitHub
2. Share repository link in Slack **#daily-standup**

---

## ✅ Evaluation Rubric

| Criteria | Weight |
|----------|--------|
| Correctness (both classifiers work, confusion analysis, fair comparison) | 40% |
| Code Quality (clean, reusable, informative visualizations) | 25% |
| Understanding (SVM vs LR distinction, debug catches missing scaling) | 20% |
| AI Usage (visualization works, analogy is vetted) | 15% |

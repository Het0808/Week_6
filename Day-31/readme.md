# Day 30 — Logistic Regression & End-to-End ML Pipeline

**Week 6 | AM + PM Session | IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**

---

## 📋 Assignment Overview

Build a complete end-to-end ML pipeline using **Logistic Regression** on the SUV Purchase Dataset.

| Detail | Info |
|--------|------|
| Dataset | [SUV Purchase Dataset (Kaggle)](https://www.kaggle.com/datasets/bittupanchal/logistics-regression-on-suv-dataset) |
| Topics | Logistic Regression, Preprocessing, Train-Test Split, Feature Scaling |
| Estimated Time | 75–90 minutes |
| Due | Day 31 · 09:15 AM |

---

## 🗂️ File Structure

```
D30_Logistic_Regression/
├── README.md
├── d30_logistic_regression.ipynb   # Main notebook
└── data/
    └── suv_data.csv                # Download from Kaggle
```

---

## ⚙️ Setup & Run Steps

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 2. Download Dataset

- Go to: https://www.kaggle.com/datasets/bittupanchal/logistics-regression-on-suv-dataset
- Download `Social_Network_Ads.csv` (or equivalent)
- Place it inside a `data/` folder

### 3. Launch Jupyter Notebook

```bash
jupyter notebook d30_logistic_regression.ipynb
```

---

## 🧪 Solution Walkthrough

### Part A — Concept Application (40%)

#### Step 1: Data Loading & Exploration
```python
import pandas as pd

df = pd.read_csv('data/suv_data.csv')
print(df.head())
print(df.shape)
print(df.dtypes)
print(df.isnull().sum())
```

#### Step 2: Data Preprocessing
```python
from sklearn.preprocessing import LabelEncoder

# Encode Gender
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# Features & target
X = df[['Age', 'EstimatedSalary']]
y = df['Purchased']
```

#### Step 3: Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Step 4: Feature Scaling
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

#### Step 5: Model Training
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

---

### Part B — Stretch Problem (30%)

#### Model Evaluation
```python
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

#### Try Different Test Sizes
```python
for test_size in [0.20, 0.25, 0.30]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = LogisticRegression()
    model.fit(X_train_s, y_train)
    print(f"Test size {test_size}: Accuracy = {accuracy_score(y_test, model.predict(X_test_s)):.4f}")
```

---

### Part C — Interview Ready (20%)

**Q1:** Logistic Regression is a **classification** algorithm despite its name. It uses the sigmoid function to map predicted values to probabilities between 0 and 1.

**Q2 — Code:**
```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

**Q3:** A confusion matrix is a 2×2 table for binary classification showing:
- **TP** (True Positives), **TN** (True Negatives)
- **FP** (False Positives), **FN** (False Negatives)

It helps evaluate precision, recall, and F1 beyond just accuracy.

---

### Part D — AI-Augmented Task (10%)

Use this prompt with an AI tool:

> *"Explain Logistic Regression with a Python example using sklearn on the SUV dataset. Include data loading, preprocessing, training, and evaluation steps."*

Then validate the AI output:
- Is every sklearn step correct?
- Are preprocessing steps complete (scaling before model)?
- Does the code run without errors?

---

## 📤 Submission

1. Push notebook to GitHub
2. Share repository link in Slack **#daily-standup**

---

## ✅ Evaluation Rubric

| Criteria | Weight |
|----------|--------|
| Correctness (complete ML pipeline) | 40% |
| Code Quality (clean, modular) | 25% |
| Understanding (explanation of model & results) | 20% |
| AI Usage (prompt + validated output) | 15% |

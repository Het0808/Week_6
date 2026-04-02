"""
============================================================
  SUV Purchase Prediction — Logistic Regression Assignment
  Dataset: Social Network Ads (SUV Purchase Dataset)
============================================================
  Parts Covered:
    A — Data Loading, Preprocessing, Splitting, Scaling, Training
    B — Evaluation, Visualization, Accuracy Comparison
    C — Interview Q&A (inline comments)
    D — AI-Augmented Task Documentation
"""

# ─────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PART A — CONCEPT APPLICATION
# ─────────────────────────────────────────────

# ── A1: Data Loading & Exploration ──────────
print("=" * 60)
print("  PART A — DATA LOADING & EXPLORATION")
print("=" * 60)

df = pd.read_csv("Social_Network_Ads.csv")

print("\n[1] First 5 Rows:")
print(df.head())

print(f"\n[2] Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

print("\n[3] Column Names:")
print(list(df.columns))

print("\n[4] Data Types:")
print(df.dtypes)

print("\n[5] Missing Values:")
print(df.isnull().sum())

print("\n[6] Basic Statistics:")
print(df.describe())

# ── A2: Data Preprocessing ──────────────────
print("\n" + "=" * 60)
print("  PART A — DATA PREPROCESSING")
print("=" * 60)

# Handle missing values
print("\n[1] Handling Missing Values...")
if df.isnull().sum().sum() == 0:
    print("    ✓ No missing values found. Dataset is clean.")
else:
    df.dropna(inplace=True)
    print(f"    Rows after dropping NaN: {len(df)}")

# Encode categorical variable: Gender → 0/1
print("\n[2] Encoding Categorical Variable 'Gender'...")
le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])   # Male=1, Female=0
print("    Encoding map:", dict(zip(le.classes_, le.transform(le.classes_))))
print(df[["Gender"]].value_counts().to_string())

# Select features and target
print("\n[3] Selecting Features: Age, EstimatedSalary → Target: Purchased")
X = df[["Age", "EstimatedSalary"]].values
y = df["Purchased"].values
print(f"    X shape: {X.shape} | y shape: {y.shape}")
print(f"    Class distribution → 0 (Not Purchased): {(y==0).sum()}  |  1 (Purchased): {(y==1).sum()}")

# ── A3: Train-Test Split ─────────────────────
print("\n" + "=" * 60)
print("  PART A — TRAIN-TEST SPLIT  (80 / 20)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Testing  samples : {X_test.shape[0]}")

# ── A4: Feature Scaling ─────────────────────
print("\n" + "=" * 60)
print("  PART A — FEATURE SCALING (StandardScaler)")
print("=" * 60)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit only on training data
X_test_sc  = scaler.transform(X_test)        # transform test with training params

print(f"  Before Scaling — Age mean: {X_train[:,0].mean():.2f}, std: {X_train[:,0].std():.2f}")
print(f"  After  Scaling — Age mean: {X_train_sc[:,0].mean():.4f}, std: {X_train_sc[:,0].std():.4f}")

# ── A5: Model Training ───────────────────────
print("\n" + "=" * 60)
print("  PART A — LOGISTIC REGRESSION MODEL TRAINING")
print("=" * 60)

model = LogisticRegression(random_state=42)
model.fit(X_train_sc, y_train)

print("  ✓ Model trained successfully!")
print(f"  Intercept  : {model.intercept_[0]:.4f}")
print(f"  Coefficients → Age: {model.coef_[0][0]:.4f}  |  EstimatedSalary: {model.coef_[0][1]:.4f}")

# ─────────────────────────────────────────────
#  PART B — STRETCH PROBLEM
# ─────────────────────────────────────────────

# ── B1: Model Evaluation ────────────────────
print("\n" + "=" * 60)
print("  PART B — MODEL EVALUATION")
print("=" * 60)

y_pred = model.predict(X_test_sc)

acc = accuracy_score(y_test, y_pred)
cm  = confusion_matrix(y_test, y_pred)

print(f"\n  Accuracy  : {acc * 100:.2f}%")
print("\n  Confusion Matrix:")
print(f"  {'':>20} Predicted 0   Predicted 1")
print(f"  Actual 0   {'|':>5}  {cm[0][0]:>7}   {'|':>3}  {cm[0][1]:>7}   {'|':>3}")
print(f"  Actual 1   {'|':>5}  {cm[1][0]:>7}   {'|':>3}  {cm[1][1]:>7}   {'|':>3}")

TP = cm[1][1]; TN = cm[0][0]
FP = cm[0][1]; FN = cm[1][0]
print(f"\n  TP (True Positives)  = {TP}  — Correctly predicted Purchased")
print(f"  TN (True Negatives)  = {TN}  — Correctly predicted Not Purchased")
print(f"  FP (False Positives) = {FP}   — Predicted Purchased but wasn't")
print(f"  FN (False Negatives) = {FN}   — Predicted Not Purchased but was")

print("\n  Full Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Purchased","Purchased"]))

# ── B2: Visualizations ──────────────────────
def plot_decision_boundary(X_sc, y, model, scaler, title, ax):
    """Plot 2D decision boundary for logistic regression."""
    h = 0.01
    x_min, x_max = X_sc[:, 0].min() - 1, X_sc[:, 0].max() + 1
    y_min, y_max = X_sc[:, 1].min() - 1, X_sc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap="RdYlGn")
    scatter = ax.scatter(X_sc[:, 0], X_sc[:, 1],
                         c=y, cmap="RdYlGn", edgecolors="k", s=30)
    ax.set_xlabel("Age (Scaled)")
    ax.set_ylabel("EstimatedSalary (Scaled)")
    ax.set_title(title)
    red_patch   = mpatches.Patch(color="#d73027", label="Not Purchased (0)")
    green_patch = mpatches.Patch(color="#1a9850", label="Purchased (1)")
    ax.legend(handles=[red_patch, green_patch], loc="upper left", fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Logistic Regression — Decision Boundary", fontsize=14, fontweight="bold")

# Train boundary
X_all_sc = scaler.transform(X)
plot_decision_boundary(X_all_sc, y, model, scaler,
                        "Decision Boundary (All Data)", axes[0])

# Confusion matrix heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Not Purchased", "Purchased"])
disp.plot(ax=axes[1], colorbar=False, cmap="Blues")
axes[1].set_title("Confusion Matrix — 80/20 Split")

plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ Saved: decision_boundary.png")

# ── B3: Improvement — Compare Test Sizes ────
print("\n" + "=" * 60)
print("  PART B — ACCURACY COMPARISON ACROSS SPLITS")
print("=" * 60)

splits = [("80/20", 0.20), ("75/25", 0.25), ("70/30", 0.30)]
results = []

for label, ts in splits:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=42)
    sc_  = StandardScaler()
    Xtr_ = sc_.fit_transform(Xtr)
    Xte_ = sc_.transform(Xte)
    m_   = LogisticRegression(random_state=42)
    m_.fit(Xtr_, ytr)
    a_   = accuracy_score(yte, m_.predict(Xte_))
    results.append((label, len(Xtr), len(Xte), round(a_ * 100, 2)))
    print(f"  Split {label} → Train: {len(Xtr):>3}  Test: {len(Xte):>3}  Accuracy: {a_*100:.2f}%")

# Bar chart comparing accuracies
fig2, ax2 = plt.subplots(figsize=(7, 4))
labels_   = [r[0] for r in results]
accs_     = [r[3] for r in results]
bars = ax2.bar(labels_, accs_, color=["#2166ac","#4dac26","#d01c8b"], width=0.4)
for bar, acc in zip(bars, accs_):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{acc}%", ha="center", va="bottom", fontweight="bold")
ax2.set_ylim(80, 95)
ax2.set_xlabel("Train/Test Split")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Accuracy Comparison — Different Split Ratios")
ax2.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  ✓ Saved: accuracy_comparison.png")

# ─────────────────────────────────────────────
#  PART C — INTERVIEW READY (inline answers)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PART C — INTERVIEW QUESTIONS")
print("=" * 60)

print("""
Q1 — What is Logistic Regression? Is it classification or regression?
──────────────────────────────────────────────────────────────────────
  Logistic Regression is a CLASSIFICATION algorithm despite having
  "regression" in its name. It predicts the PROBABILITY that an input
  belongs to a class (0 or 1) using the sigmoid function:

      σ(z) = 1 / (1 + e^(−z))     where z = w₀ + w₁x₁ + w₂x₂ + ...

  Output is always between 0 and 1. A threshold (usually 0.5) decides
  the final class label. It is binary classification when there are
  two classes, like Purchased (1) vs Not Purchased (0).

Q3 — What is a Confusion Matrix? What does it represent?
──────────────────────────────────────────────────────────
  A confusion matrix is a 2×2 table that summarises classification
  performance by comparing actual vs predicted labels:

        │  Predicted 0  │  Predicted 1  │
  ──────┼───────────────┼───────────────┤
  Act 0 │  TN (correct) │  FP (wrong)   │
  Act 1 │  FN (wrong)   │  TP (correct) │

  TP = True Positive  → Predicted 1, actually 1
  TN = True Negative  → Predicted 0, actually 0
  FP = False Positive → Predicted 1, actually 0  (Type I Error)
  FN = False Negative → Predicted 0, actually 1  (Type II Error)

  From the matrix we derive Accuracy, Precision, Recall, F1-Score.
""")

# Q2 — Code: Train-test split and scaling
print("Q2 — Code: Train-Test Split & Scaling")
print("─" * 40)
q2_code = """
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing   import StandardScaler

  # Split (80 train / 20 test)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.20, random_state=42
  )

  # Scale features (fit ONLY on train, transform both)
  scaler      = StandardScaler()
  X_train_sc  = scaler.fit_transform(X_train)
  X_test_sc   = scaler.transform(X_test)
"""
print(q2_code)

# ─────────────────────────────────────────────
#  PART D — AI-AUGMENTED TASK
# ─────────────────────────────────────────────
print("=" * 60)
print("  PART D — AI-AUGMENTED TASK DOCUMENTATION")
print("=" * 60)
print("""
  PROMPT USED:
  ────────────
  "Explain Logistic Regression with Python example using sklearn
   on SUV dataset."

  AI OUTPUT EVALUATION:
  ─────────────────────
  ✅ Code correctness  : The AI-generated pipeline was verified and
                          is functionally correct. Steps matched
                          sklearn best practices.
  ✅ Steps complete    : All key ML stages present — loading,
                          encoding, splitting, scaling, training,
                          evaluation (accuracy + confusion matrix).
  ⚠  Minor issues noted:
       • AI did not explicitly mention fitting scaler on training
         data only — a critical best practice to prevent data leakage.
       • AI did not include feature importance / coefficient analysis.
       • Decision boundary visualisation was missing.
  ✅ Conclusion        : AI output is a solid starting point but
                          requires human review to ensure correctness
                          of scaling strategy and completeness of
                          evaluation metrics.
""")

print("=" * 60)
print("  ALL PARTS COMPLETE — Assignment Finished Successfully!")
print("=" * 60)

# Day 34 AM — Clustering: K-Means & DBSCAN

**Week 6 | AM Session | IIT Gandhinagar — PG Diploma in AI-ML & Agentic AI Engineering**

---

## 📋 Assignment Overview

Segment the **Iris dataset without labels** using K-Means and DBSCAN, then compare recovered clusters against true species.

| Detail | Info |
|--------|------|
| Dataset | `sklearn.datasets.load_iris` (built-in) |
| Topics | K-Means, elbow method, silhouette score, DBSCAN, ARI, NMI, cluster evaluation |
| Estimated Time | 60–90 minutes |
| Due | Next Day · 09:15 AM |

---

## 🗂️ File Structure

```
D34_AM_Clustering_PCA/
├── README.md
├── d34_am_clustering.ipynb        # Main notebook
```

---

## ⚙️ Setup & Run Steps

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook d34_am_clustering.ipynb
```

No external dataset needed — Iris is built into sklearn.

---

## 🧪 Solution Walkthrough

### Part A — Concept Application (40%)

#### Step 1: Load Iris, Drop Target, Scale
```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

iris = load_iris()
X = iris.data
y_true = iris.target  # Keep for comparison only

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### Step 2: K-Means with K=3 & Label Comparison
```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Confusion-matrix-like table
import pandas as pd
comparison_df = pd.crosstab(y_true, y_kmeans,
                             rownames=['True Species'],
                             colnames=['KMeans Cluster'])
print(comparison_df)
```

#### Step 3: ARI and NMI
```python
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari = adjusted_rand_score(y_true, y_kmeans)
nmi = normalized_mutual_info_score(y_true, y_kmeans)
print(f"ARI = {ari:.4f}")  # Expected: ~0.73
print(f"NMI = {nmi:.4f}")
```

#### Step 4: Visualize True Labels vs K-Means
```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='Set1', edgecolors='k', s=60)
axes[0].set_title('True Species (PCA 2D)')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='Set2', edgecolors='k', s=60)
axes[1].set_title('K-Means Clusters (PCA 2D)')
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```

#### Step 5: Apply DBSCAN
```python
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Tune eps and min_samples
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X_scaled)

n_clusters = len(set(y_dbscan)) - (1 if -1 in y_dbscan else 0)
n_noise    = list(y_dbscan).count(-1)
print(f"DBSCAN clusters: {n_clusters}, Noise points: {n_noise}")

if n_clusters > 1:
    mask = y_dbscan != -1
    print(f"Silhouette: {silhouette_score(X_scaled[mask], y_dbscan[mask]):.4f}")

ari_dbscan = adjusted_rand_score(y_true[y_dbscan != -1], y_dbscan[y_dbscan != -1])
print(f"DBSCAN ARI (non-noise): {ari_dbscan:.4f}")
```

#### Step 6: Write Answer — What Does Agreement Tell Us?
> *"When unsupervised clustering (K-Means, ARI ≈ 0.73) agrees strongly with known labels, it tells us that the classes are **naturally separable** in feature space — meaning the input features alone carry enough information to distinguish the groups. For Iris, petal length/width are so discriminative that an algorithm with no label information nearly recovers the true species. High agreement validates both the feature quality and the natural structure of the data."*

---

### Part B — Stretch Problem (30%): Hierarchical Clustering

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
y_agg = agg.fit_predict(X_scaled)
print(f"Agglomerative ARI: {adjusted_rand_score(y_true, y_agg):.4f}")

# Dendrogram (on a subset for readability)
linked = linkage(X_scaled[:50], method='ward')
plt.figure(figsize=(14, 6))
dendrogram(linked, truncate_mode='lastp', p=12, leaf_rotation=45, leaf_font_size=10)
plt.title('Hierarchical Clustering Dendrogram (first 50 samples)')
plt.xlabel('Sample index / cluster size')
plt.ylabel('Distance')
plt.tight_layout()
plt.show()

# Compare ARI
print(f"K-Means ARI:        {ari:.4f}")
print(f"Agglomerative ARI:  {adjusted_rand_score(y_true, y_agg):.4f}")
```

---

### Part C — Interview Ready (20%)

**Q1 — K-Means as 'greedy' algorithm:**
K-Means is greedy because at each step it makes the locally optimal assignment (nearest centroid) without considering the global optimum. Yes, it **can get stuck in local minima** depending on initial centroid placement.

**KMeans++ helps** by seeding centroids with probability proportional to their distance from already-chosen centroids — spreading them out to avoid clustered initializations. In sklearn: `KMeans(init='k-means++')` (default).

**Q2 — K-Means from Scratch:**
```python
import numpy as np

def kmeans(X, k, max_iter=100):
    # KMeans++ initialization
    np.random.seed(42)
    centroids = [X[np.random.randint(len(X))]]
    for _ in range(k - 1):
        dists = np.min([np.linalg.norm(X - c, axis=1) for c in centroids], axis=0)
        probs = dists ** 2 / (dists ** 2).sum()
        centroids.append(X[np.random.choice(len(X), p=probs)])
    centroids = np.array(centroids)
    
    for _ in range(max_iter):
        # Assign
        dists = np.array([np.linalg.norm(X - c, axis=1) for c in centroids])
        labels = np.argmin(dists, axis=0)
        # Update
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

labels, centroids = kmeans(X_scaled, k=3)
```

**Q3 — Silhouette Score 0.25 with K=5:**
A silhouette of 0.25 is **weak** (range is -1 to 1; good clustering > 0.5). Things to investigate:
- Try different K values (use elbow method and silhouette plots)
- Check if DBSCAN works better — data may have non-spherical clusters
- Visualize with PCA — clusters may heavily overlap
- Verify feature scaling — unscaled features bias K-Means

---

### Part D — AI-Augmented Task (10%)

Prompt:

> *"Explain the difference between K-Means, DBSCAN, and Hierarchical Clustering using an analogy of sorting different types of fruits at a market."*

Then follow up with:

> *"In your fruit-sorting analogy, when would each method fail?"*

Evaluate the response:
- Does K-Means map to "dividing fruits into K groups by average size/color"? ✅
- Does DBSCAN map to "grouping dense clusters of similar fruits, leaving unusual ones as outliers"? ✅
- Does Hierarchical map to "creating a family tree from most similar pairs upward"? ✅

Improve by checking if the analogy handles the *hyperparameter sensitivity* of each method.

---

## 📤 Submission

1. Push notebook to GitHub
2. Share repository link in Slack **#daily-standup**

---

## ✅ Evaluation Rubric

| Criteria | Weight |
|----------|--------|
| Correctness (ARI computed, DBSCAN tuned, comparison meaningful) | 40% |
| Code Quality (clean notebook, publication-ready visualizations) | 25% |
| Understanding (greedy explanation, silhouette nuance) | 20% |
| AI Usage (analogy creative and verified) | 15% |

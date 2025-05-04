import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# 產生三群二維資料
X, _ = make_blobs(
    n_samples=300,
    centers=[(-5,0),(0,5),(5,0)],
    cluster_std=1.0,
    random_state=42
)

# 定義三種分群算法
algos = {
    "KMeans (k=3)": KMeans(n_clusters=3, random_state=0),
    "DBSCAN": DBSCAN(eps=1.2, min_samples=5),
    "Agglomerative(3)": AgglomerativeClustering(n_clusters=3)
}

# 畫圖
fig, axes = plt.subplots(1, 3, figsize=(15,4))
for ax, (name, algo) in zip(axes, algos.items()):
    labels = algo.fit_predict(X)
    ax.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=20)
    ax.set_title(name)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

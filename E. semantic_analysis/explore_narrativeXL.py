#%%
import os
import pickle
import numpy as np
import json
from glob import glob

from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import umap
import pandas as pd
output_folder = 'C:/Users/mkersey/Documents/GitHub/Distinct-paths-to-false-memory-/E. semantic_analysis/explore_narrativeXL_result'
#%% load text windows
def load_texts(folder_path):
    files = glob(os.path.join(folder_path, "*.json"))
    all_texts = []
    for f in files:
        with open(f, 'r') as file:
            data = json.load(file)
        all_texts.extend(data)
    return all_texts
    
folder_path = "C:/Users/mkersey/Documents/GitHub/Distinct-paths-to-false-memory-/E. semantic_analysis/explore_narrativeXL_result/narrativeXL_textWindows"
y_text = load_texts(folder_path)
print(f"Loaded text shape: {len(y_text)}")
#%% load embeddings 
# ----------------------------
# 1. Load all embeddings
# ----------------------------
def load_embeddings(folder_path):
    files = glob(os.path.join(folder_path, "*.pkl"))
    
    all_embeddings = []
    
    for f in files:
        with open(f, "rb") as fp:
            data = pickle.load(fp)
            
            data = np.array(data)
            
            # Ensure 2D
            if data.ndim == 1:
                data = data.reshape(1, -1)
            
            all_embeddings.append(data)
    
    return np.vstack(all_embeddings)

folder_path = "C:/Users/mkersey/Documents/GitHub/Distinct-paths-to-false-memory-/semantic_features/all-mpnet-base-v2"
X = load_embeddings(folder_path)
print(f"Loaded embeddings shape: {X.shape}")
#%% PCA fitting with elbow method
import json
output_folder = 'C:/Users/mkersey/Documents/GitHub/Distinct-paths-to-false-memory-/E. semantic_analysis/explore_narrativeXL_result'
X_scaled = np.load(output_folder+'/X_scaled.npy')

# fit elbow
def fit_full_pca(X_scaled):
    pca_full = PCA(svd_solver="randomized", random_state=42)
    pca_full.fit(X_scaled)
    return pca_full


# ----------------------------
# 2. Find elbow (knee point)
# ----------------------------
def find_elbow(explained_variance):
    """
    Simple geometric elbow detection
    """
    n_points = len(explained_variance)
    x = np.arange(n_points)
    y = explained_variance

    # line from first to last point
    line_vec = np.array([x[-1] - x[0], y[-1] - y[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)

    distances = []
    for i in range(n_points):
        point_vec = np.array([x[i] - x[0], y[i] - y[0]])
        proj = np.dot(point_vec, line_vec) * line_vec
        orth = point_vec - proj
        distances.append(np.linalg.norm(orth))

    elbow_idx = np.argmax(distances)
    return elbow_idx + 1  # +1 for component count


# ----------------------------
# 3. Choose dimension
# ----------------------------
def choose_pca_dim(pca_full, variance_threshold=0.9, use_elbow=True):
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    
    # threshold-based
    dim_var = np.argmax(cumvar >= variance_threshold) + 1
    
    # elbow-based
    dim_elbow = find_elbow(cumvar)
    
    print(f"Dim (variance {variance_threshold*100:.0f}%): {dim_var}")
    print(f"Dim (elbow): {dim_elbow}")
    
    if use_elbow:
        return dim_elbow, cumvar
    else:
        return dim_var, cumvar


# ----------------------------
# 4. Plot
# ----------------------------
def plot_pca_curve(cumvar, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.plot(cumvar)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    
    plt.show()

def run_pca_with_selection(X_scaled, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Fit full PCA
    pca_full = fit_full_pca(X_scaled)
    
    # Choose dimension
    pca_dim, cumvar = choose_pca_dim(
        pca_full,
        variance_threshold=0.9,
        use_elbow=False  # <-- switch here
    )
    
    # Plot
    plot_pca_curve(cumvar,
                   save_path=os.path.join(output_folder, "pca_variance.png"))
    
    # Fit final PCA
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Save everything
    np.save(os.path.join(output_folder, "X_pca.npy"), X_pca)
    
    with open(os.path.join(output_folder, "pca.pkl"), "wb") as f:
        pickle.dump(pca, f)
    
    metadata = {
        "pca_dim": int(pca_dim),
        "explained_variance_total": float(cumvar[pca_dim - 1])
    }
    
    with open(os.path.join(output_folder, "pca_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    return X_pca, pca, cumvar
X_pca, pca, cumvar = run_pca_with_selection(X_scaled, output_folder)
#%% UMAP
from sklearn.manifold import trustworthiness
X_pca = np.load(output_folder+'/X_pca.npy')
candidate_dims = [5, 10, 15, 20, 25]
n_neighbors = 15
min_dist = 0.1
random_state = 42

sample_size = 10000   # try 10k-50k depending on memory
rng = np.random.default_rng(random_state)

n_total = X_pca.shape[0]
sample_idx = rng.choice(n_total, size=min(sample_size, n_total), replace=False)

X_pca_sample = X_pca[sample_idx]

umap_results = []
umap_models = {}

for d in candidate_dims:
    reducer = umap.UMAP(
        n_components=d,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        low_memory=True,
    )
    
    X_umap_sample = reducer.fit_transform(X_pca_sample)
    
    tw = trustworthiness(X_pca_sample, X_umap_sample, n_neighbors=n_neighbors)
    
    umap_results.append((d, tw))
    umap_models[d] = (reducer, X_umap_sample)

umap_results
#%% plotting to pic best UMAP dim
for d, tw in umap_results:
    print(f"UMAP dim = {d:2d}, trustworthiness = {tw:.4f}")

best_umap_dim = max(umap_results, key=lambda x: x[1])[0]
print("Chosen UMAP dim:", best_umap_dim)
#%% UMAP transformation
reducer = umap.UMAP(
    n_components=best_umap_dim,
    n_neighbors=n_neighbors,
    min_dist=min_dist,
    metric="euclidean",
    random_state=random_state,
    low_memory=True,
)

X_umap = reducer.fit_transform(X_pca)

print("Final X_umap shape:", X_umap.shape)
np.save(os.path.join(output_folder, "X_umap.npy"), X_umap)

with open(os.path.join(output_folder, "umap.pkl"), "wb") as f:
    pickle.dump(reducer, f)
#%% clustering
X_umap = np.load(output_folder+'/X_umap.npy')
random_state=42
k_range = range(2, 21)
k_values = list(k_range)

sweep_dir = os.path.join(output_folder, "kmeans_sweep")
model_dir = os.path.join(sweep_dir, "models")
os.makedirs(model_dir, exist_ok=True)

results_path = os.path.join(sweep_dir, "kmeans_sweep_results.csv")
state_path = os.path.join(sweep_dir, "kmeans_sweep_state.json")

# Optional: use a sample for silhouette to keep this manageable
silhouette_sample_size = 50000
rng = np.random.default_rng(random_state)
sample_idx = rng.choice(X_umap.shape[0], size=min(silhouette_sample_size, X_umap.shape[0]), replace=False)
X_umap_sample = X_umap[sample_idx]

# Resume support
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    completed_ks = set(results_df["k"].astype(int).tolist())
else:
    results_df = pd.DataFrame(columns=["k", "inertia", "silhouette"])
    completed_ks = set()

for k in k_values:
    if k in completed_ks:
        print(f"Skipping k={k} (already done)")
        continue

    print(f"Fitting k={k} ...")

    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=1024,
        n_init="auto",
    )

    labels_k = model.fit_predict(X_umap)

    inertia = float(model.inertia_)

    # Silhouette on a sample only, for feasibility
    sample_labels = labels_k[sample_idx]
    sil = float(silhouette_score(X_umap_sample, sample_labels))

    # Save model for this k
    model_path = os.path.join(model_dir, f"kmeans_k{k}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Append results immediately
    new_row = pd.DataFrame([{
        "k": k,
        "inertia": inertia,
        "silhouette": sil,
    }])

    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_csv(results_path, index=False)

    # Save small state file too
    state = {
        "completed_ks": sorted(list(completed_ks | {k})),
        "random_state": random_state,
        "silhouette_sample_size": int(silhouette_sample_size),
    }
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)

    completed_ks.add(k)

    print(f"Saved k={k}: inertia={inertia:.2f}, silhouette={sil:.4f}")
#%% examine k means results
results_df = pd.read_csv(results_path)
results_df = results_df.sort_values("k").reset_index(drop=True)
plt.figure(figsize=(7, 4))
plt.plot(results_df["k"], results_df["inertia"], marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("KMeans Elbow Curve")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 4))
plt.plot(results_df["k"], results_df["silhouette"], marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores by k")
plt.tight_layout()
plt.show()
#%% choose optimal k
def elbow_from_curve(values):
    y = np.array(values, dtype=float)
    x = np.arange(len(y), dtype=float)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line_vec = p2 - p1
    line_vec = line_vec / np.linalg.norm(line_vec)

    distances = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        proj_len = np.dot(p - p1, line_vec)
        proj = p1 + proj_len * line_vec
        distances.append(np.linalg.norm(p - proj))

    return int(np.argmax(distances))

k_list = results_df["k"].tolist()
best_k_by_elbow = k_list[elbow_from_curve(results_df["inertia"].tolist())]
best_k_by_silhouette = k_list[int(results_df["silhouette"].argmax())]

print("Best k by elbow:", best_k_by_elbow)
print("Best k by silhouette:", best_k_by_silhouette)
#%% optimal k chosen by inertia (elbow curve)
chosen_k = best_k_by_elbow  # or best_k_by_silhouette

final_kmeans = MiniBatchKMeans(
    n_clusters=chosen_k,
    random_state=random_state,
    batch_size=1024,
    n_init="auto",
)

labels = final_kmeans.fit_predict(X_umap)

np.save(os.path.join(output_folder, "kmeans_labels.npy"), labels)

with open(os.path.join(output_folder, "kmeans.pkl"), "wb") as f:
    pickle.dump(final_kmeans, f)

print("Final chosen k:", chosen_k)
print("Final labels saved.")
#%% plotting UMAP 2d space with clustering labels
labels = np.load(os.path.join(output_folder, "kmeans_labels.npy"))
X_umap = np.load(os.path.join(output_folder, "X_umap.npy"))

# If X_umap has >2 dims, this plots the first two dimensions.
# For very large data, plot only a sample.
plot_sample_size = 200000  # adjust down if needed
rng = np.random.default_rng(random_state)

n_total = X_umap.shape[0]
sample_idx = rng.choice(n_total, size=min(plot_sample_size, n_total), replace=False)

X_plot = X_umap[sample_idx]
labels_plot = labels[sample_idx]

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_plot[:, 0],
    X_plot[:, 1],
    c=labels_plot,
    s=2,
    alpha=0.3,
    cmap='Set1',
    rasterized=True,
)
# range of the main cloud
# plt.xlim(9.6,10.5)
# plt.ylim(-0.4,0.4)
# range for the outlier cloud
# plt.xlim(0.5,1)
# plt.ylim(-0.2,0.2)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP projection colored by KMeans cluster")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.show()
#%% alternative plot: eclipses with centroids
from matplotlib.patches import Ellipse
random_state=42
labels = np.load(os.path.join(output_folder, "kmeans_labels.npy"))
X_umap = np.load(os.path.join(output_folder, "X_umap.npy"))

# Use a 2D visualization array here.
# If X_umap is already 2D, set X_vis = X_umap
# If X_umap is >2D, use your 2D plot embedding instead, e.g. X_vis from a 2D UMAP
X_vis = X_umap[:, :2] if X_umap.shape[1] >= 2 else X_umap

cluster_ids = np.unique(labels)

def plot_cov_ellipse(mean, cov, ax, n_std=2.0, edgecolor="black", facecolor="none", alpha=0.25):
    """
    Draw an ellipse representing the covariance of a 2D Gaussian.
    n_std=2.0 roughly corresponds to a 95% region for a Gaussian.
    """
    if cov.shape != (2, 2):
        return

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)

    ell = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        edgecolor=edgecolor,
        facecolor=facecolor,
        lw=2,
        alpha=alpha,
    )
    ax.add_patch(ell)

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.Set1(np.linspace(0, 1, len(cluster_ids)))

for c, color in zip(cluster_ids, colors):
    idx = labels == c
    pts = X_vis[idx]

    if pts.shape[0] < 2:
        continue

    mean = pts.mean(axis=0)
    cov = np.cov(pts.T)
    std = pts.std(axis=0)

    # centroid
    ax.scatter(mean[0], mean[1], s=120, color=color, edgecolor="black", zorder=3)
    ax.text(mean[0], mean[1], f"  {c}", fontsize=11, weight="bold", va="center")

    # std bars
    ax.errorbar(
        mean[0],
        mean[1],
        xerr=std[0],
        yerr=std[1],
        fmt="none",
        ecolor=color,
        elinewidth=1.5,
        alpha=0.8,
        capsize=3,
    )

    # covariance ellipse
    plot_cov_ellipse(mean, cov, ax, n_std=2.0, edgecolor=color, alpha=0.35)

ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.set_title("Cluster centroids with spread (mean, std, covariance ellipse)")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()

#%% examine representative text segments of each cluster
labels = np.load(os.path.join(output_folder, "kmeans_labels.npy"))
X_umap = np.load(os.path.join(output_folder, "X_umap.npy"))

df = pd.DataFrame({
    "text": y_text,
    "cluster": labels
})

cluster_counts = df["cluster"].value_counts().sort_index()
print('cluster counts', cluster_counts)
# X_umap should be the embedding space you clustered on
# If X_umap has more than 2 dimensions, this still works fine.
# We compute the centroid in UMAP space and find the closest texts.

rep_n = 10
representative_rows = []

for c in sorted(df["cluster"].unique()):
    idx = np.where(labels == c)[0]
    X_c = X_umap[idx]
    
    centroid = X_c.mean(axis=0)
    dists = np.linalg.norm(X_c - centroid, axis=1)
    nearest = idx[np.argsort(dists)[:rep_n]]
    
    print("\n" + "=" * 100)
    print(f"Cluster {c} | representative texts closest to centroid")
    print("=" * 100)
    
    for j, row_idx in enumerate(nearest, 1):
        txt = df.iloc[row_idx]["text"]
        print(f"\n[{j}] {txt[:500]}")
        
        representative_rows.append({
            "cluster": int(c),
            "row_idx": int(row_idx),
            "distance_to_centroid": float(np.linalg.norm(X_umap[row_idx] - centroid)),
            "text": txt
        })
representative_rows = pd.DataFrame(representative_rows)
representative_rows.to_csv(output_folder+'/representative_rows.csv')
#%% top terms in each cluster

#%% load story events
from sentence_transformers import SentenceTransformer
base_path = 'C:/Users/mkersey/Documents/GitHub/Distinct-paths-to-false-memory-'
keep_ids = np.load(base_path + '/E. semantic_features/keep_ids.npy')
story_ids = ['pieman','eyespy','oregontrail','baseball']
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
centrality_arr = []
for j, story_id in enumerate(story_ids):
    story_segs = pd.read_excel('./A. script_prompting/story_segs.xlsx',
                               sheet_name='%s_segs' % story_id)['events'].to_list()
    story_segs = np.array(story_segs)[keep_ids[j]]
    story_embedding = model.encode(story_segs)

# %%

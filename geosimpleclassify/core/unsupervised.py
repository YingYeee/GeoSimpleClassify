from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans


def unsupervised_cluster(
    features,
    method = "kmeans",
    n_clusters = 8,
    random_state = 42,
    # fast mode parameters
    fast_mode = False,
    sample_size = 200000,
    batch_size = 4096,
    max_iter = 100,
    predict_chunk_size = 500000,
):
    """
    Perform unsupervised clustering on feature vectors.

    Parameters
    ----------
    features : ndarray
        Feature matrix with shape (N, B).
    method : {"kmeans"}, optional
        Clustering method.
    n_clusters : int, optional
        Number of clusters.
    random_state : int, optional
        Random seed.
    fast_mode : bool, optional
        Use MiniBatchKMeans with sampling.
    sample_size : int, optional
        Number of samples for fitting in fast mode.
    batch_size : int, optional
        Mini-batch size.
    max_iter : int, optional
        Maximum iterations.
    predict_chunk_size : int, optional
        Chunk size for prediction.

    Returns
    -------
    labels_all : ndarray
        Cluster labels with shape (N,).
    """
    if method != "kmeans":
        raise ValueError("Only method='kmeans' is supported in current pipeline.")

    if features.ndim != 2:
        raise ValueError(f"features must be (N,B), got {features.shape}")

    n = features.shape[0]
    if n == 0:
        return np.empty((0,), dtype=np.int32)

    if not fast_mode:
        km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        labels = km.fit_predict(features)
        return labels.astype(np.int32)

    # FAST MODE: sample + MiniBatchKMeans
    rng = np.random.default_rng(random_state)

    # pick sample indices
    s = int(min(max(sample_size, n_clusters * 50), n))  # keep at least some samples per cluster
    sample_idx = rng.choice(n, size=s, replace=False)
    X_fit = features[sample_idx]

    mbk = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init="auto",
    )
    mbk.fit(X_fit)

    # predict for all pixels
    labels_all = np.empty((n,), dtype=np.int32)
    if predict_chunk_size is None or predict_chunk_size <= 0:
        labels_all[:] = mbk.predict(features).astype(np.int32)
        return labels_all

    for start in range(0, n, predict_chunk_size):
        end = min(start + predict_chunk_size, n)
        labels_all[start:end] = mbk.predict(features[start:end]).astype(np.int32)

    return labels_all

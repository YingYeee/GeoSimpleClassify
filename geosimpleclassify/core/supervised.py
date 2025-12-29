from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def _stratified_sample_indices(
    labels,
    sample_per_class,
    random_state,
):
    """
    Sample indices per class with an upper bound per class.

    Parameters
    ----------
    labels : ndarray
        Class labels.
    sample_per_class : int
        Maximum samples per class.
    random_state : int
        Random seed.

    Returns
    -------
    idx : ndarray
        Sample indices.
    """
    rng = np.random.default_rng(random_state)
    picks = []

    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        if idx_c.size == 0:
            continue
        if idx_c.size > sample_per_class:
            idx_c = rng.choice(idx_c, size=sample_per_class, replace=False)
        picks.append(idx_c)

    if not picks:
        return np.arange(labels.size)

    return np.concatenate(picks)


def supervised_classify(
    features,
    labels_init,
    model = "rf",
    sample_per_class = 20000,
    random_state = 42,
):
    """
    Train a classifier on pseudo labels and predict refined labels.

    Parameters
    ----------
    features : ndarray
        Feature matrix with shape (N, B).
    labels_init : ndarray
        Initial labels with shape (N,).
    model : {"rf", "svm"}, optional
        Classification model.
    sample_per_class : int, optional
        Training samples per class.
    random_state : int, optional
        Random seed.

    Returns
    -------
    labels_final : ndarray
        Predicted labels with shape (N,).
    """
    if features.shape[0] != labels_init.shape[0]:
        raise ValueError("features and labels_init length mismatch.")

    train_idx = _stratified_sample_indices(labels_init, sample_per_class, random_state)
    X_train = features[train_idx]
    y_train = labels_init[train_idx]

    if model == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            max_features="sqrt",
        )
    elif model == "svm":
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=False,
            random_state=random_state,
        )
    else:
        raise ValueError("model must be 'rf' or 'svm'.")

    clf.fit(X_train, y_train)
    labels_final = clf.predict(features).astype(np.int32)
    return labels_final

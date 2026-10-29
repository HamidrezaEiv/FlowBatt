"""RF-synthesis quality and diversity metrics."""

from __future__ import annotations

import numpy as np
import scipy.linalg
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def _covariance(features: np.ndarray) -> np.ndarray:
    """Return a float64 feature covariance, or zeros for fewer than two observations."""
    features = np.asarray(features, dtype=np.float64)
    if features.shape[0] < 2:
        return np.zeros((features.shape[1], features.shape[1]), dtype=np.float64)
    return np.atleast_2d(np.cov(features, rowvar=False))


def calculate_fid(f1: np.ndarray, f2: np.ndarray) -> np.float64:
    """Compute Gaussian Fréchet distance from two (samples, features) arrays."""
    f1 = np.asarray(f1, dtype=np.float64)
    f2 = np.asarray(f2, dtype=np.float64)
    mean1, sigma1 = f1.mean(axis=0), _covariance(f1)
    mean2, sigma2 = f2.mean(axis=0), _covariance(f2)
    sum_sq_diff = np.sum((mean1 - mean2)**2)
    cov_mean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    fid = sum_sq_diff + np.trace(sigma1 + sigma2 - 2.0*cov_mean)
    return fid

def pca_features(reference: np.ndarray, samples: np.ndarray, n_components: float | int=0.99) -> tuple[np.ndarray, np.ndarray]:
    """Fit PCA on reference training trajectories and project both input sets."""
    reference = np.asarray(reference, dtype=np.float32).reshape(len(reference), -1)
    samples = np.asarray(samples, dtype=np.float32).reshape(len(samples), -1)
    if isinstance(n_components, float) and n_components >= 1:
        n_components = int(n_components)
    if isinstance(n_components, int):
        n_components = min(n_components, reference.shape[0] - 1, reference.shape[1])
    pca = PCA(n_components=n_components)
    ref_features = pca.fit_transform(reference)
    sample_features = pca.transform(samples)
    return ref_features, sample_features


def precision_recall_features(ref_features: np.ndarray, eval_features: np.ndarray, n_neighbors: int=3) -> tuple[float, float]:
    """Measure overlap using each sample's nearest-reference-neighbor sphere."""
    ref_features = np.asarray(ref_features, dtype=np.float32)
    eval_features = np.asarray(eval_features, dtype=np.float32)
    n_neighbors = min(n_neighbors, len(ref_features) - 1, len(eval_features) - 1)
    if n_neighbors < 1:
        return np.nan, np.nan

    ref_nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(ref_features)
    ref_dists, _ = ref_nn.kneighbors(ref_features)
    ref_radius = ref_dists[:, -1]

    eval_nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(eval_features)
    eval_dists, _ = eval_nn.kneighbors(eval_features)
    eval_radius = eval_dists[:, -1]

    cross_eval_ref = NearestNeighbors(n_neighbors=1).fit(ref_features)
    eval_to_ref_dist, eval_to_ref_idx = cross_eval_ref.kneighbors(eval_features)
    precision = (
        eval_to_ref_dist[:, 0] <= ref_radius[eval_to_ref_idx[:, 0]]
    ).mean()

    cross_ref_eval = NearestNeighbors(n_neighbors=1).fit(eval_features)
    ref_to_eval_dist, ref_to_eval_idx = cross_ref_eval.kneighbors(ref_features)
    recall = (
        ref_to_eval_dist[:, 0] <= eval_radius[ref_to_eval_idx[:, 0]]
    ).mean()

    return float(precision), float(recall)


def synthetic_soh_metrics(reference_soh: np.ndarray, synthetic_soh: np.ndarray, n_components: float | int=0.99, n_neighbors: int=3) -> dict[str, float]:
    """Return PCA-space FID, precision, and recall using real training references."""
    ref_features, synthetic_features = pca_features(
        reference_soh,
        synthetic_soh,
        n_components=n_components,
    )
    fid = calculate_fid(ref_features, synthetic_features)
    precision, recall = precision_recall_features(
        ref_features,
        synthetic_features,
        n_neighbors=n_neighbors,
    )
    return {
        "fid": float(fid),
        "precision": precision,
        "recall": recall,
    }

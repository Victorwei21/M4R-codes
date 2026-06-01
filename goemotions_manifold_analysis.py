"""
Modell-style manifold analysis for GoEmotions embeddings.

This script is intentionally split into small, inspectable steps rather than
hiding the experiment in a notebook-shaped swamp.

What it does:
1. Builds three representation levels when available:
   - comment_centroid: one averaged embedding per emotion label. This is the
     main, most stable result.
   - individual_label_vad: individual comment embeddings paired with the VAD
     coordinates of their label. This is noisy and can be expensive.
   - label_word: embeddings of the emotion label words themselves, if created
     by goemotions_embed_pipeline.py with --embed-label-words.
2. Compares representation distances against affective ground-truth distances
   derived from valence/arousal/dominance coordinates.
3. Computes both direct distances and Modell-style KNN geodesic distances.
4. Runs PCA/K/metric sensitivity sweeps.
5. Adds permutation p-values for centroid/label-word results.
6. Optionally tests whether valence, arousal, and dominance are linearly
   recoverable from individual comment embeddings.
7. Saves clean inspection plots, including selected-label plots for
   admiration/confusion/fear/joy by default.

Fast Windows PowerShell example:
    python goemotions_manifold_analysis.py `
        --data-dir "D:/pyprojects/m4r/data/goemotions" `
        --vad-csv "D:/pyprojects/m4r/data/goemotions/emotion_label_vad_starter.csv" `
        --individual-mode direct `
        --max-individual 500

More expensive geodesic individual-comment example:
    python goemotions_manifold_analysis.py `
        --data-dir "D:/pyprojects/m4r/data/goemotions" `
        --vad-csv "D:/pyprojects/m4r/data/goemotions/emotion_label_vad_starter.csv" `
        --individual-mode geodesic `
        --individual-pca-dims "10" `
        --individual-k-values "auto" `
        --max-individual 800
"""

from __future__ import annotations

import argparse
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.stats import kendalltau, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except Exception:  # pragma: no cover
    plt = None
    Ellipse = None


VAD_SPACES = {
    "VAD": ["valence", "arousal", "dominance"],
    "VA": ["valence", "arousal"],
    "valence": ["valence"],
    "arousal": ["arousal"],
    "dominance": ["dominance"],
}


@dataclass
class RepresentationLevel:
    name: str
    X: np.ndarray
    labels: pd.DataFrame
    vad: np.ndarray


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_string_list(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def parse_k_values(text: str) -> list[str | int]:
    values: list[str | int] = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item == "auto":
            values.append("auto")
        else:
            values.append(int(item))
    return values


def load_embeddings(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return pd.read_csv(path, header=None).values


def load_vad(path: Path) -> pd.DataFrame:
    vad = pd.read_csv(path)
    required = {"label_name", "valence", "arousal", "dominance"}
    missing = required - set(vad.columns)
    if missing:
        raise ValueError(f"VAD CSV missing required columns: {sorted(missing)}")
    vad = vad.copy()
    vad["label_name"] = vad["label_name"].astype(str)
    for col in ["valence", "arousal", "dominance"]:
        vad[col] = pd.to_numeric(vad[col], errors="coerce")
    if vad[["valence", "arousal", "dominance"]].isna().any().any():
        bad = vad[vad[["valence", "arousal", "dominance"]].isna().any(axis=1)]
        raise ValueError("Some VAD rows contain missing/non-numeric values:\n" + bad.to_string(index=False))
    return vad


def attach_vad(labels: pd.DataFrame, vad_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    out = labels.merge(vad_df[["label_name", "valence", "arousal", "dominance"]], on="label_name", how="left")
    if out[["valence", "arousal", "dominance"]].isna().any().any():
        missing = sorted(out.loc[out["valence"].isna(), "label_name"].unique())
        raise ValueError(f"Missing VAD rows for labels: {missing}")
    vad = out[["valence", "arousal", "dominance"]].to_numpy(dtype=float)
    return out, vad


def make_comment_level(X: np.ndarray, metadata: pd.DataFrame, vad_df: pd.DataFrame, max_n: int, seed: int) -> RepresentationLevel:
    labels_vad, vad = attach_vad(metadata[["label_idx", "label_name"]].copy(), vad_df)
    if len(X) != len(metadata):
        raise ValueError(f"Embeddings and metadata length differ: {len(X)} vs {len(metadata)}")
    if max_n and len(X) > max_n:
        # Preserve label balance as much as possible while keeping the expensive
        # all-pairs part small. Exact stratification is unnecessary here.
        tmp = metadata.copy()
        tmp["_row"] = np.arange(len(tmp))
        frac = min(1.0, max_n / len(tmp))
        sampled_groups = []
        for _, g in tmp.groupby("label_name", sort=False):
            n_g = max(1, int(round(len(g) * frac)))
            sampled_groups.append(g.sample(n=min(len(g), n_g), random_state=seed))
        sampled = pd.concat(sampled_groups, ignore_index=True)
        if len(sampled) > max_n:
            sampled = sampled.sample(n=max_n, random_state=seed).reset_index(drop=True)
        idx = sampled["_row"].to_numpy()
        X = X[idx]
        labels_vad = labels_vad.iloc[idx].reset_index(drop=True)
        vad = vad[idx]
    return RepresentationLevel("individual_label_vad", X, labels_vad.reset_index(drop=True), vad)


def make_centroid_level(X: np.ndarray, metadata: pd.DataFrame, vad_df: pd.DataFrame) -> RepresentationLevel:
    if len(X) != len(metadata):
        raise ValueError(f"Embeddings and metadata length differ: {len(X)} vs {len(metadata)}")
    df = metadata[["label_idx", "label_name"]].copy()
    df["_row"] = np.arange(len(df))
    rows = []
    centroids = []
    for label_name, group in df.groupby("label_name", sort=True):
        idx = group["_row"].to_numpy()
        centroids.append(X[idx].mean(axis=0))
        rows.append({
            "label_name": label_name,
            "label_idx": int(group["label_idx"].iloc[0]),
            "n_comments": int(len(group)),
        })
    label_df = pd.DataFrame(rows)
    label_df, vad = attach_vad(label_df, vad_df)
    return RepresentationLevel("comment_centroid", np.vstack(centroids), label_df, vad)


def make_label_word_level(label_word_embeddings: Path, label_word_metadata: Path, vad_df: pd.DataFrame) -> RepresentationLevel | None:
    if not label_word_embeddings.exists() or not label_word_metadata.exists():
        return None
    X = load_embeddings(label_word_embeddings)
    meta = pd.read_csv(label_word_metadata)
    labels, vad = attach_vad(meta[["label_idx", "label_name"]].copy(), vad_df)
    return RepresentationLevel("label_word", X, labels, vad)


def maybe_pca(X: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, str, float]:
    if n_components <= 0:
        return X, "raw", math.nan
    max_components = min(X.shape[0] - 1, X.shape[1])
    if max_components < 1:
        return X, "raw", math.nan
    n = min(n_components, max_components)
    pca = PCA(n_components=n, random_state=seed)
    Z = pca.fit_transform(X)
    return Z, f"pca_{n}", float(pca.explained_variance_ratio_.sum())


def upper_values(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def corr_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {
            "pearson_r": np.nan, "pearson_p": np.nan,
            "spearman_rho": np.nan, "spearman_p": np.nan,
            "kendall_tau": np.nan, "kendall_p": np.nan,
            "n_pairs": len(a),
        }
    pr = pearsonr(a, b)
    sr = spearmanr(a, b)
    kt = kendalltau(a, b)
    return {
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
        "kendall_tau": float(kt.statistic),
        "kendall_p": float(kt.pvalue),
        "n_pairs": int(len(a)),
    }


def angular_pairwise(X: np.ndarray) -> np.ndarray:
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sims = np.clip(Xn @ Xn.T, -1.0, 1.0)
    return np.arccos(sims)


def pairwise_rep_distance(X: np.ndarray, metric: str) -> np.ndarray:
    if metric == "angular":
        return angular_pairwise(X)
    return pairwise_distances(X, metric=metric)


def build_symmetric_knn_graph(X: np.ndarray, k: int, metric: str) -> csr_matrix:
    n = X.shape[0]
    if k >= n:
        raise ValueError(f"k must be smaller than number of samples. Got k={k}, n={n}")

    if metric == "angular":
        Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nbrs.fit(Xn)
        _, indices = nbrs.kneighbors(Xn)
        indices = indices[:, 1:]
        rows = np.repeat(np.arange(n), k)
        cols = indices.ravel()
        sims = np.sum(Xn[rows] * Xn[cols], axis=1)
        vals = np.arccos(np.clip(sims, -1.0, 1.0))
    else:
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric=metric)
        nbrs.fit(X)
        distances, indices = nbrs.kneighbors(X)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
        rows = np.repeat(np.arange(n), k)
        cols = indices.ravel()
        vals = distances.ravel()

    G = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return G.maximum(G.T)


def connected_k(X: np.ndarray, metric: str, k_max: int) -> tuple[int, csr_matrix]:
    upper = min(k_max, X.shape[0] - 1)
    for k in range(1, upper + 1):
        G = build_symmetric_knn_graph(X, k=k, metric=metric)
        n_components, _ = connected_components(G, directed=False)
        if n_components == 1:
            return k, G
    raise ValueError(f"Graph did not become connected up to k={upper}")


def geodesic_distance(X: np.ndarray, k_value: str | int, metric: str, k_max: int) -> tuple[np.ndarray, int, int]:
    if k_value == "auto":
        k, G = connected_k(X, metric=metric, k_max=k_max)
        n_components = 1
    else:
        k = int(k_value)
        G = build_symmetric_knn_graph(X, k=k, metric=metric)
        n_components, _ = connected_components(G, directed=False)
    dist = dijkstra(csgraph=G, directed=False, return_predecessors=False)
    return dist, k, int(n_components)


def vad_distance(vad: np.ndarray, space: str) -> np.ndarray:
    cols = VAD_SPACES[space]
    col_idx = [{"valence": 0, "arousal": 1, "dominance": 2}[c] for c in cols]
    return pairwise_distances(vad[:, col_idx], metric="euclidean")


def permutation_pvalue(
    obs_abs_spearman: float,
    rep_upper: np.ndarray,
    vad: np.ndarray,
    space: str,
    n_perm: int,
    seed: int,
) -> float:
    if n_perm <= 0 or not np.isfinite(obs_abs_spearman):
        return math.nan
    rng = np.random.default_rng(seed)
    count = 0
    n = vad.shape[0]
    for _ in range(n_perm):
        perm = rng.permutation(n)
        Dp = vad_distance(vad[perm], space)
        rho = corr_stats(upper_values(Dp), rep_upper)["spearman_rho"]
        if np.isfinite(rho) and abs(rho) >= obs_abs_spearman:
            count += 1
    return float((count + 1) / (n_perm + 1))


def add_distance_results(
    rows: list[dict],
    level: RepresentationLevel,
    representation_name: str,
    explained_variance: float,
    distance_name: str,
    rep_D: np.ndarray,
    pca_dim: int,
    k: int | None,
    n_components: int | None,
    vad_spaces: list[str],
    n_perm: int,
    seed: int,
) -> None:
    rep_upper = upper_values(rep_D)
    for space in vad_spaces:
        truth_D = vad_distance(level.vad, space)
        stats = corr_stats(upper_values(truth_D), rep_upper)
        p_perm = permutation_pvalue(abs(stats["spearman_rho"]), rep_upper, level.vad, space, n_perm=n_perm, seed=seed)
        rows.append({
            "level": level.name,
            "n_points": int(level.X.shape[0]),
            "representation": representation_name,
            "pca_dim_requested": pca_dim,
            "explained_variance": explained_variance,
            "distance_method": distance_name,
            "knn_k": k if k is not None else "",
            "graph_connected_components": n_components if n_components is not None else "",
            "ground_truth_space": space,
            "permutation_p_abs_spearman": p_perm,
            **stats,
        })


def run_distance_analysis(
    levels: list[RepresentationLevel],
    pca_dims: list[int],
    individual_pca_dims: list[int],
    k_values: list[str | int],
    individual_k_values: list[str | int],
    graph_metrics: list[str],
    direct_metrics: list[str],
    vad_spaces: list[str],
    k_max: int,
    n_perm_centroid: int,
    n_perm_individual: int,
    individual_mode: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for level in levels:
        print(f"\n=== Level: {level.name} ({level.X.shape[0]} points) ===")
        is_individual = level.name == "individual_label_vad"
        if is_individual and individual_mode == "skip":
            print("Skipping individual-comment distance analysis by request.")
            continue

        level_pca_dims = individual_pca_dims if is_individual else pca_dims
        level_k_values = individual_k_values if is_individual else k_values
        run_direct = (not is_individual) or individual_mode in {"direct", "full"}
        run_geodesic = (not is_individual) or individual_mode in {"geodesic", "full"}
        n_perm = n_perm_individual if is_individual else n_perm_centroid

        for pca_dim in level_pca_dims:
            Z, rep_name, ev = maybe_pca(level.X, pca_dim, seed=seed)

            if run_direct:
                for metric in direct_metrics:
                    if metric == "cosine" and np.any(np.linalg.norm(Z, axis=1) == 0):
                        continue
                    try:
                        rep_D = pairwise_rep_distance(Z, metric=metric)
                    except Exception as exc:
                        warnings.warn(f"Skipping direct metric={metric} for {level.name}/{rep_name}: {exc}")
                        continue
                    add_distance_results(
                        rows, level, rep_name, ev,
                        distance_name=f"direct_{metric}", rep_D=rep_D,
                        pca_dim=pca_dim, k=None, n_components=None,
                        vad_spaces=vad_spaces, n_perm=n_perm, seed=seed,
                    )

            if run_geodesic:
                for graph_metric in graph_metrics:
                    for kval in level_k_values:
                        if isinstance(kval, int) and kval >= Z.shape[0]:
                            continue
                        try:
                            graph_D, actual_k, n_components = geodesic_distance(Z, k_value=kval, metric=graph_metric, k_max=k_max)
                        except Exception as exc:
                            warnings.warn(f"Skipping geodesic metric={graph_metric}, k={kval}, {level.name}/{rep_name}: {exc}")
                            continue
                        add_distance_results(
                            rows, level, rep_name, ev,
                            distance_name=f"knn_geodesic_{graph_metric}", rep_D=graph_D,
                            pca_dim=pca_dim, k=actual_k, n_components=n_components,
                            vad_spaces=vad_spaces, n_perm=n_perm, seed=seed,
                        )
                        print(f"{level.name} {rep_name} geodesic metric={graph_metric} k={actual_k} components={n_components} done")
    return pd.DataFrame(rows)


def run_axis_prediction(level: RepresentationLevel, pca_dims: list[int], seed: int) -> pd.DataFrame:
    rows = []
    if level.X.shape[0] < 50:
        return pd.DataFrame(rows)
    for pca_dim in pca_dims:
        n_comp = min(pca_dim, level.X.shape[0] - 2, level.X.shape[1])
        if n_comp < 2:
            continue
        model = make_pipeline(
            StandardScaler(with_mean=True),
            PCA(n_components=n_comp, random_state=seed),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
        for axis_name, axis_idx in [("valence", 0), ("arousal", 1), ("dominance", 2)]:
            scores = cross_val_score(model, level.X, level.vad[:, axis_idx], cv=cv, scoring="r2")
            rows.append({
                "level": level.name,
                "axis": axis_name,
                "pca_dim": n_comp,
                "cv_r2_mean": float(np.mean(scores)),
                "cv_r2_std": float(np.std(scores)),
                "n_points": int(level.X.shape[0]),
            })
    return pd.DataFrame(rows)


def save_centroid_tables(levels: list[RepresentationLevel], out_dir: Path) -> None:
    for level in levels:
        if level.name != "comment_centroid":
            continue
        labels = level.labels.copy()
        labels.to_csv(out_dir / "centroid_labels_with_vad.csv", index=False)
        D_vad = vad_distance(level.vad, "VAD")
        pd.DataFrame(D_vad, index=labels["label_name"], columns=labels["label_name"]).to_csv(out_dir / "centroid_vad_distance_matrix.csv")


def _clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linewidth=0.4, alpha=0.25)


def _add_covariance_ellipse(ax, points: np.ndarray, color, n_std: float = 1.5) -> None:
    if Ellipse is None or len(points) < 3:
        return
    cov = np.cov(points.T)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        xy=points.mean(axis=0),
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=color,
        linewidth=1.8,
        alpha=0.85,
    )
    ax.add_patch(ell)


def _plot_selected_scatter(
    Z: np.ndarray,
    labels: pd.Series,
    focus_labels: list[str],
    title: str,
    subtitle: str,
    out_path: Path,
    explained_variance: np.ndarray | None = None,
) -> None:
    if plt is None:
        return
    colors = plt.get_cmap("tab10").colors
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(focus_labels)}

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for label in focus_labels:
        mask = labels.eq(label).to_numpy()
        if not np.any(mask):
            continue
        pts = Z[mask]
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=24,
            alpha=0.58,
            color=color_map[label],
            label=f"{label} (n={len(pts)})",
            linewidths=0,
        )
        _add_covariance_ellipse(ax, pts, color_map[label])
        center = pts.mean(axis=0)
        ax.scatter(center[0], center[1], s=150, marker="X", color=color_map[label], edgecolor="black", linewidth=0.7)
        ax.text(center[0], center[1], f"  {label}", fontsize=11, weight="bold", va="center")

    if explained_variance is not None and len(explained_variance) >= 2:
        xlab = f"PC1 ({explained_variance[0] * 100:.1f}% var.)"
        ylab = f"PC2 ({explained_variance[1] * 100:.1f}% var.)"
    else:
        xlab = "PC1"
        ylab = "PC2"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title, pad=14)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, alpha=0.72)
    ax.legend(frameon=False, loc="best")
    _clean_axes(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_selected_label_plots(
    X: np.ndarray,
    metadata: pd.DataFrame,
    out_dir: Path,
    focus_labels: list[str],
    max_per_label: int,
    global_pca_fit_max: int,
    seed: int,
) -> None:
    """Save clean PCA plots containing only selected emotion labels.

    Two plots are saved:
    - selected_labels_local_pca2.png: PCA is fitted only on the selected labels.
      This is best for visual separation among the chosen emotions.
    - selected_labels_global_pca2.png: PCA is fitted on a sample of all comments,
      then only the selected labels are displayed. This is more faithful to the
      global embedding geometry, but often visually messier.
    """
    if plt is None:
        return
    if len(X) != len(metadata):
        raise ValueError(f"Embeddings and metadata length differ: {len(X)} vs {len(metadata)}")

    rng = np.random.default_rng(seed)
    meta = metadata.copy().reset_index(drop=True)
    meta["_row"] = np.arange(len(meta))
    selected_parts = []
    for label in focus_labels:
        group = meta[meta["label_name"].eq(label)]
        if group.empty:
            warnings.warn(f"Focus label not found in metadata: {label}")
            continue
        if max_per_label and len(group) > max_per_label:
            group = group.sample(n=max_per_label, random_state=seed)
        selected_parts.append(group)
    if not selected_parts:
        warnings.warn("No focus-label rows found, so selected-label plots were not created.")
        return

    selected = pd.concat(selected_parts, ignore_index=True)
    X_sel = X[selected["_row"].to_numpy()]
    y_sel = selected["label_name"].reset_index(drop=True)
    present_labels = [label for label in focus_labels if label in set(y_sel)]

    if len(X_sel) < 3:
        warnings.warn("Too few selected-label points for PCA plot.")
        return

    local_pca = PCA(n_components=2, random_state=seed)
    Z_local = local_pca.fit_transform(X_sel)
    _plot_selected_scatter(
        Z_local,
        y_sel,
        present_labels,
        title="Selected GoEmotions comment embeddings, local PCA",
        subtitle="Only admiration, confusion, fear, and joy are displayed; X marks label centroid in the plotted PCA space.",
        out_path=out_dir / "selected_labels_local_pca2.png",
        explained_variance=local_pca.explained_variance_ratio_,
    )

    if global_pca_fit_max and len(X) > global_pca_fit_max:
        fit_idx = rng.choice(len(X), size=global_pca_fit_max, replace=False)
        X_fit = X[fit_idx]
    else:
        X_fit = X
    global_pca = PCA(n_components=2, random_state=seed)
    global_pca.fit(X_fit)
    Z_global = global_pca.transform(X_sel)
    _plot_selected_scatter(
        Z_global,
        y_sel,
        present_labels,
        title="Selected GoEmotions comment embeddings, global PCA",
        subtitle="PCA is fitted on the broader embedding sample, but only the four selected labels are displayed.",
        out_path=out_dir / "selected_labels_global_pca2.png",
        explained_variance=global_pca.explained_variance_ratio_,
    )


def make_basic_plots(results: pd.DataFrame, levels: list[RepresentationLevel], out_dir: Path, seed: int) -> None:
    if plt is None or results.empty:
        return

    top = results[results["level"].eq("comment_centroid")].copy()
    if not top.empty:
        top["abs_spearman"] = top["spearman_rho"].abs()
        top = top.sort_values("abs_spearman", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = top["distance_method"] + " / " + top["representation"] + " / " + top["ground_truth_space"]
        ax.barh(np.arange(len(top)), top["spearman_rho"])
        ax.set_yticks(np.arange(len(top)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Spearman correlation with ground-truth affective distance")
        ax.set_title("Top centroid-level distance alignment results")
        ax.invert_yaxis()
        _clean_axes(ax)
        fig.tight_layout()
        fig.savefig(out_dir / "top_centroid_alignment_results.png", dpi=220)
        plt.close(fig)

    centroid = next((lvl for lvl in levels if lvl.name == "comment_centroid"), None)
    if centroid is not None and centroid.X.shape[0] >= 3:
        Z, _, ev = maybe_pca(centroid.X, 2, seed=seed)
        for axis_name, axis_idx in [("valence", 0), ("arousal", 1), ("dominance", 2)]:
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(Z[:, 0], Z[:, 1], c=centroid.vad[:, axis_idx], s=80)
            for i, name in enumerate(centroid.labels["label_name"]):
                ax.text(Z[i, 0], Z[i, 1], str(name), fontsize=8, alpha=0.8)
            fig.colorbar(sc, ax=ax, label=axis_name)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"Comment centroids in PCA space, colored by {axis_name} (EV={ev:.2f})")
            _clean_axes(ax)
            fig.tight_layout()
            fig.savefig(out_dir / f"centroid_pca2_colored_by_{axis_name}.png", dpi=220)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--vad-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--label-word-embeddings", type=Path, default=None)
    parser.add_argument("--label-word-metadata", type=Path, default=None)

    # Light defaults. The individual-comment all-pairs graph is the expensive bit.
    parser.add_argument("--max-individual", type=int, default=500, help="Cap individual-comment analysis to avoid huge all-pairs graphs")
    parser.add_argument("--individual-mode", choices=["skip", "direct", "geodesic", "full"], default="direct")
    parser.add_argument("--pca-dims", type=str, default="0,2,5,10,25")
    parser.add_argument("--individual-pca-dims", type=str, default="0,10,50")
    parser.add_argument("--axis-pca-dims", type=str, default="10,50")
    parser.add_argument("--k-values", type=str, default="auto,5,10")
    parser.add_argument("--individual-k-values", type=str, default="auto")
    parser.add_argument("--direct-metrics", type=str, default="cosine,angular")
    parser.add_argument("--graph-metrics", type=str, default="angular")
    parser.add_argument("--vad-spaces", type=str, default="VAD,VA,valence")
    parser.add_argument("--k-max", type=int, default=100)
    parser.add_argument("--n-perm-centroid", type=int, default=100)
    parser.add_argument("--n-perm-individual", type=int, default=0)
    parser.add_argument("--skip-axis-prediction", action="store_true")

    # Selected-label plot controls.
    parser.add_argument("--focus-labels", type=str, default="fear, love, caring, gratitude")
    parser.add_argument("--focus-max-per-label", type=int, default=300)
    parser.add_argument("--focus-global-pca-fit-max", type=int, default=10000)

    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = args.data_dir
    embeddings_path = args.embeddings or data_dir / "manifold_outputs" / "embeddings.npy"
    metadata_path = args.metadata or data_dir / "manifold_outputs" / "metadata.csv"
    out_dir = args.out_dir or data_dir / "manifold_outputs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    X = load_embeddings(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    vad_df = load_vad(args.vad_csv)

    levels: list[RepresentationLevel] = [
        make_centroid_level(X, metadata, vad_df),
        make_comment_level(X, metadata, vad_df, max_n=args.max_individual, seed=args.seed),
    ]

    if args.label_word_embeddings and args.label_word_metadata:
        label_level = make_label_word_level(args.label_word_embeddings, args.label_word_metadata, vad_df)
        if label_level is not None:
            levels.append(label_level)

    pca_dims = parse_int_list(args.pca_dims)
    individual_pca_dims = parse_int_list(args.individual_pca_dims)
    axis_pca_dims = parse_int_list(args.axis_pca_dims)
    k_values = parse_k_values(args.k_values)
    individual_k_values = parse_k_values(args.individual_k_values)
    direct_metrics = parse_string_list(args.direct_metrics)
    graph_metrics = parse_string_list(args.graph_metrics)
    vad_spaces = parse_string_list(args.vad_spaces)
    focus_labels = parse_string_list(args.focus_labels)
    bad_spaces = set(vad_spaces) - set(VAD_SPACES)
    if bad_spaces:
        raise ValueError(f"Unknown VAD spaces: {sorted(bad_spaces)}")

    print("Loaded embeddings:", X.shape)
    print("Loaded metadata:", metadata.shape)
    print("Levels:", [(lvl.name, lvl.X.shape) for lvl in levels])
    print("Individual mode:", args.individual_mode)

    results = run_distance_analysis(
        levels=levels,
        pca_dims=pca_dims,
        individual_pca_dims=individual_pca_dims,
        k_values=k_values,
        individual_k_values=individual_k_values,
        graph_metrics=graph_metrics,
        direct_metrics=direct_metrics,
        vad_spaces=vad_spaces,
        k_max=args.k_max,
        n_perm_centroid=args.n_perm_centroid,
        n_perm_individual=args.n_perm_individual,
        individual_mode=args.individual_mode,
        seed=args.seed,
    )
    results_path = out_dir / "distance_alignment_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Saved distance results: {results_path}")

    axis_results = pd.DataFrame()
    if not args.skip_axis_prediction:
        axis_rows = []
        for lvl in levels:
            if lvl.name == "individual_label_vad":
                axis_rows.append(run_axis_prediction(lvl, pca_dims=axis_pca_dims, seed=args.seed))
        axis_results = pd.concat(axis_rows, ignore_index=True) if axis_rows else pd.DataFrame()
    axis_path = out_dir / "axis_prediction_results.csv"
    axis_results.to_csv(axis_path, index=False)
    print(f"Saved axis prediction results: {axis_path}")

    save_centroid_tables(levels, out_dir)
    make_basic_plots(results, levels, out_dir, seed=args.seed)
    make_selected_label_plots(
        X=X,
        metadata=metadata,
        out_dir=out_dir,
        focus_labels=focus_labels,
        max_per_label=args.focus_max_per_label,
        global_pca_fit_max=args.focus_global_pca_fit_max,
        seed=args.seed,
    )

    if not results.empty:
        preview = results.copy()
        preview["abs_spearman"] = preview["spearman_rho"].abs()
        preview = preview.sort_values("abs_spearman", ascending=False).head(15)
        print("\nTop 15 results by |Spearman|:")
        cols = [
            "level", "representation", "distance_method", "knn_k", "graph_connected_components", "ground_truth_space",
            "spearman_rho", "pearson_r", "permutation_p_abs_spearman", "n_pairs",
        ]
        print(preview[cols].to_string(index=False))


if __name__ == "__main__":
    main()

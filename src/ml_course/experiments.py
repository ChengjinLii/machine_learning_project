from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_utils import load_fashion_mnist, load_iris_dataset, load_wine_dataset
from .rbf_network import RBFNetworkClassifier


RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


def ensure_result_dirs(project_root: Path) -> dict[str, Path]:
    results_root = project_root / "results"
    figures_dir = results_root / "figures"
    tables_dir = results_root / "tables"
    metrics_dir = results_root / "metrics"
    for directory in (results_root, figures_dir, tables_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return {
        "results": results_root,
        "figures": figures_dir,
        "tables": tables_dir,
        "metrics": metrics_dir,
    }


def _save_confusion_matrix(
    matrix: np.ndarray,
    labels: list[str],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _write_classification_report(report: dict, output_path: Path) -> None:
    pd.DataFrame(report).transpose().to_csv(output_path, index=True)


def _save_json(data: dict, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _prepare_wine_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X, y, _, target_names = load_wine_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, target_names


def _prepare_fashion_subset(
    train_size: int = 12000,
    test_size: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    X_train_full, y_train_full, X_test_full, y_test_full, target_names = load_fashion_mnist()
    train_indices, _ = train_test_split(
        np.arange(X_train_full.shape[0]),
        train_size=train_size,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
    test_indices, _ = train_test_split(
        np.arange(X_test_full.shape[0]),
        train_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y_test_full,
    )
    return (
        X_train_full[train_indices],
        y_train_full[train_indices],
        X_test_full[test_indices],
        y_test_full[test_indices],
        target_names,
    )


def _save_dataset_summary(paths: dict[str, Path]) -> None:
    iris_X, iris_y, iris_features, iris_names = load_iris_dataset()
    wine_X, wine_y, wine_features, wine_names = load_wine_dataset()
    fashion_train_X, fashion_train_y, fashion_test_X, fashion_test_y, fashion_names = load_fashion_mnist()

    rows = [
        {
            "dataset": "Iris",
            "samples": int(iris_X.shape[0]),
            "features": int(len(iris_features)),
            "classes": int(len(iris_names)),
            "train_samples": "",
            "test_samples": "",
            "source": "UCI Machine Learning Repository",
        },
        {
            "dataset": "Wine",
            "samples": int(wine_X.shape[0]),
            "features": int(len(wine_features)),
            "classes": int(len(wine_names)),
            "train_samples": 133,
            "test_samples": 45,
            "source": "UCI Machine Learning Repository",
        },
        {
            "dataset": "Fashion-MNIST",
            "samples": int(fashion_train_X.shape[0] + fashion_test_X.shape[0]),
            "features": int(fashion_train_X.shape[1]),
            "classes": int(len(fashion_names)),
            "train_samples": 12000,
            "test_samples": 4000,
            "source": "GitHub / Zalando Research release",
        },
    ]
    pd.DataFrame(rows).to_csv(paths["tables"] / "dataset_summary.csv", index=False)


def run_kmeans_experiment(project_root: Path, paths: dict[str, Path]) -> dict:
    X, y, feature_names, target_names = load_iris_dataset()
    X_scaled = StandardScaler().fit_transform(X)

    sweep_records: list[dict] = []
    for n_clusters in range(2, 7):
        start = time.perf_counter()
        model = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=20)
        clusters = model.fit_predict(X_scaled)
        elapsed = time.perf_counter() - start
        sweep_records.append(
            {
                "n_clusters": n_clusters,
                "inertia": float(model.inertia_),
                "silhouette": float(silhouette_score(X_scaled, clusters)),
                "adjusted_rand_index": float(adjusted_rand_score(y, clusters)),
                "normalized_mutual_info": float(normalized_mutual_info_score(y, clusters)),
                "runtime_seconds": float(elapsed),
            }
        )

        if n_clusters == 3:
            selected_model = model
            selected_clusters = clusters
            selected_elapsed = elapsed

    sweep_frame = pd.DataFrame(sweep_records)
    sweep_frame.to_csv(paths["tables"] / "kmeans_iris_k_sweep.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    axes[0].plot(sweep_frame["n_clusters"], sweep_frame["inertia"], marker="o")
    axes[0].axvline(3, color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_title("K-means Inertia vs Number of Clusters")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(
        sweep_frame["n_clusters"],
        sweep_frame["silhouette"],
        marker="o",
        label="Silhouette",
    )
    axes[1].plot(
        sweep_frame["n_clusters"],
        sweep_frame["adjusted_rand_index"],
        marker="s",
        label="ARI",
    )
    axes[1].plot(
        sweep_frame["n_clusters"],
        sweep_frame["normalized_mutual_info"],
        marker="^",
        label="NMI",
    )
    axes[1].axvline(3, color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_title("K-means Clustering Quality vs Number of Clusters")
    axes[1].set_xlabel("Number of clusters")
    axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "kmeans_iris_k_sweep.png", dpi=220)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for index, label in enumerate(target_names):
        mask = y == index
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], s=36, label=label, alpha=0.85)
    axes[0].set_title("Iris Ground Truth")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(fontsize=8)

    for index in range(3):
        mask = selected_clusters == index
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], s=36, label=f"cluster_{index}", alpha=0.85)
    axes[1].scatter(
        pca.transform(selected_model.cluster_centers_)[:, 0],
        pca.transform(selected_model.cluster_centers_)[:, 1],
        c="black",
        marker="X",
        s=120,
        label="centers",
    )
    axes[1].set_title("K-means Clusters on Iris")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(paths["figures"] / "kmeans_iris_clusters.png", dpi=220)
    plt.close(fig)

    contingency = pd.crosstab(
        pd.Series(y, name="true_label"),
        pd.Series(selected_clusters, name="cluster"),
    )
    contingency.to_csv(paths["tables"] / "kmeans_iris_contingency.csv")

    chosen = sweep_frame.loc[sweep_frame["n_clusters"] == 3].iloc[0].to_dict()
    metrics = {
        "algorithm": "K-means",
        "dataset": "Iris",
        "samples": int(X.shape[0]),
        "features": int(len(feature_names)),
        "selected_k": 3,
        "inertia": float(chosen["inertia"]),
        "silhouette": float(chosen["silhouette"]),
        "adjusted_rand_index": float(chosen["adjusted_rand_index"]),
        "normalized_mutual_info": float(chosen["normalized_mutual_info"]),
        "runtime_seconds": float(selected_elapsed),
        "best_silhouette_k": int(sweep_frame.loc[sweep_frame["silhouette"].idxmax(), "n_clusters"]),
    }
    _save_json(metrics, paths["metrics"] / "kmeans_iris_metrics.json")
    return metrics


def run_svm_experiment(project_root: Path, paths: dict[str, Path]) -> dict:
    X_train, X_test, y_train, y_test, target_names = _prepare_wine_split()

    kernel_settings = [
        {"label": "linear", "params": {"kernel": "linear", "C": 1.0}},
        {"label": "rbf", "params": {"kernel": "rbf", "C": 3.0, "gamma": "scale"}},
        {"label": "poly", "params": {"kernel": "poly", "C": 3.0, "degree": 3, "gamma": "scale"}},
        {"label": "sigmoid", "params": {"kernel": "sigmoid", "C": 3.0, "gamma": "scale"}},
    ]

    comparison_records: list[dict] = []

    for setting in kernel_settings:
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svc", SVC(**setting["params"])),
            ]
        )
        start = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.perf_counter() - start
        result = {
            "kernel": setting["label"],
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "runtime_seconds": float(elapsed),
        }
        comparison_records.append(result)

    comparison_frame = pd.DataFrame(comparison_records).sort_values(
        by=["accuracy", "macro_f1", "runtime_seconds"],
        ascending=[False, False, True],
    )
    comparison_frame.to_csv(paths["tables"] / "svm_wine_kernel_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    width = 0.35
    x = np.arange(len(comparison_frame))
    ax.bar(x - width / 2, comparison_frame["accuracy"], width=width, label="Accuracy")
    ax.bar(x + width / 2, comparison_frame["macro_f1"], width=width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_frame["kernel"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SVM Kernel Comparison on Wine")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "svm_wine_kernel_comparison.png", dpi=220)
    plt.close(fig)

    best_kernel = str(comparison_frame.iloc[0]["kernel"])
    final_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(**next(item["params"] for item in kernel_settings if item["label"] == best_kernel))),
        ]
    )
    start = time.perf_counter()
    final_model.fit(X_train, y_train)
    final_pred = final_model.predict(X_test)
    elapsed = time.perf_counter() - start

    report = classification_report(y_test, final_pred, target_names=target_names, output_dict=True)
    _write_classification_report(report, paths["tables"] / "svm_wine_report.csv")
    matrix = confusion_matrix(y_test, final_pred)
    _save_confusion_matrix(
        matrix,
        target_names,
        f"SVM ({best_kernel}) Confusion Matrix on Wine",
        paths["figures"] / "svm_wine_confusion.png",
    )

    metrics = {
        "algorithm": "SVM",
        "dataset": "Wine",
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": int(X_train.shape[1]),
        "selected_kernel": best_kernel,
        "accuracy": float(accuracy_score(y_test, final_pred)),
        "macro_f1": float(f1_score(y_test, final_pred, average="macro")),
        "runtime_seconds": float(elapsed),
    }
    _save_json(metrics, paths["metrics"] / "svm_wine_metrics.json")
    return metrics


def run_rbf_experiment(project_root: Path, paths: dict[str, Path]) -> dict:
    X_train, X_test, y_train, y_test, target_names = _prepare_wine_split()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    center_values = [6, 9, 12, 15, 18, 24, 30]
    comparison_records: list[dict] = []
    predictions_by_centers: dict[int, np.ndarray] = {}

    for n_centers in center_values:
        model = RBFNetworkClassifier(
            n_centers=n_centers,
            alpha=5e-3,
            random_state=RANDOM_STATE,
        )
        start = time.perf_counter()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        elapsed = time.perf_counter() - start
        predictions_by_centers[n_centers] = y_pred
        comparison_records.append(
            {
                "hidden_centers": n_centers,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
                "runtime_seconds": float(elapsed),
            }
        )

    comparison_frame = pd.DataFrame(comparison_records).sort_values(
        by=["accuracy", "macro_f1", "hidden_centers"],
        ascending=[False, False, True],
    )
    comparison_frame.to_csv(paths["tables"] / "rbf_wine_center_sweep.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(
        comparison_frame.sort_values("hidden_centers")["hidden_centers"],
        comparison_frame.sort_values("hidden_centers")["accuracy"],
        marker="o",
        label="Accuracy",
    )
    ax.plot(
        comparison_frame.sort_values("hidden_centers")["hidden_centers"],
        comparison_frame.sort_values("hidden_centers")["macro_f1"],
        marker="s",
        label="Macro-F1",
    )
    ax.set_xlabel("Number of RBF centers")
    ax.set_ylabel("Score")
    ax.set_ylim(0.6, 1.02)
    ax.set_title("RBF Network Sensitivity to Hidden Centers")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "rbf_wine_center_sweep.png", dpi=220)
    plt.close(fig)

    selected_centers = int(comparison_frame.iloc[0]["hidden_centers"])
    selected_pred = predictions_by_centers[selected_centers]
    report = classification_report(y_test, selected_pred, target_names=target_names, output_dict=True)
    _write_classification_report(report, paths["tables"] / "rbf_wine_report.csv")
    _save_confusion_matrix(
        confusion_matrix(y_test, selected_pred),
        target_names,
        f"RBF Network ({selected_centers} centers) on Wine",
        paths["figures"] / "rbf_wine_confusion.png",
    )

    metrics = {
        "algorithm": "RBF Network",
        "dataset": "Wine",
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": int(X_train.shape[1]),
        "hidden_centers": selected_centers,
        "accuracy": float(accuracy_score(y_test, selected_pred)),
        "macro_f1": float(f1_score(y_test, selected_pred, average="macro")),
        "runtime_seconds": float(comparison_frame.iloc[0]["runtime_seconds"]),
    }
    _save_json(metrics, paths["metrics"] / "rbf_wine_metrics.json")
    return metrics


def run_mlp_experiment(project_root: Path, paths: dict[str, Path]) -> dict:
    X_train, y_train, X_test, y_test, target_names = _prepare_fashion_subset()

    architecture_settings = [
        {"label": "MLP(128)", "hidden_layer_sizes": (128,)},
        {"label": "MLP(256,128)", "hidden_layer_sizes": (256, 128)},
        {"label": "MLP(256,128,64)", "hidden_layer_sizes": (256, 128, 64)},
    ]

    comparison_records: list[dict] = []
    model_registry: dict[str, MLPClassifier] = {}
    pred_registry: dict[str, np.ndarray] = {}

    for setting in architecture_settings:
        model = MLPClassifier(
            hidden_layer_sizes=setting["hidden_layer_sizes"],
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=30,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=RANDOM_STATE,
            verbose=False,
        )
        start = time.perf_counter()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        elapsed = time.perf_counter() - start
        result = {
            "architecture": setting["label"],
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
            "epochs": int(len(model.loss_curve_)),
            "runtime_seconds": float(elapsed),
        }
        comparison_records.append(result)
        model_registry[setting["label"]] = model
        pred_registry[setting["label"]] = y_pred

    comparison_frame = pd.DataFrame(comparison_records).sort_values(
        by=["accuracy", "macro_f1", "runtime_seconds"],
        ascending=[False, False, True],
    )
    comparison_frame.to_csv(paths["tables"] / "mlp_fashion_architecture_comparison.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    width = 0.35
    x = np.arange(len(comparison_frame))
    ax.bar(x - width / 2, comparison_frame["accuracy"], width=width, label="Accuracy")
    ax.bar(x + width / 2, comparison_frame["macro_f1"], width=width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_frame["architecture"], rotation=10)
    ax.set_ylim(0.70, 1.00)
    ax.set_ylabel("Score")
    ax.set_title("BP/MLP Architecture Comparison on Fashion-MNIST")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["figures"] / "mlp_fashion_architecture_comparison.png", dpi=220)
    plt.close(fig)

    best_label = str(comparison_frame.iloc[0]["architecture"])
    best_model = model_registry[best_label]
    best_pred = pred_registry[best_label]
    report = classification_report(y_test, best_pred, target_names=target_names, output_dict=True)
    _write_classification_report(report, paths["tables"] / "mlp_fashion_report.csv")
    _save_confusion_matrix(
        confusion_matrix(y_test, best_pred),
        target_names,
        f"BP/MLP ({best_label}) on Fashion-MNIST",
        paths["figures"] / "mlp_fashion_confusion.png",
    )

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(np.arange(1, len(best_model.loss_curve_) + 1), best_model.loss_curve_, marker="o", ms=3)
    ax.set_title(f"Training Loss of {best_label} on Fashion-MNIST")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.tight_layout()
    fig.savefig(paths["figures"] / "mlp_fashion_loss.png", dpi=220)
    plt.close(fig)

    top_row = comparison_frame.iloc[0]
    metrics = {
        "algorithm": "BP Network (MLP)",
        "dataset": "Fashion-MNIST",
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "features": int(X_train.shape[1]),
        "architecture": str(top_row["architecture"]),
        "epochs": int(top_row["epochs"]),
        "accuracy": float(top_row["accuracy"]),
        "macro_f1": float(top_row["macro_f1"]),
        "runtime_seconds": float(top_row["runtime_seconds"]),
    }
    _save_json(metrics, paths["metrics"] / "mlp_fashion_metrics.json")
    return metrics


def run_all_experiments(project_root: Path) -> dict[str, dict]:
    paths = ensure_result_dirs(project_root)
    _save_dataset_summary(paths)

    summary = {
        "kmeans": run_kmeans_experiment(project_root, paths),
        "svm": run_svm_experiment(project_root, paths),
        "rbf": run_rbf_experiment(project_root, paths),
        "mlp": run_mlp_experiment(project_root, paths),
    }

    _save_json(summary, paths["metrics"] / "summary_metrics.json")
    pd.DataFrame([{key: value for key, value in item.items()} for item in summary.values()]).to_csv(
        paths["tables"] / "summary_metrics.csv",
        index=False,
    )
    return summary

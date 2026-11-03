from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"


IRIS_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
FASHION_BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion"
)
FASHION_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

IRIS_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
    "species",
]

WINE_COLUMNS = [
    "class_id",
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]

FASHION_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def ensure_data_dirs() -> None:
    RAW_DATA_ROOT.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination

    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    with destination.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                handle.write(chunk)
    return destination


def download_iris() -> Path:
    ensure_data_dirs()
    return download_file(IRIS_URL, RAW_DATA_ROOT / "iris" / "iris.data")


def load_iris_dataset() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    iris_path = download_iris()
    frame = pd.read_csv(iris_path, header=None, names=IRIS_COLUMNS)
    frame = frame.dropna()
    target_names = sorted(frame["species"].unique())
    y = pd.Categorical(frame["species"], categories=target_names).codes
    X = frame[IRIS_COLUMNS[:-1]].to_numpy(dtype=float)
    return X, y.astype(int), IRIS_COLUMNS[:-1], target_names


def download_wine() -> Path:
    ensure_data_dirs()
    return download_file(WINE_URL, RAW_DATA_ROOT / "wine" / "wine.data")


def load_wine_dataset() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    wine_path = download_wine()
    frame = pd.read_csv(wine_path, header=None, names=WINE_COLUMNS)
    X = frame[WINE_COLUMNS[1:]].to_numpy(dtype=float)
    y = frame["class_id"].to_numpy(dtype=int) - 1
    target_names = ["class_1", "class_2", "class_3"]
    return X, y, WINE_COLUMNS[1:], target_names


def download_fashion_mnist() -> dict[str, Path]:
    ensure_data_dirs()
    base_dir = RAW_DATA_ROOT / "fashion_mnist"
    base_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, filename in FASHION_FILES.items():
        paths[key] = download_file(f"{FASHION_BASE_URL}/{filename}", base_dir / filename)
    return paths


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        header = handle.read(16)`n        if len(header) != 16:`n            raise ValueError(f"Truncated IDX image header in {path}")`n        magic, count, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError(f"Unexpected image magic number {magic} in {path}")
        buffer = handle.read()
    images = np.frombuffer(buffer, dtype=np.uint8)
    return images.reshape(count, rows * cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"Unexpected label magic number {magic} in {path}")
        buffer = handle.read()
    labels = np.frombuffer(buffer, dtype=np.uint8)
    if labels.shape[0] != count:
        raise ValueError(f"Label count mismatch in {path}")
    return labels


def load_fashion_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    paths = download_fashion_mnist()
    X_train = _read_idx_images(paths["train_images"]).astype(np.float32) / 255.0
    y_train = _read_idx_labels(paths["train_labels"]).astype(np.int64)
    X_test = _read_idx_images(paths["test_images"]).astype(np.float32) / 255.0
    y_test = _read_idx_labels(paths["test_labels"]).astype(np.int64)
    return X_train, y_train, X_test, y_test, FASHION_LABELS

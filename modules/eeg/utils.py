"""
Shared utilities for the PRĀNA EEG module.

Handles path resolution, dataset discovery, format detection,
and optional download of the UCI Epileptic Seizure Recognition dataset.
"""

from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ── Module paths ──────────────────────────────────────────────────────────────
MODULE_ROOT = Path(__file__).resolve().parent
DATASET_DIR = MODULE_ROOT / "dataset"
MODEL_DIR = MODULE_ROOT / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "eeg_model.pth"
META_PATH = DATASET_DIR / "dataset_meta.json"
INSPECT_PLOT_PATH = DATASET_DIR / "example_signal.png"

# UCI Epileptic Seizure Recognition (178 time points, 5 classes)
# Primary mirror — UCI direct link may 404; GitHub mirrors are tried in order.
UCI_EEG_URLS = [
    "https://raw.githubusercontent.com/apurvnnd/Epileptic-Seizure-Recognition-Using-ANN/master/data.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00361/data.csv",
]
UCI_EEG_FILENAME = "data.csv"
# Each segment: 4097 samples at 173.61 Hz, downsampled every 23rd point → 178 pts
UCI_SAMPLING_RATE = 173.61
UCI_SIGNAL_LENGTH = 178

# Label column candidates (case-insensitive)
LABEL_COLUMN_CANDIDATES = (
    "label", "labels", "y", "class", "target", "category", "diagnosis",
)

# Feature column patterns for wide CSV layouts
FEATURE_PREFIXES = ("x", "ch", "eeg", "feature", "sample")


@dataclass
class EEGDatasetInfo:
    """Metadata produced by dataset inspection."""

    n_samples: int
    signal_length: int
    n_channels: int
    sampling_rate: Optional[float]
    n_classes: int
    class_names: List[str]
    class_distribution: Dict[str, int]
    missing_values: int
    format_type: str
    source_files: List[str]
    is_pre_segmented: bool = True
    label_column: Optional[str] = None
    binary_mapping: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EEGDatasetInfo":
        return cls(**data)

    def save(self, path: Optional[Path] = None) -> Path:
        """Persist metadata for adaptive preprocessing."""
        out = Path(path or META_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return out

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "EEGDatasetInfo":
        src = Path(path or META_PATH)
        if not src.exists():
            raise FileNotFoundError(
                f"Dataset metadata not found at {src}. "
                "Run inspect_dataset.py first."
            )
        with open(src, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def ensure_dataset_dir() -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    return DATASET_DIR


def ensure_model_dir() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def get_model_path() -> Path:
    ensure_model_dir()
    return DEFAULT_MODEL_PATH


def discover_dataset_files() -> List[Path]:
    """Return sorted data files under dataset/ (excluding meta/plots)."""
    ensure_dataset_dir()
    skip = {META_PATH.name, INSPECT_PLOT_PATH.name, ".gitkeep"}
    extensions = {".csv", ".npy", ".npz", ".mat", ".edf", ".txt", ".tsv"}
    files: List[Path] = []
    for path in sorted(DATASET_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.name in skip or path.suffix.lower() not in extensions:
            continue
        files.append(path)
    return files


def dataset_is_empty() -> bool:
    return len(discover_dataset_files()) == 0


def download_uci_eeg_dataset(
    dest_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download the UCI Epileptic Seizure Recognition CSV if not present.

    Tries multiple mirrors, then falls back to a synthetic reference dataset.
    Returns path to the CSV.
    """
    dest = Path(dest_dir or DATASET_DIR)
    dest.mkdir(parents=True, exist_ok=True)
    out_path = dest / UCI_EEG_FILENAME

    if out_path.exists() and not force:
        print(f"  Dataset already exists: {out_path}")
        return out_path

    last_error: Optional[Exception] = None
    for url in UCI_EEG_URLS:
        print(f"  Downloading EEG dataset from {url} ...")
        try:
            urllib.request.urlretrieve(url, out_path)
            print(f"  Saved -> {out_path}")
            return out_path
        except Exception as exc:
            last_error = exc
            print(f"  [WARN] Download failed: {exc}")

    print("  All remote downloads failed - generating synthetic reference dataset...")
    try:
        return generate_synthetic_eeg_dataset(out_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to obtain EEG dataset: {last_error}\n"
            f"Synthetic fallback also failed: {exc}\n"
            "Place a CSV/NPZ/MAT file manually in modules/eeg/dataset/"
        ) from exc


def generate_synthetic_eeg_dataset(out_path: Path, n_per_class: int = 500) -> Path:
    """
    Create a synthetic EEG CSV mimicking UCI layout (178 features + label y).

    Class 1 = seizure-like (ABNORMAL); classes 2–5 = normal variants (NORMAL).
    """
    rng = np.random.default_rng(42)
    signal_length = UCI_SIGNAL_LENGTH
    rows: list[list[float]] = []

    for label in [1, 2, 3, 4, 5]:
        for _ in range(n_per_class):
            t = np.linspace(0, 1, signal_length, endpoint=False)
            if label == 1:
                # Seizure-like: elevated amplitude + sharp spikes
                signal = (
                    2.5 * np.sin(2 * np.pi * 12 * t)
                    + 1.5 * np.sin(2 * np.pi * 25 * t)
                    + rng.normal(0, 0.6, signal_length)
                )
                spikes = rng.integers(0, signal_length, size=6)
                signal[spikes] += rng.uniform(3, 6, size=len(spikes))
            else:
                # Normal resting EEG-like rhythms
                freq = 8 + label
                signal = (
                    np.sin(2 * np.pi * freq * t)
                    + 0.4 * np.sin(2 * np.pi * (freq + 3) * t)
                    + rng.normal(0, 0.15, signal_length)
                )
            rows.append(signal.tolist() + [float(label)])

    columns = [f"X{i + 1}" for i in range(signal_length)] + ["y"]
    df = pd.DataFrame(rows, columns=columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Synthetic dataset saved -> {out_path}  ({len(df)} samples)")
    return out_path


def _detect_label_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        if str(col).lower() in LABEL_COLUMN_CANDIDATES:
            return col
    # Last column is often the label in UCI-style CSVs
    if df.shape[1] >= 2:
        last = df.columns[-1]
        unique = df[last].nunique()
        if 2 <= unique <= 20:
            return last
    return None


def _feature_columns(df: pd.DataFrame, label_col: Optional[str]) -> List[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if label_col and label_col in numeric:
        numeric.remove(label_col)
    if numeric:
        return numeric

    # Unnamed UCI CSV: all columns except last
    if label_col:
        return [c for c in df.columns if c != label_col]
    return list(df.columns[:-1])


def load_csv_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load signals (N, L) and raw labels from a CSV file."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)

    # Drop pandas index columns (e.g. "Unnamed: 0")
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    # Assign generic column names when header is missing
    if df.columns.dtype == object and any(str(c).startswith("Unnamed") for c in df.columns):
        df.columns = [f"X{i + 1}" for i in range(len(df.columns))]

    label_col = _detect_label_column(df)
    feature_cols = _feature_columns(df, label_col)

    if not feature_cols:
        raise ValueError(f"No numeric feature columns found in {path}")

    signals = df[feature_cols].values.astype(np.float32)
    if label_col:
        labels = df[label_col].values
    else:
        labels = np.zeros(len(df), dtype=np.int64)

    return signals, labels, df


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    signal_key = next(
        (k for k in keys if k.lower() in ("signals", "x", "data", "eeg")),
        keys[0],
    )
    label_key = next(
        (k for k in keys if k.lower() in ("labels", "y", "target", "class")),
        keys[1] if len(keys) > 1 else None,
    )

    signals = np.asarray(data[signal_key], dtype=np.float32)
    if label_key:
        labels = np.asarray(data[label_key])
    else:
        labels = np.zeros(signals.shape[0], dtype=np.int64)

    return signals, labels


def load_mat_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise ImportError("scipy is required to load .mat files") from exc

    mat = loadmat(path)
    arrays = {
        k: v for k, v in mat.items()
        if not k.startswith("__") and isinstance(v, np.ndarray)
    }
    if not arrays:
        raise ValueError(f"No arrays found in {path}")

    # Prefer keys hinting at signals / labels
    signal_key = next(
        (k for k in arrays if re.search(r"signal|data|eeg|x", k, re.I)),
        next(iter(arrays)),
    )
    label_key = next(
        (k for k in arrays if re.search(r"label|y|class|target", k, re.I)),
        None,
    )

    signals = np.asarray(arrays[signal_key], dtype=np.float32)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)

    if label_key:
        labels = np.asarray(arrays[label_key]).squeeze()
    else:
        labels = np.zeros(signals.shape[0], dtype=np.int64)

    return signals, labels


def load_raw_dataset(
    auto_download: bool = True,
) -> Tuple[np.ndarray, np.ndarray, str, List[str]]:
    """
    Load the first usable dataset found in dataset/.

    Returns
    -------
    signals : (N, L) or (N, C, L)
    labels  : (N,)
    format_type : str
    source_files : list of paths used
    """
    if dataset_is_empty():
        if auto_download:
            print("  No EEG data found - downloading UCI reference dataset...")
            download_uci_eeg_dataset()
        else:
            raise FileNotFoundError(
                f"No EEG files in {DATASET_DIR}. "
                "Add data or run with auto_download=True."
            )

    files = discover_dataset_files()
    path = files[0]
    suffix = path.suffix.lower()

    if suffix == ".csv":
        signals, labels, _ = load_csv_dataset(path)
        fmt = "csv"
    elif suffix == ".npz":
        signals, labels = load_npz_dataset(path)
        fmt = "npz"
    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            signals = np.stack([a for a in arr])
            labels = np.zeros(len(signals), dtype=np.int64)
        else:
            signals = np.asarray(arr, dtype=np.float32)
            labels = np.zeros(signals.shape[0], dtype=np.int64)
        fmt = "npy"
    elif suffix == ".mat":
        signals, labels = load_mat_dataset(path)
        fmt = "mat"
    else:
        raise ValueError(f"Unsupported format: {suffix} ({path})")

    signals = _standardize_signal_shape(signals)
    labels = np.asarray(labels).squeeze()

    return signals, labels, fmt, [str(path)]


def _standardize_signal_shape(signals: np.ndarray) -> np.ndarray:
    """Ensure shape (N, signal_length) for single-channel windows."""
    if signals.ndim == 1:
        return signals.reshape(1, -1).astype(np.float32)
    if signals.ndim == 2:
        return signals.astype(np.float32)
    if signals.ndim == 3:
        # (N, channels, time) → use first channel for 1D CNN baseline
        return signals[:, 0, :].astype(np.float32)
    raise ValueError(f"Unsupported signal shape: {signals.shape}")


def infer_sampling_rate(
    format_type: str,
    source_files: List[str],
    signal_length: int,
) -> Optional[float]:
    """Infer sampling rate from known datasets or filename hints."""
    if UCI_EEG_FILENAME in " ".join(source_files):
        return UCI_SAMPLING_RATE
    if signal_length == UCI_SIGNAL_LENGTH:
        return UCI_SAMPLING_RATE

    for src in source_files:
        name = Path(src).stem.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*hz", name)
        if match:
            return float(match.group(1))
    return None


def build_class_names(labels: np.ndarray) -> List[str]:
    unique = sorted(np.unique(labels).tolist())
    return [str(int(u)) if isinstance(u, (int, np.integer, float)) else str(u)
            for u in unique]


def build_binary_mapping(labels: np.ndarray) -> Dict[str, str]:
    """
    Map raw labels to NORMAL / ABNORMAL for clinical inference.

    UCI convention: class 1 = seizure; classes 2–5 = non-seizure.
    """
    unique = set(np.unique(labels).tolist())
    if 1 in unique and any(c in unique for c in (2, 3, 4, 5)):
        return {"1": "ABNORMAL", **{str(i): "NORMAL" for i in range(2, 6)}}
    if len(unique) == 2:
        sorted_u = sorted(unique)
        return {str(sorted_u[0]): "NORMAL", str(sorted_u[1]): "ABNORMAL"}
    # Default: lowest label NORMAL, others ABNORMAL
    sorted_u = sorted(unique)
    mapping = {str(sorted_u[0]): "NORMAL"}
    for u in sorted_u[1:]:
        mapping[str(u)] = "ABNORMAL"
    return mapping


def inspect_dataset(
    auto_download: bool = True,
) -> EEGDatasetInfo:
    """
    Load dataset and compute inspection metadata.
    """
    signals, labels, fmt, sources = load_raw_dataset(auto_download=auto_download)
    n_samples, signal_length = signals.shape

    label_col = None
    if sources and sources[0].endswith(".csv"):
        try:
            df_head = pd.read_csv(sources[0], nrows=5)
            label_col = _detect_label_column(df_head)
        except Exception:
            pass

    class_names = build_class_names(labels)
    dist: Dict[str, int] = {}
    for lbl in labels:
        key = str(int(lbl)) if isinstance(lbl, (int, np.integer, float)) else str(lbl)
        dist[key] = dist.get(key, 0) + 1

    missing = int(np.isnan(signals).sum())

    info = EEGDatasetInfo(
        n_samples=n_samples,
        signal_length=signal_length,
        n_channels=1,
        sampling_rate=infer_sampling_rate(fmt, sources, signal_length),
        n_classes=len(class_names),
        class_names=class_names,
        class_distribution=dist,
        missing_values=missing,
        format_type=fmt,
        source_files=sources,
        is_pre_segmented=True,
        label_column=label_col,
        binary_mapping=build_binary_mapping(labels),
    )
    return info


def raw_to_binary_labels(
    labels: np.ndarray,
    mapping: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Convert raw class labels to 0=NORMAL, 1=ABNORMAL."""
    if mapping is None:
        mapping = build_binary_mapping(labels)

    binary = np.zeros(len(labels), dtype=np.int64)
    for i, lbl in enumerate(labels):
        key = str(int(lbl)) if isinstance(lbl, (int, np.integer, float)) else str(lbl)
        status = mapping.get(key, "ABNORMAL")
        binary[i] = 0 if status == "NORMAL" else 1
    return binary


def class_index_to_status(class_idx: int, meta: EEGDatasetInfo) -> str:
    """Map model class index to NORMAL / ABNORMAL."""
    if meta.n_classes == 2:
        return "NORMAL" if class_idx == 0 else "ABNORMAL"
    # Multi-class: use binary mapping on original class name
    if class_idx < len(meta.class_names):
        raw = meta.class_names[class_idx]
        return meta.binary_mapping.get(raw, "ABNORMAL")
    return "ABNORMAL"

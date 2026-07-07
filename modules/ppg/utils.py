"""
Shared utilities for the PRANA PPG module.

Handles path resolution, dataset discovery, format detection,
and synthetic/reference dataset generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Module paths ──────────────────────────────────────────────────────────────
MODULE_ROOT = Path(__file__).resolve().parent
DATASET_DIR = MODULE_ROOT / "dataset"
MODEL_DIR = MODULE_ROOT / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "ppg_model.pth"
META_PATH = DATASET_DIR / "dataset_meta.json"
INSPECT_PLOT_PATH = DATASET_DIR / "example_signal.png"

PPG_FILENAME = "ppg_data.csv"
ALT_PPG_FILENAME = "PPG_Dataset.csv"
DEFAULT_SAMPLING_RATE = 125.0   # Hz — common for 2000-sample PPG windows
DEFAULT_SIGNAL_LENGTH = 512     # synthetic fallback length

LABEL_COLUMN_CANDIDATES = (
    "label", "labels", "y", "class", "target", "category", "diagnosis",
)
SUBJECT_COLUMN_CANDIDATES = (
    "subject_id", "subject", "patient_id", "patient", "id", "record",
)

# Clinical string labels → binary status
NORMAL_LABEL_ALIASES = {"normal", "0", "n", "healthy", "good"}
ABNORMAL_LABEL_ALIASES = {"mi", "abnormal", "1", "a", "bad", "disease"}


@dataclass
class PPGDatasetInfo:
    """Metadata produced by dataset inspection."""

    n_samples: int
    signal_length: int
    sampling_rate: float
    n_subjects: int
    n_classes: int
    class_names: List[str]
    class_distribution: Dict[str, int]
    missing_values: int
    format_type: str
    source_files: List[str]
    signal_mean: float
    signal_std: float
    signal_min: float
    signal_max: float
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    is_pre_segmented: bool = True
    label_column: Optional[str] = None
    subject_column: Optional[str] = None
    binary_mapping: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PPGDatasetInfo":
        return cls(**data)

    def save(self, path: Optional[Path] = None) -> Path:
        out = Path(path or META_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return out

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "PPGDatasetInfo":
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
    """Return data files under dataset/, preferring canonical PPG filenames."""
    ensure_dataset_dir()
    skip = {META_PATH.name, INSPECT_PLOT_PATH.name, ".gitkeep"}
    extensions = {".csv", ".npy", ".npz", ".mat", ".txt", ".tsv"}

    # Prefer ppg_data.csv, then PPG_Dataset.csv
    for name in (PPG_FILENAME, ALT_PPG_FILENAME):
        preferred = DATASET_DIR / name
        if preferred.exists():
            return [preferred]

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


def _detect_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower_map:
            return lower_map[candidate]
    return None


def _feature_columns(df: pd.DataFrame, exclude: List[str]) -> List[str]:
    """Select numeric feature columns, excluding metadata columns."""
    exclude_set = set(exclude)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric if c not in exclude_set]

    if feature_cols:
        return feature_cols

    # Wide layout: all columns except label/metadata
    return [c for c in df.columns if c not in exclude_set]


def _label_to_status(raw_label: Any) -> str:
    """Map a raw label value to NORMAL or ABNORMAL."""
    key = str(raw_label).strip().lower()
    if key in NORMAL_LABEL_ALIASES:
        return "NORMAL"
    if key in ABNORMAL_LABEL_ALIASES:
        return "ABNORMAL"
    # Numeric: lowest class → NORMAL
    try:
        return "NORMAL" if int(float(key)) == 0 else "ABNORMAL"
    except (ValueError, TypeError):
        return "ABNORMAL"


def build_binary_mapping(labels: np.ndarray) -> Dict[str, str]:
    """Build raw-label → clinical status mapping."""
    mapping: Dict[str, str] = {}
    for lbl in np.unique(labels):
        key = str(lbl).strip()
        mapping[key] = _label_to_status(lbl)
    return mapping


def raw_to_binary_labels(
    labels: np.ndarray,
    mapping: Optional[Dict[str, str]] = None,
) -> np.ndarray:
    """Convert raw class labels to 0=NORMAL, 1=ABNORMAL."""
    if mapping is None:
        mapping = build_binary_mapping(labels)

    binary = np.zeros(len(labels), dtype=np.int64)
    for i, lbl in enumerate(labels):
        key = str(lbl).strip()
        status = mapping.get(key, _label_to_status(lbl))
        binary[i] = 0 if status == "NORMAL" else 1
    return binary


def load_csv_dataset(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load signals (N, L), labels (N,), subject_ids (N,) from CSV."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)

    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if df.columns.dtype == object and any(str(c).startswith("Unnamed") for c in df.columns):
        df.columns = [f"col_{i}" for i in range(len(df.columns))]

    label_col = _detect_column(df, LABEL_COLUMN_CANDIDATES)
    subject_col = _detect_column(df, SUBJECT_COLUMN_CANDIDATES)

    exclude = [c for c in (label_col, subject_col) if c]
    feature_cols = _feature_columns(df, exclude)

    if not feature_cols:
        if label_col:
            feature_cols = [c for c in df.columns if c != label_col]
        else:
            feature_cols = list(df.columns[:-1])

    signals = df[feature_cols].values.astype(np.float32)
    labels = df[label_col].values if label_col else np.zeros(len(df), dtype=np.int64)

    if subject_col:
        subjects = df[subject_col].values
    else:
        subjects = np.arange(len(df))

    return signals, labels, subjects, df


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    signal_key = next(
        (k for k in keys if k.lower() in ("signals", "ppg", "x", "data")),
        keys[0],
    )
    label_key = next(
        (k for k in keys if k.lower() in ("labels", "y", "target", "class")),
        None,
    )
    subject_key = next(
        (k for k in keys if k.lower() in ("subjects", "subject_id", "patient_id")),
        None,
    )

    signals = np.asarray(data[signal_key], dtype=np.float32)
    labels = np.asarray(data[label_key]) if label_key else np.zeros(signals.shape[0])
    subjects = (
        np.asarray(data[subject_key])
        if subject_key
        else np.arange(signals.shape[0])
    )
    return signals, labels, subjects


def _standardize_signal_shape(signals: np.ndarray) -> np.ndarray:
    if signals.ndim == 1:
        return signals.reshape(1, -1).astype(np.float32)
    if signals.ndim == 2:
        return signals.astype(np.float32)
    if signals.ndim == 3:
        return signals[:, 0, :].astype(np.float32)
    raise ValueError(f"Unsupported signal shape: {signals.shape}")


def generate_synthetic_ppg_dataset(
    out_path: Optional[Path] = None,
    n_subjects: int = 40,
    windows_per_subject: int = 15,
    signal_length: int = DEFAULT_SIGNAL_LENGTH,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    seed: int = 42,
) -> Path:
    """
    Create a reference PPG CSV with normal and abnormal pulse patterns.

    Uses neurokit2 when available; falls back to analytic PPG simulation.
    """
    rng = np.random.default_rng(seed)
    out = Path(out_path or DATASET_DIR / PPG_FILENAME)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import neurokit2 as nk
        has_nk = True
    except ImportError:
        has_nk = False

    duration = signal_length / sampling_rate
    rows: list[list[float]] = []

    for subj in range(n_subjects):
        is_abnormal = subj >= n_subjects // 2
        subject_id = f"S{subj + 1:03d}"

        for _ in range(windows_per_subject):
            if has_nk:
                hr = float(rng.uniform(95, 140) if is_abnormal else rng.uniform(58, 85))
                noise = float(rng.uniform(0.08, 0.25) if is_abnormal else rng.uniform(0.01, 0.04))
                ppg = nk.ppg_simulate(
                    duration=duration,
                    sampling_rate=int(sampling_rate),
                    heart_rate=hr,
                    noise=noise,
                )
                if len(ppg) > signal_length:
                    ppg = ppg[:signal_length]
                elif len(ppg) < signal_length:
                    ppg = np.pad(ppg, (0, signal_length - len(ppg)), mode="edge")
                if is_abnormal and rng.random() > 0.5:
                    spikes = rng.integers(0, signal_length, size=4)
                    ppg = ppg.copy()
                    ppg[spikes] += rng.uniform(0.3, 0.8, size=4)
            else:
                t = np.linspace(0, duration, signal_length, endpoint=False)
                hr = rng.uniform(95, 140) if is_abnormal else rng.uniform(58, 85)
                freq = hr / 60.0
                ppg = (
                    0.6 * np.sin(2 * np.pi * freq * t)
                    + 0.15 * np.sin(2 * np.pi * 2 * freq * t)
                    + rng.normal(0, 0.08 if is_abnormal else 0.02, signal_length)
                )

            label = 1 if is_abnormal else 0
            rows.append([subject_id, label] + ppg.tolist())

    columns = ["subject_id", "label"] + [f"X{i + 1}" for i in range(signal_length)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out, index=False)
    print(f"  Synthetic PPG dataset saved -> {out}  ({len(df)} samples, {n_subjects} subjects)")
    return out


def ensure_ppg_dataset(auto_generate: bool = True) -> Path:
    """Return path to dataset CSV, generating synthetic data if needed."""
    ensure_dataset_dir()
    files = discover_dataset_files()
    if files:
        return files[0]

    out = DATASET_DIR / PPG_FILENAME
    if auto_generate:
        print("  No PPG data found - generating synthetic reference dataset...")
        return generate_synthetic_ppg_dataset(out)
    raise FileNotFoundError(
        f"No PPG files in {DATASET_DIR}. Add data or enable auto_generate."
    )


def load_raw_dataset(
    auto_generate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, List[str]]:
    """
    Load PPG dataset.

    Returns signals (N, L), labels (N,), subjects (N,), format, source paths.
    """
    if dataset_is_empty():
        ensure_ppg_dataset(auto_generate=auto_generate)

    path = discover_dataset_files()[0]
    suffix = path.suffix.lower()

    if suffix == ".csv":
        signals, labels, subjects, _ = load_csv_dataset(path)
        fmt = "csv"
    elif suffix == ".npz":
        signals, labels, subjects = load_npz_dataset(path)
        fmt = "npz"
    elif suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        signals = np.asarray(arr, dtype=np.float32)
        labels = np.zeros(signals.shape[0], dtype=np.int64)
        subjects = np.arange(signals.shape[0])
        fmt = "npy"
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    signals = _standardize_signal_shape(signals)
    labels = np.asarray(labels).squeeze()
    subjects = np.asarray(subjects).squeeze()

    return signals, labels, subjects, fmt, [str(path)]


def infer_sampling_rate(source_files: List[str], signal_length: int) -> float:
    """Infer sampling rate from filename hints or known PPG window lengths."""
    for src in source_files:
        name = Path(src).stem.lower()
        match = re.search(r"(\d+(?:\.\d+)?)\s*hz", name)
        if match:
            return float(match.group(1))

    # Common clinical PPG: 2000 samples @ 125 Hz ≈ 16 s window
    if signal_length == 2000:
        return 125.0
    if signal_length == DEFAULT_SIGNAL_LENGTH:
        return 128.0
    return DEFAULT_SAMPLING_RATE


def inspect_dataset(auto_generate: bool = True) -> PPGDatasetInfo:
    """Load dataset and compute inspection metadata."""
    signals, labels, subjects, fmt, sources = load_raw_dataset(auto_generate=auto_generate)
    n_samples, signal_length = signals.shape

    label_col = subject_col = None
    columns: List[str] = []
    dtypes: Dict[str, str] = {}

    if sources and sources[0].endswith(".csv"):
        try:
            df_head = pd.read_csv(sources[0], nrows=5)
            label_col = _detect_column(df_head, LABEL_COLUMN_CANDIDATES)
            subject_col = _detect_column(df_head, SUBJECT_COLUMN_CANDIDATES)
            columns = [str(c) for c in df_head.columns.tolist()]
            dtypes = {str(k): str(v) for k, v in df_head.dtypes.items()}
        except Exception:
            pass

    class_names = [str(lbl).strip() for lbl in sorted(np.unique(labels), key=str)]
    dist: Dict[str, int] = {}
    for lbl in labels:
        key = str(lbl).strip()
        dist[key] = dist.get(key, 0) + 1

    n_subjects = len(np.unique(subjects))
    binary_mapping = build_binary_mapping(labels)

    return PPGDatasetInfo(
        n_samples=n_samples,
        signal_length=signal_length,
        sampling_rate=infer_sampling_rate(sources, signal_length),
        n_subjects=n_subjects,
        n_classes=len(class_names),
        class_names=class_names,
        class_distribution=dist,
        missing_values=int(np.isnan(signals).sum()),
        format_type=fmt,
        source_files=sources,
        signal_mean=float(np.nanmean(signals)),
        signal_std=float(np.nanstd(signals)),
        signal_min=float(np.nanmin(signals)),
        signal_max=float(np.nanmax(signals)),
        columns=columns,
        dtypes=dtypes,
        is_pre_segmented=True,
        label_column=label_col,
        subject_column=subject_col,
        binary_mapping=binary_mapping,
    )

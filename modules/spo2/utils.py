"""
Shared utilities for the PRANA SpO2 module.

Dataset discovery, synchronized PPG/SpO2 loading, and synthetic data generation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.spo2.clinical_rules import class_index_to_status

MODULE_ROOT = Path(__file__).resolve().parent
DATASET_DIR = MODULE_ROOT / "dataset"
MODEL_DIR = MODULE_ROOT / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "spo2_model.pth"
META_PATH = DATASET_DIR / "dataset_meta.json"
INSPECT_PLOT_PATH = DATASET_DIR / "example_signal.png"

SPO2_FILENAME = "spo2_data.csv"
DEFAULT_SAMPLING_RATE = 64.0
DEFAULT_SIGNAL_LENGTH = 256
N_CHANNELS = 2  # channel 0 = PPG, channel 1 = SpO2 trace (%)

LABEL_COLUMN_CANDIDATES = ("label", "labels", "y", "class", "target", "category")
SUBJECT_COLUMN_CANDIDATES = ("subject_id", "subject", "patient_id", "patient")
SPO2_REF_CANDIDATES = ("spo2", "spo2_mean", "oxygen_saturation", "saturation", "o2")


@dataclass
class SpO2DatasetInfo:
    """Metadata from dataset inspection."""

    n_samples: int
    signal_length: int
    sampling_rate: float
    n_channels: int
    n_subjects: int
    n_classes: int
    class_names: List[str]
    class_distribution: Dict[str, int]
    missing_values: int
    format_type: str
    source_files: List[str]
    spo2_min: float
    spo2_max: float
    spo2_mean: float
    signal_mean: float
    signal_std: float
    task_type: str = "classification"   # "classification" | "regression"
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    is_pre_segmented: bool = True
    binary_mode: bool = False
    label_column: Optional[str] = None
    subject_column: Optional[str] = None
    spo2_column: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpO2DatasetInfo":
        return cls(**data)

    def save(self, path: Optional[Path] = None) -> Path:
        out = Path(path or META_PATH)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return out

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SpO2DatasetInfo":
        src = Path(path or META_PATH)
        if not src.exists():
            raise FileNotFoundError(
                f"Metadata not found at {src}. Run inspect_dataset.py first."
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
    """Return data files under dataset/, preferring spo2_data.csv."""
    ensure_dataset_dir()
    skip = {META_PATH.name, INSPECT_PLOT_PATH.name, ".gitkeep"}
    extensions = {".csv", ".npy", ".npz", ".mat"}

    preferred = DATASET_DIR / SPO2_FILENAME
    if preferred.exists():
        return [preferred]

    files: List[Path] = []
    for path in sorted(DATASET_DIR.rglob("*")):
        if path.is_file() and path.name not in skip and path.suffix.lower() in extensions:
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


def _parse_csv_columns(
    df: pd.DataFrame,
) -> Tuple[List[str], List[str], Optional[str], Optional[str], Optional[str]]:
    """Identify PPG cols (X*), SpO2 trace cols (S*), label, subject, spo2 ref."""
    label_col = _detect_column(df, LABEL_COLUMN_CANDIDATES)
    subject_col = _detect_column(df, SUBJECT_COLUMN_CANDIDATES)
    spo2_col = _detect_column(df, SPO2_REF_CANDIDATES)

    # Match X1, X2 ... and S1, S2 ... (exclude spo2_mean which starts with 's')
    ppg_cols = [c for c in df.columns if re.match(r"^X\d+$", str(c), re.I)]
    spo2_cols = [c for c in df.columns if re.match(r"^S\d+$", str(c), re.I)]

    if not ppg_cols and not spo2_cols:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = [c for c in (label_col, subject_col, spo2_col) if c]
        numeric = [c for c in numeric if c not in exclude]
        mid = len(numeric) // 2
        ppg_cols = numeric[:mid] if mid else numeric
        spo2_cols = numeric[mid:] if mid else numeric

    return ppg_cols, spo2_cols, label_col, subject_col, spo2_col


def _detect_task_type(labels: np.ndarray) -> str:
    """Infer classification vs regression from label distribution."""
    try:
        numeric = labels.astype(np.float64)
    except (ValueError, TypeError):
        return "classification"

    unique = np.unique(numeric)
    if len(unique) <= 20 and np.all(np.equal(numeric, np.round(numeric))):
        return "classification"
    if np.all((numeric >= 50) & (numeric <= 100)):
        return "regression"
    return "classification"


def load_csv_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load (N, 2, L) signals, labels, subjects, spo2_reference."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, header=None)

    drop = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop:
        df = df.drop(columns=drop)

    ppg_cols, spo2_cols, label_col, subject_col, spo2_col = _parse_csv_columns(df)

    if not ppg_cols:
        raise ValueError(f"No PPG columns found in {path}")

    ppg = df[ppg_cols].values.astype(np.float32)
    if spo2_cols and len(spo2_cols) == len(ppg_cols):
        spo2_trace = df[spo2_cols].values.astype(np.float32)
    else:
        ref = df[spo2_col].values if spo2_col else np.full(len(df), 97.0)
        spo2_trace = np.tile(ref.reshape(-1, 1), (1, ppg.shape[1]))

    signals = np.stack([ppg, spo2_trace], axis=1)
    labels = df[label_col].values if label_col else np.zeros(len(df), dtype=np.int64)
    subjects = df[subject_col].values if subject_col else np.arange(len(df))
    spo2_ref = df[spo2_col].values.astype(np.float32) if spo2_col else spo2_trace.mean(axis=1)

    return signals, labels, subjects, spo2_ref, df


def load_npz_dataset(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())

    sig_key = next((k for k in keys if k.lower() in ("signals", "x", "data")), keys[0])
    lbl_key = next((k for k in keys if k.lower() in ("labels", "y", "class")), None)
    sub_key = next((k for k in keys if "subject" in k.lower()), None)
    spo2_key = next((k for k in keys if "spo2" in k.lower() or "sat" in k.lower()), None)

    signals = np.asarray(data[sig_key], dtype=np.float32)
    if signals.ndim == 2:
        signals = np.stack([signals, signals], axis=1)

    labels = np.asarray(data[lbl_key]) if lbl_key else np.zeros(signals.shape[0])
    subjects = np.asarray(data[sub_key]) if sub_key else np.arange(signals.shape[0])
    if spo2_key:
        spo2_ref = np.asarray(data[spo2_key], dtype=np.float32)
    else:
        spo2_ref = signals[:, 1, :].mean(axis=1) if signals.shape[1] > 1 else np.full(signals.shape[0], 97.0)

    return signals, labels, subjects, spo2_ref


def generate_synthetic_spo2_dataset(
    out_path: Optional[Path] = None,
    n_subjects: int = 40,
    windows_per_subject: int = 12,
    signal_length: int = DEFAULT_SIGNAL_LENGTH,
    sampling_rate: float = DEFAULT_SAMPLING_RATE,
    num_classes: int = 2,
    seed: int = 42,
) -> Path:
    """Generate synchronized PPG + SpO2 trace windows with clinical labels."""
    rng = np.random.default_rng(seed)
    out = Path(out_path or DATASET_DIR / SPO2_FILENAME)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        import neurokit2 as nk
        has_nk = True
    except ImportError:
        has_nk = False

    duration = signal_length / sampling_rate
    rows: list[list] = []

    class_ranges = [(95.0, 100.0), (90.0, 94.5), (85.0, 89.5), (72.0, 84.5)]
    if num_classes == 2:
        class_ranges = [(95.0, 100.0), (72.0, 94.5)]

    for subj in range(n_subjects):
        subject_id = f"S{subj + 1:03d}"
        for _ in range(windows_per_subject):
            cls = int(rng.integers(0, num_classes))
            lo, hi = class_ranges[min(cls, len(class_ranges) - 1)]
            mean_spo2 = float(rng.uniform(lo, hi))

            t = np.linspace(0, duration, signal_length, endpoint=False)
            spo2_trace = mean_spo2 + rng.normal(0, 0.4, signal_length)
            if cls >= 2:
                dip_idx = rng.integers(0, signal_length, size=int(rng.integers(1, 4)))
                spo2_trace[dip_idx] -= rng.uniform(2, 6, size=len(dip_idx))
            spo2_trace = np.clip(spo2_trace, 70, 100)

            if has_nk:
                hr_map = [72.0, 80.0, 95.0, 110.0] if num_classes >= 4 else [72.0, 100.0]
                hr = float(rng.uniform(hr_map[min(cls, len(hr_map) - 1)] - 5, hr_map[min(cls, len(hr_map) - 1)] + 5))
                noise = float(0.02 + cls * 0.04)
                ppg = nk.ppg_simulate(
                    duration=duration,
                    sampling_rate=int(sampling_rate),
                    heart_rate=hr,
                    noise=noise,
                )
                if len(ppg) != signal_length:
                    ppg = np.interp(t, np.linspace(0, duration, len(ppg)), ppg)
            else:
                freq = rng.uniform(0.9, 1.5)
                ppg = np.sin(2 * np.pi * freq * t) + rng.normal(0, 0.05, signal_length)

            label = cls if num_classes > 2 else (0 if mean_spo2 >= 95 else 1)
            row = [subject_id, label, round(float(spo2_trace.mean()), 2)]
            row.extend(ppg.tolist())
            row.extend(spo2_trace.tolist())
            rows.append(row)

    ppg_cols = [f"X{i + 1}" for i in range(signal_length)]
    spo2_cols = [f"S{i + 1}" for i in range(signal_length)]
    columns = ["subject_id", "label", "spo2_mean"] + ppg_cols + spo2_cols
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(out, index=False)
    print(f"  Synthetic SpO2 dataset saved -> {out}  ({len(df)} samples, {n_subjects} subjects)")
    return out


def ensure_spo2_dataset(auto_generate: bool = True) -> Path:
    ensure_dataset_dir()
    files = discover_dataset_files()
    if files:
        return files[0]
    if auto_generate:
        print("  No SpO2 data found - generating synthetic reference dataset...")
        return generate_synthetic_spo2_dataset(DATASET_DIR / SPO2_FILENAME)
    raise FileNotFoundError(f"No SpO2 files in {DATASET_DIR}.")


def load_raw_dataset(
    auto_generate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, List[str]]:
    """
    Returns signals (N, 2, L), labels, subjects, spo2_ref, format, sources.
    """
    if dataset_is_empty():
        ensure_spo2_dataset(auto_generate=auto_generate)

    path = discover_dataset_files()[0]
    suffix = path.suffix.lower()

    if suffix == ".csv":
        signals, labels, subjects, spo2_ref, _ = load_csv_dataset(path)
        fmt = "csv"
    elif suffix == ".npz":
        signals, labels, subjects, spo2_ref = load_npz_dataset(path)
        fmt = "npz"
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    labels = np.asarray(labels).squeeze()
    subjects = np.asarray(subjects).squeeze()
    spo2_ref = np.asarray(spo2_ref).squeeze().astype(np.float32)

    return signals, labels, subjects, spo2_ref, fmt, [str(path)]


def inspect_dataset(auto_generate: bool = True) -> SpO2DatasetInfo:
    signals, labels, subjects, spo2_ref, fmt, sources = load_raw_dataset(auto_generate)
    n_samples = signals.shape[0]
    n_channels, signal_length = signals.shape[1], signals.shape[2]

    label_col = subject_col = spo2_col = None
    columns: List[str] = []
    dtypes: Dict[str, str] = {}

    if sources and sources[0].endswith(".csv"):
        try:
            head = pd.read_csv(sources[0], nrows=5)
            label_col = _detect_column(head, LABEL_COLUMN_CANDIDATES)
            subject_col = _detect_column(head, SUBJECT_COLUMN_CANDIDATES)
            spo2_col = _detect_column(head, SPO2_REF_CANDIDATES)
            columns = [str(c) for c in head.columns.tolist()]
            dtypes = {str(k): str(v) for k, v in head.dtypes.items()}
        except Exception:
            pass

    task_type = _detect_task_type(labels)
    unique_labels = sorted(np.unique(labels).tolist(), key=lambda x: str(x))
    n_classes = len(unique_labels)
    binary_mode = n_classes == 2 and task_type == "classification"

    dist: Dict[str, int] = {}
    for lbl in labels:
        key = str(int(lbl)) if isinstance(lbl, (int, np.integer, float)) and float(lbl) == int(float(lbl)) else str(lbl)
        dist[key] = dist.get(key, 0) + 1

    if task_type == "classification":
        class_names = [class_index_to_status(int(u), n_classes) if str(u).isdigit() else str(u) for u in unique_labels]
    else:
        class_names = ["regression"]

    return SpO2DatasetInfo(
        n_samples=n_samples,
        signal_length=signal_length,
        sampling_rate=DEFAULT_SAMPLING_RATE,
        n_channels=n_channels,
        n_subjects=len(np.unique(subjects)),
        n_classes=n_classes if task_type == "classification" else 1,
        class_names=class_names,
        class_distribution=dist,
        missing_values=int(np.isnan(signals).sum()),
        format_type=fmt,
        source_files=sources,
        spo2_min=float(np.min(spo2_ref)),
        spo2_max=float(np.max(spo2_ref)),
        spo2_mean=float(np.mean(spo2_ref)),
        signal_mean=float(np.nanmean(signals)),
        signal_std=float(np.nanstd(signals)),
        task_type=task_type,
        columns=columns,
        dtypes=dtypes,
        is_pre_segmented=True,
        binary_mode=binary_mode,
        label_column=label_col,
        subject_column=subject_col,
        spo2_column=spo2_col,
    )


def encode_labels(labels: np.ndarray, task_type: str = "classification") -> np.ndarray:
    """Encode labels for classification (int64) or regression (float32)."""
    if task_type == "regression":
        return labels.astype(np.float32)
    return labels.astype(np.int64)

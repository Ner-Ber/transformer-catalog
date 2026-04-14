"""Load ingested catalogs and build supervised windows for next-magnitude regression."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from eq_mag_prediction.utilities import catalog_processing
from eq_mag_prediction.utilities import loading_utils

REQUIRED_COLUMNS = ("time", "magnitude")
OPTIONAL_NUMERIC = ("latitude", "longitude", "depth")


def default_ingested_dir() -> Path:
    """Sibling ``eq_mag_prediction`` ingested path relative to cwd."""
    return (Path.cwd().resolve().parent / "eq_mag_prediction" / "results" / "catalogs" / "ingested")


def resolve_data_dir(path: str | Path | None) -> Path:
    """Return absolute :class:`Path`.

    If *path* is relative and starts with ``results/``, resolve via
    :func:`eq_mag_prediction.utilities.loading_utils.get_resource_path` (paths
    relative to the ``eq_mag_prediction`` repo checkout).
    """
    if path is None:
        return default_ingested_dir()
    p = Path(path)
    if p.is_absolute():
        return p
    s = str(p).replace("\\", "/").lstrip("/")
    if s.startswith("results/"):
        return Path(loading_utils.get_resource_path(s))
    return (Path.cwd() / p).resolve()


def list_catalog_csvs(data_dir: Path) -> List[Path]:
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    paths = sorted(data_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files under {data_dir}")
    return paths


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            raise ValueError(f"Missing required column {c!r}; got {list(df.columns)}")
    out = df.copy()
    out = out.sort_values("time").reset_index(drop=True)
    for c in OPTIONAL_NUMERIC:
        if c not in out.columns:
            out[c] = 0.0
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            if out[c].isna().any():
                raise ValueError(f"Non-numeric or NaN in column {c!r}")
    out["time"] = pd.to_numeric(out["time"], errors="coerce")
    out["magnitude"] = pd.to_numeric(out["magnitude"], errors="coerce")
    if out["time"].isna().any() or out["magnitude"].isna().any():
        raise ValueError("NaN in time or magnitude after coercion")
    return out


def windows_from_prepared(df: pd.DataFrame, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build ``(X, y)`` with X shape ``(num_windows, seq_len, n_features)``."""
    n = len(df)
    if n < seq_len + 1:
        raise ValueError(
            f"Need at least seq_len + 1 rows (got n={n}, seq_len={seq_len})"
        )
    t = df["time"].to_numpy(dtype=np.float64)
    mag = df["magnitude"].to_numpy(dtype=np.float64)
    lat = df["latitude"].to_numpy(dtype=np.float64)
    lon = df["longitude"].to_numpy(dtype=np.float64)
    dep = df["depth"].to_numpy(dtype=np.float64)
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0.0
    dt = np.maximum(dt, 0.0)
    log_dt = np.log1p(dt)
    feats = np.stack([log_dt, mag, lat, lon, dep], axis=1)
    x_list = [feats[i : i + seq_len] for i in range(n - seq_len)]
    X = np.stack(x_list, axis=0)
    y = mag[seq_len:n]
    return X, y


def load_all_windows(
    data_dir: Path,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """Load every ``*.csv`` under *data_dir*, build windows per file, concatenate.

    Returns ``X, y, file_hashes`` where ``file_hashes`` maps basename to SHA256
    of the prepared frame (via :func:`catalog_processing.hash_pandas_object`).
    """
    paths = list_catalog_csvs(data_dir)
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    file_hashes: Dict[str, str] = {}
    for path in paths:
        raw = pd.read_csv(path)
        if len(raw) == 0:
            raise ValueError(f"Empty catalog: {path}")
        prepared = _prepare_frame(raw)
        file_hashes[path.name] = catalog_processing.hash_pandas_object(prepared)
        wx, wy = windows_from_prepared(prepared, seq_len)
        xs.append(wx)
        ys.append(wy)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return X, y, file_hashes


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler


def transform_X(X: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    shape = X.shape
    flat = X.reshape(-1, shape[-1])
    out = scaler.transform(flat)
    return out.reshape(shape)

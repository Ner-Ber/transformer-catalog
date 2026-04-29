"""Load ingested catalogs and build supervised windows (next magnitude + inter-event time)."""

from __future__ import annotations

import dataclasses
import os
import pathlib
import typing

import numpy as np
import pandas as pd
import sklearn.preprocessing
import tensorflow as tf

import catalog_tfm.discretize
import eq_mag_prediction.utilities.catalog_processing
import eq_mag_prediction.utilities.loading_utils
import eq_mag_prediction.utilities.ml_utils

REQUIRED_COLUMNS = ("time", "magnitude")
OPTIONAL_NUMERIC = ("latitude", "longitude", "depth")


@dataclasses.dataclass
class CatalogTrainValTest:
  """Train / val / test tensors and metadata for evaluation."""

  X_train: np.ndarray
  y_train: np.ndarray
  X_val: np.ndarray
  y_val: np.ndarray
  X_test: np.ndarray
  y_test: np.ndarray
  file_hashes: typing.Dict[str, str]
  magnitude_bin_edges: np.ndarray
  dt_bin_seconds: float


def default_ingested_dir() -> pathlib.Path:
  """Sibling ``eq_mag_prediction`` ingested path relative to cwd."""
  return (
      pathlib.Path.cwd().resolve().parent
      / "eq_mag_prediction"
      / "results"
      / "catalogs"
      / "ingested"
  )


def resolve_data_dir(path: str | pathlib.Path | None) -> pathlib.Path:
  """Return absolute :class:`Path`.

  If *path* is relative and starts with ``results/``, resolve via
  :func:`eq_mag_prediction.utilities.loading_utils.get_resource_path` (paths
  relative to the ``eq_mag_prediction`` repo checkout).
  """
  if path is None:
    return default_ingested_dir()
  p = pathlib.Path(path)
  if p.is_absolute():
    return p
  s = str(p).replace("\\", "/").lstrip("/")
  if s.startswith("results/"):
    return pathlib.Path(eq_mag_prediction.utilities.loading_utils.get_resource_path(s))
  return (pathlib.Path.cwd() / p).resolve()


def list_catalog_csvs(data_dir: pathlib.Path) -> typing.List[pathlib.Path]:
  data_dir = pathlib.Path(data_dir)
  if not data_dir.is_dir():
    raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
  paths = sorted(data_dir.glob("*.csv"))
  if not paths:
    raise FileNotFoundError(f"No CSV files under {data_dir}")
  return paths


def _filename_matches_any(name: str, substrings: typing.Sequence[str]) -> bool:
  """True if *name* contains any *substrings* (case-insensitive)."""
  lower = name.lower()
  return any(s.lower() in lower for s in substrings)


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


def windows_from_prepared(df: pd.DataFrame, seq_len: int) -> typing.Tuple[np.ndarray, np.ndarray]:
  """Build ``X`` and ``y`` with ``y[:,0]`` = next magnitude, ``y[:,1]`` = Δt (s) to that event."""
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
  x_list = [feats[i: i + seq_len] for i in range(n - seq_len)]
  X = np.stack(x_list, axis=0)
  m_next = mag[seq_len:n]
  # Inter-event time ending at the next event (gap before the predicted event).
  dt_next = t[seq_len:n] - t[seq_len - 1: n - 1]
  dt_next = np.maximum(dt_next, 0.0)
  y = np.stack([m_next, dt_next], axis=1)
  return X, y


def _split_three_chronological(
    n: int,
    train_frac: float,
    val_frac: float,
    test_frac: float,
) -> typing.Tuple[slice, slice, slice]:
  if n < 3:
    raise ValueError(
        f"Need at least 3 windows for train/val/test split (got n={n})"
    )
  total = train_frac + val_frac + test_frac
  if abs(total - 1.0) > 1e-6:
    raise ValueError(f"train/val/test fractions must sum to 1, got {total}")
  n_train = int(n * train_frac)
  n_val = int(n * val_frac)
  n_test = n - n_train - n_val
  if n_train < 1 or n_val < 1 or n_test < 1:
    raise ValueError(
        f"Chronological split produced empty part: n={n}, "
        f"n_train={n_train}, n_val={n_val}, n_test={n_test}"
    )
  return (
      slice(0, n_train),
      slice(n_train, n_train + n_val),
      slice(n_train + n_val, n),
  )


def load_catalog_train_val_test(
    data_dir: pathlib.Path,
    seq_len: int,
    *,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    magnitude_bin_edges: typing.Optional[np.ndarray] = None,
    dt_bin_seconds: float = 1200.0,
    max_rows_per_file: typing.Optional[int] = None,
    max_windows_per_catalog: typing.Optional[int] = None,
    exclude_filename_contains: typing.Optional[typing.Sequence[str]] = None,
    test_only_filename_keywords: typing.Optional[typing.Sequence[str]] = None,
) -> CatalogTrainValTest:
  """Load CSVs, build windows per catalog, split chronologically per catalog.

  Catalogs whose basename matches ``test_only_filename_keywords`` contribute
  **only** to the test set (full file, in time order). Remaining catalogs are
  split into train / val / test in chronological order (earlier windows train,
  later val, latest test).

  ``y`` is shape ``(n, 2)``: ``[:,0]`` next magnitude, ``[:,1]`` Δt in seconds
  to that event.
  """
  mag_edges = (
      np.asarray(magnitude_bin_edges, dtype=np.float64)
      if magnitude_bin_edges is not None
      else catalog_tfm.discretize.default_magnitude_bin_edges()
  )
  paths = list_catalog_csvs(data_dir)
  if exclude_filename_contains:
    kw = tuple(s for s in exclude_filename_contains if s)
    if not kw:
      raise ValueError(
          "exclude_filename_contains was non-empty but only empty strings were given"
      )
    paths = [p for p in paths if not _filename_matches_any(p.name, kw)]
    if not paths:
      raise FileNotFoundError(
          "All CSV files were excluded by exclude_filename_contains; "
          f"keywords={list(exclude_filename_contains)!r}"
      )

  test_kw = tuple(s for s in (test_only_filename_keywords or ()) if s)
  paths_main: typing.List[pathlib.Path] = []
  paths_test_only: typing.List[pathlib.Path] = []
  for p in paths:
    if test_kw and _filename_matches_any(p.name, test_kw):
      paths_test_only.append(p)
    else:
      paths_main.append(p)

  if not paths_main:
    raise FileNotFoundError(
        "No CSVs left for train/val after excluding test-only keywords; "
        "need at least one catalog not matching test_only_filename_keywords."
    )

  xs_train: typing.List[np.ndarray] = []
  ys_train: typing.List[np.ndarray] = []
  xs_val: typing.List[np.ndarray] = []
  ys_val: typing.List[np.ndarray] = []
  xs_test: typing.List[np.ndarray] = []
  ys_test: typing.List[np.ndarray] = []
  file_hashes: typing.Dict[str, str] = {}

  for path in paths_main:
    raw = (
        pd.read_csv(path, nrows=max_rows_per_file)
        if max_rows_per_file is not None
        else pd.read_csv(path)
    )
    if len(raw) == 0:
      raise ValueError(f"Empty catalog: {path}")
    prepared = _prepare_frame(raw)
    file_hashes[path.name] = eq_mag_prediction.utilities.catalog_processing.hash_pandas_object(
        prepared
    )
    wx, wy = windows_from_prepared(prepared, seq_len)
    if max_windows_per_catalog is not None:
      wx = wx[:max_windows_per_catalog]
      wy = wy[:max_windows_per_catalog]
    n = wx.shape[0]
    sl_tr, sl_va, sl_te = _split_three_chronological(
        n, train_fraction, val_fraction, test_fraction
    )
    xs_train.append(wx[sl_tr])
    ys_train.append(wy[sl_tr])
    xs_val.append(wx[sl_va])
    ys_val.append(wy[sl_va])
    xs_test.append(wx[sl_te])
    ys_test.append(wy[sl_te])

  for path in paths_test_only:
    raw = (
        pd.read_csv(path, nrows=max_rows_per_file)
        if max_rows_per_file is not None
        else pd.read_csv(path)
    )
    if len(raw) == 0:
      raise ValueError(f"Empty catalog: {path}")
    prepared = _prepare_frame(raw)
    file_hashes[path.name] = eq_mag_prediction.utilities.catalog_processing.hash_pandas_object(
        prepared
    )
    wx, wy = windows_from_prepared(prepared, seq_len)
    if max_windows_per_catalog is not None:
      wx = wx[:max_windows_per_catalog]
      wy = wy[:max_windows_per_catalog]
    xs_test.append(wx)
    ys_test.append(wy)

  X_train = np.concatenate(xs_train, axis=0)
  y_train = np.concatenate(ys_train, axis=0)
  X_val = np.concatenate(xs_val, axis=0)
  y_val = np.concatenate(ys_val, axis=0)
  X_test = np.concatenate(xs_test, axis=0)
  y_test = np.concatenate(ys_test, axis=0)

  return CatalogTrainValTest(
      X_train=X_train,
      y_train=y_train,
      X_val=X_val,
      y_val=y_val,
      X_test=X_test,
      y_test=y_test,
      file_hashes=file_hashes,
      magnitude_bin_edges=mag_edges,
      dt_bin_seconds=float(dt_bin_seconds),
  )


def fit_scaler(X_train: np.ndarray) -> sklearn.preprocessing.StandardScaler:
  scaler = sklearn.preprocessing.StandardScaler()
  scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
  return scaler


def transform_X(X: np.ndarray, scaler: sklearn.preprocessing.StandardScaler) -> np.ndarray:
  shape = X.shape
  flat = X.reshape(-1, shape[-1])
  out = scaler.transform(flat)
  return out.reshape(shape)


def fit_y_scaler(y_train: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
  """Per-dimension mean and std for ``y`` (shape ``(n, 2)``)."""
  mean = y_train.mean(axis=0)
  std = y_train.std(axis=0)
  std = np.where(std < 1e-8, 1.0, std)
  return mean, std


def normalize_y(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
  return (y - mean) / std


def denormalize_y(y_n: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
  return y_n * std + mean


def create_tf_dataset(
    csv_file,
    seq_length=100,
    batch_size=32,
    max_sequences=None,
    shuffle_buffer=10000,
    train_fraction=0.8,
):
  """Train/test sliding-window data. The scaler is fit on training pairs only."""
  if not os.path.exists(csv_file):
    return None

  df = pd.read_csv(csv_file)
  df = df.sort_values(by="time").reset_index(drop=True)
  n_ev = len(df)
  if n_ev < seq_length + 3:
    return None

  time = df["time"].values.astype(np.float64)
  magnitude = df["magnitude"].values.astype(np.float64)
  n_pair = n_ev - 1
  dt = np.diff(time)
  mag = magnitude[1:]
  raw = np.stack([dt, mag], axis=-1)

  n_train_ev = max(2, min(int(np.floor(train_fraction * n_ev)), n_ev - 1))
  n_train_pair = n_train_ev - 1
  if n_train_pair < seq_length + 1:
    return None

  scaler = eq_mag_prediction.utilities.ml_utils.StandardScaler(feature_axes=0)
  scaler.fit(raw[:n_train_pair])
  data = np.asarray(scaler.transform(raw), dtype=np.float32)

  i_train_max = n_train_pair - seq_length
  i_test0 = n_train_pair - seq_length + 1
  if i_test0 < n_pair - seq_length and i_test0 >= 0:
    test_window_starts = np.arange(
        i_test0, n_pair - seq_length, dtype=np.int32
    )
  else:
    test_window_starts = np.array([], dtype=np.int32)

  def build_xy(indices):
    X, y = [], []
    for i in indices:
      i = int(i)
      X.append(data[i: i + seq_length])
      y.append(data[i + 1: i + seq_length + 1])
    if not X:
      return None, None
    return (np.array(X, dtype=np.float32), np.array(y, dtype=np.float32))

  train_indices = list(range(0, i_train_max + 1))
  if max_sequences is not None:
    train_indices = train_indices[: max_sequences]
  Xtr, ytr = build_xy(train_indices)
  if Xtr is None:
    return None

  test_is = [int(t) for t in test_window_starts]
  if max_sequences is not None:
    cap = max(64, max_sequences // 4)
    test_is = test_is[: min(len(test_is), cap)]
  Xte, yte = build_xy(test_is) if test_is else (None, None)

  buf = min(shuffle_buffer, len(Xtr)) if len(Xtr) > 0 else 1
  train_dataset = (
      tf.data.Dataset.from_tensor_slices((Xtr, ytr))
      .shuffle(buf)
      .batch(batch_size)
      .prefetch(tf.data.AUTOTUNE)
  )
  if Xte is not None and len(Xte) > 0:
    bsz = min(int(batch_size), int(len(Xte)))
    bsz = max(1, bsz)
    test_dataset = (
        tf.data.Dataset.from_tensor_slices((Xte, yte))
        .batch(bsz)
        .prefetch(tf.data.AUTOTUNE)
    )
  else:
    test_dataset = None
    test_is = []

  return {
      "train_dataset": train_dataset,
      "test_dataset": test_dataset,
      "scaler": scaler,
      "df": df,
      "n_train_events": n_train_ev,
      "time": time,
      "test_window_starts": np.array(test_is, dtype=np.int32)
      if test_is
      else np.array([], dtype=np.int32),
      "num_train_sequences": int(len(Xtr)),
  }

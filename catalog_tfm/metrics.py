"""Report MAE/MSE on continuous targets and accuracy on discretized bins."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

import catalog_tfm.discretize


def evaluate_regression(
  y_true: np.ndarray,
  y_pred: np.ndarray,
) -> Dict[str, float]:
  """MAE / MSE for magnitude (column 0) and Δt seconds (column 1)."""
  y_true = np.asarray(y_true, dtype=np.float64)
  y_pred = np.asarray(y_pred, dtype=np.float64)
  e_m = y_true[:, 0] - y_pred[:, 0]
  e_t = y_true[:, 1] - y_pred[:, 1]
  return {
    "mae_magnitude": float(np.mean(np.abs(e_m))),
    "mse_magnitude": float(np.mean(e_m**2)),
    "mae_dt_seconds": float(np.mean(np.abs(e_t))),
    "mse_dt_seconds": float(np.mean(e_t**2)),
  }


def evaluate_discretized(
  y_true: np.ndarray,
  y_pred: np.ndarray,
  magnitude_bin_edges: np.ndarray,
  dt_bin_seconds: float,
  *,
  max_dt_bin_index: int,
) -> Dict[str, float]:
  """Bin-wise accuracy: compare discretized true vs pred (from continuous values)."""
  y_true = np.asarray(y_true, dtype=np.float64)
  y_pred = np.asarray(y_pred, dtype=np.float64)
  tm = catalog_tfm.discretize.magnitude_to_bin(y_true[:, 0], magnitude_bin_edges)
  pm = catalog_tfm.discretize.magnitude_to_bin(y_pred[:, 0], magnitude_bin_edges)
  tt = catalog_tfm.discretize.dt_seconds_to_bin(
    y_true[:, 1], dt_bin_seconds, max_bin_index=max_dt_bin_index
  )
  pt = catalog_tfm.discretize.dt_seconds_to_bin(
    y_pred[:, 1], dt_bin_seconds, max_bin_index=max_dt_bin_index
  )
  n = tm.shape[0]
  acc_m = float(np.mean(tm == pm))
  acc_dt = float(np.mean(tt == pt))
  acc_both = float(np.mean((tm == pm) & (tt == pt)))
  return {
    "acc_magnitude_bin": acc_m,
    "acc_dt_bin": acc_dt,
    "acc_joint_bin": acc_both,
    "n": float(n),
  }


def format_metrics_line(prefix: str, reg: Dict[str, Any], disc: Dict[str, Any]) -> str:
  return (
    f"{prefix}  continuous: MAE_mag={reg['mae_magnitude']:.6g} MAE_dt_s={reg['mae_dt_seconds']:.6g} "
    f"MSE_mag={reg['mse_magnitude']:.6g} MSE_dt={reg['mse_dt_seconds']:.6g} | "
    f"discrete: acc_mag_bin={disc['acc_magnitude_bin']:.6g} acc_dt_bin={disc['acc_dt_bin']:.6g} "
    f"acc_joint={disc['acc_joint_bin']:.6g}"
  )

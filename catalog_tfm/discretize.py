"""Discretize magnitude and inter-event time for metrics and optional reporting."""

from __future__ import annotations

import typing

import numpy as np
import pandas


def default_magnitude_bin_edges() -> np.ndarray:
  """Default magnitude histogram edges (bin i is ``[edges[i], edges[i+1])``)."""
  return np.arange(-10.0, 11.0, 0.02)


def magnitude_to_bin(
    m: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
  """Map magnitudes to bin indices in ``[0, len(edges) - 2]``."""
  m = np.asarray(m, dtype=np.float64)
  edges = np.asarray(edges, dtype=np.float64)
  if edges.ndim != 1 or len(edges) < 2:
    raise ValueError("edges must be 1-D with at least 2 values")
  # searchsorted: index of right edge; subtract 1 for left-inclusive bin
  idx = np.searchsorted(edges, m, side="right") - 1
  idx = np.clip(idx, 0, len(edges) - 2)
  return idx.astype(np.int64)


def dt_seconds_to_bin(
    dt_seconds: np.ndarray,
    bin_width_seconds: float,
    *,
    max_bin_index: typing.Optional[int] = None,
) -> np.ndarray:
  """Map ``dt >= 0`` to bin index ``floor(dt / bin_width)``, optionally clipped."""
  if bin_width_seconds <= 0:
    raise ValueError("bin_width_seconds must be positive")
  dt_seconds = np.asarray(dt_seconds, dtype=np.float64)
  idx = np.floor(dt_seconds / bin_width_seconds).astype(np.int64)
  idx = np.maximum(idx, 0)
  if max_bin_index is not None:
    idx = np.minimum(idx, max_bin_index)
  return idx


def discretize_catalog(
    catalog: pandas.DataFrame,
    magnitude_bin_edges: np.ndarray,
    dt_bin_seconds: float,
) -> pandas.DataFrame:
  """Return a copy of the catalog with magnitude and dt bins."""
  return catalog.copy().assign({
      "magnitude_bin": magnitude_to_bin(catalog["magnitude"], magnitude_bin_edges),
      "dt_bin": dt_seconds_to_bin(catalog["time_delta"], dt_bin_seconds),
  })

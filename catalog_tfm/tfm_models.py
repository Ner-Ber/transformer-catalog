# coding=utf-8
"""Factory helpers for MTP (marked temporal point process) transformer architectures.

Layer definitions live in :mod:`catalog_tfm.model_continous`. Training is run from
:file:`scripts/tfm_trainer.py`.
"""

from __future__ import annotations

import catalog_tfm.model_continous


def build_mtp_transformer(
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    num_mixtures: int = 5,
    *,
    name: str | None = None,
) -> catalog_tfm.model_continous.MTPPTransformer:
  """Return a configured MTP transformer (MDN heads over the transformer trunk).

  This mirrors the role of :func:`eq_mag_prediction.forecasting.head_models.magnitude_prediction_model`
  as a single entry point for constructing the forecast head + backbone assembly.
  """
  kwargs: dict = {}
  if name is not None:
    kwargs["name"] = name
  return catalog_tfm.model_continous.MTPPTransformer(
      d_model=d_model,
      nhead=nhead,
      num_layers=num_layers,
      num_mixtures=num_mixtures,
      **kwargs,
  )


MTPPTransformer = catalog_tfm.model_continous.MTPPTransformer
ContinuousTimeEncoding = catalog_tfm.model_continous.ContinuousTimeEncoding
TransformerBlock = catalog_tfm.model_continous.TransformerBlock

__all__ = [
    "build_mtp_transformer",
    "ContinuousTimeEncoding",
    "MTPPTransformer",
    "TransformerBlock",
]

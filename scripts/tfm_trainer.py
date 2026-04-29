#!/usr/bin/env python3
# coding=utf-8
"""Train the MTP transformer on a catalog CSV.

GPU setup, distribution strategy, gradient loop, and CLI live in this module (analogous to
``eq_mag_prediction`` ``magnitude_predictor_trainer.py`` wiring ``head_models`` + ``fit``).
Model factories: :mod:`catalog_tfm.tfm_models`; layers: :mod:`catalog_tfm.model_continous`.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import sys
import typing

import tensorflow as tf

# Repo root on ``python scripts/tfm_trainer.py`` without editable install.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import catalog_tfm.data
import catalog_tfm.metrics
import catalog_tfm.model_continous
import catalog_tfm.tfm_models


def configure_tensorflow_gpus() -> list[tf.config.PhysicalDevice]:
  """Enable per-GPU memory growth and return physical GPU devices."""
  gpus = tf.config.list_physical_devices("GPU")
  for gpu in gpus:
    try:
      tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
      pass
  return gpus


def make_distribution_strategy(
    batch_size: int,
    *,
    verbose: bool = True,
) -> tuple[tf.distribute.Strategy, list[tf.config.PhysicalDevice]]:
  """Return a strategy suitable for the custom ``strategy.run`` training loop."""
  gpus = configure_tensorflow_gpus()
  if not gpus:
    if verbose:
      print("TensorFlow: no GPU visible; training on CPU.")
    return tf.distribute.get_strategy(), []

  if len(gpus) == 1:
    if verbose:
      print(f"TensorFlow: using GPU {gpus[0].name} (with memory_growth).")
    strategy = tf.distribute.MirroredStrategy()
    if verbose:
      print(
          f"Using MirroredStrategy with {strategy.num_replicas_in_sync} replica(s) "
          f"(global batch size = {batch_size})."
      )
    return strategy, gpus

  if verbose:
    print(f"TensorFlow: {len(gpus)} GPUs visible: {[g.name for g in gpus]!r}")
  logical_gpus = tf.config.list_logical_devices("GPU")
  dev = logical_gpus[0].name if logical_gpus else gpus[0].name
  strategy = tf.distribute.OneDeviceStrategy(device=dev)
  if verbose:
    print(
        f"Using OneDeviceStrategy on {dev} "
        f"(multi-GPU MirroredStrategy disabled for this custom training loop). "
        f"Global batch size = {batch_size}."
    )
  return strategy, gpus


def build_train_step(
    model: catalog_tfm.model_continous.MTPPTransformer,
    optimizer: tf.keras.optimizers.Optimizer,
    loss_fn: typing.Callable[..., tf.Tensor] = catalog_tfm.metrics.mdn_nll_bivariate,
) -> typing.Callable[..., tf.Tensor]:
  """Return a per-replica train step: ``(batch_x, batch_y) -> loss``."""

  def train_step(batch_x, batch_y):
    with tf.GradientTape() as tape:
      pi, mu, sigma = model(batch_x, training=True)
      loss = loss_fn(pi, mu, sigma, batch_y)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

  return train_step


def compile_strategy_train_step(
    train_step: typing.Callable[..., tf.Tensor],
    strategy: tf.distribute.Strategy,
    *,
    use_multi_gpu_run: bool,
) -> typing.Callable[..., tf.Tensor]:
  """Wrap ``train_step`` with ``tf.function`` and optional ``strategy.run``."""
  if use_multi_gpu_run:

    @tf.function
    def run_train_step(batch_x, batch_y):
      per_replica = strategy.run(train_step, args=(batch_x, batch_y))
      return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica, axis=None)

    return run_train_step

  return tf.function(train_step)


def run_epoch_loop(
    run_train_step: typing.Callable[..., tf.Tensor],
    train_dataset: tf.data.Dataset,
    epochs: int,
    *,
    max_steps_per_epoch: typing.Optional[int] = None,
    verbose: bool = True,
) -> list[float]:
  """Iterate epochs and batches; return mean loss per epoch."""
  history: list[float] = []
  for epoch in range(epochs):
    total_loss = 0.0
    steps = 0
    for batch_x, batch_y in train_dataset:
      loss = run_train_step(batch_x, batch_y)
      total_loss += loss
      steps += 1
      if max_steps_per_epoch is not None and steps >= max_steps_per_epoch:
        break
    avg = float(total_loss / steps) if steps > 0 else 0.0
    history.append(avg)
    if verbose:
      print(f"Epoch {epoch + 1}/{epochs} | Average NLL Loss: {avg:.4f} (steps={steps})")
  return history


def train_mtp_from_data_pack(
    data_pack: typing.Optional[dict[str, typing.Any]],
    *,
    seq_length: int,
    batch_size: int,
    epochs: int,
    d_model: int,
    nhead: int,
    num_layers: int,
    num_mixtures: int,
    learning_rate: float = 1e-3,
    max_steps_per_epoch: typing.Optional[int] = None,
    catalog_path: typing.Optional[str | os.PathLike[str]] = None,
    run_label: str = "full",
    verbose: bool = True,
    model_builder: typing.Callable[..., catalog_tfm.model_continous.MTPPTransformer] = (
        catalog_tfm.tfm_models.build_mtp_transformer
    ),
    loss_fn: typing.Callable[..., tf.Tensor] = catalog_tfm.metrics.mdn_nll_bivariate,
) -> dict[str, typing.Any]:
  """Build model under strategy scope and run :func:`run_epoch_loop`."""
  empty: dict[str, typing.Any] = {
      "training_completed": False,
      "data_pack": None,
      "dataset": None,
      "scaler": None,
      "model": None,
      "epoch_loss_history": [],
      "gpus": [],
      "strategy": None,
      "num_train_sequences": 0,
  }

  if data_pack is None:
    if verbose and catalog_path is not None:
      print(f"Catalog not found or too small: {catalog_path}")
    return empty

  train_dataset = data_pack["train_dataset"]
  dataset_for_plots = train_dataset
  scaler = data_pack["scaler"]
  num_sequences = int(data_pack["num_train_sequences"])

  if verbose:
    print(f"[{run_label}] Sequences in this run: {num_sequences} | seq_length={seq_length}")

  strategy, gpus = make_distribution_strategy(batch_size, verbose=verbose)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)

  with strategy.scope():
    model = model_builder(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        num_mixtures=num_mixtures,
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

  train_step = build_train_step(model, optimizer, loss_fn=loss_fn)
  use_multi = bool(gpus)
  run_train_step = compile_strategy_train_step(
      train_step, strategy, use_multi_gpu_run=use_multi
  )

  if verbose:
    print("Starting training...")
  epoch_loss_history = run_epoch_loop(
      run_train_step,
      train_dataset,
      epochs,
      max_steps_per_epoch=max_steps_per_epoch,
      verbose=verbose,
  )

  return {
      "training_completed": True,
      "data_pack": data_pack,
      "dataset": dataset_for_plots,
      "scaler": scaler,
      "model": model,
      "epoch_loss_history": epoch_loss_history,
      "gpus": gpus,
      "strategy": strategy,
      "num_train_sequences": num_sequences,
  }


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(
      description=(
          "Train MTP (mixture-density) transformer on a time-sorted catalog CSV "
          "(columns include time, magnitude)."
      ),
  )
  p.add_argument(
      "--catalog-path",
      type=pathlib.Path,
      required=True,
      help="Path to ingested catalog CSV.",
  )
  p.add_argument("--seq-length", type=int, default=50)
  p.add_argument("--batch-size", type=int, default=32)
  p.add_argument("--epochs", type=int, default=10)
  p.add_argument("--learning-rate", type=float, default=1e-3)
  p.add_argument("--d-model", type=int, default=64)
  p.add_argument("--nhead", type=int, default=4)
  p.add_argument("--num-layers", type=int, default=2)
  p.add_argument("--num-mixtures", type=int, default=5)
  p.add_argument("--train-fraction", type=float, default=0.8)
  p.add_argument("--shuffle-buffer", type=int, default=10000)
  p.add_argument("--max-sequences", type=int, default=None)
  p.add_argument("--max-steps-per-epoch", type=int, default=None)
  p.add_argument("--run-label", type=str, default="cli")
  p.add_argument("--quiet", action="store_true", help="Reduce console output from the training loop.")
  return p.parse_args()


def main() -> None:
  logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
  args = parse_args()
  catalog_path = args.catalog_path.expanduser().resolve()
  if not catalog_path.is_file():
    logging.error("Catalog not found: %s", catalog_path)
    sys.exit(1)

  logging.info("Building dataset: %s", catalog_path)
  data_pack = catalog_tfm.data.create_tf_dataset(
      str(catalog_path),
      seq_length=args.seq_length,
      batch_size=args.batch_size,
      max_sequences=args.max_sequences,
      shuffle_buffer=args.shuffle_buffer,
      train_fraction=args.train_fraction,
  )

  result = train_mtp_from_data_pack(
      data_pack,
      seq_length=args.seq_length,
      batch_size=args.batch_size,
      epochs=args.epochs,
      d_model=args.d_model,
      nhead=args.nhead,
      num_layers=args.num_layers,
      num_mixtures=args.num_mixtures,
      learning_rate=args.learning_rate,
      max_steps_per_epoch=args.max_steps_per_epoch,
      catalog_path=str(catalog_path),
      run_label=args.run_label,
      verbose=not args.quiet,
  )

  if not result["training_completed"]:
    sys.exit(2)

  hist = result["epoch_loss_history"]
  logging.info(
      "Training finished; epochs=%s final_mean_nll=%s",
      len(hist),
      hist[-1] if hist else None,
  )


if __name__ == "__main__":
  main()

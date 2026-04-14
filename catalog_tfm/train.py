"""CLI: train the shallow transformer on ingested catalogs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from catalog_tfm import data
from catalog_tfm.model import build_model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Train shallow transformer on ingested catalogs.")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Directory of ingested *.csv (default: ../eq_mag_prediction/results/catalogs/ingested from cwd)",
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = data.resolve_data_dir(args.data_dir)
    logging.info("Resolved data_dir: %s", data_dir)

    X, y, hashes = data.load_all_windows(data_dir, args.seq_len)
    logging.info("Windows: X=%s y=%s", X.shape, y.shape)
    for name in sorted(hashes):
        logging.info("catalog_sha256[%s]=%s", name, hashes[name])

    rng = np.random.RandomState(args.seed)
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=args.val_fraction,
        random_state=rng,
    )
    scaler = data.fit_scaler(X_train)
    X_train_t = data.transform_X(X_train, scaler)
    X_val_t = data.transform_X(X_val, scaler)

    tf.random.set_seed(args.seed)
    model = build_model(
        args.seq_len,
        int(X.shape[-1]),
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    model.fit(
        X_train_t,
        y_train,
        validation_data=(X_val_t, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )


if __name__ == "__main__":
    main()

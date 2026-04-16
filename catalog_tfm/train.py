"""CLI: train the shallow transformer on ingested catalogs."""

from __future__ import annotations

import argparse
import logging

import numpy as np
import tensorflow as tf

import catalog_tfm.data
import catalog_tfm.metrics
import catalog_tfm.model


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
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--mag-min", type=float, default=-10.0)
    parser.add_argument("--mag-max", type=float, default=11.0)
    parser.add_argument("--mag-step", type=float, default=0.02)
    parser.add_argument("--dt-bin-seconds", type=float, default=1200.0)
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--max-windows-per-catalog", type=int, default=None)
    parser.add_argument(
        "--exclude-filename-keyword",
        action="append",
        default=None,
        metavar="SUBSTR",
        help="Exclude CSVs whose basename contains this substring (case-insensitive).",
    )
    parser.add_argument(
        "--test-only-keyword",
        action="append",
        default=None,
        metavar="SUBSTR",
        help="CSV basename matching any keyword is used for test only (chronological holdout).",
    )
    args = parser.parse_args()

    magnitude_bin_edges = np.arange(args.mag_min, args.mag_max, args.mag_step, dtype=np.float64)
    if magnitude_bin_edges.size < 2:
        raise ValueError("magnitude bin edges must have at least 2 points; check mag-min/max/step")

    data_dir = catalog_tfm.data.resolve_data_dir(args.data_dir)
    logging.info("Resolved data_dir: %s", data_dir)

    bundle = catalog_tfm.data.load_catalog_train_val_test(
        data_dir,
        args.seq_len,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        magnitude_bin_edges=magnitude_bin_edges,
        dt_bin_seconds=args.dt_bin_seconds,
        max_rows_per_file=args.max_rows_per_file,
        max_windows_per_catalog=args.max_windows_per_catalog,
        exclude_filename_contains=args.exclude_filename_keyword,
        test_only_filename_keywords=args.test_only_keyword,
    )
    logging.info(
        "Shapes: train %s %s | val %s %s | test %s %s",
        bundle.X_train.shape,
        bundle.y_train.shape,
        bundle.X_val.shape,
        bundle.y_val.shape,
        bundle.X_test.shape,
        bundle.y_test.shape,
    )
    for name in sorted(bundle.file_hashes):
        logging.info("catalog_sha256[%s]=%s", name, bundle.file_hashes[name])

    y_mean, y_std = catalog_tfm.data.fit_y_scaler(bundle.y_train)
    y_train_n = catalog_tfm.data.normalize_y(bundle.y_train, y_mean, y_std)
    y_val_n = catalog_tfm.data.normalize_y(bundle.y_val, y_mean, y_std)

    scaler_x = catalog_tfm.data.fit_scaler(bundle.X_train)
    X_train_t = catalog_tfm.data.transform_X(bundle.X_train, scaler_x)
    X_val_t = catalog_tfm.data.transform_X(bundle.X_val, scaler_x)
    X_test_t = catalog_tfm.data.transform_X(bundle.X_test, scaler_x)

    tf.random.set_seed(args.seed)
    model = catalog_tfm.model.build_model(
        args.seq_len,
        int(bundle.X_train.shape[-1]),
        d_model=args.d_model,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    model.fit(
        X_train_t,
        y_train_n,
        validation_data=(X_val_t, y_val_n),
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
    )

    max_dt = float(
        np.max(
            [
                bundle.y_train[:, 1].max(),
                bundle.y_val[:, 1].max(),
                bundle.y_test[:, 1].max(),
            ]
        )
    )
    max_dt_bin_index = int(np.ceil(max_dt / bundle.dt_bin_seconds)) + 100

    y_val_pred = catalog_tfm.data.denormalize_y(model.predict(X_val_t, verbose=0), y_mean, y_std)
    y_test_pred = catalog_tfm.data.denormalize_y(model.predict(X_test_t, verbose=0), y_mean, y_std)

    reg_v = catalog_tfm.metrics.evaluate_regression(bundle.y_val, y_val_pred)
    disc_v = catalog_tfm.metrics.evaluate_discretized(
        bundle.y_val,
        y_val_pred,
        bundle.magnitude_bin_edges,
        bundle.dt_bin_seconds,
        max_dt_bin_index=max_dt_bin_index,
    )
    reg_te = catalog_tfm.metrics.evaluate_regression(bundle.y_test, y_test_pred)
    disc_te = catalog_tfm.metrics.evaluate_discretized(
        bundle.y_test,
        y_test_pred,
        bundle.magnitude_bin_edges,
        bundle.dt_bin_seconds,
        max_dt_bin_index=max_dt_bin_index,
    )
    logging.info(catalog_tfm.metrics.format_metrics_line("validation", reg_v, disc_v))
    logging.info(catalog_tfm.metrics.format_metrics_line("test", reg_te, disc_te))


if __name__ == "__main__":
    main()

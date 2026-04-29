"""Small Keras transformer encoder for sequence regression (magnitude + Δt)."""

from __future__ import annotations

import tensorflow as tf


def build_model(
    seq_len: int,
    feat_dim: int,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> tf.keras.Model:
  """Predict next-event magnitude and inter-event time (2 outputs)."""
  if d_model % num_heads != 0:
    raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
  key_dim = d_model // num_heads
  inputs = tf.keras.Input(shape=(seq_len, feat_dim))
  x = tf.keras.layers.Dense(d_model)(inputs)
  for _ in range(num_layers):
    attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
    )(x, x)
    attn = tf.keras.layers.Dropout(dropout)(attn)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x)
    ffn = tf.keras.layers.Dense(d_model)(ffn)
    ffn = tf.keras.layers.Dropout(dropout)(ffn)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn)
  x = tf.keras.layers.GlobalAveragePooling1D()(x)
  outputs = tf.keras.layers.Dense(2)(x)
  return tf.keras.Model(inputs, outputs)

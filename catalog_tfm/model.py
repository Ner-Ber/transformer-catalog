"""Small Keras transformer encoder for sequence regression."""

from __future__ import annotations

from tensorflow import keras
from tensorflow.keras import layers


def build_model(
    seq_len: int,
    feat_dim: int,
    d_model: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    num_layers: int = 2,
    dropout: float = 0.1,
) -> keras.Model:
    """Next-magnitude regression: single linear output."""
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    key_dim = d_model // num_heads
    inputs = keras.Input(shape=(seq_len, feat_dim))
    x = layers.Dense(d_model)(inputs)
    for _ in range(num_layers):
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
        )(x, x)
        attn = layers.Dropout(dropout)(attn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ffn = layers.Dense(ff_dim, activation="relu")(x)
        ffn = layers.Dense(d_model)(ffn)
        ffn = layers.Dropout(dropout)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1)(x)
    return keras.Model(inputs, outputs)

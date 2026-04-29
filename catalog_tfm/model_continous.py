import tensorflow as tf


class ContinuousTimeEncoding(tf.keras.layers.Layer):
  """Continuous time encoding layer."""

  def __init__(self, d_model, **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=(1, self.d_model),
        initializer="glorot_uniform",
        trainable=True,
        name="time_freqs",
    )

  def call(self, dt):
    time_enc = dt * self.w
    return tf.sin(time_enc)


class TransformerBlock(tf.keras.layers.Layer):
  """Transformer block layer."""

  def __init__(self, d_model, nhead, **kwargs):
    super().__init__(**kwargs)
    self.att = tf.keras.layers.MultiHeadAttention(num_heads=nhead, key_dim=d_model)
    self.ffn = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(d_model * 4, activation="relu"),
            tf.keras.layers.Dense(d_model),
        ]
    )
    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, training=False, attention_mask=None):
    attn_output = self.att(inputs, inputs, attention_mask=attention_mask)
    out1 = self.layernorm1(inputs + attn_output)
    ffn_output = self.ffn(out1)
    return self.layernorm2(out1 + ffn_output)


class MTPPTransformer(tf.keras.Model):
  """Mixture Density Network Transformer model."""

  def __init__(self, d_model=64, nhead=4, num_layers=2, num_mixtures=5, **kwargs):
    super().__init__(**kwargs)
    self.d_model = d_model
    self.num_mixtures = num_mixtures

    self.input_proj = tf.keras.layers.Dense(d_model)
    self.time_encoding = ContinuousTimeEncoding(d_model)
    self.transformer_blocks = [
        TransformerBlock(d_model, nhead) for _ in range(num_layers)
    ]

    self.mdn_pi = tf.keras.layers.Dense(num_mixtures * 2)
    self.mdn_mu = tf.keras.layers.Dense(num_mixtures * 2)
    self.mdn_sigma = tf.keras.layers.Dense(num_mixtures * 2)

  def call(self, x, training=False):
    batch_size = tf.shape(x)[0]
    seq_length = tf.shape(x)[1]

    dt = tf.expand_dims(x[:, :, 0], -1)
    h = self.input_proj(x) + self.time_encoding(dt)
    attn_mask = tf.linalg.band_part(
        tf.ones((seq_length, seq_length), dtype=tf.bool), -1, 0
    )
    attn_mask = tf.reshape(attn_mask, [1, seq_length, seq_length])
    attn_mask = tf.broadcast_to(attn_mask, [batch_size, seq_length, seq_length])

    for block in self.transformer_blocks:
      h = block(h, training=training, attention_mask=attn_mask)

    pi_logits = tf.reshape(self.mdn_pi(h), [batch_size, seq_length, 2, self.num_mixtures])
    pi = tf.nn.softmax(pi_logits, axis=-1)

    mu = tf.reshape(self.mdn_mu(h), [batch_size, seq_length, 2, self.num_mixtures])

    sigma_logits = tf.reshape(self.mdn_sigma(h), [batch_size, seq_length, 2, self.num_mixtures])
    # Floor avoids TFP Normal(scale≈0) -> NaN log_prob and broken autoregressive sampling.
    sigma = tf.exp(tf.clip_by_value(sigma_logits, -12.0, 8.0)) + 1e-6

    return pi, mu, sigma

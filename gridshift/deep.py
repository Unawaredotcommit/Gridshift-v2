"""CNN-LSTM-Attention 24h forecaster (TensorFlow/Keras).  Optional: needs `pip install tensorflow`.

Reuses the architecture of `build_attention_cnn_lstm_model` from Notebook/updated_EROCT.ipynb
(causal Conv1D x2 + BN + MaxPool + SpatialDropout -> BiLSTM -> LSTM -> MultiHeadAttention + residual
+ LayerNorm -> GAP -> Dropout -> Dense).  Changes vs the notebook:
  * output is Dense(24): a direct 24-hour forecast instead of 1-step-ahead
  * input is a 48-step window: 24h history (with ERCOT) + 24h future steps (ERCOT masked, weather +
    calendar known) -> the model can react to the weather forecast / scenario
  * scalers are fit on the TRAIN split only (notebook fit MinMax on all data = leakage)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .features import SEQ_FEATURES, SEQ_SCALED

SCALE_IDX = [SEQ_FEATURES.index(c) for c in SEQ_SCALED]


class SeqScaler:
    """Per-column standardiser for the SEQ_SCALED columns; load target uses the same load stats."""

    def fit(self, X_train: np.ndarray) -> "SeqScaler":
        flat = X_train.reshape(-1, X_train.shape[-1])
        # Load column: statistics only from history steps (future steps are masked zeros)
        hist = X_train[:, : X_train.shape[1] // 2, :].reshape(-1, X_train.shape[-1])
        self.mean_, self.std_ = np.zeros(len(SEQ_FEATURES)), np.ones(len(SEQ_FEATURES))
        for i in SCALE_IDX:
            src = hist if SEQ_FEATURES[i] == "load" else flat
            self.mean_[i], self.std_[i] = src[:, i].mean(), max(src[:, i].std(), 1e-6)
        self.load_mean_, self.load_std_ = self.mean_[0], self.std_[0]
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        Z = X.copy()
        half = X.shape[1] // 2
        for i in SCALE_IDX:
            Z[:, :, i] = (X[:, :, i] - self.mean_[i]) / self.std_[i]
        Z[:, half:, 0] = 0.0  # keep masked future load exactly 0
        return Z.astype(np.float32)

    def transform_y(self, y):  return (y - self.load_mean_) / self.load_std_
    def inverse_y(self, y):    return y * self.load_std_ + self.load_mean_


def build_model(input_shape, horizon: int = C.HORIZON, learning_rate: float = C.DEEP_LR, dropout: float = 0.30):
    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import (Add, BatchNormalization, Bidirectional, Conv1D, Dense, Dropout,
                                         GlobalAveragePooling1D, Input, LayerNormalization, LSTM,
                                         MaxPooling1D, MultiHeadAttention, SpatialDropout1D)
    from tensorflow.keras.models import Model

    l2 = regularizers.l2(1e-4)
    inp = Input(shape=input_shape)
    x = Conv1D(64, 3, activation="relu", padding="causal", kernel_regularizer=l2)(inp)
    x = BatchNormalization()(x)
    x = Conv1D(64, 3, activation="relu", padding="causal", kernel_regularizer=l2)(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(128, return_sequences=True, activation="tanh", kernel_regularizer=l2))(x)
    x = LSTM(64, return_sequences=True, activation="tanh", kernel_regularizer=l2)(x)
    attn = MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    out = Dense(horizon)(x)
    model = Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="mse", metrics=["mae"])
    return model


def train(X_tr, y_tr, X_va, y_va, models_dir: Path = C.MODELS_DIR):
    """X/y are UNSCALED arrays from features.build_sequences. Returns (model, scaler, history)."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(C.SEED)
    sc = SeqScaler().fit(X_tr)
    model = build_model(X_tr.shape[1:])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=C.DEEP_PATIENCE, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
    hist = model.fit(sc.transform_X(X_tr), sc.transform_y(y_tr),
                     validation_data=(sc.transform_X(X_va), sc.transform_y(y_va)),
                     epochs=C.DEEP_EPOCHS, batch_size=C.DEEP_BATCH, callbacks=cb, verbose=2)
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir / "cnn_lstm_attn.keras")
    joblib.dump(sc, models_dir / "cnn_lstm_attn_scaler.joblib")
    return model, sc, hist


def load(models_dir: Path = C.MODELS_DIR):
    import tensorflow as tf
    return (tf.keras.models.load_model(models_dir / "cnn_lstm_attn.keras", compile=False),
            joblib.load(models_dir / "cnn_lstm_attn_scaler.joblib"))


def predict(model, sc: SeqScaler, X: np.ndarray) -> np.ndarray:
    """UNSCALED X (n, 48, F) -> (n, 24) MW."""
    return sc.inverse_y(model.predict(sc.transform_X(X), verbose=0))

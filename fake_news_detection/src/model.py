"""LSTM classifier + evaluation metrics (same four metrics as the base paper)."""
import numpy as np


def build_lstm(timesteps, units=64, dropout=0.2, lr=1e-3, steps=10):
    import tensorflow as tf
    if timesteps % steps:
        steps = 1
    m = tf.keras.Sequential([
        tf.keras.layers.Input((timesteps, 1)),
        tf.keras.layers.Reshape((steps, timesteps // steps)),
        tf.keras.layers.LSTM(int(units)),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="binary_crossentropy", metrics=["accuracy"])
    return m


def metrics(y_true, y_prob, thr=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob).ravel() >= thr).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    d = lambda a, b: a / b if b else 0.0
    return {"accuracy": d(tp + tn, tp + tn + fp + fn), "precision": d(tp, tp + fp),
            "recall": d(tp, tp + fn), "specificity": d(tn, tn + fp),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}

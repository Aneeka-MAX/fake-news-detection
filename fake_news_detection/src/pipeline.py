"""Inference pipeline: raw text -> clean -> TF-IDF -> PPCA -> LSTM -> P(real), P(fake)."""
import json
import os

import joblib
import numpy as np

from .ppca import PPCA
from .preprocess import clean_text


class FakeNewsPipeline:
    def __init__(self, tfidf, ppca, model):
        self.tfidf, self.ppca, self.model = tfidf, ppca, model

    @classmethod
    def load(cls, models_dir, name):
        import tensorflow as tf
        tfidf = joblib.load(os.path.join(models_dir, "tfidf.joblib"))
        ppca = PPCA.load(os.path.join(models_dir, "ppca.npz"))
        model = tf.keras.models.load_model(os.path.join(models_dir, f"lstm_{name}.keras"))
        return cls(tfidf, ppca, model)

    def predict_proba(self, texts):
        """Returns array (n, 2): column 0 = Real, column 1 = Fake (LIME-compatible)."""
        if isinstance(texts, str):
            texts = [texts]
        cleaned = [clean_text(t) for t in texts]
        Z = self.ppca.transform(self.tfidf.transform(cleaned).toarray())
        p_fake = self.model.predict(Z[..., None], verbose=0).ravel()
        return np.column_stack([1 - p_fake, p_fake])


def load_results(models_dir):
    path = os.path.join(models_dir, "results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

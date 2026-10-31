"""Train the full pipeline and benchmark optimisers.

Examples
  python -m src.train --demo                               # 1-minute smoke test
  python -m src.train --dataset isot --data_dir data       # ISOT (Fake.csv + True.csv)
  python -m src.train --dataset csv --csv data/politifact.csv --text_col text --label_col label
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from .data import demo_data, load_csv, load_isot
from .model import build_lstm, metrics
from .optimizers import grey_wolf, levy_flight
from .ppca import PPCA
from .preprocess import clean_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["isot", "csv"], default="isot")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--csv"); ap.add_argument("--text_col", default="text"); ap.add_argument("--label_col", default="label")
    ap.add_argument("--demo", action="store_true", help="use tiny synthetic data (smoke test)")
    ap.add_argument("--max_samples", type=int, default=20000)
    ap.add_argument("--max_features", type=int, default=1500, help="TF-IDF vocabulary size")
    ap.add_argument("--components", type=int, default=50, help="PPCA latent dimensions")
    ap.add_argument("--optimizers", nargs="+", default=["levy", "gwo"], choices=["levy", "gwo", "none"])
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--tune_epochs", type=int, default=2)
    ap.add_argument("--tune_samples", type=int, default=3000)
    ap.add_argument("--final_epochs", type=int, default=10)
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    log = lambda *x: print(*x, flush=True)

    # 1. data
    log("[1/6] Loading data")
    if a.demo:
        df = demo_data()
    elif a.dataset == "isot":
        df = load_isot(a.data_dir)
    else:
        df = load_csv(a.csv, a.text_col, a.label_col)
    if len(df) > a.max_samples:
                per = a.max_samples // 2
    df = pd.concat([df[df.label == c].sample(min(per, int((df.label == c).sum())), random_state=a.seed)
                        for c in (0, 1)])
    df = df.sample(frac=1, random_state=a.seed).reset_index(drop=True)
    log(f"   {len(df)} articles | fake={int(df.label.sum())} real={int((1 - df.label).sum())}")

    # 2. preprocessing
    log("[2/6] Pre-processing (lowercase, stop-words, stemming) ...")
    texts = [clean_text(t) for t in df.text]
    y = df.label.values
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=a.seed)
    tr, va = train_test_split(tr, test_size=0.15, stratify=y[tr], random_state=a.seed)

    # 3. TF-IDF + PPCA
    log("[3/6] TF-IDF + PPCA (EM) feature reduction")
    tfidf = TfidfVectorizer(max_features=a.max_features, ngram_range=(1, 1), min_df=2 if not a.demo else 1)
    tfidf.fit([texts[i] for i in tr])
    X = lambda ids: tfidf.transform([texts[i] for i in ids]).toarray().astype(np.float32)
    rng = np.random.default_rng(a.seed)
    fit_ids = rng.choice(tr, size=min(len(tr), 8000), replace=False)   # keeps EM fast/low-memory
    ppca = PPCA(n_components=a.components, seed=a.seed).fit(X(fit_ids), log=log)
    Z = {k: ppca.transform(X(v)) for k, v in {"tr": tr, "va": va, "te": te}.items()}
    ytr, yva, yte = y[tr], y[va], y[te]
    joblib.dump(tfidf, os.path.join(a.out, "tfidf.joblib"))
    ppca.save(os.path.join(a.out, "ppca.npz"))

    # 4/5. optimise + train
    T = a.components
    sub = rng.choice(len(tr), size=min(len(tr), a.tune_samples), replace=False)

    def fitness(hp):
        m = build_lstm(T, hp["units"], hp["dropout"], hp["lr"])
        m.fit(Z["tr"][sub][..., None], ytr[sub], epochs=a.tune_epochs, batch_size=64, verbose=0)
        loss = m.evaluate(Z["va"][..., None], yva, verbose=0)[0]
        tf.keras.backend.clear_session()
        return float(loss) if np.isfinite(loss) else 10.0

    results = {"dataset": {"name": "demo" if a.demo else a.dataset, "n_articles": len(df),
                           "train": len(tr), "val": len(va), "test": len(te)},
               "config": {"tfidf_features": a.max_features, "ppca_components": a.components,
                          "pop": a.pop, "iters": a.iters, "tune_epochs": a.tune_epochs},
               "models": {}}
    for name in a.optimizers:
        log(f"[4/6] Optimiser: {name.upper()}")
        t0 = time.time()
        if name == "none":
            hp, hist = {"lr": 1e-3, "units": 64, "dropout": 0.2}, []
        else:
            fn = grey_wolf if name == "gwo" else levy_flight
            hp, hist = fn(fitness, pop=a.pop, iters=a.iters, seed=a.seed, log=log)
        log(f"   best hyper-parameters: {hp}")
        log(f"[5/6] Training final LSTM ({name})")
        model = build_lstm(T, hp["units"], hp["dropout"], hp["lr"])
        model.fit(Z["tr"][..., None], ytr, validation_data=(Z["va"][..., None], yva),
                  epochs=a.final_epochs, batch_size=64, verbose=2,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])
        m = metrics(yte, model.predict(Z["te"][..., None], verbose=0))
        m_va = metrics(yva, model.predict(Z["va"][..., None], verbose=0))
        log(f"[6/6] TEST  acc={m['accuracy']:.4f} prec={m['precision']:.4f} "
            f"rec={m['recall']:.4f} spec={m['specificity']:.4f}")
        model.save(os.path.join(a.out, f"lstm_{name}.keras"))
        results["models"][name] = {"hyperparams": hp, "convergence": hist, "test": m,
                                   "val_accuracy": m_va["accuracy"],
                                   "seconds": round(time.time() - t0, 1)}
        tf.keras.backend.clear_session()

    results["default_model"] = max(results["models"], key=lambda k: results["models"][k]["val_accuracy"])
    with open(os.path.join(a.out, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"\nDone. Default model: {results['default_model']}.  Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()

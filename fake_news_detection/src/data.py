"""Stage 1 - data loading. Label convention: Real = 0, Fake = 1."""
import os
import numpy as np
import pandas as pd


def load_isot(data_dir):
    fake = pd.read_csv(os.path.join(data_dir, "Fake.csv"))
    true = pd.read_csv(os.path.join(data_dir, "True.csv"))
    fake["label"], true["label"] = 1, 0
    df = pd.concat([fake, true], ignore_index=True)
    df["text_all"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df[["text_all", "label"]].rename(columns={"text_all": "text"})


def load_csv(path, text_col="text", label_col="label"):
    df = pd.read_csv(path)
    y = df[label_col]
    if y.dtype == object:
        y = y.astype(str).str.lower().map({"fake": 1, "false": 1, "real": 0, "true": 0})
    out = pd.DataFrame({"text": df[text_col].astype(str), "label": y})
    return out.dropna().astype({"label": int})


def demo_data(n=800, seed=0):
    """Tiny synthetic corpus ONLY to smoke-test the pipeline (not for research)."""
    rng = np.random.default_rng(seed)
    real_w = ("government minister announced policy parliament budget report officials said committee economy "
              "court statement president senate according spokesperson election results vote council").split()
    fake_w = ("shocking secret exposed hoax conspiracy miracle unbelievable bombshell scandal leaked "
              "they hiding truth cure banned viral outrage destroyed insane").split()
    common = "the of and in to a for on with new people year after over".split()
    rows = []
    for i in range(n):
        lab = i % 2
        pool = fake_w if lab else real_w
        words = list(rng.choice(pool, 14)) + list(rng.choice(common, 10)) + list(rng.choice(real_w + fake_w, 3))
        rng.shuffle(words)
        rows.append((" ".join(words), lab))
    return pd.DataFrame(rows, columns=["text", "label"])

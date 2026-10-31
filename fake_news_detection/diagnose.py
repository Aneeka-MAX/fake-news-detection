import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.data import load_isot
from src.preprocess import clean_text
from src.ppca import PPCA

df = load_isot("data")
df = pd.concat([df[df.label == c].sample(4000, random_state=42) for c in (0, 1)])
X = [clean_text(t) for t in df.text]; y = df.label.values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

for feats in (1500, 5000):
    vec = TfidfVectorizer(max_features=feats, min_df=2).fit(Xtr)
    A, B = vec.transform(Xtr), vec.transform(Xte)
    print(feats, "full TF-IDF + LR :", LogisticRegression(max_iter=1000).fit(A, ytr).score(B, yte))
    Ad, Bd = A.toarray(), B.toarray()
    for m in (50, 150):
        p = PPCA(m).fit(Ad[:6000])
        s = LogisticRegression(max_iter=1000).fit(p.transform(Ad), ytr).score(p.transform(Bd), yte)
        print(feats, f"PPCA {m} + LR      :", s)
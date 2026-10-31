"""Stage 6 - Streamlit UI.  Run:  streamlit run app.py   (opens http://localhost:8501)"""
import html
import re

import pandas as pd
import streamlit as st

from src.explain import explain
from src.pipeline import FakeNewsPipeline, load_results

MODELS_DIR = "models"
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")


@st.cache_resource
def get_pipeline(name):
    return FakeNewsPipeline.load(MODELS_DIR, name)


def highlight(text, weights):
    wmap = {w.lower(): v for w, v in weights}

    def repl(m):
        v = wmap.get(m.group(0).lower())
        w = html.escape(m.group(0))
        if v is None:
            return w
        a = min(0.85, 0.25 + abs(v) * 4)
        col = f"rgba(220,53,69,{a:.2f})" if v > 0 else f"rgba(40,167,69,{a:.2f})"
        return f'<span style="background:{col};padding:1px 3px;border-radius:3px">{w}</span>'

    out, last = [], 0
    for m in re.finditer(r"\w+", text):
        out.append(html.escape(text[last:m.start()]))
        out.append(repl(m))
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out).replace("\n", "<br>")


st.title("📰 Fake News Detection")
st.caption("PPCA + Grey-Wolf-optimised LSTM with LIME explanations — runs fully on localhost")

res = load_results(MODELS_DIR)
if not res:
    st.error("No trained model found. Train first:  `python -m src.train --demo`  (or use --dataset isot)")
    st.stop()

with st.sidebar:
    st.header("Model")
    names = list(res["models"])
    label = {"gwo": "GWO-LSTM (proposed)", "levy": "LSTM-LF (base paper)", "none": "LSTM (no tuning)"}
    choice = st.selectbox("Optimiser", names, index=names.index(res["default_model"]),
                          format_func=lambda n: label.get(n, n))
    n_feat = st.slider("LIME words to show", 5, 20, 10)
    n_samp = st.select_slider("LIME samples (accuracy vs speed)", [200, 500, 1000, 2000], value=500)
    st.divider()
    st.subheader("Benchmark (held-out test set)")
    rows = [{"Model": label.get(k, k), **{m.title(): f"{v['test'][m] * 100:.2f}%" for m in
             ["accuracy", "precision", "recall", "specificity"]}} for k, v in res["models"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(f"Dataset: {res['dataset']['name']} · {res['dataset']['n_articles']} articles")
    if res["dataset"]["name"] == "demo":
        st.warning("Demo (synthetic) data — metrics are NOT research results.")

text = st.text_area("Paste a news headline or article", height=220,
                    placeholder="e.g. Shocking secret exposed: officials hiding the miracle cure...")
if st.button("Analyse", type="primary") and text.strip():
    pipe = get_pipeline(choice)
    p_real, p_fake = pipe.predict_proba(text)[0]
    c1, c2 = st.columns([1, 2])
    with c1:
        if p_fake >= 0.5:
            st.error(f"🚩 FAKE  ({p_fake * 100:.1f}% confidence)")
        else:
            st.success(f"✅ REAL  ({p_real * 100:.1f}% confidence)")
        st.progress(float(p_fake), text=f"P(fake) = {p_fake:.3f}")
    with st.spinner("Generating LIME explanation..."):
        weights = explain(pipe, text, num_features=n_feat, num_samples=n_samp)
    with c2:
        st.markdown("**Why?**  🔴 pushes toward *Fake*  ·  🟢 pushes toward *Real*")
        st.markdown(highlight(text, weights), unsafe_allow_html=True)
    df = pd.DataFrame(weights, columns=["word", "weight"]).set_index("word")
    st.bar_chart(df, horizontal=True)
    st.caption("LIME is a local approximation of the model around this one text; "
               "it explains the model's behaviour, not ground truth.")
elif not text.strip():
    st.info("Enter some text and press **Analyse**.")

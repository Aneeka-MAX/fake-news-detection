# Fake News Detection — PPCA + GWO-LSTM + LIME (local)

Pipeline: Pre-processing → TF-IDF → **PPCA (EM)** → **LSTM** tuned by **Grey Wolf Optimizer**
(compared to a **Lévy-Flight** baseline) → **LIME** explanations → Streamlit UI on localhost.

## 1. Setup (Python 3.10–3.12)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## 2. Quick smoke test (~1-2 min, synthetic data)
```bash
python -m src.train --demo --iters 2 --pop 4 --final_epochs 5
streamlit run app.py
```
Open http://localhost:8501

## 3. Real research run (ISOT)
Download ISOT from Kaggle ("Fake and real news dataset"), put `Fake.csv` and `True.csv` in `data/`.
```bash
python -m src.train --dataset isot --data_dir data --max_samples 20000
```
Other datasets (BuzzFeed / PolitiFact / GossipCop) as CSV with a text and a label column:
```bash
python -m src.train --dataset csv --csv data/politifact.csv --text_col text --label_col label
```
Useful flags: `--optimizers levy gwo` (default, benchmarks both), `--components 50`,
`--pop 6 --iters 5`, `--tune_epochs 2`, `--final_epochs 10`.

Outputs go to `models/`: `tfidf.joblib`, `ppca.npz`, `lstm_<optimizer>.keras`, `results.json`
(test accuracy/precision/recall/specificity, best hyper-parameters, convergence curves — use these in your report's Results section).

## Notes for the report (honest caveats)
* The base paper uses Lévy Flight to optimise LSTM *weights*; here both Lévy Flight and GWO tune
  *hyper-parameters* (learning rate, units, dropout) so the comparison is fair. Say this explicitly.
* ISOT is known to be easy (~99% with simple models) and the "(Reuters) -" prefix leaks the label;
  it is stripped in `src/preprocess.py`. Don't expect to match/beat 98.5% on the harder datasets.
* LIME explains a single prediction locally; it is an approximation of the model.
* `--demo` data is synthetic: use it only to verify the code runs.

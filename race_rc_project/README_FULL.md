# RACE Reading Comprehension & Quiz Generation System
AI Lab Project — BS CS Spring 2026 | FAST NUCES Islamabad

This repository implements an end-to-end classical-ML pipeline for automatic
reading-comprehension question generation, answer verification, distractor
generation, and graduated hint extraction using the RACE dataset.

Contents: preprocessing, training (Model A & B), evaluation, and a Streamlit UI.

**Quick index:**
- Code: [race_rc_project/src](race_rc_project/src)
- UI: [race_rc_project/ui/app.py](race_rc_project/ui/app.py)
- Data (place here): [race_rc_project/data/raw](race_rc_project/data/raw)
- Models output: [race_rc_project/models](race_rc_project/models)

---

**Prerequisites**

- Python 3.8+
- GPU optional for neural baseline or Colab training; classical ML runs on CPU.
- Install packages from `requirements.txt`.

Recommended: use a virtual environment.

Windows example:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Linux / macOS example:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

**Dataset**

Download the RACE CSVs from Kaggle (or your source) and place them here:

- data/raw/train.csv
- data/raw/val.csv
- data/raw/test.csv

The project scripts expect the RACE format with columns: `id, article, question, A, B, C, D, answer`.

---

**Full pipeline (one-command)**

After dependencies and dataset are placed, run the end-to-end pipeline:

```bash
python src/train_all.py
```

What this does (in order):
- `src/preprocessing.py` — builds OHE and TF-IDF vectorizers, creates feature matrices in `data/processed/`.
- `src/model_a_train.py` — trains Model A components (LR, SVM, RF, NB, ensemble), K-Means, Label Propagation, and a question ranker.
- `src/model_b_train.py` — trains the distractor ranker and hint scorer.
- `src/evaluate.py` — runs BLEU/ROUGE/METEOR + ranking evaluations and writes `models/evaluation_report.json`.

Notes:
- Training scripts save checkpoints under `models/model_a/checkpoints/` and `models/model_b/checkpoints/`.
- `train_all.py` will exit with actionable messages if `data/raw/*.csv` are missing.

---

**Run pieces individually**

- Preprocess only:

```bash
python src/preprocessing.py
```

- Train Model A only (assumes preprocessing outputs exist):

```bash
python src/model_a_train.py
```

- Train Model B only:

```bash
python src/model_b_train.py
```

- Evaluate generation and ranking metrics (BLEU / ROUGE / METEOR):

```bash
python src/evaluate.py
```

---

**Run the Streamlit UI**

Start the interactive demo locally:

```bash
streamlit run ui/app.py
```

UI notes:
- Screen list: Article Input | Quiz View | Hint Panel | Analytics.
- The app loads models from `models/` and falls back to a demo mode if models are missing.
- Use the sidebar to load a random RACE example or paste your own passage.

---

**Colab / remote GPU usage**

To run on Google Colab while keeping code files local (recommended):

1. Upload dataset CSVs to Colab or mount Google Drive.
2. Use `src/colab_runner.py` as a small wrapper to invoke the local `src/train_all.py` pipeline.

Example inside Colab:

```python
from src.colab_runner import run_training
run_training()
```

Checkpointing is implemented in training scripts — save/load will resume training and avoid lost progress.

---

**Testing**

Run unit tests (pytest recommended):

```bash
pip install pytest
pytest -q
```

Current tests live under race_rc_project/tests and check inference fallbacks and basic app imports.

---

**File map (important files)**

- `race_rc_project/src/preprocessing.py` — TF-IDF & OHE vectorizers, feature builders, question templates.
- `race_rc_project/src/model_a_train.py` — Model A training & checkpoints.
- `race_rc_project/src/model_b_train.py` — Model B training & checkpoints.
- `race_rc_project/src/inference.py` — Unified inference API consumed by the UI.
- `race_rc_project/src/evaluate.py` — BLEU/ROUGE/METEOR evaluation harness.
- `race_rc_project/ui/app.py` — Streamlit UI (four screens).

---

**Recommendations & tips**

- If training on Colab T4, use checkpoints and smaller subset sizes for quick experiments.
- If a script fails with missing model files, run `python src/preprocessing.py` then `python src/model_a_train.py` and `python src/model_b_train.py` in order.
- Use `models/model_a/results.json` and `models/model_b/results.json` to inspect training metrics.

---

If you want, I can now: (A) add a short `make` / script wrappers to standardize runs, (B) harden model-loading error messages in `src/inference.py`, or (C) add unit tests verifying vectorizer persistence — which would you prefer next?

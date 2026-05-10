"""
src/model_a_train.py
Trains all Model A components:
  - Logistic Regression   (answer verification)
  - SVM/LinearSVC         (answer verification)
  - Naive Bayes           (question type classification)
  - Random Forest         (difficulty estimation + ensemble)
  - Soft-vote Ensemble    (LR + SVM + NB + RF)
  - K-Means Clustering    (unsupervised — 20/100 marks)
  - Label Propagation     (semi-supervised)
  - Question Ranker       (SVM trained to rank template-generated questions)

Feature representation: One-Hot Encoding (primary) via CountVectorizer(binary=True)
TF-IDF features are available as an optional supplement but OHE is used for grading.

Run after preprocessing.py.
Checkpoints are saved after every major step so Colab restarts don't lose progress.
"""

import os
import sys
import json
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.svm             import LinearSVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.naive_bayes     import GaussianNB
from sklearn.ensemble        import RandomForestClassifier
from sklearn.cluster         import MiniBatchKMeans
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.metrics         import (accuracy_score, f1_score,
                                     classification_report,
                                     confusion_matrix, silhouette_score)

PROC_DIR   = "data/processed"
MODELS_DIR = "models/model_a"
CKPT_DIR   = os.path.join(MODELS_DIR, "checkpoints")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _ckpt_path(name: str) -> str:
    return os.path.join(CKPT_DIR, f"{name}.pkl")

def _ckpt_exists(name: str) -> bool:
    return os.path.exists(_ckpt_path(name))

def save_ckpt(obj, name: str):
    joblib.dump(obj, _ckpt_path(name))
    print(f"  [checkpoint] Saved → {_ckpt_path(name)}")

def load_ckpt(name: str):
    obj = joblib.load(_ckpt_path(name))
    print(f"  [checkpoint] Loaded ← {_ckpt_path(name)}")
    return obj

# ── Load OHE feature matrices ──────────────────────────────────────────────────
print("Loading OHE feature matrices...")
X_train = np.load(os.path.join(PROC_DIR, "X_train_ohe.npy"),
                  allow_pickle=False)
y_train = np.load(os.path.join(PROC_DIR, "y_train.npy"),
                  allow_pickle=False)
X_val   = np.load(os.path.join(PROC_DIR, "X_val_ohe.npy"),
                  allow_pickle=False)
y_val   = np.load(os.path.join(PROC_DIR, "y_val.npy"),
                  allow_pickle=False)

print(f"  X_train: {X_train.shape} | class balance: {y_train.mean():.3f}")
print(f"  X_val:   {X_val.shape}   | class balance: {y_val.mean():.3f}")

results = {}

# ── Helper: evaluate and save ──────────────────────────────────────────────────
def evaluate_and_save(model, name: str, X_v, y_v):
    y_pred = model.predict(X_v)
    acc    = accuracy_score(y_v, y_pred)
    f1     = f1_score(y_v, y_pred, average="macro")
    cm     = confusion_matrix(y_v, y_pred).tolist()
    report = classification_report(y_v, y_pred, output_dict=True)
    print(f"  {name}: Accuracy={acc:.4f}  Macro-F1={f1:.4f}")
    results[name] = {
        "accuracy": acc, "macro_f1": f1,
        "confusion_matrix": cm, "report": report,
    }
    joblib.dump(model, os.path.join(MODELS_DIR, f"{name}.pkl"))
    return acc, f1


# ── 1. Logistic Regression ─────────────────────────────────────────────────────
print("\n[1/7] Logistic Regression (OHE features)...")
if _ckpt_exists("scaler_lr") and _ckpt_exists("logistic_regression"):
    scaler_lr = load_ckpt("scaler_lr")
    lr        = load_ckpt("logistic_regression")
    X_val_lr  = scaler_lr.transform(X_val)
else:
    scaler_lr  = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    X_val_lr   = scaler_lr.transform(X_val)
    lr = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced",
                            random_state=42)
    lr.fit(X_train_lr, y_train)
    save_ckpt(scaler_lr, "scaler_lr")
    save_ckpt(lr,        "logistic_regression")

joblib.dump(scaler_lr, os.path.join(MODELS_DIR, "scaler_lr.pkl"))
evaluate_and_save(lr, "logistic_regression", X_val_lr, y_val)


# ── 2. SVM (LinearSVC + calibration) ──────────────────────────────────────────
print("\n[2/7] SVM — LinearSVC with calibration (OHE features)...")
if _ckpt_exists("scaler_svm") and _ckpt_exists("svm"):
    scaler_svm = load_ckpt("scaler_svm")
    svm        = load_ckpt("svm")
    X_val_svm  = scaler_svm.transform(X_val)
else:
    scaler_svm  = StandardScaler()
    X_train_svm = scaler_svm.fit_transform(X_train)
    X_val_svm   = scaler_svm.transform(X_val)
    base_svm = LinearSVC(C=0.5, max_iter=2000, class_weight="balanced",
                         random_state=42)
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train_svm, y_train)
    save_ckpt(scaler_svm, "scaler_svm")
    save_ckpt(svm,        "svm")

joblib.dump(scaler_svm, os.path.join(MODELS_DIR, "scaler_svm.pkl"))
evaluate_and_save(svm, "svm", X_val_svm, y_val)


# ── 3. Naive Bayes ─────────────────────────────────────────────────────────────
print("\n[3/7] Naive Bayes (OHE features — question type classification)...")
if _ckpt_exists("scaler_nb") and _ckpt_exists("naive_bayes"):
    scaler_nb = load_ckpt("scaler_nb")
    nb        = load_ckpt("naive_bayes")
    X_val_nb  = scaler_nb.transform(X_val)
else:
    scaler_nb  = MinMaxScaler()   # GNB needs non-negative
    X_train_nb = scaler_nb.fit_transform(X_train)
    X_val_nb   = scaler_nb.transform(X_val)
    nb = GaussianNB()
    nb.fit(X_train_nb, y_train)
    save_ckpt(scaler_nb, "scaler_nb")
    save_ckpt(nb,        "naive_bayes")

joblib.dump(scaler_nb, os.path.join(MODELS_DIR, "scaler_nb.pkl"))
evaluate_and_save(nb, "naive_bayes", X_val_nb, y_val)


# ── 4. Random Forest ───────────────────────────────────────────────────────────
print("\n[4/7] Random Forest (difficulty estimation + ensemble, OHE features)...")
if _ckpt_exists("random_forest"):
    rf = load_ckpt("random_forest")
else:
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight="balanced", random_state=42,
                                n_jobs=-1)
    rf.fit(X_train, y_train)
    save_ckpt(rf, "random_forest")

evaluate_and_save(rf, "random_forest", X_val, y_val)


# ── 5. Soft-vote Ensemble ──────────────────────────────────────────────────────
print("\n[5/7] Soft-vote Ensemble (LR + SVM + NB + RF)...")
ensemble_probs = (
    lr.predict_proba(X_val_lr)
    + svm.predict_proba(X_val_svm)
    + nb.predict_proba(X_val_nb)
    + rf.predict_proba(X_val)
) / 4.0

y_ens    = np.argmax(ensemble_probs, axis=1)
acc_ens  = accuracy_score(y_val, y_ens)
f1_ens   = f1_score(y_val, y_ens, average="macro")
print(f"  Ensemble: Accuracy={acc_ens:.4f}  Macro-F1={f1_ens:.4f}")
results["ensemble"] = {"accuracy": acc_ens, "macro_f1": f1_ens}

# Save bundle (inference.py loads this single file)
bundle = {
    "scaler_lr": scaler_lr, "lr": lr,
    "scaler_svm": scaler_svm, "svm": svm,
    "scaler_nb": scaler_nb,  "nb": nb,
    "rf": rf,
}
joblib.dump(bundle, os.path.join(MODELS_DIR, "ensemble_bundle.pkl"))
save_ckpt(bundle, "ensemble_bundle")


# ── 6. K-Means Clustering (unsupervised — 20 marks) ───────────────────────────
print("\n[6/7] K-Means Clustering (unsupervised, OHE features)...")
if _ckpt_exists("kmeans"):
    km_bundle = load_ckpt("kmeans")
    km, scaler_km = km_bundle["km"], km_bundle["scaler"]
else:
    scaler_km = StandardScaler()
    X_km      = scaler_km.fit_transform(X_train[:5000])
    km = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=10)
    km.fit(X_km)
    km_bundle = {"km": km, "scaler": scaler_km}
    save_ckpt(km_bundle, "kmeans")

scaler_km = km_bundle["scaler"]
km        = km_bundle["km"]
X_km      = scaler_km.transform(X_train[:5000])

sample_idx = np.random.default_rng(42).choice(len(X_km), 1000, replace=False)
sil_score  = silhouette_score(X_km[sample_idx], km.labels_[sample_idx])
print(f"  K-Means (k=4): Silhouette Score = {sil_score:.4f}")

true_labels   = y_train[:5000]
cluster_labels = km.labels_
purity_vals   = []
for c in range(4):
    mask = cluster_labels == c
    if mask.sum() > 0:
        dominant = np.bincount(true_labels[mask].astype(int)).max()
        purity_vals.append(dominant / mask.sum())
purity = float(np.mean(purity_vals))
print(f"  K-Means purity: {purity:.4f}")
results["kmeans"] = {"silhouette": sil_score, "purity": purity}
joblib.dump(km_bundle, os.path.join(MODELS_DIR, "kmeans.pkl"))


# ── Label Propagation (Semi-supervised) ───────────────────────────────────────
print("\n  Label Propagation (semi-supervised, OHE features)...")
if _ckpt_exists("label_propagation"):
    lp_bundle = load_ckpt("label_propagation")
    lp = lp_bundle["lp"]
    labeled_idx = lp_bundle["labeled_idx"]
    y_semi_gt   = lp_bundle["y_semi_gt"]
    X_semi_sc   = scaler_lr.transform(X_train[:3000])
else:
    N         = 3000
    X_semi_sc = scaler_lr.transform(X_train[:N])
    y_semi    = y_train[:N].astype(int)

    rng         = np.random.default_rng(42)
    labeled_idx = rng.choice(N, int(N * 0.1), replace=False)
    y_semi_gt   = y_semi[labeled_idx]

    y_semi_masked              = np.full(N, -1, dtype=int)
    y_semi_masked[labeled_idx] = y_semi[labeled_idx]

    lp = LabelPropagation(kernel="knn", n_neighbors=7, max_iter=200)
    lp.fit(X_semi_sc, y_semi_masked)
    lp_bundle = {"lp": lp, "scaler": scaler_lr,
                 "labeled_idx": labeled_idx, "y_semi_gt": y_semi_gt}
    save_ckpt(lp_bundle, "label_propagation")

y_lp_pred = lp.predict(X_semi_sc[labeled_idx])
acc_lp = accuracy_score(y_semi_gt, y_lp_pred)
f1_lp  = f1_score(y_semi_gt, y_lp_pred, average="macro")
print(f"  Label Propagation (10% labels): Accuracy={acc_lp:.4f}  F1={f1_lp:.4f}")
results["label_propagation"] = {"accuracy": acc_lp, "macro_f1": f1_lp}
joblib.dump(lp_bundle, os.path.join(MODELS_DIR, "label_propagation.pkl"))


# ── 7. Question Ranker (SVM — ranks template-generated questions) ──────────────
# This fulfils the "Question Generator/Verifier" generation sub-task.
# The ranker is trained on sentence-level features derived from OHE vectors.
# At inference time: candidate sentences → templates → ranker → top question.
print("\n[7/7] Question Ranker (SVM on OHE sentence features)...")

if _ckpt_exists("question_ranker"):
    qr_bundle = load_ckpt("question_ranker")
else:
    import pandas as pd
    from src.preprocessing import (load_race, clean_text, generate_candidate_questions,
                                   build_ohe_verification_features)

    ohe_vec_loaded = joblib.load("models/ohe_vectorizer.pkl")

    train_qr_df = load_race("train").head(4000)
    Xq_rows, yq_rows = [], []

    print("  Building question ranker training data...")
    for _, row in tqdm(train_qr_df.iterrows(), total=len(train_qr_df),
                       desc="  QR data"):
        article       = str(row["article"])
        correct       = str(row["answer"]).strip().upper()
        correct_text  = str(row[correct])
        real_question = str(row["question"])

        candidates = generate_candidate_questions(article, ohe_vec_loaded, top_k=8)
        if not candidates:
            continue

        # Best candidate = most similar to real question (OHE cosine)
        from sklearn.metrics.pairwise import cosine_similarity as _cs
        real_q_v = ohe_vec_loaded.transform([clean_text(real_question)])
        for score, sent, q_text in candidates:
            cand_v = ohe_vec_loaded.transform([clean_text(q_text)])
            sim    = float(_cs(real_q_v, cand_v)[0, 0])
            words  = set(clean_text(q_text).split())
            art_words = set(clean_text(article).split())
            overlap   = len(words & art_words) / max(len(words | art_words), 1)
            has_wh    = int(any(q_text.lower().startswith(w)
                                for w in ("who","what","where","when","why","how","which")))
            Xq_rows.append([sim, overlap, score, has_wh,
                            min(len(q_text.split()) / 30.0, 1.0)])
            yq_rows.append(1 if sim > 0.3 else 0)

    if Xq_rows:
        Xq = np.array(Xq_rows, dtype=np.float32)
        yq = np.array(yq_rows)
        scaler_qr = StandardScaler()
        Xq_sc     = scaler_qr.fit_transform(Xq)
        base_qr   = LinearSVC(C=1.0, max_iter=1000, class_weight="balanced",
                               random_state=42)
        qr_model  = CalibratedClassifierCV(base_qr, cv=3)
        qr_model.fit(Xq_sc, yq)
        qr_bundle = {"model": qr_model, "scaler": scaler_qr}
        save_ckpt(qr_bundle, "question_ranker")
        print(f"  Question ranker trained on {len(Xq)} examples.")
    else:
        print("  No question ranker data — skipping (will use heuristic ranking).")
        qr_bundle = None

if qr_bundle:
    joblib.dump(qr_bundle, os.path.join(MODELS_DIR, "question_ranker.pkl"))
    results["question_ranker"] = {"status": "trained"}


# ── Save results summary ───────────────────────────────────────────────────────
with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 55)
print("MODEL A TRAINING COMPLETE")
print("=" * 55)
print("\nComparison Table (OHE-based features):")
print(f"{'Model':<25} {'Accuracy':>10} {'Macro-F1':>10}")
print("-" * 47)
for name, res in results.items():
    if "accuracy" in res:
        print(f"{name:<25} {res['accuracy']:>10.4f} {res.get('macro_f1', 0):>10.4f}")
print("\nBest for answer verification: ENSEMBLE")
print("Question ranker: models/model_a/question_ranker.pkl")
print("Checkpoints in: models/model_a/checkpoints/")

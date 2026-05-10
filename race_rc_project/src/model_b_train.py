"""
src/model_b_train.py
Trains all Model B components:
  - Distractor ranker  (Logistic Regression on OHE-based candidate features)
  - Extractive hint scorer (Logistic Regression on OHE sentence features)

Feature representation: One-Hot Encoding (primary).
TF-IDF vectorizer is loaded as a supplementary resource but OHE drives the pipeline.

Checkpoints are saved after every major step.
Run after preprocessing.py and model_a_train.py.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (accuracy_score, f1_score, precision_score,
                                   recall_score, confusion_matrix,
                                   classification_report)
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR    = "data/raw"
PROC_DIR   = "data/processed"
MODELS_DIR = "models/model_b"
CKPT_DIR   = os.path.join(MODELS_DIR, "checkpoints")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

# ── Checkpoint helpers ─────────────────────────────────────────────────────────
def _ckpt_path(name):
    return os.path.join(CKPT_DIR, f"{name}.pkl")

def _ckpt_exists(name):
    return os.path.exists(_ckpt_path(name))

def save_ckpt(obj, name):
    joblib.dump(obj, _ckpt_path(name))
    print(f"  [checkpoint] Saved → {_ckpt_path(name)}")

def load_ckpt(name):
    obj = joblib.load(_ckpt_path(name))
    print(f"  [checkpoint] Loaded ← {_ckpt_path(name)}")
    return obj

# ── Load vectorizers ───────────────────────────────────────────────────────────
print("Loading OHE vectorizer (primary)...")
ohe_vec = joblib.load("models/ohe_vectorizer.pkl")

from src.preprocessing import (load_race, clean_text,
                                rank_sentences_by_relevance,
                                get_distractor_candidates)


# ══════════════════════════════════════════════════════════════════════════════
# DISTRACTOR RANKER
# ══════════════════════════════════════════════════════════════════════════════

def build_distractor_features_ohe(article, question, candidate_sent,
                                   correct_answer, vec, pos_norm=0.0):
    """
    8-dim feature vector using OHE (binary bag-of-words) cosine similarity + Lexical:
    [cos_cand_correct, cos_cand_q, char_overlap,
     cand_len_norm, passage_freq_norm, not_in_correct_flag,
     pos_norm, content_density]
    """
    clean_cand    = clean_text(candidate_sent)
    clean_correct = clean_text(correct_answer)
    clean_q       = clean_text(question)

    vecs = vec.transform([clean_cand, clean_correct, clean_q])

    cos_cand_correct = float(cos_sim(vecs[0], vecs[1])[0, 0])
    cos_cand_q       = float(cos_sim(vecs[0], vecs[2])[0, 0])

    # Character-level bigram overlap
    def char_bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    bg_cand    = char_bigrams(clean_cand[:100])
    bg_correct = char_bigrams(clean_correct[:100])
    char_overlap = len(bg_cand & bg_correct) / max(len(bg_cand | bg_correct), 1)

    cand_len_norm = min(len(clean_cand.split()) / 50.0, 1.0)

    article_words = clean_text(article).split()
    cand_words    = clean_cand.split()
    passage_freq  = (np.mean([article_words.count(w) for w in cand_words]) / 10.0
                     if cand_words else 0.0)

    correct_words = set(clean_correct.split())
    cand_word_set = set(cand_words)
    not_in_correct = 1.0 if len(correct_words & cand_word_set) == 0 else 0.0

    # Simple lexical features:
    # 7. position norm
    # 8. content word density (words > 5 chars as proxy for content words if no POS tagger)
    content_words = [w for w in cand_words if len(w) > 5]
    content_density = len(content_words) / max(len(cand_words), 1)

    return np.array([
        cos_cand_correct, cos_cand_q, char_overlap,
        cand_len_norm, min(passage_freq, 1.0), not_in_correct,
        pos_norm, content_density
    ], dtype=np.float32)


def build_distractor_training_data(df, vec, max_rows=5000):
    """
    Positive examples: the non-correct RACE options (genuine distractors).
    Negative examples: random passage sentences with very low OHE overlap.
    Label 1 = good distractor, 0 = bad.
    """
    X_rows, y_rows = [], []
    options = ["A", "B", "C", "D"]
    df_sample = df.head(max_rows)
    print(f"  Building distractor training data ({len(df_sample)} rows)...")

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        article      = str(row["article"])
        question     = str(row["question"])
        correct      = str(row["answer"]).strip().upper()
        correct_text = str(row[correct])

        # Positive: non-correct options are real distractors
        for opt in options:
            if opt == correct:
                continue
            feats = build_distractor_features_ohe(
                article, question, str(row[opt]), correct_text, vec, pos_norm=0.5)
            X_rows.append(feats)
            y_rows.append(1)

        # Negative: passage sentences that are very unlike the correct answer
        sentences = re.split(r"(?<=[.!?])\s+", article.strip())
        sentences = [s for s in sentences if 5 <= len(s.split()) <= 25]
        for i, sent in enumerate(sentences[:3]):
            p_norm = i / max(len(sentences) - 1, 1)
            feats = build_distractor_features_ohe(
                article, question, sent, correct_text, vec, pos_norm=p_norm)
            if feats[0] < 0.05:
                X_rows.append(feats)
                y_rows.append(0)

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    print(f"  Distractor data: {X.shape}, balance={y.mean():.3f}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# HINT SCORER
# ══════════════════════════════════════════════════════════════════════════════

def build_hint_features_ohe(question, sentence, vec, pos_norm=0.0):
    """
    5-dim feature vector using OHE cosine similarity:
    [sim_q_sent, keyword_overlap, sentence_len, position_norm, wh_word_overlap]
    """
    clean_q    = clean_text(question)
    clean_sent = clean_text(sentence)

    vecs = vec.transform([clean_q, clean_sent])
    sim  = float(cos_sim(vecs[0], vecs[1])[0, 0])

    q_words    = set(clean_q.split())
    sent_words = set(clean_sent.split())
    overlap    = len(q_words & sent_words) / max(len(q_words | sent_words), 1)
    sent_len   = min(len(clean_sent.split()) / 40.0, 1.0)

    wh_words   = {"who", "what", "where", "when", "why", "how", "which"}
    wh_overlap = len(q_words & wh_words & sent_words) / max(len(wh_words & q_words), 1)

    return np.array([sim, overlap, sent_len, pos_norm, wh_overlap],
                    dtype=np.float32)


def build_hint_training_data(df, vec, max_rows=5000):
    """
    Sentence most similar (by OHE cosine) to question+answer = best hint (1).
    Others = 0.
    """
    X_rows, y_rows = [], []
    df_sample = df.head(max_rows)
    print(f"  Building hint training data ({len(df_sample)} rows)...")

    for _, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
        article      = str(row["article"])
        question     = str(row["question"])
        correct      = str(row["answer"]).strip().upper()
        correct_text = str(row[correct])

        sentences = re.split(r"(?<=[.!?])\s+", article.strip())
        sentences = [s for s in sentences if len(s.split()) > 3]
        if len(sentences) < 2:
            continue

        query  = question + " " + correct_text
        ranked = rank_sentences_by_relevance(article, query, vec,
                                             top_k=len(sentences))

        for i, (score, sent) in enumerate(ranked):
            pos_norm = i / max(len(ranked) - 1, 1)
            feats    = build_hint_features_ohe(question, sent, vec, pos_norm)
            X_rows.append(feats)
            y_rows.append(1 if i == 0 else 0)

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    print(f"  Hint data: {X.shape}, balance={y.mean():.3f}")
    return X, y


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("MODEL B TRAINING  (OHE — primary feature representation)")
    print("=" * 55)

    train_df = load_race("train")
    val_df   = load_race("val")
    results  = {}

    # ── Distractor Ranker ──────────────────────────────────────────────────────
    print("\n[1/2] Distractor Ranker (OHE-based LR)...")

    if _ckpt_exists("dist_train_X") and _ckpt_exists("dist_train_y"):
        X_dist_train = load_ckpt("dist_train_X")
        y_dist_train = load_ckpt("dist_train_y")
    else:
        X_dist_train, y_dist_train = build_distractor_training_data(
            train_df, ohe_vec, max_rows=5000)
        save_ckpt(X_dist_train, "dist_train_X")
        save_ckpt(y_dist_train, "dist_train_y")

    if _ckpt_exists("dist_val_X") and _ckpt_exists("dist_val_y"):
        X_dist_val = load_ckpt("dist_val_X")
        y_dist_val = load_ckpt("dist_val_y")
    else:
        X_dist_val, y_dist_val = build_distractor_training_data(
            val_df, ohe_vec, max_rows=1000)
        save_ckpt(X_dist_val, "dist_val_X")
        save_ckpt(y_dist_val, "dist_val_y")

    if _ckpt_exists("distractor_ranker"):
        dr_bundle  = load_ckpt("distractor_ranker")
        dist_lr    = dr_bundle["model"]
        scaler_dist = dr_bundle["scaler"]
    else:
        scaler_dist = StandardScaler()
        X_dt_sc = scaler_dist.fit_transform(X_dist_train)
        X_dv_sc = scaler_dist.transform(X_dist_val)
        dist_lr = LogisticRegression(C=1.0, max_iter=300,
                                     class_weight="balanced", random_state=42)
        dist_lr.fit(X_dt_sc, y_dist_train)
        dr_bundle = {"model": dist_lr, "scaler": scaler_dist}
        save_ckpt(dr_bundle, "distractor_ranker")

    X_dv_sc    = scaler_dist.transform(X_dist_val)
    y_dist_pred = dist_lr.predict(X_dv_sc)
    acc_d  = accuracy_score(y_dist_val, y_dist_pred)
    f1_d   = f1_score(y_dist_val, y_dist_pred, average="macro")
    prec_d = precision_score(y_dist_val, y_dist_pred, average="macro", zero_division=0)
    rec_d  = recall_score(y_dist_val, y_dist_pred, average="macro", zero_division=0)
    cm_d   = confusion_matrix(y_dist_val, y_dist_pred).tolist()

    print(f"  Distractor LR: Acc={acc_d:.4f}  F1={f1_d:.4f}  "
          f"Prec={prec_d:.4f}  Rec={rec_d:.4f}")
    results["distractor_ranker"] = {
        "accuracy": acc_d, "macro_f1": f1_d,
        "precision": prec_d, "recall": rec_d,
        "confusion_matrix": cm_d,
        "feature_method": "One-Hot Encoding (primary)",
    }
    joblib.dump(dr_bundle, os.path.join(MODELS_DIR, "distractor_ranker.pkl"))

    # ── Hint Scorer ────────────────────────────────────────────────────────────
    print("\n[2/2] Hint Scorer (OHE-based LR)...")

    if _ckpt_exists("hint_train_X") and _ckpt_exists("hint_train_y"):
        X_hint_train = load_ckpt("hint_train_X")
        y_hint_train = load_ckpt("hint_train_y")
    else:
        X_hint_train, y_hint_train = build_hint_training_data(
            train_df, ohe_vec, max_rows=5000)
        save_ckpt(X_hint_train, "hint_train_X")
        save_ckpt(y_hint_train, "hint_train_y")

    if _ckpt_exists("hint_val_X") and _ckpt_exists("hint_val_y"):
        X_hint_val = load_ckpt("hint_val_X")
        y_hint_val = load_ckpt("hint_val_y")
    else:
        X_hint_val, y_hint_val = build_hint_training_data(
            val_df, ohe_vec, max_rows=1000)
        save_ckpt(X_hint_val, "hint_val_X")
        save_ckpt(y_hint_val, "hint_val_y")

    if _ckpt_exists("hint_scorer"):
        hs_bundle   = load_ckpt("hint_scorer")
        hint_lr     = hs_bundle["model"]
        scaler_hint = hs_bundle["scaler"]
    else:
        scaler_hint = StandardScaler()
        X_ht_sc = scaler_hint.fit_transform(X_hint_train)
        X_hv_sc = scaler_hint.transform(X_hint_val)
        hint_lr = LogisticRegression(C=0.5, max_iter=300,
                                     class_weight="balanced", random_state=42)
        hint_lr.fit(X_ht_sc, y_hint_train)
        hs_bundle = {"model": hint_lr, "scaler": scaler_hint}
        save_ckpt(hs_bundle, "hint_scorer")

    X_hv_sc     = scaler_hint.transform(X_hint_val)
    y_hint_pred = hint_lr.predict(X_hv_sc)
    acc_h  = accuracy_score(y_hint_val, y_hint_pred)
    f1_h   = f1_score(y_hint_val, y_hint_pred, average="macro")
    prec_h = precision_score(y_hint_val, y_hint_pred, average="macro", zero_division=0)

    print(f"  Hint LR: Acc={acc_h:.4f}  F1={f1_h:.4f}  Prec={prec_h:.4f}")
    results["hint_scorer"] = {
        "accuracy": acc_h, "macro_f1": f1_h, "precision": prec_h,
        "feature_method": "One-Hot Encoding (primary)",
    }
    joblib.dump(hs_bundle, os.path.join(MODELS_DIR, "hint_scorer.pkl"))

    # ── Save results ───────────────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n✓ Model B training complete. Files saved to models/model_b/")
    print("  Checkpoints in: models/model_b/checkpoints/")

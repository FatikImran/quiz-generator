"""
src/inference.py
Unified inference API for the RACE RC system.
Used by both the Streamlit UI and evaluation scripts.

Feature representation: One-Hot Encoding (primary, via ohe_vectorizer.pkl).
TF-IDF vectorizer is also loaded as an optional supplement.

Public API
----------
generate_question(article)            → dict  (NEW — question generation)
verify_answer(article, q, options)    → dict
generate_distractors(article, q, ans) → list[str]
generate_hints(article, q, ans)       → list[str]
predict_from_race_row(row)            → dict  (full pipeline, single row)
"""

import os
import re
import time
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics.pairwise import cosine_similarity as cos_sim

# ── Load all models once ───────────────────────────────────────────────────────# Global var to cache the models so we dont reload every time
_LOADED = {}


def _load_models(base="models"):
    """loads all saved pkl files. will fail if train_all hasn't bin ran."""
    global _LOADED
    if _LOADED:
        return _LOADED

    base = "models"
    ma   = os.path.join(base, "model_a")
    mb   = os.path.join(base, "model_b")

    # Primary vectorizer: OHE
    _LOADED["ohe_vec"]   = joblib.load(os.path.join(base, "ohe_vectorizer.pkl"))
    # Optional supplement: TF-IDF (kept for backwards compat / analytics)
    tfidf_path = os.path.join(base, "tfidf_vectorizer.pkl")
    _LOADED["tfidf_vec"] = (joblib.load(tfidf_path)
                            if os.path.exists(tfidf_path) else None)

    _LOADED["ensemble"]    = joblib.load(os.path.join(ma, "ensemble_bundle.pkl"))
    _LOADED["distractor"]  = joblib.load(os.path.join(mb, "distractor_ranker.pkl"))
    _LOADED["hint"]        = joblib.load(os.path.join(mb, "hint_scorer.pkl"))

    qr_path = os.path.join(ma, "question_ranker.pkl")
    _LOADED["question_ranker"] = (joblib.load(qr_path)
                                  if os.path.exists(qr_path) else None)
    return _LOADED


def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _ohe_features(article, question, option, vec):
    """8-dim OHE-based verification feature vector."""
    clean_art = _clean(article)
    clean_q   = _clean(question)
    clean_opt = _clean(option)

    vecs = vec.transform([clean_art, clean_q, clean_opt])
    art_v, q_v, opt_v = vecs[0], vecs[1], vecs[2]

    cos_art_opt = float(cos_sim(art_v, opt_v)[0, 0])
    cos_q_opt   = float(cos_sim(q_v,   opt_v)[0, 0])
    cos_art_q   = float(cos_sim(art_v, q_v  )[0, 0])

    art_words = set(clean_art.split())
    q_words   = set(clean_q.split())
    opt_words = set(clean_opt.split())

    overlap_q_opt   = len(q_words   & opt_words) / max(len(q_words   | opt_words), 1)
    overlap_art_opt = len(art_words & opt_words) / max(len(art_words | opt_words), 1)

    feature_names = vec.get_feature_names_out()
    art_arr  = np.asarray(art_v.todense()).flatten()
    opt_arr  = np.asarray(opt_v.todense()).flatten()
    art_topk = set(feature_names[np.argsort(art_arr)[-50:]])
    opt_topk = set(feature_names[np.argsort(opt_arr)[-50:]])
    ohe_ov   = len(art_topk & opt_topk) / max(len(art_topk | opt_topk), 1)

    return np.array([
        cos_art_opt, cos_q_opt, cos_art_q,
        min(len(clean_opt.split()) / 50.0, 1.0),
        min(len(clean_q.split()) / 30.0, 1.0),
        overlap_q_opt, overlap_art_opt, ohe_ov,
    ], dtype=np.float32)


# ── WH-word templates (reused from preprocessing) ─────────────────────────────
_WH_CLUES = {
    "person":   "who",
    "people":   "who",
    "place":    "where",
    "location": "where",
    "reason":   "why",
    "because":  "why",
    "time":     "when",
    "year":     "when",
    "number":   "how many",
    "amount":   "how much",
}


def _sentence_to_question(sentence: str) -> str:
    lower = sentence.lower()
    chosen_wh = "what"
    for clue, wh in _WH_CLUES.items():
        if clue in lower:
            chosen_wh = wh
            break
    words = sentence.strip().split()
    body  = " ".join(words[1:]) if len(words) > 6 else sentence.strip()
    body  = body.rstrip(".,;:!?").lower()
    return f"{chosen_wh.capitalize()} {body}?"


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def generate_question(article: str) -> dict:
    """
    Model A — Generation sub-task.

    Extracts candidate sentences via OHE keyword overlap, applies Wh-word
    templates, then ranks with the trained SVM question ranker (if available)
    or falls back to heuristic scoring.

    Returns:
      {
        "question":      str,
        "source_sent":   str,   # the passage sentence the question came from
        "candidates":    list[{"question": str, "score": float}],
        "method":        "ml_ranker" | "heuristic",
        "latency_ms":    float,
      }
    """
    t0 = time.time()
    m  = _load_models()
    vec = m["ohe_vec"]
    qr  = m["question_ranker"]  # may be None

    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 40]
    if not sentences:
        return {"question": "What is the main topic of the passage?",
                "source_sent": "",
                "candidates": [],
                "method": "fallback",
                "latency_ms": (time.time() - t0) * 1000}

    clean_art  = _clean(article)
    art_words  = set(clean_art.split())

    candidates = []
    for sent in sentences:
        clean_sent = _clean(sent)
        
        # Candidate sentence extraction using One-Hot keyword overlap!
        vecs = vec.transform([clean_art, clean_sent])
        # Since vec is a binary CountVectorizer, this overlap reflects OHE keyword overlap.
        overlap = float(cos_sim(vecs[0], vecs[1])[0, 0])
        
        boost   = sum(0.05 for cue in _WH_CLUES if cue in clean_sent)

        q_text  = _sentence_to_question(sent)
        has_wh  = int(q_text.split()[0].lower() in
                      ("who","what","where","when","why","how","which"))

        heuristic_score = overlap + boost
        candidates.append({
            "sentence": sent,
            "question": q_text,
            "heuristic_score": heuristic_score,
            "features": np.array([
                heuristic_score,         # sim proxy (heuristic)
                overlap,                 # OHE overlap
                heuristic_score,         # score again as position proxy
                has_wh,
                min(len(q_text.split()) / 30.0, 1.0),
            ], dtype=np.float32),
        })

    method = "heuristic"
    if qr is not None:
        try:
            X_cands = np.vstack([c["features"] for c in candidates])
            X_sc    = qr["scaler"].transform(X_cands)
            scores  = qr["model"].predict_proba(X_sc)[:, 1]
            for i, c in enumerate(candidates):
                c["ml_score"] = float(scores[i])
            # SVM to rank generated questions
            candidates.sort(key=lambda c: c["ml_score"], reverse=True)
            method = "ml_ranker (SVM)"
        except Exception as e:
            print(f"SVM ranking failed: {e}")
            candidates.sort(key=lambda c: c["heuristic_score"], reverse=True)
    else:
        candidates.sort(key=lambda c: c["heuristic_score"], reverse=True)

    best = candidates[0]
    top5 = [{"question": c["question"],
              "score": float(c.get("ml_score", c["heuristic_score"]))}
            for c in candidates[:5]]

    return {
        "question":    best["question"],
        "source_sent": best["sentence"],
        "candidates":  top5,
        "method":      method,
        "latency_ms":  (time.time() - t0) * 1000,
    }


def verify_answer(article: str, question: str, options: dict) -> dict:
    """
    Given article, question, and options dict {"A": text, ...},
    returns predicted best answer label and confidence scores.

    Returns:
      {
        "predicted":     "A",
        "probabilities": {"A": 0.4, "B": 0.2, "C": 0.2, "D": 0.2},
        "latency_ms":    12.4,
        "feature_method": "One-Hot Encoding (primary)",
      }
    """
    t0     = time.time()
    m      = _load_models()
    vec    = m["ohe_vec"]    # OHE is the primary feature vectorizer
    bundle = m["ensemble"]

    scaler_lr  = bundle["scaler_lr"]
    lr         = bundle["lr"]
    scaler_svm = bundle["scaler_svm"]
    svm        = bundle["svm"]
    scaler_nb  = bundle["scaler_nb"]
    nb         = bundle["nb"]
    rf         = bundle["rf"]

    labels    = ["A", "B", "C", "D"]
    probs_all = np.zeros((4, 2), dtype=np.float32)

    for i, label in enumerate(labels):
        feats = _ohe_features(article, question, options[label], vec).reshape(1, -1)
        p_lr  = lr.predict_proba(scaler_lr.transform(feats))[0]
        p_svm = svm.predict_proba(scaler_svm.transform(feats))[0]
        p_nb  = nb.predict_proba(scaler_nb.transform(feats))[0]
        p_rf  = rf.predict_proba(feats)[0]
        probs_all[i] = (p_lr + p_svm + p_nb + p_rf) / 4.0

    correct_probs  = probs_all[:, 1]
    correct_probs /= (correct_probs.sum() + 1e-9)
    predicted_idx  = int(np.argmax(correct_probs))

    return {
        "predicted":      labels[predicted_idx],
        "probabilities":  {labels[i]: float(correct_probs[i]) for i in range(4)},
        "latency_ms":     (time.time() - t0) * 1000,
        "feature_method": "One-Hot Encoding (primary)",
    }


def generate_distractors(article: str, question: str, correct_answer: str,
                          n: int = 3) -> list:
    """
    Returns n distractor strings extracted and ranked from the article
    using OHE-based cosine similarity features.
    """
    m      = _load_models()
    vec    = m["ohe_vec"]
    bundle = m["distractor"]
    model  = bundle["model"]
    scaler = bundle["scaler"]

    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 40]
    if not sentences:
        return []

    candidates_features = []
    clean_correct = _clean(correct_answer)

    for pos, sent in enumerate(sentences):
        clean_sent = _clean(sent)
        vecs = vec.transform([clean_sent, clean_correct, _clean(question)])
        c_cor = float(cos_sim(vecs[0], vecs[1])[0, 0])
        c_q   = float(cos_sim(vecs[0], vecs[2])[0, 0])

        def char_bigrams(s):
            return set(s[i:i+2] for i in range(len(s)-1))
        bg_s = char_bigrams(clean_sent[:100])
        bg_c = char_bigrams(clean_correct[:100])
        char_ov = len(bg_s & bg_c) / max(len(bg_s | bg_c), 1)

        art_words  = _clean(article).split()
        sent_words = clean_sent.split()
        pfreq = (np.mean([art_words.count(w) for w in sent_words]) / 10.0
                 if sent_words else 0.0)

        not_in_correct = 1.0 if len(bg_s & bg_c) < 3 else 0.0
        pos_norm = pos / max(len(sentences) - 1, 1)
        content_words = [w for w in sent_words if len(w) > 5]
        content_density = len(content_words) / max(len(sent_words), 1)

        feats = np.array([c_cor, c_q, char_ov,
                          min(len(sent_words) / 50.0, 1.0),
                          min(pfreq, 1.0), not_in_correct, pos_norm, content_density])
        candidates_features.append(feats)

    X     = np.vstack(candidates_features)
    X_sc  = scaler.transform(X)
    probs = model.predict_proba(X_sc)[:, 1]

    filtered_pairs = []
    for i, (p, sent) in enumerate(zip(probs, sentences)):
        c_cor_val = candidates_features[i][0]   # cos similarity to correct
        char_ov = candidates_features[i][2]     # character overlap
        
        # Hard constraint: discard trivially similar distractors (character level)
        if char_ov > 0.65:
            continue
            
        if 0.03 < c_cor_val < 0.55 and correct_answer.lower()[:20] not in sent.lower():
            filtered_pairs.append((p, sent))

    filtered_pairs.sort(key=lambda x: x[0], reverse=True)

    result = []
    selected_vecs = []
    
    for _, sent in filtered_pairs:
        # Check diversity penalty: must be somewhat different from already chosen ones
        clean_cand = _clean(sent)
        cand_v = vec.transform([clean_cand])
        
        is_diverse = True
        for sel_v in selected_vecs:
            if float(cos_sim(cand_v, sel_v)[0, 0]) > 0.5:
                is_diverse = False
                break
                
        if is_diverse:
            selected_vecs.append(cand_v)
            result.append(sent[:150])
            
        if len(result) == n:
            break

    while len(result) < n:
        result.append(f"None of the above ({len(result)+1})")

    return result[:n]


def generate_hints(article: str, question: str, correct_answer: str) -> list:
    """
    Returns 3 graduated hints (most general → near-explicit) using OHE scoring.
    """
    m      = _load_models()
    vec    = m["ohe_vec"]
    bundle = m["hint"]
    model  = bundle["model"]
    scaler = bundle["scaler"]

    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 4]

    if len(sentences) < 3:
        return (sentences[:3] + ["Hint not available"] * (3 - len(sentences)))

    query   = question + " " + correct_answer
    clean_q = _clean(query)

    # We need 3 graduated hints:
    # 1. General: Most similar to the question only (where to look)
    # 2. Specific: The context sentence immediately before the explicit clue (or second best if at pos 0)
    # 3. Near Explicit: The top scoring sentence from our model (question+answer)
    
    clean_q_only = _clean(question)
    
    scored = []
    for pos, sent in enumerate(sentences):
        clean_sent = _clean(sent)
        vecs  = vec.transform([clean_q, clean_sent, clean_q_only])
        sim   = float(cos_sim(vecs[0], vecs[1])[0, 0])
        sim_q = float(cos_sim(vecs[2], vecs[1])[0, 0])  # Sim to question only

        q_words   = set(clean_q.split())
        s_words   = set(clean_sent.split())
        overlap   = len(q_words & s_words) / max(len(q_words | s_words), 1)
        sent_len  = min(len(s_words) / 40.0, 1.0)
        pos_norm  = pos / max(len(sentences) - 1, 1)
        wh_words  = {"who", "what", "where", "when", "why", "how", "which"}
        wh_ov     = len(q_words & wh_words & s_words) / max(len(wh_words & q_words), 1)

        feats    = np.array([sim, overlap, sent_len, pos_norm, wh_ov])
        feats_sc = scaler.transform(feats.reshape(1, -1))
        score    = model.predict_proba(feats_sc)[0][1]
        scored.append((score, sim_q, sent, pos))

    # Sort by model score (which predicts the near-explicit answer sentence)
    scored_by_model = sorted(scored, key=lambda x: x[0], reverse=True)
    best_hint_3 = scored_by_model[0]
    best_pos = best_hint_3[3]

    # Hint 3: Near-Explicit
    hint3 = best_hint_3[2]

    # Hint 2: Specific (context right before the answer, or second best matching if it's the first sentence)
    if best_pos > 0:
        hint2 = sentences[best_pos - 1]
    elif len(scored_by_model) > 1:
        hint2 = scored_by_model[1][2]
    else:
        hint2 = hint3

    # Hint 1: General (most similar to the question only, excluding hint2 and hint3 if possible)
    scored_by_q = sorted(scored, key=lambda x: x[1], reverse=True)
    hint1 = scored_by_q[0][2]
    for h in scored_by_q:
        if h[2] != hint3 and h[2] != hint2:
            hint1 = h[2]
            break

    return [hint1, hint2, hint3]


def predict_from_race_row(row: dict) -> dict:
    """
    Full inference on one RACE dataset row.
    row must have keys: article, question, A, B, C, D, answer

    Also generates a question from the article to demonstrate the
    generation sub-task (Model A generation component).
    """
    article  = str(row["article"])
    question = str(row["question"])
    options  = {k: str(row[k]) for k in ["A", "B", "C", "D"]}
    correct  = str(row["answer"]).strip().upper()
    correct_text = options[correct]

    # Question generation (Model A — generation sub-task)
    gen_result   = generate_question(article)

    # Answer verification (Model A — verification sub-task)
    verify_result = verify_answer(article, question, options)

    # Distractor + hint generation (Model B)
    distractors  = generate_distractors(article, question, correct_text)
    hints        = generate_hints(article, question, correct_text)

    return {
        "article":           article,
        "question":          question,
        "generated_question": gen_result,
        "options":           options,
        "correct_label":     correct,
        "correct_text":      correct_text,
        "verification":      verify_result,
        "distractors":       distractors,
        "hints":             hints,
    }


if __name__ == "__main__":
    import pandas as pd
    df  = pd.read_csv("data/raw/val.csv")
    row = df.iloc[0].to_dict()
    result = predict_from_race_row(row)

    print("\n=== INFERENCE SANITY CHECK ===")
    print(f"Original question:   {result['question'][:80]}...")
    print(f"Generated question:  {result['generated_question']['question']}")
    print(f"  method={result['generated_question']['method']}  "
          f"latency={result['generated_question']['latency_ms']:.1f}ms")
    print(f"\nCorrect answer: ({result['correct_label']}) {result['correct_text'][:60]}")
    print(f"Predicted:      ({result['verification']['predicted']}) "
          f"— {result['verification']['probabilities']}")
    print(f"Feature method: {result['verification']['feature_method']}")
    print(f"Latency: {result['verification']['latency_ms']:.1f} ms")
    print("\nGenerated distractors:")
    for i, d in enumerate(result["distractors"], 1):
        print(f"  D{i}: {d[:80]}")
    print("\nHints:")
    for i, h in enumerate(result["hints"], 1):
        print(f"  Hint {i}: {h[:80]}")

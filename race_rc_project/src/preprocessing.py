"""
src/preprocessing.py
Feature engineering pipeline for the RACE RC project.

Primary feature representation: One-Hot Encoding (OHE) of bag-of-words vectors.
TF-IDF vectorization is also built and saved as an optional/supplementary pipeline.

Outputs:
  data/processed/X_train_ohe.npy   — OHE-based verification features (primary)
  data/processed/X_val_ohe.npy
  data/processed/y_train.npy
  data/processed/y_val.npy
  models/ohe_vectorizer.pkl         — CountVectorizer (binary=True → OHE)
  models/tfidf_vectorizer.pkl       — TF-IDF vectorizer (optional supplement)
"""

import os
import re
import string
import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR    = "data/raw"
PROC_DIR   = "data/processed"
MODELS_DIR = "models"
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs("models/model_a", exist_ok=True)
os.makedirs("models/model_b", exist_ok=True)

ANSWER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Load RACE CSV ──────────────────────────────────────────────────────────────
def load_race(split: str) -> pd.DataFrame:
    """Load train / val / test CSV and add 'answer_idx' column."""
    path = os.path.join(RAW_DIR, f"{split}.csv")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["answer_idx"] = df["answer"].map(ANSWER_MAP)
    df = df.dropna(subset=["article", "question", "A", "B", "C", "D", "answer"])
    df = df.reset_index(drop=True)
    return df


# ── One-Hot Encoding vectorizer (primary) ─────────────────────────────────────
def build_ohe_vectorizer(train_df: pd.DataFrame) -> CountVectorizer:
    """
    Build a binary bag-of-words (One-Hot Encoding) vectorizer.
    Fit on training articles + questions.
    binary=True → each term presence = 1, absence = 0 (true OHE spirit).
    """
    corpus = (train_df["article"].tolist()
              + train_df["question"].tolist()
              + train_df["A"].tolist()
              + train_df["B"].tolist()
              + train_df["C"].tolist()
              + train_df["D"].tolist())
    corpus = [clean_text(t) for t in corpus]

    vec = CountVectorizer(
        max_features=12000,
        stop_words="english",
        binary=True,          # ← One-Hot: presence/absence, not count
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.95,
    )
    vec.fit(corpus)
    return vec


# ── TF-IDF vectorizer (optional supplement) ───────────────────────────────────
def build_tfidf_vectorizer(train_df: pd.DataFrame) -> TfidfVectorizer:
    corpus = train_df["article"].tolist()
    vec = TfidfVectorizer(
        max_features=15000,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        norm="l2",
    )
    vec.fit(corpus)
    return vec


# ── OHE verification feature vector for one (article, question, option) ───────
def build_ohe_verification_features(
    article: str,
    question: str,
    option: str,
    ohe_vec: CountVectorizer,
) -> np.ndarray:
    """
    Returns a 1-D numpy array of 8 features for answer verification
    using One-Hot Encoding (binary bag-of-words) representations:

    [cos_sim_art_opt,  cos_sim_q_opt,  cos_sim_art_q,
     opt_len_norm,     q_len_norm,
     word_overlap_q_opt, word_overlap_art_opt,
     ohe_topk_overlap]
    """
    clean_art = clean_text(article)
    clean_q   = clean_text(question)
    clean_opt = clean_text(option)

    # OHE vectors (binary bag-of-words)
    vecs = ohe_vec.transform([clean_art, clean_q, clean_opt])
    art_v  = vecs[0]
    q_v    = vecs[1]
    opt_v  = vecs[2]

    cos_art_opt = float(cosine_similarity(art_v, opt_v)[0, 0])
    cos_q_opt   = float(cosine_similarity(q_v,   opt_v)[0, 0])
    cos_art_q   = float(cosine_similarity(art_v, q_v  )[0, 0])

    # Lexical overlap features
    art_words = set(clean_art.split())
    q_words   = set(clean_q.split())
    opt_words = set(clean_opt.split())

    overlap_q_opt   = len(q_words   & opt_words) / max(len(q_words   | opt_words), 1)
    overlap_art_opt = len(art_words & opt_words) / max(len(art_words | opt_words), 1)

    # OHE top-K overlap between article and option
    feature_names = ohe_vec.get_feature_names_out()
    art_arr  = np.asarray(art_v.todense()).flatten()
    opt_arr  = np.asarray(opt_v.todense()).flatten()
    top_k    = 50
    art_topk = set(feature_names[np.argsort(art_arr)[-top_k:]])
    opt_topk = set(feature_names[np.argsort(opt_arr)[-top_k:]])
    ohe_overlap = len(art_topk & opt_topk) / max(len(art_topk | opt_topk), 1)

    opt_len_norm = min(len(clean_opt.split()) / 50.0, 1.0)
    q_len_norm   = min(len(clean_q.split()) / 30.0, 1.0)

    return np.array([
        cos_art_opt, cos_q_opt, cos_art_q,
        opt_len_norm, q_len_norm,
        overlap_q_opt, overlap_art_opt,
        ohe_overlap,
    ], dtype=np.float32)


# ── TF-IDF verification feature vector (kept for optional comparison) ─────────
def build_tfidf_verification_features(
    article: str,
    question: str,
    option: str,
    tfidf_vec: TfidfVectorizer,
) -> np.ndarray:
    """
    Same 8-feature layout but using TF-IDF vectors.
    Kept for optional comparison / ablation; OHE is the primary method.
    """
    clean_art = clean_text(article)
    clean_q   = clean_text(question)
    clean_opt = clean_text(option)

    vecs = tfidf_vec.transform([clean_art, clean_q, clean_opt])
    art_v  = vecs[0]
    q_v    = vecs[1]
    opt_v  = vecs[2]

    cos_art_opt = float(cosine_similarity(art_v, opt_v)[0, 0])
    cos_q_opt   = float(cosine_similarity(q_v,   opt_v)[0, 0])
    cos_art_q   = float(cosine_similarity(art_v, q_v  )[0, 0])

    art_words = set(clean_art.split())
    q_words   = set(clean_q.split())
    opt_words = set(clean_opt.split())
    overlap_q_opt   = len(q_words   & opt_words) / max(len(q_words   | opt_words), 1)
    overlap_art_opt = len(art_words & opt_words) / max(len(art_words | opt_words), 1)

    feature_names = tfidf_vec.get_feature_names_out()
    art_arr  = np.asarray(art_v.todense()).flatten()
    opt_arr  = np.asarray(opt_v.todense()).flatten()
    top_k    = 50
    art_topk = set(feature_names[np.argsort(art_arr)[-top_k:]])
    opt_topk = set(feature_names[np.argsort(opt_arr)[-top_k:]])
    tfidf_overlap = len(art_topk & opt_topk) / max(len(art_topk | opt_topk), 1)

    opt_len_norm = min(len(clean_opt.split()) / 50.0, 1.0)
    q_len_norm   = min(len(clean_q.split()) / 30.0, 1.0)

    return np.array([
        cos_art_opt, cos_q_opt, cos_art_q,
        opt_len_norm, q_len_norm,
        overlap_q_opt, overlap_art_opt,
        tfidf_overlap,
    ], dtype=np.float32)


# ── Alias used by inference.py (keeps backward-compat) ───────────────────────
def build_verification_features(
    article: str,
    question: str,
    option: str,
    vec,            # accepts either OHE CountVectorizer or TF-IDF Vectorizer
) -> np.ndarray:
    """Dispatch to OHE or TF-IDF feature builder based on vectorizer type."""
    if isinstance(vec, CountVectorizer):
        return build_ohe_verification_features(article, question, option, vec)
    return build_tfidf_verification_features(article, question, option, vec)


# ── Build full verification matrix (all 4 options per row) ────────────────────
def build_verification_matrix(df: pd.DataFrame, vec,
                               max_rows: int = None, split_name: str = "train"):
    """
    For each row, creates 4 samples (one per option).
    Label = 1 if that option is the correct answer, else 0.
    Returns X (n_samples × 8), y (n_samples,)
    """
    if max_rows:
        df = df.head(max_rows)

    X_rows, y_rows = [], []
    options = ["A", "B", "C", "D"]

    print(f"Building verification matrix for {split_name} ({len(df)} articles)...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        article  = str(row["article"])
        question = str(row["question"])
        correct  = str(row["answer"]).strip().upper()

        for opt_label in options:
            option_text = str(row[opt_label])
            feats = build_verification_features(article, question, option_text, vec)
            X_rows.append(feats)
            y_rows.append(1 if opt_label == correct else 0)

    X = np.vstack(X_rows)
    y = np.array(y_rows)
    print(f"  → X shape: {X.shape}, class balance: {y.mean():.3f} (should be ~0.25)")
    return X, y


# ── Sentence-level cosine similarity (for hint + distractor) ──────────────────
def rank_sentences_by_relevance(
    article: str,
    query: str,
    vec,
    top_k: int = 5,
) -> list:
    """
    Splits article into sentences. Ranks each by cosine similarity to query.
    Works with both OHE and TF-IDF vectorizers.
    Returns [(score, sentence), ...] sorted descending.
    """
    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    sentences = [s.strip() for s in sentences if len(s.split()) > 3]
    if not sentences:
        return []

    all_texts = sentences + [query]
    vecs      = vec.transform([clean_text(t) for t in all_texts])
    query_v   = vecs[-1]
    sent_vecs = vecs[:-1]

    sims = cosine_similarity(query_v, sent_vecs)[0]
    ranked = sorted(zip(sims, sentences), key=lambda x: x[0], reverse=True)
    return ranked[:top_k] if top_k else ranked


# ── Distractor candidate extraction ───────────────────────────────────────────
def get_distractor_candidates(
    article: str,
    question: str,
    correct_answer: str,
    vec,
    n_candidates: int = 10,
) -> list:
    """
    Returns medium-similarity sentences from the article as distractor candidates.
    Filters: avoid too-similar (gives away answer) and too-dissimilar (irrelevant).
    """
    all_ranked = rank_sentences_by_relevance(
        article, correct_answer, vec,
        top_k=len(re.split(r"(?<=[.!?])\s+", article))
    )

    filtered = [
        (s, sent) for s, sent in all_ranked
        if 0.05 < s < 0.6
        and correct_answer.lower().strip() not in sent.lower()
    ]

    seen, unique = set(), []
    for score, sent in filtered:
        key = sent[:40]
        if key not in seen:
            seen.add(key)
            unique.append((score, sent))

    return unique[:n_candidates]


# ── Template-based question generation ────────────────────────────────────────
WH_TEMPLATES = {
    "who":   "Who {verb_phrase}?",
    "what":  "What {verb_phrase}?",
    "where": "Where {verb_phrase}?",
    "when":  "When {verb_phrase}?",
    "why":   "Why {verb_phrase}?",
    "how":   "How {verb_phrase}?",
    "which": "Which {verb_phrase}?",
}

# Simple auxiliary verb starters for heuristic detection
_WH_CLUES = {
    "person": "who",
    "people": "who",
    "place":  "where",
    "location": "where",
    "reason": "why",
    "because": "why",
    "time":   "when",
    "year":   "when",
    "number": "how many",
    "amount": "how much",
}


def sentence_to_question(sentence: str) -> str:
    """
    Apply a simple Wh-word template to transform a declarative sentence
    into a multiple-choice question stem.

    Strategy:
    1. Scan the sentence for known answer-type clues.
    2. Apply the best matching Wh-word template.
    3. Fall back to "What does the passage say about ___?"
    """
    clean = sentence.strip()
    lower = clean.lower()

    chosen_wh = "what"
    for clue, wh in _WH_CLUES.items():
        if clue in lower:
            chosen_wh = wh
            break

    # Heuristically build question body by removing subject-like prefix
    words = clean.split()
    if len(words) > 6:
        body = " ".join(words[1:])  # drop first word (often subject)
    else:
        body = clean

    # Remove trailing period / comma
    body = body.rstrip(".,;:!?")

    # Actually use the WH_TEMPLATES mapping (was missing before!)
    template = WH_TEMPLATES.get(chosen_wh, WH_TEMPLATES["what"])
    question = template.format(verb_phrase=body.lower())
    
    return question


def generate_candidate_questions(
    article: str,
    vec,
    top_k: int = 5,
) -> list:
    """
    Step 1: extract candidate sentences via OHE keyword overlap with article.
    Step 2: apply Wh-word templates to convert them to question stems.
    Returns list of (score, sentence, question) tuples — best candidates first.
    """
    sentences = re.split(r"(?<=[.!?])\s+", article.strip())
    sentences = [s.strip() for s in sentences if 5 <= len(s.split()) <= 40]
    if not sentences:
        return []

    # Score each sentence against the full article (self-similarity weighted by OHE)
    clean_art = clean_text(article)
    results = []
    for sent in sentences:
        clean_sent = clean_text(sent)
        
        # Proper One-Hot keyword overlap utilizing the passed OHE vectorizer
        vecs = vec.transform([clean_art, clean_sent])
        overlap = float(cosine_similarity(vecs[0], vecs[1])[0, 0])


        # Boost sentences with question-relevant cue words
        boost = 0.0
        for cue in _WH_CLUES:
            if cue in clean_sent:
                boost += 0.05

        score = overlap + boost
        q_text = sentence_to_question(sent)
        results.append((score, sent, q_text))

    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


# ── Main: preprocess and save everything ──────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RACE Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load data
    print("\n[1/6] Loading datasets...")
    train_df = load_race("train")
    val_df   = load_race("val")
    test_df  = load_race("test")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # 2. Build + save OHE vectorizer (primary)
    print("\n[2/6] Fitting OHE (binary bag-of-words) vectorizer — PRIMARY...")
    ohe_vec = build_ohe_vectorizer(train_df)
    joblib.dump(ohe_vec, os.path.join(MODELS_DIR, "ohe_vectorizer.pkl"))
    print(f"  OHE vocabulary size: {len(ohe_vec.vocabulary_)}")

    # 3. Build + save TF-IDF vectorizer (optional supplement)
    print("\n[3/6] Fitting TF-IDF vectorizer — OPTIONAL SUPPLEMENT...")
    tfidf_vec = build_tfidf_vectorizer(train_df)
    joblib.dump(tfidf_vec, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    print(f"  TF-IDF vocabulary size: {len(tfidf_vec.vocabulary_)}")

    # 4. Build OHE verification matrices
    MAX_TRAIN = 20000
    print(f"\n[4/6] Building OHE verification matrix (train, max={MAX_TRAIN})...")
    X_train, y_train = build_verification_matrix(
        train_df, ohe_vec, max_rows=MAX_TRAIN, split_name="train"
    )

    print("\n[5/6] Building OHE verification matrix (val, max=3000)...")
    X_val, y_val = build_verification_matrix(
        val_df, ohe_vec, max_rows=3000, split_name="val"
    )

    # 5. Save
    print("\n[6/6] Saving feature matrices...")
    np.save(os.path.join(PROC_DIR, "X_train_ohe.npy"), X_train)
    np.save(os.path.join(PROC_DIR, "y_train.npy"),     y_train)
    np.save(os.path.join(PROC_DIR, "X_val_ohe.npy"),   X_val)
    np.save(os.path.join(PROC_DIR, "y_val.npy"),       y_val)

    # Keep legacy names so older scripts still work
    np.save(os.path.join(PROC_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROC_DIR, "X_val.npy"),   X_val)

    print("\n✓ Preprocessing complete. Files saved to data/processed/")
    print("  Run: python src/model_a_train.py")

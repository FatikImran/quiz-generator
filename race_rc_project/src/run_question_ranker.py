"""
Run only the Question Ranker training (Step 7 from model_a_train.py).
Usage: python src/run_question_ranker.py

This script expects:
- models/ohe_vectorizer.pkl to exist
- data/raw/train.csv available (or the load_race fallback will look for dev.csv)
"""
import os
import sys
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

# Ensure repo root is importable
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.preprocessing import load_race, clean_text, generate_candidate_questions
from tqdm import tqdm

MODELS_DIR = os.path.join('models', 'model_a')
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print('\n[Question Ranker] Loading OHE vectorizer...')
    ohe_path = os.path.join('models', 'ohe_vectorizer.pkl')
    if not os.path.exists(ohe_path):
        raise SystemExit(f"Missing OHE vectorizer: {ohe_path}")
    ohe_vec_loaded = joblib.load(ohe_path)

    print('[Question Ranker] Loading training data (train head 4000)...')
    train_qr_df = load_race('train').head(4000)

    Xq_rows, yq_rows = [], []
    print('  Building question ranker training data...')
    for _, row in tqdm(train_qr_df.iterrows(), total=len(train_qr_df), desc='  QR data'):
        article       = str(row['article'])
        correct       = str(row['answer']).strip().upper()
        real_question = str(row['question'])

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

    if not Xq_rows:
        print('No question-ranker training data generated — aborting.')
        return

    Xq = np.array(Xq_rows, dtype=np.float32)
    yq = np.array(yq_rows)
    scaler_qr = StandardScaler()
    Xq_sc     = scaler_qr.fit_transform(Xq)
    base_qr   = LinearSVC(C=1.0, max_iter=1000, class_weight='balanced', random_state=42)
    qr_model  = CalibratedClassifierCV(base_qr, cv=3)
    print('\n[Question Ranker] Training SVM ranker...')
    qr_model.fit(Xq_sc, yq)
    qr_bundle = {'model': qr_model, 'scaler': scaler_qr}

    out_path = os.path.join(MODELS_DIR, 'question_ranker.pkl')
    joblib.dump(qr_bundle, out_path)
    print(f'[Question Ranker] Saved → {out_path}')

if __name__ == '__main__':
    main()

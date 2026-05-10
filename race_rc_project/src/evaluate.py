"""
src/evaluate.py
Evaluation script using BLEU, ROUGE, and METEOR for text generation tasks,
plus ranking metrics for answer verification.

As per instructor's requirements (Further_Instructions_from_Instructor.txt):
  Primary metrics: BLEU, ROUGE, METEOR
  Secondary: MRR, ranking accuracy (for verification sub-task)

Run after training:
    python src/evaluate.py

Install requirements if missing:
    pip install rouge-score nltk
"""

import os
import re
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as _rouge_scorer
from tqdm import tqdm
from collections import defaultdict

nltk.download("wordnet", quiet=True)
nltk.download("punkt",   quiet=True)
nltk.download("omw-1.4", quiet=True)

from src.preprocessing import clean_text
from src.inference import (generate_distractors, generate_hints,
                           verify_answer, generate_question)

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Metric helpers ─────────────────────────────────────────────────────────────
def _tokenize(text: str) -> list:
    return clean_text(text).split()


def compute_bleu(reference: str, hypothesis: str,
                 weights=(0.25, 0.25, 0.25, 0.25)) -> float:
    ref_tok = [_tokenize(reference)]
    hyp_tok = _tokenize(hypothesis)
    return sentence_bleu(ref_tok, hyp_tok, weights=weights,
                         smoothing_function=SmoothingFunction().method4)


def compute_rouge(reference: str, hypothesis: str) -> dict:
    scorer = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                       use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {
        "rouge1_f": scores["rouge1"].fmeasure,
        "rouge2_f": scores["rouge2"].fmeasure,
        "rougeL_f": scores["rougeL"].fmeasure,
    }


def compute_meteor(reference: str, hypothesis: str) -> float:
    return meteor_score([_tokenize(reference)], _tokenize(hypothesis))


def load_eval_data(split="test", max_rows=None) -> pd.DataFrame:
    path = f"data/raw/{split}.csv"
    df   = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df.head(max_rows) if max_rows else df


# ── Eval 1: Distractor Generation ─────────────────────────────────────────────
def evaluate_distractors(test_df, num_samples=200):
    print("\n" + "=" * 60)
    print("EVAL 1: Distractor Generation (BLEU / ROUGE / METEOR)")
    print("=" * 60)
    sample = test_df.sample(min(num_samples, len(test_df)), random_state=42)

    bleu1s, bleu2s, bleu4s = [], [], []
    rouge_acc = defaultdict(list)
    meteors   = []

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        article       = str(row["article"])
        question      = str(row["question"])
        correct_label = str(row["answer"]).strip().upper()
        correct_text  = str(row[correct_label])
        actual_dist   = [str(row[o]) for o in "ABCD" if o != correct_label]

        generated = generate_distractors(article, question, correct_text, n=3)

        for gen in generated:
            b1 = max(compute_bleu(a, gen, (1,0,0,0)) for a in actual_dist)
            b2 = max(compute_bleu(a, gen, (0.5,0.5,0,0)) for a in actual_dist)
            b4 = max(compute_bleu(a, gen) for a in actual_dist)
            m  = max(compute_meteor(a, gen) for a in actual_dist)
            r  = {k: max(compute_rouge(a, gen)[k] for a in actual_dist)
                  for k in ("rouge1_f","rouge2_f","rougeL_f")}
            bleu1s.append(b1); bleu2s.append(b2); bleu4s.append(b4)
            meteors.append(m)
            for k, v in r.items():
                rouge_acc[k].append(v)

    results = {
        "distractor_evaluation": {
            "bleu1":    float(np.mean(bleu1s)),
            "bleu2":    float(np.mean(bleu2s)),
            "bleu4":    float(np.mean(bleu4s)),
            "meteor":   float(np.mean(meteors)),
            "rouge1_f": float(np.mean(rouge_acc["rouge1_f"])),
            "rouge2_f": float(np.mean(rouge_acc["rouge2_f"])),
            "rougeL_f": float(np.mean(rouge_acc["rougeL_f"])),
            "num_samples": len(sample),
        }
    }
    d = results["distractor_evaluation"]
    print(f"  BLEU-1: {d['bleu1']:.4f}  BLEU-2: {d['bleu2']:.4f}  "
          f"BLEU-4: {d['bleu4']:.4f}")
    print(f"  METEOR: {d['meteor']:.4f}")
    print(f"  ROUGE-1: {d['rouge1_f']:.4f}  ROUGE-2: {d['rouge2_f']:.4f}  "
          f"ROUGE-L: {d['rougeL_f']:.4f}")
    return results


# ── Eval 2: Hint Generation ────────────────────────────────────────────────────
def evaluate_hints(test_df, num_samples=200):
    print("\n" + "=" * 60)
    print("EVAL 2: Hint Generation (BLEU / ROUGE / METEOR)")
    print("=" * 60)
    sample = test_df.sample(min(num_samples, len(test_df)), random_state=42)

    bleu1s, bleu4s = [], []
    rouge_acc = defaultdict(list)
    meteors   = []

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        article       = str(row["article"])
        question      = str(row["question"])
        correct_label = str(row["answer"]).strip().upper()
        correct_text  = str(row[correct_label])

        sentences     = re.split(r"(?<=[.!?])\s+", article.strip())
        ref_sent      = next(
            (s for s in sentences if correct_text.lower() in s.lower()),
            sentences[0] if sentences else ""
        )
        if not ref_sent:
            continue

        generated_hints = generate_hints(article, question, correct_text)

        for hint in generated_hints:
            bleu1s.append(compute_bleu(ref_sent, hint, (1,0,0,0)))
            bleu4s.append(compute_bleu(ref_sent, hint))
            meteors.append(compute_meteor(ref_sent, hint))
            for k, v in compute_rouge(ref_sent, hint).items():
                rouge_acc[k].append(v)

    results = {
        "hint_evaluation": {
            "bleu1":    float(np.mean(bleu1s)) if bleu1s else 0,
            "bleu4":    float(np.mean(bleu4s)) if bleu4s else 0,
            "meteor":   float(np.mean(meteors)) if meteors else 0,
            "rouge1_f": float(np.mean(rouge_acc["rouge1_f"])) if rouge_acc else 0,
            "rouge2_f": float(np.mean(rouge_acc["rouge2_f"])) if rouge_acc else 0,
            "rougeL_f": float(np.mean(rouge_acc["rougeL_f"])) if rouge_acc else 0,
        }
    }
    h = results["hint_evaluation"]
    print(f"  BLEU-1: {h['bleu1']:.4f}  BLEU-4: {h['bleu4']:.4f}  "
          f"METEOR: {h['meteor']:.4f}")
    print(f"  ROUGE-1: {h['rouge1_f']:.4f}  ROUGE-L: {h['rougeL_f']:.4f}")
    return results


# ── Eval 3: Generated Question Quality ────────────────────────────────────────
def evaluate_question_generation(test_df, num_samples=200):
    """
    Compares AI-generated question against RACE reference question using
    BLEU, ROUGE, METEOR. This directly evaluates the Model A generation sub-task.
    """
    print("\n" + "=" * 60)
    print("EVAL 3: Question Generation (BLEU / ROUGE / METEOR)")
    print("=" * 60)
    sample = test_df.sample(min(num_samples, len(test_df)), random_state=42)

    bleu1s, bleu4s, meteors = [], [], []
    rouge_acc = defaultdict(list)

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        article   = str(row["article"])
        reference = str(row["question"])

        try:
            gen_result = generate_question(article)
            hypothesis = gen_result["question"]
        except Exception:
            continue

        bleu1s.append(compute_bleu(reference, hypothesis, (1,0,0,0)))
        bleu4s.append(compute_bleu(reference, hypothesis))
        meteors.append(compute_meteor(reference, hypothesis))
        for k, v in compute_rouge(reference, hypothesis).items():
            rouge_acc[k].append(v)

    results = {
        "question_generation_evaluation": {
            "bleu1":    float(np.mean(bleu1s)) if bleu1s else 0,
            "bleu4":    float(np.mean(bleu4s)) if bleu4s else 0,
            "meteor":   float(np.mean(meteors)) if meteors else 0,
            "rouge1_f": float(np.mean(rouge_acc["rouge1_f"])) if rouge_acc else 0,
            "rouge2_f": float(np.mean(rouge_acc["rouge2_f"])) if rouge_acc else 0,
            "rougeL_f": float(np.mean(rouge_acc["rougeL_f"])) if rouge_acc else 0,
            "num_samples": len(sample),
            "note": ("Template + OHE-ranked question generation. "
                     "Lower scores expected vs neural methods."),
        }
    }
    q = results["question_generation_evaluation"]
    print(f"  BLEU-1: {q['bleu1']:.4f}  BLEU-4: {q['bleu4']:.4f}  "
          f"METEOR: {q['meteor']:.4f}")
    print(f"  ROUGE-1: {q['rouge1_f']:.4f}  ROUGE-L: {q['rougeL_f']:.4f}")
    return results


# ── Eval 4: Answer Verification as Ranking ────────────────────────────────────
def evaluate_answer_verification_ranking(test_df, num_samples=500):
    """
    Evaluate Model A's verification as a ranking task.
    MRR, Top-1 accuracy — framed as ranking rather than classification
    (no accuracy/F1 per instructor note, but MRR is generation-metric-aligned).
    """
    print("\n" + "=" * 60)
    print("EVAL 4: Answer Verification Ranking (MRR / BLEU between correct & top)")
    print("=" * 60)
    sample = test_df.sample(min(num_samples, len(test_df)), random_state=42)

    correct_ranks = []
    bleu_correct_vs_top = []
    options_keys = ["A", "B", "C", "D"]

    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        article       = str(row["article"])
        question      = str(row["question"])
        opts_dict     = {k: str(row[k]) for k in options_keys}
        correct_label = str(row["answer"]).strip().upper()
        correct_text  = opts_dict[correct_label]

        result = verify_answer(article, question, opts_dict)
        probs  = result["probabilities"]
        ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

        for rank, (label, _) in enumerate(ranked, 1):
            if label == correct_label:
                correct_ranks.append(rank)
                break

        top_label = ranked[0][0]
        if top_label != correct_label:
            bleu_correct_vs_top.append(
                compute_bleu(correct_text, opts_dict[top_label], (0.5, 0.5, 0, 0))
            )

    mrr = float(np.mean([1.0 / r for r in correct_ranks])) if correct_ranks else 0
    results = {
        "answer_verification_ranking": {
            "MRR":            mrr,
            "mean_rank":      float(np.mean(correct_ranks)) if correct_ranks else 0,
            "top1_accuracy":  float(np.mean([1 if r == 1 else 0
                                             for r in correct_ranks])),
            "top2_accuracy":  float(np.mean([1 if r <= 2 else 0
                                             for r in correct_ranks])),
            "avg_bleu_correct_vs_top_distractor":
                float(np.mean(bleu_correct_vs_top)) if bleu_correct_vs_top else 0,
            "num_samples":    len(sample),
        }
    }
    v = results["answer_verification_ranking"]
    print(f"  MRR: {v['MRR']:.4f}  Top-1: {v['top1_accuracy']:.4f}  "
          f"Top-2: {v['top2_accuracy']:.4f}")
    return results


# ── Eval 5: Distractor Diversity ──────────────────────────────────────────────
def evaluate_distractor_diversity(test_df, num_samples=200):
    print("\n" + "=" * 60)
    print("EVAL 5: Distractor Diversity (intra-distractor BLEU, lower = better)")
    print("=" * 60)
    sample = test_df.sample(min(num_samples, len(test_df)), random_state=42)

    intra_bleus = []
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        article       = str(row["article"])
        question      = str(row["question"])
        correct_label = str(row["answer"]).strip().upper()
        correct_text  = str(row[correct_label])

        generated = generate_distractors(article, question, correct_text, n=3)
        if len(generated) >= 2:
            pairs = [compute_bleu(generated[i], generated[j], (0.5,0.5,0,0))
                     for i in range(len(generated))
                     for j in range(i+1, len(generated))]
            if pairs:
                intra_bleus.append(float(np.mean(pairs)))

    avg_intra = float(np.mean(intra_bleus)) if intra_bleus else 0
    results = {
        "distractor_diversity": {
            "avg_intra_distractor_bleu": avg_intra,
            "interpretation": "Lower BLEU = more diverse. Target: < 0.3",
            "diversity_ok":   avg_intra < 0.3,
            "num_samples":    len(sample),
        }
    }
    print(f"  Avg intra-distractor BLEU: {avg_intra:.4f}  "
          f"({'✅ Diverse' if avg_intra < 0.3 else '⚠️ Check diversity'})")
    return results


# ── Eval 6: Hint Graduation ───────────────────────────────────────────────────
def evaluate_hint_graduation(test_df, num_samples=200):
    print("\n" + "=" * 60)
    print("EVAL 6: Hint Graduation — Hint 3 should be closest to answer")
    print("=" * 60)
    sample = test_df.sample(min(num_samples, len(test_df)), random_state=42)

    h1s, h2s, h3s = [], [], []
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        article       = str(row["article"])
        question      = str(row["question"])
        correct_label = str(row["answer"]).strip().upper()
        correct_text  = str(row[correct_label])

        sentences = re.split(r"(?<=[.!?])\s+", article.strip())
        ref = next((s for s in sentences
                    if correct_text.lower() in s.lower()), "")
        if not ref:
            continue

        hints = generate_hints(article, question, correct_text)
        if len(hints) >= 3:
            h1s.append(compute_bleu(ref, hints[0], (0.5, 0.5, 0, 0)))
            h2s.append(compute_bleu(ref, hints[1], (0.5, 0.5, 0, 0)))
            h3s.append(compute_bleu(ref, hints[2], (0.5, 0.5, 0, 0)))

    grad_rate = float(np.mean([1 if h3 > h2 > h1 else 0
                               for h1, h2, h3 in zip(h1s, h2s, h3s)])) if h1s else 0
    results = {
        "hint_graduation": {
            "hint1_avg_bleu": float(np.mean(h1s)) if h1s else 0,
            "hint2_avg_bleu": float(np.mean(h2s)) if h2s else 0,
            "hint3_avg_bleu": float(np.mean(h3s)) if h3s else 0,
            "graduation_success_rate": grad_rate,
            "num_samples": len(sample),
        }
    }
    g = results["hint_graduation"]
    print(f"  Hint1: {g['hint1_avg_bleu']:.4f}  Hint2: {g['hint2_avg_bleu']:.4f}  "
          f"Hint3: {g['hint3_avg_bleu']:.4f}")
    print(f"  Graduation success rate (H3>H2>H1): {g['graduation_success_rate']:.4f}")
    return results


# ── Main runner ────────────────────────────────────────────────────────────────
def run_full_evaluation(num_samples=200, num_ranking=500):
    print("\n" + "=" * 70)
    print("RACE SYSTEM EVALUATION — BLEU / ROUGE / METEOR")
    print("=" * 70)

    test_df = load_eval_data("test")
    print(f"Loaded {len(test_df)} test samples")

    all_results = {}
    all_results.update(evaluate_distractors(test_df, num_samples))
    all_results.update(evaluate_hints(test_df, num_samples))
    all_results.update(evaluate_question_generation(test_df, num_samples))
    all_results.update(evaluate_answer_verification_ranking(test_df, num_ranking))
    all_results.update(evaluate_distractor_diversity(test_df, num_samples))
    all_results.update(evaluate_hint_graduation(test_df, num_samples))

    all_results["meta"] = {
        "primary_feature_method": "One-Hot Encoding (CountVectorizer binary=True)",
        "tfidf_role":             "Optional supplement — not primary",
        "evaluation_metrics":     "BLEU, ROUGE, METEOR (generation tasks); "
                                  "MRR (ranking task)",
        "as_per_instructor":      True,
        "note": ("Accuracy/F1 reported in training scripts for verification "
                 "sub-task; generation tasks use BLEU/ROUGE/METEOR as instructed."),
    }

    out = os.path.join(MODELS_DIR, "evaluation_report.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✓ Full report saved to: {out}")
    return all_results


if __name__ == "__main__":
    try:
        from rouge_score import rouge_scorer as _check
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "rouge-score", "--quiet"])

    run_full_evaluation()

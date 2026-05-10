"""
train_all.py
Single-command training pipeline. Run this once after placing RACE CSVs in data/raw/.
Usage: python src/train_all.py
"""
import os
import sys
import subprocess
import time

def run_step(name, script):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    # Run in same Python process by importing
    result = subprocess.run([sys.executable, script], check=True)
    elapsed = time.time() - t0
    print(f"\n  ✓ Completed in {elapsed:.1f}s")
    return result

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  RACE RC PROJECT — FULL TRAINING PIPELINE")
    print("  This will take ~15-30 minutes on CPU")
    print("="*60)

    # Check data exists
    for split in ["train", "val", "test"]:
        path = f"data/raw/{split}.csv"
        if not os.path.exists(path):
            print(f"\n✗ Missing: {path}")
            print("  Download from: https://www.kaggle.com/datasets/ankitdhiman7/race-dataset")
            print("  Place train.csv, val.csv, test.csv in data/raw/")
            sys.exit(1)

    # Create __init__.py so src is importable as package
    open("src/__init__.py", "a").close()

    run_step("Step 1/3 — Preprocessing (TF-IDF + feature matrices)", "src/preprocessing.py")
    run_step("Step 2/3 — Model A Training (LR, SVM, RF, K-Means, Label Propagation)", "src/model_a_train.py")
    run_step("Step 3/3 — Model B Training (Distractor Ranker, Hint Scorer)", "src/model_b_train.py")
    run_step("Step 4/4 — Generation & Baseline Metrics Evaluation", "src/evaluate.py")

    print("\n" + "="*60)
    print("  ALL TRAINING COMPLETE!")
    print("  Launch app: streamlit run ui/app.py")
    print("="*60)

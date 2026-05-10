# notebooks/EDA.py  (also works as EDA.ipynb — paste cells into Jupyter)
# Run: python notebooks/EDA.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, re

os.makedirs("notebooks", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
train_df = pd.read_csv("data/raw/train.csv")
val_df   = pd.read_csv("data/raw/val.csv")
test_df  = pd.read_csv("data/raw/test.csv")

print("=" * 50)
print("RACE DATASET — EDA SUMMARY")
print("=" * 50)
print(f"Train: {len(train_df):,} rows")
print(f"Val:   {len(val_df):,} rows")
print(f"Test:  {len(test_df):,} rows")
print(f"Columns: {list(train_df.columns)}")

# ── Answer distribution ───────────────────────────────────────────────────────
print("\n--- Answer Distribution ---")
ans_counts = train_df["answer"].value_counts().sort_index()
print(ans_counts)
print("(Should be roughly balanced ~25% each)")

# ── Article length stats ──────────────────────────────────────────────────────
train_df["article_len"] = train_df["article"].apply(lambda x: len(str(x).split()))
train_df["question_len"] = train_df["question"].apply(lambda x: len(str(x).split()))
train_df["option_a_len"] = train_df["A"].apply(lambda x: len(str(x).split()))

print("\n--- Article Length (words) ---")
print(train_df["article_len"].describe().round(1))

print("\n--- Question Length (words) ---")
print(train_df["question_len"].describe().round(1))

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("RACE Dataset — EDA", fontsize=16, fontweight="bold")

# Answer balance
axes[0, 0].bar(ans_counts.index, ans_counts.values, color=["#1565c0","#0288d1","#29b6f6","#81d4fa"])
axes[0, 0].set_title("Answer Label Distribution")
axes[0, 0].set_xlabel("Answer")
axes[0, 0].set_ylabel("Count")
for i, v in enumerate(ans_counts.values):
    axes[0, 0].text(i, v + 50, f"{v/len(train_df)*100:.1f}%", ha="center")

# Article length histogram
axes[0, 1].hist(train_df["article_len"].clip(0, 1000), bins=50, color="#1565c0", alpha=0.7)
axes[0, 1].set_title("Article Length Distribution (words)")
axes[0, 1].set_xlabel("Word Count")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].axvline(train_df["article_len"].median(), color="red", linestyle="--",
                    label=f"Median={train_df['article_len'].median():.0f}")
axes[0, 1].legend()

# Question length histogram
axes[1, 0].hist(train_df["question_len"].clip(0, 50), bins=30, color="#0288d1", alpha=0.7)
axes[1, 0].set_title("Question Length Distribution (words)")
axes[1, 0].set_xlabel("Word Count")
axes[1, 0].set_ylabel("Frequency")

# Option lengths boxplot
option_lens = pd.DataFrame({
    "A": train_df["A"].apply(lambda x: len(str(x).split())),
    "B": train_df["B"].apply(lambda x: len(str(x).split())),
    "C": train_df["C"].apply(lambda x: len(str(x).split())),
    "D": train_df["D"].apply(lambda x: len(str(x).split())),
})
option_lens.boxplot(ax=axes[1, 1], patch_artist=True)
axes[1, 1].set_title("Option Length Distribution (words)")
axes[1, 1].set_ylabel("Word Count")

plt.tight_layout()
plt.savefig("notebooks/eda_plots.png", dpi=120, bbox_inches="tight")
plt.show()
print("\nEDA plots saved to notebooks/eda_plots.png")

# ── Sample rows ───────────────────────────────────────────────────────────────
print("\n--- Sample Row ---")
row = train_df.iloc[0]
print(f"Article:  {str(row['article'])[:200]}...")
print(f"Question: {row['question']}")
print(f"A: {row['A']}")
print(f"B: {row['B']}")
print(f"C: {row['C']}")
print(f"D: {row['D']}")
print(f"Answer:   {row['answer']}")

print("\n✓ EDA complete.")

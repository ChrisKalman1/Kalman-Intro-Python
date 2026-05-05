
"""
Sentiment analysis using Cardiff NLP's RoBERTa model:
    cardiffnlp/twitter-roberta-base-sentiment-latest

Model choice rationale
----------------------
This model is a RoBERTa-base transformer fine-tuned on ~124 million tweets
by the Cardiff NLP group (Barbieri et al., 2020; Loureiro et al., 2022).
Tweets share key properties with news headlines: both are short, punchy,
and sentiment-laden. The model outputs three classes (positive / neutral /
negative) matching VADER's classification scheme, enabling direct comparison.
It is widely used in peer-reviewed computational social science and is
publicly available via HuggingFace.

Limitations
-----------
- Not trained specifically on news headlines or homelessness discourse.
- CPU inference is slow; a GPU reduces runtime substantially.
- Titles longer than 128 tokens are truncated (rare for headlines).

Resume logic
------------
If a checkpoint file exists at data/processed/roberta_checkpoint.csv the
script will skip all titles already scored and resume from where it left off.
A checkpoint is saved every CHECKPOINT_EVERY batches.

Runtime estimate
----------------
~440 000 titles at BATCH_SIZE=32 on CPU: approximately 6-10 hours.
On a GPU this drops to under 1 hour.
"""

import os
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm

MODEL_NAME       = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE       = 32    # reduce to 8 if you hit memory errors on CPU
CHECKPOINT_EVERY = 500   # save progress every 500 batches (~16 000 titles)

CHECKPOINT_PATH  = "data/processed/roberta_checkpoint.csv"
CHECKPOINT_IDX   = "data/processed/roberta_checkpoint_idx.txt"

# Label map returned by the model → canonical names
LABEL_MAP = {
    "positive": "positive",
    "neutral":  "neutral",
    "negative": "negative",
    "LABEL_0":  "negative",
    "LABEL_1":  "neutral",
    "LABEL_2":  "positive",
}


# ── model ─────────────────────────────────────────────────────────────────────

def build_pipeline():
    print(f"  Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    device    = 0 if torch.cuda.is_available() else -1
    device_label = "GPU" if device == 0 else "CPU"
    print(f"  Running on: {device_label}")
    return pipeline(
        "text-classification",
        model=MODEL_NAME,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=128,
        top_k=None,       # return scores for all classes
    )


# ── scoring helpers ───────────────────────────────────────────────────────────

def _scores_to_row(result: list[dict]) -> dict:
    """
    Convert a list of {label, score} dicts to a flat row.
    compound = positive_prob - negative_prob  (range approximately -1 to +1)
    """
    row      = {LABEL_MAP.get(r["label"], r["label"]): r["score"] for r in result}
    compound = row.get("positive", 0.0) - row.get("negative", 0.0)
    dominant = max(result, key=lambda r: r["score"])["label"]
    sentiment = LABEL_MAP.get(dominant, dominant)
    return {
        "roberta_pos":       round(row.get("positive", 0.0), 6),
        "roberta_neu":       round(row.get("neutral",  0.0), 6),
        "roberta_neg":       round(row.get("negative", 0.0), 6),
        "roberta_compound":  round(compound, 6),
        "roberta_sentiment": sentiment,
    }


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> tuple[list[dict], int]:
    """
    Return (already_scored_rows, start_index).
    If no checkpoint exists, return ([], 0).
    """
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CHECKPOINT_IDX):
        rows      = pd.read_csv(CHECKPOINT_PATH).to_dict("records")
        with open(CHECKPOINT_IDX) as f:
            start = int(f.read().strip())
        print(f"  ↩  Resuming from title {start:,}  ({len(rows):,} already scored)")
        return rows, start
    return [], 0


def _save_checkpoint(rows: list[dict], next_index: int):
    """Persist scored rows and the index of the next title to process."""
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)
    with open(CHECKPOINT_IDX, "w") as f:
        f.write(str(next_index))


# ── main scorer ───────────────────────────────────────────────────────────────

def run_roberta(df: pd.DataFrame) -> pd.DataFrame:
    print("Running RoBERTa sentiment analysis...")
    df     = df.copy().reset_index(drop=True)
    clf    = build_pipeline()
    titles = df["title"].fillna("").tolist()
    total  = len(titles)

    # Resume from checkpoint if available
    results, start_idx = _load_checkpoint()

    batch_nums = range(start_idx, total, BATCH_SIZE)
    progress   = tqdm(
        batch_nums,
        desc="RoBERTa",
        unit="batch",
        initial=start_idx // BATCH_SIZE,
        total=(total + BATCH_SIZE - 1) // BATCH_SIZE,
    )

    try:
        for i in progress:
            batch = titles[i : i + BATCH_SIZE]
            batch = [t if t.strip() else "unknown" for t in batch]

            preds = clf(batch)
            results.extend([_scores_to_row(p) for p in preds])

            batch_num = i // BATCH_SIZE
            if batch_num > 0 and batch_num % CHECKPOINT_EVERY == 0:
                _save_checkpoint(results, i + BATCH_SIZE)
                tqdm.write(
                    f"  💾  Checkpoint saved — "
                    f"{i + BATCH_SIZE:,} / {total:,} titles complete"
                )

    except KeyboardInterrupt:
        tqdm.write("\n  ⚠️  Interrupted — saving checkpoint before exit...")
        _save_checkpoint(results, i)
        tqdm.write(f"  💾  Checkpoint saved at title {i:,}. "
                   f"Re-run to resume.")
        raise

    except Exception as exc:
        tqdm.write(f"\n  ❌  Error at title {i:,}: {exc}")
        tqdm.write("  Saving checkpoint before raising...")
        _save_checkpoint(results, i)
        raise

    # All titles scored — build output DataFrame
    score_df = pd.DataFrame(results, index=df.index)
    return pd.concat([df, score_df], axis=1)


# ── summariser ────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["country", "year"])
        .agg(
            avg_roberta_compound = ("roberta_compound",  "mean"),
            pct_negative         = ("roberta_sentiment", lambda x: (x == "negative").mean() * 100),
            pct_neutral          = ("roberta_sentiment", lambda x: (x == "neutral").mean()  * 100),
            pct_positive         = ("roberta_sentiment", lambda x: (x == "positive").mean() * 100),
            total_titles         = ("title",             "count"),
        )
        .reset_index()
    )


# ── pipeline entry point ──────────────────────────────────────────────────────

def run_roberta_pipeline(
    input_path   = "data/processed/vader_results.csv",
    output_path  = "data/processed/roberta_results.csv",
    summary_path = "data/processed/roberta_summary.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(input_path)
    df = run_roberta(df)

    # Clean up checkpoint files on successful completion
    for f in [CHECKPOINT_PATH, CHECKPOINT_IDX]:
        if os.path.exists(f):
            os.remove(f)
    print("  🗑   Checkpoint files removed after successful completion.")

    df.to_csv(output_path, index=False)
    print(f"  RoBERTa scores  → {output_path}")

    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    print(f"  RoBERTa summary → {summary_path}")
    return df, summary


if __name__ == "__main__":
    run_roberta_pipeline()

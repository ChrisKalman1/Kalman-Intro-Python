
"""
Framing analysis — zero-shot classification.

Model: facebook/bart-large-mnli
--------------------------------
BART-large fine-tuned on Multi-Genre Natural Language Inference (MNLI).
Zero-shot classification works by posing each headline as a premise and
each candidate label as a hypothesis ("This headline is about <label>"),
then using the model's entailment score as the probability that the label
applies.  multi_label=True means each title is scored against all 7 frames
independently — a title can match any number of frames simultaneously.

Advantages over keyword matching
---------------------------------
- Captures semantic meaning rather than surface vocabulary.
- No manually curated keyword lists required.
- Each label receives a continuous probability score.

Limitations
-----------
- Slower and more memory-intensive than keyword matching.
- Label wording choices influence scores.
- The 0.5 threshold for flagging a frame as present is a convention;
  sensitivity analysis is recommended.

Resume logic
------------
If a checkpoint exists at data/processed/zeroshot_checkpoint.csv the
script skips all titles already classified and resumes from that point.
A checkpoint is saved every CHECKPOINT_EVERY batches.

Runtime estimate
----------------
~440 000 titles at BATCH_SIZE=8 on CPU: approximately 12-20 hours.
On a GPU this drops to 2-3 hours.
"""

import os
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
BATCH_SIZE       = 8     # bart-large is memory-hungry; reduce to 4 if needed
CHECKPOINT_EVERY = 200   # save every 200 batches (~1 600 titles)
THRESHOLD        = 0.50  # minimum entailment score to flag a frame as present

CHECKPOINT_PATH  = "data/processed/zeroshot_checkpoint.csv"
CHECKPOINT_IDX   = "data/processed/zeroshot_checkpoint_idx.txt"

# Candidate labels presented to the model.
# Plain, self-explanatory wording helps the NLI model reason correctly.
FRAME_LABELS: dict[str, str] = {
    "criminalization":       "policing, arrests, or criminalization of homeless people",
    "humanization":          "personal stories or human experiences of homeless people",
    "policy_government":     "government policy, legislation, or political decisions about homelessness",
    "health_services":       "health care, mental health, addiction services, or shelters for homeless people",
    "crisis_alarm":          "a crisis, emergency, or alarming increase in homelessness",
    "dehumanizing_language": "language that dehumanizes or stigmatizes homeless people",
    "empathetic_language":   "empathetic or dignity-affirming language about homeless people",
}


# ── model ─────────────────────────────────────────────────────────────────────

def build_pipeline():
    print(f"  Loading model: {MODEL_NAME}")
    device = 0 if torch.cuda.is_available() else -1
    device_label = "GPU" if device == 0 else "CPU"
    print(f"  Running on: {device_label}")
    return pipeline(
        "zero-shot-classification",
        model=MODEL_NAME,
        device=device,
    )


# ── result helper ─────────────────────────────────────────────────────────────

def _result_to_row(result: dict) -> dict:
    """
    Convert zero-shot output to a flat row of:
      zs_<frame>_score  — raw entailment probability (0-1)
      frame_zs_<frame>  — binary flag (1 if score >= THRESHOLD)
    """
    label_to_key = {v: k for k, v in FRAME_LABELS.items()}
    row: dict = {}
    for label, score in zip(result["labels"], result["scores"]):
        key = label_to_key.get(label, label)
        row[f"zs_{key}_score"]  = round(score, 6)
        row[f"frame_zs_{key}"]  = int(score >= THRESHOLD)
    return row


# ── checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> tuple[list[dict], int]:
    """
    Return (already_classified_rows, start_index).
    If no checkpoint exists return ([], 0).
    """
    if os.path.exists(CHECKPOINT_PATH) and os.path.exists(CHECKPOINT_IDX):
        rows  = pd.read_csv(CHECKPOINT_PATH).to_dict("records")
        with open(CHECKPOINT_IDX) as f:
            start = int(f.read().strip())
        print(f"  ↩  Resuming from title {start:,}  ({len(rows):,} already classified)")
        return rows, start
    return [], 0


def _save_checkpoint(rows: list[dict], next_index: int):
    """Persist classified rows and the index of the next title to process."""
    os.makedirs("data/processed", exist_ok=True)
    pd.DataFrame(rows).to_csv(CHECKPOINT_PATH, index=False)
    with open(CHECKPOINT_IDX, "w") as f:
        f.write(str(next_index))


# ── main classifier ───────────────────────────────────────────────────────────

def run_zeroshot(df: pd.DataFrame) -> pd.DataFrame:
    print("Running zero-shot framing analysis...")
    df     = df.copy().reset_index(drop=True)
    clf    = build_pipeline()
    titles = df["title"].fillna("unknown").tolist()
    labels = list(FRAME_LABELS.values())
    total  = len(titles)

    # Resume from checkpoint if available
    rows, start_idx = _load_checkpoint()

    batch_nums = range(start_idx, total, BATCH_SIZE)
    progress   = tqdm(
        batch_nums,
        desc="Zero-shot",
        unit="batch",
        initial=start_idx // BATCH_SIZE,
        total=(total + BATCH_SIZE - 1) // BATCH_SIZE,
    )

    try:
        for i in progress:
            batch   = titles[i : i + BATCH_SIZE]
            results = clf(
                batch,
                candidate_labels=labels,
                multi_label=True,
            )
            # clf returns a single dict for batch size 1, list otherwise
            if isinstance(results, dict):
                results = [results]
            rows.extend([_result_to_row(r) for r in results])

            batch_num = i // BATCH_SIZE
            if batch_num > 0 and batch_num % CHECKPOINT_EVERY == 0:
                _save_checkpoint(rows, i + BATCH_SIZE)
                tqdm.write(
                    f"  💾  Checkpoint saved — "
                    f"{i + BATCH_SIZE:,} / {total:,} titles complete"
                )

    except KeyboardInterrupt:
        tqdm.write("\n  ⚠️  Interrupted — saving checkpoint before exit...")
        _save_checkpoint(rows, i)
        tqdm.write(f"  💾  Checkpoint saved at title {i:,}. "
                   f"Re-run to resume.")
        raise

    except Exception as exc:
        tqdm.write(f"\n  ❌  Error at title {i:,}: {exc}")
        tqdm.write("  Saving checkpoint before raising...")
        _save_checkpoint(rows, i)
        raise

    # All titles classified — build output DataFrame
    score_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, score_df], axis=1)


# ── summariser ────────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    frame_cols = [f"frame_zs_{k}" for k in FRAME_LABELS]
    score_cols = [f"zs_{k}_score" for k in FRAME_LABELS]
    keep       = [c for c in frame_cols + score_cols if c in df.columns]

    summary = (
        df.groupby(["country", "year"])[keep + ["title"]]
        .agg({**{col: "mean" for col in keep}, "title": "count"})
        .reset_index()
    )
    for col in frame_cols:
        if col in summary.columns:
            summary[col] = summary[col] * 100

    return summary.rename(columns={"title": "total_titles"})


# ── pipeline entry point ──────────────────────────────────────────────────────

def run_zeroshot_pipeline(
    input_path = "data/clean/all_countries_clean.csv",
    output_path  = "data/processed/zeroshot_framing_results.csv",
    summary_path = "data/processed/zeroshot_framing_summary.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(input_path)
    df = run_zeroshot(df)

    # Clean up checkpoint files on successful completion
    for f in [CHECKPOINT_PATH, CHECKPOINT_IDX]:
        if os.path.exists(f):
            os.remove(f)
    print("  🗑   Checkpoint files removed after successful completion.")

    df.to_csv(output_path, index=False)
    print(f"  Zero-shot framing results  → {output_path}")

    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    print(f"  Zero-shot framing summary  → {summary_path}")
    return df, summary


if __name__ == "__main__":
    run_zeroshot_pipeline()


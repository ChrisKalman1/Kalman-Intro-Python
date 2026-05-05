
"""
Master pipeline.

Steps
-----
1. Fetch titles   (fetch_titles.py)        → data/raw/
2. VADER          (sentiment_vader.py)     → data/processed/vader_*
3. RoBERTa        (sentiment_roberta.py)   → data/processed/roberta_*
4. Keyword frames (framing_keyword.py)     → data/processed/keyword_framing_*
5. Zero-shot      (framing_zeroshot.py)    → data/processed/zeroshot_framing_*
6. Visualise      (visualize.py)           → data/figures/

Each step reads from / writes to well-defined file paths so any individual
step can be re-run in isolation without repeating earlier steps.

Usage
-----
    python -m src.run_pipeline              # run all steps
    python -m src.run_pipeline --from 3    # resume from step 3
"""

import argparse
import os
import sys

RAW_CSV       = "data/raw/all_countries_raw.csv"
VADER_CSV     = "data/processed/vader_results.csv"
ROBERTA_CSV   = "data/processed/roberta_results.csv"
KW_FRAME_CSV  = "data/processed/keyword_framing_results.csv"
ZS_FRAME_CSV  = "data/processed/zeroshot_framing_results.csv"


def step1_fetch():
    print("\n" + "="*60)
    print("STEP 1: Fetching titles from Media Cloud")
    print("="*60)
    from src.fetch_titles import fetch_all
    fetch_all()


def step2_vader():
    print("\n" + "="*60)
    print("STEP 2: VADER sentiment analysis")
    print("="*60)
    from src.sentiment_vader import run_vader_pipeline
    run_vader_pipeline(input_path=RAW_CSV)


def step3_roberta():
    print("\n" + "="*60)
    print("STEP 3: RoBERTa sentiment analysis")
    print("="*60)
    from src.sentiment_roberta import run_roberta_pipeline
    run_roberta_pipeline(input_path=VADER_CSV)


def step4_keyword():
    print("\n" + "="*60)
    print("STEP 4: Keyword framing analysis")
    print("="*60)
    from src.framing_keyword import run_keyword_pipeline
    run_keyword_pipeline(input_path=RAW_CSV)


def step5_zeroshot():
    print("\n" + "="*60)
    print("STEP 5: Zero-shot framing analysis")
    print("="*60)
    from src.framing_zeroshot import run_zeroshot_pipeline
    run_zeroshot_pipeline(input_path=RAW_CSV)


def step6_visualize():
    print("\n" + "="*60)
    print("STEP 6: Visualisation")
    print("="*60)
    from src.visualize import run_all
    run_all()


STEPS = {
    1: step1_fetch,
    2: step2_vader,
    3: step3_roberta,
    4: step4_keyword,
    5: step5_zeroshot,
    6: step6_visualize,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--from", dest="from_step", type=int, default=1,
        help="Start from this step number (1-6). Default: 1"
    )
    parser.add_argument(
        "--only", dest="only_step", type=int, default=None,
        help="Run only this step number."
    )
    args = parser.parse_args()

    os.makedirs("data/raw",       exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/figures",   exist_ok=True)
    os.makedirs("logs",           exist_ok=True)

    if args.only_step:
        steps_to_run = [args.only_step]
    else:
        steps_to_run = list(range(args.from_step, 7))

    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"Unknown step {step_num}. Valid steps: 1-6.")
            sys.exit(1)
        STEPS[step_num]()

    print("\n✅  Pipeline complete.")


if __name__ == "__main__":
    main()

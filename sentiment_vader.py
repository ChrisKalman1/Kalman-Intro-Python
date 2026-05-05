
"""
Sentiment analysis using VADER (Valence Aware Dictionary and sEntiment Reasoner).

VADER is a lexicon- and rule-based tool designed for short, informal text.
It returns a compound score in [-1.0, +1.0].

Classification thresholds (standard VADER convention):
    positive : compound >= +0.05
    neutral  : -0.05 < compound < +0.05
    negative : compound <= -0.05
"""

import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def score_title(title: str) -> float | None:
    if not isinstance(title, str) or not title.strip():
        return None
    return analyzer.polarity_scores(title)["compound"]


def classify(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def run_vader(df: pd.DataFrame) -> pd.DataFrame:
    print("Running VADER sentiment analysis...")
    df = df.copy()
    df["vader_score"]     = df["title"].apply(score_title)
    df["vader_sentiment"] = df["vader_score"].apply(classify)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["country", "year"])
        .agg(
            avg_vader_score = ("vader_score",     "mean"),
            pct_negative    = ("vader_sentiment", lambda x: (x == "negative").mean() * 100),
            pct_neutral     = ("vader_sentiment", lambda x: (x == "neutral").mean()  * 100),
            pct_positive    = ("vader_sentiment", lambda x: (x == "positive").mean() * 100),
            total_titles    = ("title",           "count"),
        )
        .reset_index()
    )


def run_vader_pipeline(
    input_path   = "data/clean/all_countries_clean.csv",
    output_path  = "data/processed/vader_results.csv",
    summary_path = "data/processed/vader_summary.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(input_path)
    df = run_vader(df)
    df.to_csv(output_path, index=False)
    print(f"  VADER scores  → {output_path}")

    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    print(f"  VADER summary → {summary_path}")
    return df, summary


if __name__ == "__main__":
    run_vader_pipeline()

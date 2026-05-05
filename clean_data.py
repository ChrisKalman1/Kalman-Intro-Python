"""
Clean and deduplicate the raw titles CSV.
Works on title-only data (no full article text).
"""

import pandas as pd
import re
import os

RAW_FILE     = "data/raw/all_countries_raw.csv"
CLEANED_FILE = "data/clean/all_countries_clean.csv"


def clean_title(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)   # normalize whitespace
    return text


def load_and_clean(filepath=RAW_FILE):
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows.")

    # Drop duplicate URLs
    df = df.drop_duplicates(subset="url")
    print(f"After URL deduplication: {len(df):,} rows.")

    # Drop duplicate titles within the same country
    df = df.drop_duplicates(subset=["country", "title"])
    print(f"After title deduplication: {len(df):,} rows.")

    # Drop rows with no title
    df = df[df["title"].notna() & (df["title"].str.strip() != "")]
    print(f"After dropping empty titles: {len(df):,} rows.")

    # Clean title
    df["title"] = df["title"].apply(clean_title)

    # Ensure year is integer
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # Parse publish_date
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

    # Sort
    df = df.sort_values(["country", "year"]).reset_index(drop=True)

    return df


if __name__ == "__main__":
    os.makedirs("data/clean", exist_ok=True)
    df = load_and_clean()
    df.to_csv(CLEANED_FILE, index=False)
    print(f"Cleaned data saved to {CLEANED_FILE} ({len(df):,} rows).")
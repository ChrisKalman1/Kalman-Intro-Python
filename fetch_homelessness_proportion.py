# src/fetch_homelessness_proportion.py

"""
Queries Media Cloud for:
  1. Total article counts per country per year (no keyword filter)
  2. Homelessness article counts per country per year (using QUERY)

Saves:
  data/processed/homelessness_proportion.csv
  data/figures/18_homelessness_article_pct.png
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import mediacloud.api as mc

from src.config import API_KEY, COLLECTIONS, QUERY, START_YEAR, END_YEAR
from datetime import date

mc_search = mc.SearchApi(API_KEY)

# ── constants ─────────────────────────────────────────────────────────────────
COUNTRY_ORDER  = ["US", "UK", "Canada", "Australia"]
COUNTRY_COLORS = {
    "US":        "#1f77b4",
    "UK":        "#2ca02c",
    "Canada":    "#d62728",
    "Australia": "#ff7f0e",
}
YEARS = list(range(START_YEAR, END_YEAR + 1))


# ── helpers ───────────────────────────────────────────────────────────────────
def _count_year(query: str, collection_id: int, year: int) -> int:
    """Sum monthly counts from story_count_over_time for a full calendar year."""
    start = date(year, 1, 1)
    end   = date(year, 12, 31)
    for attempt in range(4):
        try:
            counts = mc_search.story_count_over_time(
                query, start, end,
                collection_ids=[collection_id]
            )
            return sum(entry["count"] for entry in counts)
        except Exception as e:
            wait = 30 * (attempt + 1)
            print(f"    ⚠️  Attempt {attempt+1} failed: {e} — waiting {wait}s")
            time.sleep(wait)
    print(f"    ERROR: returning 0 for collection {collection_id}, {year}")
    return 0


# ── fetch ─────────────────────────────────────────────────────────────────────
def fetch_proportions() -> pd.DataFrame:
    # Check if already cached
    out_path = "data/processed/homelessness_proportion.csv"
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        done = set(zip(existing["country"], existing["year"]))
    else:
        existing = pd.DataFrame()
        done = set()

    rows = []
    for country_label, collection_id in COLLECTIONS.items():
        base_country = country_label.split(" - ")[0]
        print(f"\n{'='*55}\n{country_label}\n{'='*55}")

        for year in YEARS:
            if (country_label, year) in done:
                print(f"  {year} — already cached, skipping")
                continue

            print(f"  {year} — fetching total...", end=" ", flush=True)
            total = _count_year("*", collection_id, year)
            print(f"{total:,}  |  fetching homelessness...", end=" ", flush=True)
            homeless = _count_year(QUERY, collection_id, year)
            pct = (homeless / total * 100) if total > 0 else None
            print(f"{homeless:,}  →  {f'{pct:.4f}%' if pct is not None else 'N/A'}")

            rows.append({
                "country":           country_label,
                "base_country":      base_country,
                "year":              year,
                "total_articles":    total,
                "homeless_articles": homeless,
                "pct_homelessness":  pct,
            })

            # Save incrementally after each year in case of failure
            df_so_far = pd.concat(
                [existing, pd.DataFrame(rows)], ignore_index=True
            ) if not existing.empty else pd.DataFrame(rows)
            df_so_far.to_csv(out_path, index=False)

            time.sleep(8)  # polite gap between calls

    # Final combined save
    if rows:
        df = pd.concat(
            [existing, pd.DataFrame(rows)], ignore_index=True
        ) if not existing.empty else pd.DataFrame(rows)
    else:
        df = existing

    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    return df


# ── plot ──────────────────────────────────────────────────────────────────────
def plot_proportion(df: pd.DataFrame):
    os.makedirs("data/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for country in COUNTRY_ORDER:
        sub = df[df["base_country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["pct_homelessness"],
                marker="o", label=country,
                color=COUNTRY_COLORS[country], linewidth=2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=3))
    ax.set_title(
        "Homelessness Articles as % of All Published Articles\n"
        "National Press (2014–2024)",
        fontsize=13
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("% of All Articles")
    ax.set_xticks(YEARS)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Country")
    fig.tight_layout()
    fig.savefig("data/figures/18_homelessness_article_pct.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: data/figures/18_homelessness_article_pct.png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching story counts from Media Cloud...")
    df = fetch_proportions()
    print("\nGenerating figure 18...")
    plot_proportion(df)
    print("\nDone!")
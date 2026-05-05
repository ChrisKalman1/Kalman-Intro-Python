
"""
Framing analysis — keyword matching approach.

Each headline is checked against seven theoretically motivated framing
categories (see FRAMING_CATEGORIES below).  A title may match multiple
frames simultaneously (multi-label).  Results are expressed as the
percentage of titles per collection per year that matched each frame.

Theoretical grounding
---------------------
Frame categories draw on:
  - Entman (1993) framing theory
  - Schneider & Ingram (1993) social construction of target populations
  - Homeless media framing literature (e.g. Bunis et al. 1996,
    Shields et al. 2015, Phelan et al. 1997)

Limitations
-----------
- Keyword matching is sensitive to vocabulary coverage; terms not in the
  list will be missed (false negatives).
- Common words (e.g. 'police', 'rise') may match off-topic articles
  (false positives), though the upstream homelessness query filter
  substantially reduces this risk.
- No inter-rater reliability has been computed; treat findings as
  exploratory unless validated against hand-coded headlines.
"""

import pandas as pd
import os
import re

FRAMING_CATEGORIES: dict[str, list[str]] = {
    "criminalization": [
        "arrested", "arrest", "charged", "charges", "banned", "ban",
        "swept", "sweep", "cleared", "crackdown", "enforcement",
        "trespassing", "loitering", "encampment cleared", "removed",
        "evicted", "eviction", "crime", "criminal", "jail", "prison",
        "police", "law enforcement", "ordinance", "citation", "cited",
        "prosecuted", "prosecution", "anti-homeless", "sit-lie",
        "no-camping", "dispersed", "dispersal",
    ],
    "humanization": [
        "veteran", "family", "families", "mother", "father", "child",
        "children", "story", "portrait", "meet", "life", "lives",
        "person", "people", "community", "dignity", "rights", "voice",
        "human", "survivor", "named", "remembers", "remembering",
        "tribute", "obituary",
    ],
    "policy_government": [
        "council", "bill", "funding", "budget", "policy", "plan",
        "program", "initiative", "government", "mayor", "minister",
        "legislation", "vote", "approved", "proposal", "strategy",
        "reform", "affordable housing", "investment", "taskforce",
        "task force", "committee", "inquiry", "review", "audit",
        "spending", "allocated", "allocation",
    ],
    "health_services": [
        "shelter", "shelters", "mental health", "addiction", "services",
        "support", "clinic", "outreach", "treatment", "overdose",
        "substance", "healthcare", "food bank", "charity", "nonprofit",
        "volunteer", "aid", "assistance", "resource", "detox",
        "rehabilitation", "counselling", "counseling", "drop-in",
        "warming centre", "warming center", "night shelter",
    ],
    "crisis_alarm": [
        "surge", "spike", "crisis", "record", "rise", "rising",
        "growing", "epidemic", "soaring", "increase", "worst",
        "alarming", "emergency", "unprecedented", "hits record",
        "all-time high", "worsening", "exploding", "skyrocketing",
    ],
    "dehumanizing_language": [
        "the homeless",
        "homeless problem",
        "homeless issue",
        "blight",
        "nuisance",
        "eyesore",
        "vagrant",
        "vagrancy",
        "transient",
        "street people",
        "drifter",
        "derelict",
    ],
    "empathetic_language": [
        "people experiencing homelessness",
        "person experiencing homelessness",
        "unhoused people",
        "unhoused person",
        "unhoused individuals",
        "people without homes",
        "housing insecurity",
        "housing insecure",
        "lived experience",
        "dignity",
        "advocates say",
        "advocates call",
        "peer support",
        "housing first",
    ],
}


def tag_title(title: str) -> dict[str, int]:
    """Return {category: 0/1} for a single title."""
    if not isinstance(title, str):
        return {cat: 0 for cat in FRAMING_CATEGORIES}

    title_lower = title.lower()
    return {
        cat: int(
            any(
                re.search(r"\b" + re.escape(kw) + r"\b", title_lower)
                for kw in keywords
            )
        )
        for cat, keywords in FRAMING_CATEGORIES.items()
    }


def run_framing_keyword(df: pd.DataFrame) -> pd.DataFrame:
    print("Running keyword framing analysis...")
    df   = df.copy()
    tags = df["title"].apply(tag_title)
    for cat in FRAMING_CATEGORIES:
        df[f"frame_{cat}"] = tags.apply(lambda x: x[cat])
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    frame_cols = [f"frame_{cat}" for cat in FRAMING_CATEGORIES]
    summary = (
        df.groupby(["country", "year"])[frame_cols + ["title"]]
        .agg({**{col: "mean" for col in frame_cols}, "title": "count"})
        .reset_index()
    )
    for col in frame_cols:
        summary[col] = summary[col] * 100
    return summary.rename(columns={"title": "total_titles"})


def run_keyword_pipeline(
    input_path   = "data/clean/all_countries_clean.csv",
    output_path  = "data/processed/keyword_framing_results.csv",
    summary_path = "data/processed/keyword_framing_summary.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:

    os.makedirs("data/processed", exist_ok=True)
    df = pd.read_csv(input_path)
    df = run_framing_keyword(df)
    df.to_csv(output_path, index=False)
    print(f"  Keyword framing results  → {output_path}")

    summary = summarize(df)
    summary.to_csv(summary_path, index=False)
    print(f"  Keyword framing summary  → {summary_path}")
    return df, summary


if __name__ == "__main__":
    run_keyword_pipeline()

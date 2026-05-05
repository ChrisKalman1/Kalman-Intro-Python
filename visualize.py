"""
Visualisation module.

Figures produced
----------------
Sentiment
  1.  sentiment_vader_national.png        — VADER avg score, national press
  2.  sentiment_vader_levels.png          — VADER avg score, national vs state/local
  3.  sentiment_pct_negative.png          — % negative headlines, national press
  4.  sentiment_vader_vs_roberta.png      — VADER compound vs RoBERTa compound
  5.  sentiment_roberta_breakdown.png     — RoBERTa pos/neu/neg % over time

Framing — keyword
  6.  kw_criminalization_trend.png        — criminalization % over time
  7.  kw_framing_heatmap.png              — avg frame % heatmap, national
  8.  kw_stacked_frames.png               — stacked bar by country × year
  9.  kw_dehu_vs_empathy.png              — dehumanizing vs empathetic over time

Framing — zero-shot
  10. zs_framing_heatmap.png              — zero-shot avg frame % heatmap
  11. zs_stacked_frames.png               — zero-shot stacked bar by country × year
  12. zs_trends_per_country.png           — ZS frame trends over time per country
  13. zs_dehu_vs_empathy.png              — ZS dehumanizing vs empathetic over time

Method comparison
  14. zs_vs_kw_comparison.png             — keyword vs zero-shot heatmap side-by-side
  15. criminalization_method_comparison.png — KW vs ZS criminalization over time
  16. humanization_method_comparison.png    — KW vs ZS humanization over time

Sentiment × Framing
  17. sentiment_vs_criminalization.png    — scatter, sentiment vs crime frame

Coverage
  18. homelessness_article_pct.png        — % of all articles about homelessness
  19. homelessness_article_counts.png     — raw homelessness article counts over time
  20. total_vs_homeless_articles.png      — total vs homelessness articles per country
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import os

os.makedirs("data/figures", exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
COUNTRY_ORDER  = ["US", "UK", "Canada", "Australia"]
COUNTRY_COLORS = {
    "US":        "#1f77b4",
    "UK":        "#2ca02c",
    "Canada":    "#d62728",
    "Australia": "#ff7f0e",
}

FRAME_COLS = [
    "frame_criminalization",
    "frame_humanization",
    "frame_policy_government",
    "frame_health_services",
    "frame_crisis_alarm",
    "frame_dehumanizing_language",
    "frame_empathetic_language",
]
FRAME_LABELS = {
    "frame_criminalization":       "Criminalization",
    "frame_humanization":          "Humanization",
    "frame_policy_government":     "Policy / Govt",
    "frame_health_services":       "Health / Services",
    "frame_crisis_alarm":          "Crisis / Alarm",
    "frame_dehumanizing_language": "Dehumanizing Language",
    "frame_empathetic_language":   "Empathetic Language",
}
ZS_FRAME_COLS = [f"frame_zs_{k}" for k in [
    "criminalization", "humanization", "policy_government",
    "health_services", "crisis_alarm", "dehumanizing_language",
    "empathetic_language",
]]
ZS_FRAME_LABELS = {
    "frame_zs_criminalization":       "Criminalization",
    "frame_zs_humanization":          "Humanization",
    "frame_zs_policy_government":     "Policy / Govt",
    "frame_zs_health_services":       "Health / Services",
    "frame_zs_crisis_alarm":          "Crisis / Alarm",
    "frame_zs_dehumanizing_language": "Dehumanizing Language",
    "frame_zs_empathetic_language":   "Empathetic Language",
}
FRAME_COLORS = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3",
    "#ff7f00","#a65628","#f781bf"
]
YEARS = list(range(2014, 2025))


# ── helpers ───────────────────────────────────────────────────────────────────
def _add_meta(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["base_country"] = df["country"].str.extract(r"^(US|UK|Canada|Australia)")
    df["level"]        = df["country"].apply(
        lambda x: "National" if "National" in x else "State & Local"
    )
    return df


def _save(fig, name: str):
    path = f"data/figures/{name}"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {name}")


def _build_total_counts() -> pd.DataFrame:
    """Count all articles per country/year from raw file."""
    raw = pd.read_csv("data/raw/all_countries_raw.csv")
    raw["base_country"] = raw["country"].str.extract(r"^(US|UK|Canada|Australia)")
    raw["level"]        = raw["country"].apply(
        lambda x: "National" if "National" in x else "State & Local"
    )
    totals = (raw.groupby(["base_country", "level", "year"])
                 .size()
                 .reset_index(name="total_all_articles"))
    return totals


# ── 1. VADER sentiment — national ─────────────────────────────────────────────
def plot_vader_national(sent: pd.DataFrame):
    df  = sent[sent["level"] == "National"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for country in COUNTRY_ORDER:
        sub = df[df["base_country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["avg_vader_score"],
                marker="o", label=country,
                color=COUNTRY_COLORS[country], linewidth=2)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("VADER Sentiment — National Press (2014–2024)", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel("Avg VADER Compound Score")
    ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Country")
    _save(fig, "1_sentiment_vader_national.png")


# ── 2. VADER sentiment — levels ───────────────────────────────────────────────
def plot_vader_levels(sent: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for i, country in enumerate(COUNTRY_ORDER):
        ax = axes.flatten()[i]
        for level, ls in [("National", "-"), ("State & Local", "--")]:
            sub = sent[(sent["base_country"] == country) &
                       (sent["level"] == level)].sort_values("year")
            if sub.empty:
                continue
            ax.plot(sub["year"], sub["avg_vader_score"],
                    marker="o", label=level, linestyle=ls,
                    color=COUNTRY_COLORS[country], linewidth=2)
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(country, fontsize=12)
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year"); ax.set_ylabel("VADER Score")
        ax.legend(fontsize=8)
    fig.suptitle("VADER Sentiment by Coverage Level (2014–2024)", fontsize=14)
    _save(fig, "2_sentiment_vader_levels.png")


# ── 3. % negative ─────────────────────────────────────────────────────────────
def plot_pct_negative(sent: pd.DataFrame):
    df  = sent[sent["level"] == "National"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for country in COUNTRY_ORDER:
        sub = df[df["base_country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["pct_negative"],
                marker="o", label=country,
                color=COUNTRY_COLORS[country], linewidth=2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("% Negative Headlines — National Press (2014–2024)", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel("% Negative")
    ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Country")
    _save(fig, "3_sentiment_pct_negative.png")


# ── 4. VADER vs RoBERTa comparison ────────────────────────────────────────────
def plot_vader_vs_roberta(vader: pd.DataFrame, roberta: pd.DataFrame):
    df = pd.merge(
        vader[vader["level"] == "National"][
            ["base_country", "year", "avg_vader_score"]],
        roberta[roberta["level"] == "National"][
            ["base_country", "year", "avg_roberta_compound"]],
        on=["base_country", "year"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for i, country in enumerate(COUNTRY_ORDER):
        ax  = axes.flatten()[i]
        sub = df[df["base_country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["avg_vader_score"],
                marker="o", label="VADER", color="#1f77b4", linewidth=2)
        ax.plot(sub["year"], sub["avg_roberta_compound"],
                marker="s", label="RoBERTa", color="#d62728",
                linestyle="--", linewidth=2)
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(country, fontsize=12)
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year"); ax.set_ylabel("Compound Score")
        ax.legend(fontsize=8)
    fig.suptitle("VADER vs RoBERTa Sentiment — National Press (2014–2024)",
                 fontsize=14)
    _save(fig, "4_sentiment_vader_vs_roberta.png")


# ── 5. RoBERTa pos/neu/neg breakdown ─────────────────────────────────────────
def plot_roberta_breakdown(roberta: pd.DataFrame):
    df  = roberta[roberta["level"] == "National"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    for i, country in enumerate(COUNTRY_ORDER):
        ax  = axes.flatten()[i]
        sub = df[df["base_country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["pct_positive"], marker="o",
                label="Positive", color="#2ca02c", linewidth=2)
        ax.plot(sub["year"], sub["pct_neutral"],  marker="s",
                label="Neutral",  color="#7f7f7f", linewidth=2, linestyle="--")
        ax.plot(sub["year"], sub["pct_negative"], marker="^",
                label="Negative", color="#d62728", linewidth=2, linestyle=":")
        ax.set_title(country, fontsize=12)
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=8)
    fig.suptitle("RoBERTa Sentiment Breakdown — National Press (2014–2024)",
                 fontsize=14)
    _save(fig, "5_sentiment_roberta_breakdown.png")


# ── 6. Criminalization trend ──────────────────────────────────────────────────
def plot_criminalization(kw: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, level in zip(axes, ["National", "State & Local"]):
        df = kw[kw["level"] == level]
        for country in COUNTRY_ORDER:
            sub = df[df["base_country"] == country].sort_values("year")
            if sub.empty:
                continue
            ax.plot(sub["year"], sub["frame_criminalization"],
                    marker="o", label=country,
                    color=COUNTRY_COLORS[country], linewidth=2)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title(f"Criminalization Frame — {level}", fontsize=12)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Country", fontsize=8)
    fig.suptitle("Criminalization Framing of Homelessness (2014–2024)", fontsize=14)
    _save(fig, "6_kw_criminalization_trend.png")


# ── 7. Keyword framing heatmap ────────────────────────────────────────────────
def plot_kw_heatmap(kw: pd.DataFrame):
    df  = kw[kw["level"] == "National"].groupby("base_country")[FRAME_COLS].mean()
    df  = df.loc[[c for c in COUNTRY_ORDER if c in df.index]]
    df.columns = [FRAME_LABELS[c] for c in df.columns]
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "% of Titles"})
    ax.set_title("Keyword Framing — National Press avg 2014–2024", fontsize=13)
    _save(fig, "7_kw_framing_heatmap.png")


# ── 8. Stacked frame bar — keyword ────────────────────────────────────────────
def plot_kw_stacked(kw: pd.DataFrame):
    df = kw[kw["level"] == "National"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for i, country in enumerate(COUNTRY_ORDER):
        ax  = axes.flatten()[i]
        sub = (df[df["base_country"] == country]
               .sort_values("year").set_index("year"))
        vals = sub[FRAME_COLS].rename(columns=FRAME_LABELS)
        vals.plot(kind="bar", stacked=True, ax=ax,
                  color=FRAME_COLORS, legend=(i == 0), width=0.75)
        ax.set_title(country, fontsize=12)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.tick_params(axis="x", rotation=45)
        if i == 0:
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Keyword Frame Composition — National Press (2014–2024)", fontsize=14)
    _save(fig, "8_kw_stacked_frames.png")


# ── 9. Dehumanizing vs empathetic language — keyword ─────────────────────────
def plot_kw_dehu_vs_empathy(kw: pd.DataFrame):
    df  = kw[kw["level"] == "National"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for i, country in enumerate(COUNTRY_ORDER):
        ax  = axes.flatten()[i]
        sub = df[df["base_country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["frame_dehumanizing_language"],
                marker="o", label="Dehumanizing", color="#d62728", linewidth=2)
        ax.plot(sub["year"], sub["frame_empathetic_language"],
                marker="s", label="Empathetic",   color="#2ca02c",
                linewidth=2, linestyle="--")
        ax.set_title(country, fontsize=12)
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=8)
    fig.suptitle("Dehumanizing vs Empathetic Language — Keyword — National Press (2014–2024)",
                 fontsize=14)
    _save(fig, "9_kw_dehu_vs_empathy.png")


# ── 10. Zero-shot heatmap ─────────────────────────────────────────────────────
def plot_zs_heatmap(zs: pd.DataFrame):
    avail = [c for c in ZS_FRAME_COLS if c in zs.columns]
    df    = zs[zs["level"] == "National"].groupby("base_country")[avail].mean()
    df    = df.loc[[c for c in COUNTRY_ORDER if c in df.index]]
    df.columns = [ZS_FRAME_LABELS.get(c, c) for c in df.columns]
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "% of Titles"})
    ax.set_title("Zero-Shot Framing — National Press avg 2014–2024", fontsize=13)
    _save(fig, "10_zs_framing_heatmap.png")


# ── 11. Stacked frame bar — zero-shot ─────────────────────────────────────────
def plot_zs_stacked(zs: pd.DataFrame):
    avail = [c for c in ZS_FRAME_COLS if c in zs.columns]
    df    = zs[zs["level"] == "National"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for i, country in enumerate(COUNTRY_ORDER):
        ax  = axes.flatten()[i]
        sub = (df[df["base_country"] == country]
               .sort_values("year").set_index("year"))
        if sub.empty:
            continue
        vals = sub[avail].rename(columns=ZS_FRAME_LABELS)
        vals.plot(kind="bar", stacked=True, ax=ax,
                  color=FRAME_COLORS, legend=(i == 0), width=0.75)
        ax.set_title(country, fontsize=12)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.tick_params(axis="x", rotation=45)
        if i == 0:
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Zero-Shot Frame Composition — National Press (2014–2024)", fontsize=14)
    _save(fig, "11_zs_stacked_frames.png")


# ── 12. Zero-shot frame trends per country ────────────────────────────────────
def plot_zs_trends(zs: pd.DataFrame):
    avail  = [c for c in ZS_FRAME_COLS if c in zs.columns]
    df     = zs[zs["level"] == "National"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for i, country in enumerate(COUNTRY_ORDER):
        ax  = axes.flatten()[i]
        sub = df[df["base_country"] == country].sort_values("year")
        if sub.empty:
            continue
        for col, color in zip(avail, FRAME_COLORS):
            ax.plot(sub["year"], sub[col],
                    marker="o", label=ZS_FRAME_LABELS.get(col, col),
                    color=color, linewidth=2)
        ax.set_title(country, fontsize=12)
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        if i == 0:
            ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.suptitle("Zero-Shot Framing Trends — National Press (2014–2024)", fontsize=14)
    _save(fig, "12_zs_trends_per_country.png")


# ── 13. Zero-shot dehumanizing vs empathetic ──────────────────────────────────
def plot_zs_dehu_vs_empathy(zs: pd.DataFrame):
    df  = zs[zs["level"] == "National"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for i, country in enumerate(COUNTRY_ORDER):
        ax  = axes.flatten()[i]
        sub = df[df["base_country"] == country].sort_values("year")
        if sub.empty:
            continue
        ax.plot(sub["year"], sub["frame_zs_dehumanizing_language"],
                marker="o", label="Dehumanizing", color="#d62728", linewidth=2)
        ax.plot(sub["year"], sub["frame_zs_empathetic_language"],
                marker="s", label="Empathetic",   color="#2ca02c",
                linewidth=2, linestyle="--")
        ax.set_title(country, fontsize=12)
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=8)
    fig.suptitle("Dehumanizing vs Empathetic Language — Zero-Shot — National Press (2014–2024)",
                 fontsize=14)
    _save(fig, "13_zs_dehu_vs_empathy.png")


# ── 14. Keyword vs zero-shot heatmap side-by-side ─────────────────────────────
def plot_zs_vs_kw(kw: pd.DataFrame, zs: pd.DataFrame):
    kw_mean = (kw[kw["level"] == "National"]
               .groupby("base_country")[FRAME_COLS].mean()
               .loc[[c for c in COUNTRY_ORDER if c in
                     kw["base_country"].unique()]])
    avail   = [c for c in ZS_FRAME_COLS if c in zs.columns]
    zs_mean = (zs[zs["level"] == "National"]
               .groupby("base_country")[avail].mean()
               .loc[[c for c in COUNTRY_ORDER if c in
                     zs["base_country"].unique()]])

    kw_mean.columns = [FRAME_LABELS[c] for c in kw_mean.columns]
    zs_mean.columns = [ZS_FRAME_LABELS.get(c, c) for c in zs_mean.columns]

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    sns.heatmap(kw_mean, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[0],
                cbar_kws={"label": "% Titles"})
    axes[0].set_title("Keyword Matching", fontsize=12)

    sns.heatmap(zs_mean, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=axes[1],
                cbar_kws={"label": "% Titles"})
    axes[1].set_title("Zero-Shot Classification", fontsize=12)

    fig.suptitle("Framing Methods Comparison — National Press avg 2014–2024",
                 fontsize=14)
    _save(fig, "14_zs_vs_kw_comparison.png")


# ── 15. Criminalization: KW vs ZS over time ───────────────────────────────────
def plot_criminalization_method_comparison(kw: pd.DataFrame, zs: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for i, country in enumerate(COUNTRY_ORDER):
        ax   = axes.flatten()[i]
        kw_s = (kw[(kw["level"] == "National") & (kw["base_country"] == country)]
                .sort_values("year"))
        zs_s = (zs[(zs["level"] == "National") & (zs["base_country"] == country)]
                .sort_values("year"))
        ax.plot(kw_s["year"], kw_s["frame_criminalization"],
                marker="o", label="Keyword",    color="#1f77b4", linewidth=2)
        if "frame_zs_criminalization" in zs_s.columns:
            ax.plot(zs_s["year"], zs_s["frame_zs_criminalization"],
                    marker="s", label="Zero-Shot", color="#d62728",
                    linewidth=2, linestyle="--")
        ax.set_title(country, fontsize=12)
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=8)
    fig.suptitle("Criminalization Frame: Keyword vs Zero-Shot — National Press (2014–2024)",
                 fontsize=14)
    _save(fig, "15_criminalization_method_comparison.png")


# ── 16. Humanization: KW vs ZS over time ─────────────────────────────────────
def plot_humanization_method_comparison(kw: pd.DataFrame, zs: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for i, country in enumerate(COUNTRY_ORDER):
        ax   = axes.flatten()[i]
        kw_s = (kw[(kw["level"] == "National") & (kw["base_country"] == country)]
                .sort_values("year"))
        zs_s = (zs[(zs["level"] == "National") & (zs["base_country"] == country)]
                .sort_values("year"))
        ax.plot(kw_s["year"], kw_s["frame_humanization"],
                marker="o", label="Keyword",    color="#1f77b4", linewidth=2)
        if "frame_zs_humanization" in zs_s.columns:
            ax.plot(zs_s["year"], zs_s["frame_zs_humanization"],
                    marker="s", label="Zero-Shot", color="#d62728",
                    linewidth=2, linestyle="--")
        ax.set_title(country, fontsize=12)
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.legend(fontsize=8)
    fig.suptitle("Humanization Frame: Keyword vs Zero-Shot — National Press (2014–2024)",
                 fontsize=14)
    _save(fig, "16_humanization_method_comparison.png")


# ── 17. Sentiment vs criminalization scatter ──────────────────────────────────
def plot_sentiment_vs_crime(sent: pd.DataFrame, kw: pd.DataFrame):
    merged = pd.merge(
        sent[sent["level"] == "National"][
            ["base_country", "year", "avg_vader_score"]],
        kw[kw["level"] == "National"][
            ["base_country", "year", "frame_criminalization"]],
        on=["base_country", "year"],
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    for country in COUNTRY_ORDER:
        sub = merged[merged["base_country"] == country]
        ax.scatter(sub["frame_criminalization"], sub["avg_vader_score"],
                   label=country, color=COUNTRY_COLORS[country], s=80, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(str(int(row["year"])),
                        (row["frame_criminalization"], row["avg_vader_score"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Sentiment vs. Criminalization Frame — National Press", fontsize=13)
    ax.set_xlabel("% Titles with Criminalization Frame")
    ax.set_ylabel("Avg VADER Compound Score")
    ax.legend(title="Country")
    _save(fig, "17_sentiment_vs_criminalization.png")


# ── 18. % of all articles about homelessness ─────────────────────────────────
def plot_homelessness_pct(kw: pd.DataFrame, totals: pd.DataFrame):
    df = kw[kw["level"] == "National"][
        ["base_country", "year", "total_titles"]].copy()
    df = df.merge(
        totals[totals["level"] == "National"][
            ["base_country", "year", "total_all_articles"]],
        on=["base_country", "year"], how="left"
    )
    df["pct_homelessness"] = (df["total_titles"] / df["total_all_articles"]) * 100

    fig, ax = plt.subplots(figsize=(12, 5))
    for country in COUNTRY_ORDER:
        sub = df[df["base_country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["pct_homelessness"],
                marker="o", label=country,
                color=COUNTRY_COLORS[country], linewidth=2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=2))
    ax.set_title("Homelessness Articles as % of All Articles — National Press (2014–2024)",
                 fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of All Articles")
    ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Country")
    _save(fig, "18_homelessness_article_pct.png")


# ── 19. Raw homelessness article counts ───────────────────────────────────────
def plot_homelessness_counts(kw: pd.DataFrame):
    df  = kw[kw["level"] == "National"]
    fig, ax = plt.subplots(figsize=(12, 5))
    for country in COUNTRY_ORDER:
        sub = df[df["base_country"] == country].sort_values("year")
        ax.plot(sub["year"], sub["total_titles"],
                marker="o", label=country,
                color=COUNTRY_COLORS[country], linewidth=2)
    ax.set_title("Homelessness Article Count — National Press (2014–2024)", fontsize=13)
    ax.set_xlabel("Year"); ax.set_ylabel("Number of Articles")
    ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Country")
    _save(fig, "19_homelessness_article_counts.png")


# ── 20. Total vs homelessness articles per country ────────────────────────────
def plot_total_vs_homeless(kw: pd.DataFrame, totals: pd.DataFrame):
    df = kw[kw["level"] == "National"][
        ["base_country", "year", "total_titles"]].copy()
    df = df.merge(
        totals[totals["level"] == "National"][
            ["base_country", "year", "total_all_articles"]],
        on=["base_country", "year"], how="left"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=False)
    for i, country in enumerate(COUNTRY_ORDER):
        ax  = axes.flatten()[i]
        sub = df[df["base_country"] == country].sort_values("year")
        ax2 = ax.twinx()
        ax.bar(sub["year"], sub["total_all_articles"],
               color="lightsteelblue", label="All Articles", width=0.6, alpha=0.7)
        ax2.plot(sub["year"], sub["total_titles"],
                 marker="o", color="#d62728",
                 label="Homelessness Articles", linewidth=2)
        ax.set_title(country, fontsize=12)
        ax.set_xlabel("Year")
        ax.set_ylabel("Total Articles", color="steelblue")
        ax2.set_ylabel("Homelessness Articles", color="#d62728")
        ax.set_xticks(sub["year"]); ax.tick_params(axis="x", rotation=45)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")
    fig.suptitle("Total vs Homelessness Articles — National Press (2014–2024)",
                 fontsize=14)
    _save(fig, "20_total_vs_homeless_articles.png")


# ── master runner ─────────────────────────────────────────────────────────────
def run_all():
    print("Loading summary files...")

    vader_s   = _add_meta(pd.read_csv("data/processed/vader_summary.csv"))
    kw_s      = _add_meta(pd.read_csv("data/processed/keyword_framing_summary.csv"))
    totals    = _build_total_counts()

    roberta_path = "data/processed/roberta_summary.csv"
    zs_path      = "data/processed/zeroshot_framing_summary.csv"

    roberta_s = _add_meta(pd.read_csv(roberta_path)) \
                if os.path.exists(roberta_path) else None
    zs_s      = _add_meta(pd.read_csv(zs_path)) \
                if os.path.exists(zs_path) else None

    print("Generating figures...")
    plot_vader_national(vader_s)
    plot_vader_levels(vader_s)
    plot_pct_negative(vader_s)
    if roberta_s is not None:
        plot_vader_vs_roberta(vader_s, roberta_s)
        plot_roberta_breakdown(roberta_s)
    else:
        print("  Skipping figs 4-5 (RoBERTa not yet run)")
    plot_criminalization(kw_s)
    plot_kw_heatmap(kw_s)
    plot_kw_stacked(kw_s)
    plot_kw_dehu_vs_empathy(kw_s)
    if zs_s is not None:
        plot_zs_heatmap(zs_s)
        plot_zs_stacked(zs_s)
        plot_zs_trends(zs_s)
        plot_zs_dehu_vs_empathy(zs_s)
        plot_zs_vs_kw(kw_s, zs_s)
        plot_criminalization_method_comparison(kw_s, zs_s)
        plot_humanization_method_comparison(kw_s, zs_s)
    else:
        print("  Skipping figs 10-16 (zero-shot not yet run)")
    plot_sentiment_vs_crime(vader_s, kw_s)
    plot_homelessness_pct(kw_s, totals)
    plot_homelessness_counts(kw_s)
    plot_total_vs_homeless(kw_s, totals)

    print("\nAll figures saved to data/figures/")


if __name__ == "__main__":
    run_all()
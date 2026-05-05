
"""
Visualisation module.

Figures produced
----------------
Sentiment
  1. sentiment_vader_national.png       — VADER avg score, national press
  2. sentiment_vader_levels.png         — VADER avg score, national vs state/local
  3. sentiment_pct_negative.png         — % negative headlines, national press
  4. sentiment_vader_vs_roberta.png     — VADER compound vs RoBERTa compound

Framing — keyword
  5. kw_criminalization_trend.png       — criminalization % over time
  6. kw_framing_heatmap.png             — avg frame % heatmap, national
  7. kw_stacked_frames.png              — stacked bar by country × year

Framing — zero-shot
  8. zs_framing_heatmap.png             — zero-shot avg frame % heatmap
  9. zs_vs_kw_comparison.png            — keyword vs zero-shot side-by-side

Sentiment × Framing
  10. sentiment_vs_criminalization.png  — scatter, sentiment vs crime frame
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
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


# ── 5. Criminalization trend ──────────────────────────────────────────────────
def plot_criminalization(kw: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, level in zip(axes, ["National", "State & Local"]):
        df = kw[kw["level"] == level]
        for country in COUNTRY_ORDER:
            sub = df[df["base_country"] == country].sort_values("year")
            ax.plot(sub["year"], sub["frame_criminalization"],
                    marker="o", label=country,
                    color=COUNTRY_COLORS[country], linewidth=2)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_title(f"Criminalization Frame — {level}", fontsize=12)
        ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
        ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
        ax.legend(title="Country", fontsize=8)
    fig.suptitle("Criminalization Framing of Homelessness (2014–2024)", fontsize=14)
    _save(fig, "5_kw_criminalization_trend.png")


# ── 6. Keyword framing heatmap ────────────────────────────────────────────────
def plot_kw_heatmap(kw: pd.DataFrame):
    df  = kw[kw["level"] == "National"].groupby("base_country")[FRAME_COLS].mean()
    df  = df.loc[[c for c in COUNTRY_ORDER if c in df.index]]
    df.columns = [FRAME_LABELS[c] for c in df.columns]
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "% of Titles"})
    ax.set_title("Keyword Framing — National Press avg 2014–2024", fontsize=13)
    _save(fig, "6_kw_framing_heatmap.png")


# ── 7. Stacked frame bar ──────────────────────────────────────────────────────
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
    fig.suptitle("Frame Composition — National Press (2014–2024)", fontsize=14)
    _save(fig, "7_kw_stacked_frames.png")


# ── 8. Zero-shot heatmap ──────────────────────────────────────────────────────
def plot_zs_heatmap(zs: pd.DataFrame):
    avail = [c for c in ZS_FRAME_COLS if c in zs.columns]
    df    = zs[zs["level"] == "National"].groupby("base_country")[avail].mean()
    df    = df.loc[[c for c in COUNTRY_ORDER if c in df.index]]
    df.columns = [FRAME_LABELS.get(
        c.replace("frame_zs_", "frame_"), c) for c in df.columns]
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlOrRd",
                linewidths=0.5, ax=ax, cbar_kws={"label": "% of Titles"})
    ax.set_title("Zero-Shot Framing — National Press avg 2014–2024", fontsize=13)
    _save(fig, "8_zs_framing_heatmap.png")


# ── 9. Keyword vs zero-shot side-by-side ─────────────────────────────────────
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
    zs_mean.columns = [FRAME_LABELS.get(
        c.replace("frame_zs_", "frame_"), c) for c in zs_mean.columns]

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
    _save(fig, "9_zs_vs_kw_comparison.png")


# ── 10. Sentiment vs criminalization scatter ──────────────────────────────────
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
    _save(fig, "10_sentiment_vs_criminalization.png")


# ── master runner ─────────────────────────────────────────────────────────────
def run_all():
    print("Loading summary files...")

    vader_s   = _add_meta(pd.read_csv("data/processed/vader_summary.csv"))
    kw_s      = _add_meta(pd.read_csv("data/processed/keyword_framing_summary.csv"))

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
    else:
        print("  Skipping fig 4 (RoBERTa not yet run)")
    plot_criminalization(kw_s)
    plot_kw_heatmap(kw_s)
    plot_kw_stacked(kw_s)
    if zs_s is not None:
        plot_zs_heatmap(zs_s)
        plot_zs_vs_kw(kw_s, zs_s)
    else:
        print("  Skipping figs 8-9 (zero-shot not yet run)")
    plot_sentiment_vs_crime(vader_s, kw_s)

    print("\nAll figures saved to data/figures/")


if __name__ == "__main__":
    run_all()

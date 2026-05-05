# src/plot_kw_trends.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os

os.makedirs("data/figures", exist_ok=True)

COUNTRY_ORDER  = ["US", "UK", "Canada", "Australia"]
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
FRAME_COLORS = [
    "#e41a1c","#377eb8","#4daf4a","#984ea3",
    "#ff7f00","#a65628","#f781bf"
]
YEARS = list(range(2014, 2025))

kw = pd.read_csv("data/processed/keyword_framing_summary.csv")
kw["base_country"] = kw["country"].str.extract(r"^(US|UK|Canada|Australia)")
kw["level"]        = kw["country"].apply(
    lambda x: "National" if "National" in x else "State & Local"
)
df = kw[kw["level"] == "National"]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for i, country in enumerate(COUNTRY_ORDER):
    ax  = axes.flatten()[i]
    sub = df[df["base_country"] == country].sort_values("year")
    if sub.empty:
        continue
    for col, color in zip(FRAME_COLS, FRAME_COLORS):
        ax.plot(sub["year"], sub[col],
                marker="o", label=FRAME_LABELS[col],
                color=color, linewidth=2)
    ax.set_title(country, fontsize=12)
    ax.set_xticks(YEARS); ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Year"); ax.set_ylabel("% of Titles")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    if i == 0:
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")

fig.suptitle("Keyword Framing Trends — National Press (2014–2024)", fontsize=14)
fig.tight_layout()
fig.savefig("data/figures/12b_kw_trends_per_country.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: data/figures/12b_kw_trends_per_country.png")
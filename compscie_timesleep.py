# ============================================
# Sleep Deprivation and Cognitive Performance Analyzer
# Option A (FIXED): Raw xlsx columns
#   Sleep-related  → PIE only
#   Non-sleep      → BAR/HISTOGRAM only
#   Age, BMI, Gender → REMOVED
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# -----------------------------
# 1. Load the Excel dataset
# -----------------------------
df = pd.read_excel("sleep_calupdated1.xlsx", sheet_name="stresslvl")

print("First 5 rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values per column:")
print(df.isnull().sum())

# -----------------------------
# 2. Data cleaning
# -----------------------------
df = df.dropna()

# ============================================================
# 3. FULL STATISTICS FOR EVERY NUMERICAL COLUMN (raw xlsx)
#    Age and BMI have been REMOVED
# ============================================================
numerical_cols = [
    "Sleep_Hours", "Sleep_Quality_Score", "Daytime_Sleepiness",
    "Stroop_Task_Reaction_Time", "N_Back_Accuracy", "Emotion_Regulation_Score",
    "PVT_Reaction_Time", "Caffeine_Intake", "Physical_Activity_Level",
    "Stress_Level"
]

stats_rows = []
for col in numerical_cols:
    arr = df[col].to_numpy(dtype=float)
    stats_rows.append({
        "Variable": col,
        "Count": len(arr),
        "Mean": round(np.mean(arr), 2),
        "Median": round(np.median(arr), 2),
        "Std Dev": round(np.std(arr, ddof=1), 2),
        "Min": round(np.min(arr), 2),
        "Max": round(np.max(arr), 2),
        "Range": round(np.ptp(arr), 2),
        "Q1 (25%)": round(np.percentile(arr, 25), 2),
        "Q3 (75%)": round(np.percentile(arr, 75), 2),
        "IQR": round(np.percentile(arr, 75) - np.percentile(arr, 25), 2)
    })

stats_df = pd.DataFrame(stats_rows)
print("\n" + "="*80)
print("FULL STATISTICS FOR EVERY NUMERICAL VARIABLE (raw xlsx)")
print("="*80)
print(stats_df.to_string(index=False))
stats_df.to_csv("variable_statistics.csv", index=False)

# ============================================================
# 4. VALUE COUNTS FOR DISCRETE / CATEGORICAL RAW COLUMNS
#    Age and Gender have been REMOVED
# ============================================================
discrete_cols = [
    "Sleep_Quality_Score", "Daytime_Sleepiness",
    "Caffeine_Intake", "Physical_Activity_Level",
    "Stress_Level"
]

print("\n" + "="*80)
print("VALUE COUNTS (raw xlsx discrete columns)")
print("="*80)

for col in discrete_cols:
    print(f"\n--- {col} ---")
    vc = df[col].value_counts().sort_index()
    pct = (vc / vc.sum() * 100).round(2)
    table = pd.DataFrame({"Count": vc, "Percentage (%)": pct})
    print(table)
    table.to_csv(f"counts_{col}.csv")

# ============================================================
# 5. CHART GENERATORS
# ============================================================
palette = [
    '#7F3FBF', '#E41A1C', '#4DAF4A', '#377EB8', '#FF7F00',
    '#00CED1', '#FF00FF', '#FFD700', '#8B4513', '#808080',
    '#1F77B4', '#FF9896', '#98DF8A', '#AEC7E8', '#FFBB78',
    '#F4A582', '#B2DF8A', '#FDBF6F', '#CAB2D6', '#FFFF99',
    '#FB9A99', '#A6CEE3', '#33A02C', '#E31A1C', '#6A3D9A'
]

# --- PIE (SLEEP-RELATED ONLY) ---
def draw_pie(series, title, filename):
    """Small pie + side legend + clear percentage labels on each slice."""
    vc = series.value_counts().sort_index()
    labels = [str(x) for x in vc.index]
    values = vc.values
    total = values.sum()
    colors = [palette[i % len(palette)] for i in range(len(values))]

    fig, ax = plt.subplots(figsize=(11, 6))

    wedges, texts, autotexts = ax.pie(
        values,
        labels=None,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        colors=colors,
        radius=0.85,
        pctdistance=0.75,
        textprops={"fontsize": 9, "fontweight": "bold", "color": "white"}
    )
    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    legend_labels = [f"{l}  (n={v})" for l, v in zip(labels, values)]
    ax.legend(wedges, legend_labels,
              loc="center left",
              bbox_to_anchor=(1.0, 0.5),
              fontsize=8,
              title=series.name,
              title_fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {filename}")

# --- HISTOGRAM (NON-SLEEP CONTINUOUS ONLY) ---
def draw_hist(column_name, bins, filename, color='#377EB8'):
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name], bins=bins, edgecolor="black",
             color=color, alpha=0.85)
    plt.title(f"Distribution of {column_name} (Histogram)",
              fontsize=13, fontweight="bold")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {filename}")

# --- BAR (NON-SLEEP DISCRETE ONLY) ---
def draw_bar(column_name, title, filename):
    vc = df[column_name].value_counts().sort_index()
    labels = [str(x) for x in vc.index]
    values = vc.values
    total = values.sum()
    colors = [palette[i % len(palette)] for i in range(len(values))]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, values, edgecolor="black", color=colors)
    for bar, cnt in zip(bars, values):
        height = bar.get_height()
        pct = cnt / total * 100
        plt.text(bar.get_x() + bar.get_width() / 2,
                 height + max(values) * 0.02,
                 f"{int(cnt)}\n({pct:.1f}%)",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.title(title + " (Bar Graph)", fontsize=13, fontweight="bold")
    plt.xlabel(column_name)
    plt.ylabel("Count")
    plt.ylim(0, max(values) * 1.25)
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {filename}")

# ============================================================
# 6. SLEEP-RELATED COLUMNS → PIE ONLY
# ============================================================
def bin_sleep_hours(x):
    if x <= 2:
        return "<=2h"
    elif x < 5:
        return "3-4h"
    elif x < 7:
        return "5-6h"
    elif x < 9:
        return "7-8h"
    else:
        return ">=9h"

sleep_hours_binned = df["Sleep_Hours"].apply(bin_sleep_hours)
sleep_hours_binned.name = "Sleep_Hours"

draw_pie(sleep_hours_binned,
         "Sleep_Hours – Participants per Sleep Range",
         "Sleep_Hours_pie.png")

draw_pie(df["Sleep_Quality_Score"],
         "Sleep_Quality_Score – Participants per Score",
         "Sleep_Quality_Score_pie.png")

draw_pie(df["Daytime_Sleepiness"],
         "Daytime_Sleepiness – Participants per Score",
         "Daytime_Sleepiness_pie.png")

# ============================================================
# 7. NON-SLEEP COLUMNS → BAR / HISTOGRAM ONLY (no pies)
#    Age, BMI, Gender have been REMOVED
# ============================================================
# Discrete raw columns → bar chart
discrete_non_sleep = [
    ("Caffeine_Intake", "Caffeine_Intake"),
    ("Physical_Activity_Level", "Physical_Activity_Level"),
    ("Stress_Level", "Stress_Level")
]

for col, title in discrete_non_sleep:
    draw_bar(col, title, f"{col}_bar.png")

# Continuous raw columns → histogram
continuous_non_sleep = [
    ("PVT_Reaction_Time", 15),
    ("N_Back_Accuracy", 15),
    ("Stroop_Task_Reaction_Time", 15),
    ("Emotion_Regulation_Score", 15)
]

for col, bins in continuous_non_sleep:
    draw_hist(col, bins=bins, filename=f"{col}_histogram.png",
              color='#377EB8')

# ============================================================
# 8. SUMMARY
# ============================================================
print("\n" + "="*80)
print("SLEEP-RELATED OUTPUTS (PIE ONLY)")
print("="*80)
print(" - Sleep_Hours_pie.png")
print(" - Sleep_Quality_Score_pie.png")
print(" - Daytime_Sleepiness_pie.png")

print("\nNON-SLEEP OUTPUTS (BAR / HISTOGRAM ONLY)")
print("="*80)
for col, _ in discrete_non_sleep:
    print(f" - {col}_bar.png")
for col, _ in continuous_non_sleep:
    print(f" - {col}_histogram.png")

print("\nCSV FILES")
print(" - variable_statistics.csv")
for col in discrete_cols:
    print(f" - counts_{col}.csv")
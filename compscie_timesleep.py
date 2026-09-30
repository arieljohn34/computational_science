# ============================================
# Sleep Deprivation and Cognitive Performance Analyzer
# Sleep_Hours     → PIE only
# Other sleep     → SCATTER PLOT LINE
# Stress_Level    → SCATTER PLOT LINE
# Other non-sleep → BAR / HISTOGRAM only
# Age, BMI, Gender → REMOVED
# ALL CHARTS NOW HAVE LEGENDS
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
# 4. VALUE COUNTS FOR DISCRETE RAW COLUMNS
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

# --- PIE (only for Sleep_Hours) ---
def draw_pie(series, title, filename):
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
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    legend_labels = [f"{l}  (n={v})" for l, v in zip(labels, values)]
    ax.legend(wedges, legend_labels,
              loc="center left",
              bbox_to_anchor=(1.0, 0.5),
              fontsize=9,
              title=series.name,
              title_fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {filename}")

# --- SCATTER PLOT LINE ---
def draw_scatter_line(series, title, filename, line_color='#7F3FBF'):
    vc = series.value_counts().sort_index()
    x = list(vc.index)
    y = list(vc.values)
    total = sum(y)

    plt.figure(figsize=(11, 6))

    plt.scatter(x, y,
                s=140, color=line_color,
                edgecolor='black', zorder=3,
                label=f'{series.name} (count)')

    plt.plot(x, y, color=line_color,
             linewidth=2, alpha=0.7, zorder=2,
             label=f'{series.name} (trend)')

    for xi, yi in zip(x, y):
        pct = yi / total * 100
        plt.annotate(f"{yi}\n({pct:.1f}%)",
                     (xi, yi),
                     textcoords="offset points",
                     xytext=(0, 12),
                     ha='center',
                     fontsize=8, fontweight='bold')

    plt.title(title + " (Scatter Plot Line)",
              fontsize=13, fontweight="bold")
    plt.xlabel(series.name)
    plt.ylabel("Number of Participants")
    plt.ylim(0, max(y) * 1.25)
    plt.xticks(rotation=15)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {filename}")

# --- HISTOGRAM ---
def draw_hist(column_name, bins, filename, color='#377EB8'):
    plt.figure(figsize=(10, 6))
    plt.hist(df[column_name], bins=bins, edgecolor="black",
             color=color, alpha=0.85,
             label=f"{column_name} distribution")
    plt.axvline(df[column_name].mean(), color='red',
                linestyle='--', linewidth=2,
                label=f"Mean = {df[column_name].mean():.2f}")
    plt.axvline(df[column_name].median(), color='green',
                linestyle='--', linewidth=2,
                label=f"Median = {df[column_name].median():.2f}")
    plt.title(f"Distribution of {column_name} (Histogram)",
              fontsize=13, fontweight="bold")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {filename}")

# --- BAR ---
def draw_bar(column_name, title, filename):
    vc = df[column_name].value_counts().sort_index()
    labels = [str(x) for x in vc.index]
    values = vc.values
    total = values.sum()
    colors = [palette[i % len(palette)] for i in range(len(values))]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, values, edgecolor="black", color=colors,
                   label=f"{column_name} count")
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
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Saved] {filename}")

# ============================================================
# 6. SLEEP_HOURS → PIE ONLY
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

order = ["<=2h", "3-4h", "5-6h", "7-8h", ">=9h"]
sleep_hours_binned = pd.Categorical(sleep_hours_binned, categories=order, ordered=True)
sleep_hours_binned = pd.Series(sleep_hours_binned, name="Sleep_Hours")

draw_pie(sleep_hours_binned,
         "Sleep_Hours – Participants per Sleep Range",
         "Sleep_Hours_pie.png")

# ============================================================
# 7. OTHER SLEEP-RELATED COLUMNS → SCATTER PLOT LINE
# ============================================================
draw_scatter_line(df["Sleep_Quality_Score"],
                  "Sleep_Quality_Score – Participants per Score",
                  "Sleep_Quality_Score_scatter_line.png",
                  line_color='#E41A1C')

draw_scatter_line(df["Daytime_Sleepiness"],
                  "Daytime_Sleepiness – Participants per Score",
                  "Daytime_Sleepiness_scatter_line.png",
                  line_color='#377EB8')

# ============================================================
# 8. STRESS_LEVEL → SCATTER PLOT LINE
# ============================================================
draw_scatter_line(df["Stress_Level"],
                  "Stress_Level – Participants per Score",
                  "Stress_Level_scatter_line.png",
                  line_color='#FF7F00')

# ============================================================
# 9. OTHER NON-SLEEP COLUMNS → BAR / HISTOGRAM ONLY
# ============================================================
discrete_non_sleep = [
    ("Caffeine_Intake", "Caffeine_Intake"),
    ("Physical_Activity_Level", "Physical_Activity_Level")
]

for col, title in discrete_non_sleep:
    draw_bar(col, title, f"{col}_bar.png")

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
# 10. SUMMARY
# ============================================================
print("\n" + "="*80)
print("SLEEP_HOURS OUTPUT (PIE ONLY)")
print("="*80)
print(" - Sleep_Hours_pie.png")

print("\nSLEEP-RELATED + STRESS OUTPUTS (SCATTER PLOT LINE)")
print("="*80)
print(" - Sleep_Quality_Score_scatter_line.png")
print(" - Daytime_Sleepiness_scatter_line.png")
print(" - Stress_Level_scatter_line.png")

print("\nOTHER NON-SLEEP OUTPUTS (BAR / HISTOGRAM ONLY)")
print("="*80)
for col, _ in discrete_non_sleep:
    print(f" - {col}_bar.png")
for col, _ in continuous_non_sleep:
    print(f" - {col}_histogram.png")

print("\nCSV FILES")
print(" - variable_statistics.csv")
for col in discrete_cols:
    print(f" - counts_{col}.csv")
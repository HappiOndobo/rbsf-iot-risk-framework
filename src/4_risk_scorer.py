# ============================================================
# SCRIPT 4 — 4_risk_scorer.py
# RBSF IoT Security Framework — Risk Scoring Engine
# Author: Happi Ondobo Steve | MTUCI 2026
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the trained models
#   2. Takes the test data as "new incoming traffic"
#   3. For each record, computes:
#      - Supervised classification (Random Forest)
#      - Anomaly score (Isolation Forest)
#      - Combined RISK SCORE (0.0 to 1.0)
#      - Risk Level: Low / Medium / High / Critical
#   4. Saves a CSV with all results
#   5. Generates a Risk Score Distribution chart
#   6. Prints a summary dashboard
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import json

np.random.seed(42)

PURPLE  = "#4B0082"
LPURPLE = "#7B2FBE"
PALE    = "#F0E6FF"
DARK    = "#2D0057"
GRAY    = "#555555"

# Risk level colors
LEVEL_COLORS = {
    "Low":      "#1D9E75",   # green
    "Medium":   "#BA7517",   # amber
    "High":     "#D85A30",   # orange
    "Critical": "#E24B4A",   # red
}

print("=" * 60)
print("  RBSF — Step 4: Risk Scoring Engine")
print("=" * 60)

# ── LOAD EVERYTHING ───────────────────────────────────────────
print("\n[1/5] Loading models and data...")

with open("rbsf_output/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("rbsf_output/if_model.pkl", "rb") as f:
    if_model = pickle.load(f)
with open("rbsf_output/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

X_test       = pd.read_csv("rbsf_output/X_test.csv")
y_test_names = pd.read_csv("rbsf_output/y_test_names.csv").squeeze()

print(f"    Records to score: {len(X_test):,}")

# ── RISK SCORE FORMULA ────────────────────────────────────────
# R = alpha * C + beta * A + gamma * H
#
#   C = Supervised classification score
#       (probability that record is an attack, from Random Forest)
#
#   A = Anomaly score
#       (how unusual this record is, from Isolation Forest)
#       0 = perfectly normal, 1 = very unusual
#
#   H = Historical risk score
#       (simplified: we use a rolling mean of the last 10 scores)
#       In a real system this would track each device over time.
#
#   Weights (sum to 1.0):
ALPHA = 0.55    # supervised score weight
BETA  = 0.30    # anomaly score weight
GAMMA = 0.15    # history weight

print(f"\n    Risk formula: R = {ALPHA}×C + {BETA}×A + {GAMMA}×H")
print(f"    Thresholds:")
print(f"      Low      : 0.00 – 0.30")
print(f"      Medium   : 0.31 – 0.60")
print(f"      High     : 0.61 – 0.85")
print(f"      Critical : 0.86 – 1.00")

# ── STEP 2: SUPERVISED SCORE (C) ─────────────────────────────
print("\n[2/5] Computing supervised classification scores...")

rf_proba  = rf_model.predict_proba(X_test)     # shape: (n, n_classes)
rf_labels = rf_model.predict(X_test)
rf_names  = label_encoder.inverse_transform(rf_labels)

normal_idx = list(label_encoder.classes_).index("Normal")

# C = probability of being an ATTACK = 1 - P(Normal)
C = 1.0 - rf_proba[:, normal_idx]

print(f"    Mean C score (Normal records)  : {C[y_test_names == 'Normal'].mean():.4f}")
print(f"    Mean C score (Attack records)  : {C[y_test_names != 'Normal'].mean():.4f}")

# ── STEP 3: ANOMALY SCORE (A) ─────────────────────────────────
print("\n[3/5] Computing anomaly scores (Isolation Forest)...")

if_raw    = if_model.score_samples(X_test)     # more negative = more anomalous
# Normalize to 0–1, then flip so 1 = highly anomalous
if_min, if_max = if_raw.min(), if_raw.max()
A = 1.0 - (if_raw - if_min) / (if_max - if_min + 1e-9)

print(f"    Mean A score (Normal records)  : {A[y_test_names == 'Normal'].mean():.4f}")
print(f"    Mean A score (Attack records)  : {A[y_test_names != 'Normal'].mean():.4f}")

# ── STEP 4: HISTORY SCORE (H) ─────────────────────────────────
# Simplified: use a rolling average of C score over last 10 records
# In a real deployment each device would have its own history
C_series = pd.Series(C)
H = C_series.rolling(window=10, min_periods=1).mean().values

# ── STEP 5: COMBINED RISK SCORE ───────────────────────────────
print("\n[4/5] Computing combined risk scores...")

R = ALPHA * C + BETA * A + GAMMA * H
R = np.clip(R, 0.0, 1.0)   # make sure score stays in [0, 1]

def risk_level(score):
    if score <= 0.30:
        return "Low"
    elif score <= 0.60:
        return "Medium"
    elif score <= 0.85:
        return "High"
    else:
        return "Critical"

risk_levels = [risk_level(r) for r in R]

# ── BUILD RESULTS DATAFRAME ───────────────────────────────────
results = pd.DataFrame({
    "true_label":       y_test_names.values,
    "predicted_label":  rf_names,
    "supervised_score": np.round(C, 4),
    "anomaly_score":    np.round(A, 4),
    "history_score":    np.round(H, 4),
    "risk_score":       np.round(R, 4),
    "risk_level":       risk_levels,
    "correctly_classified": (y_test_names.values == rf_names),
})

results.to_csv("rbsf_output/risk_scores.csv", index=False)
print("    Saved → rbsf_output/risk_scores.csv")

# ── PRINT DASHBOARD SUMMARY ───────────────────────────────────
print("\n" + "=" * 60)
print("  RISK SCORE DASHBOARD SUMMARY")
print("=" * 60)

level_counts = results["risk_level"].value_counts()
total = len(results)

print(f"\n  Total devices/records evaluated: {total:,}")
print(f"\n  Risk Level Distribution:")
for lvl in ["Low", "Medium", "High", "Critical"]:
    cnt = level_counts.get(lvl, 0)
    pct = cnt / total * 100
    bar = "█" * int(pct / 2)
    color_label = {"Low":"SAFE","Medium":"WARN","High":"ALERT","Critical":"EMERGENCY"}[lvl]
    print(f"    [{color_label:<9}] {lvl:<10} {cnt:>5} records  {pct:5.1f}%  {bar}")

print(f"\n  Average Risk Scores by True Label:")
print(f"    {'Label':<12} {'Avg Risk Score':>15} {'Avg C':>10} {'Avg A':>10}")
print(f"    {'-'*50}")
for label in sorted(results["true_label"].unique()):
    mask = results["true_label"] == label
    avg_r = results.loc[mask, "risk_score"].mean()
    avg_c = results.loc[mask, "supervised_score"].mean()
    avg_a = results.loc[mask, "anomaly_score"].mean()
    bar = "█" * int(avg_r * 20)
    print(f"    {label:<12} {avg_r:>14.4f}  {avg_c:>9.4f}  {avg_a:>9.4f}  {bar}")

# Detection summary
print(f"\n  Detection Summary:")
attack_mask = results["true_label"] != "Normal"
detected    = results[attack_mask & (results["risk_level"].isin(["Medium","High","Critical"]))]
missed      = results[attack_mask & (results["risk_level"] == "Low")]
fp_mask     = (results["true_label"] == "Normal") & (results["risk_level"] != "Low")

print(f"    Real attacks in test set    : {attack_mask.sum():,}")
print(f"    Detected (risk >= Medium)   : {len(detected):,}  "
      f"({len(detected)/attack_mask.sum()*100:.1f}%)")
print(f"    Missed   (risk = Low)       : {len(missed):,}  "
      f"({len(missed)/attack_mask.sum()*100:.1f}%)")
print(f"    False alarms (normal→flagged): {fp_mask.sum():,}  "
      f"({fp_mask.sum()/(~attack_mask).sum()*100:.1f}%)")

# ── RISK SCORE DISTRIBUTION CHART ─────────────────────────────
print("\n[5/5] Generating risk score distribution chart...")

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# --- Plot 1: Risk score histogram by true label ---
ax1 = axes[0]
normal_scores = results.loc[results["true_label"] == "Normal", "risk_score"]
attack_scores = results.loc[results["true_label"] != "Normal", "risk_score"]

ax1.hist(normal_scores, bins=50, alpha=0.6, color=LPURPLE,
         label="Normal traffic", density=True)
ax1.hist(attack_scores, bins=50, alpha=0.6, color="#D85A30",
         label="Attack traffic", density=True)

# Add threshold lines
for threshold, label_t, color_t in [
    (0.30, "Low→Medium", "#BA7517"),
    (0.60, "Medium→High", "#D85A30"),
    (0.85, "High→Critical", "#E24B4A"),
]:
    ax1.axvline(threshold, color=color_t, linestyle="--", lw=1.5, alpha=0.8)
    ax1.text(threshold + 0.01, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 1,
             label_t, fontsize=8, color=color_t, rotation=90, va="top")

ax1.set_xlabel("Risk Score", fontsize=11, color=GRAY)
ax1.set_ylabel("Density", fontsize=11, color=GRAY)
ax1.set_title("Risk Score Distribution: Normal vs Attack Traffic",
              fontsize=13, fontweight="bold", color=DARK)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, color=GRAY)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- Plot 2: Risk level counts by true label (stacked bar) ---
ax2 = axes[1]
labels_ordered = ["Normal"] + [c for c in sorted(results["true_label"].unique()) if c != "Normal"]
risk_levels_ordered = ["Low", "Medium", "High", "Critical"]

bar_data = {}
for lvl in risk_levels_ordered:
    bar_data[lvl] = []
    for lbl in labels_ordered:
        mask = (results["true_label"] == lbl) & (results["risk_level"] == lvl)
        bar_data[lvl].append(mask.sum())

x = np.arange(len(labels_ordered))
width = 0.6
bottom = np.zeros(len(labels_ordered))

for lvl, color_val in LEVEL_COLORS.items():
    vals = np.array(bar_data[lvl])
    ax2.bar(x, vals, width, bottom=bottom, label=lvl, color=color_val, alpha=0.85)
    bottom += vals

ax2.set_xticks(x)
ax2.set_xticklabels(labels_ordered, rotation=30, ha="right", fontsize=9)
ax2.set_ylabel("Number of Records", fontsize=11, color=GRAY)
ax2.set_xlabel("True Traffic Label", fontsize=11, color=GRAY)
ax2.set_title("Risk Level Assignment by Traffic Category",
              fontsize=13, fontweight="bold", color=DARK)
ax2.legend(title="Risk Level", fontsize=9, loc="upper right")
ax2.grid(True, axis="y", alpha=0.3, color=GRAY)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

plt.tight_layout(pad=3.0)
plt.savefig("rbsf_output/risk_distribution.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close()
print("    Saved → rbsf_output/risk_distribution.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  RISK SCORING COMPLETE")
print("=" * 60)
print(f"\n  Files saved in rbsf_output/:")
print(f"    risk_scores.csv          — full results for every record")
print(f"    risk_distribution.png    — risk score distribution chart")
print(f"\n  ALL 4 SCRIPTS COMPLETE!")
print(f"\n  Your RBSF framework is working. Summary of results:")
print(f"    → Models trained and saved (rf_model.pkl, if_model.pkl)")
print(f"    → Confusion matrix generated (confusion_matrix.png)")
print(f"    → ROC curves generated (roc_curve.png)")
print(f"    → Feature importance chart (feature_importance.png)")
print(f"    → Risk scores computed for all test records")
print(f"\n  Next step: Step 2 — Build the Flask web dashboard")
print("=" * 60)

# ============================================================
# SCRIPT 3 — 3_evaluate.py
# RBSF IoT Security Framework — Evaluation & Visualizations
# Author: Happi Ondobo Steve | MTUCI 2026
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the trained models and test data
#   2. Makes predictions on the test set
#   3. Calculates all performance metrics
#   4. Generates and saves:
#      - Confusion matrix heatmap (PNG)
#      - ROC curve (PNG)
#      - Feature importance chart (PNG)
#      - Full metrics report (TXT)
# ============================================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # no display needed — saves directly to file
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pickle
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

np.random.seed(42)

# University purple color palette
PURPLE      = "#4B0082"
LPURPLE     = "#7B2FBE"
PALE        = "#F0E6FF"
DARK        = "#2D0057"
GRAY        = "#555555"

print("=" * 60)
print("  RBSF — Step 3: Evaluation & Visualizations")
print("=" * 60)

# ── LOAD MODELS AND DATA ──────────────────────────────────────
print("\n[1/6] Loading models and test data...")

with open("rbsf_output/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)
with open("rbsf_output/if_model.pkl", "rb") as f:
    if_model = pickle.load(f)
with open("rbsf_output/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

X_test      = pd.read_csv("rbsf_output/X_test.csv")
y_test      = pd.read_csv("rbsf_output/y_test.csv").squeeze()
y_test_names= pd.read_csv("rbsf_output/y_test_names.csv").squeeze()

classes = list(label_encoder.classes_)
print(f"    Test set: {len(X_test):,} records")
print(f"    Classes : {classes}")

# ── STEP 2: RANDOM FOREST PREDICTIONS ────────────────────────
print("\n[2/6] Making predictions with Random Forest...")

y_pred     = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)   # probability for each class

# ── STEP 3: COMPUTE METRICS ───────────────────────────────────
print("\n[3/6] Computing performance metrics...")

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1        = f1_score(y_test, y_pred, average="weighted", zero_division=0)
mcc       = matthews_corrcoef(y_test, y_pred)

# AUC-ROC (one-vs-rest for multiclass)
y_test_bin = label_binarize(y_test, classes=list(range(len(classes))))
auc_roc = roc_auc_score(y_test_bin, y_pred_proba, average="weighted", multi_class="ovr")

# False positive and false negative rates
cm = confusion_matrix(y_test, y_pred)
FP = cm.sum(axis=0) - np.diag(cm)
FN = cm.sum(axis=1) - np.diag(cm)
TP = np.diag(cm)
TN = cm.sum() - (FP + FN + TP)
fpr_mean = (FP / (FP + TN + 1e-9)).mean()
fnr_mean = (FN / (FN + TP + 1e-9)).mean()

print(f"\n    ┌─────────────────────────────────────┐")
print(f"    │  PERFORMANCE METRICS — TEST SET     │")
print(f"    ├─────────────────────────────────────┤")
print(f"    │  Accuracy          : {accuracy*100:>7.2f}%       │")
print(f"    │  Precision         : {precision*100:>7.2f}%       │")
print(f"    │  Recall            : {recall*100:>7.2f}%       │")
print(f"    │  F1-Score          : {f1*100:>7.2f}%       │")
print(f"    │  AUC-ROC           : {auc_roc:>8.4f}        │")
print(f"    │  Matthews CC       : {mcc:>8.4f}        │")
print(f"    │  False Positive Rate: {fpr_mean*100:>6.2f}%       │")
print(f"    │  False Negative Rate: {fnr_mean*100:>6.2f}%       │")
print(f"    └─────────────────────────────────────┘")

# Per-class results
print(f"\n    Per-class results:")
print(f"    {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
print(f"    {'-'*55}")
per_class_p = precision_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(len(classes))))
per_class_r = recall_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(len(classes))))
per_class_f = f1_score(y_test, y_pred, average=None, zero_division=0, labels=list(range(len(classes))))
per_class_s = np.bincount(y_test, minlength=len(classes))

for i, cls in enumerate(classes):
    print(f"    {cls:<12} {per_class_p[i]*100:>9.2f}%  {per_class_r[i]*100:>9.2f}%  "
          f"{per_class_f[i]*100:>9.2f}%  {per_class_s[i]:>9}")

# Isolation Forest anomaly scores on test set
if_scores     = if_model.score_samples(X_test)
# Convert: more negative = more anomalous → flip and normalize to 0–1
if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min() + 1e-9)
if_anomaly    = 1 - if_scores_norm   # high value = high anomaly

# Quick check: do anomaly scores separate normal from attack?
is_normal_mask  = y_test_names == "Normal"
if_normal_mean  = if_anomaly[is_normal_mask].mean()
if_attack_mean  = if_anomaly[~is_normal_mask].mean()
print(f"\n    Isolation Forest sanity check:")
print(f"      Mean anomaly score — Normal traffic : {if_normal_mean:.4f}")
print(f"      Mean anomaly score — Attack traffic : {if_attack_mean:.4f}")
print(f"      (attack score should be higher than normal score)")

# ── STEP 4: CONFUSION MATRIX ──────────────────────────────────
print("\n[4/6] Generating confusion matrix...")

fig, ax = plt.subplots(figsize=(12, 10))
cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

sns.heatmap(
    cm_pct,
    annot=True,
    fmt=".1f",
    cmap=sns.light_palette(PURPLE, as_cmap=True),
    xticklabels=classes,
    yticklabels=classes,
    linewidths=0.5,
    linecolor="#CCCCCC",
    ax=ax,
    cbar_kws={"label": "Percentage (%)"}
)

ax.set_title(
    "Confusion Matrix — RBSF IoT Security Framework\n(values in %)",
    fontsize=14, fontweight="bold", color=DARK, pad=20
)
ax.set_xlabel("Predicted Label", fontsize=12, color=GRAY, labelpad=10)
ax.set_ylabel("True Label", fontsize=12, color=GRAY, labelpad=10)
ax.tick_params(axis="x", rotation=45, labelsize=9)
ax.tick_params(axis="y", rotation=0,  labelsize=9)

# Add accuracy annotation
ax.text(
    0.02, 0.02,
    f"Overall Accuracy: {accuracy*100:.2f}%   |   F1-Score: {f1*100:.2f}%   |   AUC: {auc_roc:.4f}",
    transform=ax.transAxes,
    fontsize=9, color=GRAY,
    bbox=dict(boxstyle="round,pad=0.3", facecolor=PALE, edgecolor=LPURPLE, alpha=0.8)
)

plt.tight_layout()
plt.savefig("rbsf_output/confusion_matrix.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
print("    Saved → rbsf_output/confusion_matrix.png")

# ── STEP 5: ROC CURVE ─────────────────────────────────────────
print("\n[5/6] Generating ROC curve...")

fig, ax = plt.subplots(figsize=(10, 8))

colors = [
    "#4B0082","#7B2FBE","#1D9E75","#D85A30",
    "#378ADD","#BA7517","#D4537E","#639922",
    "#E24B4A","#888780"
]

# Plot one ROC curve per class (one-vs-rest)
auc_scores = []
for i, (cls_name, color) in enumerate(zip(classes, colors)):
    fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    auc_c = auc(fpr_c, tpr_c)
    auc_scores.append(auc_c)
    ax.plot(fpr_c, tpr_c, color=color, lw=1.5,
            label=f"{cls_name} (AUC = {auc_c:.3f})")

# Overall weighted AUC line
ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Classifier (AUC = 0.50)")

ax.fill_between([0, 1], [0, 1], alpha=0.03, color=PURPLE)

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12, color=GRAY)
ax.set_ylabel("True Positive Rate (Recall)", fontsize=12, color=GRAY)
ax.set_title(
    f"ROC Curves — RBSF IoT Security Framework\nWeighted AUC = {auc_roc:.4f}",
    fontsize=14, fontweight="bold", color=DARK
)
ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
ax.grid(True, alpha=0.3, color=GRAY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("rbsf_output/roc_curve.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
print("    Saved → rbsf_output/roc_curve.png")

# ── BONUS: FEATURE IMPORTANCE CHART ───────────────────────────
feat_importance = pd.read_csv("rbsf_output/feature_importance.csv",
                              index_col=0).squeeze()
top15 = feat_importance.head(15)

fig, ax = plt.subplots(figsize=(10, 7))
bars = ax.barh(
    range(len(top15)), top15.values,
    color=[PURPLE if i < 5 else LPURPLE if i < 10 else "#AFA9EC"
           for i in range(len(top15))],
    edgecolor="white", height=0.7
)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Feature Importance Score", fontsize=11, color=GRAY)
ax.set_title(
    "Top 15 Most Important Features — Random Forest\n(higher = more useful for attack detection)",
    fontsize=13, fontweight="bold", color=DARK
)
ax.grid(True, axis="x", alpha=0.3, color=GRAY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Value labels on bars
for bar, val in zip(bars, top15.values):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=8, color=DARK)

legend_patches = [
    mpatches.Patch(color=PURPLE,  label="Top 1–5"),
    mpatches.Patch(color=LPURPLE, label="Top 6–10"),
    mpatches.Patch(color="#AFA9EC", label="Top 11–15"),
]
ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("rbsf_output/feature_importance.png", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.close()
print("    Saved → rbsf_output/feature_importance.png")

# ── STEP 6: SAVE FULL METRICS REPORT ─────────────────────────
print("\n[6/6] Saving full metrics report...")

report_lines = [
    "=" * 60,
    "  RBSF IoT SECURITY FRAMEWORK — EVALUATION REPORT",
    "  Author: Happi Ondobo Steve | MTUCI 2026",
    "=" * 60,
    "",
    "OVERALL METRICS (RANDOM FOREST — WEIGHTED AVERAGE)",
    "-" * 45,
    f"  Accuracy              : {accuracy*100:.4f}%",
    f"  Precision             : {precision*100:.4f}%",
    f"  Recall                : {recall*100:.4f}%",
    f"  F1-Score              : {f1*100:.4f}%",
    f"  AUC-ROC               : {auc_roc:.4f}",
    f"  Matthews Corr. Coeff  : {mcc:.4f}",
    f"  False Positive Rate   : {fpr_mean*100:.4f}%",
    f"  False Negative Rate   : {fnr_mean*100:.4f}%",
    "",
    "PER-CLASS RESULTS",
    "-" * 45,
    f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
    f"  {'-'*50}",
]

for i, cls in enumerate(classes):
    report_lines.append(
        f"  {cls:<12} {per_class_p[i]*100:>9.2f}%  "
        f"{per_class_r[i]*100:>9.2f}%  "
        f"{per_class_f[i]*100:>9.2f}%  "
        f"{per_class_s[i]:>9}"
    )

report_lines += [
    "",
    "ISOLATION FOREST — ANOMALY DETECTION",
    "-" * 45,
    f"  Mean anomaly score (Normal traffic)  : {if_normal_mean:.4f}",
    f"  Mean anomaly score (Attack traffic)  : {if_attack_mean:.4f}",
    f"  Separation ratio                     : {if_attack_mean/if_normal_mean:.2f}x",
    "",
    "OUTPUT FILES",
    "-" * 45,
    "  confusion_matrix.png    — confusion matrix heatmap",
    "  roc_curve.png           — ROC curves for all classes",
    "  feature_importance.png  — top 15 features bar chart",
    "  metrics_report.txt      — this file",
    "",
    "=" * 60,
]

report_text = "\n".join(report_lines)
with open("rbsf_output/metrics_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("    Saved → rbsf_output/metrics_report.txt")
print("\n" + report_text)

print("\n  Next step: run  python 4_risk_scorer.py")
print("=" * 60)

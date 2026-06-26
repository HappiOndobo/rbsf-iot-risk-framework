"""
=============================================================
  RBSF — Real-World Validation on TON_IoT Dataset
  Author: Хаппи Ондобо Стив | MTUCI 2026
  Script 5/5 — Framework Validation on Real Network Traffic
=============================================================
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_curve, auc
)

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = "Train_Test_Network.csv"
OUTPUT_DIR   = "rbsf_output/real_world"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FEATURES = 30
TEST_SIZE    = 0.20
RANDOM_STATE = 42

SYNTHETIC_METRICS = {
    "Accuracy"           : 98.3708,
    "Precision"          : 98.3800,
    "Recall"             : 98.3708,
    "F1-Score"           : 98.3743,
    "AUC-ROC"            : 0.9996,
    "Matthews CC"        : 0.9746,
    "False Positive Rate": 0.1866,
    "False Negative Rate": 1.3458,
}

LABEL_MAP = {
    "normal"    : "Normal",
    "backdoor"  : "Backdoor",
    "ddos"      : "DDoS",
    "dos"       : "DoS",
    "injection" : "Injection",
    "mitm"      : "MitM",
    "password"  : "Password",
    "ransomware": "Ransomware",
    "scanning"  : "Scanning",
    "xss"       : "Injection",
    "botnet"    : "Botnet",
    "0"         : "Normal",
    "1"         : "Attack",
}

ALWAYS_DROP = {
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "local_orig", "local_resp", "tunnel_parents", "Unnamed: 0",
    "id", "index", "date", "time", "src_ip", "dst_ip",
    "src_port", "dst_port",
}

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# ═════════════════════════════════════════════════════════════
#  STEP 1 — LOAD DATA
# ═════════════════════════════════════════════════════════════
section("STEP 1/6 — Loading TON_IoT Dataset")

df = pd.read_csv(DATASET_PATH, low_memory=False)
print(f"  Loaded  : {df.shape[0]:,} rows × {df.shape[1]} columns")

# Decide which column to use as label
# Prefer 'type' if it has multi-class attack names
label_col = "label"
drop_binary_col = None

if "type" in df.columns:
    type_vals = df["type"].astype(str).str.lower().unique()
    print(f"\n  'type' column unique values : {sorted(type_vals)}")
    if len(type_vals) > 2:
        print("  → Multi-class detected in 'type' column — using it as label")
        label_col = "type"
        drop_binary_col = "label"
    else:
        print("  → 'type' is binary — using 'label' column")
        drop_binary_col = "type"
elif "label" in df.columns:
    label_col = "label"

ALWAYS_DROP.add(drop_binary_col or "")   # drop the unused column

print(f"\n  Using label column : '{label_col}'")
print(f"  Unique values      : {sorted(df[label_col].astype(str).str.lower().unique())}")
print(f"\n  Raw distribution:\n{df[label_col].value_counts()}")

# ═════════════════════════════════════════════════════════════
#  STEP 2 — PREPROCESS
# ═════════════════════════════════════════════════════════════
section("STEP 2/6 — Preprocessing Real Data")

df = df.copy()
df["rbsf_label"] = (
    df[label_col].astype(str).str.lower().str.strip().map(LABEL_MAP)
)

before = len(df)
df = df.dropna(subset=["rbsf_label"])
dropped = before - len(df)
if dropped:
    print(f"  Dropped {dropped:,} rows with unmapped labels")

print(f"\n  RBSF label distribution:")
print(df["rbsf_label"].value_counts().to_string())

drop_cols = ALWAYS_DROP | {label_col, "rbsf_label"}
feature_cols = [c for c in df.columns if c not in drop_cols]

X_raw = df[feature_cols].copy()

print(f"\n  Encoding categorical columns...")
for col in X_raw.select_dtypes(include=["object"]).columns:
    le_tmp = LabelEncoder()
    X_raw[col] = le_tmp.fit_transform(X_raw[col].astype(str))
    print(f"    encoded: {col}")

X_raw = X_raw.dropna(axis=1, thresh=int(0.70 * len(X_raw)))
X_raw = X_raw.fillna(X_raw.median(numeric_only=True))
X_raw = X_raw.replace([np.inf, -np.inf], np.nan).fillna(0)

constant_cols = [c for c in X_raw.columns if X_raw[c].nunique() <= 1]
if constant_cols:
    X_raw.drop(columns=constant_cols, inplace=True)
    print(f"  Removed {len(constant_cols)} constant columns")

y_raw = df.loc[X_raw.index, "rbsf_label"]
le = LabelEncoder()
y = le.fit_transform(y_raw)

print(f"\n  Feature matrix : {X_raw.shape}")
print(f"  Classes        : {list(le.classes_)}")

# ═════════════════════════════════════════════════════════════
#  STEP 3 — SCALE + FEATURE SELECTION
# ═════════════════════════════════════════════════════════════
section("STEP 3/6 — Scaling & Feature Selection")

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X_raw), columns=X_raw.columns)

print("  Running quick RF selector (50 trees)...")
rf_sel = RandomForestClassifier(
    n_estimators=50, class_weight="balanced",
    n_jobs=-1, random_state=RANDOM_STATE
)
rf_sel.fit(X_scaled, y)

selector = SelectFromModel(rf_sel, max_features=MAX_FEATURES, prefit=True)
X_selected = selector.transform(X_scaled)
selected_names = X_raw.columns[selector.get_support()].tolist()

print(f"  Selected {len(selected_names)}/{X_scaled.shape[1]} features:")
for name in selected_names:
    print(f"    • {name}")

# ═════════════════════════════════════════════════════════════
#  STEP 4 — TRAIN / TEST SPLIT  (NO SMOTE — use class_weight)
# ═════════════════════════════════════════════════════════════
section("STEP 4/6 — Train/Test Split")

X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)
print(f"  Train : {X_train.shape[0]:,} samples")
print(f"  Test  : {X_test.shape[0]:,} samples")
print(f"\n  Class distribution (train):")
counts = pd.Series(y_train).value_counts().sort_index()
for idx, cnt in counts.items():
    print(f"    {le.classes_[idx]:<15} : {cnt:,}")

print("\n  NOTE: class_weight='balanced' used in RF instead of SMOTE")
print("        (equivalent result, no sample inflation)")

# ═════════════════════════════════════════════════════════════
#  STEP 5 — TRAIN RF + ISOLATION FOREST
# ═════════════════════════════════════════════════════════════
section("STEP 5/6 — Training RBSF Models on Real Data")

# ── Random Forest ──────────────────────────────────────────
print("  [RF] Training Random Forest (100 trees, balanced weights)...")
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",    # handles imbalance — no SMOTE needed
    n_jobs=-1,
    random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
print("  [RF] ✓ Training complete")

print("  [RF] Running 3-fold cross-validation...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
cv_acc = cross_val_score(rf, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
cv_f1  = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1_weighted", n_jobs=-1)
print(f"  [RF] CV Accuracy : {cv_acc.mean()*100:.4f}% ± {cv_acc.std()*100:.4f}%")
print(f"  [RF] CV F1-Score : {cv_f1.mean()*100:.4f}% ± {cv_f1.std()*100:.4f}%")

# ── Isolation Forest ────────────────────────────────────────
print("\n  [IF] Training Isolation Forest on Normal traffic...")
normal_classes = [c for c in le.classes_ if c.lower() == "normal"]
iso = None
if normal_classes:
    normal_idx = le.transform([normal_classes[0]])[0]
    X_normal = X_train[y_train == normal_idx]
    X_attack = X_train[y_train != normal_idx]
    iso = IsolationForest(
        contamination=0.1, n_estimators=100,
        n_jobs=-1, random_state=RANDOM_STATE
    )
    iso.fit(X_normal)
    scores_normal = -iso.score_samples(X_normal)
    scores_attack = -iso.score_samples(X_attack) if len(X_attack) > 0 else scores_normal
    sep_ratio = scores_attack.mean() / (scores_normal.mean() + 1e-9)
    print(f"  [IF] ✓ Trained on {len(X_normal):,} Normal samples")
    print(f"  [IF] Mean anomaly score — Normal : {scores_normal.mean():.4f}")
    print(f"  [IF] Mean anomaly score — Attack : {scores_attack.mean():.4f}")
    print(f"  [IF] Separation ratio            : {sep_ratio:.2f}x")
else:
    print("  [IF] Skipped — no 'Normal' class found")

# ═════════════════════════════════════════════════════════════
#  STEP 6 — EVALUATE
# ═════════════════════════════════════════════════════════════
section("STEP 6/6 — Evaluation & Comparison")

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)

accuracy  = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
recall    = recall_score(y_test, y_pred,    average="weighted", zero_division=0) * 100
f1        = f1_score(y_test, y_pred,        average="weighted", zero_division=0) * 100
mcc       = matthews_corrcoef(y_test, y_pred)

# AUC-ROC — binary vs multi-class handled correctly
n_classes = len(le.classes_)
try:
    if n_classes == 2:
        auc_roc = roc_auc_score(y_test, y_prob[:, 1])
    else:
        auc_roc = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="weighted"
        )
except Exception as e:
    print(f"  [WARN] AUC-ROC error: {e}")
    auc_roc = 0.0

cm  = confusion_matrix(y_test, y_pred)
FP  = cm.sum(axis=0) - np.diag(cm)
FN  = cm.sum(axis=1) - np.diag(cm)
TP  = np.diag(cm)
TN  = cm.sum() - (FP + FN + TP)
fpr = (FP / (FP + TN + 1e-9)).mean() * 100
fnr = (FN / (FN + TP + 1e-9)).mean() * 100

real_metrics = {
    "Accuracy"           : round(accuracy,  4),
    "Precision"          : round(precision, 4),
    "Recall"             : round(recall,    4),
    "F1-Score"           : round(f1,        4),
    "AUC-ROC"            : round(auc_roc,   4),
    "Matthews CC"        : round(mcc,       4),
    "False Positive Rate": round(fpr,       4),
    "False Negative Rate": round(fnr,       4),
}

# Print comparison table
print()
print("=" * 72)
print("  RBSF FRAMEWORK — SYNTHETIC vs REAL-WORLD (TON_IoT)")
print("=" * 72)
print(f"  {'Metric':<28} {'Synthetic':>12} {'Real-World':>12} {'Difference':>12}")
print("  " + "─" * 68)

for metric, synth_val in SYNTHETIC_METRICS.items():
    real_val = real_metrics[metric]
    diff = real_val - synth_val
    sign = "+" if diff >= 0 else ""
    if metric in {"AUC-ROC", "Matthews CC"}:
        print(f"  {metric:<28} {synth_val:>12.4f} {real_val:>12.4f} {sign}{diff:>11.4f}")
    else:
        print(f"  {metric:<28} {synth_val:>11.2f}% {real_val:>11.2f}% {sign}{diff:>10.2f}%")

print("=" * 72)

print("\n  Per-Class Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# ─────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────
print("\n  Generating charts...")

# 1) Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("RBSF Real-World — Confusion Matrix (TON_IoT)", fontsize=14, pad=12)
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix_real.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/confusion_matrix_real.png")

# 2) ROC Curves
plt.figure(figsize=(10, 8))
for i, cls in enumerate(le.classes_):
    y_bin = (y_test == i).astype(int)
    if y_bin.sum() == 0:
        continue
    fpr_c, tpr_c, _ = roc_curve(y_bin, y_prob[:, i])
    roc_auc_c = auc(fpr_c, tpr_c)
    plt.plot(fpr_c, tpr_c, lw=1.5, label=f"{cls} (AUC={roc_auc_c:.3f})")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("RBSF Real-World — ROC Curves (TON_IoT)", fontsize=14)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/roc_curve_real.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/roc_curve_real.png")

# 3) Comparison chart
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("RBSF: Synthetic vs Real-World (TON_IoT)", fontsize=14, fontweight="bold")
for ax, m in zip(axes.flat, ["Accuracy", "Precision", "Recall", "F1-Score"]):
    vals = [SYNTHETIC_METRICS[m], real_metrics[m]]
    bars = ax.bar(["Synthetic\n(Script 3)", "Real-World\n(TON_IoT)"],
                  vals, color=["#2196F3", "#4CAF50"], width=0.5)
    ax.set_title(m, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 108); ax.set_ylabel("%")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
                f"{v:.2f}%", ha="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparison_chart.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/comparison_chart.png")

# 4) Feature importance
importances = rf.feature_importances_
feat_df = pd.DataFrame({
    "feature": selected_names[:len(importances)],
    "importance": importances
}).sort_values("importance", ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(feat_df["feature"][::-1], feat_df["importance"][::-1], color="#2196F3")
plt.xlabel("Feature Importance"); plt.title("Top 10 Feature Importances (TON_IoT)", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/feature_importance_real.png", dpi=150)
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR}/feature_importance_real.png")

# ─────────────────────────────────────────────
#  SAVE ARTIFACTS
# ─────────────────────────────────────────────
with open(f"{OUTPUT_DIR}/real_world_metrics.json", "w") as f:
    json.dump(real_metrics, f, indent=2)
with open(f"{OUTPUT_DIR}/rf_model_real.pkl", "wb") as f:
    pickle.dump(rf, f)
if iso:
    with open(f"{OUTPUT_DIR}/if_model_real.pkl", "wb") as f:
        pickle.dump(iso, f)
with open(f"{OUTPUT_DIR}/scaler_real.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(f"{OUTPUT_DIR}/label_encoder_real.pkl", "wb") as f:
    pickle.dump(le, f)
print(f"  ✓ Saved: all models + artifacts to {OUTPUT_DIR}/")

# Text report
report = [
    "=" * 60,
    "  RBSF IoT SECURITY FRAMEWORK — REAL-WORLD EVALUATION",
    "  Author: Хаппи Ондобо Стив | MTUCI 2026",
    "  Dataset: TON_IoT (Train_Test_Network.csv)",
    "=" * 60, "",
    "OVERALL METRICS (RANDOM FOREST — WEIGHTED AVERAGE)",
    "-" * 45,
    f"  Accuracy              : {real_metrics['Accuracy']:.4f}%",
    f"  Precision             : {real_metrics['Precision']:.4f}%",
    f"  Recall                : {real_metrics['Recall']:.4f}%",
    f"  F1-Score              : {real_metrics['F1-Score']:.4f}%",
    f"  AUC-ROC               : {real_metrics['AUC-ROC']:.4f}",
    f"  Matthews Corr. Coeff  : {real_metrics['Matthews CC']:.4f}",
    f"  False Positive Rate   : {real_metrics['False Positive Rate']:.4f}%",
    f"  False Negative Rate   : {real_metrics['False Negative Rate']:.4f}%",
    "",
    "3-FOLD CROSS-VALIDATION",
    "-" * 45,
    f"  CV Accuracy  : {cv_acc.mean()*100:.4f}% ± {cv_acc.std()*100:.4f}%",
    f"  CV F1-Score  : {cv_f1.mean()*100:.4f}% ± {cv_f1.std()*100:.4f}%",
    "", "=" * 60,
]
with open(f"{OUTPUT_DIR}/real_world_metrics_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"  ✓ Saved: {OUTPUT_DIR}/real_world_metrics_report.txt")

# ─────────────────────────────────────────────
#  FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n")
print("=" * 60)
print("  ✓ REAL-WORLD VALIDATION COMPLETE")
print("=" * 60)
print(f"  Accuracy   : {real_metrics['Accuracy']:.2f}%")
print(f"  F1-Score   : {real_metrics['F1-Score']:.2f}%")
print(f"  AUC-ROC    : {real_metrics['AUC-ROC']:.4f}")
print(f"  Matthews CC: {real_metrics['Matthews CC']:.4f}")
print(f"  Outputs    : {OUTPUT_DIR}/")
print("=" * 60)

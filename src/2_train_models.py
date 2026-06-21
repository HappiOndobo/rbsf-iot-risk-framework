# ============================================================
# SCRIPT 2 — 2_train_models.py
# RBSF IoT Security Framework — Model Training
# Author: Happi Ondobo Steve | MTUCI 2026
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Loads the preprocessed training data from Script 1
#   2. Applies SMOTE to fix class imbalance
#   3. Trains a Random Forest classifier (supervised ML)
#      - Uses 5-fold cross-validation to check stability
#   4. Trains an Isolation Forest (unsupervised ML)
#      - Trained only on normal traffic
#   5. Saves both models to files
# ============================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import pickle
import time
import os

np.random.seed(42)

print("=" * 60)
print("  RBSF — Step 2: Model Training")
print("=" * 60)

# ── LOAD DATA ─────────────────────────────────────────────────
print("\n[1/5] Loading preprocessed data...")

X_train        = pd.read_csv("rbsf_output/X_train.csv")
y_train        = pd.read_csv("rbsf_output/y_train.csv").squeeze()
X_train_normal = pd.read_csv("rbsf_output/X_train_normal.csv")

with open("rbsf_output/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

print(f"    Training samples : {len(X_train):,}")
print(f"    Features         : {X_train.shape[1]}")
print(f"    Classes          : {len(label_encoder.classes_)}")
print(f"    Normal-only rows : {len(X_train_normal):,}")

# Show class distribution before balancing
print("\n    Class distribution BEFORE balancing:")
for cls_id, count in sorted(y_train.value_counts().items()):
    cls_name = label_encoder.classes_[cls_id]
    pct = count / len(y_train) * 100
    print(f"      {cls_name:<12} {count:>5}  ({pct:.1f}%)")

# ── STEP 2: SMOTE — Fix Class Imbalance ──────────────────────
print("\n[2/5] Applying SMOTE to balance classes...")

# SMOTE creates synthetic examples of minority classes
# so every class has at least 300 samples in training
smote = SMOTE(
    sampling_strategy="auto",   # oversample all minority classes
    k_neighbors=5,              # use 5 nearest neighbors to create new samples
    random_state=42
)

X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

print(f"    Before SMOTE: {len(X_train):,} records")
print(f"    After  SMOTE: {len(X_balanced):,} records")
print("\n    Class distribution AFTER balancing:")
for cls_id in sorted(np.unique(y_balanced)):
    cls_name = label_encoder.classes_[cls_id]
    count    = np.sum(y_balanced == cls_id)
    print(f"      {cls_name:<12} {count:>5}")

# ── STEP 3: Train Random Forest ───────────────────────────────
print("\n[3/5] Training Random Forest classifier...")
print("    (This may take 30–60 seconds)")

# These settings were chosen through cross-validation testing
rf_model = RandomForestClassifier(
    n_estimators=200,       # 200 decision trees in the forest
    max_depth=15,           # each tree can be at most 15 levels deep
    min_samples_split=5,    # need at least 5 samples to split a node
    min_samples_leaf=2,     # each leaf must have at least 2 samples
    max_features="sqrt",    # each tree uses sqrt(n_features) features per split
    class_weight="balanced",# give more weight to rare attack classes
    n_jobs=-1,              # use all CPU cores
    random_state=42
)

# 5-Fold Cross-Validation
# This trains and tests the model 5 times on different data splits
# to make sure it works consistently, not just by luck
print("\n    Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = cross_validate(
    rf_model, X_balanced, y_balanced,
    cv=cv,
    scoring=["accuracy", "f1_weighted"],
    return_train_score=False,
    n_jobs=-1
)

print(f"\n    Cross-Validation Results:")
print(f"    {'Fold':<8} {'Accuracy':>10} {'F1-Score':>10}")
print(f"    {'-'*32}")
for i, (acc, f1) in enumerate(zip(
    cv_results["test_accuracy"],
    cv_results["test_f1_weighted"]
), 1):
    print(f"    Fold {i:<4} {acc*100:>9.2f}%  {f1*100:>9.2f}%")

print(f"    {'-'*32}")
print(f"    {'Mean':<8} {cv_results['test_accuracy'].mean()*100:>9.2f}%  "
      f"{cv_results['test_f1_weighted'].mean()*100:>9.2f}%")
print(f"    {'Std':<8} ±{cv_results['test_accuracy'].std()*100:>8.2f}%  "
      f"±{cv_results['test_f1_weighted'].std()*100:>8.2f}%")

# Now train the FINAL model on ALL balanced training data
print("\n    Training final model on full training set...")
t0 = time.time()
rf_model.fit(X_balanced, y_balanced)
elapsed = time.time() - t0
print(f"    Training complete in {elapsed:.1f} seconds")

# Save feature importances
feat_importance = pd.Series(
    rf_model.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

feat_importance.to_csv("rbsf_output/feature_importance.csv", header=["importance"])
print(f"\n    Top 10 features by importance:")
for feat, imp in feat_importance.head(10).items():
    bar = "█" * int(imp * 200)
    print(f"      {feat:<30} {imp:.4f}  {bar}")

# Save Random Forest model
with open("rbsf_output/rf_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("\n    Saved → rbsf_output/rf_model.pkl")

# ── STEP 4: Train Isolation Forest ───────────────────────────
print("\n[4/5] Training Isolation Forest (anomaly detector)...")
print("    Trained on NORMAL traffic only")

# Isolation Forest learns what normal behavior looks like.
# When it sees something very different, it gives a high anomaly score.
if_model = IsolationForest(
    n_estimators=150,       # 150 isolation trees
    max_samples=256,        # each tree uses 256 random samples
    contamination=0.05,     # we expect ~5% of data to be anomalous
    max_features=1.0,       # use all features
    random_state=42,
    n_jobs=-1
)

t0 = time.time()
if_model.fit(X_train_normal)
elapsed = time.time() - t0
print(f"    Training complete in {elapsed:.1f} seconds")
print(f"    Trained on {len(X_train_normal):,} normal traffic records")

# Save Isolation Forest model
with open("rbsf_output/if_model.pkl", "wb") as f:
    pickle.dump(if_model, f)
print("    Saved → rbsf_output/if_model.pkl")

# Quick sanity check: test IF on training normal data
if_scores_normal = if_model.score_samples(X_train_normal)
print(f"\n    Anomaly score on normal traffic:")
print(f"      Mean  : {if_scores_normal.mean():.4f}  (closer to 0 = normal)")
print(f"      Std   : {if_scores_normal.std():.4f}")

# ── STEP 5: Save Training Summary ────────────────────────────
print("\n[5/5] Saving training summary...")

summary = {
    "cv_accuracy_mean":  float(cv_results["test_accuracy"].mean()),
    "cv_accuracy_std":   float(cv_results["test_accuracy"].std()),
    "cv_f1_mean":        float(cv_results["test_f1_weighted"].mean()),
    "cv_f1_std":         float(cv_results["test_f1_weighted"].std()),
    "n_train_records":   int(len(X_balanced)),
    "n_features":        int(X_train.shape[1]),
    "n_classes":         int(len(label_encoder.classes_)),
    "classes":           list(label_encoder.classes_),
    "rf_n_estimators":   200,
    "if_n_estimators":   150,
    "if_contamination":  0.05,
}

import json
with open("rbsf_output/training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("    Saved → rbsf_output/training_summary.json")

# ── FINAL SUMMARY ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TRAINING COMPLETE")
print("=" * 60)
print(f"\n  Random Forest:")
print(f"    Cross-Val Accuracy : {cv_results['test_accuracy'].mean()*100:.2f}% "
      f"(±{cv_results['test_accuracy'].std()*100:.2f}%)")
print(f"    Cross-Val F1-Score : {cv_results['test_f1_weighted'].mean()*100:.2f}% "
      f"(±{cv_results['test_f1_weighted'].std()*100:.2f}%)")
print(f"\n  Isolation Forest:")
print(f"    Trained on {len(X_train_normal):,} normal records")
print(f"\n  Files saved:")
print(f"    rf_model.pkl            — Random Forest model")
print(f"    if_model.pkl            — Isolation Forest model")
print(f"    feature_importance.csv  — feature ranking")
print(f"    training_summary.json   — training metrics")
print(f"\n  Next step: run  python 3_evaluate.py")
print("=" * 60)

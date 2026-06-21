# ============================================================
# SCRIPT 1 — preprocess.py
# RBSF IoT Security Framework — Data Preprocessing
# Author: Happi Ondobo Steve | MTUCI 2026
# ============================================================
# WHAT THIS SCRIPT DOES:
#   1. Generates a realistic synthetic IoT security dataset
#   2. Cleans the data (removes duplicates, fixes missing values)
#   3. Encodes text columns into numbers
#   4. Normalizes all numbers to the same scale (0 to 1)
#   5. Selects the best features
#   6. Splits into training and test sets
#   7. Saves everything to files for the next scripts
# ============================================================

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
import pickle
import os

# Fix random seed so results are the same every time you run
np.random.seed(42)

print("=" * 60)
print("  RBSF — Step 1: Data Preprocessing")
print("=" * 60)

# ── STEP 1: Generate Synthetic IoT Dataset ───────────────────
print("\n[1/6] Generating synthetic IoT dataset...")

# Number of records per class
CLASS_SIZES = {
    "Normal":    5000,
    "DDoS":      1500,
    "DoS":        800,
    "Botnet":     600,
    "Scanning":   400,
    "Backdoor":   200,
    "Injection":  150,
    "Password":   120,
    "Ransomware":  80,
    "MitM":        50,
}

PROTOCOLS  = ["tcp", "udp", "icmp", "mqtt", "coap"]
SERVICES   = ["http", "ftp", "ssh", "smtp", "dns", "other"]
FLAGS      = ["SF", "S0", "REJ", "RSTO", "SH", "OTH"]
CONN_STATE = ["established", "closed", "failed", "half_open"]

def make_records(label, n):
    """
    Generate n realistic network traffic records for a given label.
    Each attack type has different statistical patterns.
    """
    is_attack = label != "Normal"

    # --- Base traffic volume ---
    if label == "DDoS":
        # DDoS = massive packet floods, very short duration
        duration    = np.random.exponential(0.5, n)
        src_bytes   = np.random.randint(50, 200, n).astype(float)
        dst_bytes   = np.random.randint(0, 50, n).astype(float)
        pkt_count   = np.random.randint(1000, 50000, n).astype(float)
    elif label == "DoS":
        duration    = np.random.exponential(1.0, n)
        src_bytes   = np.random.randint(100, 500, n).astype(float)
        dst_bytes   = np.random.randint(0, 100, n).astype(float)
        pkt_count   = np.random.randint(500, 10000, n).astype(float)
    elif label == "Botnet":
        # Botnet = regular beaconing, moderate traffic
        duration    = np.random.uniform(30, 300, n)
        src_bytes   = np.random.randint(200, 1000, n).astype(float)
        dst_bytes   = np.random.randint(200, 1000, n).astype(float)
        pkt_count   = np.random.randint(20, 200, n).astype(float)
    elif label == "Scanning":
        # Scanning = many short connections to different ports
        duration    = np.random.exponential(0.1, n)
        src_bytes   = np.random.randint(40, 100, n).astype(float)
        dst_bytes   = np.random.randint(0, 40, n).astype(float)
        pkt_count   = np.random.randint(1, 10, n).astype(float)
    elif label == "Backdoor":
        duration    = np.random.uniform(60, 3600, n)
        src_bytes   = np.random.randint(500, 5000, n).astype(float)
        dst_bytes   = np.random.randint(500, 5000, n).astype(float)
        pkt_count   = np.random.randint(50, 500, n).astype(float)
    elif label == "Injection":
        duration    = np.random.exponential(2.0, n)
        src_bytes   = np.random.randint(200, 2000, n).astype(float)
        dst_bytes   = np.random.randint(100, 800, n).astype(float)
        pkt_count   = np.random.randint(5, 50, n).astype(float)
    elif label == "Password":
        # Brute force = many failed logins
        duration    = np.random.exponential(0.3, n)
        src_bytes   = np.random.randint(60, 120, n).astype(float)
        dst_bytes   = np.random.randint(60, 120, n).astype(float)
        pkt_count   = np.random.randint(2, 8, n).astype(float)
    elif label == "Ransomware":
        # Ransomware = large file reads, then encrypted upload
        duration    = np.random.uniform(10, 120, n)
        src_bytes   = np.random.randint(5000, 50000, n).astype(float)
        dst_bytes   = np.random.randint(1000, 10000, n).astype(float)
        pkt_count   = np.random.randint(100, 1000, n).astype(float)
    elif label == "MitM":
        duration    = np.random.uniform(5, 60, n)
        src_bytes   = np.random.randint(300, 3000, n).astype(float)
        dst_bytes   = np.random.randint(300, 3000, n).astype(float)
        pkt_count   = np.random.randint(10, 100, n).astype(float)
    else:
        # Normal traffic — wide natural variation
        duration    = np.random.exponential(15, n)
        src_bytes   = np.random.randint(100, 8000, n).astype(float)
        dst_bytes   = np.random.randint(100, 8000, n).astype(float)
        pkt_count   = np.random.randint(5, 300, n).astype(float)

    # Add small random noise to all volumes
    src_bytes += np.random.normal(0, 10, n)
    dst_bytes += np.random.normal(0, 10, n)

    # --- Connection behavior ---
    if label in ("DDoS", "DoS"):
        serror_rate   = np.random.uniform(0.7, 1.0, n)
        rerror_rate   = np.random.uniform(0.0, 0.3, n)
        same_srv_rate = np.random.uniform(0.8, 1.0, n)
        diff_srv_rate = np.random.uniform(0.0, 0.1, n)
        failed_logins = np.zeros(n)
    elif label == "Scanning":
        serror_rate   = np.random.uniform(0.3, 0.8, n)
        rerror_rate   = np.random.uniform(0.3, 0.8, n)
        same_srv_rate = np.random.uniform(0.0, 0.2, n)
        diff_srv_rate = np.random.uniform(0.7, 1.0, n)
        failed_logins = np.zeros(n)
    elif label == "Password":
        serror_rate   = np.random.uniform(0.0, 0.2, n)
        rerror_rate   = np.random.uniform(0.5, 1.0, n)
        same_srv_rate = np.random.uniform(0.9, 1.0, n)
        diff_srv_rate = np.random.uniform(0.0, 0.1, n)
        failed_logins = np.random.randint(3, 20, n).astype(float)
    else:
        serror_rate   = np.random.uniform(0.0, 0.1, n) if not is_attack else np.random.uniform(0.1, 0.5, n)
        rerror_rate   = np.random.uniform(0.0, 0.1, n) if not is_attack else np.random.uniform(0.1, 0.4, n)
        same_srv_rate = np.random.uniform(0.5, 1.0, n)
        diff_srv_rate = np.random.uniform(0.0, 0.3, n)
        failed_logins = np.zeros(n)

    # --- Derived statistical features ---
    byte_ratio        = np.where(dst_bytes > 0, src_bytes / (dst_bytes + 1), src_bytes)
    pkt_rate          = np.where(duration > 0, pkt_count / (duration + 0.001), pkt_count)
    payload_entropy   = np.random.uniform(3.5, 8.0, n) if not is_attack else np.random.uniform(1.0, 7.5, n)
    inter_arrival_std = np.random.uniform(0.001, 0.5, n) if label != "Botnet" else np.random.uniform(0.0001, 0.01, n)

    # --- Categorical features ---
    if label in ("DDoS", "DoS"):
        protocol = np.random.choice(["icmp", "udp"], n, p=[0.6, 0.4])
    elif label == "Scanning":
        protocol = np.random.choice(["tcp", "udp"], n, p=[0.7, 0.3])
    else:
        protocol = np.random.choice(PROTOCOLS, n)

    service    = np.random.choice(SERVICES, n)
    flag       = np.random.choice(FLAGS, n,
                    p=[0.7,0.1,0.1,0.05,0.03,0.02] if not is_attack
                      else [0.3,0.3,0.2,0.1,0.05,0.05])
    conn_state = np.random.choice(CONN_STATE, n,
                    p=[0.7,0.2,0.05,0.05] if not is_attack
                      else [0.2,0.2,0.4,0.2])

    # --- Device telemetry (simulated) ---
    if label in ("DDoS", "DoS", "Ransomware"):
        cpu_usage  = np.random.uniform(70, 100, n)
        mem_usage  = np.random.uniform(60, 100, n)
    elif label == "Botnet":
        cpu_usage  = np.random.uniform(20, 50, n)
        mem_usage  = np.random.uniform(30, 60, n)
    else:
        cpu_usage  = np.random.uniform(5, 40, n)
        mem_usage  = np.random.uniform(10, 50, n)

    # --- Unique destinations contacted (scanning indicator) ---
    if label == "Scanning":
        unique_dst = np.random.randint(50, 500, n).astype(float)
        unique_ports = np.random.randint(100, 1000, n).astype(float)
    elif label in ("DDoS", "DoS", "Botnet"):
        unique_dst   = np.random.randint(1, 10, n).astype(float)
        unique_ports = np.random.randint(1, 5, n).astype(float)
    else:
        unique_dst   = np.random.randint(1, 20, n).astype(float)
        unique_ports = np.random.randint(1, 30, n).astype(float)

    df = pd.DataFrame({
        "duration":         np.clip(duration, 0, None),
        "src_bytes":        np.clip(src_bytes, 0, None),
        "dst_bytes":        np.clip(dst_bytes, 0, None),
        "pkt_count":        np.clip(pkt_count, 0, None),
        "pkt_rate":         np.clip(pkt_rate, 0, None),
        "byte_ratio":       np.clip(byte_ratio, 0, None),
        "serror_rate":      np.clip(serror_rate, 0, 1),
        "rerror_rate":      np.clip(rerror_rate, 0, 1),
        "same_srv_rate":    np.clip(same_srv_rate, 0, 1),
        "diff_srv_rate":    np.clip(diff_srv_rate, 0, 1),
        "failed_logins":    np.clip(failed_logins, 0, None),
        "payload_entropy":  payload_entropy,
        "inter_arrival_std":inter_arrival_std,
        "cpu_usage":        np.clip(cpu_usage, 0, 100),
        "mem_usage":        np.clip(mem_usage, 0, 100),
        "unique_dst":       unique_dst,
        "unique_ports":     unique_ports,
        "protocol":         protocol,
        "service":          service,
        "flag":             flag,
        "conn_state":       conn_state,
        "label":            label,
    })
    return df

# Build the full dataset
frames = [make_records(lbl, n) for lbl, n in CLASS_SIZES.items()]
df_raw = pd.concat(frames, ignore_index=True)

# Shuffle rows
df_raw = df_raw.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"    Dataset created: {df_raw.shape[0]:,} records, {df_raw.shape[1]} columns")
print(f"    Attack classes : {df_raw['label'].nunique()}")
print(f"\n    Class distribution:")
for cls, cnt in df_raw['label'].value_counts().items():
    pct = cnt / len(df_raw) * 100
    bar = "█" * int(pct / 2)
    print(f"      {cls:<12} {cnt:>5} records  {pct:5.1f}%  {bar}")

# Save raw dataset
os.makedirs("rbsf_output", exist_ok=True)
df_raw.to_csv("rbsf_output/dataset_raw.csv", index=False)
print("\n    Saved → rbsf_output/dataset_raw.csv")

# ── STEP 2: Clean the Data ────────────────────────────────────
print("\n[2/6] Cleaning data...")

df = df_raw.copy()
before = len(df)

# Remove duplicate rows
df = df.drop_duplicates()
print(f"    Duplicates removed : {before - len(df)}")

# Fix missing values
missing_before = df.isnull().sum().sum()
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=["object"]).columns:
    if col != "label":
        df[col] = df[col].fillna(df[col].mode()[0])
print(f"    Missing values fixed: {missing_before}")

# Clip infinite values
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print(f"    Infinite values handled")
print(f"    Remaining records: {len(df):,}")

# ── STEP 3: Encode Categorical Columns ───────────────────────
print("\n[3/6] Encoding categorical features...")

# One-hot encode: protocol, service, flag, conn_state
cat_cols = ["protocol", "service", "flag", "conn_state"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
print(f"    One-hot encoding done → {df.shape[1]-1} features (before selection)")

# Encode the label column to numbers
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])

# Save the label encoder (needed later for prediction)
with open("rbsf_output/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print(f"    Label classes: {list(label_encoder.classes_)}")

# ── STEP 4: Normalize Numerical Features ─────────────────────
print("\n[4/6] Normalizing features to [0, 1] scale...")

# Separate features and label
X = df.drop(columns=["label", "label_encoded"])
y = df["label_encoded"]
y_names = df["label"]

# Apply Min-Max normalization
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=X.columns
)

# Save the scaler (needed to scale new data the same way)
with open("rbsf_output/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"    All {X_scaled.shape[1]} features normalized to [0, 1]")
print(f"    Scaler saved → rbsf_output/scaler.pkl")

# ── STEP 5: Select Best Features ─────────────────────────────
print("\n[5/6] Selecting best features using Mutual Information...")

# Score each feature by how useful it is for predicting the label
mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_series = pd.Series(mi_scores, index=X_scaled.columns).sort_values(ascending=False)

# Keep top 30 features
TOP_N = 30
selected_features = mi_series.head(TOP_N).index.tolist()
X_selected = X_scaled[selected_features]

# Save feature list
with open("rbsf_output/selected_features.pkl", "wb") as f:
    pickle.dump(selected_features, f)

print(f"    Top {TOP_N} features selected out of {X_scaled.shape[1]}")
print(f"\n    Top 10 most useful features:")
for feat, score in mi_series.head(10).items():
    bar = "█" * int(score * 30)
    print(f"      {feat:<30} {score:.4f}  {bar}")

# ── STEP 6: Split into Train and Test Sets ───────────────────
print("\n[6/6] Splitting data: 80% train / 20% test...")

X_train, X_test, y_train, y_test, names_train, names_test = train_test_split(
    X_selected, y, y_names,
    test_size=0.20,
    random_state=42,
    stratify=y          # keep same class proportions in both splits
)

# Save all splits
X_train.to_csv("rbsf_output/X_train.csv", index=False)
X_test.to_csv("rbsf_output/X_test.csv", index=False)
y_train.to_csv("rbsf_output/y_train.csv", index=False)
y_test.to_csv("rbsf_output/y_test.csv", index=False)
names_test.to_csv("rbsf_output/y_test_names.csv", index=False)

# Also save normal-only training data for Isolation Forest
X_train_normal = X_train[y_train == label_encoder.transform(["Normal"])[0]]
X_train_normal.to_csv("rbsf_output/X_train_normal.csv", index=False)

print(f"    Training set : {len(X_train):,} records")
print(f"    Test set     : {len(X_test):,} records")
print(f"    Normal-only  : {len(X_train_normal):,} records (for Isolation Forest)")

# ── FINAL SUMMARY ────────────────────────────────────────────
print("\n" + "=" * 60)
print("  PREPROCESSING COMPLETE")
print("=" * 60)
print(f"\n  Files saved in folder: rbsf_output/")
print(f"    dataset_raw.csv         — original generated dataset")
print(f"    X_train.csv             — training features")
print(f"    X_test.csv              — test features")
print(f"    y_train.csv             — training labels")
print(f"    y_test.csv              — test labels (numbers)")
print(f"    y_test_names.csv        — test labels (names)")
print(f"    X_train_normal.csv      — normal traffic only")
print(f"    scaler.pkl              — normalization scaler")
print(f"    label_encoder.pkl       — label encoder")
print(f"    selected_features.pkl   — list of 30 best features")
print(f"\n  Next step: run  python 2_train_models.py")
print("=" * 60)

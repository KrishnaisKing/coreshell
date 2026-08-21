"""
Phase 4 (real candidates): K-Means group-safe split + XGBoost training.

WHY GROUP-BASED SPLIT (not row-based random split):
Each of the 44 candidates contributes ~50 rows (one per device-sweep
point: Vmax, shell_thick_nm, core_radius_nm, Nt_cm3 combo). Rows from
the SAME candidate share the same dEc_eV/dEv_eV/eps_shell -- they are
highly correlated, not independent samples. A random row-level
train/test split would put some rows from a candidate in train and
other rows from the SAME candidate in test, letting the model
"memorize" that candidate's fixed material properties rather than
learn a generalizable structure->behavior relationship. This is
exactly the kind of leakage the friend's earlier project fell into
(Image 3's suspiciously perfect parity plot).

K-Means here clusters the 44 CANDIDATES (not the 2200 rows) by their
material-level descriptors, then whole clusters are assigned to
train/test -- guaranteeing no candidate's rows appear in both.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import xgboost as xgb

RANDOM_STATE = 42

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("rs_training_data_real_candidates.csv")
print(f"Loaded {len(df)} rows.")

candidate_id_col = "candidate_id" if "candidate_id" in df.columns else None
if candidate_id_col is None:
    # fall back: build a candidate id from the material-level columns if
    # no explicit id column exists in the Phase 4-real output
    df["candidate_id"] = df["formula_core"].astype(str) + "__" + df["formula_shell"].astype(str)
    candidate_id_col = "candidate_id"

n_candidates = df[candidate_id_col].nunique()
print(f"{n_candidates} unique candidates, ~{len(df)/n_candidates:.0f} rows each.")

# ---------------------------------------------------------------
# 2. K-Means on CANDIDATE-LEVEL descriptors (one row per candidate)
# ---------------------------------------------------------------
material_cols = [c for c in ["dE_LUMO_eV", "dE_HOMO_eV", "confinement_score_eV",
                              "lattice_mismatch_pct"] if c in df.columns]
if not material_cols:
    raise SystemExit(f"None of the expected material-level columns found. "
                      f"Available columns: {df.columns.tolist()}")

candidate_level = df.groupby(candidate_id_col)[material_cols].first().reset_index()
print(f"Clustering on: {material_cols}")

scaler = StandardScaler()
X_cluster = scaler.fit_transform(candidate_level[material_cols])

# small candidate pool (44) -> keep cluster count modest so each
# cluster has enough members to be meaningfully assignable to train/test
n_clusters = min(8, n_candidates // 4)
km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
candidate_level["cluster"] = km.fit_predict(X_cluster)
print(f"K-Means: {n_clusters} clusters over {n_candidates} candidates.")
print(candidate_level["cluster"].value_counts().sort_index())

# ---------------------------------------------------------------
# 3. Assign whole clusters to train/test (~75/25 split by candidate count)
# ---------------------------------------------------------------
rng = np.random.default_rng(RANDOM_STATE)
cluster_sizes = candidate_level["cluster"].value_counts()
clusters_shuffled = cluster_sizes.index.to_numpy().copy()
rng.shuffle(clusters_shuffled)

target_test_n = max(1, round(n_candidates * 0.25))
test_clusters, running = [], 0
for c in clusters_shuffled:
    if running >= target_test_n:
        break
    test_clusters.append(c)
    running += cluster_sizes[c]

candidate_level["split"] = np.where(
    candidate_level["cluster"].isin(test_clusters), "test", "train"
)
print(f"\nCandidates -> train: {(candidate_level['split']=='train').sum()}, "
      f"test: {(candidate_level['split']=='test').sum()}")

df = df.merge(candidate_level[[candidate_id_col, "cluster", "split"]],
              on=candidate_id_col, how="left")

train_df = df[df["split"] == "train"].copy()
test_df = df[df["split"] == "test"].copy()
print(f"Rows -> train: {len(train_df)}, test: {len(test_df)}")

# sanity check: zero candidate overlap between train/test
overlap = set(train_df[candidate_id_col]) & set(test_df[candidate_id_col])
assert len(overlap) == 0, f"LEAK: {len(overlap)} candidates appear in both splits!"
print("Verified: zero candidate overlap between train and test.")

# ---------------------------------------------------------------
# 4. Feature set for XGBoost (device-sweep params + material descriptors)
# ---------------------------------------------------------------
feature_cols = [c for c in [
    "dE_LUMO_eV", "dE_HOMO_eV", "shell_thick_nm", "core_radius_nm",
    "Nt_cm3", "eps_shell", "Vmax_V", "lattice_mismatch_pct",
] if c in df.columns]
print(f"\nFeature columns used: {feature_cols}")

targets = {
    "hysteresis_window_V": "hysteresis_window_V",
    "log10_on_off_ratio": "on_off_ratio",       # train on log if raw present but no log col
    "log10_retention_tau_s": "retention_tau_s",
}

results = {}
for target_name, fallback_col in targets.items():
    if target_name in df.columns:
        ycol = target_name
    elif fallback_col in df.columns:
        # derive log column on the fly if the raw sweep script didn't save it
        df[f"log10_{fallback_col}"] = np.log10(df[fallback_col].clip(lower=1e-12))
        train_df[f"log10_{fallback_col}"] = np.log10(train_df[fallback_col].clip(lower=1e-12))
        test_df[f"log10_{fallback_col}"] = np.log10(test_df[fallback_col].clip(lower=1e-12))
        ycol = f"log10_{fallback_col}" if "log10" in target_name else fallback_col
    else:
        print(f"Skipping {target_name} -- no matching column found.")
        continue

    X_train, y_train = train_df[feature_cols], train_df[ycol]
    X_test, y_test = test_df[feature_cols], test_df[ycol]

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    r2 = r2_score(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    results[ycol] = {"r2": r2, "mae": mae}
    print(f"\n[{ycol}]  R^2 = {r2:.4f}   MAE = {mae:.4f}   "
          f"(n_train={len(X_train)}, n_test={len(X_test)})")

    model.save_model(f"xgb_model_{ycol}.json")

    out = test_df[[candidate_id_col]].copy()
    out["y_true"] = y_test.values
    out["y_pred"] = pred
    out.to_csv(f"predictions_{ycol}.csv", index=False)

print("\nSaved models (xgb_model_*.json) and per-target prediction "
      "files (predictions_*.csv) for parity-plot inspection.")
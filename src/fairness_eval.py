"""
fairness_eval.py  (v4 — all 4 models)
--------------------------------------
Run:  python fairness_eval.py

Prerequisites (run each server first to save models):
  python server_fed.py       → results/federated_lr_final.pkl
                               results/federated_dt_final.pkl
                               results/federated_rf_final.pkl
  python server_xgb_fed.py   → results/federated_xgb_final.json

Outputs:
  results/fairness_<model>.csv   — per-model per-group metrics
  results/fairness_summary.csv   — DP ratio + EO ratio across all models
"""

import pandas as pd
import numpy as np
import os, sys, pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_ratio,
    equalized_odds_ratio,
    false_negative_rate,
    false_positive_rate,
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import (clean_los, DROP_COLS, BINARY_COLS,
                        CATEGORICAL_COLS, get_feature_schema)

DATA_DIR  = "../data/hospitals"
DEMO_COLS = ["race", "gender", "age_group", "payment_typology_1"]

MODELS = {
    "lr":  ("../results/federated_lr_final.pkl",   "sklearn"),
    "dt":  ("../results/federated_dt_final.pkl",   "sklearn"),
    "rf":  ("../results/federated_rf_final.pkl",   "sklearn"),
    "xgb": ("../results/federated_xgb_final.json", "xgboost"),
}


# ── model loading and prediction ─────────────────────────────────────────────

def load_model(path, framework):
    if framework == "sklearn":
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        import xgboost as xgb
        bst = xgb.Booster()
        bst.load_model(path)
        return bst


def get_predictions(model, framework, X):
    if framework == "sklearn":
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
    else:
        import xgboost as xgb
        dmat   = xgb.DMatrix(X)
        y_prob = model.predict(dmat)
        y_pred = (y_prob > 0.5).astype(int)
    return y_pred, y_prob


# ── per-hospital data loading (mirrors each client exactly) ──────────────────

def load_federated_test_pool(data_dir, feature_schema):
    """
    Processes each hospital file exactly as client_fed.py does:
      - per-hospital MinMaxScaler fit on train split
      - 70/30 split with random_state=42
    Returns pooled X_test (scaled), y_test, demo_test (raw demographic strings).
    """
    X_parts, y_parts, demo_parts = [], [], []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".csv"):
            continue

        df = pd.read_csv(os.path.join(data_dir, fname), low_memory=False)
        df["length_of_stay"] = df["length_of_stay"].apply(clean_los)
        df = df.dropna(subset=["length_of_stay"])

        y = (df["length_of_stay"] > 7).astype(int).values
        df = df.drop(columns=["length_of_stay"])
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])

        # save demographics BEFORE any encoding
        demo = pd.DataFrame()
        for c in DEMO_COLS:
            if c in df.columns:
                demo[c] = df[c].copy()
        demo["gender"] = demo["gender"].map(
            {"M": "Male", "F": "Female", "U": "Unknown"}
        )

        for col, mapping in BINARY_COLS.items():
            if col in df.columns:
                df[col] = df[col].map(mapping).fillna(0).astype(int)

        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        df = pd.get_dummies(df, columns=cat_cols)

        for col in feature_schema:
            if col not in df.columns:
                df[col] = 0
        df = df[feature_schema]
        X = df.values

        idx = np.arange(len(y))
        idx_tr, idx_te, y_tr, y_te = train_test_split(
            idx, y, test_size=0.3, random_state=42
        )

        scaler = MinMaxScaler()
        scaler.fit(X[idx_tr])
        X_te_scaled = scaler.transform(X[idx_te])

        X_parts.append(X_te_scaled)
        y_parts.append(y_te)
        demo_parts.append(demo.iloc[idx_te].reset_index(drop=True))

    return (np.vstack(X_parts),
            np.concatenate(y_parts),
            pd.concat(demo_parts, ignore_index=True))


# ── fairness metrics ──────────────────────────────────────────────────────────

def eval_group(y_true, y_pred, y_prob, sensitive):
    mf_bin = MetricFrame(
        metrics={"FNR": false_negative_rate, "FPR": false_positive_rate},
        y_true=y_true, y_pred=y_pred,
        sensitive_features=sensitive,
    )
    mf_auc = MetricFrame(
        metrics={"AUROC": roc_auc_score},
        y_true=y_true, y_pred=y_prob,
        sensitive_features=sensitive,
    )
    counts     = sensitive.value_counts().rename("N")
    per_group  = mf_bin.by_group.join(mf_auc.by_group).join(counts)
    dp_r = demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive)
    eo_r = equalized_odds_ratio(y_true,    y_pred, sensitive_features=sensitive)
    return per_group, dp_r, eo_r


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building feature schema...")
    feature_schema = get_feature_schema(DATA_DIR)

    print("Loading per-hospital test pool...")
    X_test, y_test, demo_test = load_federated_test_pool(DATA_DIR, feature_schema)
    print(f"  Pooled records: {len(y_test):,} | long-stay: {y_test.mean():.1%}")

    os.makedirs("../results", exist_ok=True)
    summary_rows = []

    for model_name, (model_path, framework) in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\n[SKIP] {model_name} — model file not found: {model_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  {model_name.upper()} ({framework})")
        print(f"{'='*60}")

        model  = load_model(model_path, framework)
        y_pred, y_prob = get_predictions(model, framework, X_test)
        overall_auroc  = roc_auc_score(y_test, y_prob)
        print(f"  Overall AUROC: {overall_auroc:.4f}")

        all_results = []

        for col in DEMO_COLS:
            if col not in demo_test.columns:
                continue

            per_group, dp_r, eo_r = eval_group(
                y_test, y_pred, y_prob, demo_test[col]
            )

            print(f"\n  {col.replace('_',' ').title()}")
            print(f"  DP ratio: {dp_r:.4f}  |  EO ratio: {eo_r:.4f}")
            print(per_group[["N","AUROC","FNR","FPR"]].round(4).to_string())

            tmp = per_group.copy()
            tmp["model"]                   = model_name
            tmp["demographic_variable"]    = col
            tmp["demographic_parity_ratio"] = round(dp_r, 4)
            tmp["equalized_odds_ratio"]     = round(eo_r, 4)
            tmp.index.name = "group"
            all_results.append(tmp.reset_index())

            summary_rows.append({
                "model":     model_name,
                "dimension": col,
                "dp_ratio":  round(dp_r, 4),
                "eo_ratio":  round(eo_r, 4),
                "flag":      "below_0.80" if dp_r < 0.80 else "ok",
            })

        out = pd.concat(all_results, ignore_index=True)
        cols = ["model","demographic_variable","group","N",
                "AUROC","FNR","FPR",
                "demographic_parity_ratio","equalized_odds_ratio"]
        out = out[[c for c in cols if c in out.columns]]
        out.to_csv(f"../results/fairness_{model_name}.csv", index=False)

    # summary across all models
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("../results/fairness_summary.csv", index=False)

    print("\n" + "="*60)
    print("  SUMMARY — DP ratio by model and dimension")
    print("="*60)
    if not summary.empty:
        pivot = summary.pivot(index="dimension", columns="model", values="dp_ratio")
        print(pivot.round(4).to_string())
    print("\nFiles saved to results/fairness_<model>.csv + fairness_summary.csv")


if __name__ == "__main__":
    main()
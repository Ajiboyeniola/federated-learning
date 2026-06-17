import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge

DATA_DIR = "../data/hospitals"

DROP_COLS = [
    "hospital_service_area", "hospital_county", "operating_certificate_number",
    "permanent_facility_id", "facility_name", "zip_code_3_digits", "discharge_year",
    "ccs_diagnosis_description", "ccs_procedure_description", "apr_drg_description",
    "apr_mdc_description", "apr_severity_of_illness_description",
    "payment_typology_2", "payment_typology_3", "total_charges", "total_costs"
]

BINARY_COLS = {
    "gender": {"M": 1, "F": 0, "U": 0},
    "emergency_department_indicator": {"Y": 1, "N": 0},
    "abortion_edit_indicator": {"Y": 1, "N": 0}
}

CATEGORICAL_COLS = [
    "age_group", "race", "ethnicity", "type_of_admission",
    "patient_disposition", "apr_risk_of_mortality",
    "apr_medical_surgical_description", "payment_typology_1"
]


def clean_los(value):
    if str(value).strip() == "120+":
        return 120
    try:
        return int(value)
    except:
        return np.nan


def load_all_hospitals():
    """
    Loads all hospital CSVs, combines them into one dataframe,
    then preprocesses together so all hospitals share the same feature space.
    """
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    dfs = []

    for f in files:
        df = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)
        dfs.append(df)
        print(f"Loaded {f}: {len(df)} rows")

    # Combine all hospitals into one dataframe
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined: {len(combined)} rows")

    # Clean target
    combined["length_of_stay"] = combined["length_of_stay"].apply(clean_los)
    combined = combined.dropna(subset=["length_of_stay"])

    # Drop columns
    cols_to_drop = [c for c in DROP_COLS if c in combined.columns]
    combined = combined.drop(columns=cols_to_drop)

    # Safety check - force drop any remaining high-variance non-feature columns
    force_drop = ["total_charges", "total_costs", "operating_certificate_number", 
              "permanent_facility_id", "discharge_year"]
    force_drop = [c for c in force_drop if c in combined.columns]
    if force_drop:
        print(f"Force dropping: {force_drop}")
        combined = combined.drop(columns=force_drop)

    # Separate target
    y = combined["length_of_stay"].values
    combined = combined.drop(columns=["length_of_stay"])

    # Handle nulls
    for col in combined.select_dtypes(include=[np.number]).columns:
        combined[col] = combined[col].fillna(combined[col].median())
    for col in combined.select_dtypes(include=["object"]).columns:
        combined[col] = combined[col].fillna(combined[col].mode()[0])

    # Binary encode
    for col, mapping in BINARY_COLS.items():
        if col in combined.columns:
            combined[col] = combined[col].map(mapping).fillna(0).astype(int)

    # One-hot encode
    cat_to_encode = [c for c in CATEGORICAL_COLS if c in combined.columns]
    combined = pd.get_dummies(combined, columns=cat_to_encode)

    # Split before scaling
    X_train, X_test, y_train, y_test = train_test_split(
        combined.values, y, test_size=0.3, random_state=42
    )

    # Scale on train only
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
    return {"model": model_name, "MAE": mae, "RMSE": rmse, "R2": r2}


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_all_hospitals()

    results = []

    print("Training Ridge Regression...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    results.append(evaluate(y_test, ridge.predict(X_test), "Ridge Regression"))

    print("Training Decision Tree...")
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    results.append(evaluate(y_test, dt.predict(X_test), "Decision Tree"))

    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=5,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    results.append(evaluate(y_test, rf.predict(X_test), "Random Forest"))

    print("Training XGBoost...")
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate(y_test, xgb.predict(X_test), "XGBoost"))

    os.makedirs("../results", exist_ok=True)
    pd.DataFrame(results).to_csv("../results/centralized_results.csv", index=False)
    print("\nResults saved to results/centralized_results.csv")
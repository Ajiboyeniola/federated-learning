import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os

# Columns to drop
DROP_COLS = [
    "hospital_service_area",
    "hospital_county",
    "operating_certificate_number",
    "permanent_facility_id",
    "facility_name",
    "zip_code_3_digits",
    "discharge_year",
    "ccs_diagnosis_description",
    "ccs_procedure_description",
    "apr_drg_description",
    "apr_mdc_description",
    "apr_severity_of_illness_description",
    "payment_typology_2",
    "payment_typology_3",
    "total_charges",
    "total_costs"
]

# Binary encode
BINARY_COLS = {
    "gender": {"M": 1, "F": 0, "U": 0},
    "emergency_department_indicator": {"Y": 1, "N": 0},
    "abortion_edit_indicator": {"Y": 1, "N": 0}
}

# One-hot encode
CATEGORICAL_COLS = [
    "age_group",
    "race",
    "ethnicity",
    "type_of_admission",
    "patient_disposition",
    "apr_risk_of_mortality",
    "apr_medical_surgical_description",
    "payment_typology_1"
]


def clean_los(value):
    if str(value).strip() == "120+":
        return 120
    try:
        return int(value)
    except:
        return np.nan


def get_feature_schema(data_dir):
    """
    Load all hospital files, encode together, return the master column list.
    Every federated client must use this schema to ensure identical features.
    """
    dfs = []
    for file in sorted(os.listdir(data_dir)):
        if file.endswith(".csv"):
            dfs.append(pd.read_csv(
                os.path.join(data_dir, file), low_memory=False
            ))

    df = pd.concat(dfs, ignore_index=True)
    df["length_of_stay"] = df["length_of_stay"].apply(clean_los)
    df = df.dropna(subset=["length_of_stay"])
    df = df.drop(columns=["length_of_stay"])

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col, mapping in BINARY_COLS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    cat_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_to_encode)

    return df.columns.tolist()


def preprocess(filepath, scaler=None, fit_scaler=True):
    """
    Original preprocess for centralized use.
    """
    df = pd.read_csv(filepath, low_memory=False)

    df["length_of_stay"] = df["length_of_stay"].apply(clean_los)
    df = df.dropna(subset=["length_of_stay"])

    # Binary target
    y = (df["length_of_stay"] > 7).astype(int).values
    df = df.drop(columns=["length_of_stay"])

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col, mapping in BINARY_COLS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    cat_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_to_encode)

    feature_names = df.columns.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        df.values, y, test_size=0.3, random_state=42
    )

    if fit_scaler:
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
    else:
        X_train = scaler.transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_names


def preprocess_with_schema(filepath, feature_schema):
    """
    Federated preprocess. Enforces a fixed feature schema across all clients.
    Missing columns are filled with 0. Extra columns are dropped.
    """
    df = pd.read_csv(filepath, low_memory=False)

    df["length_of_stay"] = df["length_of_stay"].apply(clean_los)
    df = df.dropna(subset=["length_of_stay"])

    # Binary target
    y = (df["length_of_stay"] > 7).astype(int).values
    df = df.drop(columns=["length_of_stay"])

    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    for col, mapping in BINARY_COLS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    cat_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_to_encode)

    # Enforce schema
    for col in feature_schema:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_schema]

    X_train, X_test, y_train, y_test = train_test_split(
        df.values, y, test_size=0.3, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, feature_schema
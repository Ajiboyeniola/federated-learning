import flwr as fl
import numpy as np
import pickle
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, precision_score, recall_score)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_with_schema


def get_model(model_type):
    if model_type == "lr":
        return LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
    elif model_type == "dt":
        return DecisionTreeClassifier(
            max_depth=10,
            class_weight='balanced',
            random_state=42
        )
    elif model_type == "rf":
        return RandomForestClassifier(
            n_estimators=20,
            max_depth=15,
            min_samples_split=5,
            max_features=0.5,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def model_to_parameters(model, model_type):
    if model_type == "lr":
        return [model.coef_.astype(np.float32),
                model.intercept_.astype(np.float32)]
    else:
        model_bytes = pickle.dumps(model)
        arr = np.frombuffer(model_bytes, dtype=np.uint8)
        return [arr]


def parameters_to_model(parameters, model_type, base_model):
    if model_type == "lr":
        base_model.coef_      = parameters[0].astype(np.float64)
        base_model.intercept_ = parameters[1].astype(np.float64)
        base_model.classes_   = np.array([0, 1])
        return base_model
    else:
        model_bytes = parameters[0].tobytes()
        return pickle.loads(model_bytes)


class FederatedClient(fl.client.NumPyClient):

    def __init__(self, client_id, model_type, feature_schema,
                 data_dir="../data/hospitals"):
        self.client_id      = client_id
        self.model_type     = model_type
        self.feature_schema = feature_schema
        self.model          = get_model(model_type)
        self.fitted         = False

        filepath = os.path.join(data_dir, f"client_{client_id}.csv")
        (self.X_train, self.X_test,
         self.y_train, self.y_test, _, _) = preprocess_with_schema(
             filepath, feature_schema
         )

        print(f"[Client {client_id}] "
              f"{self.X_train.shape[0]} train / "
              f"{self.X_test.shape[0]} test | "
              f"features: {self.X_train.shape[1]} | "
              f"long-stay: {self.y_train.mean():.1%}")

    def get_parameters(self, config):
        if not self.fitted:
            self.model.fit(self.X_train, self.y_train)
            self.fitted = True
        return model_to_parameters(self.model, self.model_type)

    def fit(self, parameters, config):
        self.model = get_model(self.model_type)
        self.model.fit(self.X_train, self.y_train)
        self.fitted = True

        y_pred = self.model.predict(self.X_train)
        acc    = float(accuracy_score(self.y_train, y_pred))

        return (model_to_parameters(self.model, self.model_type),
                len(self.X_train),
                {"accuracy": acc})

    def evaluate(self, parameters, config):
        try:
            self.model = parameters_to_model(
                parameters, self.model_type, self.model
            )
        except Exception:
            if not self.fitted:
                self.model.fit(self.X_train, self.y_train)
                self.fitted = True

        y_pred = self.model.predict(self.X_test)
        y_prob = self.model.predict_proba(self.X_test)[:, 1]

        acc   = float(accuracy_score(self.y_test, y_pred))
        auroc = float(roc_auc_score(self.y_test, y_prob))
        f1    = float(f1_score(self.y_test, y_pred, zero_division=0))
        prec  = float(precision_score(self.y_test, y_pred, zero_division=0))
        rec   = float(recall_score(self.y_test, y_pred, zero_division=0))

        print(f"  [Client {self.client_id}] "
              f"Acc: {acc:.4f} | AUROC: {auroc:.4f} | F1: {f1:.4f}")

        return (float(1 - acc), len(self.X_test),
                {"accuracy": acc, "auroc": auroc,
                 "f1": f1, "precision": prec, "recall": rec})
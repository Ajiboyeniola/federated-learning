import flwr as fl
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
import numpy as np
import pickle
import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression as _LR


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import get_feature_schema
from client_fed import FederatedClient

NUM_CLIENTS = 10
NUM_ROUNDS  = 5
DATA_DIR    = "../data/hospitals"


class FedAvgLR(FedAvg):
    """True FedAvg for LR: weighted average of coefficients and intercept."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        total         = sum(fit_res.num_examples for _, fit_res in results)
        coef_agg      = None
        intercept_agg = None

        for _, fit_res in results:
            params = parameters_to_ndarrays(fit_res.parameters)
            weight = fit_res.num_examples / total
            if coef_agg is None:
                coef_agg      = weight * params[0].astype(np.float64)
                intercept_agg = weight * params[1].astype(np.float64)
            else:
                coef_agg      += weight * params[0].astype(np.float64)
                intercept_agg += weight * params[1].astype(np.float64)

        aggregated = ndarrays_to_parameters(
        [coef_agg.astype(np.float32),
         intercept_agg.astype(np.float32)]
        )
    
        _lr = _LR(max_iter=1000, class_weight='balanced', random_state=42)
        _lr.coef_          = coef_agg.astype(np.float64)
        _lr.intercept_     = intercept_agg.astype(np.float64)
        _lr.classes_       = np.array([0, 1])
        _lr.n_features_in_ = int(coef_agg.shape[-1])
        os.makedirs("../results", exist_ok=True)
        with open("../results/federated_lr_final.pkl", "wb") as f:
            pickle.dump(_lr, f)
    
        return aggregated, {}


class FedForest(FedAvg):
    """
    Federated Forest for RF and DT.
    RF:  merges all client trees into one global forest.
    DT:  selects best client tree by accuracy.
    """

    def __init__(self, model_type, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        if self.model_type == "rf":
            all_estimators = []
            base_model     = None

            for _, fit_res in results:
                params    = parameters_to_ndarrays(fit_res.parameters)
                client_rf = pickle.loads(params[0].tobytes())
                all_estimators.extend(client_rf.estimators_)
                if base_model is None:
                    base_model = client_rf

            # Merge all client trees into one global forest
            global_rf              = base_model
            global_rf.estimators_  = all_estimators
            global_rf.n_estimators = len(all_estimators)

            # ── ADD THESE 3 LINES ──────────────────────────────────────
            os.makedirs("../results", exist_ok=True)
            with open("../results/federated_rf_final.pkl", "wb") as f:
                pickle.dump(global_rf, f)
            # ───────────────────────────────────────────────────────────


            model_bytes = pickle.dumps(global_rf)
            arr         = np.frombuffer(model_bytes, dtype=np.uint8)
            return ndarrays_to_parameters([arr]), {}

        elif self.model_type == "dt":
            best_params   = None
            best_accuracy = -1
 
            for _, fit_res in results:
                acc = fit_res.metrics.get("accuracy", 0)
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_params   = parameters_to_ndarrays(fit_res.parameters)
 
            _dt = pickle.loads(best_params[0].tobytes())
            os.makedirs("../results", exist_ok=True)
            with open("../results/federated_dt_final.pkl", "wb") as f:
                pickle.dump(_dt, f)
 
            return ndarrays_to_parameters(best_params), {}


def aggregate_eval_metrics(metrics):
    total = sum(n for n, _ in metrics)
    acc   = sum(n * m["accuracy"]  for n, m in metrics) / total
    auroc = sum(n * m["auroc"]     for n, m in metrics) / total
    f1    = sum(n * m["f1"]        for n, m in metrics) / total
    prec  = sum(n * m["precision"] for n, m in metrics) / total
    rec   = sum(n * m["recall"]    for n, m in metrics) / total
    return {"accuracy": acc, "auroc": auroc, "f1": f1,
            "precision": prec, "recall": rec}


def aggregate_fit_metrics(metrics):
    total = sum(n for n, _ in metrics)
    acc   = sum(n * m["accuracy"] for n, m in metrics) / total
    return {"accuracy": acc}


def run_federated(model_type, feature_schema):
    print(f"\n{'='*55}")
    print(f"  FedAvg | {model_type.upper()}")
    print(f"{'='*55}")

    def client_fn(cid):
        return FederatedClient(
            client_id=int(cid) + 1,
            model_type=model_type,
            feature_schema=feature_schema,
            data_dir=DATA_DIR
        ).to_client()

    common = dict(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    if model_type == "lr":
        strategy = FedAvgLR(**common)
    else:
        strategy = FedForest(model_type=model_type, **common)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 2, "num_gpus": 0},
        ray_init_args={"num_cpus": 4}
    )
    return history


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)

    # Build master feature schema from all hospitals combined
    print("Building feature schema from all hospitals...")
    FEATURE_SCHEMA = get_feature_schema(DATA_DIR)
    print(f"Global feature schema: {len(FEATURE_SCHEMA)} features")

    all_results = []

    for model_type in ["lr", "dt", "rf"]:
        history = run_federated(model_type, FEATURE_SCHEMA)

        metrics = history.metrics_distributed
        if metrics:
            rounds_acc   = metrics.get("accuracy",  [])
            rounds_auroc = metrics.get("auroc",     [])
            rounds_f1    = metrics.get("f1",        [])
            rounds_prec  = metrics.get("precision", [])
            rounds_rec   = metrics.get("recall",    [])

            if rounds_acc:
                final_acc   = rounds_acc[-1][1]
                final_auroc = rounds_auroc[-1][1] if rounds_auroc else None
                final_f1    = rounds_f1[-1][1]    if rounds_f1    else None
                final_prec  = rounds_prec[-1][1]  if rounds_prec  else None
                final_rec   = rounds_rec[-1][1]   if rounds_rec   else None

                print(f"\nFinal | {model_type.upper()} | FedAvg")
                print(f"  Acc: {final_acc:.4f} | AUROC: {final_auroc:.4f} "
                      f"| F1: {final_f1:.4f}")

                all_results.append({
                    "model":     model_type,
                    "strategy":  "fedavg",
                    "accuracy":  round(final_acc,   4),
                    "auroc":     round(final_auroc, 4) if final_auroc else None,
                    "f1":        round(final_f1,    4) if final_f1    else None,
                    "precision": round(final_prec,  4) if final_prec  else None,
                    "recall":    round(final_rec,   4) if final_rec   else None,
                })

    df = pd.DataFrame(all_results)
    df.to_csv("../results/federated_sklearn_results.csv", index=False)
    print("\n" + "="*55)
    print("Saved to results/federated_sklearn_results.csv")
    print("="*55)
    print(df.to_string(index=False))
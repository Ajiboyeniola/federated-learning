import flwr as fl
from flwr.server.strategy import FedXgbBagging
import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import get_feature_schema
from client_xgb_fed import XGBFederatedClient

NUM_CLIENTS = 10
NUM_ROUNDS  = 5
DATA_DIR    = "../data/hospitals"


def aggregate_eval_metrics(eval_res):
    total = sum(num for num, _ in eval_res)
    acc   = sum(num * m["accuracy"]  for num, m in eval_res) / total
    auroc = sum(num * m["auroc"]     for num, m in eval_res) / total
    f1    = sum(num * m["f1"]        for num, m in eval_res) / total
    prec  = sum(num * m["precision"] for num, m in eval_res) / total
    rec   = sum(num * m["recall"]    for num, m in eval_res) / total
    return {"accuracy": acc, "auroc": auroc, "f1": f1,
            "precision": prec, "recall": rec}


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)

    # Build master feature schema
    print("Building feature schema from all hospitals...")
    FEATURE_SCHEMA = get_feature_schema(DATA_DIR)
    print(f"Global feature schema: {len(FEATURE_SCHEMA)} features")

    print(f"\n{'='*55}")
    print(f"  FedXgbBagging | XGBOOST")
    print(f"{'='*55}")

    def client_fn(cid):
        return XGBFederatedClient(
            client_id=int(cid) + 1,
            feature_schema=FEATURE_SCHEMA,
            data_dir=DATA_DIR
        ).to_client()

    strategy = FedXgbBagging(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0}
    )

    metrics = history.metrics_distributed
    results = []

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

            print(f"\nFinal | XGBoost | FedXgbBagging")
            print(f"  Acc: {final_acc:.4f} | AUROC: {final_auroc:.4f} "
                  f"| F1: {final_f1:.4f} | Prec: {final_prec:.4f} "
                  f"| Rec: {final_rec:.4f}")

            results.append({
                "model":     "xgb",
                "strategy":  "fedxgb_bagging",
                "accuracy":  round(final_acc,   4),
                "auroc":     round(final_auroc, 4) if final_auroc else None,
                "f1":        round(final_f1,    4) if final_f1    else None,
                "precision": round(final_prec,  4) if final_prec  else None,
                "recall":    round(final_rec,   4) if final_rec   else None,
            })

    df = pd.DataFrame(results)
    df.to_csv("../results/federated_xgb_results.csv", index=False)
    print("\n" + "="*55)
    print("Saved to results/federated_xgb_results.csv")
    print("="*55)
    print(df.to_string(index=False))
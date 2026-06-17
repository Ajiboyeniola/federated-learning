import flwr as fl
from flwr.common import (
    Code, EvaluateIns, EvaluateRes,
    FitIns, FitRes, GetParametersRes,
    Parameters, Status,
)
import xgboost as xgb
import numpy as np
import os
import sys
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             f1_score, precision_score, recall_score)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import preprocess_with_schema


class XGBFederatedClient(fl.client.Client):

    def __init__(self, client_id, feature_schema,
                 data_dir="../data/hospitals"):
        self.client_id = client_id

        self.params = {
            "objective":        "binary:logistic",
            "eval_metric":      "auc",
            "eta":              0.05,
            "max_depth":        8,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 4.2,
            "seed":             42,
        }
        self.num_local_round = 1

        filepath = os.path.join(data_dir, f"client_{client_id}.csv")
        (X_train, X_test,
         y_train, y_test, _, _) = preprocess_with_schema(
             filepath, feature_schema
         )

        self.train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        self.test_dmatrix  = xgb.DMatrix(X_test,  label=y_test)
        self.y_test        = y_test
        self.num_train     = len(y_train)
        self.num_test      = len(y_test)

        print(f"[XGB Client {client_id}] "
              f"{self.num_train} train / {self.num_test} test | "
              f"long-stay: {y_train.mean():.1%}")

    def get_parameters(self, ins):
        return GetParametersRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(tensor_type="", tensors=[])
        )

    def _local_boost(self, bst_input):
        for _ in range(self.num_local_round):
            bst_input.update(
                self.train_dmatrix,
                bst_input.num_boosted_rounds()
            )
        bst = bst_input[
            bst_input.num_boosted_rounds() - self.num_local_round:
            bst_input.num_boosted_rounds()
        ]
        return bst

    def fit(self, ins: FitIns) -> FitRes:
        global_round = int(ins.config.get("global_round", 1))

        if global_round == 1:
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[(self.test_dmatrix, "eval")],
                verbose_eval=False
            )
        else:
            global_model = bytearray(ins.parameters.tensors[0])
            bst = xgb.Booster(params=self.params)
            bst.load_model(global_model)
            bst = self._local_boost(bst)

        local_model       = bst.save_raw("json")
        local_model_bytes = np.array(list(local_model), dtype=np.uint8)

        return FitRes(
            status=Status(code=Code.OK, message="OK"),
            parameters=Parameters(
                tensor_type="",
                tensors=[local_model_bytes.tobytes()]
            ),
            num_examples=self.num_train,
            metrics={}
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        global_model = bytearray(ins.parameters.tensors[0])
        bst = xgb.Booster(params=self.params)
        bst.load_model(global_model)

        y_prob = bst.predict(self.test_dmatrix)
        y_pred = (y_prob > 0.5).astype(int)

        acc   = float(accuracy_score(self.y_test, y_pred))
        auroc = float(roc_auc_score(self.y_test, y_prob))
        f1    = float(f1_score(self.y_test, y_pred, zero_division=0))
        prec  = float(precision_score(self.y_test, y_pred, zero_division=0))
        rec   = float(recall_score(self.y_test, y_pred, zero_division=0))

        print(f"  [XGB Client {self.client_id}] "
              f"Acc: {acc:.4f} | AUROC: {auroc:.4f} | F1: {f1:.4f}")

        return EvaluateRes(
            status=Status(code=Code.OK, message="OK"),
            loss=float(1 - acc),
            num_examples=self.num_test,
            metrics={"accuracy": acc, "auroc": auroc,
                     "f1": f1, "precision": prec, "recall": rec}
        )
"""Метрики, нормализация и модели NIR-исследования (Ridge/RF/XGB)."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor

SEED = 42
METRICS = ("mse", "mae", "r2", "ic", "spearman_ic", "sign_acc",
           "pos_sign_acc", "neg_sign_acc")
GRIDS = {
    "ridge": [{"alpha": float(alpha)} for alpha in np.logspace(-2, 4, 13)],
    "rf": [{"max_depth": depth, "min_samples_leaf": leaf,
            "max_features": 0.3}
           for depth in (4, 8) for leaf in (10, 25)],
    "xgb": [{"n_estimators": trees, "max_depth": depth,
             "learning_rate": 0.05}
            for trees in (300, 600) for depth in (3, 5)],
}


def normalize(train, val, test, kind):
    """Нормализовать X по train: std или min-max в [-1, 1]."""
    scaler = StandardScaler() if kind == "std" else MinMaxScaler((-1, 1))
    return (scaler.fit_transform(train), scaler.transform(val),
            scaler.transform(test))


def flatten(X):
    """Развернуть окна (n, w, 27) в матрицу (n, w*27)."""
    return np.asarray(X, dtype=np.float64).reshape(len(X), -1)


def evaluate(y_true, y_pred):
    """Метрики: MSE, MAE, R2, IC, SpearmanIC и три accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        ic = spearman = np.nan
    else:
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])
        spearman = float(pd.Series(y_true).corr(pd.Series(y_pred),
                                                method="spearman"))
    positive, negative = y_true > 0, y_true < 0
    return {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "ic": ic,
        "spearman_ic": spearman,
        "sign_acc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        "pos_sign_acc": (float(np.mean(y_pred[positive] > 0))
                         if positive.any() else np.nan),
        "neg_sign_acc": (float(np.mean(y_pred[negative] < 0))
                         if negative.any() else np.nan),
    }


def build_model(name, params):
    """Собрать регрессор ridge/rf/xgb; n_jobs=1 (ограничение песочницы)."""
    if name == "ridge":
        return Ridge(**params)
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, n_jobs=1,
                                     random_state=SEED, **params)
    return XGBRegressor(subsample=0.8, colsample_bytree=0.8,
                        tree_method="hist", random_state=SEED, n_jobs=1,
                        verbosity=0, **params)

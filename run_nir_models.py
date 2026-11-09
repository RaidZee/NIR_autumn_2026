"""Обучение NIR-моделей: сетки, выбор по val MSE, отчёты stage-04."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import nir_data, nir_models

ROOT = Path("splits/nir")
RESULTS = Path("reports/stage-04-results.csv")
BEST = Path("reports/stage-04-best.csv")
NORMS = ("std", "minmax")
MODELS = ("ridge", "rf", "xgb")
N_FOLDS = nir_data.N_SPLITS


def load_fold(k, w, h):
    """Загрузить окна фолда k для окна w и горизонта h."""
    data = np.load(ROOT / f"split_{k}" / f"w{w}_h{h}.npz")
    return {name: data[name] for name in data.files}


def prepare(fold, norm):
    """Развернуть окна и нормализовать X по train; y не нормализуем."""
    return nir_models.normalize(
        nir_models.flatten(fold["X_train"]),
        nir_models.flatten(fold["X_val"]),
        nir_models.flatten(fold["X_test"]),
        norm,
    )


def run_config(norm, w, h, model_name):
    """Подобрать гиперпараметры по val MSE и посчитать метрики по фолдам."""
    folds = [load_fold(k, w, h) for k in range(N_FOLDS)]
    prepared = [prepare(fold, norm) for fold in folds]

    grid = nir_models.GRIDS[model_name]
    scores = []
    for params in grid:
        fold_scores = []
        for fold, (X_train, X_val, _) in zip(folds, prepared):
            model = nir_models.build_model(model_name, params)
            model.fit(X_train, fold["y_train"])
            fold_scores.append(nir_models.evaluate(
                fold["y_val"], model.predict(X_val))["mse"])
        scores.append(float(np.mean(fold_scores)))
    best = grid[int(np.argmin(scores))]

    rows = []
    for k, (fold, (X_train, X_val, X_test)) in enumerate(zip(folds, prepared)):
        model = nir_models.build_model(model_name, best)
        model.fit(X_train, fold["y_train"])
        row = {"norm": norm, "window": w, "horizon": h,
               "model": model_name,
               "params": json.dumps(best, sort_keys=True), "split": k}
        for name, X, y in (("train", X_train, fold["y_train"]),
                           ("val", X_val, fold["y_val"]),
                           ("test", X_test, fold["y_test"])):
            for metric, value in nir_models.evaluate(
                    y, model.predict(X)).items():
                row[f"{name}_{metric}"] = value
        rows.append(row)
    return rows


def main():
    """Перебрать нормализации, окна, горизонты и модели; сохранить отчёты."""
    frames, summaries = [], []
    for norm in NORMS:
        for w in nir_data.WINDOWS:
            for h in nir_data.HORIZONS:
                for model_name in MODELS:
                    rows = run_config(norm, w, h, model_name)
                    frame = pd.DataFrame(rows)
                    frames.append(frame)
                    means = frame.drop(
                        columns=["norm", "window", "horizon", "model",
                                 "params", "split"]).mean()
                    summary = {"norm": norm, "window": w, "horizon": h,
                               "model": model_name,
                               "params": rows[0]["params"]}
                    summary.update({f"{name}_mean": value
                                    for name, value in means.items()})
                    summaries.append(summary)

    pd.concat(frames, ignore_index=True).to_csv(RESULTS, index=False)
    best = pd.DataFrame(summaries)
    best["is_global_best"] = best["val_mse_mean"] == best["val_mse_mean"].min()
    best.sort_values("val_mse_mean").to_csv(BEST, index=False)


if __name__ == "__main__":
    main()

"""Данные NIR-исследования GAZP: доходности, сплиты, винзоризация, окна."""

from pathlib import Path

import numpy as np
import pandas as pd

MOEX = ["GAZP", "LKOH", "ROSN", "SNGS", "SBER", "MGNT", "MOEXOG", "GMKN",
        "CHMF", "NLMK", "SNGSP", "SIBN", "TRNFP", "PLZL"]
YF = ["USO", "UNG", "XOM", "CVX", "SHEL", "COP", "LNG", "GLNG", "CL=F",
      "BZ=F", "SLB", "HAL", "BKR"]
FEATURES = MOEX + YF
TARGET = "GAZP"
TARGET_COL = FEATURES.index(TARGET)
N_CUTS = 6
N_SPLITS = 5
WINDOWS = (7, 10)
HORIZONS = (2, 3)
COEF = 3.0
SOURCE = Path("data/my_data.csv")
PRICES = Path("data/nir_prices.csv")


def load_prices(path=PRICES):
    """Прочитать mid-цены: индекс DATE, 27 колонок признаков."""
    return pd.read_csv(path, index_col="DATE", parse_dates=["DATE"])[FEATURES]


def feature_returns(prices):
    """Доходности MOEX без сдвига и YF со сдвигом на 1 торговый день."""
    moex = prices[MOEX].pct_change(fill_method=None)
    yf = prices[YF].shift(1).pct_change(fill_method=None)
    return pd.concat([moex, yf], axis=1).dropna()


def make_cuts(returns, n=N_CUTS):
    """Разбить доходности на n равных последовательных кусков (все строки)."""
    positions = np.array_split(np.arange(len(returns)), n)
    return [returns.iloc[part] for part in positions]


def medcouple(column):
    """Векторный medcouple: медиана h по всем парам левых/правых значений."""
    q2 = np.median(column)
    left = column[column <= q2]
    right = column[column >= q2]
    if left.size == 0 or right.size == 0:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        h = ((right - q2) - (q2 - left[:, None])) / (right - left[:, None])
    h[(left[:, None] == q2) & (right == q2)] = np.nan
    finite = h[~np.isnan(h)]
    return float(np.median(finite)) if finite.size else 0.0


def winsor_bounds(matrix):
    """MC-границы винзоризации (coef=3) по каждому столбцу train-матрицы."""
    q1, q2, q3 = np.percentile(matrix, [25, 50, 75], axis=0)
    iqr = q3 - q1
    mc = np.array([medcouple(matrix[:, j]) for j in range(matrix.shape[1])])
    left_exp = np.where(mc >= 0, np.exp(-4 * mc), np.exp(-3 * mc))
    right_exp = np.where(mc >= 0, np.exp(3 * mc), np.exp(4 * mc))
    left = q1 - COEF * left_exp * iqr
    right = q3 + COEF * right_exp * iqr
    return left, right


def train_chunks(cuts, k):
    """Индексы кусков train сплита k (в исходном порядке)."""
    return [j for j in range(len(cuts)) if j not in (k, k + 1)]


def make_windows(x_matrix, y_matrix, w, h):
    """Окна X (n,w,27) из x_matrix и цели y (n,) из y_matrix внутри куска."""
    n = len(x_matrix) - w - h + 1
    if n <= 0:
        return np.empty((0, w, x_matrix.shape[1])), np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(x_matrix, w, axis=0)[:n]
    X = np.ascontiguousarray(windows.transpose(0, 2, 1))
    target = y_matrix[:, TARGET_COL]
    y = np.array([np.prod(1.0 + target[i + w:i + w + h]) - 1.0
                  for i in range(n)])
    return X, y


def build_split(cuts, k, w, h, bounds=None):
    """Окна train/val/test сплита k с MC-границами, посчитанными по train.

    Отклонение от NIR: X val/test тоже клипаем train-границами. Причина —
    CL=F уходит в минус в апреле 2020 (доходности -152%/-774%, z до -502),
    из-за чего val/test без клипа искажены. Цель y не клипается.
    """
    ids = train_chunks(cuts, k)
    if bounds is None:
        bounds = winsor_bounds(np.vstack([cuts[j].to_numpy() for j in ids]))
    left, right = bounds
    X_list, y_list = [], []
    for j in ids:
        # У train и признаки, и цель считаются по клипнутым доходностям.
        clipped = np.clip(cuts[j].to_numpy(), left, right)
        x_part, y_part = make_windows(clipped, clipped, w, h)
        X_list.append(x_part)
        y_list.append(y_part)
    val = cuts[k].to_numpy()
    X_val, y_val = make_windows(np.clip(val, left, right), val, w, h)
    test = cuts[k + 1].to_numpy()
    X_test, y_test = make_windows(np.clip(test, left, right), test, w, h)
    return {
        "X_train": np.vstack(X_list),
        "y_train": np.concatenate(y_list),
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }

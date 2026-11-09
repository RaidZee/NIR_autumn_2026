# Архитектура

## Поток данных

1. `data/my_data.csv` — 27 mid-цен (14 MOEX + 13 yfinance), 2016–2025.
2. `build_nir_data.py`:
   - копирует CSV в `data/nir_prices.csv`, пишет `data/manifest_nir.json`;
   - `src/nir_data.py::feature_returns`: MOEX `pct_change`, yfinance
     `prices[YF].shift(1).pct_change` (сдвиг на 1 торговый день);
   - 6 кусков (`np.array_split`) → 5 сплитов: `val=cut[k]`,
     `test=cut[k+1]`, `train` — остальные 4;
   - MC-винзоризация (`coef=3`, medcouple, Q1/Q2/Q3/IQR) только по train;
   - train X/y — по клипнутым доходностям; X val/test клипаются теми же
     train-границами, y val/test — сырые;
   - окна `X (n,w,27)`, цели `y = prod(1+r_GAZP[i+w:i+w+h])−1`;
   - сохраняет `splits/nir/split_k/w{w}_h{h}.npz` и `splits/nir/meta.json`.
3. `run_nir_models.py`:
   - `std` / `min-max [−1,1]`, fit только на train;
   - Ridge / RF / XGB, выбор по среднему val MSE;
   - `reports/stage-04-results.csv` (long, 5 фолдов) и
     `reports/stage-04-best.csv` (24 группы).
4. `notebooks/nir_gazp_results.ipynb` — визуализация CSV, без обучения.

## Контракты

- Признаки: 27 колонок в порядке `MOEX + YF`; `GAZP` — индекс 0.
- Окна: `w ∈ {7,10}`, `h ∈ {2,3}`; пересечение границ кусков запрещено.
- Метрики: MSE, MAE, R², IC (Pearson), Spearman IC, sign accuracy,
  positive/negative sign accuracy.
- Выбор модели: только val; test — финальная оценка.

## Ограничения

- `cut0` не тестируется, `cut5` не валидируется; train fold0–3 включает
  будущие куски.
- Клип X val/test — отклонение от исходного NIR из-за отрицательных цен
  `CL=F` (апрель 2020); y val/test сырые.
- 156 строк с нулевыми yfinance-доходностями (ffill по US-праздникам).

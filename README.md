# NIR_autumn_2026 — прогноз mid-доходности GAZP

Исследование прогноза доходности GAZP по средней цене дня `(open+close)/2`
на горизонт **2 и 3 дня** по окну **7 и 10 дней**. Используются 27 тикеров:
14 MOEX + 13 yfinance-инструментов (нефть/газ/энергетика).

## Результат

Лучшая конфигурация — Random Forest, окно 7, горизонт 2,
`max_depth=8`, `min_samples_leaf=25`, `max_features=0.3`:

| Метрика | Val | Test |
|---|---:|---:|
| R² | **0,0253** | **0,0322** |
| IC (Pearson) | 0,1749 | 0,2053 |
| Sign accuracy | 0,5551 | 0,5506 |

Ridge: val/test R² 0,008/0,011; XGBoost переобучается
(train R² ≈ 0,98, val/test R² < 0). Подробности —
в [reports/stage-04-nir-gazp.md](reports/stage-04-nir-gazp.md).

## Данные

В репозиторий **не коммитятся** (см. `.gitignore`). Нужно положить
`data/my_data.csv`:

- 2577 строк × 27 колонок, mid-цены, 2016-01-04 … 2025-12-30, NaN 0;
- MOEX (14): GAZP, LKOH, ROSN, SNGS, SBER, MGNT, MOEXOG, GMKN, CHMF,
  NLMK, SNGSP, SIBN, TRNFP, PLZL;
- yfinance (13): USO, UNG, XOM, CVX, SHEL, COP, LNG, GLNG, CL=F,
  BZ=F, SLB, HAL, BKR;
- SHA-256: `ee0fea3c586c1c72b6633ddc41da8b8742ea9b12df8035491d9b7465e0ceaa95`.

yfinance-уровни в пайплайне сдвигаются на 1 торговый день
(`shift(1)`), MOEX — нет.

## Запуск

```powershell
pip install -r requirements.txt
python build_nir_data.py     # data/my_data.csv -> data/nir_prices.csv, splits/nir/*
python run_nir_models.py     # 24 группы × 5 фолдов, ~39 мин на CPU
jupyter notebook notebooks/nir_gazp_results.ipynb
```

`build_nir_data.py` копирует исходный CSV в `data/nir_prices.csv`,
считает доходности, MC-винзоризацию (coef=3, только train), 6 кусков →
5 сплитов, окна 7/10 и цели 2/3 дня. `run_nir_models.py` обучает
Ridge/RF/XGB, выбирает гиперпараметры по среднему val MSE и пишет
`reports/stage-04-results.csv` / `stage-04-best.csv`.

## Структура

```text
NIR_autumn_2026/
├── src/
│   ├── nir_data.py       # данные, yf shift, MC-винзоризация, окна
│   └── nir_models.py     # метрики, нормализация, Ridge/RF/XGB
├── build_nir_data.py     # подготовка data/ и splits/nir/
├── run_nir_models.py     # CV, сетки, результаты
├── data/                 # my_data.csv (локально, gitignored)
├── splits/               # npz + meta.json (локально, gitignored)
├── reports/              # отчёт и CSV с результатами
├── notebooks/            # визуализация результатов
├── docs/                 # архитектура
├── requirements.txt
└── README.md
```

## Ограничения

- 5 сплитов из 6 кусков: `cut0` не тестируется, `cut5` не валидируется;
  train для fold0–3 включает будущие куски (осознанно, как в исходном NIR).
- X val/test клипаются train-границами MC (отклонение от NIR) из-за
  отрицательных цен `CL=F` в апреле 2020; `y` val/test остаются сырыми.
- 156 строк с нулевыми yfinance-доходностями (ffill по US-праздникам).
- Нет ablation «с yfinance / без yfinance», «shift 0 / shift 1» и нет
  экономического бэктеста.
- Сравнение с исходным NIR некорректно: там не было `shift(1)` yfinance
  и другой набор кусков.

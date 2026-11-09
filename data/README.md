# Данные

Положите сюда `my_data.csv` (не коммитится, см. `.gitignore`).

Требования к файлу:

- индекс `DATE` (дневные даты), 2016-01-04 … 2025-12-30;
- 2577 строк × 27 колонок, mid-цены `(open+close)/2`, NaN 0;
- MOEX (14): GAZP, LKOH, ROSN, SNGS, SBER, MGNT, MOEXOG, GMKN, CHMF,
  NLMK, SNGSP, SIBN, TRNFP, PLZL;
- yfinance (13): USO, UNG, XOM, CVX, SHEL, COP, LNG, GLNG, CL=F,
  BZ=F, SLB, HAL, BKR;
- SHA-256: `ee0fea3c586c1c72b6633ddc41da8b8742ea9b12df8035491d9b7465e0ceaa95`.

`build_nir_data.py` создаст `data/nir_prices.csv` (копия) и
`data/manifest_nir.json` с SHA-256; оба файла игнорируются git.

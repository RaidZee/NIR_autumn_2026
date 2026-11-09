"""Сборка данных NIR: копия цен, окна по 5 сплитам, meta.json и манифест."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from src import nir_data

ROOT = Path("splits/nir")
MANIFEST = Path("data/manifest_nir.json")


def sha256(path):
    """SHA-256 файла."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    """Скопировать цены, построить все окна и записать meta/manifest."""
    nir_data.PRICES.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(nir_data.SOURCE, nir_data.PRICES)
    prices = nir_data.load_prices()
    returns = nir_data.feature_returns(prices)
    cuts = nir_data.make_cuts(returns)

    manifest = {
        "source": str(nir_data.SOURCE),
        "prices_path": str(nir_data.PRICES),
        "sha256": sha256(nir_data.PRICES),
        "rows": int(len(prices)),
        "start": str(prices.index[0].date()),
        "end": str(prices.index[-1].date()),
        "columns": list(prices.columns),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    meta = {
        "source": str(nir_data.PRICES),
        "sha256": manifest["sha256"],
        "rows": int(len(returns)),
        "features": list(returns.columns),
        "target": nir_data.TARGET,
        "cuts": len(cuts),
        "cut_sizes": [len(c) for c in cuts],
        "windows": list(nir_data.WINDOWS),
        "horizons": list(nir_data.HORIZONS),
        # Клип val/test X train-границами — отклонение от NIR (см. nir_data).
        "clip_val_test": True,
        "folds": [],
        "shapes": {},
    }
    for k in range(nir_data.N_SPLITS):
        ids = nir_data.train_chunks(cuts, k)
        train = np.vstack([cuts[j].to_numpy() for j in ids])
        left, right = nir_data.winsor_bounds(train)
        meta["folds"].append({
            "split": k,
            "train_chunks": ids,
            "train_start": str(cuts[ids[0]].index[0].date()),
            "train_end": str(cuts[ids[-1]].index[-1].date()),
            "val": {"start": str(cuts[k].index[0].date()),
                    "end": str(cuts[k].index[-1].date()),
                    "rows": len(cuts[k])},
            "test": {"start": str(cuts[k + 1].index[0].date()),
                     "end": str(cuts[k + 1].index[-1].date()),
                     "rows": len(cuts[k + 1])},
            "winsor_left": left.round(12).tolist(),
            "winsor_right": right.round(12).tolist(),
        })
        folder = ROOT / f"split_{k}"
        folder.mkdir(parents=True, exist_ok=True)
        meta["shapes"][f"split_{k}"] = {}
        for w in nir_data.WINDOWS:
            for h in nir_data.HORIZONS:
                data = nir_data.build_split(cuts, k, w, h,
                                            bounds=(left, right))
                for value in data.values():
                    if not np.isfinite(value).all():
                        raise RuntimeError(
                            f"нефинитные значения: split {k}, w{w}, h{h}")
                np.savez(folder / f"w{w}_h{h}.npz",
                         **{name: value.astype(np.float32)
                            for name, value in data.items()})
                meta["shapes"][f"split_{k}"][f"w{w}_h{h}"] = {
                    name: list(value.shape) for name, value in data.items()}

    (ROOT / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

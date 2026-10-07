"""Etapa 6: amostragem estratificada por pares t0->t1, divisão espacial por blocos (70/15/15), scaler do treino
e matrizes de transição populacionais (para a demanda de Markov e a correção de priors)."""
from _common import *
import json, numpy as np, pandas as pd, rasterio
from sbsrv2 import features as F
from sbsrv2.sampling import block_ids, split_blocks, stratified_sample

cfg = load_config(); grid = load_grid(cfg); proc = p(cfg, cfg["paths"]["processed"]); y0 = cfg["annual_range"][0]
ds = cfg["dataset"]; rng = np.random.default_rng(ds["seed"])
with rasterio.open(p(cfg, cfg["paths"]["aoi"]) / "aoi_mask.tif") as s:
    aoi_flat = np.flatnonzero(s.read(1).ravel() > 0)
cat = lambda y: rasterio.open(proc / "lulc_reclass_v1.tif").read(y - y0 + 1)
block_px = int(round(ds["block_size_km"] * 1000 / grid.res))
blocks = block_ids(grid, aoi_flat, block_px)
lab = split_blocks(blocks, ds["split"], ds["seed"])
print(f"blocos de {ds['block_size_km']} km ({block_px} px): {len(lab)} | "
      + ", ".join(f"{k}={sum(v==k for v in lab.values())}" for k in ("train", "val", "test")))

names_all = F.names_for(cfg, "C2")
rows, npop_all, nsm_train, npop_train = [], {}, np.zeros((6, 6), np.int64), []
pairs = [tuple(x) for x in cfg["train_pairs"]] + [tuple(cfg["validation_pair"])]
for t0, t1 in pairs:
    c0, c1 = cat(t0), cat(t1)
    v0, v1 = c0.ravel()[aoi_flat], c1.ravel()[aoi_flat]
    ok = (v0 > 0) & (v1 > 0)
    sel, npop, nsm = stratified_sample(v0[ok], v1[ok], None, ds["n_per_stratum"], rng)
    npop_all[f"{t0}_{t1}"] = npop.tolist()
    changed = 1 - np.trace(npop) / npop.sum()
    print(f"{t0}->{t1}: {npop.sum():,} px válidos | mudança {100*changed:.2f}% | amostra {len(sel):,}")
    if (t0, t1) == tuple(cfg["validation_pair"]):
        continue                                            # validação 2020->2025: nunca entra no treino
    npop_train.append(npop); nsm_train += nsm
    flat = aoi_flat[np.flatnonzero(ok)[sel]]
    X = F.features_at(cfg, grid, t0, c0, flat, names_all)
    df = pd.DataFrame(X, columns=names_all)
    df.insert(0, "flat", flat.astype(np.int64)); df.insert(1, "t0", t0); df.insert(2, "t1", t1)
    df["c0"] = c0.ravel()[flat]; df["c1"] = c1.ravel()[flat]
    df["split"] = [lab[int(b)] for b in block_ids(grid, flat, block_px)]
    rows.append(df)
data = pd.concat(rows, ignore_index=True)
out = proc / "samples_v1.parquet"; data.to_parquet(out)
print(f"\namostras: {len(data):,} | por partição: {data.split.value_counts().to_dict()} | NaN por preditor (%):")
print((100 * data[names_all].isna().mean()).round(2)[lambda s: s > 0].to_dict())

# scaler: só linhas de treino, ignora NaN
tr = data[data.split == "train"]
scaler = {n: dict(mean=float(np.nanmean(tr[n])), std=float(max(np.nanstd(tr[n]), 1e-6)), median=float(np.nanmedian(tr[n])))
          for n in names_all}
npop_sum = np.sum(npop_train, axis=0)
w = np.where(nsm_train > 0, npop_sum / np.maximum(nsm_train, 1), 0.0)
meta = dict(scaler=scaler, names_all=names_all, npop=npop_all, npop_train_sum=npop_sum.tolist(),
            nsample_train=nsm_train.tolist(), w_prior=w.tolist(), block_labels={str(k): v for k, v in lab.items()})
(proc / "dataset_meta_v1.json").write_text(json.dumps(meta))
print("\nMatriz de transição agrupada (treino, linhas=origem; % por passo de 5 anos):")
M = npop_sum / npop_sum.sum(1, keepdims=True)
print(pd.DataFrame(100 * M, index=range(1, 7), columns=range(1, 7)).round(2).to_string())

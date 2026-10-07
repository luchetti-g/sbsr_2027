"""Etapa 11: projeta 2030-2050 com o MLP C2 treinado em TODOS os pares (1990-2025).
Cenários de demanda (matriz de Markov por passo de 5 anos):
  tendencial        : todos os pares da série
  expansao_agricola : decênio de maior ganho líquido de agricultura+urbano (2005-2015)
  pressao_floresta  : decênio de maior perda líquida de floresta (1995-2005)
Uso: 11_project.py [--reuse-model] [cenario ...]   (sem cenários = os três)"""
from _common import *
import json, os, gc, sys, numpy as np, pandas as pd, rasterio
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from sbsrv2 import features as F, ca, mlp
from sbsrv2.sampling import block_ids, stratified_sample
from sbsrv2.geoio import write_raster

args = [a for a in sys.argv[1:] if not a.startswith("--")]; reuse = "--reuse-model" in sys.argv
cfg = load_config(); grid = load_grid(cfg); proc = p(cfg, cfg["paths"]["processed"]); rep = p(cfg, cfg["paths"]["reports"]) / "model"
y0 = cfg["annual_range"][0]; names = F.names_for(cfg, "C2"); pj = cfg["projection"]; ds = cfg["dataset"]
meta = json.loads((proc / "dataset_meta_v1.json").read_text()); scaler = meta["scaler"]
with rasterio.open(p(cfg, cfg["paths"]["aoi"]) / "aoi_mask.tif") as s:
    aoi_flat = np.flatnonzero(s.read(1).ravel() > 0)
lulc = rasterio.open(proc / "lulc_reclass_v1.tif"); cat = lambda y: lulc.read(y - y0 + 1)
model_path = p(cfg, "models") / "mlp_C2_final_v1.keras"

# ---- 1) amostra do par 2020->2025 (determinística, semente+1): contagens para w e, se treinar, as linhas ----
rng = np.random.default_rng(ds["seed"] + 1)
t0, t1 = cfg["validation_pair"]; c0, c1 = cat(t0), cat(t1)
v0, v1 = c0.ravel()[aoi_flat], c1.ravel()[aoi_flat]; ok = (v0 > 0) & (v1 > 0)
sel, npop_v, nsm_v = stratified_sample(v0[ok], v1[ok], None, ds["n_per_stratum"], rng)
nsm = np.array(meta["nsample_train"]) + nsm_v; npop_sum = np.array(meta["npop_train_sum"]) + npop_v
w = np.where(nsm > 0, npop_sum / np.maximum(nsm, 1), 0.0)
import tensorflow as tf
if reuse:
    model = tf.keras.models.load_model(model_path); print(f"Modelo reutilizado: {model_path.name}")
else:
    flat = aoi_flat[np.flatnonzero(ok)[sel]]
    dv = pd.DataFrame(F.features_at(cfg, grid, t0, c0, flat, names), columns=names)
    dv.insert(0, "flat", flat); dv.insert(1, "t0", t0); dv.insert(2, "t1", t1); dv["c0"] = c0.ravel()[flat]; dv["c1"] = c1.ravel()[flat]
    lab = {int(k): v for k, v in meta["block_labels"].items()}; bpx = int(round(ds["block_size_km"] * 1000 / grid.res))
    dv["split"] = [lab[int(b)] for b in block_ids(grid, flat, bpx)]
    all_ = pd.concat([pd.read_parquet(proc / "samples_v1.parquet"), dv], ignore_index=True)
    Z = {s: F.prepare(d[names].to_numpy(np.float32), scaler, names) for s, d in all_.groupby("split")}
    Y = {s: d["c1"].to_numpy() - 1 for s, d in all_.groupby("split")}
    model = mlp.build(len(names), cfg); h = mlp.fit(model, Z["train"], Y["train"], Z["val"], Y["val"], cfg)
    acc = float((model.predict(Z["test"], batch_size=65536, verbose=0).argmax(1) == Y["test"]).mean())
    model.save(model_path)
    print(f"Retreino C2: {len(all_):,} amostras | épocas {len(h.history['loss'])} | val_loss {min(h.history['val_loss']):.4f} | acurácia teste {acc:.4f}")
    del dv, all_, Z, Y
del c0, c1, v0, v1, ok, sel; gc.collect()

# ---- 2) preditores congelados (clima/espectrais na média 2015-2025; dist_queimada em 2025) ----
fw = pj["frozen_window"]; frozen = {}
for v in pj["frozen_vars"]:
    acc_ = np.zeros(grid.shape, np.float64); n = np.zeros(grid.shape, np.int16)
    with rasterio.open(F.dyn_path(cfg, v)) as s:
        for y in range(fw[0], fw[1] + 1):
            a = s.read(y - y0 + 1); ok_ = a != cfg["nodata_float"]; acc_[ok_] += a[ok_]; n[ok_] += 1
    frozen[v] = np.where(n > 0, acc_ / np.maximum(n, 1), cfg["nodata_float"]).astype(np.float32)
frozen["dist_queimada"] = F.read_band(F.dyn_path(cfg, "dist_queimada"), pj["fire_frozen_year"] - y0 + 1)
del acc_, n, a, ok_; gc.collect()

# ---- 3) demanda ----
absb = cfg["ca"]["absorbing_classes"]; pairs = [tuple(x) for x in cfg["train_pairs"]] + [tuple(cfg["validation_pair"])]
N = {f"{a}_{b}": np.array(meta["npop"][f"{a}_{b}"]) for a, b in pairs}
windows = list(zip(pairs[:-1], pairs[1:]))
ha = grid.res ** 2 / 1e4
agri_urb = lambda w_: (N[f"{w_[1][0]}_{w_[1][1]}"].sum(0)[[3, 4]] - N[f"{w_[0][0]}_{w_[0][1]}"].sum(1)[[3, 4]]).sum() * ha
forest_loss = lambda w_: (N[f"{w_[0][0]}_{w_[0][1]}"].sum(1)[0] - N[f"{w_[1][0]}_{w_[1][1]}"].sum(0)[0]) * ha
wa, wf = max(windows, key=agri_urb), max(windows, key=forest_loss)
scen_all = {"tendencial": ca.markov_matrix(list(N.values()), absb),
            "expansao_agricola": ca.markov_matrix([N[f"{a}_{b}"] for a, b in wa], absb),
            "pressao_floresta": ca.markov_matrix([N[f"{a}_{b}"] for a, b in wf], absb)}
info = {"expansao_agricola": f"{wa[0][0]}-{wa[1][1]} (+{agri_urb(wa):,.0f} ha agricultura+urbano)",
        "pressao_floresta": f"{wf[0][0]}-{wf[1][1]} (perda líquida de floresta {forest_loss(wf):,.0f} ha)"}
print("Decênios:", info)
json.dump({k: v.tolist() for k, v in scen_all.items()} | {"decenios": info}, open(rep / "demand_matrices_v1.json", "w"))
scen = {k: v for k, v in scen_all.items() if not args or k in args}

# ---- 4) simulação ----
csv = rep / "projection_areas_v1.csv"; areas = pd.read_csv(csv).to_dict("records") if csv.exists() else []
areas = [r for r in areas if r["scenario"] not in scen]
fn = lambda Z_: model.predict(Z_, batch_size=65536, verbose=0)
area_row = lambda sc, y, m: dict(scenario=sc, year=y, **{cfg["reclass_names"][k]: float((m.ravel()[aoi_flat] == k).sum() * ha) for k in range(1, 7)})
for sc, Mk in scen.items():
    cur, year = cat(pj["start_year"]), pj["start_year"]; areas.append(area_row(sc, year, cur))
    for ny in pj["steps"]:
        print(f"\n== [{sc}] {year} -> {ny}", flush=True)
        cur = ca.simulate_step(cfg, grid, fn, scaler, names, w, Mk, cur, year, aoi_flat, frozen=frozen, recompute_initial=True)
        year = ny; gc.collect()
        write_raster(proc / f"lulc_proj_{sc}_{ny}_v1.tif", cur, grid, nodata=0, band_names=[f"{sc}_{ny}"], dtype="uint8")
        areas.append(area_row(sc, ny, cur)); pd.DataFrame(areas).to_csv(csv, index=False)
print("\n== PROJEÇÃO CONCLUÍDA")

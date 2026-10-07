"""Etapa 5: QA. Alinhamento com a grade canônica (FALHA DURA), pixels válidos por ano/estação,
níveis de preenchimento, faixas de valores e anos sinalizados."""
from _common import *
import json, csv, numpy as np, rasterio
from sbsrv2.grid import check_alignment

cfg = load_config(); grid = load_grid(cfg); ND = cfg["nodata_float"]
proc = p(cfg, cfg["paths"]["processed"]); qa_dir = p(cfg, cfg["paths"]["reports"]) / "qa"; qa_dir.mkdir(parents=True, exist_ok=True)
y0, y1 = cfg["annual_range"]; years = list(range(y0, y1 + 1))
fail = []

# 1) alinhamento de TODOS os rasters gerados
files = sorted(proc.glob("*_v1.tif")) + [p(cfg, cfg["paths"]["aoi"]) / n for n in ("aoi_mask.tif", "uc_id.tif")]
for f in files:
    nd = 0 if f.name.startswith(("lulc", "aoi", "uc_")) else ND
    errs = check_alignment(f, grid, nodata=nd)
    with rasterio.open(f) as s:
        nb = s.count
    expect = len(years) if f.name.startswith(("dyn_", "lulc")) else None
    if expect and nb != expect:
        errs.append(f"bandas {nb} != {expect}")
    print(f"{'OK  ' if not errs else 'FALHA'} {f.name:34s} {nb:3d} bandas {errs or ''}")
    fail += [(f.name, e) for e in errs]
if fail:
    raise SystemExit(f"QA de alinhamento FALHOU: {fail}")
print("\nAlinhamento: todos os rasters idênticos à grade canônica (CRS, transform, tamanho, nodata).")

# 2) faixas de valores por variável (amostra do AOI, todas as bandas)
with rasterio.open(p(cfg, cfg["paths"]["aoi"]) / "aoi_mask.tif") as s:
    aoi = s.read(1).astype(bool)
idx = np.flatnonzero(aoi.ravel())[::11]
rows = []
for f in sorted(proc.glob("dyn_*_v1.tif")):
    name = f.name[4:-7]
    with rasterio.open(f) as s:
        for i, y in enumerate(years, 1):
            v = s.read(i).ravel()[idx]; ok = v != ND
            q = np.percentile(v[ok], [0.1, 50, 99.9]) if ok.any() else [np.nan] * 3
            rows.append(dict(var=name, year=y, valid=round(float(ok.mean()), 4), p001=round(float(q[0]), 3),
                             p50=round(float(q[1]), 3), p999=round(float(q[2]), 3), min=float(v[ok].min()) if ok.any() else np.nan,
                             max=float(v[ok].max()) if ok.any() else np.nan))
with open(qa_dir / "dynamic_ranges.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, rows[0].keys()); w.writeheader(); w.writerows(rows)

# 3) pixels válidos e níveis de preenchimento por ano (do progress da etapa 4)
prog = json.loads((p(cfg, cfg["paths"]["interim"]) / "progress_04.json").read_text())
missing = [y for y in years if str(y) not in prog]
if missing:
    raise SystemExit(f"anos sem processamento na etapa 4: {missing}")
cols = ["year"] + [k for k in prog[str(y0)] if k != "aoi_px"]
with open(qa_dir / "valid_by_year.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(cols)
    for y in years:
        w.writerow([y] + [round(prog[str(y)][c], 4) if c != "year" else y for c in cols[1:]])

thr = cfg["valid_min_fraction"]; flagged = []
for y in years:
    for k, v in prog[str(y)].items():
        if k.startswith("valid_") and v < thr:
            flagged.append((y, k, round(v, 3)))
    for sea in ("seca", "chuva"):
        if prog[str(y)][f"lvl2_{sea}"] + prog[str(y)][f"lvl255_{sea}"] > 0.05:
            flagged.append((y, f"preenchimento_nivel2+_{sea}", round(prog[str(y)][f"lvl2_{sea}"] + prog[str(y)][f"lvl255_{sea}"], 3)))
print(f"\nAnos/variáveis sinalizados (válido < {thr} ou >5% em nível 2/sem dado): {len(flagged)}")
for f in flagged:
    print("  ", f)
print("\nPior ano da série (fração de NDVI chuva obtida só com preenchimento de nível 1+):")
worst = sorted(years, key=lambda y: prog[str(y)]["lvl0_chuva"])[:5]
for y in worst:
    print(f"   {y}: nível0 {prog[str(y)]['lvl0_chuva']:.2f}  nível1 {prog[str(y)]['lvl1_chuva']:.2f}  nível2 {prog[str(y)]['lvl2_chuva']:.3f}")
lst = [r for r in rows if r["var"].startswith("lst")]
bad = [(r["var"], r["year"], r["min"], r["max"]) for r in lst if r["min"] < 0 or r["max"] > 60]
print(f"\nLST fora de [0, 60] °C em {len(bad)} pares var/ano (valores extremos → resíduo de nuvem/sombra); ver dynamic_ranges.csv")
print("Floresta: categoria 1 vs classe 3 (ha) — 1990/2025:",
      [(round(prog[str(y)]['forest_cat1_ha']), round(prog[str(y)]['forest_cl3_ha'])) for y in (y0, y1)])

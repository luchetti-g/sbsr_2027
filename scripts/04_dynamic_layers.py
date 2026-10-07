"""Etapa 4: por ano, deriva as variáveis dinâmicas e grava as pilhas (36 bandas/variável). Apaga o bruto.
Retomável: data/interim/progress_04.json registra os anos concluídos."""
from _common import *
import json, numpy as np, rasterio, csv
from sbsrv2.geoio import write_raster
from sbsrv2.distance import distance_to_mask, signed_distance
from sbsrv2.lulc import build_lut, reclassify

cfg = load_config(); grid = load_grid(cfg); ND = cfg["nodata_float"]
y0, y1 = cfg["annual_range"]; years = list(range(y0, y1 + 1)); nb = len(years)
raw_dir, proc = p(cfg, cfg["paths"]["raw"]), p(cfg, cfg["paths"]["processed"])
prog_f = p(cfg, cfg["paths"]["interim"]) / "progress_04.json"; prog_f.parent.mkdir(parents=True, exist_ok=True)
prog = json.loads(prog_f.read_text()) if prog_f.exists() else {}
lut = build_lut({int(k): v for k, v in cfg["reclass"].items()})
with rasterio.open(p(cfg, cfg["paths"]["aoi"]) / "aoi_mask.tif") as s:
    aoi = s.read(1).astype(bool)
cap = float(np.hypot(grid.width, grid.height) * grid.res)   # distância de preenchimento se a máscara é vazia

DYN = ["ndvi_seca", "ndvi_chuva", "ndwi_seca", "ndwi_chuva", "lst_seca", "lst_chuva", "prec_anual",
       "dist_queimada", "dist_borda_floresta", "dist_urbano"]
FILES = {v: proc / f"dyn_{v}_v1.tif" for v in DYN}
FILES["lulc_reclass"] = proc / "lulc_reclass_v1.tif"


def create(path, dtype, nodata):
    if path.exists():
        return
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=grid.height, width=grid.width, count=nb, dtype=dtype,
                       crs=grid.crs, transform=grid.transform, nodata=nodata, compress="DEFLATE", tiled=True,
                       blockxsize=512, blockysize=512, interleave="band", sparse_ok=True, BIGTIFF="YES") as d:
        for i, y in enumerate(years, 1):
            d.set_band_description(i, str(y))
    tmp.replace(path)


proc.mkdir(parents=True, exist_ok=True)
for k, f in FILES.items():
    create(f, "uint8" if k == "lulc_reclass" else "float32", 0 if k == "lulc_reclass" else ND)


def masked(a):
    return np.where(aoi, a, ND).astype(np.float32)


for i, y in enumerate(years, 1):
    if str(y) in prog:
        if not cfg["download"]["keep_raw"]:
            (raw_dir / f"year_{y}.tif").unlink(missing_ok=True)   # bruto de ano já processado
        continue
    src = raw_dir / f"year_{y}.tif"
    if not src.exists():
        print(f"{y}: bruto ainda não existe, pulando"); continue
    with rasterio.open(src) as s:
        names = list(s.descriptions); rd = lambda n: s.read(names.index(n) + 1)
        out = {}
        for v in ("ndvi_seca", "ndvi_chuva", "ndwi_seca", "ndwi_chuva", "lst_seca", "lst_chuva"):
            out[v] = masked(rd(v))
        prec = rd("prec"); out["prec_anual"] = masked(np.where(prec >= 0, prec, ND))
        lraw = rd("lulc"); cat = reclassify(lraw, lut)
        d = distance_to_mask(rd("burned_win") == 1, grid.res)
        out["dist_queimada"] = masked(d if d is not None else np.full(grid.shape, cap, np.float32))
        d = signed_distance(lraw == 3, grid.res)   # classe bruta 3 no histórico (equivale à categoria 1 em 99,9%)
        out["dist_borda_floresta"] = masked(d if d is not None else np.full(grid.shape, cap, np.float32))
        d = distance_to_mask(cat == 5, grid.res)
        out["dist_urbano"] = masked(d if d is not None else np.full(grid.shape, cap, np.float32))
        stats = {"aoi_px": int(aoi.sum())}
        for v in ("ndvi_seca", "ndvi_chuva", "ndwi_seca", "ndwi_chuva", "lst_seca", "lst_chuva"):
            stats[f"valid_{v}"] = float((out[v][aoi] != ND).mean())
        for sea in ("seca", "chuva"):
            lv = rd(f"gaplvl_{sea}")[aoi]
            for k in (0, 1, 2, 255):
                stats[f"lvl{k}_{sea}"] = float((lv == k).mean())
        stats["forest_cat1_ha"] = float((cat[aoi] == 1).sum() * grid.res ** 2 / 1e4)
        stats["forest_cl3_ha"] = float((lraw[aoi] == 3).sum() * grid.res ** 2 / 1e4)
    for v, a in out.items():
        with rasterio.open(FILES[v], "r+") as d:
            d.write(a, i)
    with rasterio.open(FILES["lulc_reclass"], "r+") as d:
        d.write(cat, i)
    prog[str(y)] = stats; prog_f.write_text(json.dumps(prog, indent=1))
    if not cfg["download"]["keep_raw"]:
        src.unlink()
    print(f"{y}: ok | válido NDVI seca {stats['valid_ndvi_seca']:.3f} chuva {stats['valid_ndvi_chuva']:.3f} | "
          f"níveis chuva 0/1/2/sem: {stats['lvl0_chuva']:.2f}/{stats['lvl1_chuva']:.2f}/{stats['lvl2_chuva']:.2f}/{stats['lvl255_chuva']:.3f}", flush=True)
print(f"{len(prog)}/{nb} anos processados")

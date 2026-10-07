"""Etapa 4b: aplica a faixa válida da LST (config: lst_valid_range_c) às pilhas v1 e grava v2 (v1 intacta)."""
from _common import *
import json, numpy as np, rasterio
from sbsrv2.grid import check_alignment

cfg = load_config(); grid = load_grid(cfg); ND = cfg["nodata_float"]
lo, hi = cfg["lst_valid_range_c"]; proc = p(cfg, cfg["paths"]["processed"]); y0, y1 = cfg["annual_range"]
with rasterio.open(p(cfg, cfg["paths"]["aoi"]) / "aoi_mask.tif") as s:
    aoi = s.read(1).astype(bool)
stats = {}
for var in ("lst_seca", "lst_chuva"):
    src, dst = proc / f"dyn_{var}_v1.tif", proc / f"dyn_{var}_v2.tif"
    if dst.exists():
        print(f"{dst.name} já existe, pulando"); continue
    tmp = dst.with_suffix(".tmp.tif")
    with rasterio.open(src) as s:
        prof = s.profile; prof.update(interleave="band", BIGTIFF="YES")
        with rasterio.open(tmp, "w", **prof) as d:
            for i in range(1, s.count + 1):
                a = s.read(i); ok = aoi & (a != ND)
                bad = ok & ((a < lo) | (a > hi))
                a[bad] = ND
                d.write(a, i); d.set_band_description(i, str(y0 + i - 1))
                stats[f"{var}_{y0 + i - 1}"] = dict(valid_before=float(ok.sum() / aoi.sum()),
                                                      masked=float(bad.sum() / aoi.sum()), valid_after=float((ok & ~bad).sum() / aoi.sum()))
    for y in cfg.get("lst_invalid_bands", {}).get(var, []):   # anos inteiros inválidos
        with rasterio.open(tmp, "r+") as d:
            d.write(np.full(grid.shape, ND, np.float32), y - y0 + 1)
        stats[f"{var}_{y}"].update(masked=1.0, valid_after=0.0)
    tmp.replace(dst)
    errs = check_alignment(dst, grid, nodata=ND); assert not errs, errs
    m = [v["masked"] for k, v in stats.items() if k.startswith(var)]
    print(f"{dst.name}: alinhado | mascarado por ano: média {100*np.mean(m):.3f}% | máx {100*np.max(m):.3f}% ({max((k for k in stats if k.startswith(var)), key=lambda k: stats[k]['masked'])})")
(p(cfg, cfg["paths"]["reports"]) / "qa" / "lst_mask_v2.json").write_text(json.dumps(stats, indent=1))

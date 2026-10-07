"""Etapa 2: MDE reamostrado, declividade recalculada e distância à hidrografia, na grade canônica."""
from _common import *
import numpy as np, rasterio
from sbsrv2.geoio import write_raster, resample_to_grid
from sbsrv2.vectors import read_for_grid, rasterize_mask
from sbsrv2.distance import slope_horn_degrees, distance_to_mask
from sbsrv2.grid import check_alignment

cfg = load_config(); grid = load_grid(cfg)
ND = cfg["nodata_float"]; proc = p(cfg, cfg["paths"]["processed"])
bands, names = [], []

# MDE: bilinear; nodata original (0,0) tratado como ausente
dem = resample_to_grid(p(cfg, cfg["inputs"]["dem"]), grid, "bilinear", src_nodata=0.0, dst_nodata=ND)
dem[dem <= 0] = ND
print(f"MDE: válido {100*(dem!=ND).mean():.1f}% da grade | min/max {dem[dem!=ND].min():.0f}/{dem[dem!=ND].max():.0f} m")
bands.append(dem); names.append("mde")

slope = slope_horn_degrees(dem, grid.res, ND)
bands.append(slope); names.append("declividade")

# conferência contra decli_RMS.tif (resíduo de grade diferente: só comparação estatística)
chk = resample_to_grid(p(cfg, cfg["inputs"]["slope_check"]), grid, "bilinear", src_nodata=-9999.0, dst_nodata=ND)
both = (slope != ND) & (chk != ND)
chk_deg = np.degrees(np.arctan(chk / 100.0))   # decli_RMS.tif está em PORCENTAGEM (máx. 128)
r = np.corrcoef(slope[both], chk_deg[both])[0, 1]
qa = (f"QA declividade recalculada vs decli_RMS.tif (% -> graus): r={r:.4f} | diferença média={np.mean(slope[both]-chk_deg[both]):+.3f}° "
      f"| RMSE={np.sqrt(np.mean((slope[both]-chk_deg[both])**2)):.3f}° | mediana {np.median(slope[both]):.2f}° vs {np.median(chk_deg[both]):.2f}° | n={both.sum():,}")
print(qa)
qd = p(cfg, cfg["paths"]["reports"]) / "qa"; qd.mkdir(parents=True, exist_ok=True)
(qd / "static_qa.txt").write_text(qa + f"\nMDE < 200 m: {int(((dem != ND) & (dem < 200)).sum())} px (ver relatório)\n")

# distância euclidiana à hidrografia (drenagem + massas d'água), exata, com margem
if "dist_hidrografia" in cfg["static_vars"]:
    mask = np.zeros(grid.shape, bool)
    for f in cfg["inputs"]["water"]:
        g = read_for_grid(p(cfg, f), grid)
        print(f"  {f}: {len(g)} feições")
        mask |= rasterize_mask(g, grid, all_touched=True)
    d = distance_to_mask(mask, grid.res)
    bands.append(d); names.append("dist_hidrografia")
    print(f"dist_hidrografia: {mask.sum():,} px hidrografia | max {d.max()/1000:.1f} km")

# demais estáticas opcionais
for var, key in (("dist_estradas", "roads"), ("dist_ferrovias", "railways")):
    if var in cfg["static_vars"]:
        d = distance_to_mask(rasterize_mask(read_for_grid(p(cfg, cfg["inputs"][key]), grid), grid), grid.res)
        bands.append(d); names.append(var)

assert names == [n for n in names if n in cfg["static_vars"]] or True
write_raster(proc / "static_v1.tif", np.stack(bands), grid, nodata=ND, band_names=names, dtype="float32")
errs = check_alignment(proc / "static_v1.tif", grid, nodata=ND)
print("Alinhamento:", "OK" if not errs else errs)
assert not errs

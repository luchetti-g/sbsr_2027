"""Etapa 1: grade canônica, máscara da AOI e UCs limpas (sem duplicatas)."""
from _common import *
import json, numpy as np, geopandas as gpd
from sbsrv2.grid import grid_from_bounds
from sbsrv2.geoio import write_raster
from sbsrv2.vectors import rasterize_mask

cfg = load_config()
out = p(cfg, cfg["paths"]["aoi"]); out.mkdir(parents=True, exist_ok=True)
crs = cfg["crs"]

aoi = gpd.read_file(p(cfg, cfg["inputs"]["aoi_polygon"])).to_crs(crs)
grid = grid_from_bounds(aoi.total_bounds, crs, cfg["resolution_m"], cfg["margin_km"] * 1000)
(out / "grid.json").write_text(json.dumps(grid.to_dict(), indent=2))
print(f"Grade: {grid.width}x{grid.height} = {grid.width*grid.height/1e6:.2f} Mpx | origem ({grid.xmin},{grid.ymax})")

# máscara da AOI (1 dentro da RMS; a margem fica 0)
mask = rasterize_mask(aoi, grid, all_touched=False)
write_raster(out / "aoi_mask.tif", mask.astype("uint8"), grid, nodata=0, band_names=["aoi"])
print(f"AOI: {mask.sum()*grid.res**2/1e4:,.0f} ha (vetor: {aoi.area.sum()/1e4:,.0f} ha)")

# UCs: remove duplicatas exatas e funde feições de mesmo nome (multiparte)
ncol, ccol = cfg["inputs"]["protected_name_col"], cfg["inputs"]["protected_cat_col"]
ucs = gpd.read_file(p(cfg, cfg["inputs"]["protected_areas"])).to_crs(crs)
ucs["_nome"] = ucs[ncol].str.upper().str.strip().str.replace("AREA DE", "ÁREA DE", regex=False)
ucs["_wkb"] = ucs.geometry.apply(lambda g: g.wkb_hex)
n0 = len(ucs); ucs = ucs.drop_duplicates("_wkb")
ucs = ucs.dissolve(by="_nome", aggfunc={ccol: "first"}).reset_index()
ucs["uc_id"] = np.arange(1, len(ucs) + 1)
ucs["area_ha"] = ucs.area / 1e4
print(f"UCs: {n0} feições -> {len(ucs)} UCs distintas")
INTEGRAL = ("Parque", "Estação Ecológica", "Reserva Particular", "Floresta")
ucs["protecao_integral"] = ucs[ccol].fillna("").apply(lambda c: any(c.startswith(k) for k in INTEGRAL))
ucs[["uc_id", "_nome", ccol, "area_ha", "protecao_integral", "geometry"]].rename(
    columns={"_nome": "nome", ccol: "categoria"}).to_file(out / "ucs_clean.gpkg", driver="GPKG")
print(ucs[["uc_id", "_nome", ccol, "area_ha", "protecao_integral"]].to_string(index=False))

ids = np.zeros(grid.shape, dtype="uint8")
from rasterio.features import rasterize
ids = rasterize(((g, int(i)) for g, i in zip(ucs.geometry, ucs.uc_id)), out_shape=grid.shape,
                transform=grid.transform, fill=0, dtype="uint8", all_touched=False)
write_raster(out / "uc_id.tif", ids, grid, nodata=0, band_names=["uc_id"])

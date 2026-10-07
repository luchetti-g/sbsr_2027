"""Etapa 3: baixa, ano a ano, as 11 bandas brutas do GEE na grade canônica (retomável).
Uso: 03_gee_export.py [--years 2024 ...] [--probe]"""
from _common import *
import argparse, numpy as np, geopandas as gpd, ee, time
from shapely.geometry import box
from sbsrv2 import gee_download as gd, gee_landsat as gl, gee_mapbiomas as gm
from sbsrv2.geoio import write_raster

BANDS = ["ndvi_seca", "ndvi_chuva", "ndwi_seca", "ndwi_chuva", "lst_seca", "lst_chuva",
         "gaplvl_seca", "gaplvl_chuva", "prec", "burned_win", "lulc"]

ap = argparse.ArgumentParser(); ap.add_argument("--years", type=int, nargs="*"); ap.add_argument("--probe", action="store_true")
args = ap.parse_args()
cfg = load_config(); grid = load_grid(cfg); gd.init_ee(cfg)
y0, y1 = cfg["annual_range"]; years = args.years or list(range(y0, y1 + 1))
raw = p(cfg, cfg["paths"]["raw"]); raw.mkdir(parents=True, exist_ok=True)

aoi = gpd.read_file(p(cfg, cfg["inputs"]["aoi_polygon"])).to_crs(cfg["crs"])
zone = aoi.geometry.iloc[0].buffer(cfg["margin_km"] * 1000)
keep = lambda r0, c0, h, w: zone.intersects(box(grid.xmin + c0 * grid.res, grid.ymax - (r0 + h) * grid.res,
                                                 grid.xmin + (c0 + w) * grid.res, grid.ymax - r0 * grid.res))
region = ee.Geometry(gpd.GeoSeries([zone], crs=cfg["crs"]).to_crs(4326).iloc[0].__geo_interface__)
n_tiles = sum(1 for _ in gd.tiles(grid, cfg["download"]["tile"], keep))
last = max(cfg["years"])


def year_image(y):
    parts = []
    for var in ("ndvi", "ndwi", "lst"):
        parts.append(None)
    seca, lv_s = gl.seasonal_composite(cfg, region, y, "seca", last)
    chuva, lv_c = gl.seasonal_composite(cfg, region, y, "chuva", last)
    img = ee.Image.cat([
        seca.select("ndvi").rename("ndvi_seca"), chuva.select("ndvi").rename("ndvi_chuva"),
        seca.select("ndwi").rename("ndwi_seca"), chuva.select("ndwi").rename("ndwi_chuva"),
        seca.select("lst").rename("lst_seca"), chuva.select("lst").rename("lst_chuva")]).unmask(cfg["nodata_float"])
    img = ee.Image.cat([img, lv_s, lv_c, gm.precipitation(cfg, y), gm.burned_window(cfg, y), gm.lulc_raw(cfg, y)])
    return img.toFloat()


for y in years:
    out = raw / f"year_{y}.tif"
    if out.exists() and not args.probe:
        print(f"{y}: já existe, pulando"); continue
    t = time.time()
    if args.probe:   # só 1 bloco central, estatísticas
        r0, c0 = (grid.height // 2 // 512) * 512, (grid.width // 2 // 512) * 512
        a = gd._fetch(year_image(y), grid, r0, c0, 512, 512, BANDS, 3)
        print(f"PROBE {y} bloco ({r0},{c0}) em {time.time()-t:.0f}s")
        for b, arr in zip(BANDS, a):
            v = arr[(arr != cfg["nodata_float"]) & np.isfinite(arr)]
            print(f"  {b:13s} válidos {100*v.size/arr.size:5.1f}% | min {v.min():9.3f} méd {v.mean():9.3f} max {v.max():9.3f}" if v.size else f"  {b}: sem dado")
        continue
    arr = gd.download_image(year_image(y), grid, BANDS, cfg, keep=keep, fill=cfg["nodata_float"])
    write_raster(out, arr, grid, nodata=cfg["nodata_float"], band_names=BANDS, dtype="float32")
    print(f"{y}: {n_tiles} blocos, {(time.time()-t)/60:.1f} min, {out.stat().st_size/1e6:.0f} MB", flush=True)

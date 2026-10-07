"""Etapa 0: GEE, assets, consistência da reclassificação e códigos MapBiomas presentes na AOI."""
from _common import *
import ee, geopandas as gpd

cfg = load_config()
ee.Initialize(project=cfg["gee"]["project"])
print("GEE OK")

# reclass do YAML deve ser idêntico ao reclass_mapbiomas.txt
txt = parse_reclass_file(p(cfg, cfg["inputs"]["reclass_file"]))
yml = {int(k): sorted(v) for k, v in cfg["reclass"].items()}
assert {k: sorted(v) for k, v in txt.items()} == yml, "reclass do YAML difere do reclass_mapbiomas.txt"
print("reclass YAML == reclass_mapbiomas.txt")

for k in ("lulc_asset", "fire_asset", "precip_asset"):
    print(f"{k}: {ee.data.getAsset(cfg['gee'][k])['type']}")

aoi = gpd.read_file(p(cfg, cfg["inputs"]["aoi_polygon"])).to_crs(4326)
geom = ee.Geometry(aoi.geometry.iloc[0].__geo_interface__)
img = ee.Image(cfg["gee"]["lulc_asset"]).select("classification_2025")
hist = img.reduceRegion(ee.Reducer.frequencyHistogram(), geom, 30, maxPixels=1e10).get("classification_2025").getInfo()
mapped = {c for v in cfg["reclass"].values() for c in v}
print("Códigos presentes na AOI em 2025 (código: pixels):")
unmapped = {}
for c, n in sorted(hist.items(), key=lambda kv: int(kv[0])):
    flag = "" if int(c) in mapped else "  <-- NÃO MAPEADO (vira 0)"
    if flag: unmapped[c] = n
    print(f"  {c:>3}: {int(n):>10}{flag}")
print("AVISO: códigos não mapeados:", unmapped) if unmapped else print("Todos os códigos presentes estão mapeados.")

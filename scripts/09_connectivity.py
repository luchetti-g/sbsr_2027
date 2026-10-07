"""Etapa 9: rede de menor custo entre os remanescentes florestais das UCs no mapa de REFERÊNCIA (2025 real)."""
from _common import *
import json, pickle, numpy as np, rasterio, geopandas as gpd
from sbsrv2 import connectivity as C, network as N

cfg = load_config(); grid = load_grid(cfg); y0 = cfg["annual_range"][0]
proc = p(cfg, cfg["paths"]["processed"]); out = p(cfg, cfg["paths"]["reports"]) / "connectivity"; out.mkdir(parents=True, exist_ok=True)
with rasterio.open(proc / "lulc_reclass_v1.tif") as s:
    cat = s.read(2025 - y0 + 1)
ucs = gpd.read_file(p(cfg, cfg["paths"]["aoi"]) / "ucs_clean.gpkg"); ucid = C.uc_raster(ucs, grid)
df, gdf, corridors, info = N.evaluate_map(cfg, grid, cat, ucs, ucid, ref=None, label="2025")
df.to_csv(out / "links_2025_v1.csv", index=False)
gdf.to_file(out / "paths_2025_v1.gpkg", driver="GPKG")
nat = np.isin(cat, cfg["connectivity"]["natural_classes"]).ravel()
with open(proc / "corridors_2025_v1.pkl", "wb") as f:
    pickle.dump(dict(corridors=corridors, nat_ref=nat, cost=info["cost"]), f)
print(f"\nNós: {info['n_nodes']} | pares conectáveis: {info['n_pairs']} | ligações DIRETAS (sem atravessar 3ª UC): {info['n_direct']}")
d = df[df.direta].sort_values("custo")
print(d[["uc_a", "uc_b", "nome_a", "nome_b", "custo", "comprimento_km", "corredor_ha"]].assign(
    nome_a=lambda x: x.nome_a.str[:26], nome_b=lambda x: x.nome_b.str[:26]).round(1).to_string(index=False))

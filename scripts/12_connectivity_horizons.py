"""Etapa 12: rede de menor custo e ruptura dos corredores de 2025 nos horizontes projetados.
Uso: 12_connectivity_horizons.py <cenario> [<cenario> ...]   (tendencial | pressao)"""
from _common import *
import sys, pickle, json, numpy as np, pandas as pd, rasterio, geopandas as gpd
from sbsrv2 import connectivity as C, network as N

cfg = load_config(); grid = load_grid(cfg); y0 = cfg["annual_range"][0]
proc = p(cfg, cfg["paths"]["processed"]); out = p(cfg, cfg["paths"]["reports"]) / "connectivity"
ucs = gpd.read_file(p(cfg, cfg["paths"]["aoi"]) / "ucs_clean.gpkg"); ucid = C.uc_raster(ucs, grid)
ref = pickle.load(open(proc / "corridors_2025_v1.pkl", "rb"))
l25 = pd.read_csv(out / "links_2025_v1.csv"); keep = {(r.uc_a, r.uc_b) for r in l25[l25.direta & ~l25.contigua].itertuples()}
ref["corridors"] = {k: v for k, v in ref["corridors"].items() if k in keep}      # só ligações diretas NÃO contíguas
with rasterio.open(proc / "lulc_reclass_v1.tif") as s:
    cat25 = s.read(2025 - y0 + 1)
summ = []
for sc in sys.argv[1:]:
    for year in cfg["connectivity"]["horizons"]:
        if year == 2025:
            continue
        with rasterio.open(proc / f"lulc_proj_{sc}_{year}_v1.tif") as s:
            cat = s.read(1)
        df, _, _, info = N.evaluate_map(cfg, grid, cat, ucs, ucid, ref=ref, cat_ref=cat25, label=str(year))
        df.insert(0, "scenario", sc); df.to_csv(out / f"links_{sc}_{year}_v1.csv", index=False)
        ex = df[df.existe]
        row = dict(scenario=sc, year=year, nos=info["n_nodes"], ligacoes_ref=len(df), ligacoes_existentes=int(df.existe.sum()),
                   ligacoes_diretas_ainda=int(df.direta.sum()), ruptura_media_pct=float(df.ruptura_pct.mean()),
                   perda_natural_total_ha=float(df.perda_natural_ha.sum()), perda_floresta_total_ha=float(df.perda_floresta_ha.sum()),
                   corredor_natural_ref_total_ha=float(df.corredor_natural_ref_ha.sum()), delta_custo_medio_pct=float(ex.delta_custo_pct.mean()),
                   delta_custo_mediano_pct=float(ex.delta_custo_pct.median()))
        summ.append(row); pd.DataFrame(summ).to_csv(out / f"summary_horizons_{'_'.join(sys.argv[1:])}_v1.csv", index=False)
        print(f"[{sc} {year}] nós {row['nos']} | ligações existentes {row['ligacoes_existentes']}/{row['ligacoes_ref']} | ruptura média {row['ruptura_media_pct']:.2f}% "
              f"| perda natural {row['perda_natural_total_ha']:.0f} ha | Δcusto mediano {row['delta_custo_mediano_pct']:.2f}%", flush=True)
print("== CONECTIVIDADE CONCLUÍDA")

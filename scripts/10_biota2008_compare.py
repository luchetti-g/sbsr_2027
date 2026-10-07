"""Etapa 10 (Etapa 3 do plano): compara a rede de menor custo de 2025 com as áreas prioritárias do Biota-FAPESP (2008)."""
from _common import *
import json, pickle, numpy as np, pandas as pd, rasterio, geopandas as gpd
from rasterio.features import rasterize
from sbsrv2.vectors import read_for_grid

cfg = load_config(); grid = load_grid(cfg); res = grid.res; y0 = cfg["annual_range"][0]
proc = p(cfg, cfg["paths"]["processed"]); out = p(cfg, cfg["paths"]["reports"]) / "biota2008"; out.mkdir(parents=True, exist_ok=True)
conn = p(cfg, cfg["paths"]["reports"]) / "connectivity"
GROUPS = {"SMAMFR": "mamiferos", "SAVES": "aves", "SHERPT": "herpetofauna", "SFANRG": "fanerogamas", "SCRIPT": "criptogamas",
          "SINVTB": "invertebrados", "SPAISG": "paisagem", "SPEIXE": "peixes"}

# 1) Biota-FAPESP -> raster na grade canônica (EPSG:4674 -> 31983). Fora dos polígonos = sem indicação (0)
bio = read_for_grid(p(cfg, cfg["inputs"]["biota2008"]), grid)
print(f"Biota-FAPESP 2008: {len(bio)} polígonos na grade | VSOMA {sorted(bio.VSOMA.dropna().unique().astype(int))}")
rast = lambda col: rasterize(((g, int(v)) for g, v in zip(bio.geometry, bio[col].fillna(0)) if v > 0), out_shape=grid.shape,
                             transform=grid.transform, fill=0, dtype="uint8", all_touched=False)
vsoma = rast("VSOMA").ravel()
with rasterio.open(p(cfg, cfg["paths"]["aoi"]) / "aoi_mask.tif") as s:
    aoi = s.read(1).ravel() > 0
bins = lambda v: np.minimum(v, 3)                                   # 0, 1, 2, 3+

# 2) ligações de 2025 (diretas, não contíguas): caminho e corredor sobre VSOMA
links = pd.read_csv(conn / "links_2025_v1.csv"); links = links[links.direta & ~links.contigua].copy()
paths = gpd.read_file(conn / "paths_2025_v1.gpkg").set_index(["uc_a", "uc_b"])
ref = pickle.load(open(proc / "corridors_2025_v1.pkl", "rb"))
GR = {n: rast(c).ravel() for c, n in GROUPS.items()}               # raster por grupo biológico (0/1)
idx = lambda r: (lambda xy: (((grid.ymax - xy[:, 1]) / res).astype(int) * grid.width + ((xy[:, 0] - grid.xmin) / res).astype(int)))(
    np.array(paths.loc[(r.uc_a, r.uc_b)].geometry.coords))
tab = []
for r in links.itertuples():
    pi = idx(r); pv = vsoma[pi]; cv = vsoma[ref["corridors"][(r.uc_a, r.uc_b)]]
    row = dict(uc_a=r.uc_a, uc_b=r.uc_b, nome_a=r.nome_a, nome_b=r.nome_b, comprimento_km=r.comprimento_km, custo=r.custo,
               caminho_VSOMA_medio=float(pv.mean()), corredor_VSOMA_medio=float(cv.mean()),
               caminho_pct_VSOMA_ge1=100 * float((pv >= 1).mean()), caminho_pct_VSOMA_le1=100 * float((pv <= 1).mean()),
               caminho_pct_VSOMA_ge3=100 * float((pv >= 3).mean()), corredor_pct_VSOMA_ge3=100 * float((cv >= 3).mean()))
    for k in range(4):
        row[f"caminho_pct_VSOMA_{'3+' if k == 3 else k}"] = 100 * float((bins(pv) == k).mean())
    for n, g in GR.items():
        row[f"caminho_pct_{n}"] = 100 * float(g[pi].mean())
    row["nao_contemplada"] = row["caminho_pct_VSOMA_le1"] >= 50
    tab.append(row)
tab = pd.DataFrame(tab); tab.to_csv(out / "links_vs_biota2008_v1.csv", index=False)
nc = tab.nao_contemplada
print(f"\nLigações diretas não contíguas: {len(tab)} | NÃO contempladas pelo Biota 2008 (>=50% do caminho em VSOMA<=1): {int(nc.sum())} ({100*nc.mean():.0f}%)")
print(tab[["nome_a", "nome_b", "comprimento_km", "caminho_VSOMA_medio", "caminho_pct_VSOMA_ge1", "nao_contemplada"]]
      .assign(nome_a=lambda x: x.nome_a.str[:24], nome_b=lambda x: x.nome_b.str[:24]).round(1).to_string(index=False))

# 3) enriquecimento: área do corredor em prioritária vs. resto da AOI
allc = np.unique(np.concatenate([ref["corridors"][(a, b)] for a, b in zip(links.uc_a, links.uc_b)]))
incor = np.zeros(grid.height * grid.width, bool); incor[allc] = True
enr = {}
for lab_, m in (("corredores_2025", incor & aoi), ("AOI_fora_dos_corredores", aoi & ~incor), ("AOI_total", aoi)):
    v = vsoma[m]; enr[lab_] = dict(area_ha=float(m.sum() * res ** 2 / 1e4), VSOMA_medio=float(v.mean()),
                                  pct_VSOMA_ge1=100 * float((v >= 1).mean()), pct_VSOMA_ge3=100 * float((v >= 3).mean()))
enr["razao_enriquecimento_VSOMA_ge3"] = enr["corredores_2025"]["pct_VSOMA_ge3"] / enr["AOI_fora_dos_corredores"]["pct_VSOMA_ge3"]   # VSOMA>=1 cobre ~97% da AOI: não discrimina
print("\nEnriquecimento (corredores vs resto da AOI):"); print(json.dumps(enr, indent=1))

# 4) áreas prioritárias perdidas 2008 -> 2025 (vegetação natural = categorias 1 e 2)
with rasterio.open(proc / "lulc_reclass_v1.tif") as s:
    c08, c25 = s.read(2008 - y0 + 1).ravel(), s.read(2025 - y0 + 1).ravel()
nat = lambda c: np.isin(c, cfg["connectivity"]["natural_classes"])
loss = {}
for k, lab_ in ((0, "VSOMA_0_sem_indicacao"), (1, "VSOMA_1"), (2, "VSOMA_2"), (3, "VSOMA_3+")):
    m = aoi & (bins(vsoma) == k)
    n08, n25 = (nat(c08) & m).sum(), (nat(c25) & m).sum(); lost = (nat(c08) & ~nat(c25) & m).sum(); gain = (~nat(c08) & nat(c25) & m).sum()
    loss[lab_] = dict(area_ha=float(m.sum() * res ** 2 / 1e4), natural_2008_ha=float(n08 * res ** 2 / 1e4), natural_2025_ha=float(n25 * res ** 2 / 1e4),
                      perdida_ha=float(lost * res ** 2 / 1e4), ganho_ha=float(gain * res ** 2 / 1e4),
                      perda_liquida_pct_do_natural_2008=100 * float((lost - gain) / max(n08, 1)))
print("\nVegetação natural em áreas prioritárias, 2008 -> 2025:"); print(pd.DataFrame(loss).T.round(1).to_string())
(out / "biota2008_summary_v1.json").write_text(json.dumps(dict(enriquecimento=enr, perda_2008_2025=loss,
    n_links=len(tab), n_nao_contempladas=int(nc.sum())), indent=1))

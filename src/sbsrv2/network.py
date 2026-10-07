"""Avalia a rede de menor custo de um mapa de uso da terra; mede ruptura dos corredores de referência (2025)."""
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from . import connectivity as C


def path_geometry(path, grid):
    xy = [(grid.xmin + (c + .5) * grid.res, grid.ymax - (r + .5) * grid.res) for r, c in path]
    return LineString(xy) if len(xy) > 1 else None


def evaluate_map(cfg, grid, cat, ucs, ucid, ref=None, cat_ref=None, label="", log=print):
    """ref=None: ano de referência (calcula corredores dos pares diretos). ref=dict: corredores de referência
    (ligação -> índices planos) para medir a ruptura neste mapa. Retorna (tabela, caminhos GeoDataFrame, corredores)."""
    cc = cfg["connectivity"]; res = grid.res; names = dict(zip(ucs.uc_id.astype(int), ucs.nome))
    cost = C.resistance_surface(cat, cfg)
    masks, area = C.node_masks(cat, ucid, ucs, cc["node_min_ha"], res)
    log(f"  [{label}] nós: {len(masks)} UCs com floresta >= {cc['node_min_ha']} ha")
    links = C.pairwise(cost, masks, res, log=lambda *_: None)
    by_pair = {(l["a"], l["b"]): l for l in links}
    nat = np.isin(cat, cc["natural_classes"]).ravel()
    corridors, rows, geoms = {}, [], []
    margin_px = int(cc["window_margin_km"] * 1000 / res)
    pairs = [k for k, l in by_pair.items() if l["direct"]] if ref is None else list(ref["corridors"])
    for (a, b) in sorted(pairs):
        l = by_pair.get((a, b))
        row = dict(uc_a=a, uc_b=b, nome_a=names[a], nome_b=names[b], year=label, existe=l is not None,
                   direta=bool(l and l["direct"]), custo=l["cost"] if l else np.nan,
                   comprimento_km=l["length_km"] if l else np.nan, cruza=",".join(map(str, l["crosses"])) if l else "",
                   contigua=bool(l and l["length_km"] <= cc.get("contiguous_max_km", 0.2)))
        if ref is None and l is not None:
            cm = C.corridor_mask(cost, (masks[a], masks[b]), l["path"], res, cc["corridor_tolerance"], margin_px, cat.shape)
            if cm is not None:
                idx = np.flatnonzero(cm.ravel()); corridors[(a, b)] = idx
                row["corredor_ha"] = idx.size * res ** 2 / 1e4
                row["corredor_natural_ha"] = float(nat[idx].sum() * res ** 2 / 1e4)
        if ref is not None:
            idx = ref["corridors"][(a, b)]
            was_nat = ref["nat_ref"][idx]; now_nat = nat[idx]
            row["corredor_natural_ref_ha"] = float(was_nat.sum() * res ** 2 / 1e4)
            row["perda_natural_ha"] = float((was_nat & ~now_nat).sum() * res ** 2 / 1e4)
            row["ruptura_pct"] = 100 * row["perda_natural_ha"] / row["corredor_natural_ref_ha"] if row["corredor_natural_ref_ha"] else np.nan
            row["perda_floresta_ha"] = float(((cat_ref.ravel()[idx] == 1) & (cat.ravel()[idx] != 1)).sum() * res ** 2 / 1e4)
            row["custo_ref"] = ref["cost"].get((a, b), np.nan)
            row["delta_custo_pct"] = 100 * (row["custo"] / row["custo_ref"] - 1) if l and row["custo_ref"] else np.nan
        rows.append(row)
        if l is not None:
            geoms.append(path_geometry(l["path"], grid))
    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(df.assign(geometry=geoms[:len(df)] if len(geoms) == len(df) else None), crs=grid.crs) if len(geoms) == len(df) else None
    n_direct_all = sum(l["direct"] for l in links)
    return df, gdf, corridors, dict(n_nodes=len(masks), n_pairs=len(links), n_direct=n_direct_all,
                                    cost={k: l["cost"] for k, l in by_pair.items()}, forest_ha=area)

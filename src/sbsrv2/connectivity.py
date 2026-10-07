"""Rede de caminhos de menor custo entre os remanescentes florestais das UCs."""
import numpy as np
import geopandas as gpd
from rasterio.features import rasterize
from skimage.graph import MCP_Geometric


def resistance_surface(cat, cfg):
    c = cfg["connectivity"]; s = np.full(cat.shape, float(c["nodata_resistance"]), np.float64)
    for k, v in c["resistance"].items():
        s[cat == int(k)] = v
    return s


def uc_raster(ucs: gpd.GeoDataFrame, grid):
    """uc_id por pixel; sobreposição resolvida por prioridade: proteção integral por último, e UCs menores sobre as maiores."""
    g = ucs.sort_values(["protecao_integral", "area_ha"], ascending=[True, False])
    return rasterize(((geom, int(i)) for geom, i in zip(g.geometry, g.uc_id)), out_shape=grid.shape,
                     transform=grid.transform, fill=0, dtype="uint8", all_touched=False)


def node_masks(cat, ucid, ucs, min_ha, res):
    """{uc_id: máscara de floresta (categoria 1) dentro da UC}, só UCs com floresta >= min_ha."""
    out, area = {}, {}
    for i in ucs.uc_id:
        m = (ucid == int(i)) & (cat == 1)
        a = m.sum() * res ** 2 / 1e4; area[int(i)] = float(a)
        if a >= min_ha:
            out[int(i)] = m
    return out, area


def cost_from(cost, mask):
    """Custo acumulado de menor custo a partir de `mask` (todos os pixels do nó são origem). Retorna (mcp, custo)."""
    starts = np.argwhere(mask)
    m = MCP_Geometric(cost, fully_connected=True)
    cum, _ = m.find_costs(starts)
    return m, cum


def pairwise(cost, masks, res, log=print):
    """Custo mínimo e caminho entre todos os pares de nós. `direct`=True se o caminho não atravessa o nó de uma 3ª UC."""
    ids = sorted(masks); links = []
    owner = np.zeros(cost.shape, np.int32)
    for i in ids:
        owner[masks[i]] = i
    for a_pos, a in enumerate(ids):
        m, cum = cost_from(cost, masks[a])
        for b in ids[a_pos + 1:]:
            cb = np.where(masks[b], cum, np.inf)
            end = np.unravel_index(np.argmin(cb), cb.shape); c = float(cb[end])
            if not np.isfinite(c):
                continue
            path = np.array(m.traceback(end))
            own = owner[path[:, 0], path[:, 1]]
            third = set(np.unique(own[(own != 0) & (own != a) & (own != b)]).tolist())
            links.append(dict(a=a, b=b, cost=c, length_km=len(path) * res / 1000, direct=len(third) == 0,
                              crosses=sorted(third), path=path))
        del m, cum
        log(f"    nó {a}: ok")
    return links


def corridor_mask(cost, masks_ab, path, res, tol, margin_px, shape):
    """Corredor do par: pixels cujo custo a->x + x->b <= (1+tol)*mínimo, numa janela em torno do caminho."""
    r0 = max(path[:, 0].min() - margin_px, 0); r1 = min(path[:, 0].max() + margin_px + 1, shape[0])
    c0 = max(path[:, 1].min() - margin_px, 0); c1 = min(path[:, 1].max() + margin_px + 1, shape[1])
    sub = cost[r0:r1, c0:c1]; ma, mb = (m[r0:r1, c0:c1] for m in masks_ab)
    if not ma.any() or not mb.any():
        return None
    _, ca = cost_from(sub, ma); _, cb = cost_from(sub, mb)
    tot = ca + cb; best = tot.min()
    out = np.zeros(shape, bool); out[r0:r1, c0:c1] = tot <= best * (1 + tol)
    return out

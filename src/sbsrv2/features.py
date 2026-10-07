"""Preditores por pixel: estáticos, dinâmicos em t0, fração de cada categoria na vizinhança e one-hot de t0."""
import numpy as np
import rasterio
from scipy import ndimage
from .config import p
from .distance import distance_to_mask, signed_distance

STATIC = ["mde", "declividade", "dist_hidrografia"]
NCLS = 6
NBR = [f"nbr_{k}" for k in range(1, NCLS + 1)]
OH = [f"oh_{k}" for k in range(1, NCLS + 1)]


def names_for(cfg, scenario: str) -> list[str]:
    dyn = list(cfg["dynamic_vars"]) if scenario == "C2" else []
    return STATIC + dyn + NBR + OH


def dyn_path(cfg, var):
    d = p(cfg, cfg["paths"]["processed"])
    v2 = d / f"dyn_{var}_v2.tif"          # v2 = LST com máscara de faixa; demais variáveis só têm v1
    return v2 if v2.exists() else d / f"dyn_{var}_v1.tif"


def read_band(path, band: int) -> np.ndarray:
    with rasterio.open(path) as s:
        return s.read(band)


def neighborhood(cat: np.ndarray, k: int, size: int) -> np.ndarray:
    return ndimage.uniform_filter((cat == k).astype(np.float32), size=size, mode="nearest")


def dist_urban(cat, res, cap):
    d = distance_to_mask(cat == 5, res)
    return d if d is not None else np.full(cat.shape, cap, np.float32)


def dist_forest_edge(cat, res, cap):
    d = signed_distance(cat == 1, res)
    return d if d is not None else np.full(cat.shape, cap, np.float32)


def features_at(cfg, grid, year, cat, flat, names, extra=None):
    """Matriz (n, len(names)) float32 em `flat` (índices planos da grade). nodata -> NaN.
    `extra`: dict nome->mapa 2D já calculado (p.ex. distâncias recalculadas pelo AC)."""
    extra = extra or {}
    nd = cfg["nodata_float"]; proc = p(cfg, cfg["paths"]["processed"]); y0 = cfg["annual_range"][0]
    X = np.empty((len(flat), len(names)), np.float32)
    with rasterio.open(proc / "static_v1.tif") as s:
        sdesc = list(s.descriptions)
    for j, n in enumerate(names):
        if n in extra:
            a = extra[n]
        elif n in STATIC:
            a = read_band(proc / "static_v1.tif", sdesc.index(n) + 1)
        elif n in cfg["dynamic_vars"]:
            a = read_band(dyn_path(cfg, n), year - y0 + 1)
        elif n in NBR:
            a = neighborhood(cat, int(n.split("_")[1]), cfg["neighborhood_size"])
        elif n in OH:
            a = (cat == int(n.split("_")[1])).astype(np.float32)
        else:
            raise KeyError(n)
        v = a.ravel()[flat].astype(np.float32)
        v[v == nd] = np.nan
        X[:, j] = v
        del a
    return X


def prepare(X, scaler, names):
    """Imputa NaN pela mediana do treino e padroniza com média/desvio do treino."""
    med = np.array([scaler[n]["median"] for n in names], np.float32)
    mu = np.array([scaler[n]["mean"] for n in names], np.float32)
    sd = np.array([scaler[n]["std"] for n in names], np.float32)
    Z = np.where(np.isnan(X), med, X)
    return (Z - mu) / sd

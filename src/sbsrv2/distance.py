"""Distância euclidiana exata (EDT do scipy) sobre a grade inteira, com margem."""
import numpy as np
from scipy import ndimage


def distance_to_mask(mask: np.ndarray, res: float) -> np.ndarray | None:
    """Distância (m) de cada pixel ao pixel verdadeiro mais próximo; None se a máscara é vazia."""
    if not mask.any():
        return None
    return (ndimage.distance_transform_edt(~mask) * res).astype(np.float32)


def signed_distance(mask: np.ndarray, res: float) -> np.ndarray | None:
    """+ fora da máscara, − dentro (distância à borda)."""
    if not mask.any() or mask.all():
        return None
    out = ndimage.distance_transform_edt(~mask) * res
    inn = ndimage.distance_transform_edt(mask) * res
    return np.where(mask, -inn, out).astype(np.float32)


def slope_horn_degrees(dem: np.ndarray, res: float, nodata=-9999.0) -> np.ndarray:
    """Declividade (graus), Horn 3x3. Pixels com nodata na janela -> nodata."""
    valid = dem != nodata
    z = np.where(valid, dem, np.nan).astype(np.float64)
    p = np.pad(z, 1, mode="edge")
    a, b, c = p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:]
    d, f = p[1:-1, :-2], p[1:-1, 2:]
    g, h, i = p[2:, :-2], p[2:, 1:-1], p[2:, 2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8 * res)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8 * res)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    return np.where(np.isnan(slope), nodata, slope).astype(np.float32)

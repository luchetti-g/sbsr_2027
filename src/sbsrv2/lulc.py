"""Reclassificação MapBiomas -> 6 categorias."""
import numpy as np


def build_lut(reclass: dict) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for cat, codes in reclass.items():
        for c in codes:
            lut[int(c)] = int(cat)
    return lut


def reclassify(raw: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """raw float/int (nodata negativo ou >255 -> 0)."""
    r = np.where((raw >= 0) & (raw <= 255), raw, 0).astype(np.uint8)
    return lut[r]

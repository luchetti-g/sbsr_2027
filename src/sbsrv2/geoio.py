"""Leitura/escrita de GeoTIFFs na grade canônica e reamostragem."""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from .grid import Grid


def write_raster(path, data: np.ndarray, grid: Grid, nodata, band_names=None, dtype=None):
    """Escreve (bandas, h, w) ou (h, w) em GeoTIFF DEFLATE tiled; nomes de banda nos metadados."""
    data = data[None] if data.ndim == 2 else data
    dtype = dtype or data.dtype
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", driver="GTiff", height=grid.height, width=grid.width,
                       count=data.shape[0], dtype=dtype, crs=grid.crs, transform=grid.transform,
                       nodata=nodata, compress="DEFLATE", tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER") as dst:
        dst.write(data.astype(dtype, copy=False))
        for i, n in enumerate(band_names or [], 1):
            dst.set_band_description(i, n)
    tmp.replace(path)


def resample_to_grid(src_path, grid: Grid, resampling="bilinear", src_nodata=None, dst_nodata=-9999.0):
    """Reamostra a banda 1 de um raster para a grade canônica (float32, nodata=dst_nodata)."""
    out = np.full(grid.shape, dst_nodata, dtype=np.float32)
    with rasterio.open(src_path) as s:
        reproject(rasterio.band(s, 1), out, src_transform=s.transform, src_crs=s.crs,
                  src_nodata=src_nodata if src_nodata is not None else s.nodata,
                  dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=dst_nodata,
                  resampling=getattr(Resampling, resampling))
    return out

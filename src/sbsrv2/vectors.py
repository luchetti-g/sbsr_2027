"""Leitura, reprojeção e rasterização de vetores na grade canônica."""
import geopandas as gpd
import numpy as np
import pyproj
from rasterio.features import rasterize
from .grid import Grid


def read_for_grid(path, grid: Grid, allow_network=True) -> gpd.GeoDataFrame:
    """Lê o vetor recortado à extensão da grade (bbox no CRS da fonte) e reprojeta para o CRS da grade."""
    if allow_network:
        pyproj.network.set_network_enabled(True)
    meta = gpd.read_file(path, rows=1)
    box = gpd.GeoSeries.from_xy(*np.array([[grid.bounds[0], grid.bounds[2]], [grid.bounds[1], grid.bounds[3]]]),
                                crs=grid.crs).to_crs(meta.crs).total_bounds
    pad = 0.0 if meta.crs.is_geographic is False else 0.0
    gdf = gpd.read_file(path, bbox=tuple(box))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf.to_crs(grid.crs)


def rasterize_mask(gdf: gpd.GeoDataFrame, grid: Grid, all_touched=True) -> np.ndarray:
    if len(gdf) == 0:
        return np.zeros(grid.shape, dtype=bool)
    return rasterize(((g, 1) for g in gdf.geometry), out_shape=grid.shape, transform=grid.transform,
                     fill=0, dtype="uint8", all_touched=all_touched).astype(bool)

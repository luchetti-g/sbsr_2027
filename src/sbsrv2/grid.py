"""Grade canônica: toda camada raster do projeto é escrita nela."""
from dataclasses import dataclass
import math
import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass(frozen=True)
class Grid:
    crs: str
    res: float
    xmin: float
    ymax: float
    width: int
    height: int

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.xmin, 0, -self.res, self.ymax)

    @property
    def bounds(self):
        return (self.xmin, self.ymax - self.height * self.res,
                self.xmin + self.width * self.res, self.ymax)

    @property
    def shape(self):
        return (self.height, self.width)

    def to_dict(self) -> dict:
        return dict(crs=self.crs, res=self.res, xmin=self.xmin, ymax=self.ymax,
                    width=self.width, height=self.height)


def grid_from_bounds(bounds, crs: str, res: float, margin_m: float) -> Grid:
    """Grade alinhada a múltiplos de `res`, cobrindo bounds + margem."""
    xmin = math.floor((bounds[0] - margin_m) / res) * res
    ymin = math.floor((bounds[1] - margin_m) / res) * res
    xmax = math.ceil((bounds[2] + margin_m) / res) * res
    ymax = math.ceil((bounds[3] + margin_m) / res) * res
    return Grid(crs, res, xmin, ymax, int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res)))


def check_alignment(path, grid: Grid, nodata=None) -> list[str]:
    """Lista de divergências entre um raster e a grade canônica (vazia = OK)."""
    errs = []
    with rasterio.open(path) as s:
        if s.crs is None or s.crs.to_string() != grid.crs:
            errs.append(f"crs {s.crs} != {grid.crs}")
        if (s.width, s.height) != (grid.width, grid.height):
            errs.append(f"tamanho {s.width}x{s.height} != {grid.width}x{grid.height}")
        if not np.allclose(tuple(s.transform)[:6], tuple(grid.transform)[:6], atol=1e-6):
            errs.append(f"transform {tuple(s.transform)[:6]} != {tuple(grid.transform)[:6]}")
        if nodata is not None and s.nodata != nodata:
            errs.append(f"nodata {s.nodata} != {nodata}")
    return errs

"""Download em blocos via ee.data.computePixels na grade canônica (NPY estruturado)."""
import io, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import ee
from .grid import Grid

HV = "https://earthengine-highvolume.googleapis.com"


def init_ee(cfg):
    ee.Initialize(project=cfg["gee"]["project"], opt_url=HV)


def tiles(grid: Grid, size: int, keep=None):
    """(r0, c0, h, w) de cada bloco; `keep(r0,c0,h,w)->bool` filtra blocos."""
    for r0 in range(0, grid.height, size):
        for c0 in range(0, grid.width, size):
            h, w = min(size, grid.height - r0), min(size, grid.width - c0)
            if keep is None or keep(r0, c0, h, w):
                yield r0, c0, h, w


def _fetch(image, grid: Grid, r0, c0, h, w, bands, retries):
    req = {"expression": image, "fileFormat": "NPY", "bandIds": bands,
           "grid": {"dimensions": {"width": w, "height": h},
                    "affineTransform": {"scaleX": grid.res, "shearX": 0, "translateX": grid.xmin + c0 * grid.res,
                                        "shearY": 0, "scaleY": -grid.res, "translateY": grid.ymax - r0 * grid.res},
                    "crsCode": grid.crs}}
    for k in range(retries):
        try:
            arr = np.load(io.BytesIO(ee.data.computePixels(req)))
            return np.stack([arr[b].astype(np.float32) for b in bands])
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise RuntimeError(f"tile ({r0},{c0}) falhou: {e}") from e
            # 429 (limite de concorrência do modo restrito): recuo longo; demais erros: recuo curto
            time.sleep(min(15 * (k + 1), 120) if "Too Many Requests" in str(e) else 2 ** k)


def download_image(image: ee.Image, grid: Grid, bands: list[str], cfg, keep=None, fill=-9999.0, log=print):
    """Baixa `image` inteira para um array (n_bandas, h, w) float32; blocos fora de `keep` ficam em `fill`."""
    d = cfg["download"]
    out = np.full((len(bands), grid.height, grid.width), fill, dtype=np.float32)
    specs = list(tiles(grid, d["tile"], keep))
    with ThreadPoolExecutor(d["workers"]) as ex:
        futs = {ex.submit(_fetch, image, grid, *s, bands, d["retries"]): s for s in specs}
        for n, f in enumerate(as_completed(futs), 1):
            r0, c0, h, w = futs[f]
            out[:, r0:r0 + h, c0:c0 + w] = f.result()
    return out

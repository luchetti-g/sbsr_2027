"""Compostos sazonais Landsat C2-L2 (NDVI, NDWI de Gao, LST em °C) com preenchimento por janelas crescentes."""
import ee

OUT_BANDS = ["ndvi", "ndwi", "lst"]


def prep_image(img: ee.Image, bands: dict, ls: dict) -> ee.Image:
    """Aplica máscara QA_PIXEL e escala+offset ANTES de calcular os índices (o offset não se cancela)."""
    qa_mask = 0
    for b in ls["qa_bits"]:
        qa_mask |= 1 << b
    clear = img.select("QA_PIXEL").bitwiseAnd(qa_mask).eq(0)
    sr = img.select([bands["red"], bands["nir"], bands["swir1"]], ["red", "nir", "swir1"]) \
        .multiply(ls["sr_scale"]).add(ls["sr_offset"])
    valid_sr = sr.gte(0).And(sr.lte(1)).reduce(ee.Reducer.min())          # reflectância fora de [0,1] é inválida
    ndvi = sr.select("nir").subtract(sr.select("red")).divide(sr.select("nir").add(sr.select("red"))).rename("ndvi")
    ndwi = sr.select("nir").subtract(sr.select("swir1")).divide(sr.select("nir").add(sr.select("swir1"))).rename("ndwi")
    st_dn = img.select(bands["tir"])
    lst = st_dn.multiply(ls["st_scale"]).add(ls["st_offset"]).subtract(273.15).rename("lst")   # K -> °C
    lst = lst.updateMask(st_dn.gt(0))
    return ee.Image.cat([ndvi.updateMask(valid_sr), ndwi.updateMask(valid_sr), lst]).updateMask(clear)


def collection(cfg: dict, region: ee.Geometry, y0: int, y1: int, months: list) -> ee.ImageCollection:
    ls = cfg["landsat"]
    cols = []
    for asset, bands in ls["sensors"].items():
        c = (ee.ImageCollection(asset).filterBounds(region)
             .filterDate(f"{y0}-01-01", f"{y1 + 1}-01-01")
             .filter(ee.Filter.calendarRange(months[0], months[1], "month"))
             .map(lambda im, b=bands: prep_image(im, b, ls)))
        cols.append(c)
    out = cols[0]
    for c in cols[1:]:
        out = out.merge(c)
    return out


def seasonal_composite(cfg: dict, region: ee.Geometry, year: int, season: str, last_year: int):
    """(composto[ndvi,ndwi,lst], banda de nível de preenchimento 0/1/2/255) para a estação e o ano."""
    months = cfg["landsat"]["seasons"][season]
    comps, masks = [], []
    for before, after in cfg["landsat"]["gapfill_windows"]:
        y0, y1 = year + before, min(year + after, last_year)
        c = collection(cfg, region, y0, y1, months).mean().select(OUT_BANDS)
        comps.append(c)
        masks.append(c.select("ndvi").mask())
    comp = comps[0]
    for c in comps[1:]:
        comp = comp.unmask(c)       # onde o nível anterior é inválido, usa o próximo
    # nível de preenchimento: o menor nível com observação válida vence (0 = melhor; 255 = sem dado)
    lvl_min = ee.Image(255)
    for k in reversed(range(len(masks))):
        lvl_min = lvl_min.where(masks[k], k)
    return comp, lvl_min.rename(f"gaplvl_{season}")

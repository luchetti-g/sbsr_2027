"""LULC, fogo e precipitação do MapBiomas na grade canônica."""
import ee


def lulc_raw(cfg, year):
    return ee.Image(cfg["gee"]["lulc_asset"]).select(f"classification_{year}").unmask(0).rename("lulc")


def burned_window(cfg, year, first_year=1985):
    """1 se queimou em qualquer ano de [t-n+1, t]; 0 caso contrário."""
    n = cfg["fire_window_years"]
    img = ee.Image(cfg["gee"]["fire_asset"])
    yrs = range(max(first_year, year - n + 1), year + 1)
    bands = [img.select(f"burned_area_{y}").unmask(0).gt(0).rename("b") for y in yrs]
    return ee.ImageCollection(bands).max().rename("burned_win")


def precipitation(cfg, year):
    y = cfg["gee"]["year_fallback"].get(year, year)
    return (ee.Image(cfg["gee"]["precip_asset"]).select(f"precipitation_{y}")
            .resample("bilinear").rename("prec"))

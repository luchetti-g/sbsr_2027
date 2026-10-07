import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbsrv2.config import load_config, p, parse_reclass_file  # noqa: E402,F401
from sbsrv2.grid import Grid  # noqa: E402,F401
import json  # noqa: E402


def load_grid(cfg) -> Grid:
    return Grid(**json.loads((p(cfg, cfg["paths"]["aoi"]) / "grid.json").read_text()))

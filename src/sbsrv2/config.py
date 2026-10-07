"""Carga do config_V2.yaml e resolução de caminhos relativos à raiz do projeto."""
from pathlib import Path
import os
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | None = None) -> dict:
    path = path or os.environ.get("SBSR_CONFIG", str(ROOT / "config_V2.yaml"))
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_root"] = ROOT
    return cfg


def p(cfg: dict, rel: str) -> Path:
    """Caminho absoluto a partir de um caminho relativo à raiz do projeto."""
    return cfg["_root"] / rel


def parse_reclass_file(path: Path) -> dict[int, list[int]]:
    """Lê reclass_mapbiomas.txt ('1 Floresta-3, 4, 5, 6, 49') -> {1: [3,4,5,6,49]}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if "-" not in line:
            continue
        head, codes = line.rsplit("-", 1)
        out[int(head.split()[0])] = [int(c) for c in codes.replace(" ", "").split(",") if c]
    return out

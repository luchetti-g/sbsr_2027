import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from sbsrv2.grid import grid_from_bounds
from sbsrv2.distance import distance_to_mask, signed_distance, slope_horn_degrees
from sbsrv2.config import load_config, p, parse_reclass_file


def test_grid_aligned_30m():
    g = grid_from_bounds((152479.9, 7327982.8, 297867.2, 7467425.1), "EPSG:31983", 30, 10000)
    assert (g.width, g.height) == (5514, 5316)
    assert g.xmin % 30 == 0 and g.ymax % 30 == 0


def test_edt_exact():
    m = np.zeros((11, 11), bool); m[5, 5] = True
    d = distance_to_mask(m, 30)
    assert d[5, 5] == 0 and d[5, 8] == 90 and np.isclose(d[8, 9], 30 * 5)
    assert distance_to_mask(np.zeros((3, 3), bool), 30) is None


def test_signed_distance():
    m = np.zeros((9, 9), bool); m[3:6, 3:6] = True
    s = signed_distance(m, 30)
    assert s[4, 4] < 0 and s[0, 0] > 0


def test_slope_plane():
    # plano inclinado 1 m a cada 30 m em x -> atan(1/30)
    x = np.tile(np.arange(20, dtype=float), (20, 1))
    s = slope_horn_degrees(x, 1.0)
    assert np.isclose(s[10, 10], np.degrees(np.arctan(1.0)), atol=1e-4)
    assert np.isclose(slope_horn_degrees(x, 30.0)[10, 10], np.degrees(np.arctan(1 / 30)), atol=1e-4)


def test_slope_nodata():
    d = np.full((5, 5), -9999.0); s = slope_horn_degrees(d, 30.0)
    assert (s == -9999.0).all()


def test_reclass_yaml_equals_txt():
    cfg = load_config()
    txt = parse_reclass_file(p(cfg, cfg["inputs"]["reclass_file"]))
    assert {k: sorted(v) for k, v in txt.items()} == {int(k): sorted(v) for k, v in cfg["reclass"].items()}


def test_landsat_offset_matters():
    # offset -0,2 não se cancela na razão normalizada (valores do pixel validado no Jurupará)
    sf, off = 2.75e-5, -0.2
    red, nir = 8016 * sf + off, 18525 * sf + off
    assert np.isclose((nir - red) / (nir + red), 0.8761, atol=1e-3)
    assert abs((18525 * sf - 8016 * sf) / (18525 * sf + 8016 * sf) - 0.8761) > 0.4


def test_reclassify():
    from sbsrv2.lulc import build_lut, reclassify
    cfg = load_config()
    lut = build_lut({int(k): v for k, v in cfg["reclass"].items()})
    raw = np.array([3, 15, 24, 33, 0, 27, -9999, 300], dtype=np.float32)
    assert reclassify(raw, lut).tolist() == [1, 3, 5, 6, 0, 0, 0, 0]


def test_metrics_qd_ad_identity_and_kappa():
    from sbsrv2.metrics import qd_ad, overall_accuracy, kappa, confusion
    rng = np.random.default_rng(0)
    ref = rng.integers(1, 7, 5000); sim = np.where(rng.random(5000) < 0.7, ref, rng.integers(1, 7, 5000))
    cm = confusion(ref, sim)
    qd, ad = qd_ad(cm)
    assert np.isclose(qd + ad, 1 - overall_accuracy(cm))        # Pontius & Millones (2011)
    assert 0 < kappa(cm) < overall_accuracy(cm) + 1e-9
    perfect = confusion(ref, ref)
    assert kappa(perfect) == 1.0 and qd_ad(perfect) == (0.0, 0.0)


def test_kappa_known_value():
    from sbsrv2.metrics import kappa
    cm = np.array([[20, 5], [10, 15]])   # po=0,7; pe=0,5 -> kappa 0,4
    assert np.isclose(kappa(cm), 0.4)


def test_fom_toy():
    from sbsrv2.metrics import fom
    t0 = np.array([1, 1, 1, 1, 2, 2]); ref = np.array([1, 3, 3, 1, 2, 4]); sim = np.array([1, 3, 2, 3, 2, 2])
    f = fom(t0, ref, sim)   # px1 acerto; px2 cat errada; px3 falso alarme; px5 omissão
    assert (f["B_acerto"], f["C_categoria_errada"], f["D_falso_alarme"], f["A_omissao"]) == (1, 1, 1, 1)
    assert np.isclose(f["FoM"], 0.25)


def test_allocation_respects_quota_and_freezes():
    from sbsrv2.ca import allocate_origin
    c0 = np.array([1] * 10); cur = c0.copy(); moved = np.zeros(10, bool)
    prob = np.zeros((10, 6), np.float32); prob[:, 2] = np.arange(10) / 10; prob[:, 3] = np.arange(10)[::-1] / 10
    q = np.zeros((6, 6), np.int64); q[0, 2] = 2; q[0, 3] = 2
    for share in (0.5, 1.0):
        cur, moved, q = allocate_origin(prob, c0, cur, moved, q, share)
    assert (cur == 3).sum() == 2 and (cur == 4).sum() == 2 and moved.sum() == 4 and q.sum() == 0
    assert set(np.flatnonzero(cur == 3)) == {8, 9} and set(np.flatnonzero(cur == 4)) == {0, 1}


def test_markov_absorbing_and_prior_correction():
    from sbsrv2.ca import markov_matrix, correct_priors
    N = np.array([np.diag([90, 90, 90, 90, 90, 90]) + 1])
    M = markov_matrix(N, absorbing=(5, 6))
    assert np.allclose(M.sum(1), 1) and M[4, 4] == 1 and M[5, 5] == 1
    w = np.ones((6, 6)); w[0, 1] = 0.1
    p = correct_priors(np.full((2, 6), 1 / 6, np.float32), np.array([1, 1]), w)
    assert np.allclose(p.sum(1), 1) and p[0, 1] < p[0, 0]

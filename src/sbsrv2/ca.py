"""Autômato celular guiado por probabilidades do MLP + demanda por cadeia de Markov."""
import numpy as np
from . import features as F


def markov_matrix(npop_list, absorbing=(), ncls=6):
    """Matriz de transição média por passo de 5 anos (contagens populacionais agrupadas); linhas absorventes = identidade."""
    N = np.sum(npop_list, axis=0).astype(np.float64)
    M = N / np.maximum(N.sum(1, keepdims=True), 1)
    for a in absorbing:
        M[a - 1, :] = 0; M[a - 1, a - 1] = 1
    return M


def correct_priors(prob, c0, w, chunk=1_000_000):
    """Corrige o viés da amostragem estratificada: p ∝ p_amostra * w[c0, j], w = N_pop / n_amostra.
    Em blocos e float32 (evita cópias float64 de n x 6)."""
    w32 = np.asarray(w, np.float32); out = np.empty_like(prob, dtype=np.float32)
    for a in range(0, len(prob), chunk):
        q = prob[a:a + chunk] * w32[c0[a:a + chunk] - 1]
        s = q.sum(1, keepdims=True); out[a:a + chunk] = q / np.where(s > 0, s, 1)
    return out


def simulate_step(cfg, grid, model_fn, scaler, names, w, M, cat0, year0, aoi_flat, log=print, frozen=None, recompute_initial=False):
    """Simula um passo de 5 anos de `cat0` (2D uint8). `model_fn(Z)->prob (n,6)`.
    Retorna o mapa simulado (2D uint8). Preditores não ligados ao uso da terra ficam em t0; distâncias
    ao urbano/borda florestal e vizinhança são recalculadas a cada iteração (realimentação)."""
    ncls = 6; iters = cfg["ca"]["iterations"]; res = grid.res
    cap = float(np.hypot(grid.width, grid.height) * res)
    absorbing = set(cfg["ca"]["absorbing_classes"])
    flat_all = aoi_flat[cat0.ravel()[aoi_flat] > 0]                # pixels com classe válida
    c0 = cat0.ravel()[flat_all].astype(np.int64)
    mut = ~np.isin(c0, list(absorbing))                            # classes absorventes não saem do lugar
    flat = flat_all[mut]; c0m = c0[mut]
    counts = np.bincount(c0, minlength=ncls + 1)[1:]
    quotas = np.rint(counts[:, None] * M).astype(np.int64)
    np.fill_diagonal(quotas, 0)
    log(f"  demanda (pixels que mudam): {int(quotas.sum()):,} de {int(counts.sum()):,} ({100*quotas.sum()/counts.sum():.2f}%)")
    frozen = dict(frozen or {})                                    # preditores mantidos constantes (clima/espectrais na projeção)
    if recompute_initial:                                          # passos da projeção: distâncias a partir do mapa corrente
        frozen.update({"dist_urbano": F.dist_urban(cat0, res, cap), "dist_borda_floresta": F.dist_forest_edge(cat0, res, cap)})
    X = F.features_at(cfg, grid, year0, cat0, flat, names, extra=frozen)   # estado em t0
    cur = c0m.copy(); moved = np.zeros(len(cur), bool)
    cmap = cat0.copy()
    for it in range(iters):
        if it > 0:                                                  # realimentação: recalcula o que depende do uso
            cmap.ravel()[flat] = cur.astype(np.uint8)
            extra = {"dist_urbano": F.dist_urban(cmap, res, cap), "dist_borda_floresta": F.dist_forest_edge(cmap, res, cap)}
            upd = [n for n in names if n in F.NBR or n in F.OH or n in extra]
            X[:, [names.index(n) for n in upd]] = F.features_at(cfg, grid, year0, cmap, flat, upd, extra)
        prob = np.empty((len(flat), ncls), np.float32); ch = cfg["ca"]["predict_chunk"]
        for a in range(0, len(flat), ch):
            prob[a:a + ch] = model_fn(F.prepare(X[a:a + ch], scaler, names))
        prob = correct_priors(prob, c0m, w)                # priors pela classe de ORIGEM (w = N_pop/n_amostra)
        left = iters - it
        cur, moved, quotas = allocate_origin(prob, c0m, cur, moved, quotas, 1.0 / left)
        log(f"  iteração {it+1}/{iters}: pixels alocados até agora {int(moved.sum()):,}")
    out = cat0.copy(); out.ravel()[flat] = cur.astype(np.uint8)
    return out


def allocate_origin(prob, c0m, cur, moved, quotas, share, ncls=6):
    """Como `allocate`, mas candidatos são identificados pela classe de ORIGEM c0m (pixels ainda não movidos)."""
    for i in range(1, ncls + 1):
        order = [j for j in np.argsort([quotas[i - 1, j] if j != i - 1 else np.inf for j in range(ncls)]) if j != i - 1]
        for j in order:
            if quotas[i - 1, j] <= 0:
                continue
            q = min(int(np.ceil(quotas[i - 1, j] * share)), int(quotas[i - 1, j]))
            cand = np.flatnonzero((c0m == i) & ~moved)
            if cand.size == 0 or q <= 0:
                continue
            q = min(q, cand.size)
            pj = prob[cand, j]
            top = cand[np.argpartition(-pj, q - 1)[:q]] if q < cand.size else cand
            cur[top] = j + 1; moved[top] = True; quotas[i - 1, j] -= top.size
    return cur, moved, quotas

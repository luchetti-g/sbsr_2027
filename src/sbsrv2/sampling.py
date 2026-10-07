"""Blocos espaciais e amostragem estratificada por (classe_t0, classe_t1)."""
import numpy as np


def block_ids(grid, flat, block_px):
    r, c = np.divmod(flat, grid.width)
    nbc = -(-grid.width // block_px)
    return (r // block_px) * nbc + (c // block_px)


def split_blocks(blocks_present, frac, seed):
    """Atribui cada bloco a train/val/test (dicionário bloco->rótulo), reprodutível."""
    rng = np.random.default_rng(seed)
    b = rng.permutation(np.unique(blocks_present))
    n = len(b); nt, nv = int(round(n * frac["train"])), int(round(n * frac["val"]))
    lab = {}
    for i, k in enumerate(b):
        lab[int(k)] = "train" if i < nt else ("val" if i < nt + nv else "test")
    return lab


def stratified_sample(c0, c1, flat, n_per, rng, ncls=6):
    """Retorna (índices em `flat` amostrados, matriz N_pop[ncls,ncls], matriz n_amostra[ncls,ncls])."""
    key = c0.astype(np.int16) * 10 + c1.astype(np.int16)
    npop = np.zeros((ncls, ncls), np.int64); nsm = np.zeros((ncls, ncls), np.int64); sel = []
    for i in range(1, ncls + 1):
        for j in range(1, ncls + 1):
            idx = np.flatnonzero(key == i * 10 + j)
            npop[i - 1, j - 1] = idx.size
            if idx.size:
                take = idx if idx.size <= n_per else rng.choice(idx, n_per, replace=False)
                nsm[i - 1, j - 1] = take.size; sel.append(take)
    return (np.concatenate(sel) if sel else np.array([], int)), npop, nsm

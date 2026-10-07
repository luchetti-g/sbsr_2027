"""Matriz de confusão, acurácia global, Kappa, FoM, Quantity/Allocation Disagreement."""
import numpy as np


def confusion(ref, sim, ncls=6):
    """Linhas = referência (mapa real); colunas = simulado."""
    k = (ref.astype(np.int64) - 1) * ncls + (sim.astype(np.int64) - 1)
    return np.bincount(k, minlength=ncls * ncls).reshape(ncls, ncls)


def overall_accuracy(cm):
    return float(np.trace(cm) / cm.sum())


def kappa(cm):
    n = cm.sum(); po = np.trace(cm) / n
    pe = float((cm.sum(1) * cm.sum(0)).sum() / n ** 2)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def kappa_per_class(cm):
    """Kappa condicional por categoria de referência: (p_jj/p_j+ - p_+j)/(1 - p_+j), onde j+ = linha (ref)."""
    n = cm.sum(); row = cm.sum(1) / n; col = cm.sum(0) / n; d = np.diag(cm) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        return (d / row - col) / (1 - col)


def qd_ad(cm):
    """Quantity e Allocation Disagreement (Pontius & Millones, 2011). QD + AD = 1 - OA."""
    p = cm / cm.sum(); row, col = p.sum(1), p.sum(0); d = np.diag(p)
    qd = float(np.abs(row - col).sum() / 2)
    ad = float(np.minimum(row - d, col - d).sum())
    return qd, ad


def fom(t0, ref, sim):
    """Figure of Merit = B/(A+B+C+D): A=omissão, B=acerto, C=mudança em categoria errada, D=falso alarme."""
    ch_ref = ref != t0; ch_sim = sim != t0
    A = int((ch_ref & ~ch_sim).sum()); B = int((ch_ref & (sim == ref)).sum())
    C = int((ch_ref & ch_sim & (sim != ref)).sum()); D = int((~ch_ref & ch_sim).sum())
    tot = A + B + C + D
    return dict(A_omissao=A, B_acerto=B, C_categoria_errada=C, D_falso_alarme=D,
                FoM=B / tot if tot else float("nan"),
                producer_change=B / (A + B + C) if (A + B + C) else float("nan"),
                user_change=B / (B + C + D) if (B + C + D) else float("nan"))

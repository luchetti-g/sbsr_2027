"""Etapa 8: simula 2020->2025 com modelo treinado em 1990-2020 e compara com o MapBiomas real de 2025.
Entrega: matriz de confusão, acurácia global, Kappa (global e por categoria), FoM, QD/AD e linha de base de persistência."""
from _common import *
import json, os, numpy as np, pandas as pd, rasterio
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from sbsrv2 import features as F, ca, metrics as M
from sbsrv2.geoio import write_raster

cfg = load_config(); grid = load_grid(cfg); proc = p(cfg, cfg["paths"]["processed"]); rep = p(cfg, cfg["paths"]["reports"]) / "model"
y0 = cfg["annual_range"][0]; t0, t1 = cfg["validation_pair"]
meta = json.loads((proc / "dataset_meta_v1.json").read_text()); scaler = meta["scaler"]; w = np.array(meta["w_prior"])
with rasterio.open(p(cfg, cfg["paths"]["aoi"]) / "aoi_mask.tif") as s:
    aoi_flat = np.flatnonzero(s.read(1).ravel() > 0)
with rasterio.open(proc / "lulc_reclass_v1.tif") as s:
    cat0, cat1 = s.read(t0 - y0 + 1), s.read(t1 - y0 + 1)

# demanda: matriz de Markov média dos pares de TREINO (1990-2020); água e urbano absorventes
Mk = ca.markov_matrix([np.array(meta["npop"][f"{a}_{b}"]) for a, b in cfg["train_pairs"]], cfg["ca"]["absorbing_classes"])
print("Demanda Markov por passo (%):\n", pd.DataFrame(100 * Mk, index=range(1, 7), columns=range(1, 7)).round(2).to_string())

v = lambda c: c.ravel()[aoi_flat]
valid = (v(cat0) > 0) & (v(cat1) > 0)
names_cls = [cfg["reclass_names"][k] for k in range(1, 7)]
results = {}


def evaluate(label, sim):
    ref0, ref1, sm = v(cat0)[valid], v(cat1)[valid], v(sim)[valid]
    ok = sm > 0; cm = M.confusion(ref1[ok], sm[ok])
    qd, ad = M.qd_ad(cm); f = M.fom(ref0[ok], ref1[ok], sm[ok])
    r = dict(overall_accuracy=M.overall_accuracy(cm), kappa=M.kappa(cm), quantity_disagreement=qd, allocation_disagreement=ad, **f,
             pct_pixels_changed_sim=float(100 * (sm != ref0).mean()), pct_pixels_changed_ref=float(100 * (ref1 != ref0).mean()))
    pd.DataFrame(cm, index=names_cls, columns=names_cls).to_csv(rep / f"confusion_2025_{label}_v1.csv", index_label="referencia(real)\\simulado")
    pd.Series(M.kappa_per_class(cm), index=names_cls).to_csv(rep / f"kappa_por_classe_2025_{label}_v1.csv", header=["kappa_condicional"])
    return r


results["persistencia"] = evaluate("persistencia", cat0)
print(f"\n[persistência: 2025 = 2020] OA {results['persistencia']['overall_accuracy']:.4f} | Kappa {results['persistencia']['kappa']:.4f}")
import tensorflow as tf
for scn in cfg["scenarios"]:
    names = F.names_for(cfg, scn)
    model = tf.keras.models.load_model(p(cfg, "models") / f"mlp_{scn}_v1.keras")
    print(f"\n== Simulando {t0}->{t1} [{scn}]")
    sim = ca.simulate_step(cfg, grid, lambda Z: model.predict(Z, batch_size=65536, verbose=0), scaler, names, w, Mk,
                           cat0, t0, aoi_flat)
    write_raster(proc / f"lulc_sim{t1}_{scn}_v1.tif", sim, grid, nodata=0, band_names=[f"sim_{t1}"], dtype="uint8")
    results[scn] = evaluate(scn, sim)
    r = results[scn]
    print(f"[{scn}] OA {r['overall_accuracy']:.4f} | Kappa {r['kappa']:.4f} | FoM {r['FoM']:.4f} | QD {r['quantity_disagreement']:.4f} AD {r['allocation_disagreement']:.4f}")
    print(f"      mudança: simulada {r['pct_pixels_changed_sim']:.2f}% | real {r['pct_pixels_changed_ref']:.2f}% | acertos {r['B_acerto']:,} / omissões {r['A_omissao']:,} / cat.errada {r['C_categoria_errada']:,} / falso alarme {r['D_falso_alarme']:,}")
best = max(cfg["scenarios"], key=lambda s: results[s]["FoM"])
results["cenario_selecionado_por_FoM"] = best
(rep / "validation_2025_v1.json").write_text(json.dumps(results, indent=1))
print(f"\nCenário com maior FoM: {best} (C1 {results['C1']['FoM']:.4f} | C2 {results['C2']['FoM']:.4f})")

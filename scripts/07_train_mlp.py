"""Etapa 7: treina o MLP nos cenários C1 (estático) e C2 (estático + dinâmico). Métricas no teste espacial."""
from _common import *
import json, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, log_loss, f1_score
from sbsrv2 import features as F, mlp

cfg = load_config(); proc = p(cfg, cfg["paths"]["processed"]); rep = p(cfg, cfg["paths"]["reports"]) / "model"; rep.mkdir(parents=True, exist_ok=True)
meta = json.loads((proc / "dataset_meta_v1.json").read_text()); scaler = meta["scaler"]
data = pd.read_parquet(proc / "samples_v1.parquet")
(p(cfg, "models")).mkdir(exist_ok=True)
summary = {}
for scn in cfg["scenarios"]:
    names = F.names_for(cfg, scn)
    Z = {s: F.prepare(d[names].to_numpy(np.float32), scaler, names) for s, d in data.groupby("split")}
    y = {s: d["c1"].to_numpy() - 1 for s, d in data.groupby("split")}
    model = mlp.build(len(names), cfg)
    h = mlp.fit(model, Z["train"], y["train"], Z["val"], y["val"], cfg)
    pd.DataFrame(h.history).to_csv(rep / f"history_{scn}_v1.csv", index_label="epoch")
    prob = model.predict(Z["test"], batch_size=65536, verbose=0); pred = prob.argmax(1)
    acc = float((pred == y["test"]).mean()); ll = float(log_loss(y["test"], prob, labels=range(6)))
    mf1 = float(f1_score(y["test"], pred, average="macro"))
    cm = confusion_matrix(y["test"], pred, labels=range(6))
    pd.DataFrame(cm, index=range(1, 7), columns=range(1, 7)).to_csv(rep / f"confusion_test_{scn}_v1.csv")
    rpt = classification_report(y["test"], pred, labels=range(6), target_names=[cfg["reclass_names"][k] for k in range(1, 7)],
                                output_dict=True, zero_division=0)
    pd.DataFrame(rpt).T.to_csv(rep / f"classification_report_{scn}_v1.csv")
    # importância por permutação (aumento do log-loss no teste)
    rng = np.random.default_rng(cfg["model"]["seed"]); sub = rng.choice(len(Z["test"]), min(30000, len(Z["test"])), replace=False)
    Xs, ys = Z["test"][sub], y["test"][sub]; base = log_loss(ys, model.predict(Xs, batch_size=65536, verbose=0), labels=range(6))
    imp = {}
    for j, n in enumerate(names):
        Xp = Xs.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
        imp[n] = float(log_loss(ys, model.predict(Xp, batch_size=65536, verbose=0), labels=range(6)) - base)
    pd.Series(imp).sort_values(ascending=False).to_csv(rep / f"permutation_importance_{scn}_v1.csv", header=["delta_logloss"])
    model.save(p(cfg, "models") / f"mlp_{scn}_v1.keras")
    summary[scn] = dict(n_features=len(names), epochs=len(h.history["loss"]), best_val_loss=float(min(h.history["val_loss"])),
                        test_accuracy_stratified=acc, test_logloss=ll, test_macro_f1=mf1,
                        n_train=int(len(Z["train"])), n_val=int(len(Z["val"])), n_test=int(len(Z["test"])))
    print(f"\n[{scn}] {len(names)} preditores | épocas {summary[scn]['epochs']} | val_loss {summary[scn]['best_val_loss']:.4f}")
    print(f"      TESTE (amostra estratificada): acurácia {acc:.4f} | macro-F1 {mf1:.4f} | log-loss {ll:.4f}")
    print("      top-5 importância:", dict(list(pd.Series(imp).sort_values(ascending=False).round(4).items())[:5]))
(rep / "train_summary_v1.json").write_text(json.dumps(summary, indent=1))

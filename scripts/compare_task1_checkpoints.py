"""Compara os 4 checkpoints candidatos da Tarefa 1 (repositorio isolado
IC_Kalile_Experimentos_Contexto/runs/task1) rodando cada um no pipeline AO VIVO
ja existente (run_live_context), sem reimplementar nada.

Candidatos (seed0, os unicos treinados por config no repositorio isolado):
  V2_dim7_frozen  (paper original)
  V2_dim7_joint   (vencedor informal da Tarefa 1)
  V2_dim10_frozen (paper original)
  V2_dim10_joint  (vencedor informal da Tarefa 1)

Registra os 4 checkpoints como VARIANTS extras (mesmo layout runs_dir/seedN/
checkpoint.pth que repos.checkpoint_paths ja espera) e roda run_live_context
com send_window=8 (o ajuste ja adotado por analyze_live.py) em todas as 8
sessoes. Reporta top-1 e balanceada AO VIVO por checkpoint, e a diferenca
para o numero offline ja conhecido (task1_comparativo.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from live_context_eval import repos  # noqa: E402
from live_context_eval.live_replay import run_live_context  # noqa: E402

EXPCTX = Path.home() / "IC_Kalile_Experimentos_Contexto"
SEND_WINDOW = 8
CLASSES = list(repos.INTENTION_LIST.keys())

CANDIDATES = {
    "V2_dim7_frozen":  dict(context_dim=7,  runs_dir=EXPCTX / "runs/task1/V2_dim7_frozen_fromV1_lr1e-04_ep60",
                            offline_key="V2_dim7_frozen_fromV1_lr1e-04_ep60"),
    "V2_dim7_joint":   dict(context_dim=7,  runs_dir=EXPCTX / "runs/task1/V2_dim7_joint_fromV1_lr3e-04_ep60",
                            offline_key="V2_dim7_joint_fromV1_lr3e-04_ep60"),
    "V2_dim10_frozen": dict(context_dim=10, runs_dir=EXPCTX / "runs/task1/V2_dim10_frozen_fromV1_lr1e-04_ep60",
                            offline_key="V2_dim10_frozen_fromV1_lr1e-04_ep60"),
    "V2_dim10_joint":  dict(context_dim=10, runs_dir=EXPCTX / "runs/task1/V2_dim10_joint_fromV1_lr3e-04_ep60",
                            offline_key="V2_dim10_joint_fromV1_lr3e-04_ep60"),
}


def _parse_offline_table(path: Path) -> dict:
    """Le reports/task1_comparativo.md (tabela markdown) -> {config: {top1, bal}}."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| V2_dim") and not line.startswith("| V1") and not line.startswith("| V0"):
            continue
        cols = [c.strip() for c in line.strip("|").split("|")]
        name = cols[0]
        top1 = float(cols[6].split("±")[0])
        bal = float(cols[7].split("±")[0])
        out[name] = {"top1": top1, "bal": bal}
    return out


def acc_from_conf(c: np.ndarray) -> float:
    return float(np.trace(c) / c.sum()) if c.sum() else 0.0


def balanced_from_conf(c: np.ndarray) -> float:
    recalls = [c[i][i] / c[i].sum() for i in range(len(c)) if c[i].sum() > 0]
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> int:
    offline = _parse_offline_table(EXPCTX / "reports" / "task1_comparativo.md")

    # Registra os 4 candidatos em repos.VARIANTS (mesmo formato que
    # checkpoint_paths()/run_live_context ja esperam: runs_dir/seedN/checkpoint.pth).
    for name, cfg in CANDIDATES.items():
        ckpt = cfg["runs_dir"] / "seed0" / "checkpoint.pth"
        if not ckpt.exists():
            print(f"ERRO: checkpoint nao encontrado para {name}: {ckpt}")
            return 1
        repos.VARIANTS[name] = {"context_dim": cfg["context_dim"], "runs_dir": cfg["runs_dir"]}

    rows = []
    for name, cfg in CANDIDATES.items():
        conf_all = np.zeros((4, 4), dtype=np.int64)
        conf_train = np.zeros((4, 4), dtype=np.int64)
        conf_test = np.zeros((4, 4), dtype=np.int64)
        errors = []
        for sid in repos.ALL_SESSIONS:
            sd = repos.SESSIONS_ROOT / sid
            try:
                r = run_live_context(sd, name, restrict="no", send_window=SEND_WINDOW, seed=0)
            except Exception as exc:  # nao contornar: reportar e seguir para o proximo candidato
                errors.append(f"{sid}: {type(exc).__name__}: {exc}")
                continue
            c = np.array(r["confusion_matrix"], dtype=np.int64)
            conf_all += c
            if sid in repos.TRAIN_SESSIONS:
                conf_train += c
            else:
                conf_test += c
            print(f"  [{name}] {sid} live_acc={r['streaming_accuracy']:.3f} "
                 f"confirm={r['num_model_confirmations']}/{r['num_gt_actionable']} "
                 f"plan_err={r['num_plan_errors']}")

        if errors:
            print(f"[{name}] ERROS (nao contornados):")
            for e in errors:
                print("   ", e)

        off = offline.get(cfg["offline_key"])
        rows.append(dict(
            name=name,
            live_top1=acc_from_conf(conf_all), live_bal=balanced_from_conf(conf_all),
            live_top1_train=acc_from_conf(conf_train), live_bal_train=balanced_from_conf(conf_train),
            live_top1_test=acc_from_conf(conf_test), live_bal_test=balanced_from_conf(conf_test),
            offline_top1=off["top1"] if off else None, offline_bal=off["bal"] if off else None,
            errors=errors,
        ))

    rows.sort(key=lambda r: r["live_top1"], reverse=True)

    print("\n=== Tabela final (ordenada por top-1 AO VIVO, todas as 8 sessoes) ===")
    print(f"{'checkpoint':18s} {'live_top1':>10s} {'live_bal':>10s} {'offline_top1':>13s} "
         f"{'d_top1':>8s} {'offline_bal':>12s} {'d_bal':>8s}")
    for r in rows:
        d_top1 = r["live_top1"] - r["offline_top1"] if r["offline_top1"] is not None else float("nan")
        d_bal = r["live_bal"] - r["offline_bal"] if r["offline_bal"] is not None else float("nan")
        print(f"{r['name']:18s} {r['live_top1']:10.4f} {r['live_bal']:10.4f} "
             f"{r['offline_top1']:13.4f} {d_top1:+8.4f} {r['offline_bal']:12.4f} {d_bal:+8.4f}")

    print("\n=== Detalhe train/test (split original do paper) ===")
    print(f"{'checkpoint':18s} {'train_top1':>11s} {'train_bal':>10s} {'test_top1':>10s} {'test_bal':>9s}")
    for r in rows:
        print(f"{r['name']:18s} {r['live_top1_train']:11.4f} {r['live_bal_train']:10.4f} "
             f"{r['live_top1_test']:10.4f} {r['live_bal_test']:9.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

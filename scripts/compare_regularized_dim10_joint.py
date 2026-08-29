"""So V2_dim10_joint: compara o vencedor original da Tarefa 1 (60 epocas
fixas, 6 sessoes de treino) contra a versao com early stopping por val loss
(runs/regularization/baseline_wd0, 5 sessoes de treino + S08 como validacao,
mesmo lr=3e-4, weight_decay nao teve efeito mensuravel no offline -> nao
comparamos variantes de weight_decay aqui, so a mudanca que importou:
early stopping).

Roda os 2 checkpoints (seed0) no pipeline AO VIVO ja existente
(run_live_context, send_window=8), mesmo metodo de
compare_task1_checkpoints.py, para ver se o gap TEST ao vivo (que era 0.667
de balanceada no V2_dim10_joint original) melhora.
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

CANDIDATES = {
    "V2_dim10_joint_original": dict(
        context_dim=10, runs_dir=EXPCTX / "runs/task1/V2_dim10_joint_fromV1_lr3e-04_ep60"),
    "V2_dim10_joint_earlystop": dict(
        context_dim=10, runs_dir=EXPCTX / "runs/regularization/baseline_wd0"),
}


def acc_from_conf(c: np.ndarray) -> float:
    return float(np.trace(c) / c.sum()) if c.sum() else 0.0


def balanced_from_conf(c: np.ndarray) -> float:
    recalls = [c[i][i] / c[i].sum() for i in range(len(c)) if c[i].sum() > 0]
    return float(np.mean(recalls)) if recalls else 0.0


def main() -> int:
    for name, cfg in CANDIDATES.items():
        ckpt = cfg["runs_dir"] / "seed0" / "checkpoint.pth"
        if not ckpt.exists():
            print(f"ERRO: checkpoint nao encontrado para {name}: {ckpt}")
            return 1
        repos.VARIANTS[name] = {"context_dim": cfg["context_dim"], "runs_dir": cfg["runs_dir"]}

    rows = []
    for name in CANDIDATES:
        conf_all = np.zeros((4, 4), dtype=np.int64)
        conf_train = np.zeros((4, 4), dtype=np.int64)
        conf_test = np.zeros((4, 4), dtype=np.int64)
        errors = []
        for sid in repos.ALL_SESSIONS:
            sd = repos.SESSIONS_ROOT / sid
            try:
                r = run_live_context(sd, name, restrict="no", send_window=SEND_WINDOW, seed=0)
            except Exception as exc:
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
            print(f"[{name}] ERROS:")
            for e in errors:
                print("   ", e)
        rows.append(dict(
            name=name,
            live_top1=acc_from_conf(conf_all), live_bal=balanced_from_conf(conf_all),
            live_top1_train=acc_from_conf(conf_train), live_bal_train=balanced_from_conf(conf_train),
            live_top1_test=acc_from_conf(conf_test), live_bal_test=balanced_from_conf(conf_test),
        ))

    print("\n=== Comparacao (mesmo checkpoint base, so muda o regime de treino) ===")
    print(f"{'checkpoint':28s} {'live_top1':>10s} {'live_bal':>9s} {'test_top1':>10s} "
         f"{'test_bal':>9s} {'train_bal':>10s}")
    for r in rows:
        print(f"{r['name']:28s} {r['live_top1']:10.4f} {r['live_bal']:9.4f} "
             f"{r['live_top1_test']:10.4f} {r['live_bal_test']:9.4f} {r['live_bal_train']:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

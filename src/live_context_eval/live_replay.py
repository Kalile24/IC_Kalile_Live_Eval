"""Construcao AO VIVO do vetor de contexto a partir do preditor de intencao.

Roda a logica de ``run_webcam_context.py`` sobre as sessoes gravadas (lendo o
``skeleton.pkl`` frame a frame, sem camera). A diferenca central em relacao a
Parte A: aqui o contexto NAO vem do ground truth. Ele e construido ao vivo,
sendo mutado pelas intencoes CONFIRMADAS DO PROPRIO MODELO. Assim avaliamos
como a predicao ocorre quando o contexto e realimentado pelo proprio preditor.

Baseado no repositorio original (``run_webcam_context.run_live``), com dois
ajustes minimos e necessarios para as sessoes gravadas:

1. Evento ``begin_four_tubes`` (comando de voz)
   ------------------------------------------------
   Ao vivo esse evento viria da fala do operador. Nas sessoes gravadas ele ja
   esta anotado em ``plan_events.json`` -- entao o aplicamos no seu frame como
   substituto elegante do comando de voz. E a UNICA informacao externa a visao;
   todas as intencoes continuam vindo do modelo.

2. Reset de ``old_intention`` no repouso
   -------------------------------------
   A guarda ``old_intention`` do ``run_live`` impede reconfirmar a mesma
   intencao consecutivamente (para nao repetir comandos ao robo). Mas a tarefa
   real tem sequencias de intencao repetida (connectors x4, screws x8, ...),
   cada uma num alcance discreto separado por ``no_action``. Resetamos
   ``old_intention`` quando o modelo preve ``no_action`` (pessoa voltou ao
   repouso): assim cada alcance discreto pode gerar uma confirmacao, como o
   PlanGraph precisa. E a correcao minima que faz a construcao ao vivo
   funcionar -- sem ela o contexto trava apos a 1a confirmacao.

Pre-processamento de pose
-------------------------
Usamos a pose CRUA do ``skeleton.pkl`` (identico ao treino/Parte A), porque os
checkpoints V1/V2 foram treinados sobre esses valores crus (``build_json.py``).
A normalizacao do ``run_live`` (``2*(x-min)/(max-min)`` + ``camera_to_world``)
era para o pipeline original (IC), com escala de coordenadas diferente, e
degrada estes checkpoints -- por isso nao a aplicamos aqui.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import repos  # registra sys.path
from plan_sim import PlanGraph  # type: ignore  # mesma copia do run ao vivo
from predict import IntentionPredictor  # type: ignore

from .context_builder import EVENT_BEGIN_FOUR_TUBES, apply_begin_four_tubes, build_context_timeline
from .evaluation import build_model, load_checkpoint_into
from .repos import INTENTION_LIST, PLAN_POLICY
from .windows import load_session_labels

NAME_BY_IDX = {v: k for k, v in INTENTION_LIST.items()}
ACTIONABLE = ("get_connectors", "get_screws", "get_wheels")
SEQ_LEN = repos.MODEL_WINDOW
CHANNELS = repos.MODEL_CHANNELS


def run_live_context(
    session_dir: Path,
    variant: str,
    *,
    restrict: str = "no",
    send_window: int = 3,
    seed: int = 0,
    teacher_forced: bool = False,
    apply_voice_event: bool = True,
    skip_ignore: bool = True,
) -> Dict[str, Any]:
    """Roda a construcao ao vivo do contexto numa sessao.

    Args:
        restrict: modo do preditor (``no`` = argmax com contexto; ``ood`` = com
            corte por entropia). Faithful ao ``run_webcam_context`` (que usa
            ``ood``), mas o padrao aqui e ``no`` porque o corte por entropia
            (calibrado no dataset do artigo) suprime justamente as predicoes de
            acao que queremos observar (ver relatorio).
        teacher_forced: se True, o contexto de cada frame e imposto do ground
            truth (baseline A/B, no MESMO pipeline streaming), para medir quanto
            da variacao vem da construcao ao vivo vs. do proprio streaming.
        skip_ignore: se True (padrao), os frames rotulados ``ignore`` NAO geram
            predicao nem confirmacao, e a janela de pose e zerada nas bordas de
            ignore (nenhuma janela cruza um bloco ignore). Os blocos ignore sao
            momentos que nao sao de predicao de intencao (setup inicial/final e,
            sobretudo, a entrega manual dos 4 tubos curtos do estagio
            four_tubes, cujo estado ja e atualizado pelo evento externo de voz);
            deixar a visao confirmar ali gera confirmacoes espurias que corrompem
            o contexto. O evento ``begin_four_tubes`` continua sendo aplicado.

    Returns:
        dict com a trilha por frame (para o video) e as metricas.
    """
    context_dim = repos.VARIANTS[variant]["context_dim"]
    skeleton, labels, plan_events = load_session_labels(session_dir)

    # Contexto de referencia (oraculo) por frame, para comparacao.
    gt_timeline = build_context_timeline(labels, context_dim, plan_events)
    gt_context_by_frame = [entry["context"] for entry in gt_timeline]
    gt_actionable = [
        label
        for i, label in enumerate(labels)
        if (i == len(labels) - 1 or labels[i + 1] != label) and label in ACTIONABLE
    ]

    ckpt = repos.VARIANTS[variant]["runs_dir"] / f"seed{seed}" / "checkpoint.pth"
    model = build_model(context_dim)
    load_checkpoint_into(model, ckpt)
    model.eval()
    predictor = IntentionPredictor(model=model, context_dim=context_dim)

    plan = PlanGraph(policy=PLAN_POLICY)
    cached_context = None if context_dim == 0 else plan.to_context_vector(context_dim)
    event_frames = {e["frame_idx"]: e["event"] for e in plan_events.get("events", [])}

    traj_queue: List[np.ndarray] = []
    intention_queue: List[str] = []
    old_intention: Optional[str] = None
    confirmations: List[Dict[str, Any]] = []
    plan_errors: List[Dict[str, Any]] = []
    trace: List[Dict[str, Any]] = []  # um registro por frame, para o video

    confusion = np.zeros((4, 4), dtype=np.int64)

    for frame_idx in range(len(skeleton)):
        # (1) evento externo begin_four_tubes (substituto do comando de voz).
        # apply_voice_event=False = ablacao "sem comando de voz" (visao pura).
        event = event_frames.get(frame_idx)
        if event is not None and not teacher_forced and apply_voice_event:
            if event != EVENT_BEGIN_FOUR_TUBES:
                raise ValueError(f"evento desconhecido: {event!r}")
            try:
                apply_begin_four_tubes(plan)
                cached_context = plan.to_context_vector(context_dim) if context_dim else None
            except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                plan_errors.append({"frame_idx": frame_idx, "kind": "begin_four_tubes", "msg": str(exc)})

        # Modo teacher-forced: contexto imposto do ground truth.
        if teacher_forced and context_dim:
            cached_context = list(gt_context_by_frame[frame_idx])

        # Contexto e estado vigentes NESTE frame (o que o modelo usa/mostra).
        frame_snapshot = plan.snapshot()
        frame_context = [] if context_dim == 0 else list(cached_context)

        # Frame 'ignore': periodo que nao e de predicao de intencao (setup e,
        # sobretudo, a entrega manual dos 4 tubos curtos). A VISAO NAO CONFIRMA
        # aqui -- deixar confirmar geraria confirmacoes espurias que corrompem o
        # contexto (o estado do four_tubes ja vem do evento externo de voz). A
        # predicao continua rodando (para o video e a continuidade da janela),
        # mas o frame nao entra em nenhuma metrica (rotulo != das 4 classes).
        is_ignore = skip_ignore and labels[frame_idx] == "ignore"
        if is_ignore:
            intention_queue = []  # quebra a sequencia de confirmacao

        traj_queue.append(skeleton[frame_idx])
        if len(traj_queue) > SEQ_LEN:
            traj_queue.pop(0)

        predicted = None
        if len(traj_queue) == SEQ_LEN:
            poses = np.asarray(traj_queue, dtype=np.float32)
            inputs = torch.tensor(poses.reshape(1, SEQ_LEN, CHANNELS), dtype=torch.float32)
            ctx_tensor = (
                None if context_dim == 0
                else torch.tensor(cached_context, dtype=torch.float32).unsqueeze(0)
            )
            _, pred = predictor.predict(inputs, restrict=restrict, context=ctx_tensor)
            predicted = NAME_BY_IDX[int(pred[0].item())]

            true = labels[frame_idx]
            if true in INTENTION_LIST:
                confusion[INTENTION_LIST[true]][INTENTION_LIST[predicted]] += 1

            # (2) maquina de confirmacao (run_live) + reset no repouso.
            # Suprimida em frames ignore (is_ignore).
            if not teacher_forced and not is_ignore:
                if predicted != "no_action":
                    if len(intention_queue) < send_window:
                        if not intention_queue or predicted == intention_queue[-1]:
                            intention_queue.append(predicted)
                        else:
                            intention_queue = [predicted]
                    else:
                        if predicted == intention_queue[-1] and predicted != old_intention:
                            old_intention = predicted
                            if context_dim:
                                try:
                                    plan.step(predicted)
                                    cached_context = plan.to_context_vector(context_dim)
                                    confirmations.append({"frame_idx": frame_idx, "intention": predicted})
                                except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                                    plan_errors.append(
                                        {"frame_idx": frame_idx, "kind": f"step:{predicted}", "msg": str(exc)}
                                    )
                            else:
                                confirmations.append({"frame_idx": frame_idx, "intention": predicted})
                        intention_queue = []
                else:
                    intention_queue = []
                    old_intention = None  # reset no repouso (melhoria minima)

        trace.append(
            {
                "frame_idx": frame_idx,
                "label": labels[frame_idx],
                "predicted": predicted,
                "context": frame_context,
                "snapshot": frame_snapshot,
                "event": event,
                "confirmed": confirmations[-1]["intention"]
                if (confirmations and confirmations[-1]["frame_idx"] == frame_idx)
                else None,
            }
        )

    # ── Metricas ─────────────────────────────────────────────────────────────
    class_names = list(INTENTION_LIST.keys())
    total = int(confusion.sum())
    correct = int(np.trace(confusion))
    per_class = {
        name: (float(confusion[i][i] / confusion[i].sum()) if confusion[i].sum() else 0.0)
        for i, name in enumerate(class_names)
    }
    model_seq = [c["intention"] for c in confirmations]
    lcs = _lcs_len(model_seq, gt_actionable)
    seq_recovery = lcs / len(gt_actionable) if gt_actionable else 1.0

    final_snapshot_matches = _snapshot_equal(plan.snapshot(), gt_timeline[-1]["snapshot"])
    if context_dim and not teacher_forced:
        final_ctx = [float(v) for v in plan.to_context_vector(context_dim)]
        final_ctx_matches = final_ctx == [float(v) for v in gt_context_by_frame[-1]]
    else:
        final_ctx_matches = None

    return {
        "session_id": Path(session_dir).name,
        "variant": variant,
        "restrict": restrict,
        "teacher_forced": teacher_forced,
        "num_frames": len(skeleton),
        "streaming_num_samples": total,
        "streaming_accuracy": (correct / total) if total else 0.0,
        "streaming_per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "num_model_confirmations": len(model_seq),
        "num_gt_actionable": len(gt_actionable),
        "confirmations": confirmations,
        "model_confirmation_sequence": model_seq,
        "gt_actionable_sequence": gt_actionable,
        "sequence_recovery": seq_recovery,
        "final_snapshot_matches_gt": final_snapshot_matches,
        "final_context_matches_gt": final_ctx_matches,
        "num_plan_errors": len(plan_errors),
        "plan_errors": plan_errors,
        "trace": trace,
    }


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Comprimento da maior subsequencia comum (mede quanto do plano foi recuperado)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def _snapshot_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = ("tube_count", "screw_count", "wheels_count", "stage", "stage_history")
    return all(a.get(k) == b.get(k) for k in keys)

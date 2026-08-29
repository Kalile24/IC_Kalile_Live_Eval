"""Reconstrucao do contexto do PlanGraph "a moda ao vivo" (deliverable central).

RESTRICAO CENTRAL do experimento
--------------------------------
O vetor de contexto que alimenta o modelo neste repositorio e construido
AQUI, pela logica deste modulo, instanciando ``PlanGraph`` diretamente e
aplicando os eventos discretos um a um. NADA aqui delega a
``build_json.build_plan_timeline`` -- essa funcao so aparece no teste de
sanidade, como ORACULO de comparacao.

Por que "a moda ao vivo"
------------------------
Em ``run_webcam_context.py`` o contexto:
  * comeca a frio (``PlanGraph`` recem-instanciado -> ``to_context_vector``);
  * so e recalculado no EVENTO DISCRETO DE CONFIRMACAO de uma intencao
    (``confirm_intention_context`` -> ``plan.step(intention)`` ->
    ``to_context_vector``), nunca a cada frame bruto;
  * o estagio manual ``four_tubes`` e aberto por um evento externo
    (``begin_four_tubes``), equivalente ao comando de voz do receptor real.

Reproduzimos exatamente essa maquina de estados percorrendo os rotulos por
frame: em cada frame o contexto vigente e o estado ATUAL do plano (antes de
confirmar a intencao cujo intervalo termina naquele frame). Quando um
intervalo anotado de intencao != ``no_action`` termina, disparamos a
confirmacao (``plan.step``), que muta os contadores e passa a valer para os
frames seguintes -- espelhando ``confirm_intention_context`` do loop ao vivo.

Isto e, propositalmente, o MESMO algoritmo de ``build_plan_timeline``; a
diferenca exigida pelo experimento e que ele esta ESCRITO AQUI (a mesma
logica que rodaria ao vivo), e nao chamado como caixa-preta.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from . import repos  # registra sys.path

# PlanGraph importado da COPIA usada pelo loop ao vivo (hrc-finetune/plan_sim.py).
# Ver repos.py para a justificativa. A logica e byte-identica a de
# datacol.plan_sim (o oraculo), o que o teste de sanidade confirma.
from plan_sim import PlanGraph  # type: ignore

from .repos import INTENTION_LIST, PLAN_POLICY

# Evento externo que abre o estagio manual four_tubes. Mesmo nome/semantica de
# annotate_pkl.PLAN_EVENT_BEGIN_FOUR_TUBES; definido localmente para que o
# walk-through seja auto-contido.
EVENT_BEGIN_FOUR_TUBES = "begin_four_tubes"


def apply_begin_four_tubes(plan: PlanGraph) -> None:
    """Aplica o evento discreto ``begin_four_tubes`` ao PlanGraph.

    Espelha a entrega dos quatro tubos curtos do receptor real: abre o estagio
    manual e confirma quatro comandos ``short`` em sequencia. E a mesma
    operacao que o loop ao vivo faria ao ouvir o comando de voz correspondente.
    """
    plan.begin_four_tubes_stage()
    for _ in range(4):
        action = plan.apply_command("short")
        if action is None:
            raise RuntimeError(
                "begin_four_tubes: PlanGraph nao entregou os quatro tubos curtos"
            )
        plan.apply_action(action[0])


def confirm_intention(plan: PlanGraph, intention: str) -> None:
    """Evento de confirmacao de intencao (identico a confirm_intention_context).

    Muta o PlanGraph exatamente uma vez, ao FIM de um intervalo anotado de
    intencao acionavel -- nunca por frame bruto.
    """
    plan.step(intention)


def build_context_timeline(
    labels: Sequence[str],
    context_dim: int,
    plan_events: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[Dict[str, Any]]:
    """Reconstroi contexto + estado do PlanGraph frame a frame, a moda ao vivo.

    Args:
        labels: rotulo por frame (inclui ``ignore``), na ordem de ``frame_idx``.
            Mesmo vetor produzido por ``annotations_to_labels``.
        context_dim: 0, 7 ou 10. Com 0, ``context`` fica ``[]``.
        plan_events: documento ``{"events": [...]}`` de ``plan_events.json``
            (ou ``None`` = sem eventos).

    Returns:
        Uma entrada por frame com ``frame_idx``, ``label``, ``context`` (o
        vetor vigente NAQUELE frame, antes de confirmar o intervalo que
        termina nele), ``snapshot`` do estado e ``event`` aplicado no frame.

    A ordem das operacoes por frame e exatamente a do loop ao vivo:
      1. aplica evento externo do frame (ex.: begin_four_tubes), se houver;
      2. captura o contexto vigente (estado ANTES de confirmar a intencao);
      3. se o intervalo de intencao acionavel termina neste frame, confirma
         (muta o plano), passando a valer para os proximos frames.
    """
    if context_dim not in (0, 7, 10):
        raise ValueError(f"context_dim deve ser 0, 7 ou 10; recebido {context_dim}")

    plan = PlanGraph(policy=PLAN_POLICY)
    events = (plan_events or {"events": []}).get("events", [])
    event_by_frame = {event["frame_idx"]: event["event"] for event in events}

    timeline: List[Dict[str, Any]] = []
    n_frames = len(labels)
    for frame_idx, label in enumerate(labels):
        # (1) evento externo discreto neste frame.
        event = event_by_frame.get(frame_idx)
        if event is not None:
            if event != EVENT_BEGIN_FOUR_TUBES:
                raise ValueError(f"evento de plano desconhecido: {event!r}")
            apply_begin_four_tubes(plan)

        # (2) contexto vigente (estado atual, antes de confirmar o intervalo).
        context: List[float] = (
            [] if context_dim == 0 else plan.to_context_vector(context_dim)
        )
        timeline.append(
            {
                "frame_idx": frame_idx,
                "label": label,
                "context": list(context),
                "snapshot": plan.snapshot(),
                "event": event,
            }
        )

        # (3) fim de intervalo de intencao acionavel -> evento de confirmacao.
        interval_ends = frame_idx == n_frames - 1 or labels[frame_idx + 1] != label
        if interval_ends and label in INTENTION_LIST and label != "no_action":
            confirm_intention(plan, label)

    return timeline


def context_per_frame(
    labels: Sequence[str],
    context_dim: int,
    plan_events: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> List[List[float]]:
    """Atalho: so os vetores de contexto por frame (sem snapshot/evento)."""
    return [
        entry["context"]
        for entry in build_context_timeline(labels, context_dim, plan_events)
    ]

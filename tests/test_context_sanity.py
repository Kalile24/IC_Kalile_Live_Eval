"""TESTE DE SANIDADE DO CONTEXTO (Parte A, item 3) -- obrigatorio.

Confronta, JANELA A JANELA e FRAME A FRAME, em TODAS as 8 sessoes, o vetor de
contexto reconstruido pela logica propria deste repositorio
(``context_builder``, a moda ao vivo) contra o vetor "correto":

  1. o contexto embutido em cada janela dos datasets pre-computados
     ``datasets/v1/dataset_dim{7,10}.json`` (gerado por build_json.py);
  2. o contexto por frame de ``build_plan_timeline`` (oraculo direto).

Se qualquer janela/frame divergir, o teste falha e imprime, para cada
divergencia, o ``frame_idx``, o vetor esperado e o vetor obtido. A correcao
esperada e ajustar a logica propria ate zerar -- nunca trocar por
build_plan_timeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from live_context_eval import repos
from live_context_eval.context_builder import context_per_frame
from live_context_eval.windows import build_windows, load_session_labels

# Oraculo direto (SO no teste, como comparacao -- nunca na construcao do contexto).
from datacol.build_json import build_plan_timeline  # type: ignore

DIMS = (7, 10)


def _load_oracle_windows(context_dim: int) -> Dict[str, Dict[int, List[float]]]:
    """Contexto por janela dos datasets pre-computados: session -> {frame_idx: ctx}."""
    data = json.loads(Path(repos.DATASETS[context_dim]).read_text(encoding="utf-8"))
    by_session: Dict[str, Dict[int, List[float]]] = {}
    for split in ("train", "test"):
        for _label, sessions in data[split].items():
            for session_id, record in sessions.items():
                bucket = by_session.setdefault(session_id, {})
                for window in record.get("windows", []):
                    bucket[window["frame_idx"]] = list(window["context"])
    return by_session


@pytest.mark.parametrize("context_dim", DIMS)
def test_windows_context_matches_dataset(context_dim: int) -> None:
    """Contexto por janela (logica propria) == contexto do dataset, nas 8 sessoes."""
    oracle = _load_oracle_windows(context_dim)
    divergences: List[Tuple[str, int, List[float], List[float]]] = []
    identical = 0
    compared = 0

    for session_id in repos.ALL_SESSIONS:
        session_dir = repos.SESSIONS_ROOT / session_id
        windows = build_windows(session_dir, context_dim)
        oracle_session = oracle.get(session_id, {})
        # As janelas mantidas tem de ser exatamente as mesmas (mesma regra).
        assert set(w["frame_idx"] for w in windows) == set(oracle_session), (
            f"{session_id}: conjunto de janelas difere do dataset"
        )
        for window in windows:
            compared += 1
            got = [float(v) for v in window["context"]]
            expected = [float(v) for v in oracle_session[window["frame_idx"]]]
            if got == expected:
                identical += 1
            else:
                divergences.append((session_id, window["frame_idx"], expected, got))

    print(
        f"\n[sanidade dim{context_dim}] janelas comparadas={compared} "
        f"identicas={identical} divergentes={len(divergences)}"
    )
    if divergences:
        for session_id, frame_idx, expected, got in divergences[:50]:
            print(f"  DIVERGE {session_id} frame_idx={frame_idx} "
                  f"esperado={expected} obtido={got}")
    assert not divergences, (
        f"dim{context_dim}: {len(divergences)} janelas divergentes "
        f"(de {compared}); ver saida acima"
    )


@pytest.mark.parametrize("context_dim", DIMS)
def test_full_timeline_matches_build_plan_timeline(context_dim: int) -> None:
    """Contexto por FRAME (todos os frames) == build_plan_timeline, nas 8 sessoes."""
    divergences: List[Tuple[str, int, List[float], List[float]]] = []
    identical = 0
    compared = 0

    for session_id in repos.ALL_SESSIONS:
        session_dir = repos.SESSIONS_ROOT / session_id
        _skeleton, labels, plan_events = load_session_labels(session_dir)

        mine = context_per_frame(labels, context_dim, plan_events)
        oracle = [
            frame_state["context"]
            for frame_state in build_plan_timeline(labels, context_dim, plan_events)
        ]
        assert len(mine) == len(oracle) == len(labels)

        for frame_idx, (got, expected) in enumerate(zip(mine, oracle)):
            compared += 1
            got_f = [float(v) for v in got]
            exp_f = [float(v) for v in expected]
            if got_f == exp_f:
                identical += 1
            else:
                divergences.append((session_id, frame_idx, exp_f, got_f))

    print(
        f"\n[sanidade frame dim{context_dim}] frames comparados={compared} "
        f"identicos={identical} divergentes={len(divergences)}"
    )
    if divergences:
        for session_id, frame_idx, expected, got in divergences[:50]:
            print(f"  DIVERGE {session_id} frame_idx={frame_idx} "
                  f"esperado={expected} obtido={got}")
    assert not divergences, (
        f"dim{context_dim}: {len(divergences)} frames divergentes "
        f"(de {compared}); ver saida acima"
    )

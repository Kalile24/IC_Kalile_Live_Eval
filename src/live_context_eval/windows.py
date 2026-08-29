"""Geracao de janelas de entrada, identica ao build_json.py (mesmo tamanho/stride).

Reproduz ``build_json.build_windows`` (janela de 5 frames, stride 1, sem cruzar
rotulos, sem intervalos ``ignore``), MAS o vetor de contexto de cada janela vem
da reconstrucao propria deste repositorio (``context_builder``), nao de
``build_plan_timeline``. Os poses sao os mesmos frames crus do ``skeleton.pkl``
(float32), exatamente como no dataset de treino -- nenhuma normalizacao, para
casar com os checkpoints (que foram treinados sobre esses valores crus).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import repos  # registra sys.path

# Loaders/validadores dos artefatos de sessao (somente leitura).
from datacol.annotate_pkl import (  # type: ignore
    annotations_to_labels,
    load_annotations,
    load_plan_events,
)

from .context_builder import context_per_frame
from .repos import INTENTION_LIST, MODEL_CHANNELS, MODEL_COORDS, MODEL_JOINTS, MODEL_WINDOW


def load_skeleton(path: Path) -> np.ndarray:
    """Carrega ``skeleton.pkl`` como (N, 15, 3) float32 (igual ao build_json)."""
    with Path(path).open("rb") as handle:
        skeleton = np.asarray(pickle.load(handle), dtype=np.float32)
    if skeleton.ndim != 3 or skeleton.shape[1:] != (MODEL_JOINTS, MODEL_COORDS):
        raise ValueError(
            f"skeleton.pkl deve ter shape (N, 15, 3); recebido {skeleton.shape}"
        )
    if len(skeleton) == 0:
        raise ValueError("skeleton.pkl vazio")
    return skeleton


def load_session_labels(session_dir: Path) -> tuple:
    """Retorna ``(skeleton, labels, plan_events)`` de uma sessao."""
    session = Path(session_dir)
    skeleton = load_skeleton(session / "skeleton.pkl")
    annotations = load_annotations(session / "annotations.json", len(skeleton))
    labels = annotations_to_labels(annotations, len(skeleton))
    plan_events = load_plan_events(session / "plan_events.json", len(skeleton), labels)
    return skeleton, labels, plan_events


def build_windows(
    session_dir: Path,
    context_dim: int,
    window_size: int = MODEL_WINDOW,
) -> List[Dict[str, Any]]:
    """Gera as janelas de uma sessao (mesma regra do build_json.build_windows).

    Cada janela contem:
        session_id, frame_idx (inicio), end_frame_idx, intention (str),
        label (id numerico), pose (np.float32 [window, 45]), context (list).
    """
    session = Path(session_dir)
    skeleton, labels, plan_events = load_session_labels(session)
    contexts = context_per_frame(labels, context_dim, plan_events)
    session_id = session.name

    windows: List[Dict[str, Any]] = []
    for start in range(0, len(skeleton) - window_size + 1):
        end = start + window_size - 1
        label = labels[start]
        if label == "ignore":
            continue
        # Janela nao pode cruzar rotulos.
        if any(candidate != label for candidate in labels[start : end + 1]):
            continue
        pose = skeleton[start : end + 1].reshape(window_size, MODEL_CHANNELS)
        windows.append(
            {
                "session_id": session_id,
                "frame_idx": start,
                "end_frame_idx": end,
                "intention": label,
                "label": INTENTION_LIST[label],
                "pose": np.ascontiguousarray(pose, dtype=np.float32),
                "context": list(contexts[start]),
            }
        )
    return windows

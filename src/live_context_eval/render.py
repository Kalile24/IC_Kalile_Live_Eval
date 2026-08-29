"""Parte B -- renderizacao estendida (video + contexto + PREDICAO).

As funcoes de renderizacao abaixo sao COPIADAS (nao importadas) de
``context_replay.py`` e ``annotate_pkl.py`` do hrc-data-collection, conforme
pedido -- para que este repositorio seja independente. Sobre a base copiada
adicionamos:

  * a intencao PREDITA (usando o contexto reconstruido pela logica propria)
    lado a lado com o rotulo anotado, com cores diferentes quando concordam
    (verde) ou divergem (vermelho);
  * a entropia/confianca da predicao, reaproveitando a logica de
    ``compute_diag`` de ``run_webcam_context.py`` (aqui numa variante que passa
    o contexto ao forward, ``compute_diag_ctx``).

O contexto por frame vem de ``context_builder`` (a moda ao vivo). A predicao
exibida usa o argmax COM contexto (a predicao que de fato usa o vetor
reconstruido), com V2_dim10.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn.functional import softmax

from . import repos  # registra sys.path
from .repos import INTENTION_LIST

Rect = Tuple[int, int, int, int]

# ── Constantes copiadas de annotate_pkl.py ───────────────────────────────────
TORSO_CONNECTIONS = ((0, 1), (1, 13), (13, 12), (12, 0))
ARM_CONNECTIONS = ((0, 2), (2, 4), (1, 3), (3, 5))
LABEL_COLORS = {
    "no_action": (160, 160, 160),
    "get_connectors": (0, 200, 255),
    "get_screws": (255, 120, 0),
    "get_wheels": (180, 0, 255),
    "ignore": (0, 0, 255),
}
PANEL_BACKGROUND = (24, 24, 28)
PANEL_BORDER = (70, 70, 78)
TEXT_PRIMARY = (240, 240, 240)
TEXT_SECONDARY = (170, 170, 180)

# ── Constantes copiadas de context_replay.py ─────────────────────────────────
CONTEXT_NAMES = {
    7: (
        "stage:none", "stage:bottom", "stage:four_tubes", "stage:top",
        "connectors / 8", "screws / 12", "wheels / 4",
    ),
    10: (
        "stage:none", "stage:bottom", "stage:four_tubes", "stage:top",
        "short tubes / 8", "long tubes / 4", "screws bottom / 4",
        "screws four_tubes / 4", "screws top / 4", "wheels / 4",
    ),
}
STAGE_COLORS = {
    "bottom": (80, 180, 255),
    "four_tubes": (90, 220, 130),
    "top": (220, 150, 80),
}
BUTTON_BACKGROUND = (52, 52, 60)
BUTTON_ACTIVE = (55, 135, 75)

CLASS_NAMES = [k for k, _ in sorted(INTENTION_LIST.items(), key=lambda x: x[1])]


# ── _VideoReplay copiado de annotate_pkl.py ──────────────────────────────────
class _VideoReplay:
    """Leitura aleatoria com cache de um quadro, sem carregar o video em RAM."""

    def __init__(self, path: Path) -> None:
        self._capture = cv2.VideoCapture(str(path))
        if not self._capture.isOpened():
            raise RuntimeError(f"could not open video: {path}")
        self.frame_count = int(round(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if self.frame_count <= 0:
            self.close()
            raise RuntimeError(f"video contains no frames: {path}")
        if self.fps <= 0:
            self.fps = 30.0
        self._cached_index: Optional[int] = None
        self._cached_frame: Optional[np.ndarray] = None

    def read(self, frame_idx: int) -> np.ndarray:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(f"video frame index out of range: {frame_idx}")
        if self._cached_index == frame_idx and self._cached_frame is not None:
            return self._cached_frame.copy()
        if self._cached_index is None or frame_idx != self._cached_index + 1:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read video frame {frame_idx}")
        self._cached_index = frame_idx
        self._cached_frame = frame
        return frame.copy()

    def close(self) -> None:
        self._capture.release()


# ── draw_skeleton copiado de annotate_pkl.py ─────────────────────────────────
def draw_skeleton(frame: np.ndarray, joints15: np.ndarray) -> np.ndarray:
    display = frame.copy()
    joints = np.asarray(joints15, dtype=np.float32)
    if joints.shape != (15, 3) or not np.isfinite(joints).all():
        return display
    if np.allclose(joints, 0):
        return display

    height, width = display.shape[:2]
    scale = float(np.clip(height / 720.0, 0.8, 1.8))
    torso_thickness = max(2, int(round(4 * scale)))
    limb_thickness = max(2, int(round(3 * scale)))
    joint_radius = max(4, int(round(6 * scale)))
    points: List[Optional[Tuple[int, int]]] = []
    for x_value, y_value, _z in joints:
        if 0.0 <= x_value <= 1.0 and 0.0 <= y_value <= 1.0:
            points.append(
                (int(round(float(x_value) * (width - 1))),
                 int(round(float(y_value) * (height - 1))))
            )
        else:
            points.append(None)

    def draw_connections(connections, color, thickness):
        for first, second in connections:
            if points[first] is not None and points[second] is not None:
                cv2.line(display, points[first], points[second], color, thickness, cv2.LINE_AA)

    draw_connections(TORSO_CONNECTIONS, (255, 80, 0), torso_thickness)
    draw_connections(ARM_CONNECTIONS, (0, 0, 255), limb_thickness)
    for joint_index in (0, 1, 2, 3, 4, 5, 12, 13):
        point = points[joint_index]
        if point is not None:
            cv2.circle(display, point, joint_radius + 2, (0, 70, 0), 1, cv2.LINE_AA)
            cv2.circle(display, point, joint_radius, (0, 220, 0), -1, cv2.LINE_AA)
    return display


# ── Funcoes de painel copiadas de context_replay.py ──────────────────────────
def _put_text(canvas, text, point, *, color=TEXT_PRIMARY, scale=0.48, thickness=1):
    cv2.putText(canvas, text, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def _draw_context_bars(canvas: np.ndarray, rect: Rect, context: Sequence[float]) -> None:
    left, top, right, bottom = rect
    names = CONTEXT_NAMES[len(context)]
    row_height = max(18, (bottom - top) // len(context))
    label_width = 150
    value_width = 42
    bar_left = left + label_width
    bar_right = right - value_width
    for index, (name, value) in enumerate(zip(names, context)):
        y_value = top + index * row_height
        _put_text(canvas, name, (left, y_value + 14), color=TEXT_SECONDARY, scale=0.38)
        cv2.rectangle(canvas, (bar_left, y_value + 3), (bar_right, y_value + 15), (55, 55, 62), -1)
        fill_right = bar_left + int(np.clip(value, 0.0, 1.0) * (bar_right - bar_left))
        cv2.rectangle(canvas, (bar_left, y_value + 3), (fill_right, y_value + 15), (80, 210, 120), -1)
        _put_text(canvas, f"{value:.3f}", (bar_right + 6, y_value + 14), scale=0.36)


def _draw_button(canvas: np.ndarray, rect: Rect, text: str, *, active: bool = False) -> None:
    left, top, right, bottom = rect
    background = BUTTON_ACTIVE if active else BUTTON_BACKGROUND
    cv2.rectangle(canvas, (left, top), (right, bottom), background, -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL_BORDER, 1)
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
    _put_text(canvas, text,
              (left + max(4, (right - left - text_width) // 2),
               top + (bottom - top + text_height) // 2), scale=0.38)


def draw_proxy_diagram(canvas: np.ndarray, rect: Rect, snapshot: Dict[str, Any]) -> None:
    """Desenha uma representacao esquematica do progresso da montagem."""
    left, top, right, bottom = rect
    cv2.rectangle(canvas, (left, top), (right, bottom), (34, 34, 40), -1)
    cv2.rectangle(canvas, (left, top), (right, bottom), PANEL_BORDER, 1)
    _put_text(canvas, "PROXY ASSEMBLY", (left + 12, top + 22), color=(100, 255, 100),
              scale=0.48, thickness=2)

    stage_record = snapshot["stage_record"]
    screw_count = snapshot["screw_count"]
    active_stage = snapshot["stage"]
    empty_color = (62, 62, 70)

    car_left = left + 72
    car_right = right - 34
    car_top = top + 55
    car_bottom = bottom - 48
    upper_y = car_top + 22
    lower_y = car_bottom - 34
    front_x = car_right - 26
    rear_x = car_left + 26
    post_x = (rear_x, rear_x + (front_x - rear_x) // 3,
              rear_x + 2 * (front_x - rear_x) // 3, front_x)

    def stage_color(stage: str, complete: bool) -> Tuple[int, int, int]:
        if complete:
            return STAGE_COLORS[stage]
        if active_stage == stage:
            return tuple(channel // 2 for channel in STAGE_COLORS[stage])
        return empty_color

    bottom_count = min(4, len(stage_record["bottom"]))
    top_count = min(4, len(stage_record["top"]))
    post_count = min(4, len(stage_record["four_tubes"]))
    chassis_width = car_right - car_left
    bottom_fill = car_left + int(chassis_width * bottom_count / 4)
    top_fill = car_left + int(chassis_width * top_count / 4)

    cv2.line(canvas, (car_left, lower_y), (car_right, lower_y), empty_color, 12)
    cv2.line(canvas, (car_left, lower_y), (bottom_fill, lower_y),
             stage_color("bottom", bottom_count == 4), 12, cv2.LINE_AA)
    cv2.line(canvas, (car_left, upper_y), (car_right, upper_y), empty_color, 12)
    cv2.line(canvas, (car_left, upper_y), (top_fill, upper_y),
             stage_color("top", top_count == 4), 12, cv2.LINE_AA)

    for index, x_value in enumerate(post_x):
        color = stage_color("four_tubes", index < post_count)
        cv2.line(canvas, (x_value, upper_y + 7), (x_value, lower_y - 7), color, 9, cv2.LINE_AA)

    screw_positions = {
        "top": [(x_value, upper_y) for x_value in post_x],
        "four_tubes": [(x_value, (upper_y + lower_y) // 2) for x_value in post_x],
        "bottom": [(x_value, lower_y) for x_value in post_x],
    }
    for stage, positions in screw_positions.items():
        completed = min(4, screw_count[stage])
        for index, point in enumerate(positions):
            fill = (245, 245, 245) if index < completed else (82, 82, 90)
            cv2.circle(canvas, point, 6, fill, -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 6, STAGE_COLORS[stage], 1, cv2.LINE_AA)

    wheel_count = min(4, snapshot["wheels_count"])
    wheel_positions = (
        (rear_x, lower_y + 31, 18), (front_x, lower_y + 31, 18),
        (rear_x + 35, lower_y + 20, 12), (front_x - 35, lower_y + 20, 12),
    )
    for index, (x_value, y_value, radius) in enumerate(wheel_positions):
        fill = LABEL_COLORS["get_wheels"] if index < wheel_count else empty_color
        cv2.circle(canvas, (x_value, y_value), radius, fill, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x_value, y_value), radius, (210, 210, 215), 2)
        cv2.circle(canvas, (x_value, y_value), max(3, radius // 3), (28, 28, 32), -1)

    labels = (
        ("TOP", top_count, screw_count["top"], upper_y),
        ("FOUR TUBES", post_count, screw_count["four_tubes"], (upper_y + lower_y) // 2),
        ("BOTTOM", bottom_count, screw_count["bottom"], lower_y),
    )
    for name, pieces, screws, y_value in labels:
        stage_key = name.lower().replace(" ", "_")
        _put_text(canvas, name, (left + 8, y_value + 4), color=STAGE_COLORS[stage_key],
                  scale=0.31, thickness=1)
        _put_text(canvas, f"{pieces}/4  S:{min(4, screws)}/4", (right - 89, y_value - 12),
                  color=TEXT_SECONDARY, scale=0.29)
    _put_text(canvas, f"WHEELS {wheel_count}/4", (right - 103, bottom - 8),
              color=TEXT_SECONDARY, scale=0.31)


def render_context_ui(
    frame: np.ndarray,
    joints15: np.ndarray,
    *,
    frame_idx: int,
    frame_count: int,
    label: str,
    labels: Sequence[str],
    frame_state: Dict[str, Any],
    playing: bool = False,
    exporting: bool = False,
) -> Tuple[np.ndarray, Dict[str, Rect], Rect]:
    """Compoe replay, classificacao, contexto e o desenho da proxy (copiado)."""
    height, width = frame.shape[:2]
    target_height = min(height, 720)
    target_width = max(1, int(round(width * target_height / height)))
    resized = cv2.resize(frame, (target_width, target_height))
    video = draw_skeleton(resized, joints15)

    panel_width = 520
    canvas = np.full((target_height, target_width + panel_width, 3),
                     PANEL_BACKGROUND, dtype=np.uint8)
    canvas[:, :target_width] = video
    cv2.line(canvas, (target_width, 0), (target_width, target_height), PANEL_BORDER, 2)
    left = target_width + 16
    right = target_width + panel_width - 16
    snapshot = frame_state["snapshot"]
    context = frame_state["context"]
    stage = snapshot["stage"] or "none"

    _put_text(canvas, "PLAN CONTEXT AUDIT", (left, 28), color=(100, 255, 100),
              scale=0.62, thickness=2)
    _put_text(canvas, f"Frame {frame_idx + 1}/{frame_count}  {'PLAY' if playing else 'PAUSE'}",
              (left, 55), scale=0.48, thickness=2)
    _put_text(canvas, f"Action: {label}", (left, 80), color=LABEL_COLORS[label],
              scale=0.52, thickness=2)
    _put_text(canvas, f"Active stage: {stage}", (left, 103),
              color=STAGE_COLORS.get(stage, TEXT_SECONDARY), scale=0.44)
    if frame_state["event"] is not None:
        _put_text(canvas, f"Event: {frame_state['event']}", (left + 220, 103),
                  color=(0, 190, 255), scale=0.40, thickness=2)

    bars_bottom = 250 if len(context) == 7 else 292
    _draw_context_bars(canvas, (left, 116, right, bars_bottom), context)
    diagram_top = bars_bottom + 10
    controls_top = target_height - 76
    timeline_top = target_height - 42
    draw_proxy_diagram(canvas, (left, diagram_top, right, controls_top - 10), snapshot)

    buttons: Dict[str, Rect] = {}
    timeline_rect = (left, timeline_top, right, timeline_top + 16)
    timeline_width = right - left
    run_start = 0
    run_label = labels[0]
    for index in range(1, frame_count + 1):
        if index == frame_count or labels[index] != run_label:
            segment_left = left + int(run_start / frame_count * timeline_width)
            segment_right = left + int(index / frame_count * timeline_width)
            cv2.rectangle(canvas, (segment_left, timeline_top),
                          (max(segment_left + 1, segment_right), timeline_top + 16),
                          LABEL_COLORS[run_label], -1)
            if index < frame_count:
                run_start = index
                run_label = labels[index]
    progress = 0.0 if frame_count <= 1 else frame_idx / (frame_count - 1)
    progress_x = left + int(progress * timeline_width)
    cv2.line(canvas, (progress_x, timeline_top - 4), (progress_x, timeline_top + 20),
             (255, 255, 255), 2)
    return canvas, buttons, timeline_rect


# ── Novo (Parte B): diagnostico da predicao + overlay predito vs anotado ──────
def compute_diag_ctx(model, inputs: torch.Tensor, context: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Variante de ``compute_diag`` (run_webcam_context.py) que injeta contexto.

    Retorna probs (softmax), entropia de Shannon e confianca (prob max). O
    forward usa o contexto reconstruido, para que a confianca reflita a
    predicao efetivamente usada.
    """
    with torch.no_grad():
        _, logits = model(inputs, context)
    probs_t = softmax(logits, dim=1)[0].detach()
    entropy_val = float(Categorical(probs=probs_t).entropy())
    probs_np = probs_t.numpy()
    return {
        "probs": probs_np,
        "entropy": entropy_val,
        "confidence": float(probs_np.max()),
        "pred_idx": int(probs_np.argmax()),
    }


def draw_prediction_overlay(
    canvas: np.ndarray,
    video_width: int,
    *,
    annotated: str,
    predicted: str,
    diag: Optional[Dict[str, Any]],
    variant_name: str,
    confirmed: Optional[str] = None,
) -> None:
    """Sobrepoe, no canto superior esquerdo do video, anotado vs predito + diag.

    Verde quando predito == anotado; vermelho quando divergem. Mostra entropia,
    confianca e as 4 probabilidades por classe. Quando ``confirmed`` != None,
    sinaliza que uma confirmacao acaba de mutar o contexto ao vivo neste frame.
    """
    agree = predicted == annotated
    accent = (90, 220, 130) if agree else (60, 60, 235)

    pad = 10
    box_w = min(video_width - 16, 360)
    n_rows = 4 + len(CLASS_NAMES) + (1 if confirmed else 0)
    box_h = 34 + n_rows * 22
    x0, y0 = 8, 8
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
    cv2.rectangle(canvas, (x0, y0), (x0 + box_w, y0 + box_h), accent, 2)

    row = y0 + 22
    _put_text(canvas, f"PREDICAO ({variant_name})", (x0 + pad, row),
              color=(100, 255, 100), scale=0.5, thickness=2)
    row += 26
    _put_text(canvas, f"anotado:  {annotated}", (x0 + pad, row),
              color=LABEL_COLORS.get(annotated, TEXT_PRIMARY), scale=0.5, thickness=2)
    row += 24
    _put_text(canvas, f"predito:  {predicted}", (x0 + pad, row),
              color=accent, scale=0.5, thickness=2)
    row += 24
    status = "CONCORDAM" if agree else "DIVERGEM"
    _put_text(canvas, status, (x0 + pad, row), color=accent, scale=0.46, thickness=2)
    row += 24
    if diag is not None:
        _put_text(canvas,
                  f"entropia: {diag['entropy']:.3f}   conf: {diag['confidence']:.2f}",
                  (x0 + pad, row), color=TEXT_SECONDARY, scale=0.44)
        row += 22
        bar_x = x0 + pad + 150
        bar_w = box_w - (bar_x - x0) - pad
        for i, name in enumerate(CLASS_NAMES):
            prob = float(diag["probs"][i])
            _put_text(canvas, f"{name}", (x0 + pad, row + 12),
                      color=LABEL_COLORS[name], scale=0.38)
            cv2.rectangle(canvas, (bar_x, row + 2), (bar_x + bar_w, row + 14), (55, 55, 62), -1)
            cv2.rectangle(canvas, (bar_x, row + 2),
                          (bar_x + int(np.clip(prob, 0, 1) * bar_w), row + 14),
                          LABEL_COLORS[name], -1)
            _put_text(canvas, f"{prob:.2f}", (bar_x + bar_w - 34, row + 12),
                      color=TEXT_PRIMARY, scale=0.36)
            row += 20
    if confirmed:
        _put_text(canvas, f"CONFIRMADO -> {confirmed}  (contexto atualizado)",
                  (x0 + pad, row + 12), color=(0, 230, 120), scale=0.44, thickness=2)

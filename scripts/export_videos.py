"""Parte D (video) -- exporta 1 MP4 por sessao com o contexto construido AO VIVO.

Para cada sessao, roda ``run_live_context`` (contexto dirigido pelas intencoes
confirmadas do modelo, begin_four_tubes vindo do evento gravado) e renderiza:

  * painel de contexto/proxy mostrando o estado construido AO VIVO (evolui a
    cada confirmacao do modelo, nao vem do ground truth);
  * intencao PREDITA vs. rotulo anotado (verde = concordam, vermelho = divergem);
  * entropia/confianca e probabilidades por classe (compute_diag com contexto);
  * aviso "CONFIRMADO" no frame em que uma confirmacao muta o contexto.

Assim da para avaliar visualmente como o contexto e construido ao vivo e como a
predicao se comporta com ele. Usa V2_dim10, seed0, pose crua, restrict=no.
Saida: outputs/<sessao>_live_V2_dim10.mp4.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from live_context_eval import repos  # noqa: E402
from live_context_eval.evaluation import build_model, load_checkpoint_into  # noqa: E402
from live_context_eval.live_replay import run_live_context  # noqa: E402
from live_context_eval.windows import load_skeleton  # noqa: E402
from live_context_eval.render import (  # noqa: E402
    _VideoReplay,
    compute_diag_ctx,
    draw_prediction_overlay,
    render_context_ui,
)

WINDOW = repos.MODEL_WINDOW
CHANNELS = repos.MODEL_CHANNELS
VARIANT = "V2_dim10"
CONTEXT_DIM = 10
RESTRICT = "no"
SEND_WINDOW = 8  # ajustado (ver analyze_live.py / relatorio, seção «Ajuste»)


def export_session(session_id: str, model, out_dir: Path, seed_tag: str) -> Path:
    session_dir = repos.SESSIONS_ROOT / session_id
    result = run_live_context(session_dir, VARIANT, restrict=RESTRICT, send_window=SEND_WINDOW)
    trace = result["trace"]
    labels = [t["label"] for t in trace]
    skeleton = load_skeleton(session_dir / "skeleton.pkl")

    replay = _VideoReplay(session_dir / "video.mp4")
    if replay.frame_count != len(skeleton):
        replay.close()
        raise ValueError(f"{session_id}: video != skeleton")

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{session_id}_live_{VARIANT}.mp4"
    writer = None
    ffmpeg_process = None
    ffmpeg_path = shutil.which("ffmpeg")
    try:
        for frame_idx in range(replay.frame_count):
            entry = trace[frame_idx]
            canvas, _b, _t = render_context_ui(
                replay.read(frame_idx),
                skeleton[frame_idx],
                frame_idx=frame_idx,
                frame_count=replay.frame_count,
                label=labels[frame_idx],
                labels=labels,
                frame_state=entry,  # snapshot/context/event construidos AO VIVO
                playing=True,
                exporting=True,
            )
            video_width = canvas.shape[1] - 520

            if entry["predicted"] is not None:
                start = frame_idx - (WINDOW - 1)
                pose = skeleton[start:frame_idx + 1].reshape(1, WINDOW, CHANNELS)
                inputs = torch.tensor(pose, dtype=torch.float32)
                ctx = torch.tensor(entry["context"], dtype=torch.float32).unsqueeze(0)
                diag = compute_diag_ctx(model, inputs, ctx)
                draw_prediction_overlay(
                    canvas, video_width, annotated=labels[frame_idx],
                    predicted=entry["predicted"], diag=diag,
                    variant_name=f"{VARIANT} {seed_tag} AO VIVO",
                    confirmed=entry["confirmed"],
                )
            else:
                draw_prediction_overlay(
                    canvas, video_width, annotated=labels[frame_idx],
                    predicted="(sem janela)", diag=None,
                    variant_name=f"{VARIANT} {seed_tag} AO VIVO",
                )

            if writer is None:
                height, width = canvas.shape[:2]
                if ffmpeg_path is not None:
                    ffmpeg_process = subprocess.Popen(
                        [ffmpeg_path, "-y", "-loglevel", "error", "-f", "rawvideo",
                         "-pix_fmt", "bgr24", "-s:v", f"{width}x{height}",
                         "-r", f"{replay.fps:.6f}", "-i", "-", "-an",
                         "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
                        stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                    )
                    writer = ffmpeg_process.stdin
                else:
                    import cv2
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"),
                                             replay.fps, (width, height))
            if ffmpeg_process is not None:
                writer.write(canvas.tobytes())
            else:
                writer.write(canvas)
    finally:
        if writer is not None:
            if ffmpeg_process is not None:
                writer.close()
                err = ffmpeg_process.stderr.read().decode("utf-8", errors="replace")
                if ffmpeg_process.wait() != 0:
                    raise RuntimeError(f"ffmpeg falhou: {err.strip()}")
            else:
                writer.release()
        replay.close()
    return output


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Exporta MP4s com contexto construido ao vivo.")
    parser.add_argument("--sessions", nargs="*", default=repos.ALL_SESSIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "outputs")
    args = parser.parse_args(argv)

    ckpt = repos.VARIANTS[VARIANT]["runs_dir"] / f"seed{args.seed}" / "checkpoint.pth"
    model = build_model(CONTEXT_DIM)
    load_checkpoint_into(model, ckpt)
    model.eval()

    for session_id in args.sessions:
        out = export_session(session_id, model, args.out_dir, f"seed{args.seed}")
        print(f"[video] {session_id} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

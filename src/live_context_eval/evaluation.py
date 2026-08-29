"""Avaliacao dos checkpoints V1/V2 -- espelho fiel de train_finetune.evaluate.

Para que os numeros das sessoes de TESTE batam com os ja publicados em
``hrc-finetune/reports/``, esta avaliacao reproduz exatamente o caminho de
``train_finetune.evaluate``:

  * restrict="no"  -> ``argmax`` cru de ``model(pose, ctx)`` (CONTEXTO injetado).
  * restrict="ood" -> ``IntentionPredictor(model=model).predict(pose, "ood",
    context=ctx)``. Detalhe fiel (e importante) do pipeline original: como o
    predictor e construido sem ``context_dim``, ``predict`` chama
    ``model(poses)`` SEM contexto no caminho do argmax; so aplica o corte por
    entropia sobre os logits sem contexto. Ou seja, no modo "ood" o vetor de
    contexto reconstruido NAO entra no argmax -- isso e uma propriedade do
    pipeline publicado, reproduzida aqui de proposito.

A confianca/ECE usa sempre ``model(pose, ctx)`` (com contexto), como no
original. Alimentamos janelas com o contexto RECONSTRUIDO por este repositorio;
como o teste de sanidade garante contexto identico ao do dataset, os numeros
de teste tem de coincidir com os publicados.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from . import repos  # registra sys.path

from DLinear import Model_FinalIntention  # type: ignore
from predict import IntentionPredictor  # type: ignore

from .repos import INTENTION_LIST, SEEDS, VARIANTS, checkpoint_paths


class ModelArgs:
    """Namespace minimo consumido por Model_FinalIntention."""

    def __init__(self, seq_len=5, pred_len=5, class_num=4, individual=False, channels=45):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.class_num = class_num
        self.individual = individual
        self.channels = channels


@dataclass
class EvalMetrics:
    accuracy: float
    per_class_accuracy: Dict[str, float]
    confusion_matrix: List[List[int]]
    ece: float
    num_samples: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "per_class_accuracy": self.per_class_accuracy,
            "confusion_matrix": self.confusion_matrix,
            "ece": self.ece,
            "num_samples": self.num_samples,
        }


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE com bins uniformes em [0, 1] (identico a train_finetune.compute_ece)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    if total == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.sum() / total) * abs(bin_conf - bin_acc)
    return float(ece)


def build_model(context_dim: int) -> Model_FinalIntention:
    return Model_FinalIntention(ModelArgs(), context_dim=context_dim)


def load_checkpoint_into(model: Model_FinalIntention, checkpoint_path: Path) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"checkpoint {checkpoint_path} tem chaves inesperadas: {unexpected}")


@torch.no_grad()
def evaluate_windows(
    model: Model_FinalIntention,
    windows: List[Dict[str, Any]],
    context_dim: int,
    restrict: str = "ood",
) -> EvalMetrics:
    """Avalia uma lista de janelas -- copia fiel de train_finetune.evaluate."""
    model.eval()
    class_names = list(INTENTION_LIST.keys())
    num_classes = len(class_names)
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    all_confidences: List[np.ndarray] = []
    all_correct: List[np.ndarray] = []
    total = 0
    correct_total = 0

    predictor = IntentionPredictor(model=model) if restrict != "no" else None

    for window in windows:
        pose = torch.tensor(window["pose"], dtype=torch.float32).unsqueeze(0)  # (1,5,45)
        label = int(window["label"])
        if context_dim > 0 and window["context"]:
            ctx = torch.tensor(window["context"], dtype=torch.float32).unsqueeze(0)
        else:
            ctx = None

        _, logits = model(pose, ctx)
        confidence = torch.softmax(logits, dim=1).max(dim=1).values
        if predictor is not None:
            _, pred = predictor.predict(pose, restrict=restrict, context=ctx)
        else:
            pred = torch.argmax(logits, dim=1)

        pred_label = int(pred[0].item())
        confusion[label][pred_label] += 1
        is_correct = float(pred_label == label)
        all_confidences.append(confidence.detach().cpu().numpy())
        all_correct.append(np.array([is_correct], dtype=np.float32))
        correct_total += int(is_correct)
        total += 1

    accuracy = correct_total / total if total > 0 else 0.0
    per_class_accuracy = {}
    for idx, name in enumerate(class_names):
        class_total = confusion[idx].sum()
        per_class_accuracy[name] = (
            float(confusion[idx][idx] / class_total) if class_total > 0 else 0.0
        )
    confidences = np.concatenate(all_confidences) if all_confidences else np.array([])
    correct_arr = np.concatenate(all_correct) if all_correct else np.array([])
    ece = compute_ece(confidences, correct_arr)

    return EvalMetrics(
        accuracy=accuracy,
        per_class_accuracy=per_class_accuracy,
        confusion_matrix=confusion.tolist(),
        ece=ece,
        num_samples=total,
    )


def evaluate_variant_over_seeds(
    windows: List[Dict[str, Any]],
    variant: str,
    restrict: str,
) -> Dict[str, Any]:
    """Avalia uma variante (3 seeds) sobre janelas ja construidas.

    Retorna metricas por seed e o agregado (media/dp de acuracia e ECE, media
    da acuracia por classe entre seeds, confusao somada). Mesma metodologia das
    tabelas publicadas.
    """
    context_dim = VARIANTS[variant]["context_dim"]
    per_seed: List[EvalMetrics] = []
    for ckpt in checkpoint_paths(variant):
        model = build_model(context_dim)
        load_checkpoint_into(model, ckpt)
        per_seed.append(evaluate_windows(model, windows, context_dim, restrict))

    accs = [m.accuracy for m in per_seed]
    eces = [m.ece for m in per_seed]
    class_names = list(INTENTION_LIST.keys())
    per_class_mean = {
        name: float(np.mean([m.per_class_accuracy[name] for m in per_seed]))
        for name in class_names
    }
    confusion_sum = np.sum([np.array(m.confusion_matrix) for m in per_seed], axis=0)

    return {
        "variant": variant,
        "context_dim": context_dim,
        "restrict": restrict,
        "num_seeds": len(per_seed),
        "num_samples": per_seed[0].num_samples if per_seed else 0,
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "ece_mean": float(np.mean(eces)),
        "ece_std": float(np.std(eces)),
        "per_class_accuracy_mean": per_class_mean,
        "confusion_matrix_sum": confusion_sum.tolist(),
        "per_seed": [m.to_dict() for m in per_seed],
    }

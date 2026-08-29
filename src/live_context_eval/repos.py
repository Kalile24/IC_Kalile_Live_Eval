"""Bootstrap de imports dos tres repositorios originais (somente leitura).

Este repositorio (hrc-live-context-eval) e novo e independente. Ele NAO
modifica nenhum arquivo de:

  - ~/IC_Kalile_Intention_Prediction_HRC  (framework de predicao original)
  - ~/hrc-data-collection                 (coleta, PlanGraph, build_json, sessoes)
  - ~/hrc-finetune                        (checkpoints V1/V2, predict/DLinear copiados)

Aqui apenas registramos os caminhos no ``sys.path`` para poder importar as
classes publicas (PlanGraph, IntentionPredictor, Model_FinalIntention,
loaders de anotacao) sem tocar nos arquivos originais.

Decisao de projeto sobre qual ``plan_sim`` usar
------------------------------------------------
Existem duas copias byte-identicas (a menos do docstring/import de
INTENTION_LIST) de ``plan_sim.py``:

  - ``hrc-data-collection/src/datacol/plan_sim.py`` -> usada por build_json.py
    para gerar o contexto do dataset (o ORACULO de comparacao).
  - ``hrc-finetune/plan_sim.py``                    -> copia local usada pelo
    loop AO VIVO (run_webcam_context.py).

A logica de decisao/estado (``decide_send_action``, contadores, one-hot de
estagio) e identica nas duas. Para tornar a afirmacao de equivalencia a mais
forte possivel, a reconstrucao "a moda ao vivo" deste repositorio importa
``PlanGraph`` da COPIA DE hrc-finetune (a mesma que roda ao vivo). O teste de
sanidade (tests/test_context_sanity.py) confronta o resultado contra o oraculo
gerado pela copia de datacol, provando que as duas produzem o mesmo contexto.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

HOME = Path.home()

IC_REPO = HOME / "IC_Kalile_Intention_Prediction_HRC"
DATACOL_REPO = HOME / "hrc-data-collection"
FINETUNE_REPO = HOME / "hrc-finetune"

# ``datacol`` e um pacote sob src/; ``predict``/``DLinear``/``plan_sim`` sao
# modulos de topo em hrc-finetune/.
_PATHS = [
    str(DATACOL_REPO / "src"),  # pacote datacol (build_json, annotate_pkl, plan_sim)
    str(FINETUNE_REPO),         # predict, DLinear, plan_sim (copia usada ao vivo)
]


class _NumpyCoreFinder:
    """Meta path finder: mapeia ``numpy._core.*`` (numpy 2.x) para ``numpy.core.*``.

    Os ``skeleton.pkl`` das sessoes foram serializados com numpy 2.x, que usa o
    caminho de modulo ``numpy._core``. O ambiente de avaliacao (mesmo do treino
    publicado, torch 2.8 / numpy 1.26) nao tem esse alias, entao o unpickle
    falharia. Este finder reescreve o import para ``numpy.core.*`` (equivalente
    em 1.26), permitindo carregar os esqueletos sem depender da versao do numpy.
    """

    _PREFIX = "numpy._core"

    def find_module(self, name, path=None):  # noqa: D401 (interface antiga do import)
        if name == self._PREFIX or name.startswith(self._PREFIX + "."):
            return self
        return None

    def load_module(self, name):
        if name in sys.modules:
            return sys.modules[name]
        alt = "numpy.core" + name[len(self._PREFIX):]
        module = __import__(alt, fromlist=["_"])
        sys.modules[name] = module
        return module


def _install_numpy_core_shim() -> None:
    try:
        import numpy
    except ImportError:
        return
    # numpy 2.x ja tem numpy._core nativo -> nada a fazer.
    if int(numpy.__version__.split(".")[0]) >= 2:
        return
    import numpy.core  # noqa: F401  (garante o alvo do alias carregado)
    # Remove qualquer entrada parcial deixada por uma tentativa de import.
    sys.modules.pop("numpy._core", None)
    if not any(isinstance(f, _NumpyCoreFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _NumpyCoreFinder())


def bootstrap() -> None:
    """Insere os caminhos necessarios no ``sys.path`` (idempotente)."""
    for path in _PATHS:
        if path not in sys.path:
            sys.path.insert(0, path)
    _install_numpy_core_shim()


bootstrap()

# ── Constantes do experimento ────────────────────────────────────────────────

INTENTION_LIST: Dict[str, int] = {
    "no_action": 0,
    "get_connectors": 1,
    "get_screws": 2,
    "get_wheels": 3,
}

# Politica do PlanGraph usada por build_json.py na geracao do dataset e por
# run_webcam_context.py ao vivo. Precisa bater para que o contexto coincida.
PLAN_POLICY = "proxy_graph"

MODEL_WINDOW = 5
MODEL_JOINTS = 15
MODEL_COORDS = 3
MODEL_CHANNELS = MODEL_JOINTS * MODEL_COORDS  # 45

SESSIONS_ROOT = DATACOL_REPO / "sessions"

# Split oficial (identico ao _meta.splits dos dataset_dim*.json).
TRAIN_SESSIONS: List[str] = [
    "S01_20260712",
    "S03_20260712",
    "S04_20260712",
    "S06_20260712",
    "S07_20260712",
    "S08_20260712",
]
TEST_SESSIONS: List[str] = ["S02_20260712", "S05_20260712"]
ALL_SESSIONS: List[str] = sorted(TRAIN_SESSIONS + TEST_SESSIONS)

# Datasets pre-computados (oraculo de contexto do item 3).
DATASETS = {
    0: DATACOL_REPO / "datasets/v1/dataset_dim0.json",
    7: DATACOL_REPO / "datasets/v1/dataset_dim7.json",
    10: DATACOL_REPO / "datasets/v1/dataset_dim10.json",
}

# Variantes avaliadas: nome -> (context_dim, diretorio dos checkpoints por seed).
# Os checkpoints em runs/ e runs_no_cutoff/ sao byte-identicos (o corte por
# entropia so muda a AVALIACAO, nao o treino), entao usamos runs/ para os dois
# modos restrict.
VARIANTS = {
    "V1": {"context_dim": 0, "runs_dir": FINETUNE_REPO / "runs/V1"},
    "V2_dim7": {"context_dim": 7, "runs_dir": FINETUNE_REPO / "runs/V2_dim7"},
    "V2_dim10": {"context_dim": 10, "runs_dir": FINETUNE_REPO / "runs/V2_dim10"},
}
SEEDS = [0, 1, 2]


def checkpoint_paths(variant: str) -> List[Path]:
    """Retorna os checkpoints (um por seed) de uma variante."""
    runs_dir = VARIANTS[variant]["runs_dir"]
    return [runs_dir / f"seed{seed}" / "checkpoint.pth" for seed in SEEDS]

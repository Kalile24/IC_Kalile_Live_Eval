"""hrc-live-context-eval: avaliacao com contexto reconstruido a moda ao vivo.

Repositorio novo e independente. Nao modifica nenhum arquivo dos tres
repositorios originais (IC_Kalile_Intention_Prediction_HRC, hrc-data-collection,
hrc-finetune); apenas le/importa deles.
"""
from . import repos  # garante o bootstrap de sys.path ao importar o pacote

__all__ = ["repos"]

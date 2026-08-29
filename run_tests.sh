#!/usr/bin/env bash
# Roda a suite de testes (inclui o teste de sanidade do contexto, item 3).
# Usa o ambiente conda `hrc` (torch 2.8 / numpy 1.26, mesmo do treino publicado).
# - `-u PYTHONPATH`: evita que o PYTHONPATH do ROS injete plugins de pytest.
# - PYTEST_DISABLE_PLUGIN_AUTOLOAD=1: desativa o autoload do launch_testing (ROS).
set -euo pipefail
cd "$(dirname "$0")"
PY="${HRC_PY:-$HOME/anaconda3/envs/hrc/bin/python}"
env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$PY" -m pytest tests/ "$@"

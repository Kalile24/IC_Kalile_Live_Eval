# hrc-live-context-eval

Avalia como o modelo de predição de intenção (V2_dim10, treinado offline no
`hrc-finetune`) se comporta **ao vivo**, quando o vetor de contexto do
`PlanGraph` é construído pelo **próprio preditor** — usando as 8 sessões
gravadas (`skeleton.pkl`, sem câmera) como banco de replay. Pergunta central:
**é escalável? Se não, o que fazer?**

Repositório novo e independente. Não modifica nenhum arquivo dos três
repositórios originais (`IC_Kalile_Intention_Prediction_HRC`,
`hrc-data-collection`, `hrc-finetune`); apenas lê/importa deles.

## Ideia

Roda a lógica de `run_webcam_context.py` sobre cada sessão gravada, frame a
frame. O contexto começa a frio e é mutado **só pelas intenções confirmadas do
modelo** (janela de confirmação `send_window=3`). Dois ajustes mínimos, baseados
no repositório original, tornam isso viável nas sessões gravadas:

1. **`begin_four_tubes` = evento gravado** (`plan_events.json`), como substituto
   elegante do comando de voz — a única informação externa à visão.
2. **Reset de `old_intention` no repouso** (`no_action`), para permitir uma
   confirmação por alcance discreto (sem isso o contexto trava após a 1ª).

Tudo é comparado contra o **contexto ideal** (teacher-forced = ground truth no
mesmo pipeline) para separar o custo de construir o contexto ao vivo do
desempenho bruto do modelo.

## Estrutura

```
src/live_context_eval/
  repos.py            Bootstrap de sys.path p/ os 3 repos + constantes + shim numpy._core
  context_builder.py  Reconstrução do contexto (usada como oráculo/baseline teacher-forced)
  windows.py          Carga de sessão (skeleton.pkl, rótulos, plan_events)
  evaluation.py       Construção/carga do modelo (checkpoints V1/V2)
  live_replay.py      run_live_context: constrói o contexto AO VIVO pelo preditor
  render.py           Render de vídeo (copiado de context_replay.py) + overlay de predição
scripts/
  analyze_live.py     ANÁLISE completa (live vs ideal, escalabilidade, ablação da voz,
                      erros de PlanGraph, confusão de confirmações) -> relatório
  export_videos.py    1 MP4 por sessão com o contexto construído AO VIVO + predição
  compare_task1_checkpoints.py       roda os 4 checkpoints candidatos da Tarefa 1 do
                      IC_Kalile_Experimentos_Contexto (frozen/joint × dim7/dim10) neste
                      pipeline ao vivo (send_window=8), comparando com o número offline
  compare_regularized_dim10_joint.py roda V2_dim10_joint original vs. a versão com
                      early stopping (runs/regularization/baseline_wd0) neste pipeline
                      ao vivo, para ver se o gap ao vivo melhora
tests/
  test_context_sanity.py  Garante que o contexto-oráculo (baseline) == build_plan_timeline
reports/
  analise_ao_vivo.md   RELATÓRIO principal (LEIA ESTE)
  metrics_ao_vivo.json Todos os números
outputs/               8 MP4s (contexto construído ao vivo)
```

## Instalação

Este repositório precisa de `hrc-data-collection` e `hrc-finetune` clonados
como diretórios **irmãos** (mesmo diretório pai) — `repos.py` monta o
`sys.path` a partir da posição relativa dos três. Cada um tem seu próprio
`requirements.txt`; instale os três antes de rodar qualquer script daqui:

```bash
pip install -r requirements.txt
pip install -r ../hrc-data-collection/requirements.txt
pip install -r ../hrc-finetune/requirements.txt
```

## Ambiente

Conda `hrc` (torch 2.8 / numpy 1.26 / cv2 4.13 — o mesmo do treino):
`~/anaconda3/envs/hrc/bin/python`. Os `skeleton.pkl` foram gravados com numpy
2.x; `repos.py` instala um shim de import para carregá-los sob numpy 1.26.
`env -u PYTHONPATH` evita que o `PYTHONPATH` do ROS injete plugins de pytest.

## Uso

```bash
env -u PYTHONPATH ~/anaconda3/envs/hrc/bin/python scripts/analyze_live.py    # análise + relatório
env -u PYTHONPATH ~/anaconda3/envs/hrc/bin/python scripts/export_videos.py   # vídeos
./run_tests.sh -v -s                                                         # sanidade do baseline
```

## Resultado (resumo — ver `reports/analise_ao_vivo.md`)

- **Ajuste de maior impacto: `send_window` 3 → 8** (nº de confirmações iguais
  consecutivas antes de mutar o contexto). Filtra os picos espúrios de predição
  no repouso — a causa nº 1 da deriva. No teste: `get_connectors` recall
  0.79→0.82, confirmações espúrias 20→13, erros de PlanGraph → 0, recuperação da
  sequência do plano 0.79→0.88, acurácia no fim da sessão 0.49→0.63.
- **Com o ajuste, o custo de construir o contexto ao vivo é pequeno** (top-1
  teste 0.685 vs 0.709 ideal) e some o colapso de classe que existia em
  `send_window=3` (onde o top-1 só parecia ok porque `no_action` absorvia o erro).
- **Ainda há deriva temporal residual** e gap de generalização (teste 0.685 vs
  treino 0.737) → para uso contínuo, re-sincronizar o estado por eventos externos.
- A **hipótese do comando de voz se confirma** para a contagem de tubos (sem a
  voz o erro nas dims de tubo/four_tubes salta), mas o modelo atual não converte
  esse contexto em predição melhor — estágio *four_tubes* sub-representado.

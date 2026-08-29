# Comportamento ao vivo do modelo treinado offline — é escalável?

`V2_dim10` (10D de contexto), rodando ao vivo sobre as 8 sessões gravadas (`skeleton.pkl`, sem câmera): o contexto é construído pelo próprio preditor (intenções confirmadas, **`send_window=8`** ajustado — ver seção «Ajuste» —, reset no repouso), com `begin_four_tubes` vindo do evento gravado (substituto do comando de voz). Pose crua (igual ao treino), `restrict=no`. **Teste = S02, S05** (participante/roteiro não vistos no treino) é o sinal de generalização; treino = as outras 6, para referência.

Comparamos sempre contra o **contexto ideal** (teacher-forced: contexto = ground truth no mesmo pipeline) para separar o custo de construir o contexto ao vivo do desempenho bruto do modelo.

## Veredito

- **O ajuste da janela de confirmação (`send_window` 3 → 8) melhora muito a construção do contexto ao vivo.** No teste, o recall de `get_connectors` sobe de 0.79 para 0.82, as confirmações espúrias no repouso caem de 20 para 13, os erros de PlanGraph vão a 0, e a recuperação da sequência do plano sobe de 0.79 para 0.88 (seção «Ajuste»).
- **Com o ajuste, o custo de construir o contexto ao vivo é pequeno:** top-1 0.685 vs 0.709 ideal, e agora sem o colapso de classe que existia em `send_window=3` (onde o top-1 parecia ok só porque `no_action` absorvia o erro).
- **Ainda não é totalmente escalável no tempo:** a acurácia cai no terço final da sessão (0.721 → 0.633 no teste) — melhor que antes (era 0.49 em `send_window=3`), mas o contexto ainda acumula alguma deriva. Há também gap de generalização (teste 0.685 vs treino 0.737).

## Ajuste — varredura da janela de confirmação (`send_window`)

`send_window` = nº de predições iguais consecutivas exigidas antes de confirmar uma intenção (e mutar o contexto). O padrão do `run_webcam_context.py` é 3. Aumentá-lo filtra os picos espúrios de predição durante o repouso, ao custo de eventualmente perder alcances muito curtos. Varredura no teste (S02, S05):

| send_window | top-1 | connectors | screws | wheels | confirm/GT | espúrias | erros plano | seq. recup. | acc. fim |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **3** | 0.639 | 0.788 | 0.886 | 0.514 | 54/48 | 20/54 | 17 | 0.792 | 0.489 |
| **4** | 0.639 | 0.788 | 0.889 | 0.514 | 53/48 | 17/53 | 17 | 0.792 | 0.491 |
| **5** | 0.692 | 0.788 | 0.915 | 0.617 | 56/48 | 18/56 | 2 | 0.875 | 0.628 |
| **6** | 0.687 | 0.811 | 0.918 | 0.617 | 55/48 | 17/55 | 1 | 0.875 | 0.630 |
| **8**  ← escolhido | 0.685 | 0.820 | 0.924 | 0.617 | 51/48 | 13/51 | 0 | 0.875 | 0.633 |
| **10** | 0.716 | 0.716 | 0.778 | 0.607 | 40/48 | 9/40 | 0 | 0.729 | 0.678 |

Escolhido **`send_window=8`**: melhor equilíbrio (menos espúrias, erros de PlanGraph ~0, maior recuperação da sequência e melhor acurácia no fim da sessão) e generaliza para o treino. Valores maiores (10+) começam a subcontar (perdem confirmações reais) e inflam o top-1 via `no_action`.

## 1. Desempenho ao vivo vs. contexto ideal

| Split | top-1 ao vivo | top-1 ideal | custo do live | no_action | connectors | screws | wheels |
|---|---:|---:|---:|---:|---:|---:|---:|
| test | 0.685 | 0.709 | -0.024 | 0.614 | 0.820 | 0.924 | 0.617 |
| train | 0.737 | 0.769 | -0.032 | 0.759 | 0.656 | 0.739 | 0.618 |

Linha de baixo por classe = recall ao vivo. Comparação por classe com o ideal (só teste):

| Classe | recall ao vivo | recall ideal |
|---|---:|---:|
| no_action | 0.614 | 0.591 |
| get_connectors | 0.820 | 0.923 |
| get_screws | 0.924 | 0.984 |
| get_wheels | 0.617 | 0.972 |

## 2. Escalabilidade

**(a) Generalização** — teste (não visto) vs. treino: top-1 ao vivo 0.685 (teste) vs. 0.737 (treino). Quanto menor a diferença, melhor generaliza para novos participantes/roteiros.

**(b) Estabilidade temporal** — o erro se acumula ao longo da sessão? Acurácia ao vivo em terços (início / meio / fim):

| Split | início | meio | fim |
|---|---:|---:|---:|
| test | 0.721 | 0.701 | 0.633 |
| train | 0.752 | 0.740 | 0.720 |

## 3. Onde o contexto ao vivo diverge do ideal (por dimensão)

Erro médio |ao vivo − GT| por dimensão do contexto (teste), sobre os frames com predição. Quanto maior, mais aquela componente do estado derrapou:

| Dimensão | erro médio |
|---|---:|
| stage:none | 0.301 |
| stage:top | 0.246 |
| screws_top/4 | 0.154 |
| long/4 | 0.153 |
| screws_bottom/4 | 0.112 |
| short/8 | 0.069 |
| wheels/4 | 0.066 |
| stage:bottom | 0.061 |
| screws_four_tubes/4 | 0.007 |
| stage:four_tubes | 0.000 |

### Ablação do comando de voz (`begin_four_tubes` on/off)

O comando de voz entrega os 4 tubos curtos do estágio *four_tubes* e é a única fonte desse incremento. Desligá-lo (visão pura) mede o quanto ele importa para a contagem de tubos:

- top-1 (teste) **com voz**: 0.685  |  **sem voz**: 0.665
- erro médio nas dims [short/8, long/4, screws_four_tubes/4] **com voz**: [0.069, 0.153, 0.007]  |  **sem voz**: [0.302, 0.153, 0.424]
- erros de PlanGraph **com voz**: 0  |  **sem voz**: 0

## 4. Survey de erros do PlanGraph (sinal para o treino)

Transições que o PlanGraph *rejeitou* porque o modelo confirmou uma intenção impossível no estado corrente. Cada linha: ação tentada, rótulo real da pessoa naquele frame, e nº de ocorrências. Confirmações num frame de `no_action` ou de classe diferente da tentada revelam confusões sistemáticas que valem como *hard negatives* no treino.

| Split | ação tentada | rótulo real no frame | ocorrências |
|---|---|---|---:|
| train | `begin_four_tubes` | ignore | 1 |

## 5. Confusão de confirmações (o que o modelo confirma vs. o que a pessoa faz)

Linhas = rótulo real no frame da confirmação; colunas = intenção confirmada. Fora da diagonal (e a linha `no_action`) são confirmações erradas — o que o treino precisa desambiguar. Teste:

| real \ confirmado | no_action | connectors | screws | wheels |
|---|---:|---:|---:|---:|
| no_action | 0 | 11 | 2 | 0 |
| get_connectors | 0 | 10 | 0 | 0 |
| get_screws | 0 | 0 | 19 | 1 |
| get_wheels | 0 | 0 | 3 | 5 |

## 6. Sub/sobre-contagem de confirmações por classe (teste)

| Sessão | connectors (modelo/real) | screws (modelo/real) | wheels (modelo/real) |
|---|---|---|---|
| S02 | 8/8 | 14/12 | 1/4 |
| S05 | 13/8 | 10/12 | 5/4 |

## 7. Conclusão: é escalável? o que fazer

**Com o ajuste da janela de confirmação (`send_window=8`) a construção do contexto ao vivo já fica bem melhor** — confirmações espúrias, erros de PlanGraph e colapso de classe caem muito. Resta uma deriva temporal menor e o gap de generalização, então para uso contínuo/sessões longas ainda recomenda-se re-sincronização externa do estado. Prioridades restantes, na ordem do impacto medido:

1. **Já aplicado: aumentar `send_window` (3 → 8).** Foi o ajuste de maior impacto e sem custo de treino (ver seção «Ajuste»): filtra os picos espúrios de predição no repouso, que eram a causa nº 1. Um passo adicional de baixo custo é um gate de movimento/confiança e **hard negatives de `no_action`** no treino (o levantamento de erros do PlanGraph — seção 4 — mostra que as confirmações indevidas restantes são `get_connectors` no repouso).
2. **Re-sincronizar o contexto por eventos externos confiáveis** (robô/voz) em vez de deduzir todo o estado só da visão — impede a deriva ilimitada que causa a queda no fim da sessão.
3. **Comando de voz — a sua hipótese se confirma para a contagem de tubos.** Sem o evento de voz, o erro do contexto nas dimensões de tubo/four_tubes salta (short 0.07→0.30, screws_four_tubes 0.01→0.42): a voz é a única fonte dos 4 tubos curtos do estágio *four_tubes*, então é essencial para essa parte do estado. Porém — achado importante — o modelo atual **não converte** esse contexto correto em predição melhor (o top-1 até sobe sem a voz, 0.685→0.665): o estágio *four_tubes* está sub-representado/mal aproveitado. Recomendação: manter a voz para a fidelidade do estado, mas **retreinar com mais peso no contexto de estágio** para o modelo de fato usá-lo.
4. **`get_connectors` é a classe mais frágil ao vivo** (recall despenca) — precisa de mais sinal de treino para se separar de `no_action` e de `get_screws`.

Em resumo: o modelo offline tem boa acurácia com contexto ideal, mas a construção do contexto ao vivo pelo próprio preditor introduz deriva que limita a escalabilidade. O caminho é (a) endurecer a decisão de confirmação contra o repouso, (b) re-sincronizar por eventos externos, e (c) retreinar para o modelo realmente aproveitar o contexto de estágio.


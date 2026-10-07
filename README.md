# Conectividade ecológica entre unidades de conservação da Região Metropolitana de Sorocaba (RMS)
### MLP + autômato celular sobre o uso da terra do MapBiomas e caminhos de menor custo — 1990–2025, projeções para 2030, 2040 e 2050

## Objetivo

Avaliar como a dinâmica do uso da terra afeta a **conectividade de menor custo entre as unidades de conservação (UCs)** da RMS (SP), e como ela pode evoluir até 2050.
O trabalho combina:

1. **Aquisição e pré-processamento** (Google Earth Engine + Python) de variáveis estáticas e dinâmicas, todas na mesma grade (EPSG:31983, 30 m, 5514 × 5316 px);
2. **MLP + autômato celular (AC)** para simular o uso da terra, com treino/teste espacial, validação 2020→2025 (matriz de confusão, Kappa, acurácia global, FoM) e comparação com a linha de base de persistência;
3. **Comparação** da rede de corredores de 2025 com as áreas prioritárias do Biota-FAPESP (2008);
4. **Projeção** do uso da terra e da conectividade para 2030, 2040 e 2050, em três cenários de demanda.

A resistência da paisagem por categoria de uso da terra segue Saranholi et al. (2022), Fig. 5 (leitura visual, reescalada para 1–100).

## Área de estudo, grade e reclassificação

- **Área:** limite da RMS, 1.162.379 ha, com margem de 10 km usada só para distâncias e vizinhança (mascarada nas saídas).
- **Grade canônica:** EPSG:31983, 30 m, origem (142470; 7477440), 5514 × 5316 px. A etapa de QA **aborta o pipeline** se qualquer raster divergir (CRS, transform, tamanho ou nodata).
- **Uso da terra:** MapBiomas Coleção 11 (`coverage_v3`), reclassificado em 6 categorias (`data_base/reclass_mapbiomas.txt`):

| Cat. | Nome | Códigos MapBiomas |
|---|---|---|
| 1 | Floresta | 3, 4, 5, 6, 49 |
| 2 | Formação natural não florestal | 10, 11, 12, 29, 50 |
| 3 | Pastagem | 15, 21 |
| 4 | Agricultura/silvicultura | 9, 14, 18, 19, 20, 35, 36, 39, 40, 41, 46, 47, 48, 62 |
| 5 | Urbano | 22, 23, 24, 25, 30, 32, 75 |
| 6 | Água | 26, 31, 33 |

## Variáveis

**Estáticas** (`data/processed/static_v1.tif`, 1 arquivo, válidas para todo o período)

| Variável | Fonte e processamento |
|---|---|
| `mde` | MDE da RMS reamostrado (bilinear) para a grade canônica |
| `declividade` | **Recalculada** do MDE reamostrado (Horn 3×3, graus); `decli_RMS.tif` (em %) serve só de conferência |
| `dist_hidrografia` | Distância euclidiana exata (EDT, `scipy`) à drenagem + massas d'água |

**Dinâmicas** (1 GeoTIFF por variável, 36 bandas = 1990–2025)

| Variável | Fonte e processamento |
|---|---|
| `ndvi_seca`, `ndvi_chuva` | Landsat Coleção 2 Nível 2 (LT05/LE07/LC08/LC09); máscara `QA_PIXEL` (bits 1–4); escala e offset aplicados **antes** do índice; média de jun–set (seca) e jan–mar (chuva) |
| `ndwi_seca`, `ndwi_chuva` | NDWI de Gao = (NIR−SWIR1)/(NIR+SWIR1), mesma composição |
| `lst_seca`, `lst_chuva` | Temperatura de superfície Landsat (`ST_B6`/`ST_B10`), em °C, **mascarada fora de 0–45 °C** (`dyn_lst_*_v2`); banda `lst_chuva` de 2002 invalidada |
| `prec_anual` | MapBiomas Atmosfera, precipitação anual (11,1 km, reamostrada); 2025 = 2024 |
| `dist_queimada` | Distância ao fogo (MapBiomas Fogo col. 5.1) na janela [t−4, t] |
| `dist_borda_floresta` | Distância com sinal à borda florestal (classe MapBiomas 3 no histórico; categoria 1 na projeção) |
| `dist_urbano` | Distância à categoria 5 (urbano) |

Preenchimento de falhas Landsat por janelas crescentes (`[t]` → `[t−2, t]` → `[t−4, t+2]`); a fração de pixels em cada nível está em `reports/qa/valid_by_year.csv`.

## Modelo

- **Preditores (C1, 15):** 3 estáticas + frações de cada categoria na vizinhança 5×5 (6) + categoria em t0 (one-hot, 6).
  **(C2, 25):** C1 + as 10 dinâmicas.
- **Amostragem:** estratificada por (classe t0, classe t1), até 3.000 por estrato; divisão **espacial** por blocos de 5 km (70/15/15). Correção de priors após o treino (`w = N_pop / n_amostra`).
- **MLP:** 64 → 32 neurônios (ReLU, dropout 0,2), softmax de 6 classes; Adam (lr 0,001), early stopping (paciência 15).
- **Autômato celular:** demanda por cadeia de Markov; alocação em 3 iterações por passo de 5 anos, com recálculo da vizinhança e das distâncias ao urbano e à borda florestal a cada iteração. Urbano e água são **absorventes** (a área urbana nunca diminui).

### Treino e teste (partição espacial, amostra estratificada)

| | C1 | C2 |
|---|---|---|
| Preditores | 15 | 25 |
| Acurácia no teste | 0,581 | 0,631 |
| Macro-F1 | 0,585 | 0,630 |
| Log-loss | 1,118 | 0,981 |

Maior importância por permutação no C2: vizinhança (`nbr_*`) e `dist_urbano` (`reports/model/permutation_importance_C2_v1.csv`).

### Validação 2020 → 2025 (modelo treinado em 1990–2020)

| | Persistência | C1 | C2 |
|---|---|---|---|
| Acurácia global | **0,9449** | 0,8897 | 0,8903 |
| Kappa | **0,9224** | 0,8450 | 0,8458 |
| FoM | 0 | 0,1139 | **0,1155** |
| Desacordo de quantidade | 0,0077 | 0,0068 | 0,0068 |
| Desacordo de alocação | 0,0474 | 0,1035 | 0,1029 |
| Pixels que mudam (simulado / real) | 0 / 5,51% | 8,57% / 5,51% | 8,57% / 5,51% |

A persistência supera os modelos em acurácia global e Kappa (apenas 5,5% dos pixels mudaram); o FoM mede o acerto da mudança. C1 e C2 empatam na prática.
Matrizes de confusão em `reports/model/confusion_2025_*_v1.csv`. O cenário **C2** foi o adotado nas projeções.

## Conectividade de menor custo

- **Resistência (1–100):** floresta 1, formação natural não florestal 90, pastagem 95, agricultura/silvicultura 53, urbano 100, água 74 (Saranholi et al., 2022, Fig. 5; estimada para *Tapirus terrestris* em Paranapiacaba — **a aplicação à RMS é extrapolação**).
- **Nós:** floresta dentro de cada UC (UCs com ≥ 10 ha de floresta): 17 nós; prioridade de sobreposição: proteção integral e UCs menores.
- **Ligações:** custo mínimo entre todos os pares (`skimage.graph.MCP_Geometric`); ligação **direta** = caminho que não atravessa o nó de uma terceira UC. Em 2025: 34 diretas, das quais 5 contíguas (≤ 0,2 km) e **29 não contíguas** (as avaliadas).
- **Corredor:** pixels com custo total até 10% acima do mínimo do par. **Ruptura** = perda de vegetação natural (categorias 1 e 2) dentro dos corredores de 2025. Custo em unidades de resistência × pixel (interpretar a variação relativa).

### Comparação com o Biota-FAPESP (2008)

- `VSOMA` (soma das indicações por grupo biológico) varia de **0 a 8** e `VSOMA ≥ 1` cobre ~97% da AOI, portanto não discrimina; usa-se `VSOMA ≥ 3`.
- Os corredores de 2025 têm **2,06×** mais área em `VSOMA ≥ 3` que o restante da AOI (28,9% × 14,0%).
- Ligações "não contempladas" (≥ 50% do caminho em `VSOMA ≤ 1`): **15 de 29**; a classificação depende do limiar (0, 15 e 26 ligações para `VSOMA ≤ 0`, `≤ 1`, `≤ 2`).
- A vegetação natural **aumentou** em todas as faixas de `VSOMA` entre 2008 e 2025 (ganho líquido).
Tabelas: `reports/biota2008/`.

## Cenários de projeção (2030, 2040 e 2050)

Passos de 5 anos (2030, 2035, 2040, 2045, 2050); relatam-se 2030, 2040 e 2050. Clima e variáveis espectrais ficam **constantes na média 2015–2025**; `dist_queimada` fica em 2025 (hipótese de estacionariedade).

| Cenário | Demanda (matriz de Markov) |
|---|---|
| `tendencial` | todos os pares de 1990–2025 |
| `expansao_agricola` | decênio 2005–2015 (maior ganho de agricultura + urbano; **a floresta cresce** nesse decênio) |
| `pressao_floresta` | decênio 1995–2005 (maior perda líquida de floresta; **limite superior hipotético**) |

**Variação de área 2025 → 2050 (mil ha, dentro da AOI)**

| | Tendencial | Expansão agrícola | Pressão-floresta |
|---|---|---|---|
| Floresta | −9,5 | +10,7 | −33,5 |
| Pastagem | −47,3 | −111,3 | −1,8 |
| Agricultura/silvicultura | +28,9 | +78,8 | +5,5 |
| Urbano | +25,6 | +23,5 | +25,0 |

**Conectividade (29 ligações; todas existem em todos os horizontes)**

| Cenário | Ano | Ruptura média | Perda natural nos corredores (ha) | Variação mediana do custo |
|---|---|---|---|---|
| Tendencial | 2030 / 2040 / 2050 | 5,0% / 10,6% / 12,8% | 11.194 / 23.322 / 28.318 | +9,6% / +15,7% / +19,7% |
| Expansão agrícola | 2030 / 2040 / 2050 | 4,1% / 8,2% / 9,6% | 9.358 / 17.754 / 20.325 | +5,5% / +6,0% / +1,5% |
| Pressão-floresta | 2030 / 2040 / 2050 | 6,2% / 13,8% / 17,9% | 14.015 / 31.412 / 40.748 | +13,9% / +35,7% / +56,1% |

As ligações mais atingidas envolvem a APA Avecuia (ruptura de até 35% no tendencial e 46% no pressão-floresta em 2050). Detalhe por ligação: `reports/connectivity/links_<cenario>_<ano>_v1.csv`.

## Limitações

- A demanda tendencial muda ~8% dos pixels por passo, contra 5,5% observado em 2020–2025; custo e ruptura do tendencial tendem a estar superestimados.
- A estação chuvosa de 1990–1997 depende de preenchimento por janela (em 1995, 97% dos pixels vêm da janela [t−2, t]); LST de chuva de 2002 inválida.
- Precipitação (11,1 km) praticamente não varia no espaço dentro da AOI e termina em 2024.
- Sem harmonização entre sensores Landsat (TM/ETM+ × OLI).
- A resistência de Saranholi et al. (2022) é uma extrapolação (outra região e espécie-alvo), com leitura visual da figura.
- A margem de 10 km não é simulada (permanece em 2025 nas projeções).
- O Biota-FAPESP (2008) tem escala estadual, critério de consulta a especialistas e base de uso anterior a 2008: a comparação é de complementaridade, não de validação.
- A incerteza do MapBiomas não é propagada.

## Estrutura do repositório

```
├── config_V2.yaml         # todos os parâmetros (nada fixo no código)
├── requirements.txt
├── run_pipeline.sh        # etapas 03 → 05 (download GEE, derivadas, QA), retomável
├── src/sbsrv2/            # módulos: grade, E/S, vetores, distâncias, GEE, features, amostragem, MLP, AC, métricas, conectividade
├── scripts/               # etapas numeradas (00 a 12) — ver abaixo
├── tests/                 # testes unitários (grade, EDT, declividade, métricas, alocação do AC, Markov)
├── models/                # MLP treinados (C1, C2, C2 final com todos os pares)
├── data_base/             # entradas (README com a estrutura esperada)
├── data/aoi/              # grade e UCs limpas
├── data/processed/        # mapas simulados 2025 e projetados 2030–2050 (GeoTIFF uint8, 6 categorias) e metadados do dataset
└── reports/               # resultados: model/, connectivity/, biota2008/, qa/
```

| Script | Função |
|---|---|
| `00_setup.py` | Checa GEE, assets e a reclassificação |
| `01_aoi_grid.py` | Grade canônica, máscara da AOI, UCs sem duplicatas |
| `02_static_layers.py` | MDE, declividade e distância à hidrografia |
| `03_gee_export.py` | Download anual por blocos (`computePixels`), retomável |
| `04_dynamic_layers.py` | Variáveis dinâmicas e pilhas de 36 bandas |
| `04b_mask_lst.py` | Máscara de faixa da LST (v2) |
| `05_qa.py` | QA de alinhamento e de pixels válidos |
| `06_build_dataset.py` | Amostras, blocos espaciais, scaler, matrizes de transição |
| `07_train_mlp.py` | Treino e teste do MLP (C1 e C2) |
| `08_validate_2025.py` | Simulação 2020→2025 e métricas |
| `09_connectivity.py` | Rede de menor custo de 2025 |
| `10_biota2008_compare.py` | Comparação com o Biota-FAPESP 2008 |
| `11_project.py` | Projeção 2030–2050 (cenários de demanda) |
| `12_connectivity_horizons.py` | Conectividade e ruptura nos horizontes |

## Como reproduzir

```bash
uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python -r requirements.txt
earthengine authenticate                 # e ajuste gee.project no config_V2.yaml
# coloque as entradas em data_base/ (ver data_base/README.md)
cd scripts
../.venv/bin/python 00_setup.py && ../.venv/bin/python 01_aoi_grid.py && ../.venv/bin/python 02_static_layers.py
cd .. && ./run_pipeline.sh               # 03 -> 05 (várias horas; sujeito às cotas do Earth Engine)
cd scripts
../.venv/bin/python 04b_mask_lst.py
for s in 06_build_dataset 07_train_mlp 08_validate_2025 09_connectivity 10_biota2008_compare; do ../.venv/bin/python $s.py; done
../.venv/bin/python 11_project.py        # treina o C2 final e projeta os três cenários (ou: --reuse-model <cenario>)
../.venv/bin/python 12_connectivity_horizons.py tendencial pressao_floresta expansao_agricola
cd .. && .venv/bin/python -m pytest tests
```

**Requisitos de máquina:** ~6,6 GB de RAM foram suficientes, mas a etapa do AC chega a ~5 GB (recomenda-se ≥ 8 GB); ~35 GB de disco no pico do download (os brutos anuais são apagados após o processamento). O download anual completo levou várias horas e foi retomado após esgotar a cota mensal do Earth Engine.
Rodar `11_project.py` com tensorflow-cpu em máquinas diferentes pode alterar levemente os pesos do MLP (treino não bit a bit reprodutível); os resultados deste repositório vêm do modelo em `models/mlp_C2_final_v1.keras`.

## Referências

- Saranholi, B.H. et al. (2022). *Long-term persistence of the large mammal lowland tapir is at risk in the largest Atlantic forest corridor*. Perspectives in Ecology and Conservation, 20, 263–271.
- Pontius Jr., R.G.; Millones, M. (2011). Death to Kappa: birth of quantity disagreement and allocation disagreement for accuracy assessment. International Journal of Remote Sensing, 32(15), 4407–4429.
- Gorelick, N. et al. (2017). Google Earth Engine: planetary-scale geospatial analysis for everyone. Remote Sensing of Environment, 202, 18–27.
- Projeto MapBiomas — Coleção 11 (cobertura e uso da terra), Coleção 5.1 (fogo) e Coleção 1 (atmosfera): <https://mapbiomas.org>.
- Biota-FAPESP (2008). Áreas prioritárias para incremento de conectividade — camada `CONECTIVIDADEFAPESP`, DataGeo/SP.
- Figure of Merit: **(NEEDED REF)**.

## Licença e citação

Licença a definir pela autora. Autoria: Gabriela dos Santos Luchetti Vieira (UNESP/Sorocaba, LABGEMM).

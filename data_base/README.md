# Dados de entrada (`data_base/`)

Estes arquivos **não** são versionados (tamanho e direitos de terceiros). Para reproduzir o pipeline, coloque-os com esta estrutura
(os caminhos estão em `inputs:` no `config_V2.yaml`):

```
data_base/
├── reclass_mapbiomas.txt                      # incluído no repositório
├── limites/
│   ├── RMS_reproj.shp                         # limite da Região Metropolitana de Sorocaba (EPSG:31983)
│   └── areas-protegidas.shp                   # unidades de conservação da área (EPSG:31983)
├── Variaveis_estaticas/
│   ├── MDE_RMS.tif                            # modelo digital de elevação da RMS
│   └── decli_RMS.tif                          # declividade em % (usada só como conferência no QA)
├── hidrografia/
│   ├── Rep_DrenagemSP.shp                     # rede de drenagem (SAD69/96 UTM 22S)
│   └── hid_massa_dagua_a.shp                  # massas d'água
├── rodovias/                                  # opcional: só se dist_estradas/dist_ferrovias forem ativadas
│   ├── rod_trecho_rodoviario_l.shp
│   └── fer_trecho_ferroviario_l.shp
└── conectividade_fapespbiota2008/
    └── CONECTIVIDADEFAPESP.shp                # áreas prioritárias p/ incremento de conectividade (Biota-FAPESP, 2008)
```

A camada `CONECTIVIDADEFAPESP` vem do DataGeo/SP (WFS: camada `CONECTIVIDADEFAPESP` em `datageo.ambiente.sp.gov.br`).
Os produtos do MapBiomas, do fogo e da atmosfera são lidos diretamente do Google Earth Engine (não precisam ser baixados).

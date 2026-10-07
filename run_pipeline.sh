#!/usr/bin/env bash
# Bloco B em sequência (retomável): 03 download -> 04 derivadas -> 05 QA. Uso: ./run_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/scripts"
for s in 03_gee_export 04_dynamic_layers 05_qa; do
  echo "== INICIO $s $(date +%H:%M:%S)"
  ../.venv/bin/python $s.py > ../reports/logs/$s.out 2>&1 || { echo "== FALHOU $s"; tail -20 ../reports/logs/$s.out; exit 1; }
  echo "== FIM $s $(date +%H:%M:%S)"
done
echo "== BLOCO B CONCLUIDO"

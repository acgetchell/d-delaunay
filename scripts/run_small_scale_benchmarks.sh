#!/usr/bin/env bash
set -euo pipefail
mkdir -p benchmarks/results
export CRITERION_OUTPUT_FORMAT=json
for dim in 2 3 4; do
  cargo bench --bench small_scale_triangulation -- --dimension $dim \
    --save-baseline benchmarks/results/bench_dim${dim}.json
done

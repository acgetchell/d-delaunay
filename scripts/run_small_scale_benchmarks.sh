#!/usr/bin/env bash
set -euo pipefail

# Error handling function
error_exit() {
    local message="$1"
    local code="${2:-1}"
    echo "ERROR: $message" >&2
    exit "$code"
}
mkdir -p benchmarks/results
export CRITERION_OUTPUT_FORMAT=json

echo "Running small scale triangulation benchmarks..."
# Run the benchmark which includes all dimensions (2D, 3D, 4D)
cargo bench --bench small_scale_triangulation \
  --message-format=json 2>&1 | tee benchmarks/results/small_scale_benchmarks.json

echo "Benchmark results saved to benchmarks/results/small_scale_benchmarks.json"

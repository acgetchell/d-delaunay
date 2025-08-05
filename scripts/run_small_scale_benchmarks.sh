#!/usr/bin/env bash
set -euo pipefail

# Error handling function
error_exit() {
    local message="$1"
    local code="${2:-1}"
    echo "ERROR: $message" >&2
    exit "$code"
}
# Find project root (directory containing Cargo.toml)
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)

# Create results directory and ensure we're in project root for cargo commands
mkdir -p "${PROJECT_ROOT}/benches/results"
cd "${PROJECT_ROOT}"
export CRITERION_OUTPUT_FORMAT=json

echo "Running small scale triangulation benchmarks..."
# Run the benchmark which includes all dimensions (2D, 3D, 4D)
# Remove --message-format=json to get normal cargo output, let Criterion handle JSON output
cargo bench --bench small_scale_triangulation 2>&1 | tee "${PROJECT_ROOT}/benches/results/small_scale_benchmarks.txt"

# Also save Criterion's JSON output if it exists
if [[ -d "${PROJECT_ROOT}/target/criterion" ]]; then
    echo "Extracting Criterion JSON results..."
    "${PROJECT_ROOT}/scripts/extract_benchmarks.sh" "${PROJECT_ROOT}/target/criterion" > "${PROJECT_ROOT}/benches/results/small_scale_benchmarks.json"
    echo "Benchmark JSON results saved to ${PROJECT_ROOT}/benches/results/small_scale_benchmarks.json"
fi

echo "Benchmark text output saved to ${PROJECT_ROOT}/benches/results/small_scale_benchmarks.txt"

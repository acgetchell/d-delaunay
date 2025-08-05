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
    
    # Check if extract_benchmarks.sh exists and is executable
    if [[ ! -f "${PROJECT_ROOT}/scripts/extract_benchmarks.sh" ]]; then
        error_exit "extract_benchmarks.sh not found at ${PROJECT_ROOT}/scripts/extract_benchmarks.sh"
    fi
    
    if [[ ! -x "${PROJECT_ROOT}/scripts/extract_benchmarks.sh" ]]; then
        error_exit "extract_benchmarks.sh is not executable. Run: chmod +x ${PROJECT_ROOT}/scripts/extract_benchmarks.sh"
    fi
    
    "${PROJECT_ROOT}/scripts/extract_benchmarks.sh" "${PROJECT_ROOT}/target/criterion" > "${PROJECT_ROOT}/benches/results/small_scale_benchmarks.json"
    echo "Benchmark JSON results saved to ${PROJECT_ROOT}/benches/results/small_scale_benchmarks.json"
    
    # Generate baseline results file
    if [[ -f "${PROJECT_ROOT}/scripts/generate_baseline.sh" ]] && [[ -x "${PROJECT_ROOT}/scripts/generate_baseline.sh" ]]; then
        echo "Generating baseline results..."
        "${PROJECT_ROOT}/scripts/generate_baseline.sh"
        echo "Baseline results saved to ${PROJECT_ROOT}/benches/baseline_results.txt"
    else
        echo "Warning: generate_baseline.sh not found or not executable. Baseline results not generated."
    fi
fi

echo "Benchmark text output saved to ${PROJECT_ROOT}/benches/results/small_scale_benchmarks.txt"

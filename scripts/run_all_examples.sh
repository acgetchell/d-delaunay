#!/bin/bash
set -euo pipefail

# Error handling function
error_exit() {
    local message="$1"
    local code="${2:-1}"
    echo "ERROR: $message" >&2
    exit "$code"
}

# Script to run all examples in the d-delaunay project

echo "Running all examples for d-delaunay project..."
echo "=============================================="

# Simple examples that don't take arguments (excluding test_circumsphere which takes args)
simple_examples=(
    "boundary_analysis_trait"
    "check_float_traits"
    "implicit_conversion"
    "point_comparison_and_hashing"
    "test_alloc_api"
)

# Run simple examples
for example in "${simple_examples[@]}"; do
    echo
    echo "=== Running $example ==="
    echo "--------------------------------------"
    if ! cargo run --example "$example"; then
        error_exit "Example $example failed!"
    fi
    echo
done

# Run test_circumsphere with comprehensive test categories
test_circumsphere_tests=(
    "all"              # All basic dimensional tests and orientation tests
    "test-all-points"   # Single point tests in all dimensions
    "debug-all"         # All debug tests
)

echo
echo "=== Running test_circumsphere comprehensive tests ==="
echo "---------------------------------------------------"

for test_name in "${test_circumsphere_tests[@]}"; do
    echo
    echo "--- Running test_circumsphere $test_name ---"
    if ! cargo run --example test_circumsphere -- "$test_name"; then
        error_exit "test_circumsphere $test_name failed!"
    fi
done

echo
echo "=============================================="
echo "All examples and tests completed successfully!"

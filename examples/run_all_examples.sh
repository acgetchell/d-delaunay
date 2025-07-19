#!/bin/bash

# Script to run all examples in the d-delaunay project

echo "Running all examples for d-delaunay project..."
echo "=============================================="

# Simple examples that don't take arguments
simple_examples=(
    "check_float_traits"
    "implicit_conversion"
    "point_comparison_and_hashing"
)

# Run simple examples
for example in "${simple_examples[@]}"; do
    echo
    echo "=== Running $example ==="
    echo "--------------------------------------"
    if ! cargo run --example "$example"; then
        echo "ERROR: Example $example failed!"
        exit 1
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
    if ! cargo run --example test_circumsphere "$test_name"; then
        echo "ERROR: test_circumsphere $test_name failed!"
        exit 1
    fi
done

echo
echo "=============================================="
echo "All examples and tests completed successfully!"

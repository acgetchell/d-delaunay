#!/bin/bash

# Script to run all examples in the d-delaunay project

echo "Running all examples for d-delaunay project..."
echo "=============================================="

examples=(
    "check_float_traits"
    "implicit_conversion"
    "point_comparison_and_hashing"
    "test_circumsphere"
)

for example in "${examples[@]}"; do
    echo
    echo "=== Running $example ==="
    echo "--------------------------------------"
    if ! cargo run --example "$example"; then
        echo "ERROR: Example $example failed!"
        exit 1
    fi
    echo
done

echo "=============================================="
echo "All examples completed successfully!"

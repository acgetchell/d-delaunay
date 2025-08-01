#!/bin/bash

# Script to record baseline performance metrics for assign_neighbors
# Run this before making optimizations to establish baseline

echo "Recording baseline performance for assign_neighbors..."
echo "Date: $(date)" > baseline_results.txt
echo "Git commit: $(git rev-parse HEAD)" >> baseline_results.txt
echo "" >> baseline_results.txt

echo "Running baseline benchmarks..."

# Run specific benchmarks and capture output
echo "=== 10 random points (3D) ===" >> baseline_results.txt
cargo bench --bench assign_neighbors_performance -- "assign_neighbors_random/random_points/10" 2>&1 | grep -E "(time:|thrpt:)" >> baseline_results.txt

echo "" >> baseline_results.txt
echo "=== 20 random points (3D) ===" >> baseline_results.txt
cargo bench --bench assign_neighbors_performance -- "assign_neighbors_random/random_points/20" 2>&1 | grep -E "(time:|thrpt:)" >> baseline_results.txt

echo "" >> baseline_results.txt
echo "=== 10 points (2D) ===" >> baseline_results.txt
cargo bench --bench assign_neighbors_performance -- "assign_neighbors_2d_vs_3d/2d/10" 2>&1 | grep -E "(time:|thrpt:)" >> baseline_results.txt

echo "" >> baseline_results.txt
echo "=== 10 points (3D) ===" >> baseline_results.txt
cargo bench --bench assign_neighbors_performance -- "assign_neighbors_2d_vs_3d/3d/10" 2>&1 | grep -E "(time:|thrpt:)" >> baseline_results.txt

echo "" >> baseline_results.txt
echo "=== Original microbenchmark comparison ===" >> baseline_results.txt
cargo bench --bench microbenchmarks -- "assign_neighbors/assign_neighbors/10" 2>&1 | grep -E "(time:|thrpt:)" >> baseline_results.txt

echo "" >> baseline_results.txt
cargo bench --bench microbenchmarks -- "assign_neighbors/assign_neighbors/25" 2>&1 | grep -E "(time:|thrpt:)" >> baseline_results.txt

echo "Baseline results saved to baseline_results.txt"
echo ""
echo "Summary of current performance:"
cat baseline_results.txt

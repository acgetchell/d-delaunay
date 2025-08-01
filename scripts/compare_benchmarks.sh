#!/bin/bash

# Script to compare extracted benchmark results with baseline
summary_file="results_dim2_summary.json"

echo "Benchmark Results Summary:"
echo "========================"

# Extract baseline values from baseline_results.txt for comparison
# The baseline contains: 10 points (2D) -> 7.3600 µs, 10 points (3D) -> 26.255 µs, 20 points (3D) -> 103.84 µs
baseline_2d_10=7.3600  # µs -> ns conversion: * 1000
baseline_3d_10=26.255  # µs -> ns conversion: * 1000  
baseline_3d_20=103.84  # µs -> ns conversion: * 1000

# Convert µs to ns for comparison
baseline_2d_10_ns=$(echo "$baseline_2d_10 * 1000" | bc)
baseline_3d_10_ns=$(echo "$baseline_3d_10 * 1000" | bc)
baseline_3d_20_ns=$(echo "$baseline_3d_20 * 1000" | bc)

echo "Current Benchmark Results:"
echo "--------------------------"
jq -r '.benchmarks[] | "\(.id): \(.mean) ns"' "$summary_file"

echo ""
echo "Regression Analysis (>5% threshold):"
echo "====================================="

# Check specific benchmarks that might match baseline
jq -c '.benchmarks[]' "$summary_file" | while read benchmark; do
  id=$(echo "$benchmark" | jq -r '.id')
  mean=$(echo "$benchmark" | jq -r '.mean')
  
  # Check for potential matches and calculate regression
  case "$id" in
    *"tds_new_2d/tds_new/10"*)
      change=$(echo "scale=2; ($mean - $baseline_2d_10_ns) / $baseline_2d_10_ns * 100" | bc)
      if (( $(echo "$change > 5.0 || $change < -5.0" | bc -l) )); then
        printf "⚠️  REGRESSION: %s: %.2f%% change\n" "$id" "$change"
      else
        printf "✅ OK: %s: %.2f%% change\n" "$id" "$change"
      fi
      ;;
    *"tds_new_3d/tds_new/10"*)
      change=$(echo "scale=2; ($mean - $baseline_3d_10_ns) / $baseline_3d_10_ns * 100" | bc)
      if (( $(echo "$change > 5.0 || $change < -5.0" | bc -l) )); then
        printf "⚠️  REGRESSION: %s: %.2f%% change\n" "$id" "$change"
      else
        printf "✅ OK: %s: %.2f%% change\n" "$id" "$change"
      fi
      ;;
    *"tds_new_3d/tds_new/20"*)
      change=$(echo "scale=2; ($mean - $baseline_3d_20_ns) / $baseline_3d_20_ns * 100" | bc)
      if (( $(echo "$change > 5.0 || $change < -5.0" | bc -l) )); then
        printf "⚠️  REGRESSION: %s: %.2f%% change\n" "$id" "$change"
      else
        printf "✅ OK: %s: %.2f%% change\n" "$id" "$change"
      fi
      ;;
  esac
done


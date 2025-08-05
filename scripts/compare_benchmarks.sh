#!/usr/bin/env bash

# Enable strict mode: exit on error, undefined variables, pipe failures
set -euo pipefail

# Error handling function
error_exit() {
    local message="$1"
    local code="${2:-1}"
    echo "ERROR: $message" >&2
    exit "$code"
}

# Convert scientific notation to decimal format that bc can handle
# Criterion benchmarks often output values in scientific notation (e.g., 1.23e-6)
# but bc cannot parse scientific notation, so we need to convert it to decimal format
convert_scientific() {
    local value="$1"
    
    # Check if the value contains 'e' or 'E' (scientific notation)
    if [[ "$value" == *[eE]* ]]; then
        # Use awk to convert scientific notation to decimal format with 12 decimal places
        # The +0 operation forces awk to interpret the value as a number
        local converted
        converted=$(echo "$value" | awk 'BEGIN{OFMT="%.12f"} {print $1+0}' 2>/dev/null)
        if [[ $? -ne 0 ]] || [[ -z "$converted" ]]; then
            error_exit "Failed to convert scientific notation: $value"
        fi
        echo "$converted"
    else
        # Already in decimal format, return as-is
        echo "$value"
    fi
}

# Print usage information
usage() {
    echo "Usage: compare_benchmarks.sh [-h|--help] [directory]"
    echo
    echo "Compares extracted benchmark results with baseline values."
    echo
    echo "Options:"
    echo "  -h, --help      Show this help message and exit"
    echo
    echo "Arguments:"
    echo "  directory       Optional directory containing results (default: benches/results)"
    echo
    echo "Exit Codes:"
    echo "  0  Success"
    echo "  1  Error occurred"
    exit 0
}

# Find project root and set default values
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
result_dir="${PROJECT_ROOT}/benches/results"

# Check for help option and parse directory argument
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            ;;
        *)
            result_dir=$1
            shift
            ;;
    esac
done

# Trap to catch unexpected errors
trap 'error_exit "Unexpected error at line $LINENO"' ERR

# Check dependencies first
command -v jq >/dev/null 2>&1 || error_exit "jq is required but not found. Install with: brew install jq"
command -v bc >/dev/null 2>&1 || error_exit "bc is required but not found. Install with: brew install bc"

# Script to compare extracted benchmark results with baseline
# Look for the most likely benchmark results file
if [[ -f "$result_dir/small_scale_benchmarks.json" ]]; then
    summary_file="$result_dir/small_scale_benchmarks.json"
elif [[ -f "$result_dir/results_dim2_summary.json" ]]; then
    summary_file="$result_dir/results_dim2_summary.json"
else
    error_exit "No benchmark results file found in $result_dir. Expected small_scale_benchmarks.json or results_dim2_summary.json"
fi

echo "Using benchmark results file: $summary_file"

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

# Verify the file exists and is readable
[[ -f "$summary_file" ]] || error_exit "Benchmark results file not found: $summary_file"
[[ -r "$summary_file" ]] || error_exit "Benchmark results file not readable: $summary_file"

# Check specific benchmarks that might match baseline
jq -c '.benchmarks[]' "$summary_file" 2>/dev/null | while read -r benchmark; do
  id=$(echo "$benchmark" | jq -r '.id')
  mean_raw=$(echo "$benchmark" | jq -r '.mean')
  
  # Convert scientific notation to decimal format for bc
  mean=$(convert_scientific "$mean_raw")
  
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


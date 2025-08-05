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

# Check if benchmarks have been run
if [[ ! -f "${PROJECT_ROOT}/benches/results/small_scale_benchmarks.txt" ]]; then
    error_exit "Benchmark results not found. Run scripts/run_small_scale_benchmarks.sh first."
fi

# Get current date and git commit
CURRENT_DATE=$(date)
GIT_COMMIT=$(git rev-parse HEAD)

# Output file
OUTPUT_FILE="${PROJECT_ROOT}/benches/baseline_results.txt"

echo "Generating baseline results from benchmark output..."
echo "Output file: $OUTPUT_FILE"

# Create the baseline_results.txt file
cat > "$OUTPUT_FILE" << EOF
Date: $CURRENT_DATE
Git commit: $GIT_COMMIT

EOF

# Extract performance data from the benchmark text output
# This parses the format from criterion benchmarks
grep -A 4 "=== [0-9]* Points" "${PROJECT_ROOT}/benches/results/small_scale_benchmarks.txt" | \
while IFS= read -r line; do
    if [[ "$line" =~ ^===.*Points.*=== ]]; then
        echo "$line"
    elif [[ "$line" =~ ^Time: ]]; then
        echo "$line"
    elif [[ "$line" =~ ^Throughput: ]]; then
        echo "$line"
    elif [[ "$line" =~ ^Time\ Change: ]]; then
        echo "$line"
    elif [[ "$line" =~ ^Throughput\ Change: ]]; then
        echo "$line"
        echo ""  # Add blank line after each section
    fi
done >> "$OUTPUT_FILE"

# Parse the raw benchmark output for sections that might not match the above pattern
# Extract from the small_scale_benchmarks.txt using the actual criterion output format
awk '
/^tds_new_[0-9]d\/tds_new\/[0-9]+/ {
    # Extract benchmark info
    benchmark_line = $0
    getline time_line
    getline throughput_line
    getline change_line
    getline throughput_change_line
    
    # Extract dimensions and points
    match(benchmark_line, /tds_new_([0-9])d\/tds_new\/([0-9]+)/, groups)
    dimension = groups[1]
    points = groups[2]
    
    # Parse time line
    match(time_line, /time:\s+\[([^\]]+)\]/, time_groups)
    time_values = time_groups[1]
    
    # Parse throughput line  
    match(throughput_line, /thrpt:\s+\[([^\]]+)\]/, thrpt_groups)
    thrpt_values = thrpt_groups[1]
    
    # Parse change lines
    match(change_line, /time:\s+\[([^\]]+)\]/, change_groups)
    time_change = change_groups[1]
    
    match(throughput_change_line, /thrpt:\s+\[([^\]]+)\]/, thrpt_change_groups)
    thrpt_change = thrpt_change_groups[1]
    
    # Output in our format
    printf "=== %s Points (%sD) ===\n", points, dimension
    printf "Time:   [%s]\n", time_values
    printf "Throughput:  [%s]\n", thrpt_values
    printf "Time Change:   [%s]\n", time_change
    printf "Throughput Change:  [%s]\n", thrpt_change
    printf "\n"
}
' "${PROJECT_ROOT}/benches/results/small_scale_benchmarks.txt" >> "$OUTPUT_FILE"

echo "=== Original microbenchmark comparison ===" >> "$OUTPUT_FILE"

echo "Baseline results generated successfully at $OUTPUT_FILE"

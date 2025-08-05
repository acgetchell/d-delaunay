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

# Dependency checking function
check_dependencies() {
    # Function to get package name for a command
    get_package_name() {
        case "$1" in
            "jq") echo "jq" ;;
            "find") echo "findutils" ;;
            "sort") echo "coreutils" ;;
            *) echo "$1" ;;
        esac
    }
    
    # Array of required commands
    local required_commands=("jq" "find" "sort")
    
    # Check each required command
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            local package_name
            package_name=$(get_package_name "$cmd")
            error_exit "$cmd is required but not found.
Install on macOS with brew: brew install $package_name"
        fi
    done
}

# Print usage information
usage() {
    echo "Usage: extract_benchmarks.sh [-h|--help] [directory]"
    echo
    echo "Extracts benchmark results from Criterion JSON files and outputs as JSON."
    echo
    echo "Dependencies:"
    echo "  Requires jq, find, and sort commands (install on macOS: brew install jq findutils coreutils)"
    echo
    echo "Options:"
    echo "  -h, --help      Show this help message and exit"
    echo
    echo "Arguments:"
    echo "  directory       Optional directory containing criterion results (default: target/criterion)"
    echo
    echo "Exit Codes:"
    echo "  0  Success"
    echo "  1  Error occurred"
    exit 0
}

# Find project root and set default values
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)
result_dir="${PROJECT_ROOT}/target/criterion"

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

# Check dependencies after argument parsing
check_dependencies

# Set TARGET_DIR variable for validation
TARGET_DIR=${result_dir}

# Verify target directory exists and is readable
[[ -d "$TARGET_DIR" ]] || error_exit "Directory '$TARGET_DIR' not found" 3
[[ -r "$TARGET_DIR" ]] || error_exit "Directory '$TARGET_DIR' is not readable" 3

# Convert to absolute path for consistent processing
TARGET_DIR=$(cd "$TARGET_DIR" && pwd)

# Trap to catch unexpected errors
trap 'error_exit "Unexpected error at line $LINENO"' ERR

# Script to extract benchmark results from Criterion JSON files
echo '{"benchmarks": ['

# Find all estimates.json files and process them
estimate_files=()
while IFS= read -r -d '' file; do
    estimate_files+=("$file")
done < <(find "$TARGET_DIR" -path "*/new/estimates.json" -print0 | sort -z)

# Check if any benchmark files were found
if [[ ${#estimate_files[@]} -eq 0 ]]; then
    echo ']}'
    error_exit "No benchmark results found in $TARGET_DIR. Run benchmarks first with 'cargo bench'."
fi

first=true
for estimates_file in "${estimate_files[@]}"; do
    # Extract the benchmark path (e.g., tds_new_2d/tds_new/50)
    benchmark_path=$(echo "$estimates_file" | sed "s|$TARGET_DIR/||" | sed 's|/new/estimates.json||')
    
    # Extract mean point estimate from JSON, handle potential errors
    if ! mean=$(jq -r '.mean.point_estimate' "$estimates_file" 2>/dev/null); then
        echo "Warning: Failed to extract mean from $estimates_file" >&2
        continue
    fi
    
    # Skip if mean is null or not a valid number
    if [[ "$mean" == "null" ]] || ! [[ "$mean" =~ ^[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?$ ]]; then
        echo "Warning: Invalid mean value '$mean' in $estimates_file" >&2
        continue
    fi
    
    if [ "$first" = true ]; then
        first=false
    else
        echo ","
    fi
    
    echo -n "  {\"id\": \"$benchmark_path\", \"mean\": $mean, \"alloc_bytes\": null}"
done

echo ''
echo ']}'

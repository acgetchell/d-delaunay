#!/usr/bin/env bash

# benchmark_parser.sh - Shared utility functions for parsing benchmark data
# This script provides reusable functions for parsing benchmark results across different scripts

#==============================================================================
# DEPENDENCY CHECKS
#==============================================================================

# Function to check if required dependencies are available
# Usage: check_benchmark_parser_dependencies
check_benchmark_parser_dependencies() {
    if ! command -v bc >/dev/null 2>&1; then
        echo "ERROR: bc is required but not found. Please install via your system package manager (e.g., apt, brew, winget)" >&2
        return 1
    fi
}

#==============================================================================
# BENCHMARK PARSING FUNCTIONS
#==============================================================================

# Function to detect benchmark start lines and extract metadata
# Usage: parse_benchmark_start "line_content"
# Returns: "point_count,dimension" or empty string if not a benchmark line
parse_benchmark_start() {
    local line="$1"
    
    # Detect "Benchmarking tds_new_2d/tds_new/10" format
    if [[ "$line" =~ ^Benchmarking[[:space:]]+tds_new_([0-9])d/tds_new/([0-9]+)$ ]]; then
        local dimension="${BASH_REMATCH[1]}D"
        local point_count="${BASH_REMATCH[2]}"
        echo "$point_count,$dimension"
        return 0
    fi
    
    return 1
}

# Function to extract timing data from benchmark result lines
# Usage: extract_timing_data "timing_line"
# Returns: "low,mean,high" timing values or empty string if not found
extract_timing_data() {
    local line="$1"
    
    # Match lines like: "tds_new_2d/tds_new/10   time:   [354.30 µs 356.10 µs 357.91 µs]"
    if [[ "$line" =~ ^tds_new_[0-9]d/tds_new/[0-9]+[[:space:]]+time:[[:space:]]+\[ ]]; then
        # Extract timing values from within brackets
        local timing_values
        timing_values=$(echo "$line" | grep -o '\[[^]]*\]' | grep -o '[0-9.]\+' | head -3)
        
        # Convert to array and validate we have 3 values
        local timing_array
        read -ra timing_array <<< "$timing_values"
        if [[ ${#timing_array[@]} -eq 3 ]]; then
            echo "${timing_array[0]},${timing_array[1]},${timing_array[2]}"
            return 0
        fi
    fi
    
    return 1
}

# Function to parse benchmark identifier from result lines to extract metadata
# Usage: parse_benchmark_identifier "tds_new_2d/tds_new/10   time:   [...]"
# Returns: "point_count,dimension" or empty string if not parsable
parse_benchmark_identifier() {
    local line="$1"
    
    if [[ "$line" =~ ^tds_new_([0-9])d/tds_new/([0-9]+) ]]; then
        local dimension="${BASH_REMATCH[1]}D"
        local point_count="${BASH_REMATCH[2]}"
        echo "$point_count,$dimension"
        return 0
    fi
    
    return 1
}

# Function to format benchmark output with consistent spacing
# Usage: format_benchmark_result "points" "dimension" "low" "mean" "high" "output_file"
format_benchmark_result() {
    local points="$1"
    local dimension="$2"
    local low="$3"
    local mean="$4"
    local high="$5"
    local output_file="$6"
    
    # Print the header and timing line using printf for consistent spacing
    {
        printf "=== %s Points (%s) ===\n" "$points" "$dimension"
        printf "Time: [%s, %s, %s] μs\n" "$low" "$mean" "$high"
        printf "\n"
    } >> "$output_file"
}

#==============================================================================
# WHILE READ PARSING FUNCTION
#==============================================================================

# Main parsing function using while read loop
# Usage: parse_benchmarks_with_while_read "input_file" "output_file"
parse_benchmarks_with_while_read() {
    local input_file="$1"
    local output_file="$2"
    
    # Initialize variables for tracking current benchmark
    local current_points=""
    local current_dimension=""
    
    # Use while read to process each line and detect benchmark starts
    while IFS= read -r line; do
        # Detect start of each benchmark ("Benchmarking …")
        if metadata=$(parse_benchmark_start "$line"); then
            # On detection, extract point count and dimension
            IFS=',' read -r current_points current_dimension <<< "$metadata"
            
        # Check for timing result lines
        elif timing_data=$(extract_timing_data "$line"); then
            # If we have timing data and current metadata, format output
            if [[ -n "$timing_data" && -n "$current_points" && -n "$current_dimension" ]]; then
                IFS=',' read -r low mean high <<< "$timing_data"
                format_benchmark_result "$current_points" "$current_dimension" "$low" "$mean" "$high" "$output_file"
                
                # Reset current benchmark tracking
                current_points=""
                current_dimension=""
            fi
        fi
    done < "$input_file"
}

#==============================================================================
# AWK PARSING FUNCTION
#==============================================================================

# AWK-based parsing function for comprehensive parsing
# Usage: parse_benchmarks_with_awk "input_file" "output_file"
parse_benchmarks_with_awk() {
    local input_file="$1"
    local output_file="$2"
    
    awk '
    # Detect benchmark result lines with timing data
    /^tds_new_[0-9]d\/tds_new\/[0-9]+[ 	]+time:/ {
        # Use simple field splitting to extract data
        split($1, id_parts, "/")
        
        # Extract dimension from "tds_new_2d" format
        if (match(id_parts[1], /tds_new_([0-9])d/)) {
            dimension = substr(id_parts[1], RSTART+8, 1) "D"
        }
        
        # Extract points from third part
        points = id_parts[3]
        
        # Find timing values in brackets
        bracket_start = index($0, "[")
        bracket_end = index($0, "]")
        if (bracket_start > 0 && bracket_end > bracket_start) {
            timing_str = substr($0, bracket_start + 1, bracket_end - bracket_start - 1)
            
            # Split timing string by spaces and extract numeric values
            n = split(timing_str, timing_parts, " ")
            time_values_count = 0
            for (i = 1; i <= n; i++) {
                if (match(timing_parts[i], /^[0-9.]+$/)) {
                    time_values_count++
                    if (time_values_count == 1) time_low = timing_parts[i]
                    if (time_values_count == 2) time_mean = timing_parts[i]
                    if (time_values_count == 3) time_high = timing_parts[i]
                }
            }
            
            # Print if we have all required values
            if (time_values_count >= 3 && points != "" && dimension != "") {
                printf "=== %s Points (%s) ===\n", points, dimension
                printf "Time: [%s, %s, %s] μs\n", time_low, time_mean, time_high
                printf "\n"
            }
        }
    }
    ' "$input_file" >> "$output_file"
}

#==============================================================================
# UTILITY FUNCTIONS FOR BASELINE PARSING
#==============================================================================

# Function to extract baseline timing values from baseline_results.txt
# Usage: extract_baseline_time "points" "dimension" "baseline_file"
# Returns: timing value in nanoseconds
extract_baseline_time() {
    local points="$1"
    local dimension="$2"
    local baseline_file="$3"
    
    # Look for the section header and extract the mean time value
    # Handle dimension parameter that may or may not already include 'D' suffix
    local dimension_with_d="$dimension"
    if [[ "$dimension" != *"D" ]]; then
        dimension_with_d="${dimension}D"
    fi
    
    local time_line
    time_line=$(grep -A 1 "=== $points Points ($dimension_with_d) ===" "$baseline_file" | grep "Time:" | head -1)
    
    if [[ -z "$time_line" ]]; then
        # Try alternate format for lowercase points
        time_line=$(grep -A 1 "=== $points points ($dimension_with_d) ===" "$baseline_file" | grep -i "time:" | head -1)
    fi
    
    if [[ -n "$time_line" ]]; then
        # Extract the middle value from [low, mean, high] and convert to nanoseconds  
        local time_value
        time_value=$(echo "$time_line" | grep -o '\[[^]]*\]' | tr -d '[]' | cut -d',' -f2 | tr -d ' ')
        
        # Convert to nanoseconds based on unit (check original line for unit)
        if [[ "$time_line" == *"μs"* ]] || [[ "$time_line" == *"µs"* ]]; then
            printf "%.0f\n" "$(echo "$time_value * 1000" | bc -l)"
        elif [[ "$time_line" == *"ms"* ]]; then
            printf "%.0f\n" "$(echo "$time_value * 1000000" | bc -l)"
        elif [[ "$time_line" == *" s"* ]] && [[ "$time_line" != *"μs"* ]] && [[ "$time_line" != *"ms"* ]]; then
            printf "%.0f\n" "$(echo "$time_value * 1000000000" | bc -l)"
        else
            echo "$time_value"
        fi
    else
        echo "0"  # Default if not found
    fi
}

# Function to check if benchmark_parser.sh is being sourced or executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "benchmark_parser.sh - Shared utility functions for parsing benchmark data"
    echo "This script is meant to be sourced by other scripts, not executed directly."
    echo ""
    echo "Available functions:"
    echo "  - check_benchmark_parser_dependencies"
    echo "  - parse_benchmark_start"
    echo "  - extract_timing_data"
    echo "  - parse_benchmark_identifier"
    echo "  - format_benchmark_result"
    echo "  - parse_benchmarks_with_while_read"
    echo "  - parse_benchmarks_with_awk"
    echo "  - extract_baseline_time"
    echo ""
    echo "Usage: source scripts/benchmark_parser.sh"
    exit 1
fi

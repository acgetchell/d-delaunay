#!/bin/bash

# Script to extract benchmark results from Criterion JSON files
echo '{"benchmarks": ['

first=true
for estimates_file in $(find ./target/criterion -path "*/new/estimates.json" | sort); do
    # Extract the benchmark path (e.g., tds_new_2d/tds_new/50)
    benchmark_path=$(echo "$estimates_file" | sed 's|./target/criterion/||' | sed 's|/new/estimates.json||')
    
    # Extract mean point estimate from JSON
    mean=$(jq -r '.mean.point_estimate' "$estimates_file")
    
    if [ "$first" = true ]; then
        first=false
    else
        echo ","
    fi
    
    echo -n "  {\"id\": \"$benchmark_path\", \"mean\": $mean, \"alloc_bytes\": null}"
done

echo ''
echo ']}'

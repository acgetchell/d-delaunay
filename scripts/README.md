# Scripts Directory

This directory contains utility scripts for building, testing, and benchmarking the d-delaunay library.

## Prerequisites

Before running these scripts, ensure you have the following dependencies installed:

### macOS (using Homebrew)

```bash
brew install jq findutils coreutils
```

### Ubuntu/Debian

```bash
sudo apt-get install jq findutils coreutils bc
```

### Other Systems

Install equivalent packages for `jq`, `find`, `sort`, and `bc` using your system's package manager.

## Scripts Overview

### Benchmarking Scripts

#### `run_small_scale_benchmarks.sh`

**Purpose**: Executes small-scale triangulation benchmarks and generates performance results.

**Features**:

- Runs 2D, 3D, and 4D triangulation benchmarks with point counts from 10 to 50
- Generates both text and JSON output files
- Automatically creates baseline performance results
- Uses Criterion for statistical benchmarking

**Usage**:

```bash
./scripts/run_small_scale_benchmarks.sh
```

**Output Files**:

- `benches/results/small_scale_benchmarks.txt` - Human-readable benchmark results
- `benches/results/small_scale_benchmarks.json` - JSON-formatted results for analysis
- `benches/baseline_results.txt` - Baseline performance metrics for regression testing

**Dependencies**: Requires `cargo`, `jq`, `find`, `sort`

---

#### `extract_benchmarks.sh`

**Purpose**: Extracts and aggregates benchmark results from Criterion JSON files.

**Features**:

- Parses Criterion's JSON output files from `target/criterion`
- Extracts mean performance values for each benchmark
- Outputs consolidated JSON format for analysis
- Handles scientific notation and validates numeric data

**Usage**:

```bash
./scripts/extract_benchmarks.sh [directory]

# Examples:
./scripts/extract_benchmarks.sh                    # Uses default target/criterion
./scripts/extract_benchmarks.sh target/criterion   # Explicit directory
./scripts/extract_benchmarks.sh > results.json     # Save to file
```

**Dependencies**: Requires `jq`, `find`, `sort`

---

#### `compare_benchmarks.sh`

**Purpose**: Compares current benchmark results against baseline performance metrics.

**Features**:

- Reads baseline values from `benches/baseline_results.txt`
- Compares against current benchmark results JSON
- Identifies performance regressions (>5% threshold)
- Supports scientific notation conversion
- Provides clear pass/fail indicators

**Usage**:

```bash
./scripts/compare_benchmarks.sh [results_directory]

# Examples:
./scripts/compare_benchmarks.sh                      # Uses default benches/results
./scripts/compare_benchmarks.sh /path/to/results     # Custom results directory
```

**Output**:

- ✅ OK: Performance within acceptable range
- ⚠️ REGRESSION: Performance degradation detected

**Dependencies**: Requires `jq`, `bc`

---

#### `generate_baseline.sh`

**Purpose**: Generates baseline performance results from benchmark output.

**Features**:

- Parses `benches/results/small_scale_benchmarks.txt`
- Creates formatted `benches/baseline_results.txt`
- Includes git commit hash and timestamp
- Handles various time units (µs, ms, s)

**Usage**:

```bash
./scripts/generate_baseline.sh
```

**Prerequisites**: Must run `run_small_scale_benchmarks.sh` first to generate source data.

**Dependencies**: Requires `git`, `date`, `grep`, `awk`

---

### Testing Scripts

#### `run_all_examples.sh`

**Purpose**: Executes all example programs in the project to verify functionality.

**Features**:

- Automatically discovers all examples in the `examples/` directory
- Runs simple examples with standard execution
- Provides comprehensive testing for `test_circumsphere` example
- Creates results directory structure

**Usage**:

```bash
./scripts/run_all_examples.sh
```

**Test Categories for `test_circumsphere`**:

- `all` - Basic dimensional tests and orientation tests
- `test-all-points` - Single point tests in all dimensions
- `debug-all` - All debug tests

**Dependencies**: Requires `cargo`, `find`

---

## Workflow Examples

### Performance Regression Testing

```bash
# 1. Run benchmarks and generate baseline
./scripts/run_small_scale_benchmarks.sh

# 2. Make code changes
# ... your modifications ...

# 3. Run benchmarks again
./scripts/run_small_scale_benchmarks.sh

# 4. Compare performance
./scripts/compare_benchmarks.sh
```

### Manual Benchmark Analysis

```bash
# 1. Run benchmarks
cargo bench --bench small_scale_triangulation

# 2. Extract results to JSON
./scripts/extract_benchmarks.sh > current_results.json

# 3. Generate new baseline
./scripts/generate_baseline.sh

# 4. Compare against baseline
./scripts/compare_benchmarks.sh
```

### Continuous Integration

```bash
# Validate all functionality
./scripts/run_all_examples.sh

# Run performance tests
./scripts/run_small_scale_benchmarks.sh

# Check for regressions
./scripts/compare_benchmarks.sh
```

## Error Handling and Troubleshooting

### Common Issues

1. **Missing Dependencies**: Install required packages using your system's package manager
2. **Permission Errors**: Ensure scripts are executable with `chmod +x scripts/*.sh`
3. **Path Issues**: Run scripts from the project root directory
4. **Missing Baseline**: Run `run_small_scale_benchmarks.sh` to generate initial baseline

### Exit Codes

- `0` - Success
- `1` - General error
- `2` - Missing dependency
- `3` - File/directory not found

### Debug Mode

Add `set -x` to any script for verbose execution output:

```bash
bash -x ./scripts/run_small_scale_benchmarks.sh
```

## Script Maintenance

All scripts follow consistent patterns:

- **Error Handling**: Strict mode with `set -euo pipefail`
- **Dependency Checking**: Validation of required commands
- **Usage Information**: Help text with `--help` flag
- **Project Root Detection**: Automatic detection of project directory
- **Error Messages**: Descriptive error output to stderr

When modifying scripts, maintain these patterns for consistency and reliability.

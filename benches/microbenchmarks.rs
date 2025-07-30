//! Microbenchmarks for key d-delaunay methods
//!
//! This benchmark suite focuses on measuring the performance of individual key methods
//! in the d-delaunay triangulation library, particularly those that are performance-critical:
//!
//! 1. **`Tds::new` (Bowyer-Watson triangulation)**: Complete triangulation creation
//! 2. **`assign_neighbors`**: Neighbor relationship assignment between cells
//! 3. **`remove_duplicate_cells`**: Duplicate cell removal and cleanup
//! 4. **Memory usage patterns**: Allocation and deallocation patterns
//!
//! These benchmarks establish baseline performance metrics before optimization.

#![allow(missing_docs)] // Criterion macros generate undocumented functions

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use d_delaunay::prelude::*;
use rand::Rng;
use std::hint::black_box;

/// Creates random points for benchmarking
fn generate_random_points(n_points: usize) -> Vec<Point<f64, 3>> {
    let mut rng = rand::rng();
    (0..n_points)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

/// Benchmark the complete Bowyer-Watson triangulation process via `Tds::new`
fn benchmark_bowyer_watson_triangulation(c: &mut Criterion) {
    let point_counts = [10, 25, 50, 100, 250];

    let mut group = c.benchmark_group("bowyer_watson_triangulation");

    for &n_points in &point_counts {
        #[allow(clippy::cast_sign_loss)]
        let throughput = n_points as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("tds_new", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_random_points(n_points);
                        points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                    },
                    |vertices| black_box(Tds::<f64, (), (), 3>::new(&vertices).unwrap()),
                );
            },
        );
    }

    group.finish();
}

/// Benchmark the `assign_neighbors` method specifically
fn benchmark_assign_neighbors(c: &mut Criterion) {
    let point_counts = [10, 25, 50, 100];

    let mut group = c.benchmark_group("assign_neighbors");

    for &n_points in &point_counts {
        #[allow(clippy::cast_sign_loss)]
        let throughput = n_points as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("assign_neighbors", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_random_points(n_points);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let mut tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();
                        // Clear existing neighbors to benchmark the assignment process
                        for cell in tds.cells.values_mut() {
                            cell.neighbors = None;
                        }
                        tds
                    },
                    |mut tds| {
                        tds.assign_neighbors();
                        black_box(tds);
                    },
                );
            },
        );
    }

    group.finish();
}

/// Benchmark the `remove_duplicate_cells` method
fn benchmark_remove_duplicate_cells(c: &mut Criterion) {
    let point_counts = [10, 25, 50, 100];

    let mut group = c.benchmark_group("remove_duplicate_cells");

    for &n_points in &point_counts {
        #[allow(clippy::cast_sign_loss)]
        let throughput = n_points as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("remove_duplicate_cells", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let points = generate_random_points(n_points);
                        let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                        let mut tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();

                        // Add some duplicate cells to make the benchmark meaningful
                        let cell_vertices: Vec<_> = tds.vertices.values().copied().collect();
                        if cell_vertices.len() >= 4 {
                            // Create a few duplicate cells
                            for _ in 0..3 {
                                let duplicate_cell = cell!(cell_vertices[0..4].to_vec());
                                tds.cells.insert(duplicate_cell);
                            }
                        }
                        tds
                    },
                    |mut tds| {
                        let removed = tds.remove_duplicate_cells();
                        black_box((tds, removed));
                    },
                );
            },
        );
    }

    group.finish();
}

/// Benchmark 2D triangulations for comparison
fn benchmark_2d_triangulation(c: &mut Criterion) {
    let point_counts = [10, 25, 50, 100, 250];

    let mut group = c.benchmark_group("2d_triangulation");

    for &n_points in &point_counts {
        #[allow(clippy::cast_sign_loss)]
        let throughput = n_points as u64;
        group.throughput(Throughput::Elements(throughput));

        group.bench_with_input(
            BenchmarkId::new("tds_new_2d", n_points),
            &n_points,
            |b, &n_points| {
                b.iter_with_setup(
                    || {
                        let mut rng = rand::rng();
                        let points: Vec<Point<f64, 2>> = (0..n_points)
                            .map(|_| {
                                Point::new([
                                    rng.random_range(-100.0..100.0),
                                    rng.random_range(-100.0..100.0),
                                ])
                            })
                            .collect();
                        points.iter().map(|p| vertex!(*p)).collect::<Vec<_>>()
                    },
                    |vertices| black_box(Tds::<f64, (), (), 2>::new(&vertices).unwrap()),
                );
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation patterns
fn benchmark_memory_usage(c: &mut Criterion) {
    let point_counts = [50, 100, 200];

    let mut group = c.benchmark_group("memory_usage");

    for &n_points in &point_counts {
        group.bench_with_input(
            BenchmarkId::new("triangulation_memory", n_points),
            &n_points,
            |b, &n_points| {
                b.iter(|| {
                    // Measure complete triangulation creation and destruction
                    let points = generate_random_points(n_points);
                    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();
                    let tds = Tds::<f64, (), (), 3>::new(&vertices).unwrap();
                    black_box((tds.number_of_vertices(), tds.number_of_cells()))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default();
    targets = benchmark_bowyer_watson_triangulation, benchmark_assign_neighbors, benchmark_remove_duplicate_cells, benchmark_2d_triangulation, benchmark_memory_usage
);
criterion_main!(benches);

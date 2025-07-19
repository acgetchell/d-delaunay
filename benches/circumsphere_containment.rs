//! Benchmarks for circumsphere containment algorithms.
//!
//! This benchmark suite compares the performance of two different algorithms for
//! determining whether a vertex is contained within the circumsphere of a simplex:
//! the standard algorithm and a matrix-based implementation.
//!
//! The benchmarks include:
//! - Basic performance tests with fixed simplices
//! - Random query tests with multiple vertices
//! - Tests across different dimensions (2D, 3D, 4D)
//! - Edge case tests with boundary and distant vertices
//! - Numerical consistency validation between algorithms

#![allow(missing_docs)]

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
use d_delaunay::geometry::point::Point;
use d_delaunay::geometry::predicates::{
    circumsphere_contains_vertex, circumsphere_contains_vertex_matrix,
};
use rand::Rng;

/// Generate a random simplex for benchmarking
fn generate_random_simplex_3d(rng: &mut impl Rng) -> Vec<Vertex<f64, i32, 3>> {
    (0..4)
        .map(|i| {
            let x = rng.random_range(-10.0..10.0);
            let y = rng.random_range(-10.0..10.0);
            let z = rng.random_range(-10.0..10.0);
            VertexBuilder::default()
                .point(Point::new([x, y, z]))
                .data(i)
                .build()
                .unwrap()
        })
        .collect()
}

/// Generate a random test vertex
fn generate_random_test_vertex_3d(rng: &mut impl Rng) -> Vertex<f64, i32, 3> {
    let x = rng.random_range(-5.0..5.0);
    let y = rng.random_range(-5.0..5.0);
    let z = rng.random_range(-5.0..5.0);
    VertexBuilder::default()
        .point(Point::new([x, y, z]))
        .data(999)
        .build()
        .unwrap()
}

/// Benchmark basic circumsphere containment with fixed simplex
fn benchmark_basic_circumsphere_containment(c: &mut Criterion) {
    let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([1.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 1.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 1.0]))
        .data(2)
        .build()
        .unwrap();
    let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
    let test_vertex: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.5, 0.5, 0.5]))
        .data(3)
        .build()
        .unwrap();

    c.bench_function("basic/circumsphere_contains_vertex", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex(black_box(&simplex_vertices), black_box(test_vertex))
                    .unwrap(),
            )
        });
    });

    c.bench_function("basic/circumsphere_contains_vertex_matrix", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex_matrix(
                    black_box(&simplex_vertices),
                    black_box(test_vertex),
                )
                .unwrap(),
            )
        });
    });
}

/// Benchmark with many random queries
fn benchmark_random_queries(c: &mut Criterion) {
    let mut rng = rand::rng();

    // Generate a fixed simplex for consistent benchmarking
    let simplex_vertices = generate_random_simplex_3d(&mut rng);

    // Generate many test vertices
    let test_vertices: Vec<_> = (0..1000)
        .map(|_| generate_random_test_vertex_3d(&mut rng))
        .collect();

    c.bench_function("random/circumsphere_contains_vertex_1000_queries", |b| {
        b.iter(|| {
            for test_vertex in &test_vertices {
                black_box(
                    circumsphere_contains_vertex(
                        black_box(&simplex_vertices),
                        black_box(*test_vertex),
                    )
                    .unwrap(),
                );
            }
        });
    });

    c.bench_function(
        "random/circumsphere_contains_vertex_matrix_1000_queries",
        |b| {
            b.iter(|| {
                for test_vertex in &test_vertices {
                    black_box(
                        circumsphere_contains_vertex_matrix(
                            black_box(&simplex_vertices),
                            black_box(*test_vertex),
                        )
                        .unwrap(),
                    );
                }
            });
        },
    );
}

/// Benchmark with different simplex sizes (2D, 3D, 4D)
fn benchmark_different_dimensions(c: &mut Criterion) {
    let _rng = rand::rng();

    // 2D case
    let vertex1_2d: Vertex<f64, i32, 2> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex2_2d: Vertex<f64, i32, 2> = VertexBuilder::default()
        .point(Point::new([1.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex3_2d: Vertex<f64, i32, 2> = VertexBuilder::default()
        .point(Point::new([0.0, 1.0]))
        .data(1)
        .build()
        .unwrap();
    let simplex_2d = vec![vertex1_2d, vertex2_2d, vertex3_2d];
    let test_vertex_2d: Vertex<f64, i32, 2> = VertexBuilder::default()
        .point(Point::new([0.3, 0.3]))
        .data(3)
        .build()
        .unwrap();

    c.bench_function("2d/circumsphere_contains_vertex", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex(black_box(&simplex_2d), black_box(test_vertex_2d))
                    .unwrap(),
            )
        });
    });

    c.bench_function("2d/circumsphere_contains_vertex_matrix", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex_matrix(
                    black_box(&simplex_2d),
                    black_box(test_vertex_2d),
                )
                .unwrap(),
            )
        });
    });

    // 4D case
    let vertex1_4d: Vertex<f64, i32, 4> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex2_4d: Vertex<f64, i32, 4> = VertexBuilder::default()
        .point(Point::new([1.0, 0.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex3_4d: Vertex<f64, i32, 4> = VertexBuilder::default()
        .point(Point::new([0.0, 1.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex4_4d: Vertex<f64, i32, 4> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 1.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex5_4d: Vertex<f64, i32, 4> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 0.0, 1.0]))
        .data(1)
        .build()
        .unwrap();
    let simplex_4d = vec![vertex1_4d, vertex2_4d, vertex3_4d, vertex4_4d, vertex5_4d];
    let test_vertex_4d: Vertex<f64, i32, 4> = VertexBuilder::default()
        .point(Point::new([0.2, 0.2, 0.2, 0.2]))
        .data(3)
        .build()
        .unwrap();

    c.bench_function("4d/circumsphere_contains_vertex", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex(black_box(&simplex_4d), black_box(test_vertex_4d))
                    .unwrap(),
            )
        });
    });

    c.bench_function("4d/circumsphere_contains_vertex_matrix", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex_matrix(
                    black_box(&simplex_4d),
                    black_box(test_vertex_4d),
                )
                .unwrap(),
            )
        });
    });
}

/// Benchmark edge cases (points on boundary, far away, etc.)
fn benchmark_edge_cases(c: &mut Criterion) {
    let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([1.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 1.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 1.0]))
        .data(2)
        .build()
        .unwrap();
    let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];

    // Test with vertex on the boundary (one of the simplex vertices)
    c.bench_function("edge_cases/boundary_vertex", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex(black_box(&simplex_vertices), black_box(vertex1))
                    .unwrap(),
            )
        });
    });

    c.bench_function("edge_cases/boundary_vertex_matrix", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex_matrix(
                    black_box(&simplex_vertices),
                    black_box(vertex1),
                )
                .unwrap(),
            )
        });
    });

    // Test with far away vertex
    let far_vertex: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([1000.0, 1000.0, 1000.0]))
        .data(3)
        .build()
        .unwrap();
    c.bench_function("edge_cases/far_vertex", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex(black_box(&simplex_vertices), black_box(far_vertex))
                    .unwrap(),
            )
        });
    });

    c.bench_function("edge_cases/far_vertex_matrix", |b| {
        b.iter(|| {
            black_box(
                circumsphere_contains_vertex_matrix(
                    black_box(&simplex_vertices),
                    black_box(far_vertex),
                )
                .unwrap(),
            )
        });
    });
}

/// Numerical consistency test - compare results of both methods
fn numerical_consistency_test() {
    println!("\n=== Numerical Consistency Test ===");
    let mut rng = rand::rng();
    let mut matches = 0;
    let mut total = 0;
    let mut disagreements = Vec::new();

    for _ in 0..1000 {
        let simplex_vertices = generate_random_simplex_3d(&mut rng);
        let test_vertex = generate_random_test_vertex_3d(&mut rng);

        let result1 = circumsphere_contains_vertex(&simplex_vertices, test_vertex);
        let result2 = circumsphere_contains_vertex_matrix(&simplex_vertices, test_vertex);

        if let (Ok(r1), Ok(r2)) = (result1, result2) {
            total += 1;
            if r1 == r2 {
                matches += 1;
            } else {
                disagreements.push((simplex_vertices, test_vertex, r1, r2));
            }
        }
    }

    println!(
        "Consistency: {}/{} ({:.2}%)",
        matches,
        total,
        (f64::from(matches) / f64::from(total)) * 100.0
    );

    if !disagreements.is_empty() {
        println!("Found {} disagreements:", disagreements.len());
        for (i, (simplex, test, r1, r2)) in disagreements.iter().take(5).enumerate() {
            println!("  Disagreement {}: standard={}, matrix={}", i + 1, r1, r2);
            println!("    Test point: {:?}", test.point().coordinates());
            println!(
                "    Simplex: {:?}",
                simplex
                    .iter()
                    .map(|v| v.point().coordinates())
                    .collect::<Vec<_>>()
            );
        }
    }
}

/// Main benchmark function that runs consistency test before benchmarks
fn benchmark_with_consistency_check(c: &mut Criterion) {
    // Run consistency test first
    numerical_consistency_test();

    // Then run benchmarks
    benchmark_basic_circumsphere_containment(c);
    benchmark_random_queries(c);
    benchmark_different_dimensions(c);
    benchmark_edge_cases(c);
}

criterion_group!(benches, benchmark_with_consistency_check);
criterion_main!(benches);

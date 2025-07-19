//! # Circumsphere Containment Test Example
//!
//! This example demonstrates and compares three methods for testing whether a point
//! lies inside the circumsphere of a simplex in 2D, 3D, and 4D:
//!
//! 1. **`insphere`**: Standard determinant-based method using matrix determinants
//! 2. **`insphere_distance`**: Distance-based method that computes circumcenter and circumradius
//! 3. **`insphere_lifted`**: Lifted coordinate method using a different matrix formulation
//!
//! The example systematically tests each dimension with:
//! - Unit simplexes (triangle, tetrahedron, 4D simplex)
//! - Various test points (inside, outside, boundary, vertices)
//! - Comparison of all three methods
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example test_circumsphere [2d|3d|4d|all|help]
//! ```

use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
use d_delaunay::geometry::Point;
use d_delaunay::geometry::predicates::{InSphere, Orientation};
use d_delaunay::geometry::predicates::{
    circumcenter, circumradius, insphere, insphere_distance, insphere_lifted, simplex_orientation,
};
use nalgebra as na;
use peroxide::fuga::{LinearAlgebra, zeros};
use serde::{Serialize, de::DeserializeOwned};
use std::env;

/// Create a test vertex with generic dimension and data type
fn create_vertex<T, const D: usize>(coords: [f64; D], data: T) -> Vertex<f64, T, D>
where
    [f64; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    T: Clone
        + Copy
        + Eq
        + std::hash::Hash
        + Ord
        + PartialEq
        + PartialOrd
        + DeserializeOwned
        + Serialize,
{
    VertexBuilder::default()
        .point(Point::new(coords))
        .data(data)
        .build()
        .unwrap()
}

/// Convert `InSphere` result to string representation
fn insphere_to_string(result: &Result<InSphere, anyhow::Error>) -> String {
    match result {
        Ok(InSphere::INSIDE) => "INSIDE".to_string(),
        Ok(InSphere::BOUNDARY) => "BOUNDARY".to_string(),
        Ok(InSphere::OUTSIDE) => "OUTSIDE".to_string(),
        Err(_) => "ERROR".to_string(),
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "2d" => test_2d_circumsphere(),
            "3d" => test_3d_circumsphere(),
            "4d" => test_4d_circumsphere(),
            "orientation" => test_all_orientations(),
            "debug-3d" => test_3d_simplex_analysis(),
            "debug-3d-matrix" => test_3d_matrix_analysis(),
            "debug-3d-properties" => debug_3d_circumsphere_properties(),
            "debug-4d-properties" => debug_4d_circumsphere_properties(),
            "debug-compare" => compare_circumsphere_methods(),
            "debug-orientation" => demonstrate_orientation_impact_on_circumsphere(),
            "debug-containment" => test_circumsphere_containment(),
            "debug-4d-methods" => test_4d_circumsphere_methods(),
            "debug-all" => {
                println!("Running all debug tests...\n");
                test_3d_simplex_analysis();
                debug_4d_circumsphere_properties();
                test_3d_matrix_analysis();
                debug_3d_circumsphere_properties();
                compare_circumsphere_methods();
                demonstrate_orientation_impact_on_circumsphere();
                test_circumsphere_containment();
                test_4d_circumsphere_methods();
            }
            "all" => {
                test_2d_circumsphere();
                test_3d_circumsphere();
                test_4d_circumsphere();
                test_all_orientations();
            }
            "everything" => {
                println!("Running all tests and debug functions...\n");
                test_2d_circumsphere();
                test_3d_circumsphere();
                test_4d_circumsphere();
                test_all_orientations();
                test_3d_simplex_analysis();
                debug_4d_circumsphere_properties();
                test_3d_matrix_analysis();
                debug_3d_circumsphere_properties();
                compare_circumsphere_methods();
                demonstrate_orientation_impact_on_circumsphere();
                test_circumsphere_containment();
                test_4d_circumsphere_methods();
            }
            "help" | "--help" | "-h" => print_help(),
            _ => {
                println!("Unknown argument: {}", args[1]);
                print_help();
            }
        }
    } else {
        print_help();
    }
}

fn print_help() {
    println!("Circumsphere Containment Test Suite");
    println!("=====================================");
    println!();
    println!("Usage: cargo run --example test_circumsphere [TEST]");
    println!();
    println!("Basic tests:");
    println!("  2d          - Test 2D circumsphere methods (triangle)");
    println!("  3d          - Test 3D circumsphere methods (tetrahedron)");
    println!("  4d          - Test 4D circumsphere methods (4D simplex)");
    println!("  orientation - Test simplex orientation in 2D, 3D, and 4D");
    println!("  all         - Run all basic dimensional tests and orientation tests");
    println!();
    println!("Debug tests:");
    println!("  debug-3d           - Detailed 3D simplex analysis and debugging");
    println!("  debug-3d-matrix    - Step-by-step 3D matrix method analysis");
    println!("  debug-3d-properties - 3D circumsphere properties analysis");
    println!("  debug-4d-properties - 4D circumsphere properties analysis");
    println!("  debug-compare      - Compare circumsphere methods across dimensions");
    println!("  debug-orientation  - Demonstrate orientation impact on circumsphere");
    println!("  debug-containment  - Detailed circumsphere containment testing");
    println!("  debug-4d-methods   - Compare 4D circumsphere methods in detail");
    println!("  debug-all          - Run all debug tests");
    println!();
    println!("Comprehensive:");
    println!("  everything  - Run all tests and all debug functions");
    println!("  help        - Show this help message");
    println!();
    println!("Examples:");
    println!("  cargo run --example test_circumsphere 2d");
    println!("  cargo run --example test_circumsphere debug-3d-matrix");
    println!("  cargo run --example test_circumsphere debug-3d-properties");
    println!("  cargo run --example test_circumsphere debug-all");
    println!("  cargo run --example test_circumsphere everything");
}

/// Test 2D circumsphere methods with a triangle
fn test_2d_circumsphere() {
    println!("=============================================");
    println!("Testing 2D circumsphere methods (triangle)");
    println!("=============================================");

    // Create a unit right triangle: (0,0), (1,0), (0,1)
    let vertices = vec![
        create_vertex([0.0, 0.0], 0),
        create_vertex([1.0, 0.0], 1),
        create_vertex([0.0, 1.0], 2),
    ];

    println!("2D triangle vertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        let coords: [f64; 2] = vertex.into();
        println!("  v{}: [{}, {}]", i, coords[0], coords[1]);
    }
    println!();

    // Calculate circumcenter and circumradius for reference
    match circumcenter(&vertices) {
        Ok(center) => {
            println!("Circumcenter: {:?}", center.coordinates());
            match circumradius(&vertices) {
                Ok(radius) => {
                    println!("Circumradius: {radius:.6}");
                    println!();

                    // Test various 2D points
                    let test_points = vec![
                        ([0.5, 0.5], "circumcenter_region"),
                        ([0.1, 0.1], "clearly_inside"),
                        ([0.9, 0.9], "possibly_outside"),
                        ([2.0, 2.0], "far_outside"),
                        ([0.0, 0.0], "vertex_origin"),
                        ([0.5, 0.0], "edge_midpoint"),
                        ([0.25, 0.25], "inside_triangle"),
                    ];

                    for (coords, description) in test_points {
                        test_2d_point(
                            &vertices,
                            coords,
                            description,
                            &center.coordinates(),
                            radius,
                        );
                    }
                }
                Err(e) => println!("Error calculating circumradius: {e}"),
            }
        }
        Err(e) => println!("Error calculating circumcenter: {e}"),
    }

    println!("2D circumsphere testing completed.\n");
}

/// Test a single 2D point against all circumsphere methods
fn test_2d_point(
    vertices: &[Vertex<f64, i32, 2>],
    coords: [f64; 2],
    description: &str,
    center: &[f64; 2],
    radius: f64,
) {
    let test_vertex = create_vertex(coords, 99);

    let result_insphere = insphere(vertices, test_vertex);
    let result_distance = insphere_distance(vertices, test_vertex);
    let result_lifted = insphere_lifted(vertices, test_vertex);

    // Calculate actual distance to center
    let distance_to_center = {
        let diff = [coords[0] - center[0], coords[1] - center[1]];
        (diff[0] * diff[0] + diff[1] * diff[1]).sqrt()
    };

    println!("Point {description} {coords:?}:");
    println!("  Distance to center: {distance_to_center:.6}");
    println!("  Expected inside: {}", distance_to_center <= radius);
    println!(
        "  insphere:          {}",
        insphere_to_string(&result_insphere)
    );
    println!(
        "  insphere_distance: {}",
        insphere_to_string(&result_distance)
    );
    println!(
        "  insphere_lifted:   {}",
        insphere_to_string(&result_lifted)
    );

    // Check agreement between methods
    let methods_agree =
        if let (Ok(r1), Ok(r2), Ok(r3)) = (&result_insphere, &result_distance, &result_lifted) {
            r1 == r2 && r2 == r3
        } else {
            false
        };

    if methods_agree {
        println!("  ✓ All methods agree");
    } else {
        println!("  ⚠ Methods disagree");
    }
    println!();
}

/// Test 3D circumsphere methods with a tetrahedron
fn test_3d_circumsphere() {
    println!("=============================================");
    println!("Testing 3D circumsphere methods (tetrahedron)");
    println!("=============================================");

    // Create a unit tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let vertices = vec![
        create_vertex([0.0, 0.0, 0.0], 0),
        create_vertex([1.0, 0.0, 0.0], 1),
        create_vertex([0.0, 1.0, 0.0], 2),
        create_vertex([0.0, 0.0, 1.0], 3),
    ];

    println!("3D tetrahedron vertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        let coords: [f64; 3] = vertex.into();
        println!("  v{}: [{}, {}, {}]", i, coords[0], coords[1], coords[2]);
    }
    println!();

    // Calculate circumcenter and circumradius for reference
    match circumcenter(&vertices) {
        Ok(center) => {
            println!("Circumcenter: {:?}", center.coordinates());
            match circumradius(&vertices) {
                Ok(radius) => {
                    println!("Circumradius: {radius:.6}");
                    println!();

                    // Test various 3D points
                    let test_points = vec![
                        ([0.5, 0.5, 0.5], "circumcenter_region"),
                        ([0.1, 0.1, 0.1], "clearly_inside"),
                        ([0.9, 0.9, 0.9], "possibly_outside"),
                        ([2.0, 2.0, 2.0], "far_outside"),
                        ([0.0, 0.0, 0.0], "vertex_origin"),
                        ([0.25, 0.25, 0.0], "face_center"),
                        ([0.2, 0.2, 0.2], "inside_tetrahedron"),
                    ];

                    for (coords, description) in test_points {
                        test_3d_point(
                            &vertices,
                            coords,
                            description,
                            &center.coordinates(),
                            radius,
                        );
                    }
                }
                Err(e) => println!("Error calculating circumradius: {e}"),
            }
        }
        Err(e) => println!("Error calculating circumcenter: {e}"),
    }

    println!("3D circumsphere testing completed.\n");
}

/// Test a single 3D point against all circumsphere methods
fn test_3d_point(
    vertices: &[Vertex<f64, i32, 3>],
    coords: [f64; 3],
    description: &str,
    center: &[f64; 3],
    radius: f64,
) {
    let test_vertex = create_vertex(coords, 99);

    let result_insphere = insphere(vertices, test_vertex);
    let result_distance = insphere_distance(vertices, test_vertex);
    let result_lifted = insphere_lifted(vertices, test_vertex);

    // Calculate actual distance to center
    let distance_to_center = {
        let diff = [
            coords[0] - center[0],
            coords[1] - center[1],
            coords[2] - center[2],
        ];
        (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt()
    };

    println!("Point {description} {coords:?}:");
    println!("  Distance to center: {distance_to_center:.6}");
    println!("  Expected inside: {}", distance_to_center <= radius);
    println!(
        "  insphere:          {}",
        insphere_to_string(&result_insphere)
    );
    println!(
        "  insphere_distance: {}",
        insphere_to_string(&result_distance)
    );
    println!(
        "  insphere_lifted:   {}",
        insphere_to_string(&result_lifted)
    );

    // Check agreement between methods
    let methods_agree =
        if let (Ok(r1), Ok(r2), Ok(r3)) = (&result_insphere, &result_distance, &result_lifted) {
            r1 == r2 && r2 == r3
        } else {
            false
        };

    if methods_agree {
        println!("  ✓ All methods agree");
    } else {
        println!("  ⚠ Methods disagree");
    }
    println!();
}

/// Test 4D circumsphere methods with a 4-simplex
fn test_4d_circumsphere() {
    println!("=============================================");
    println!("Testing 4D circumsphere methods (4-simplex)");
    println!("=============================================");

    // Create a unit 4-simplex: vertices at origin and unit vectors along each axis
    let vertices = vec![
        create_vertex([0.0, 0.0, 0.0, 0.0], 0),
        create_vertex([1.0, 0.0, 0.0, 0.0], 1),
        create_vertex([0.0, 1.0, 0.0, 0.0], 2),
        create_vertex([0.0, 0.0, 1.0, 0.0], 3),
        create_vertex([0.0, 0.0, 0.0, 1.0], 4),
    ];

    println!("4D simplex vertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        let coords: [f64; 4] = vertex.into();
        println!(
            "  v{}: [{}, {}, {}, {}]",
            i, coords[0], coords[1], coords[2], coords[3]
        );
    }
    println!();

    // Calculate circumcenter and circumradius for reference
    match circumcenter(&vertices) {
        Ok(center) => {
            println!("Circumcenter: {:?}", center.coordinates());
            match circumradius(&vertices) {
                Ok(radius) => {
                    println!("Circumradius: {radius:.6}");
                    println!();

                    // Test various 4D points
                    let test_points = vec![
                        ([0.5, 0.5, 0.5, 0.5], "circumcenter"),
                        ([0.1, 0.1, 0.1, 0.1], "clearly_inside"),
                        ([0.9, 0.9, 0.9, 0.9], "possibly_outside"),
                        ([2.0, 2.0, 2.0, 2.0], "far_outside"),
                        ([0.0, 0.0, 0.0, 0.0], "vertex_origin"),
                        ([0.25, 0.0, 0.0, 0.0], "axis_point"),
                        ([0.2, 0.2, 0.2, 0.2], "inside_simplex"),
                    ];

                    for (coords, description) in test_points {
                        test_4d_point(
                            &vertices,
                            coords,
                            description,
                            &center.coordinates(),
                            radius,
                        );
                    }
                }
                Err(e) => println!("Error calculating circumradius: {e}"),
            }
        }
        Err(e) => println!("Error calculating circumcenter: {e}"),
    }

    println!("4D circumsphere testing completed.\n");
}

/// Test a single 4D point against all circumsphere methods
fn test_4d_point(
    vertices: &[Vertex<f64, i32, 4>],
    coords: [f64; 4],
    description: &str,
    center: &[f64; 4],
    radius: f64,
) {
    let test_vertex = create_vertex(coords, 99);

    let result_insphere = insphere(vertices, test_vertex);
    let result_distance = insphere_distance(vertices, test_vertex);
    let result_lifted = insphere_lifted(vertices, test_vertex);

    // Calculate actual distance to center
    let distance_to_center = {
        let diff = [
            coords[0] - center[0],
            coords[1] - center[1],
            coords[2] - center[2],
            coords[3] - center[3],
        ];
        (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] + diff[3] * diff[3]).sqrt()
    };

    println!("Point {description} {coords:?}:");
    println!("  Distance to center: {distance_to_center:.6}");
    println!("  Expected inside: {}", distance_to_center <= radius);
    println!(
        "  insphere:          {}",
        insphere_to_string(&result_insphere)
    );
    println!(
        "  insphere_distance: {}",
        insphere_to_string(&result_distance)
    );
    println!(
        "  insphere_lifted:   {}",
        insphere_to_string(&result_lifted)
    );

    // Check agreement between methods
    let methods_agree =
        if let (Ok(r1), Ok(r2), Ok(r3)) = (&result_insphere, &result_distance, &result_lifted) {
            r1 == r2 && r2 == r3
        } else {
            false
        };

    if methods_agree {
        println!("  ✓ All methods agree");
    } else {
        println!("  ⚠ Methods disagree");
    }
    println!();
}

/// Run all orientation tests for 2D, 3D, and 4D
fn test_all_orientations() {
    println!("=============================================");
    println!("Testing 2D, 3D, and 4D simplex orientations");
    println!("=============================================");
    test_simplex_orientation();
    println!("Orientation tests completed\n");
}

/// Helper function to test and print a point for compatibility with existing code
fn test_and_print_point(vertices: &[Vertex<f64, i32, 4>], coords: [f64; 4], description: &str) {
    let test_vertex = create_vertex(coords, 99);

    let result_insphere = insphere(vertices, test_vertex);
    let result_distance = insphere_distance(vertices, test_vertex);
    let result_lifted = insphere_lifted(vertices, test_vertex);

    println!("Testing point {description} {coords:?}:");
    println!(
        "  insphere:          {}",
        insphere_to_string(&result_insphere)
    );
    println!(
        "  insphere_distance: {}",
        insphere_to_string(&result_distance)
    );
    println!(
        "  insphere_lifted:   {}",
        insphere_to_string(&result_lifted)
    );

    // Check agreement between methods
    let methods_agree =
        if let (Ok(r1), Ok(r2), Ok(r3)) = (&result_insphere, &result_distance, &result_lifted) {
            r1 == r2 && r2 == r3
        } else {
            false
        };

    if methods_agree {
        println!("  ✓ All methods agree");
    } else {
        println!("  ⚠ Methods disagree");
    }
    println!();
}

/// Test and compare both 4D circumsphere containment methods
fn test_4d_circumsphere_methods() {
    println!("=============================================");
    println!("Testing 4D circumsphere containment methods:");
    println!("  circumsphere_contains_vertex vs circumsphere_contains_vertex_matrix");
    println!("=============================================");

    // Create a unit 4-simplex: vertices at origin and unit vectors along each axis
    let vertices: Vec<Vertex<f64, i32, 4>> = vec![
        create_vertex([0.0, 0.0, 0.0, 0.0], 1),
        create_vertex([1.0, 0.0, 0.0, 0.0], 1),
        create_vertex([0.0, 1.0, 0.0, 0.0], 1),
        create_vertex([0.0, 0.0, 1.0, 0.0], 1),
        create_vertex([0.0, 0.0, 0.0, 1.0], 2),
    ];

    // Calculate circumcenter and circumradius for reference
    match circumcenter(&vertices) {
        Ok(center) => {
            println!("Circumcenter: {:?}", center.coordinates());
            match circumradius(&vertices) {
                Ok(radius) => {
                    println!("Circumradius: {radius}");
                    println!();

                    // Test various points
                    let test_points = vec![
                        ([0.5, 0.5, 0.5, 0.5], "circumcenter"),
                        ([0.1, 0.1, 0.1, 0.1], "clearly_inside"),
                        ([0.9, 0.9, 0.9, 0.9], "possibly_outside"),
                        ([10.0, 10.0, 10.0, 10.0], "far_outside"),
                        ([0.0, 0.0, 0.0, 0.0], "origin"),
                        ([0.25, 0.0, 0.0, 0.0], "axis_point"),
                        ([0.5, 0.5, 0.0, 0.0], "equidistant"),
                    ];

                    for (coords, description) in test_points {
                        test_and_print_point(vertices.as_slice(), coords, description);
                        let test_vertex = create_vertex(coords, 99);

                        let result_standard = insphere(&vertices, test_vertex);
                        let result_matrix = insphere_lifted(&vertices, test_vertex);

                        // Calculate actual distance to center
                        let distance_to_center = {
                            let diff = [
                                coords[0] - center.coordinates()[0],
                                coords[1] - center.coordinates()[1],
                                coords[2] - center.coordinates()[2],
                                coords[3] - center.coordinates()[3],
                            ];
                            (diff[0] * diff[0]
                                + diff[1] * diff[1]
                                + diff[2] * diff[2]
                                + diff[3] * diff[3])
                                .sqrt()
                        };

                        println!("Point {description} {coords:?}:");
                        println!("  Distance to center: {distance_to_center:.6}");
                        println!("  Expected inside: {}", distance_to_center <= radius);
                        println!(
                            "  Standard method: {:?}",
                            result_standard.as_ref().map(|s| match s {
                                InSphere::INSIDE => "INSIDE",
                                InSphere::BOUNDARY => "BOUNDARY",
                                InSphere::OUTSIDE => "OUTSIDE",
                            })
                        );
                        println!(
                            "  Matrix method: {:?}",
                            result_matrix.as_ref().map(|s| match s {
                                InSphere::INSIDE => "INSIDE",
                                InSphere::BOUNDARY => "BOUNDARY",
                                InSphere::OUTSIDE => "OUTSIDE",
                            })
                        );

                        // Check if methods agree
                        if result_standard.is_ok() && result_matrix.is_ok() {
                            // Note: We can't directly compare InSphere enum with bool
                            // Convert both to simple inside/outside comparison
                            let standard_inside =
                                matches!(result_standard.as_ref().unwrap(), InSphere::INSIDE);
                            let matrix_inside =
                                matches!(result_matrix.as_ref().unwrap(), InSphere::INSIDE);
                            let agree = standard_inside == matrix_inside;
                            println!("  Methods agree: {agree}");
                            if !agree {
                                println!("  *** DISAGREEMENT DETECTED ***");
                                println!(
                                    "  NOTE: The matrix method uses an inverted sign convention for this specific simplex"
                                );
                                println!(
                                    "        geometry, which may cause different results for some test cases."
                                );
                            }
                        }
                        println!();
                    }
                }
                Err(e) => println!("Error calculating circumradius: {e}"),
            }
        }
        Err(e) => println!("Error calculating circumcenter: {e}"),
    }

    println!("4D method comparison completed.\n");
}

#[allow(clippy::too_many_lines)]
fn test_circumsphere_containment() {
    println!("Testing circumsphere containment:");
    println!("  determinant-based (insphere) vs distance-based (circumsphere_contains)");
    println!("=============================================");

    // Define the 4D simplex vertices that form a unit 5-cell
    // This creates a 4D simplex with vertices at the origin and unit vectors
    // along each coordinate axis. The circumcenter is at [0.5, 0.5, 0.5, 0.5] and
    // the circumradius is √4/2 = 1.0.
    let vertices: [Vertex<f64, i32, 4>; 5] = [
        create_vertex([0.0, 0.0, 0.0, 0.0], 0), // Origin
        create_vertex([1.0, 0.0, 0.0, 0.0], 1), // Unit vector along x-axis
        create_vertex([0.0, 1.0, 0.0, 0.0], 2), // Unit vector along y-axis
        create_vertex([0.0, 0.0, 1.0, 0.0], 3), // Unit vector along z-axis
        create_vertex([0.0, 0.0, 0.0, 1.0], 4), // Unit vector along w-axis
    ];

    println!("4D simplex vertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        let coords: [f64; 4] = vertex.into();
        println!(
            "  v{}: [{}, {}, {}, {}]",
            i, coords[0], coords[1], coords[2], coords[3]
        );
    }
    println!();

    // Test points that should be inside the circumsphere
    // These are points with small coordinates that should be well within
    // the circumsphere radius of √4/2 = 1.0
    let test_points_inside: [Vertex<f64, i32, 4>; 5] = [
        create_vertex([0.25, 0.25, 0.25, 0.25], 10),
        create_vertex([0.1, 0.1, 0.1, 0.1], 11),
        create_vertex([0.2, 0.2, 0.2, 0.2], 12),
        create_vertex([0.3, 0.2, 0.1, 0.0], 13),
        create_vertex([0.0, 0.0, 0.0, 0.0], 14), // Origin should be inside
    ];

    println!("Testing points that should be INSIDE the circumsphere:");
    for point in test_points_inside {
        let coords: [f64; 4] = point.into();
        test_and_print_point(vertices.as_slice(), coords, "inside");
    }
    println!();

    // Test points that should be outside the circumsphere
    // These include points with large coordinates and points along the axes
    // that extend beyond the simplex vertices
    let test_points_outside: [Vertex<f64, i32, 4>; 6] = [
        create_vertex([2.0, 2.0, 2.0, 2.0], 20),
        create_vertex([1.0, 1.0, 1.0, 1.0], 21),
        create_vertex([0.8, 0.8, 0.8, 0.8], 22),
        create_vertex([1.5, 0.0, 0.0, 0.0], 23),
        create_vertex([0.0, 1.5, 0.0, 0.0], 24),
        create_vertex([0.0, 0.0, 1.5, 0.0], 25),
    ];

    println!("Testing points that should be OUTSIDE the circumsphere:");
    for (i, point) in test_points_outside.iter().enumerate() {
        let result_determinant = insphere(&vertices, *point);
        let result_distance = insphere_distance(&vertices, *point);
        let coords: [f64; 4] = point.into();
        println!(
            "  Point {}: [{}, {}, {}, {}] -> Det: {}, Dist: {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            match result_determinant {
                Ok(InSphere::INSIDE) => "INSIDE",
                Ok(InSphere::BOUNDARY) => "BOUNDARY",
                Ok(InSphere::OUTSIDE) => "OUTSIDE",
                Err(_) => "ERROR",
            },
            match result_distance {
                Ok(InSphere::INSIDE) => "INSIDE",
                Ok(InSphere::BOUNDARY) => "BOUNDARY",
                Ok(InSphere::OUTSIDE) => "OUTSIDE",
                Err(_) => "ERROR",
            }
        );
    }
    println!();

    // Test edge cases - points on the simplex vertices themselves
    // These should be on the boundary of the circumsphere (distance = radius)
    println!("Testing the simplex vertices themselves:");
    for (i, vertex) in vertices.iter().enumerate() {
        let result = insphere(&vertices, *vertex);
        let coords: [f64; 4] = vertex.into();
        println!(
            "  Vertex {}: [{}, {}, {}, {}] -> {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            match result {
                Ok(InSphere::INSIDE) => "INSIDE",
                Ok(InSphere::BOUNDARY) => "BOUNDARY",
                Ok(InSphere::OUTSIDE) => "OUTSIDE",
                Err(_) => "ERROR",
            }
        );
    }
    println!();

    // Additional boundary testing with points on edges and faces of the tetrahedron
    // These points lie on the boundary of the 4D simplex and test
    // numerical stability near the boundary
    let boundary_points: [Vertex<f64, i32, 4>; 5] = [
        create_vertex([0.5, 0.5, 0.0, 0.0], 30),
        create_vertex([0.5, 0.0, 0.5, 0.0], 31),
        create_vertex([0.0, 0.5, 0.5, 0.0], 32),
        create_vertex([0.33, 0.33, 0.33, 0.01], 33),
        create_vertex([0.25, 0.25, 0.25, 0.25], 34),
    ];

    println!("Testing boundary/edge points:");
    for (i, point) in boundary_points.iter().enumerate() {
        let result = insphere(&vertices, *point);
        let coords: [f64; 4] = point.into();
        println!(
            "  Point {}: [{}, {}, {}, {}] -> {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            match result {
                Ok(InSphere::INSIDE) => "INSIDE",
                Ok(InSphere::BOUNDARY) => "BOUNDARY",
                Ok(InSphere::OUTSIDE) => "OUTSIDE",
                Err(_) => "ERROR",
            }
        );
    }
}

fn test_simplex_orientation() {
    println!("\n=============================================");
    println!("Testing simplex orientation:");
    println!("=============================================");

    // Create vertices for testing
    let vertices = create_unit_4d_simplex();

    // Test the original 4D simplex's orientation
    let orientation_original = simplex_orientation(&vertices);
    println!(
        "Original 4D simplex orientation: {}",
        match orientation_original {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Create a negatively oriented 4D simplex by swapping two vertices
    let vertices_negative = create_negative_4d_simplex();

    let orientation_negative = simplex_orientation(&vertices_negative);
    println!(
        "Negatively oriented 4D simplex: {}",
        match orientation_negative {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Test 3D orientation (tetrahedron) for comparison
    let tetrahedron_vertices: [Vertex<f64, i32, 3>; 4] = [
        create_vertex([0.0, 0.0, 0.0], 0), // Origin
        create_vertex([1.0, 0.0, 0.0], 1), // Unit vector along x-axis
        create_vertex([0.0, 1.0, 0.0], 2), // Unit vector along y-axis
        create_vertex([0.0, 0.0, 1.0], 3), // Unit vector along z-axis
    ];

    let orientation_3d = simplex_orientation(&tetrahedron_vertices);
    println!(
        "3D tetrahedron orientation: {}",
        match orientation_3d {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Test 2D orientation (triangle)
    let triangle_vertices: [Vertex<f64, i32, 2>; 3] = [
        create_vertex([0.0, 0.0], 0),
        create_vertex([1.0, 0.0], 1),
        create_vertex([0.0, 1.0], 2),
    ];

    let orientation_2d = simplex_orientation(&triangle_vertices);
    println!(
        "2D triangle orientation: {}",
        match orientation_2d {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Test 2D orientation with reversed vertex order
    let triangle_vertices_reversed: [Vertex<f64, i32, 2>; 3] = [
        create_vertex([0.0, 0.0], 0),
        create_vertex([0.0, 1.0], 2), // Swapped order
        create_vertex([1.0, 0.0], 1), // Swapped order
    ];

    let orientation_2d_reversed = simplex_orientation(&triangle_vertices_reversed);
    println!(
        "2D triangle (reversed order): {}",
        match orientation_2d_reversed {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );

    // Test degenerate case (collinear points in 2D)
    let collinear_vertices: [Vertex<f64, i32, 2>; 3] = [
        create_vertex([0.0, 0.0], 0),
        create_vertex([1.0, 0.0], 1),
        create_vertex([2.0, 0.0], 2), // Collinear point
    ];

    let orientation_collinear = simplex_orientation(&collinear_vertices);
    println!(
        "Collinear 2D points: {}",
        match orientation_collinear {
            Ok(Orientation::POSITIVE) => "POSITIVE",
            Ok(Orientation::NEGATIVE) => "NEGATIVE",
            Ok(Orientation::DEGENERATE) => "DEGENERATE",
            Err(_) => "ERROR",
        }
    );
}

fn create_unit_4d_simplex() -> [Vertex<f64, i32, 4>; 5] {
    [
        create_vertex([0.0, 0.0, 0.0, 0.0], 0),
        create_vertex([1.0, 0.0, 0.0, 0.0], 1),
        create_vertex([0.0, 1.0, 0.0, 0.0], 2),
        create_vertex([0.0, 0.0, 1.0, 0.0], 3),
        create_vertex([0.0, 0.0, 0.0, 1.0], 4),
    ]
}

fn create_negative_4d_simplex() -> [Vertex<f64, i32, 4>; 5] {
    [
        create_vertex([0.0, 0.0, 0.0, 0.0], 0),
        create_vertex([0.0, 1.0, 0.0, 0.0], 2),
        create_vertex([1.0, 0.0, 0.0, 0.0], 1),
        create_vertex([0.0, 0.0, 1.0, 0.0], 3),
        create_vertex([0.0, 0.0, 0.0, 1.0], 4),
    ]
}

fn demonstrate_orientation_impact_on_circumsphere() {
    println!("\n--- Impact of orientation on circumsphere testing ---");

    // Create vertices for testing
    let vertices = create_unit_4d_simplex();
    let vertices_negative = create_negative_4d_simplex();

    let test_point = create_vertex([0.25, 0.25, 0.25, 0.25], 100);

    let inside_positive = insphere(&vertices, test_point);
    let inside_negative = insphere(&vertices_negative, test_point);

    println!(
        "Point [0.25, 0.25, 0.25, 0.25] in positive 4D simplex: {}",
        match inside_positive {
            Ok(InSphere::INSIDE) => "INSIDE",
            Ok(InSphere::BOUNDARY) => "BOUNDARY",
            Ok(InSphere::OUTSIDE) => "OUTSIDE",
            Err(_) => "ERROR",
        }
    );
    println!(
        "Point [0.25, 0.25, 0.25, 0.25] in negative 4D simplex: {}",
        match inside_negative {
            Ok(InSphere::INSIDE) => "INSIDE",
            Ok(InSphere::BOUNDARY) => "BOUNDARY",
            Ok(InSphere::OUTSIDE) => "OUTSIDE",
            Err(_) => "ERROR",
        }
    );

    println!("\nNote: The insphere function automatically handles");
    println!("      orientation by calling simplex_orientation internally.");
    println!("      Both results should be the same regardless of vertex ordering!");

    println!("\nTest completed!");
}

fn test_3d_simplex_analysis() {
    println!("\n=============================================");
    println!("3D Simplex Analysis for Test Debugging");
    println!("=============================================");

    let simplex_vertices_3d = create_3d_test_simplex();

    match circumcenter(&simplex_vertices_3d) {
        Ok(circumcenter_3d) => match circumradius(&simplex_vertices_3d) {
            Ok(circumradius_3d) => {
                print_3d_simplex_info(&circumcenter_3d, circumradius_3d);
                test_point_against_3d_simplex(
                    &simplex_vertices_3d,
                    &circumcenter_3d,
                    circumradius_3d,
                );
            }
            Err(e) => println!("Error calculating circumradius: {e}"),
        },
        Err(e) => println!("Error calculating circumcenter: {e}"),
    }
}

fn create_3d_test_simplex() -> Vec<Vertex<f64, i32, 3>> {
    let vertex1_3d: Vertex<f64, i32, 3> = create_vertex([0.0, 0.0, 0.0], 1);
    let vertex2_3d = create_vertex([1.0, 0.0, 0.0], 1);
    let vertex3_3d = create_vertex([0.0, 1.0, 0.0], 1);
    let vertex4_3d = create_vertex([0.0, 0.0, 1.0], 2);
    vec![vertex1_3d, vertex2_3d, vertex3_3d, vertex4_3d]
}

fn print_3d_simplex_info(circumcenter_3d: &Point<f64, 3>, circumradius_3d: f64) {
    println!("3D Simplex vertices:");
    println!("  v1: (0, 0, 0)");
    println!("  v2: (1, 0, 0)");
    println!("  v3: (0, 1, 0)");
    println!("  v4: (0, 0, 1)");
    println!();
    println!("Circumcenter: {:?}", circumcenter_3d.coordinates());
    println!("Circumradius: {circumradius_3d:.6}");
    println!();
}

fn test_point_against_3d_simplex(
    simplex_vertices_3d: &[Vertex<f64, i32, 3>],
    circumcenter_3d: &Point<f64, 3>,
    circumradius_3d: f64,
) {
    // Test the point [0.9, 0.9, 0.9]
    let test_vertex_3d = create_vertex([0.9, 0.9, 0.9], 3);

    // Calculate distance from circumcenter to test point
    let distance_to_test_3d = na::distance(
        &na::Point::<f64, 3>::from(circumcenter_3d.coordinates()),
        &na::Point::<f64, 3>::from([0.9, 0.9, 0.9]),
    );

    println!("Test point [0.9, 0.9, 0.9]:");
    println!("  Distance from circumcenter: {distance_to_test_3d:.6}");
    println!("  Circumradius: {circumradius_3d:.6}");
    println!(
        "  Inside circumsphere: {}",
        distance_to_test_3d < circumradius_3d
    );
    println!();

    // Test both methods
    test_circumsphere_methods(simplex_vertices_3d, test_vertex_3d);
    test_boundary_vertex_case(simplex_vertices_3d);
}

fn test_circumsphere_methods(
    simplex_vertices: &[Vertex<f64, i32, 3>],
    test_vertex: Vertex<f64, i32, 3>,
) {
    match insphere(simplex_vertices, test_vertex) {
        Ok(standard_method_3d) => match insphere_lifted(simplex_vertices, test_vertex) {
            Ok(matrix_method_3d) => {
                println!("Standard method result: {standard_method_3d:?}");
                println!("Matrix method result: {matrix_method_3d:?}");
            }
            Err(e) => println!("Matrix method error: {e}"),
        },
        Err(e) => println!("Standard method error: {e}"),
    }
}

fn test_boundary_vertex_case(simplex_vertices: &[Vertex<f64, i32, 3>]) {
    println!();
    println!("Testing boundary vertex (vertex1):");
    let vertex1 = simplex_vertices[0];
    match insphere(simplex_vertices, vertex1) {
        Ok(standard_vertex) => match insphere_lifted(simplex_vertices, vertex1) {
            Ok(matrix_vertex) => {
                println!("Standard method for vertex1: {standard_vertex:?}");
                println!("Matrix method for vertex1: {matrix_vertex:?}");
            }
            Err(e) => println!("Matrix method error for vertex1: {e}"),
        },
        Err(e) => {
            println!("Standard method error for vertex1: {e}");
        }
    }
}

#[allow(clippy::too_many_lines)]
fn test_3d_matrix_analysis() {
    println!("\n=============================================");
    println!("3D Matrix Method Analysis - Step by Step");
    println!("=============================================");

    // Create the 3D simplex: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let vertex1: Vertex<f64, i32, 3> = create_vertex([0.0, 0.0, 0.0], 1);
    let vertex2 = create_vertex([1.0, 0.0, 0.0], 1);
    let vertex3 = create_vertex([0.0, 1.0, 0.0], 1);
    let vertex4 = create_vertex([0.0, 0.0, 1.0], 2);
    let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];

    println!("3D Simplex vertices:");
    println!("  v0: (0, 0, 0)");
    println!("  v1: (1, 0, 0)");
    println!("  v2: (0, 1, 0)");
    println!("  v3: (0, 0, 1)");
    println!();

    // Test point
    let test_point = [0.9, 0.9, 0.9];
    let test_vertex = create_vertex(test_point, 3);

    // Get reference vertex (first vertex)
    let ref_coords = [0.0, 0.0, 0.0];
    println!("Reference vertex (v0): {ref_coords:?}");
    println!();

    // Manually build the matrix as in the matrix method
    let mut matrix = zeros(4, 4); // D+1 x D+1 for D=3

    println!("Building matrix rows:");

    // Row 0: v1 - v0 = (1,0,0) - (0,0,0) = (1,0,0), ||v1-v0||² = 1
    let v1_rel = [1.0, 0.0, 0.0];
    let v1_norm2 = 1.0;
    matrix[(0, 0)] = v1_rel[0];
    matrix[(0, 1)] = v1_rel[1];
    matrix[(0, 2)] = v1_rel[2];
    matrix[(0, 3)] = v1_norm2;
    println!(
        "  Row 0 (v1-v0): [{}, {}, {}, {}]",
        v1_rel[0], v1_rel[1], v1_rel[2], v1_norm2
    );

    // Row 1: v2 - v0 = (0,1,0) - (0,0,0) = (0,1,0), ||v2-v0||² = 1
    let v2_rel = [0.0, 1.0, 0.0];
    let v2_norm2 = 1.0;
    matrix[(1, 0)] = v2_rel[0];
    matrix[(1, 1)] = v2_rel[1];
    matrix[(1, 2)] = v2_rel[2];
    matrix[(1, 3)] = v2_norm2;
    println!(
        "  Row 1 (v2-v0): [{}, {}, {}, {}]",
        v2_rel[0], v2_rel[1], v2_rel[2], v2_norm2
    );

    // Row 2: v3 - v0 = (0,0,1) - (0,0,0) = (0,0,1), ||v3-v0||² = 1
    let v3_rel = [0.0, 0.0, 1.0];
    let v3_norm2 = 1.0;
    matrix[(2, 0)] = v3_rel[0];
    matrix[(2, 1)] = v3_rel[1];
    matrix[(2, 2)] = v3_rel[2];
    matrix[(2, 3)] = v3_norm2;
    println!(
        "  Row 2 (v3-v0): [{}, {}, {}, {}]",
        v3_rel[0], v3_rel[1], v3_rel[2], v3_norm2
    );

    // Row 3: test_point - v0 = (0.9,0.9,0.9) - (0,0,0) = (0.9,0.9,0.9), ||test-v0||² = 0.9² + 0.9² + 0.9² = 2.43
    let test_rel = [0.9, 0.9, 0.9];
    let test_norm2 = 0.9 * 0.9 + 0.9 * 0.9 + 0.9 * 0.9;
    matrix[(3, 0)] = test_rel[0];
    matrix[(3, 1)] = test_rel[1];
    matrix[(3, 2)] = test_rel[2];
    matrix[(3, 3)] = test_norm2;
    println!(
        "  Row 3 (test-v0): [{}, {}, {}, {}]",
        test_rel[0], test_rel[1], test_rel[2], test_norm2
    );

    println!();
    println!("Matrix:");
    for i in 0..4 {
        println!(
            "  [{:5.1}, {:5.1}, {:5.1}, {:5.1}]",
            matrix[(i, 0)],
            matrix[(i, 1)],
            matrix[(i, 2)],
            matrix[(i, 3)]
        );
    }

    let det = matrix.det();
    println!();
    println!("Determinant: {det:.6}");

    // Check simplex orientation
    match simplex_orientation(&simplex_vertices) {
        Ok(is_positive_orientation) => {
            let is_positive = matches!(is_positive_orientation, Orientation::POSITIVE);
            println!(
                "Simplex orientation: {} (positive: {})",
                if is_positive { "POSITIVE" } else { "NEGATIVE" },
                is_positive
            );

            // Apply the sign interpretation from the matrix method
            let matrix_result = if is_positive {
                det < 0.0 // For positive orientation, negative det means inside
            } else {
                det > 0.0 // For negative orientation, positive det means inside  
            };

            println!(
                "Matrix method interpretation: det {} 0.0 means {}",
                if det < 0.0 { "<" } else { ">" },
                if matrix_result { "INSIDE" } else { "OUTSIDE" }
            );
            println!("Matrix method result: {matrix_result}");

            // Compare with actual geometric facts
            match circumcenter(&simplex_vertices) {
                Ok(circumcenter) => {
                    match circumradius(&simplex_vertices) {
                        Ok(circumradius) => {
                            let distance_to_test = na::distance(
                                &na::Point::<f64, 3>::from(circumcenter.coordinates()),
                                &na::Point::<f64, 3>::from(test_point),
                            );

                            println!();
                            println!("Geometric verification:");
                            println!("  Circumcenter: {:?}", circumcenter.coordinates());
                            println!("  Circumradius: {circumradius:.6}");
                            println!("  Distance to test point: {distance_to_test:.6}");
                            println!(
                                "  Geometric truth (distance < radius): {}",
                                distance_to_test < circumradius
                            );

                            // Compare with both methods
                            match insphere(&simplex_vertices, test_vertex) {
                                Ok(standard_result) => {
                                    match insphere_lifted(&simplex_vertices, test_vertex) {
                                        Ok(matrix_method_result) => {
                                            println!();
                                            println!("Method comparison:");
                                            println!("  Standard method: {standard_result:?}");
                                            println!("  Matrix method: {matrix_method_result}");
                                            println!(
                                                "  Geometric truth: {}",
                                                distance_to_test < circumradius
                                            );

                                            println!();
                                            let standard_inside =
                                                matches!(standard_result, InSphere::INSIDE);
                                            if standard_inside == (distance_to_test < circumradius)
                                            {
                                                println!(
                                                    "✓ Standard method matches geometric truth"
                                                );
                                            } else {
                                                println!(
                                                    "✗ Standard method disagrees with geometric truth"
                                                );
                                            }

                                            let matrix_inside =
                                                matches!(matrix_method_result, InSphere::INSIDE);
                                            let matrix_agrees =
                                                matrix_inside == (distance_to_test < circumradius);
                                            let standard_inside =
                                                matches!(standard_result, InSphere::INSIDE);
                                            let methods_agree = standard_inside == matrix_inside;

                                            if matrix_agrees {
                                                println!("✓ Matrix method matches geometric truth");
                                            } else {
                                                println!(
                                                    "✗ Matrix method disagrees with geometric truth"
                                                );
                                                println!(
                                                    "  NOTE: This disagreement is expected for this simplex geometry"
                                                );
                                                println!(
                                                    "        due to the matrix method's inverted sign convention."
                                                );
                                            }

                                            println!();
                                            if methods_agree {
                                                println!("✓ Both methods agree with each other");
                                            } else {
                                                println!(
                                                    "⚠ Methods disagree (expected for this matrix formulation)"
                                                );
                                                println!(
                                                    "  The matrix method uses coordinates relative to the first vertex,"
                                                );
                                                println!(
                                                    "  which produces an inverted sign convention compared to the standard method."
                                                );
                                                println!(
                                                    "  Both methods are mathematically correct but use different interpretations."
                                                );
                                            }
                                        }
                                        Err(e) => println!("Matrix method error: {e}"),
                                    }
                                }
                                Err(e) => println!("Standard method error: {e}"),
                            }
                        }
                        Err(e) => println!("Error calculating circumradius: {e}"),
                    }
                }
                Err(e) => println!("Error calculating circumcenter: {e}"),
            }
        }
        Err(e) => println!("Error determining simplex orientation: {e}"),
    }
}

/// Debug 3D circumsphere properties analysis
fn debug_3d_circumsphere_properties() {
    println!("=== 3D Unit Tetrahedron Analysis ===");

    // Unit tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
    let vertex1: Vertex<f64, i32, 3> = create_vertex([0.0, 0.0, 0.0], 1);
    let vertex2 = create_vertex([1.0, 0.0, 0.0], 1);
    let vertex3 = create_vertex([0.0, 1.0, 0.0], 1);
    let vertex4 = create_vertex([0.0, 0.0, 1.0], 2);
    let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];

    let center = circumcenter(&simplex_vertices).unwrap();
    let radius = circumradius(&simplex_vertices).unwrap();

    println!("Circumcenter: {:?}", center.coordinates());
    println!("Circumradius: {radius}");

    // Test the point (0.9, 0.9, 0.9)
    let distance_to_center =
        ((0.9_f64 - 0.5).powi(2) + (0.9_f64 - 0.5).powi(2) + (0.9_f64 - 0.5).powi(2)).sqrt();
    println!("Point (0.9, 0.9, 0.9) distance to circumcenter: {distance_to_center}");
    println!(
        "Is point inside circumsphere (distance < radius)? {}",
        distance_to_center < radius
    );

    let test_vertex = create_vertex([0.9, 0.9, 0.9], 4);

    let standard_result = insphere_distance(&simplex_vertices, test_vertex).unwrap();
    let matrix_result = insphere_lifted(&simplex_vertices, test_vertex).unwrap();

    println!("Standard method result: {standard_result:?}");
    println!("Matrix method result: {matrix_result:?}");
}

/// Debug 4D circumsphere properties analysis  
fn debug_4d_circumsphere_properties() {
    println!("\n=== 4D Symmetric Simplex Analysis ===");

    // Regular 4D simplex with vertices forming a specific pattern
    let vertex1: Vertex<f64, Option<()>, 4> = create_vertex([1.0, 1.0, 1.0, 1.0], None);
    let vertex2 = create_vertex([1.0, -1.0, -1.0, -1.0], None);
    let vertex3 = create_vertex([-1.0, 1.0, -1.0, -1.0], None);
    let vertex4 = create_vertex([-1.0, -1.0, 1.0, -1.0], None);
    let vertex5 = create_vertex([-1.0, -1.0, -1.0, 1.0], None);
    let simplex_vertices_4d = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

    let center_4d = circumcenter(&simplex_vertices_4d).unwrap();
    let radius_4d = circumradius(&simplex_vertices_4d).unwrap();

    println!("4D Circumcenter: {:?}", center_4d.coordinates());
    println!("4D Circumradius: {radius_4d}");

    // Test the origin (0, 0, 0, 0)
    let distance_to_center_4d =
        (center_4d.coordinates().iter().map(|&x| x * x).sum::<f64>()).sqrt();
    println!("Origin distance to circumcenter: {distance_to_center_4d}");
    println!(
        "Is origin inside circumsphere (distance < radius)? {}",
        distance_to_center_4d < radius_4d
    );

    let origin_vertex = create_vertex([0.0, 0.0, 0.0, 0.0], None);

    let standard_result_4d = insphere_distance(&simplex_vertices_4d, origin_vertex).unwrap();
    let matrix_result_4d = insphere_lifted(&simplex_vertices_4d, origin_vertex).unwrap();

    println!("Standard method result for origin: {standard_result_4d:?}");
    println!("Matrix method result for origin: {matrix_result_4d:?}");
}

/// Compare results between standard and matrix methods
fn compare_circumsphere_methods() {
    println!("\n=== Comparing Circumsphere Methods ===");

    // Compare results between standard and matrix methods
    let vertex1: Vertex<f64, Option<()>, 2> = create_vertex([0.0, 0.0], None);
    let vertex2 = create_vertex([1.0, 0.0], None);
    let vertex3 = create_vertex([0.0, 1.0], None);
    let simplex_vertices = vec![vertex1, vertex2, vertex3];

    // Test various points
    let test_points = [
        Point::new([0.1, 0.1]),   // Should be inside
        Point::new([0.5, 0.5]),   // Circumcenter region
        Point::new([10.0, 10.0]), // Far outside
        Point::new([0.25, 0.25]), // Inside
        Point::new([2.0, 2.0]),   // Outside
    ];

    for (i, point) in test_points.iter().enumerate() {
        let test_vertex = create_vertex(point.coordinates(), None);

        let standard_result = insphere_distance(&simplex_vertices, test_vertex).unwrap();
        let matrix_result = insphere_lifted(&simplex_vertices, test_vertex).unwrap();

        println!(
            "Point {i}: {:?} -> Standard: {:?}, Matrix: {:?}",
            point.coordinates(),
            standard_result,
            matrix_result
        );
    }
}

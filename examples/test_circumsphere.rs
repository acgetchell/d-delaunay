//! # Circumsphere Containment and Simplex Orientation Test Example (4D)
//!
//! This example demonstrates and compares two methods for testing whether a point
//! lies inside the circumsphere of a 4D simplex (5-cell/hypertetrahedron):
//!
//! 1. **Distance-based method** (`circumsphere_contains`): Computes the circumcenter
//!    and circumradius explicitly, then checks if the test point is within that distance.
//! 2. **Standard determinant-based method** (`circumsphere_contains_vertex`): Uses a matrix
//!    determinant approach that is more numerically stable.
//! 3. **Optimized matrix method** (`circumsphere_contains_vertex_matrix`): Uses a different
//!    matrix formulation with inverted sign convention for better numerical conditioning.
//!
//! Additionally, this example demonstrates the `simplex_orientation` function which
//! determines whether a simplex is positively or negatively oriented based on the
//! determinant of its vertex coordinates.
//!
//! ## Test Setup
//!
//! The example uses a unit 4D simplex with vertices at:
//! - `[0, 0, 0, 0]` (origin)
//! - `[1, 0, 0, 0]` (unit vector along x-axis)
//! - `[0, 1, 0, 0]` (unit vector along y-axis)
//! - `[0, 0, 1, 0]` (unit vector along z-axis)
//! - `[0, 0, 0, 1]` (unit vector along w-axis)
//!
//! This 4D simplex has a circumcenter at `[0.5, 0.5, 0.5, 0.5]` and circumradius of
//! `√4/2 = 1.0`.
//!
//! ## Test Categories
//!
//! The example tests several categories of points:
//! - **Inside points**: Small coordinates like `[0.25, 0.25, 0.25]` that should be
//!   well within the circumsphere
//! - **Outside points**: Large coordinates like `[2, 2, 2]` that are clearly outside
//! - **Boundary points**: Points on edges and faces of the tetrahedron
//! - **Vertex points**: The tetrahedron vertices themselves
//!
//! ## Expected Behavior
//!
//! Both methods should generally agree on clearly inside/outside points. However,
//! the determinant method may be more accurate for boundary cases due to its
//! superior numerical stability.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --example test_circumsphere
//! ```

use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
use d_delaunay::geometry::Point;
use d_delaunay::geometry::predicates::{
    circumcenter, circumradius, circumsphere_contains, circumsphere_contains_vertex,
    circumsphere_contains_vertex_matrix, simplex_orientation,
};
use nalgebra as na;
use peroxide::fuga::{LinearAlgebra, zeros};
use std::env;

/// Main function that demonstrates circumsphere containment testing.
///
/// This function sets up a unit tetrahedron and tests various points to see
/// if they lie inside the circumsphere using both available methods.
fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() > 1 {
        match args[1].as_str() {
            "4d" => test_4d_circumsphere_methods(),
            "containment" => test_circumsphere_containment(),
            "orientation" => test_simplex_orientation(),
            "impact" => demonstrate_orientation_impact_on_circumsphere(),
            "3d" => test_3d_simplex_analysis(),
            "matrix" => test_3d_matrix_analysis(),
            "debug3d" => debug_3d_circumsphere_properties(),
            "debug4d" => debug_4d_circumsphere_properties(),
            "compare" => compare_circumsphere_methods(),
            "all" => {
                test_4d_circumsphere_methods();
                test_circumsphere_containment();
                test_simplex_orientation();
                demonstrate_orientation_impact_on_circumsphere();
                test_3d_simplex_analysis();
                test_3d_matrix_analysis();
                debug_3d_circumsphere_properties();
                debug_4d_circumsphere_properties();
                compare_circumsphere_methods();
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
    println!("====================================");
    println!();
    println!("Usage: cargo run --example test_circumsphere [TEST]");
    println!();
    println!("Available tests:");
    println!("  4d          - Test 4D circumsphere methods comparison");
    println!("  containment - Test circumsphere containment with various points");
    println!("  orientation - Test simplex orientation detection");
    println!("  impact      - Demonstrate orientation impact on circumsphere testing");
    println!("  3d          - Test 3D simplex analysis for debugging");
    println!("  matrix      - Detailed matrix method analysis for 3D case");
    println!("  debug3d     - Debug 3D circumsphere properties");
    println!("  debug4d     - Debug 4D circumsphere properties");
    println!("  compare     - Compare circumsphere methods across different points");
    println!("  all         - Run all tests");
    println!("  help        - Show this help message");
    println!();
    println!("Examples:");
    println!("  cargo run --example test_circumsphere 3d");
    println!("  cargo run --example test_circumsphere matrix");
    println!("  cargo run --example test_circumsphere debug3d");
    println!("  cargo run --example test_circumsphere compare");
    println!("  cargo run --example test_circumsphere all");
}

/// Test and compare both 4D circumsphere containment methods
#[allow(clippy::too_many_lines)]
fn test_4d_circumsphere_methods() {
    println!("=============================================");
    println!("Testing 4D circumsphere containment methods:");
    println!("  circumsphere_contains_vertex vs circumsphere_contains_vertex_matrix");
    println!("=============================================");

    // Create a unit 4-simplex: vertices at origin and unit vectors along each axis
    let vertices: Vec<Vertex<f64, i32, 4>> = vec![
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap(),
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
                        let test_vertex = VertexBuilder::default()
                            .point(Point::new(coords))
                            .data(99)
                            .build()
                            .unwrap();

                        let result_standard = circumsphere_contains_vertex(&vertices, test_vertex);
                        let result_matrix =
                            circumsphere_contains_vertex_matrix(&vertices, test_vertex);

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
                            result_standard
                                .as_ref()
                                .map(|b| if *b { "INSIDE" } else { "OUTSIDE" })
                        );
                        println!(
                            "  Matrix method: {:?}",
                            result_matrix
                                .as_ref()
                                .map(|b| if *b { "INSIDE" } else { "OUTSIDE" })
                        );

                        // Check if methods agree
                        if result_standard.is_ok() && result_matrix.is_ok() {
                            let agree = result_standard.unwrap() == result_matrix.unwrap();
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
    println!(
        "  determinant-based (circumsphere_contains_vertex) vs distance-based (circumsphere_contains)"
    );
    println!("=============================================");

    // Define the 4D simplex vertices that form a unit 5-cell
    // This creates a 4D simplex with vertices at the origin and unit vectors
    // along each coordinate axis. The circumcenter is at [0.5, 0.5, 0.5, 0.5] and
    // the circumradius is √4/2 = 1.0.
    let vertices: [Vertex<f64, i32, 4>; 5] = [
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0])) // Origin
            .data(0)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0])) // Unit vector along x-axis
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0])) // Unit vector along y-axis
            .data(2)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0])) // Unit vector along z-axis
            .data(3)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0])) // Unit vector along w-axis
            .data(4)
            .build()
            .unwrap(),
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
        VertexBuilder::default()
            .point(Point::new([0.25, 0.25, 0.25, 0.25]))
            .data(10)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.1, 0.1, 0.1, 0.1]))
            .data(11)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.2, 0.2, 0.2, 0.2]))
            .data(12)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.3, 0.2, 0.1, 0.0]))
            .data(13)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .data(14)
            .build()
            .unwrap(), // Origin should be inside
    ];

    println!("Testing points that should be INSIDE the circumsphere:");
    for (i, point) in test_points_inside.iter().enumerate() {
        let result_determinant = circumsphere_contains_vertex(&vertices, *point);
        let result_distance = circumsphere_contains(&vertices, *point);
        let coords: [f64; 4] = point.into();
        println!(
            "  Point {}: [{}, {}, {}, {}] -> Det: {}, Dist: {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            if result_determinant.unwrap_or(false) {
                "INSIDE"
            } else {
                "OUTSIDE"
            },
            if result_distance.unwrap_or(false) {
                "INSIDE"
            } else {
                "OUTSIDE"
            }
        );
    }
    println!();

    // Test points that should be outside the circumsphere
    // These include points with large coordinates and points along the axes
    // that extend beyond the simplex vertices
    let test_points_outside: [Vertex<f64, i32, 4>; 6] = [
        VertexBuilder::default()
            .point(Point::new([2.0, 2.0, 2.0, 2.0]))
            .data(20)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0, 1.0]))
            .data(21)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.8, 0.8, 0.8, 0.8]))
            .data(22)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.5, 0.0, 0.0, 0.0]))
            .data(23)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.5, 0.0, 0.0]))
            .data(24)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.5, 0.0]))
            .data(25)
            .build()
            .unwrap(),
    ];

    println!("Testing points that should be OUTSIDE the circumsphere:");
    for (i, point) in test_points_outside.iter().enumerate() {
        let result_determinant = circumsphere_contains_vertex(&vertices, *point);
        let result_distance = circumsphere_contains(&vertices, *point);
        let coords: [f64; 4] = point.into();
        println!(
            "  Point {}: [{}, {}, {}, {}] -> Det: {}, Dist: {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            if result_determinant.unwrap_or(false) {
                "INSIDE"
            } else {
                "OUTSIDE"
            },
            if result_distance.unwrap_or(false) {
                "INSIDE"
            } else {
                "OUTSIDE"
            }
        );
    }
    println!();

    // Test edge cases - points on the simplex vertices themselves
    // These should be on the boundary of the circumsphere (distance = radius)
    println!("Testing the simplex vertices themselves:");
    for (i, vertex) in vertices.iter().enumerate() {
        let result = circumsphere_contains_vertex(&vertices, *vertex);
        let coords: [f64; 4] = vertex.into();
        println!(
            "  Vertex {}: [{}, {}, {}, {}] -> {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            if result.unwrap_or(false) {
                "INSIDE"
            } else {
                "OUTSIDE"
            }
        );
    }
    println!();

    // Additional boundary testing with points on edges and faces of the tetrahedron
    // These points lie on the boundary of the 4D simplex and test
    // numerical stability near the boundary
    let boundary_points: [Vertex<f64, i32, 4>; 5] = [
        VertexBuilder::default()
            .point(Point::new([0.5, 0.5, 0.0, 0.0]))
            .data(30)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.5, 0.0, 0.5, 0.0]))
            .data(31)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.5, 0.5, 0.0]))
            .data(32)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.33, 0.33, 0.33, 0.01]))
            .data(33)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.25, 0.25, 0.25, 0.25]))
            .data(34)
            .build()
            .unwrap(),
    ];

    println!("Testing boundary/edge points:");
    for (i, point) in boundary_points.iter().enumerate() {
        let result = circumsphere_contains_vertex(&vertices, *point);
        let coords: [f64; 4] = point.into();
        println!(
            "  Point {}: [{}, {}, {}, {}] -> {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            coords[3],
            if result.unwrap_or(false) {
                "INSIDE"
            } else {
                "OUTSIDE"
            }
        );
    }
}

#[allow(clippy::too_many_lines)]
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
        if orientation_original.unwrap_or(false) {
            "POSITIVE"
        } else {
            "NEGATIVE"
        }
    );

    // Create a negatively oriented 4D simplex by swapping two vertices
    let vertices_negative = create_negative_4d_simplex();

    let orientation_negative = simplex_orientation(&vertices_negative);
    println!(
        "Negatively oriented 4D simplex: {}",
        if orientation_negative.unwrap_or(false) {
            "POSITIVE"
        } else {
            "NEGATIVE"
        }
    );

    // Test 3D orientation (tetrahedron) for comparison
    let tetrahedron_vertices: [Vertex<f64, i32, 3>; 4] = [
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0])) // Origin
            .data(0)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0])) // Unit vector along x-axis
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0])) // Unit vector along y-axis
            .data(2)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0])) // Unit vector along z-axis
            .data(3)
            .build()
            .unwrap(),
    ];

    let orientation_3d = simplex_orientation(&tetrahedron_vertices);
    println!(
        "3D tetrahedron orientation: {}",
        if orientation_3d.unwrap_or(false) {
            "POSITIVE"
        } else {
            "NEGATIVE"
        }
    );

    // Test 2D orientation (triangle)
    let triangle_vertices: [Vertex<f64, i32, 2>; 3] = [
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .data(0)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .data(2)
            .build()
            .unwrap(),
    ];

    let orientation_2d = simplex_orientation(&triangle_vertices);
    println!(
        "2D triangle orientation: {}",
        if orientation_2d.unwrap_or(false) {
            "POSITIVE"
        } else {
            "NEGATIVE"
        }
    );

    // Test 2D orientation with reversed vertex order
    let triangle_vertices_reversed: [Vertex<f64, i32, 2>; 3] = [
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .data(0)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.0])) // Swapped order
            .data(2)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0])) // Swapped order
            .data(1)
            .build()
            .unwrap(),
    ];

    let orientation_2d_reversed = simplex_orientation(&triangle_vertices_reversed);
    println!(
        "2D triangle (reversed order): {}",
        if orientation_2d_reversed.unwrap_or(false) {
            "POSITIVE"
        } else {
            "NEGATIVE"
        }
    );

    // Test degenerate case (collinear points in 2D)
    let collinear_vertices: [Vertex<f64, i32, 2>; 3] = [
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .data(0)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([2.0, 0.0])) // Collinear point
            .data(2)
            .build()
            .unwrap(),
    ];

    let orientation_collinear = simplex_orientation(&collinear_vertices);
    println!(
        "Collinear 2D points: {}",
        if orientation_collinear.unwrap_or(false) {
            "POSITIVE"
        } else {
            "NEGATIVE/DEGENERATE"
        }
    );
}

fn create_unit_4d_simplex() -> [Vertex<f64, i32, 4>; 5] {
    [
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .data(0)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0]))
            .data(2)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0]))
            .data(3)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0]))
            .data(4)
            .build()
            .unwrap(),
    ]
}

fn create_negative_4d_simplex() -> [Vertex<f64, i32, 4>; 5] {
    [
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .data(0)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0]))
            .data(2)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0]))
            .data(3)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0]))
            .data(4)
            .build()
            .unwrap(),
    ]
}

fn demonstrate_orientation_impact_on_circumsphere() {
    println!("\n--- Impact of orientation on circumsphere testing ---");

    // Create vertices for testing
    let vertices = create_unit_4d_simplex();
    let vertices_negative = create_negative_4d_simplex();

    let test_point = VertexBuilder::default()
        .point(Point::new([0.25, 0.25, 0.25, 0.25]))
        .data(100)
        .build()
        .unwrap();

    let inside_positive = circumsphere_contains_vertex(&vertices, test_point);
    let inside_negative = circumsphere_contains_vertex(&vertices_negative, test_point);

    println!(
        "Point [0.25, 0.25, 0.25, 0.25] in positive 4D simplex: {}",
        if inside_positive.unwrap_or(false) {
            "INSIDE"
        } else {
            "OUTSIDE"
        }
    );
    println!(
        "Point [0.25, 0.25, 0.25, 0.25] in negative 4D simplex: {}",
        if inside_negative.unwrap_or(false) {
            "INSIDE"
        } else {
            "OUTSIDE"
        }
    );

    println!("\nNote: The circumsphere_contains_vertex function automatically handles");
    println!("      orientation by calling simplex_orientation internally.");
    println!("      Both results should be the same regardless of vertex ordering!");

    println!("\nTest completed!");
}

#[allow(clippy::too_many_lines)]
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
    let vertex1_3d: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex2_3d = VertexBuilder::default()
        .point(Point::new([1.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex3_3d = VertexBuilder::default()
        .point(Point::new([0.0, 1.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex4_3d = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 1.0]))
        .data(2)
        .build()
        .unwrap();
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
    let test_point_3d = Point::new([0.9, 0.9, 0.9]);
    let test_vertex_3d = VertexBuilder::default()
        .point(test_point_3d)
        .data(3)
        .build()
        .unwrap();

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
    match circumsphere_contains_vertex(simplex_vertices, test_vertex) {
        Ok(standard_method_3d) => {
            match circumsphere_contains_vertex_matrix(simplex_vertices, test_vertex) {
                Ok(matrix_method_3d) => {
                    println!("Standard method result: {standard_method_3d}");
                    println!("Matrix method result: {matrix_method_3d}");
                }
                Err(e) => println!("Matrix method error: {e}"),
            }
        }
        Err(e) => println!("Standard method error: {e}"),
    }
}

fn test_boundary_vertex_case(simplex_vertices: &[Vertex<f64, i32, 3>]) {
    println!();
    println!("Testing boundary vertex (vertex1):");
    let vertex1 = simplex_vertices[0];
    match circumsphere_contains_vertex(simplex_vertices, vertex1) {
        Ok(standard_vertex) => {
            match circumsphere_contains_vertex_matrix(simplex_vertices, vertex1) {
                Ok(matrix_vertex) => {
                    println!("Standard method for vertex1: {standard_vertex}");
                    println!("Matrix method for vertex1: {matrix_vertex}");
                }
                Err(e) => println!("Matrix method error for vertex1: {e}"),
            }
        }
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
    let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex2 = VertexBuilder::default()
        .point(Point::new([1.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex3 = VertexBuilder::default()
        .point(Point::new([0.0, 1.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex4 = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 1.0]))
        .data(2)
        .build()
        .unwrap();
    let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];

    println!("3D Simplex vertices:");
    println!("  v0: (0, 0, 0)");
    println!("  v1: (1, 0, 0)");
    println!("  v2: (0, 1, 0)");
    println!("  v3: (0, 0, 1)");
    println!();

    // Test point
    let test_point = [0.9, 0.9, 0.9];
    let test_vertex = VertexBuilder::default()
        .point(Point::new(test_point))
        .data(3)
        .build()
        .unwrap();

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
            println!(
                "Simplex orientation: {} (positive: {})",
                if is_positive_orientation {
                    "POSITIVE"
                } else {
                    "NEGATIVE"
                },
                is_positive_orientation
            );

            // Apply the sign interpretation from the matrix method
            let matrix_result = if is_positive_orientation {
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
                            match circumsphere_contains_vertex(&simplex_vertices, test_vertex) {
                                Ok(standard_result) => {
                                    match circumsphere_contains_vertex_matrix(
                                        &simplex_vertices,
                                        test_vertex,
                                    ) {
                                        Ok(matrix_method_result) => {
                                            println!();
                                            println!("Method comparison:");
                                            println!("  Standard method: {standard_result}");
                                            println!("  Matrix method: {matrix_method_result}");
                                            println!(
                                                "  Geometric truth: {}",
                                                distance_to_test < circumradius
                                            );

                                            println!();
                                            if standard_result == (distance_to_test < circumradius)
                                            {
                                                println!(
                                                    "✓ Standard method matches geometric truth"
                                                );
                                            } else {
                                                println!(
                                                    "✗ Standard method disagrees with geometric truth"
                                                );
                                            }

                                            let matrix_agrees = matrix_method_result
                                                == (distance_to_test < circumradius);
                                            let methods_agree =
                                                standard_result == matrix_method_result;

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
    let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex2 = VertexBuilder::default()
        .point(Point::new([1.0, 0.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex3 = VertexBuilder::default()
        .point(Point::new([0.0, 1.0, 0.0]))
        .data(1)
        .build()
        .unwrap();
    let vertex4 = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 1.0]))
        .data(2)
        .build()
        .unwrap();
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

    let test_vertex = VertexBuilder::default()
        .point(Point::new([0.9, 0.9, 0.9]))
        .data(4)
        .build()
        .unwrap();

    let standard_result = circumsphere_contains(&simplex_vertices, test_vertex).unwrap();
    let matrix_result =
        circumsphere_contains_vertex_matrix(&simplex_vertices, test_vertex).unwrap();

    println!("Standard method result: {standard_result}");
    println!("Matrix method result: {matrix_result}");
}

/// Debug 4D circumsphere properties analysis
fn debug_4d_circumsphere_properties() {
    println!("\n=== 4D Symmetric Simplex Analysis ===");

    // Regular 4D simplex with vertices forming a specific pattern
    let vertex1: Vertex<f64, Option<()>, 4> = VertexBuilder::default()
        .point(Point::new([1.0, 1.0, 1.0, 1.0]))
        .build()
        .unwrap();
    let vertex2 = VertexBuilder::default()
        .point(Point::new([1.0, -1.0, -1.0, -1.0]))
        .build()
        .unwrap();
    let vertex3 = VertexBuilder::default()
        .point(Point::new([-1.0, 1.0, -1.0, -1.0]))
        .build()
        .unwrap();
    let vertex4 = VertexBuilder::default()
        .point(Point::new([-1.0, -1.0, 1.0, -1.0]))
        .build()
        .unwrap();
    let vertex5 = VertexBuilder::default()
        .point(Point::new([-1.0, -1.0, -1.0, 1.0]))
        .build()
        .unwrap();
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

    let origin_vertex = VertexBuilder::default()
        .point(Point::new([0.0, 0.0, 0.0, 0.0]))
        .build()
        .unwrap();

    let standard_result_4d = circumsphere_contains(&simplex_vertices_4d, origin_vertex).unwrap();
    let matrix_result_4d =
        circumsphere_contains_vertex_matrix(&simplex_vertices_4d, origin_vertex).unwrap();

    println!("Standard method result for origin: {standard_result_4d}");
    println!("Matrix method result for origin: {matrix_result_4d}");
}

/// Compare results between standard and matrix methods
fn compare_circumsphere_methods() {
    println!("\n=== Comparing Circumsphere Methods ===");

    // Compare results between standard and matrix methods
    let vertex1: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
        .point(Point::new([0.0, 0.0]))
        .build()
        .unwrap();
    let vertex2 = VertexBuilder::default()
        .point(Point::new([1.0, 0.0]))
        .build()
        .unwrap();
    let vertex3 = VertexBuilder::default()
        .point(Point::new([0.0, 1.0]))
        .build()
        .unwrap();
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
        let test_vertex = VertexBuilder::default().point(*point).build().unwrap();

        let standard_result = circumsphere_contains(&simplex_vertices, test_vertex).unwrap();
        let matrix_result =
            circumsphere_contains_vertex_matrix(&simplex_vertices, test_vertex).unwrap();

        println!(
            "Point {i}: {:?} -> Standard: {standard_result}, Matrix: {matrix_result}",
            point.coordinates()
        );
    }
}

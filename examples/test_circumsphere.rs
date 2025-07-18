//! # Circumsphere Containment Test Example
//!
//! This example demonstrates and compares two methods for testing whether a point
//! lies inside the circumsphere of a tetrahedron:
//!
//! 1. **Distance-based method** (`circumsphere_contains`): Computes the circumcenter
//!    and circumradius explicitly, then checks if the test point is within that distance.
//! 2. **Determinant-based method** (`circumsphere_contains_vertex`): Uses a matrix
//!    determinant approach that is more numerically stable.
//!
//! ## Test Setup
//!
//! The example uses a unit tetrahedron with vertices at:
//! - `[0, 0, 0]` (origin)
//! - `[1, 0, 0]` (unit vector along x-axis)
//! - `[0, 1, 0]` (unit vector along y-axis)
//! - `[0, 0, 1]` (unit vector along z-axis)
//!
//! This tetrahedron has a circumcenter at `[0.5, 0.5, 0.5]` and circumradius of
//! `√3/2 ≈ 0.866`.
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
use d_delaunay::geometry::predicates::{circumsphere_contains, circumsphere_contains_vertex};

/// Main function that demonstrates circumsphere containment testing.
///
/// This function sets up a unit tetrahedron and tests various points to see
/// if they lie inside the circumsphere using both available methods.
#[allow(clippy::too_many_lines)]
fn main() {
    println!("Testing circumsphere containment:");
    println!(
        "  determinant-based (circumsphere_contains_vertex) vs distance-based (circumsphere_contains)"
    );
    println!("=============================================");

    // Define the tetrahedron vertices that form a unit simplex
    // This creates a tetrahedron with vertices at the origin and unit vectors
    // along each coordinate axis. The circumcenter is at [0.5, 0.5, 0.5] and
    // the circumradius is √3/2 ≈ 0.866.
    let vertices: [Vertex<f64, i32, 3>; 4] = [
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

    println!("Tetrahedron vertices:");
    for (i, vertex) in vertices.iter().enumerate() {
        let coords: [f64; 3] = vertex.into();
        println!("  v{}: [{}, {}, {}]", i, coords[0], coords[1], coords[2]);
    }
    println!();

    // Test points that should be inside the circumsphere
    // These are points with small coordinates that should be well within
    // the circumsphere radius of √3/2 ≈ 0.866
    let test_points_inside: [Vertex<f64, i32, 3>; 5] = [
        VertexBuilder::default()
            .point(Point::new([0.25, 0.25, 0.25]))
            .data(10)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.1, 0.1, 0.1]))
            .data(11)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.2, 0.2, 0.2]))
            .data(12)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.3, 0.2, 0.1]))
            .data(13)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(14)
            .build()
            .unwrap(), // Origin should be inside
    ];

    println!("Testing points that should be INSIDE the circumsphere:");
    for (i, point) in test_points_inside.iter().enumerate() {
        let result_determinant = circumsphere_contains_vertex(&vertices, *point);
        let result_distance = circumsphere_contains(&vertices, *point);
        let coords: [f64; 3] = point.into();
        println!(
            "  Point {}: [{}, {}, {}] -> Det: {}, Dist: {}",
            i,
            coords[0],
            coords[1],
            coords[2],
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
    // that extend beyond the tetrahedron vertices
    let test_points_outside: [Vertex<f64, i32, 3>; 6] = [
        VertexBuilder::default()
            .point(Point::new([2.0, 2.0, 2.0]))
            .data(20)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(21)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.8, 0.8, 0.8]))
            .data(22)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([1.5, 0.0, 0.0]))
            .data(23)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 1.5, 0.0]))
            .data(24)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.5]))
            .data(25)
            .build()
            .unwrap(),
    ];

    println!("Testing points that should be OUTSIDE the circumsphere:");
    for (i, point) in test_points_outside.iter().enumerate() {
        let result_determinant = circumsphere_contains_vertex(&vertices, *point);
        let result_distance = circumsphere_contains(&vertices, *point);
        let coords: [f64; 3] = point.into();
        println!(
            "  Point {}: [{}, {}, {}] -> Det: {}, Dist: {}",
            i,
            coords[0],
            coords[1],
            coords[2],
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

    // Test edge cases - points on the tetrahedron vertices themselves
    // These should be on the boundary of the circumsphere (distance = radius)
    println!("Testing the tetrahedron vertices themselves:");
    for (i, vertex) in vertices.iter().enumerate() {
        let result = circumsphere_contains_vertex(&vertices, *vertex);
        let coords: [f64; 3] = vertex.into();
        println!(
            "  Vertex {}: [{}, {}, {}] -> {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            if result.unwrap_or(false) {
                "INSIDE"
            } else {
                "OUTSIDE"
            }
        );
    }
    println!();

    // Additional boundary testing with points on edges and faces of the tetrahedron
    // These points lie on the edges and faces of the tetrahedron and test
    // numerical stability near the boundary
    let boundary_points: [Vertex<f64, i32, 3>; 4] = [
        VertexBuilder::default()
            .point(Point::new([0.5, 0.5, 0.0]))
            .data(30)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.5, 0.0, 0.5]))
            .data(31)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.0, 0.5, 0.5]))
            .data(32)
            .build()
            .unwrap(),
        VertexBuilder::default()
            .point(Point::new([0.33, 0.33, 0.33]))
            .data(33)
            .build()
            .unwrap(),
    ];

    println!("Testing boundary/edge points:");
    for (i, point) in boundary_points.iter().enumerate() {
        let result = circumsphere_contains_vertex(&vertices, *point);
        let coords: [f64; 3] = point.into();
        println!(
            "  Point {}: [{}, {}, {}] -> {}",
            i,
            coords[0],
            coords[1],
            coords[2],
            if result.unwrap_or(false) {
                "INSIDE"
            } else {
                "OUTSIDE"
            }
        );
    }

    println!("\nTest completed!");
}

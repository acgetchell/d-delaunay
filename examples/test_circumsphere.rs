//! # Circumsphere Containment and Simplex Orientation Test Example (4D)
//!
//! This example demonstrates and compares two methods for testing whether a point
//! lies inside the circumsphere of a 4D simplex (5-cell/hypertetrahedron):
//!
//! 1. **Distance-based method** (`circumsphere_contains`): Computes the circumcenter
//!    and circumradius explicitly, then checks if the test point is within that distance.
//! 2. **Determinant-based method** (`circumsphere_contains_vertex`): Uses a matrix
//!    determinant approach that is more numerically stable.
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
    circumsphere_contains, circumsphere_contains_vertex, simplex_orientation,
};

/// Main function that demonstrates circumsphere containment testing.
///
/// This function sets up a unit tetrahedron and tests various points to see
/// if they lie inside the circumsphere using both available methods.
fn main() {
    test_circumsphere_containment();
    test_simplex_orientation();
    demonstrate_orientation_impact_on_circumsphere();
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

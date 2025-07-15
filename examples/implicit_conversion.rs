//! # Implicit Conversion Example
//!
//! This example demonstrates the implicit conversion functionality that allows
//! `vertex.point.coordinates()` and `point.coordinates()` to be automatically
//! converted to coordinate arrays using Rust's `From` trait.
//!
//! ## Key Features Demonstrated
//!
//! - Converting owned vertices to coordinate arrays
//! - Converting vertex references to coordinate arrays (preserving the original)
//! - Converting owned points to coordinate arrays
//! - Converting point references to coordinate arrays (preserving the original)
//!
//! Run this example with: `cargo run --example implicit_conversion`

use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
use d_delaunay::geometry::point::PointND;

/// Demonstrates implicit conversion from vertices and points to coordinate arrays.
///
/// This function shows various ways to convert vertices and points to coordinate
/// arrays using the newly implemented `From` traits, which provide more ergonomic
/// alternatives to explicitly calling `.coordinates()`.
fn main() {
    // Create a vertex with 3D coordinates
    // Note: Using usize for vertex data type since f64 doesn't implement Hash/Ord
    let vertex: Vertex<usize, 3> = VertexBuilder::default()
        .point(PointND::new([1.0, 2.0, 3.0]))
        .data(42_usize) // Add some vertex data
        .build()
        .unwrap();

    // Before: You had to call .coordinates() explicitly
    let coords_explicit: [f64; 3] = vertex.point().coordinates();
    println!("Explicit coordinates: {coords_explicit:?}");

    // After: You can now use implicit conversion from the vertex
    let coords_from_vertex: [f64; 3] = vertex.into();
    println!("Implicit conversion from vertex: {coords_from_vertex:?}");

    // Create another vertex for reference conversion
    let another_vertex: Vertex<usize, 3> = VertexBuilder::default()
        .point(PointND::new([4.0, 5.0, 6.0]))
        .data(100_usize)
        .build()
        .unwrap();

    // You can also convert from a reference to preserve the original vertex
    let coords_from_ref: [f64; 3] = (&another_vertex).into();
    println!("Implicit conversion from vertex reference: {coords_from_ref:?}");
    println!(
        "Original vertex still available: {:?}",
        another_vertex.point().coordinates()
    );

    // Point implicit conversion also works
    let point = PointND::new([7.0, 8.0, 9.0]);
    let coords_from_point: [f64; 3] = point.into();
    println!("Implicit conversion from point: {coords_from_point:?}");

    // And from point reference
    let another_point = PointND::new([10.0, 11.0, 12.0]);
    let coords_from_point_ref: [f64; 3] = (&another_point).into();
    println!("Implicit conversion from point reference: {coords_from_point_ref:?}");
    println!(
        "Original point still available: {:?}",
        another_point.coordinates()
    );
}

#[cfg(test)]
mod tests {
    use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    use d_delaunay::geometry::point::PointND;

    #[test]
    fn test_vertex_implicit_conversion() {
        // Test owned vertex conversion
        let vertex: Vertex<usize, 3> = VertexBuilder::default()
            .point(PointND::new([1.0, 2.0, 3.0]))
            .data(42_usize)
            .build()
            .unwrap();

        let coords: [f64; 3] = vertex.into();
        assert_eq!(coords, [1.0, 2.0, 3.0]);

        // Test vertex reference conversion
        let vertex_ref: Vertex<usize, 3> = VertexBuilder::default()
            .point(PointND::new([4.0, 5.0, 6.0]))
            .data(100_usize)
            .build()
            .unwrap();

        let coords_ref: [f64; 3] = (&vertex_ref).into();
        assert_eq!(coords_ref, [4.0, 5.0, 6.0]);

        // Verify original is still available
        assert_eq!(vertex_ref.point().coordinates(), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_point_implicit_conversion() {
        // Test owned point conversion
        let point = PointND::new([7.0, 8.0, 9.0]);
        let coords: [f64; 3] = point.into();
        assert_eq!(coords, [7.0, 8.0, 9.0]);

        // Test point reference conversion
        let point_ref = PointND::new([10.0, 11.0, 12.0]);
        let coords_ref: [f64; 3] = (&point_ref).into();
        assert_eq!(coords_ref, [10.0, 11.0, 12.0]);

        // Verify original is still available
        assert_eq!(point_ref.coordinates(), [10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_different_dimensions() {
        // Test 2D points
        let point_2d = PointND::new([1.0, 2.0]);
        let coords_2d: [f64; 2] = point_2d.into();
        assert_eq!(coords_2d, [1.0, 2.0]);

        // Test 4D points
        let point_4d = PointND::new([1.0, 2.0, 3.0, 4.0]);
        let coords_4d: [f64; 4] = point_4d.into();
        assert_eq!(coords_4d, [1.0, 2.0, 3.0, 4.0]);

        // Test vertices with different dimensions
        let vertex_2d: Vertex<usize, 2> = VertexBuilder::default()
            .point(PointND::new([5.0, 6.0]))
            .data(123_usize)
            .build()
            .unwrap();

        let coords_v2d: [f64; 2] = vertex_2d.into();
        assert_eq!(coords_v2d, [5.0, 6.0]);
    }

    #[test]
    fn test_conversion_consistency() {
        let point = PointND::new([1.5, 2.5, 3.5]);
        let explicit_coords = point.coordinates();
        let implicit_coords: [f64; 3] = point.into();

        assert_eq!(explicit_coords, implicit_coords);
    }

    #[test]
    fn test_vertex_conversion_consistency() {
        let vertex: Vertex<usize, 3> = VertexBuilder::default()
            .point(PointND::new([1.1, 2.2, 3.3]))
            .data(999_usize)
            .build()
            .unwrap();

        let explicit_coords = vertex.point().coordinates();
        let implicit_coords: [f64; 3] = vertex.into();

        assert_eq!(explicit_coords, implicit_coords);
    }

    #[test]
    fn test_functional_usage() {
        // Test that implicit conversion works in functional contexts
        let vertices: Vec<Vertex<usize, 3>> = vec![
            VertexBuilder::default()
                .point(PointND::new([1.0, 2.0, 3.0]))
                .data(1_usize)
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(PointND::new([4.0, 5.0, 6.0]))
                .data(2_usize)
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(PointND::new([7.0, 8.0, 9.0]))
                .data(3_usize)
                .build()
                .unwrap(),
        ];

        let coords: Vec<[f64; 3]> = vertices.into_iter().map(|v| v.into()).collect();
        assert_eq!(
            coords,
            vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0],]
        );
    }

    #[test]
    fn test_edge_cases() {
        // Test with zero coordinates
        let zero_point = PointND::new([0.0, 0.0, 0.0]);
        let zero_coords: [f64; 3] = zero_point.into();
        assert_eq!(zero_coords, [0.0, 0.0, 0.0]);

        // Test with negative coordinates
        let neg_point = PointND::new([-1.0, -2.0, -3.0]);
        let neg_coords: [f64; 3] = neg_point.into();
        assert_eq!(neg_coords, [-1.0, -2.0, -3.0]);

        // Test with very small coordinates
        let small_point = PointND::new([1e-10, 2e-10, 3e-10]);
        let small_coords: [f64; 3] = small_point.into();
        assert_eq!(small_coords, [1e-10, 2e-10, 3e-10]);

        // Test with very large coordinates
        let large_point = PointND::new([1e10, 2e10, 3e10]);
        let large_coords: [f64; 3] = large_point.into();
        assert_eq!(large_coords, [1e10, 2e10, 3e10]);
    }
}

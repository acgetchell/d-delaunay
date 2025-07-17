//! # Implicit Conversion Example
//!
//! This example demonstrates the implicit conversion functionality that allows
//! `vertex.point.coordinates()` and `point.coordinates()` to be automatically
//! converted to coordinate arrays using Rust's `From` trait.
//!
//! ## Features Demonstrated
//!
//! - Converting owned vertices to coordinate arrays
//! - Converting vertex references to coordinate arrays (preserving the original)
//! - Converting owned points to coordinate arrays
//! - Converting point references to coordinate arrays (preserving the original)
//!
//! ## Benefits
//!
//! This refactoring provides more ergonomic code while maintaining backward
//! compatibility. Instead of always having to call `.coordinates()` explicitly,
//! you can now use implicit conversion via `.into()` or let type inference
//! handle the conversion automatically in many contexts.
//!
//! Run this example with: `cargo run --example implicit_conversion`

use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
use d_delaunay::geometry::point::Point;

/// Demonstrates implicit conversion from vertices and points to coordinate arrays.
///
/// This function shows various ways to convert vertices and points to coordinate
/// arrays using the newly implemented `From` traits, which provide more ergonomic
/// alternatives to explicitly calling `.coordinates()`.
fn main() {
    // Create a vertex with 3D coordinates
    let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
        .point(Point::new([1.0, 2.0, 3.0]))
        .build()
        .unwrap();

    // Before: You had to call .coordinates() explicitly
    let coords_explicit: [f64; 3] = vertex.point().coordinates();
    println!("Explicit coordinates: {coords_explicit:?}");

    // After: You can now use implicit conversion from the vertex
    let coords_from_vertex: [f64; 3] = vertex.into();
    println!("Implicit conversion from vertex: {coords_from_vertex:?}");

    // Create another vertex for reference conversion
    let another_vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
        .point(Point::new([4.0, 5.0, 6.0]))
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
    let point = Point::new([7.0, 8.0, 9.0]);
    let coords_from_point: [f64; 3] = point.into();
    println!("Implicit conversion from point: {coords_from_point:?}");

    // And from point reference
    let another_point = Point::new([10.0, 11.0, 12.0]);
    let coords_from_point_ref: [f64; 3] = (&another_point).into();
    println!("Implicit conversion from point reference: {coords_from_point_ref:?}");
    println!(
        "Original point still available: {:?}",
        another_point.coordinates()
    );
}

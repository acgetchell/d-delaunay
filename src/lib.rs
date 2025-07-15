//! # d-delaunay
//!
//! This is a library for computing the Delaunay triangulation of a set of n-dimensional points
//! in a [simplicial complex](https://en.wikipedia.org/wiki/Simplicial_complex)
//! inspired by [CGAL](https://www.cgal.org).
//!
//! # Features
//!
//! - d-dimensional Delaunay triangulations
//! - Arbitrary data types associated with vertices and cells
//! - Serialization/Deserialization with [serde](https://serde.rs)

#[macro_use]
extern crate derive_builder;
extern crate peroxide;

/// The main module of the library. This module contains the public interface
/// for the library.
pub mod delaunay_core {
    pub mod cell;
    pub mod facet;
    pub mod matrix;
    pub mod triangulation_data_structure;
    pub mod utilities;
    pub mod vertex;
    // Re-export the `delaunay_core` modules.
    pub use cell::*;
    pub use facet::*;
    pub use matrix::*;
    pub use triangulation_data_structure::*;
    pub use utilities::*;
    pub use vertex::*;
}

/// The geometry module contains geometric data structures and operations.
///
/// This module provides fundamental geometric primitives and operations needed for
/// d-dimensional Delaunay triangulation computations. It includes:
///
/// - Point representations in d-dimensional space
/// - Vector operations and abstractions
/// - Geometric validation and error handling
///
/// # Key Components
///
/// * [`Point`](point::Point) - A generic point in d-dimensional space backed by
///   abstract vector storage
/// * [`PointND`](point::PointND) - Type alias for d-dimensional points using
///   `nalgebra::SVector` as storage
/// * [`VectorN`](point::VectorN) - Trait for abstract vector operations in
///   d-dimensional space
///
/// # Special Floating-Point Semantics
///
/// The geometry module implements custom equality semantics for floating-point
/// coordinates that treat NaN values as equal to themselves. This enables Points
/// to be used as keys in hash-based collections, which is essential for the
/// triangulation algorithms.
///
/// # Example
///
/// ```rust
/// use d_delaunay::geometry::{PointND, HashCoordinate};
///
/// // Create a 3D point
/// let point: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
/// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
/// assert_eq!(point.dim(), 3);
///
/// // Validate that coordinates are finite
/// assert!(point.is_valid().is_ok());
/// ```
pub mod geometry {
    pub mod point;
    // Re-export the `geometry` modules.
    pub use point::*;
}

/// The function `is_normal` checks that structs implement `auto` traits.
/// Traits are checked at compile time, so this function is only used for
/// testing.
#[allow(clippy::extra_unused_type_parameters)]
#[must_use]
pub fn is_normal<T: Sized + Send + Sync + Unpin>() -> bool {
    true
}

#[cfg(test)]
mod lib_tests {
    use crate::{
        delaunay_core::{
            cell::Cell, facet::Facet, triangulation_data_structure::Tds, vertex::Vertex,
        },
        geometry::point::PointND,
        is_normal,
    };

    #[test]
    fn normal_types() {
        assert!(is_normal::<PointND<3>>());
        assert!(is_normal::<PointND<2>>());
        assert!(is_normal::<Vertex<usize, 3>>());
        assert!(is_normal::<Facet<usize, Option<()>, 3>>());
        assert!(is_normal::<Cell<usize, Option<()>, 4>>());
        assert!(is_normal::<Tds<f64, usize, Option<()>, 4>>());
    }
}

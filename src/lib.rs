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

/// Contains the `Point` struct and geometry predicates.
pub mod geometry {
    pub mod point;
    pub mod predicates;
    pub use point::*;
    pub use predicates::*;
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
        geometry::Point,
        is_normal,
    };

    #[test]
    fn normal_types() {
        assert!(is_normal::<Point<f64, 3>>());
        assert!(is_normal::<Point<f32, 3>>());
        assert!(is_normal::<Vertex<f64, Option<()>, 3>>());
        assert!(is_normal::<Facet<f64, Option<()>, Option<()>, 3>>());
        assert!(is_normal::<Cell<f64, Option<()>, Option<()>, 4>>());
        assert!(is_normal::<Tds<f64, Option<()>, Option<()>, 4>>());
    }
}

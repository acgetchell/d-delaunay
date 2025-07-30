//! # d-delaunay
//!
//! This is a library for computing the Delaunay triangulation of a set of n-dimensional points
//! in a [simplicial complex](https://en.wikipedia.org/wiki/Simplicial_complex)
//! inspired by [CGAL](https://www.cgal.org).
//!
//! # Features
//!
//! - d-dimensional Delaunay triangulations
//! - Generic floating-point coordinate types (supports `f32`, `f64`, and other types implementing `CoordinateScalar`)
//! - Arbitrary data types associated with vertices and cells
//! - Serialization/Deserialization with [serde](https://serde.rs)

// Allow multiple crate versions due to transitive dependencies
#![allow(clippy::multiple_crate_versions)]

#[macro_use]
extern crate derive_builder;

/// The main module of the library. This module contains the public interface
/// for the library.
pub mod delaunay_core {
    pub mod cell;
    pub mod facet;
    pub mod triangulation_data_structure;
    pub mod utilities;
    pub mod vertex;
    /// Traits for Delaunay triangulation data structures.
    pub mod traits {
        pub mod data;
        pub use data::*;
    }
    // Re-export the `delaunay_core` modules.
    pub use cell::*;
    pub use facet::*;
    pub use traits::*;
    pub use triangulation_data_structure::*;
    pub use utilities::*;
    pub use vertex::*;
}

/// Contains geometric types including the `Point` struct and geometry predicates.
///
/// The geometry module provides a coordinate abstraction through the `Coordinate` trait
/// that unifies coordinate operations across different storage mechanisms. The `Point`
/// type implements this abstraction, providing generic floating-point coordinate support
/// (for `f32`, `f64`, and other types implementing `CoordinateScalar`) with proper NaN
/// handling, validation, and hashing.
pub mod geometry {
    pub mod matrix;
    pub mod point;
    pub mod predicates;
    /// Traits module containing coordinate abstractions and reusable trait definitions.
    ///
    /// This module contains the core `Coordinate` trait that abstracts coordinate
    /// operations, along with supporting traits for validation (`FiniteCheck`),
    /// equality comparison (`OrderedEq`), and hashing (`HashCoordinate`) of
    /// floating-point coordinate values.
    pub mod traits {
        pub mod coordinate;
        pub mod finitecheck;
        pub mod hashcoordinate;
        pub mod orderedeq;
        pub use coordinate::*;
        pub use finitecheck::*;
        pub use hashcoordinate::*;
        pub use orderedeq::*;
    }
    pub use matrix::*;
    pub use point::*;
    pub use predicates::*;
    pub use traits::*;
}

/// A prelude module that re-exports commonly used types and macros.
/// This makes it easier to import the most commonly used items from the crate.
pub mod prelude {
    // Re-export from delaunay_core
    pub use crate::delaunay_core::{
        cell::*, facet::*, traits::data::*, triangulation_data_structure::*, utilities::*,
        vertex::*,
    };

    // Re-export from geometry
    pub use crate::geometry::{
        matrix::*,
        point::*,
        predicates::*,
        traits::{coordinate::*, finitecheck::*, hashcoordinate::*, orderedeq::*},
    };
}

/// The function `is_normal` checks that structs implement `auto` traits.
/// Traits are checked at compile time, so this function is only used for
/// testing.
#[allow(clippy::extra_unused_type_parameters)]
#[must_use]
pub const fn is_normal<T: Sized + Send + Sync + Unpin>() -> bool {
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

//! # d-delaunay
//!
//! This is a library for computing the Delaunay triangulation of a set of n-dimensional points in a [simplicial complex](https://en.wikipedia.org/wiki/Simplicial_complex)
//! inspired by [CGAL](https://www.cgal.org).
//!
//! # Features
//! * d-dimensional Delaunay triangulations
//! * Arbitrary data types associated with vertices and cells

/// The main module of the library. This module contains the public interface for the library.
pub mod delaunay_core {
    pub mod cell;
    pub mod point;
    pub mod triangulation_data_structure;
    pub mod utilities;
    pub mod vertex;
}

/// The function `is_normal` checks that structs implement `auto` traits.
fn is_normal<T: Sized + Send + Sync + Unpin>() {}

#[cfg(test)]
mod tests {
    use crate::{
        delaunay_core::{
            cell::Cell, point::Point, triangulation_data_structure::Tds, vertex::Vertex,
        },
        is_normal,
    };

    #[test]
    fn normal_types() {
        is_normal::<Point<f64, 3>>();
        is_normal::<Point<f32, 3>>();
        is_normal::<Vertex<f64, Option<()>, 3>>();
        is_normal::<Cell<f64, Option<()>, Option<()>, 4>>();
        is_normal::<Tds<f64, Option<()>, Option<()>, 3>>();
    }
}

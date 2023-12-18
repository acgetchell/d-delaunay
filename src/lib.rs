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

#[cfg(test)]
mod tests {
    use crate::delaunay_core::triangulation_data_structure::start;

    #[test]
    fn it_works() {
        let result = start();
        assert_eq!(result, 1);
    }
}

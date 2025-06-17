//! A facet is a d-1 sub-simplex of a d-dimensional simplex.
//! It is defined in terms of a cell and the vertex in the cell opposite to it,
//! as per [CGAL](https://doc.cgal.org/latest/TDS_3/index.html#title3).
//! This provides convenience methods used in the
//! [Bowyer-Watson algorithm](https://en.wikipedia.org/wiki/Bowyer–Watson_algorithm).
//! Facets are not stored in the `Triangulation Data Structure` (TDS)
//! directly, but created on the fly when needed.

use super::{cell::Cell, vertex::Vertex};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::hash::Hash;
use thiserror::Error;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, PartialOrd, Serialize)]
/// The [Facet] struct represents a facet of a d-dimensional simplex.
/// Passing in a [Vertex] and a [Cell] containing that vertex to the
/// constructor will create a [Facet] struct.
///
/// # Properties
///
/// * `cell` - The [Cell] that contains this facet.
/// * `vertex` - The [Vertex] in the [Cell] opposite to this [Facet].
///
/// Note that `D` is the dimensionality of the [Cell] and [Vertex];
/// the [Facet] is one dimension less than the [Cell] (co-dimension 1).
pub struct Facet<T, U, V, const D: usize>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The [Cell] that contains this facet.
    pub cell: Cell<T, U, V, D>,

    /// The [Vertex] opposite to this facet.
    pub vertex: Vertex<T, U, D>,
}

impl<T, U, V, const D: usize> Facet<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The `new` function is a constructor for the [Facet]. It takes
    /// in a [Cell] and a [Vertex] as arguments and returns a [Result]
    /// containing a [Facet] or an error message.
    ///
    /// # Arguments
    ///
    /// * `cell`: The [Cell] that contains the [Facet].
    /// * `vertex`: The [Vertex] opposite to the [Facet].
    ///
    /// # Returns
    ///
    /// A [Result] containing a [Facet] or an error message as to why
    /// the [Facet] could not be created.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::facet::Facet;
    /// use d_delaunay::delaunay_core::point::Point;
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex4 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).build().unwrap();
    /// let facet = Facet::new(cell.clone(), vertex1).unwrap();
    /// assert_eq!(facet.cell, cell);
    /// ```
    pub fn new(cell: Cell<T, U, V, D>, vertex: Vertex<T, U, D>) -> Result<Self, anyhow::Error> {
        if !cell.vertices.contains(&vertex) {
            return Err(FacetError::CellDoesNotContainVertex.into());
        }

        if cell.vertices.len() == 1 {
            return Err(FacetError::CellIsZeroSimplex.into());
        }

        Ok(Facet { cell, vertex })
    }

    /// The `vertices` method in the [Facet] returns a container of
    /// [Vertex] objects that are in the [Facet].
    pub fn vertices(&self) -> Vec<Vertex<T, U, D>> {
        self.cell
            .vertices
            .iter()
            .filter(|v| **v != self.vertex)
            .cloned()
            .collect()
    }
}

/// Error type for facet operations.
#[derive(Debug, Error)]
pub enum FacetError {
    /// The cell does not contain the vertex.
    #[error("The cell does not contain the vertex!")]
    CellDoesNotContainVertex,
    /// The cell is a 0-simplex with no facet.
    #[error("The cell is a 0-simplex with no facet!")]
    CellIsZeroSimplex,
}
#[cfg(test)]
mod tests {

    use super::*;
    use crate::delaunay_core::{cell::CellBuilder, point::Point, vertex::VertexBuilder};

    #[test]
    fn facet_new() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();

        assert_eq!(facet.cell, cell);

        // Human readable output for cargo test -- --nocapture
        println!("Facet: {:?}", facet);
    }

    #[test]
    fn facet_new_with_incorrect_vertex() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .build()
            .unwrap();

        assert!(Facet::new(cell.clone(), vertex5).is_err());
    }

    #[test]
    fn facet_new_with_1_simplex() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1])
            .build()
            .unwrap();

        assert!(Facet::new(cell.clone(), vertex1).is_err());
    }

    #[test]
    fn facet_vertices() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();
        let vertices = facet.clone().vertices();

        assert_eq!(vertices.len(), 3);
        assert_eq!(vertices[0], vertex2);
        assert_eq!(vertices[1], vertex3);
        assert_eq!(vertices[2], vertex4);

        // Human readable output for cargo test -- --nocapture
        println!("Facet: {:?}", facet);
    }

    #[test]
    fn facet_to_and_from_json() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();
        let serialized = serde_json::to_string(&facet).unwrap();

        assert!(serialized.contains("[1.0,0.0,0.0]"));
        assert!(serialized.contains("[0.0,1.0,0.0]"));
        assert!(serialized.contains("[0.0,0.0,1.0]"));

        let deserialized: Facet<f64, Option<()>, Option<()>, 3> =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, facet);

        // Human readable output for cargo test -- --nocapture
        println!("Serialized = {:?}", serialized);
    }

    #[test]
    fn facet_partial_eq() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let facet1 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet2 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet3 = Facet::new(cell.clone(), vertex2).unwrap();

        assert_eq!(facet1, facet2);
        assert_ne!(facet1, facet3);
    }

    #[test]
    fn facet_partial_ord() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let facet1 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet2 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet3 = Facet::new(cell.clone(), vertex2).unwrap();
        let facet4 = Facet::new(cell.clone(), vertex3).unwrap();

        assert!(facet1 < facet3);
        assert!(facet2 < facet3);
        assert!(facet3 > facet1);
        assert!(facet3 > facet2);
        assert!(facet3 > facet4);
    }
}

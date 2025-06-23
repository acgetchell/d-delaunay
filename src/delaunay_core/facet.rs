//! A facet is a d-1 sub-simplex of a d-dimensional simplex.
//! It is defined in terms of a cell and the vertex in the cell opposite to it,
//! as per [CGAL](https://doc.cgal.org/latest/TDS_3/index.html#title3).
//! This provides convenience methods used in the
//! [Bowyer-Watson algorithm](https://en.wikipedia.org/wiki/Bowyer–Watson_algorithm).
//! Facets are not stored in the `Triangulation Data Structure` (TDS)
//! directly, but created on the fly when needed.

use super::{cell::Cell, point::OrderedEq, vertex::Vertex};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::hash::{Hash, Hasher};
use thiserror::Error;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
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
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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

/// Generic Eq implementation for Facet that requires Hash bounds
impl<T, U, V, const D: usize> Eq for Facet<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Vertex<T, U, D>: Hash,
    Cell<T, U, V, D>: Hash,
{
}

impl<T, U, V, const D: usize> Hash for Facet<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Vertex<T, U, D>: Hash,
    Cell<T, U, V, D>: Hash,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cell.hash(state);
        self.vertex.hash(state);
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

    #[test]
    fn facet_clone() {
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
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();
        let cloned_facet = facet.clone();

        assert_eq!(facet, cloned_facet);
        assert_eq!(facet.cell.uuid, cloned_facet.cell.uuid);
        assert_eq!(facet.vertex.uuid(), cloned_facet.vertex.uuid());
    }

    #[test]
    fn facet_default() {
        let facet: Facet<f64, Option<()>, Option<()>, 3> = Facet::default();

        // Default facet should have empty cell and default vertex
        assert_eq!(facet.cell.vertices.len(), 0);
        let default_coords: [f64; 3] = (&facet.vertex).into();
        assert_eq!(default_coords, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn facet_debug() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([4.0, 5.0, 6.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();
        let debug_str = format!("{:?}", facet);

        assert!(debug_str.contains("Facet"));
        assert!(debug_str.contains("cell"));
        assert!(debug_str.contains("vertex"));
    }

    #[test]
    fn facet_with_typed_data() {
        let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(2)
            .build()
            .unwrap();
        let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(3)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, &str, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .data("triangle")
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();

        assert_eq!(facet.cell.data, Some("triangle"));
        assert_eq!(facet.vertex.data, Some(1));

        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 2);
        assert!(vertices.iter().any(|v| v.data == Some(2)));
        assert!(vertices.iter().any(|v| v.data == Some(3)));
    }

    #[test]
    fn facet_2d_triangle() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.5, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();

        // Facet of 2D triangle is an edge (1D)
        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 2);
        assert_eq!(vertices[0], vertex2);
        assert_eq!(vertices[1], vertex3);
    }

    #[test]
    fn facet_1d_edge() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 1> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();

        // Facet of 1D edge is a point (0D)
        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 1);
        assert_eq!(vertices[0], vertex2);
    }

    #[test]
    fn facet_4d_simplex() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 4> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4, vertex5])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();

        // Facet of 4D simplex is a 3D tetrahedron
        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 4);
        assert!(vertices.contains(&vertex2));
        assert!(vertices.contains(&vertex3));
        assert!(vertices.contains(&vertex4));
        assert!(vertices.contains(&vertex5));
        assert!(!vertices.contains(&vertex1));
    }

    #[test]
    fn facet_error_cell_does_not_contain_vertex() {
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
        let vertex_not_in_cell = VertexBuilder::default()
            .point(Point::new([2.0, 2.0, 2.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        let result = Facet::new(cell, vertex_not_in_cell);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error
            .to_string()
            .contains("The cell does not contain the vertex!"));
    }

    #[test]
    fn facet_error_cell_is_zero_simplex() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1])
            .build()
            .unwrap();

        let result = Facet::new(cell, vertex1);
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(error
            .to_string()
            .contains("The cell is a 0-simplex with no facet!"));
    }

    #[test]
    fn facet_error_display() {
        let cell_error = FacetError::CellDoesNotContainVertex;
        let simplex_error = FacetError::CellIsZeroSimplex;

        assert_eq!(
            cell_error.to_string(),
            "The cell does not contain the vertex!"
        );
        assert_eq!(
            simplex_error.to_string(),
            "The cell is a 0-simplex with no facet!"
        );
    }

    #[test]
    fn facet_error_debug() {
        let cell_error = FacetError::CellDoesNotContainVertex;
        let simplex_error = FacetError::CellIsZeroSimplex;

        let cell_debug = format!("{:?}", cell_error);
        let simplex_debug = format!("{:?}", simplex_error);

        assert!(cell_debug.contains("CellDoesNotContainVertex"));
        assert!(simplex_debug.contains("CellIsZeroSimplex"));
    }

    #[test]
    fn facet_vertices_empty_cell() {
        // This tests the edge case where a cell might be empty
        // Although this shouldn't happen in practice due to validation
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let empty_cell: Cell<f64, Option<()>, Option<()>, 3> =
            CellBuilder::default().vertices(vec![]).build().unwrap();

        // Create facet directly without using new() to bypass validation
        let facet = Facet {
            cell: empty_cell,
            vertex: vertex1,
        };

        let vertices = facet.vertices();
        assert_eq!(vertices.len(), 0);
    }

    #[test]
    fn facet_vertices_ordering() {
        // Test that vertices are returned in the same order as in the cell
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

        // Create 3D cell with exactly 4 vertices (3+1)
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();

        let facet = Facet::new(cell.clone(), vertex3).unwrap();
        let vertices = facet.vertices();

        // Should have all vertices except vertex3
        assert_eq!(vertices.len(), 3);
        assert!(vertices.contains(&vertex1));
        assert!(vertices.contains(&vertex2));
        assert!(vertices.contains(&vertex4));
        assert!(!vertices.contains(&vertex3));

        // Check ordering is preserved (vertices should appear in same order as in cell)
        assert_eq!(vertices[0], vertex1);
        assert_eq!(vertices[1], vertex2);
        assert_eq!(vertices[2], vertex4);
    }

    #[test]
    fn facet_serialization_with_different_types() {
        let vertex1: Vertex<f32, u8, 2> = VertexBuilder::default()
            .point(Point::new([0.0f32, 0.0f32]))
            .data(1u8)
            .build()
            .unwrap();
        let vertex2: Vertex<f32, u8, 2> = VertexBuilder::default()
            .point(Point::new([1.0f32, 0.0f32]))
            .data(2u8)
            .build()
            .unwrap();
        let vertex3: Vertex<f32, u8, 2> = VertexBuilder::default()
            .point(Point::new([0.5f32, 1.0f32]))
            .data(3u8)
            .build()
            .unwrap();

        let cell: Cell<f32, u8, u16, 2> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .data(100u16)
            .build()
            .unwrap();

        let facet = Facet::new(cell, vertex1).unwrap();
        let serialized = serde_json::to_string(&facet).unwrap();

        assert!(serialized.contains("1.0"));
        assert!(serialized.contains("0.5"));

        let deserialized: Facet<f32, u8, u16, 2> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, facet);
    }

    #[test]
    fn facet_eq_different_vertices() {
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
        let facet2 = Facet::new(cell.clone(), vertex2).unwrap();
        let facet3 = Facet::new(cell.clone(), vertex3).unwrap();
        let facet4 = Facet::new(cell.clone(), vertex4).unwrap();

        // All facets should be different because they have different opposite vertices
        assert_ne!(facet1, facet2);
        assert_ne!(facet1, facet3);
        assert_ne!(facet1, facet4);
        assert_ne!(facet2, facet3);
        assert_ne!(facet2, facet4);
        assert_ne!(facet3, facet4);
    }

    #[test]
    fn facet_eq_same_cells_same_vertices() {
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

        // Use the same vertices for both cells
        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        let cell2: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        let facet1 = Facet::new(cell1, vertex1).unwrap();
        let facet2 = Facet::new(cell2, vertex1).unwrap();

        // Facets should be equal because cells have the same vertices
        // (same UUIDs, same coordinates) and same opposite vertex
        assert_eq!(facet1, facet2);
    }

    #[test]
    fn facet_ne_different_vertices() {
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

        // Create completely different vertices with different coordinates
        let vertex4 = VertexBuilder::default()
            .point(Point::new([2.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([3.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex6 = VertexBuilder::default()
            .point(Point::new([2.0, 1.0, 0.0]))
            .build()
            .unwrap();

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        let cell2: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex4, vertex5, vertex6])
            .build()
            .unwrap();

        let facet1 = Facet::new(cell1, vertex1).unwrap();
        let facet2 = Facet::new(cell2, vertex4).unwrap();

        // Facets should be different because cells have completely different vertices
        assert_ne!(facet1, facet2);
    }

    #[test]
    fn facet_hash() {
        use crate::delaunay_core::{cell::CellBuilder, point::Point, vertex::VertexBuilder};
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Create a cell with some vertices
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

        // Create two facets that should be equal and hash to the same value
        let facet1 = Facet::new(cell.clone(), vertex1).unwrap();
        let facet2 = Facet::new(cell.clone(), vertex1).unwrap();

        // Create a different facet that should hash to a different value
        let facet3 = Facet::new(cell.clone(), vertex2).unwrap();

        // Helper function to get hash value
        fn get_hash<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        // Test that equal facets hash to the same value
        assert_eq!(get_hash(&facet1), get_hash(&facet2));

        // Test that different facets hash to different values
        assert_ne!(get_hash(&facet1), get_hash(&facet3));
    }
}

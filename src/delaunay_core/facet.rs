//! A facet is a d-1 sub-simplex of a d-dimensional simplex.
//! It is defined in terms of a cell and the vertex in the cell opposite to it,
//! as per [CGAL](https://doc.cgal.org/latest/TDS_3/index.html#title3).
//! This provides convenience methods used in the
//! [Bowyer-Watson algorithm](https://en.wikipedia.org/wiki/Bowyer–Watson_algorithm).
//! Facets are not stored in the `Triangulation Data Structure` (TDS)
//! directly, but created on the fly when needed.

use super::{cell::Cell, vertex::Vertex};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
/// The `Facet` struct represents a facet of a d-dimensional simplex.
/// Passing in a `Vertex` and a `Cell` containing that vertex to the
/// constructor will create a `Facet` struct.
///
/// # Properties
///
/// * `cell` - The `Cell` that contains this facet.
/// * `vertex` - The `Vertex` in the `Cell` opposite to this facet.
///
/// Note that `D` is the dimensionality of the `Cell` and `Vertex`;
/// the `Facet` is one dimension less than the `Cell` (co-dimension 1).
pub struct Facet<T: Clone + Copy + Default, U, V, const D: usize>
where
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
    U: Clone + Copy,
    V: Clone + Copy,
{
    /// The `Cell` that contains this facet.
    pub cell: Cell<T, U, V, D>,

    /// The `Vertex` opposite to this facet.
    pub vertex: Vertex<T, U, D>,
}

impl<T, U, V, const D: usize> Facet<T, U, V, D>
where
    T: Copy + Default + PartialEq,
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
    U: Clone + Copy + PartialEq,
    V: Clone + Copy,
{
    /// The `new` function is a constructor for the `Facet` struct. It takes
    /// in a `Cell` and a `Vertex` as arguments and returns a `Result`
    /// containing a `Facet` struct or an error message (`&'static str`).
    ///
    /// # Arguments
    ///
    /// * `cell`: The `Cell` that contains the `Facet`.
    /// * `vertex`: The `Vertex` opposite to the `Facet`.
    ///
    /// # Returns
    ///
    /// A `Result` containing a `Facet` struct or an error message as to why
    /// the `Facet` could not be created.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::Cell;
    /// use d_delaunay::delaunay_core::facet::Facet;
    /// use d_delaunay::delaunay_core::point::Point;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// let vertex1 = Vertex::new(Point::new([0.0, 0.0, 0.0]));
    /// let vertex2 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
    /// let vertex3 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
    /// let vertex4 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
    /// let facet = Facet::new(cell.clone(), vertex1).unwrap();
    /// assert_eq!(facet.cell, cell);
    /// ```
    pub fn new(cell: Cell<T, U, V, D>, vertex: Vertex<T, U, D>) -> Result<Self, &'static str> {
        if !cell.vertices.contains(&vertex) {
            return Err("The cell does not contain the vertex!");
        }

        if cell.vertices.len() == 1 {
            return Err("The cell is a 0-simplex with no facet!");
        }

        Ok(Facet { cell, vertex })
    }

    /// The `vertices` method in the `Facet` struct returns a vector of
    /// `Vertices` that are in the facet.
    pub fn vertices(&mut self) -> Vec<Vertex<T, U, D>> {
        let mut vertices = self.cell.clone().vertices;
        vertices.retain(|v| *v != self.vertex);

        vertices
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::delaunay_core::point::Point;

    #[test]
    fn facet_new() {
        let vertex1 = Vertex::new(Point::new([0.0, 0.0, 0.0]));
        let vertex2 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
        let vertex3 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
        let vertex4 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
        let facet = Facet::new(cell.clone(), vertex1).unwrap();

        assert_eq!(facet.cell, cell);

        // Human readable output for cargo test -- --nocapture
        println!("Facet: {:?}", facet);
    }

    #[test]
    fn facet_new_with_incorrect_vertex() {
        let vertex1 = Vertex::new(Point::new([0.0, 0.0, 0.0]));
        let vertex2 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
        let vertex3 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
        let vertex4 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
        let vertex5 = Vertex::new(Point::new([1.0, 1.0, 1.0]));

        assert!(Facet::new(cell.clone(), vertex5).is_err());
    }

    #[test]
    fn facet_new_with_1_simplex() {
        let vertex1 = Vertex::new(Point::new([0.0, 0.0, 0.0]));
        let cell: Cell<f64, Option<()>, Option<()>, 3> = Cell::new(vec![vertex1]).unwrap();

        assert!(Facet::new(cell.clone(), vertex1).is_err());
    }

    #[test]
    fn facet_vertices() {
        let vertex1 = Vertex::new(Point::new([0.0, 0.0, 0.0]));
        let vertex2 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
        let vertex3 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
        let vertex4 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
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
        let vertex1 = Vertex::new(Point::new([0.0, 0.0, 0.0]));
        let vertex2 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
        let vertex3 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
        let vertex4 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
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
}

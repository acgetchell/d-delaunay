//! Data and operations on d-dimensional triangulation data structures.
//!
//! Intended to match functionality of [CGAL Triangulations](https://doc.cgal.org/latest/Triangulation/index.html).

use super::utilities::find_min_coordinate;
use super::{cell::Cell, point::Point, vertex::Vertex};
use std::cmp::PartialEq;
use std::{cmp::min, collections::HashMap};
use uuid::Uuid;

#[derive(Debug, Clone)]
/// The `Tds` struct represents a triangulation data structure with vertices and cells, where the vertices
/// and cells are identified by UUIDs.
///
/// # Properties:
///
/// * `vertices`: A HashMap that stores vertices with their corresponding UUIDs as keys. Each `Vertex` has
/// a `Point` of type T, vertex data of type U, and a constant D representing the dimension.
/// * `cells`: The `cells` property is a `HashMap` that stores `Cell` objects. Each `Cell` has
/// one or more `Vertex<T, U, D>` with cell data of type V. Note the dimensionality of the cell may differ
/// from D, though the TDS only stores cells of maximal dimensionality D and infers other lower dimensional
/// cells from the maximal cells and their vertices.
///
/// For example, in 3 dimensions:
///
/// * A 0-dimensional cell is a `Vertex`.
/// * A 1-dimensional cell is an `Edge` given by the `Tetrahedron` and two `Vertex` endpoints.
/// * A 2-dimensional cell is a `Facet` given by the `Tetrahedron` and the opposite `Vertex`.
/// * A 3-dimensional cell is a `Tetrahedron`, the maximal cell.
///
/// A similar pattern holds for higher dimensions.
///
/// In general, vertices are embedded into D-dimensional Euclidean space, and so the `Tds` is a finite simplicial complex.
pub struct Tds<T, U, V, const D: usize> {
    /// A HashMap that stores vertices with their corresponding UUIDs as keys.
    /// Each `Vertex` has a `Point` of type T, vertex data of type U, and a constant D representing the dimension.
    pub vertices: HashMap<Uuid, Vertex<T, U, D>>,

    /// The `cells` property is a `HashMap` that stores `Cell` objects.
    /// Each `Cell` has one or more `Vertex<T, U, D>` with cell data of type V.
    /// Note the dimensionality of the cell may differ from D, though the TDS only stores cells of maximal dimensionality D
    /// and infers other lower dimensional cells from the maximal cells and their vertices.
    pub cells: HashMap<Uuid, Cell<T, U, V, D>>,
}

impl<T, U, V, const D: usize> Tds<T, U, V, D> {
    /// The function creates a new instance of a triangulation data structure with given points, initializing the vertices and
    /// cells.
    ///
    /// # Arguments:
    ///
    /// * `points`: A vector of points with which to initialize the triangulation.
    ///
    /// # Returns:
    ///
    /// A delaunay triangulation with cells and neighbors aligned, and vertices associated with cells.
    pub fn new(points: Vec<Point<T, D>>) -> Self {
        // handle case where vertices are constructed with data
        let vertices = Vertex::into_hashmap(Vertex::from_points(points));
        // let cells_vec = bowyer_watson(vertices);
        // assign_neighbors(cells_vec);
        // assign_incident_cells(vertices);

        // Put cells_vec into hashmap
        let cells = HashMap::new();
        Self { vertices, cells }
    }

    /// The `add` function checks if a vertex with the same coordinates already exists in a hashmap, and
    /// if not, inserts the vertex into the hashmap.
    ///
    /// # Arguments:
    ///
    /// * `vertex`: The `vertex` parameter is of type `Vertex<T, U, D>`.
    ///
    /// # Returns:
    ///
    /// The function `add` returns `Ok(())` if the vertex was successfully added to the hashmap, or
    /// an error message if the vertex already exists or if there is a uuid collision.
    ///
    /// # Example:
    ///
    /// ```
    /// use d_delaunay::delaunay_core::triangulation_data_structure::Tds;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let mut tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
    /// let point = Point::new([1.0, 2.0, 3.0]);
    /// let vertex = Vertex::new(point);
    /// let result = tds.add(vertex);
    /// assert!(result.is_ok());
    /// ```
    pub fn add(&mut self, vertex: Vertex<T, U, D>) -> Result<(), &'static str>
    where
        T: PartialEq,
    {
        // Don't add if vertex with that point already exists
        for val in self.vertices.values() {
            if val.point.coords == vertex.point.coords {
                return Err("Vertex already exists");
            }
        }

        // Hashmap::insert returns the old value if the key already exists and updates it with the new value
        let result = self.vertices.insert(vertex.uuid, vertex);

        // Return an error if there is a uuid collision
        match result {
            Some(_) => Err("Uuid already exists"),
            None => Ok(()),
        }
    }

    /// The function returns the number of vertices in the triangulation data structure.
    ///
    /// # Returns:
    ///
    /// The number of vertices in the Tds.
    ///
    /// # Example:
    ///
    /// ```
    /// use d_delaunay::delaunay_core::triangulation_data_structure::Tds;
    /// let tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let mut tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
    /// let new_vertex1: Vertex<f64, usize, 3> = Vertex::new(Point::new([1.0, 2.0, 3.0]));
    /// let _ = tds.add(new_vertex1);
    /// assert_eq!(tds.number_of_vertices(), 1);
    /// ```
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// The `dim` function returns the dimensionality of the triangulation data structure.
    ///
    /// # Returns:
    ///
    /// The `dim` function returns the minimum value between the number of vertices minus one and the
    /// value of `D` as an `i32`.
    ///
    /// # Example:
    ///
    /// ```
    /// use d_delaunay::delaunay_core::triangulation_data_structure::Tds;
    /// let tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
    /// assert_eq!(tds.dim(), -1);
    /// ```
    pub fn dim(&self) -> i32 {
        let len = self.number_of_vertices() as i32;

        min(len - 1, D as i32)
    }

    /// The function `number_of_cells` returns the number of cells in a triangulation data structure.
    ///
    /// # Returns:
    ///
    /// The number of cells in the Tds.
    pub fn number_of_cells(&self) -> usize {
        self.cells.len()
    }

    fn bowyer_watson(&mut self) -> Result<Vec<Cell<T, U, V, D>>, &'static str>
    where
        T: Copy + Default + PartialOrd,
        Vertex<T, U, D>: Clone, // Add the Clone trait bound for Vertex
    {
        let cells: Vec<Cell<T, U, V, D>> = Vec::new();

        // Create super-cell that contains all vertices
        // First, find the min and max coordinates
        let _min_coords = find_min_coordinate(self.vertices.clone());

        Ok(cells)
    }

    fn assign_neighbors(&mut self, _cells: Vec<Cell<T, U, V, D>>) -> Result<(), &'static str> {
        todo!("Assign neighbors")
    }

    fn assign_incident_cells(
        &mut self,
        _vertices: Vec<Vertex<T, U, D>>,
    ) -> Result<(), &'static str> {
        todo!("Assign incident cells")
    }
}

/// The function "start" will eventually return a triangulation data structure.
///
/// # Returns:
///
/// The function `start()` is returning an `i32` value of `1`.
pub fn start() -> i32 {
    println!("Starting ...");
    1
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn tds_new() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];

        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), 3);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", tds);
    }

    #[test]
    fn tds_add_dim() {
        let points: Vec<Point<f64, 3>> = Vec::new();

        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), -1);

        let new_vertex1: Vertex<f64, usize, 3> = Vertex::new(Point::new([1.0, 2.0, 3.0]));
        let _ = tds.add(new_vertex1);
        assert_eq!(tds.number_of_vertices(), 1);
        assert_eq!(tds.dim(), 0);

        let new_vertex2: Vertex<f64, usize, 3> = Vertex::new(Point::new([4.0, 5.0, 6.0]));
        let _ = tds.add(new_vertex2);
        assert_eq!(tds.number_of_vertices(), 2);
        assert_eq!(tds.dim(), 1);

        let new_vertex3: Vertex<f64, usize, 3> = Vertex::new(Point::new([7.0, 8.0, 9.0]));
        let _ = tds.add(new_vertex3);
        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.dim(), 2);

        let new_vertex4: Vertex<f64, usize, 3> = Vertex::new(Point::new([10.0, 11.0, 12.0]));
        let _ = tds.add(new_vertex4);
        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.dim(), 3);

        let new_vertex5: Vertex<f64, usize, 3> = Vertex::new(Point::new([13.0, 14.0, 15.0]));
        let _ = tds.add(new_vertex5);
        assert_eq!(tds.number_of_vertices(), 5);
        assert_eq!(tds.dim(), 3);
    }

    #[test]
    fn tds_no_add() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];

        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.cells.len(), 0);
        assert_eq!(tds.dim(), 3);

        let new_vertex1: Vertex<f64, usize, 3> = Vertex::new(Point::new([1.0, 2.0, 3.0]));
        let result = tds.add(new_vertex1);
        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.dim(), 3);

        assert!(result.is_err());
    }
}

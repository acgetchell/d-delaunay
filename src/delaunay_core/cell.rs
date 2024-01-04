//! Data and operations on d-dimensional cells or [simplices](https://en.wikipedia.org/wiki/Simplex).

use super::{point::Point, utilities::make_uuid, vertex::Vertex};
use na::{ComplexField, Const, OPoint};
use nalgebra as na;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug, iter::Sum, ops::Div};
use uuid::Uuid;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
/// The `Cell` struct represents a d-dimensional [simplex](https://en.wikipedia.org/wiki/Simplex)
/// with vertices, a unique identifier, optional neighbors, and optional data.
///
/// # Properties:
///
/// * `vertices`: A vector of vertices. Each `Vertex`` has a type T, optional data U, and a constant
/// D representing the number of dimensions.
/// * `uuid`: The `uuid` property is of type `Uuid` and represents a universally unique identifier for
/// a `Cell`. It is used to uniquely identify each instance of a `Cell`.
/// * `neighbors`: The `neighbors` property is an optional vector of `Uuid` values. It represents the
/// UUIDs of the neighboring cells that are connected to the current cell, indexed such that the `i-th`
/// neighbor is opposite the `i-th`` vertex.
/// * `data`: The `data` property is an optional field that can hold a value of type `V`. It allows
/// storage of additional data associated with the `Cell`.
pub struct Cell<T: Clone + Copy + Default, U, V, const D: usize>
where
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
    U: Clone + Copy,
    V: Clone + Copy,
{
    /// The vertices of the cell.
    pub vertices: Vec<Vertex<T, U, D>>,
    /// The unique identifier of the cell.
    pub uuid: Uuid,
    /// The neighboring cells connected to the current cell.
    pub neighbors: Option<Vec<Uuid>>,
    /// The optional data associated with the cell.
    pub data: Option<V>,
}

impl<T: Clone + ComplexField<RealField = T> + Copy + Default + Sum, U, V, const D: usize>
    Cell<T, U, V, D>
where
    for<'a> &'a T: Div<f64>,
    f64: From<T>,
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
    U: Clone + Copy,
    V: Clone + Copy,
{
    /// The function `new` creates a new `Cell`` object with the given vertices.
    /// A D-dimensional cell has D + 1 vertices, so the number of vertices must be less than or equal to D + 1.
    ///
    /// # Arguments:
    ///
    /// * `vertices`: The vertices of the Cell to be constructed.
    ///
    /// # Returns:
    ///
    /// a `Result` type. If the condition `vertices.len() > D + 1` is true, it returns an `Err` variant
    /// with the message "Number of vertices must be less than or equal to D + 1". Otherwise, it returns
    /// an `Ok` variant with a `Cell` containing the provided `vertices`, a generated `uuid`, and
    /// optional neighbor and data fields.
    ///
    /// Neighbors will be calculated by the `delaunay_core::triangulation_data_structure::Tds`.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::Cell;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
    /// let vertex2 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
    /// let vertex3 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
    /// let vertex4 = Vertex::new(Point::new([1.0, 1.0, 1.0]));
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
    /// assert!(cell.neighbors.is_none());
    /// ```
    pub fn new(vertices: Vec<Vertex<T, U, D>>) -> Result<Self, &'static str> {
        if vertices.len() > D + 1 {
            return Err("Number of vertices must be less than or equal to D + 1");
        }
        let uuid = make_uuid();
        let neighbors = None;
        let data = None;
        Ok(Cell {
            vertices,
            uuid,
            neighbors,
            data,
        })
    }

    /// The function `new_with_data` creates a new `Cell` object with the given vertices and data.
    /// A D-dimensional cell has D + 1 vertices, so the number of vertices must be less than or equal to D + 1.
    ///
    /// # Arguments:
    ///
    /// * `vertices`: The vertices of the Cell to be constructed.
    /// * `data`: The data associated with the cell.
    ///
    /// # Returns:
    ///
    /// a `Result` type. If the condition `vertices.len() > D + 1` is true, it returns an `Err` variant
    /// with the message "Number of vertices must be less than or equal to D + 1". Otherwise, it returns
    /// an `Ok` variant with a `Cell` containing the provided `vertices`, a generated `uuid`, the
    /// provided data, and optional neighbor fields which will be later be calculated by the
    /// `delaunay_core::triangulation_data_structure::Tds`.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::Cell;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
    /// let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
    /// let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
    /// let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
    /// let cell = Cell::new_with_data(vec![vertex1, vertex2, vertex3, vertex4], "three-one cell").unwrap();
    /// assert_eq!(cell.data.unwrap(), "three-one cell");
    /// ```
    pub fn new_with_data(vertices: Vec<Vertex<T, U, D>>, data: V) -> Result<Self, &'static str> {
        if vertices.len() > D + 1 {
            return Err("Number of vertices must be less than or equal to D + 1");
        }
        let uuid = make_uuid();
        let neighbors = None;
        let data = Some(data);
        Ok(Cell {
            vertices,
            uuid,
            neighbors,
            data,
        })
    }

    /// The function `into_hashmap` converts a vector of cells into a hashmap, using the cells' UUIDs as keys.
    pub fn into_hashmap(cells: Vec<Self>) -> HashMap<Uuid, Self> {
        cells.into_iter().map(|c| (c.uuid, c)).collect()
    }

    /// The function returns the number of vertices in the `Cell`.
    ///
    /// # Returns:
    ///
    /// The number of vertices in the `Cell`.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::Cell;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
    /// let vertex2 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
    /// let vertex3 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = Cell::new(vec![vertex1, vertex2, vertex3]).unwrap();
    /// assert_eq!(cell.number_of_vertices(), 3);
    /// ```
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// The `dim` function returns the dimensionality of the `Cell`.
    ///
    /// # Returns:
    ///
    /// The `dim` function returns the dimension, which is calculated by subtracting 1 from
    /// the number of vertices in the `Cell`.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::Cell;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
    /// let vertex2 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
    /// let vertex3 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = Cell::new(vec![vertex1, vertex2, vertex3]).unwrap();
    /// assert_eq!(cell.dim(), 2);
    /// ```
    pub fn dim(&self) -> usize {
        self.vertices.len() - 1
    }

    /// The function is_valid checks if a `Cell` is valid.
    /// struct.
    ///
    /// # Returns:
    ///
    /// True if the `Cell` is valid; the `Vertices` are correct, the `UUID` is valid and unique, the
    /// `neighbors` contains `UUID`s of neighboring `Cell`s, and the `neighbors` are indexed such that
    /// the index of the `Vertex` opposite the neighboring cell is the same.
    pub fn is_valid(self) -> bool {
        todo!("Implement is_valid for Cell")
    }

    /// The function `contains_vertex` checks if a given vertex is present in the Cell.
    ///
    /// # Arguments:
    ///
    /// * `vertex`: The vertex to check.
    ///
    /// # Returns:
    ///
    /// Returns `true` if the given `Vertex` is present in the `Cell`, and `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::Cell;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
    /// let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
    /// let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
    /// let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
    /// let cell = Cell::new_with_data(vec![vertex1, vertex2, vertex3, vertex4], "three-one cell").unwrap();
    /// assert!(cell.contains_vertex(vertex1));
    /// ```
    pub fn contains_vertex(&self, vertex: Vertex<T, U, D>) -> bool
    where
        T: PartialEq,
        U: PartialEq,
    {
        self.vertices.contains(&vertex)
    }

    /// The function `circumcenter` returns the circumcenter of the cell.
    ///
    /// Using the approach from:
    ///
    /// Lévy, Bruno, and Yang Liu. “Lp Centroidal Voronoi Tessellation and Its Applications.”
    /// ACM Transactions on Graphics 29, no. 4 (July 26, 2010): 119:1-119:11.
    /// https://doi.org/10.1145/1778765.1778856.
    ///
    /// The circumcenter C of a cell with vertices x_0, x_1, ..., x_n is the solution to the system
    ///
    /// C = 1/2 (A^-1*B)
    ///
    /// where A is a matrix (to be inverted) of the form:
    ///     (x_1-x0) for all coordinates in x1, x0
    ///     (x2-x0) for all coordinates in x2, x0
    ///     ... for all x_n in the cell
    ///
    /// These are the perpendicular bisectors of the edges of the cell.
    ///
    /// and B is a vector of the form:
    ///     (x_1^2-x0^2) for all coordinates in x1, x0
    ///     (x_2^2-x0^2) for all coordinates in x2, x0
    ///     ... for all x_n in the cell
    ///
    /// The resulting vector gives the coordinates of the circumcenter.
    ///
    /// # Returns:
    ///
    /// If the function is successful, it will return an Ok variant containing the circumcenter as a
    /// Point<f64, D> value. If there is an error, it will return an Err variant containing an error message.
    fn circumcenter(&self) -> Result<Point<f64, D>, &'static str>
    where
        T: Clone + Copy + Debug + PartialEq,
        [T; D]: Default + DeserializeOwned + Serialize + Sized,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let dim = self.dim();
        if self.vertices[0].dim() != dim {
            return Err("Not a simplex!");
        }

        let mut matrix: na::SMatrix<T, D, D> = na::SMatrix::zeros();
        // Column-major matrix, so data in debugger will be opposite of the row,col indices
        for i in 0..dim {
            // rows
            for j in 0..dim {
                // cols
                matrix[(i, j)] =
                    self.vertices[i + 1].point.coords[j] - self.vertices[0].point.coords[j];
            }
        }

        let a_inv = matrix.try_inverse().ok_or("Singular matrix!")?;

        let mut b: na::SMatrix<T, D, 1> = na::SMatrix::zeros();
        for i in 0..dim {
            b[i] = na::distance_squared(
                &na::Point::from(self.vertices[i + 1].point.coords),
                &na::Point::from(self.vertices[0].point.coords),
            );
        }

        let solution = a_inv * b;

        let mut solution_array: [T; D] = solution
            .data
            .as_slice()
            .try_into()
            .expect("Failed to convert solution to array");

        for value in solution_array.iter_mut() {
            *value /= na::convert::<f64, T>(2.0);
        }

        let solution_point: Point<f64, D> = Point::<f64, D>::from(solution_array);

        Ok(Point::<f64, D>::new(solution_point.coords))
    }

    /// The function `circumradius` returns the circumradius of the cell.
    /// The circumradius is the distance from the circumcenter to any vertex.
    ///
    /// # Returns:
    /// If successful, returns an Ok containing the circumradius of the cell, otherwise returns an Err with an error message.
    fn circumradius(&self) -> Result<T, &'static str>
    where
        T: Clone + Copy + Default,
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let circumcenter = self.circumcenter()?;
        // Change the type of vertex to match circumcenter
        let vertex = Point::<f64, D>::from(self.vertices[0].point.coords);
        Ok(na::distance(
            &na::Point::<T, D>::from(circumcenter.coords),
            &na::Point::<T, D>::from(vertex.coords),
        ))
    }

    /// The function `circumsphere_contains` checks if a given vertex is contained in the circumsphere of the Cell.
    ///
    /// # Arguments:
    ///
    /// * `vertex`: vertex to check.
    ///
    /// # Returns:
    ///
    /// Returns `true` if the given `Vertex` is contained in the circumsphere of the `Cell`, and `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::Cell;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
    /// let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
    /// let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
    /// let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
    /// let cell = Cell::new_with_data(vec![vertex1, vertex2, vertex3, vertex4], "three-one cell").unwrap();
    /// assert!(cell.circumsphere_contains(Vertex::new(Point::origin())).unwrap());
    /// ```
    pub fn circumsphere_contains(&self, vertex: Vertex<T, U, D>) -> Result<bool, &'static str>
    where
        T: Clone + Copy + Default + PartialOrd,
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let circumradius = self.circumradius()?;
        let radius = na::distance(
            &na::Point::<T, D>::from(self.circumcenter()?.coords),
            &na::Point::<T, D>::from(Point::<f64, D>::from(vertex.point.coords).coords),
        );

        Ok(circumradius >= radius)
    }
}

#[cfg(test)]
mod tests {

    use crate::delaunay_core::point::Point;

    use super::*;

    #[test]
    fn cell_new() {
        let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
        let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
        let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
        let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
        let cell: Cell<f64, i32, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();

        assert_eq!(cell.vertices[0], vertex1);
        assert_eq!(cell.vertices[1], vertex2);
        assert_eq!(cell.vertices[2], vertex3);
        assert_eq!(cell.vertices[3], vertex4);
        assert_eq!(cell.vertices[0].data.unwrap(), 1);
        assert_eq!(cell.vertices[1].data.unwrap(), 1);
        assert_eq!(cell.vertices[2].data.unwrap(), 1);
        assert_eq!(cell.vertices[3].data.unwrap(), 2);
        assert_eq!(cell.dim(), 3);
        assert_eq!(cell.number_of_vertices(), 4);
        assert!(cell.neighbors.is_none());
        assert!(cell.data.is_none());

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn cell_new_with_too_many_vertices() {
        let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
        let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
        let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
        let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
        let vertex5 = Vertex::new_with_data(Point::new([2.0, 2.0, 2.0]), 3);
        let cell: Result<Cell<f64, i32, Option<()>, 3>, &'static str> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4, vertex5]);

        assert!(cell.is_err());

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", cell);
    }

    #[test]
    fn cell_new_with_data() {
        let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
        let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
        let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
        let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
        let cell = Cell::new_with_data(vec![vertex1, vertex2, vertex3, vertex4], "three-one cell")
            .unwrap();

        assert_eq!(cell.vertices[0], vertex1);
        assert_eq!(cell.vertices[1], vertex2);
        assert_eq!(cell.vertices[2], vertex3);
        assert_eq!(cell.vertices[3], vertex4);
        assert_eq!(cell.vertices[0].data.unwrap(), 1);
        assert_eq!(cell.vertices[1].data.unwrap(), 1);
        assert_eq!(cell.vertices[2].data.unwrap(), 1);
        assert_eq!(cell.vertices[3].data.unwrap(), 2);
        assert_eq!(cell.dim(), 3);
        assert_eq!(cell.number_of_vertices(), 4);
        assert!(cell.neighbors.is_none());
        assert!(cell.data.is_some());
        assert_eq!(cell.data.unwrap(), "three-one cell");

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn cell_new_with_data_with_too_many_vertices() {
        let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
        let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
        let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
        let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
        let vertex5 = Vertex::new_with_data(Point::new([2.0, 2.0, 2.0]), 3);
        let cell = Cell::new_with_data(
            vec![vertex1, vertex2, vertex3, vertex4, vertex5],
            "three-one cell",
        );

        assert!(cell.is_err());

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", cell);
    }

    #[test]
    fn cell_clone() {
        let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
        let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
        let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
        let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
        let cell1: Cell<f64, i32, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
        let cell2 = cell1.clone();

        assert_eq!(cell1, cell2);
    }

    #[test]
    fn cell_into_hashmap() {
        let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
        let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
        let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
        let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
        let cell: Cell<f64, i32, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
        let hashmap = Cell::into_hashmap(vec![cell.clone()]);
        let values: Vec<Cell<f64, i32, Option<()>, 3>> = hashmap.into_values().collect();

        assert_eq!(values.len(), 1);
        assert_eq!(values[0], cell);

        // Human readable output for cargo test -- --nocapture
        println!("values: {:?}", values);
        println!("cells = {:?}", cell);
    }

    #[test]
    fn cell_number_of_vertices() {
        let vertex1 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
        let vertex2 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
        let vertex3 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3]).unwrap();

        assert_eq!(cell.number_of_vertices(), 3);
    }

    #[test]
    fn cell_dim() {
        let vertex1 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
        let vertex2 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
        let vertex3 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3]).unwrap();

        assert_eq!(cell.dim(), 2);
    }

    #[test]
    fn cell_contains_vertex() {
        let vertex1 = Vertex::new_with_data(Point::new([0.0, 0.0, 1.0]), 1);
        let vertex2 = Vertex::new_with_data(Point::new([0.0, 1.0, 0.0]), 1);
        let vertex3 = Vertex::new_with_data(Point::new([1.0, 0.0, 0.0]), 1);
        let vertex4 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 2);
        let cell: Cell<f64, i32, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();

        assert!(cell.contains_vertex(vertex1));
        assert!(cell.contains_vertex(vertex2));
        assert!(cell.contains_vertex(vertex3));
        assert!(cell.contains_vertex(vertex4));

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn cell_circumcenter() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            Cell::new(Vertex::from_points(points)).unwrap();
        let circumcenter = cell.circumcenter().unwrap();

        assert_eq!(circumcenter, Point::new([0.5, 0.5, 0.5]));

        // Human readable output for cargo test -- --nocapture
        println!("Circumcenter: {:?}", circumcenter);
    }

    #[test]
    fn cell_circumcenter_fail() {
        let point1 = Point::new([0.0, 0.0, 0.0]);
        let point2 = Point::new([1.0, 0.0, 0.0]);
        let point3 = Point::new([0.0, 1.0, 0.0]);
        let vertex1 = Vertex::new_with_data(point1, 1);
        let vertex2 = Vertex::new_with_data(point2, 1);
        let vertex3 = Vertex::new_with_data(point3, 1);
        let cell: Cell<f64, i32, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3]).unwrap();
        let circumcenter = cell.circumcenter();

        assert!(circumcenter.is_err());
    }

    #[test]
    fn cell_circumradius() {
        let point1 = Point::new([0.0, 0.0, 0.0]);
        let point2 = Point::new([1.0, 0.0, 0.0]);
        let point3 = Point::new([0.0, 1.0, 0.0]);
        let point4 = Point::new([0.0, 0.0, 1.0]);
        let vertex1 = Vertex::new_with_data(point1, 1);
        let vertex2 = Vertex::new_with_data(point2, 1);
        let vertex3 = Vertex::new_with_data(point3, 1);
        let vertex4 = Vertex::new_with_data(point4, 2);
        let cell: Cell<f64, i32, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
        let circumradius = cell.circumradius().unwrap();
        let radius: f64 = 3.0_f64.sqrt() / 2.0;

        assert_eq!(circumradius, radius);

        // Human readable output for cargo test -- --nocapture
        println!("Circumradius: {:?}", circumradius);
    }

    #[test]
    fn cell_circumsphere_contains() {
        let point1 = Point::new([0.0, 0.0, 0.0]);
        let point2 = Point::new([1.0, 0.0, 0.0]);
        let point3 = Point::new([0.0, 1.0, 0.0]);
        let point4 = Point::new([0.0, 0.0, 1.0]);
        let vertex1 = Vertex::new_with_data(point1, 1);
        let vertex2 = Vertex::new_with_data(point2, 1);
        let vertex3 = Vertex::new_with_data(point3, 1);
        let vertex4 = Vertex::new_with_data(point4, 2);
        let cell: Cell<f64, i32, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
        let vertex5 = Vertex::new_with_data(Point::new([1.0, 1.0, 1.0]), 3);

        assert!(cell.circumsphere_contains(vertex5).unwrap());

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn cell_circumsphere_does_not_contain() {
        let point1 = Point::new([0.0, 0.0, 0.0]);
        let point2 = Point::new([1.0, 0.0, 0.0]);
        let point3 = Point::new([0.0, 1.0, 0.0]);
        let point4 = Point::new([0.0, 0.0, 1.0]);
        let vertex1 = Vertex::new_with_data(point1, 1);
        let vertex2 = Vertex::new_with_data(point2, 1);
        let vertex3 = Vertex::new_with_data(point3, 1);
        let vertex4 = Vertex::new_with_data(point4, 2);
        let cell: Cell<f64, i32, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
        let vertex5 = Vertex::new_with_data(Point::new([2.0, 2.0, 2.0]), 3);

        assert!(!cell.circumsphere_contains(vertex5).unwrap());

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn cell_to_and_from_json() {
        let vertex1 = Vertex::new(Point::new([0.0, 0.0, 1.0]));
        let vertex2 = Vertex::new(Point::new([0.0, 1.0, 0.0]));
        let vertex3 = Vertex::new(Point::new([1.0, 0.0, 0.0]));
        let vertex4 = Vertex::new(Point::new([1.0, 1.0, 1.0]));
        let cell: Cell<f64, Option<()>, Option<()>, 3> =
            Cell::new(vec![vertex1, vertex2, vertex3, vertex4]).unwrap();
        let serialized = serde_json::to_string(&cell).unwrap();

        assert!(serialized.contains("[0.0,0.0,1.0]"));
        assert!(serialized.contains("[0.0,1.0,0.0]"));
        assert!(serialized.contains("[1.0,0.0,0.0]"));
        assert!(serialized.contains("[1.0,1.0,1.0]"));

        let deserialized: Cell<f64, Option<()>, Option<()>, 3> =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, cell);

        // Human readable output for cargo test -- --nocapture
        println!("Serialized: {:?}", serialized);
    }
}

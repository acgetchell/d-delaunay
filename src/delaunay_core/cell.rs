//! Data and operations on d-dimensional cells or [simplices](https://en.wikipedia.org/wiki/Simplex).

use std::{
    fmt::Debug,
    ops::{Div, Sub},
};

use uuid::Uuid;

use super::{point::Point, utilities::make_uuid, vertex::Vertex};

extern crate nalgebra as na;

use na::ComplexField;

use crate::delaunay_core;

#[derive(Debug, Clone)]
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
pub struct Cell<T, U, V, const D: usize> {
    /// The vertices of the cell.
    pub vertices: Vec<Vertex<T, U, D>>,
    /// The unique identifier of the cell.
    pub uuid: Uuid,
    /// The neighboring cells connected to the current cell.
    pub neighbors: Option<Vec<Uuid>>,
    /// The optional data associated with the cell.
    pub data: Option<V>,
}

impl<
        T: delaunay_core::cell::na::ComplexField<RealField = T> + std::iter::Sum,
        U,
        V,
        const D: usize,
    > Cell<T, U, V, D>
where
    for<'a> &'a T: Div<f64>,
    f64: From<T>,
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

    /// The function returns the number of vertices in the `Cell`.
    ///
    /// # Returns:
    ///
    /// The number of vertices in the `Cell`.
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// The `dim` function returns the dimensionality of the `Cell`.
    ///
    /// # Returns:
    ///
    /// The `dim` function returns the dimension, which is calculated by subtracting 1 from
    /// the number of vertices in the `Cell`.
    pub fn dim(&self) -> usize {
        self.vertices.len() - 1
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
        T: Clone
            + Copy
            + PartialEq
            + num_traits::Zero
            + Debug
            + Sub<Output = T>
            + ComplexField
            + 'static,
    {
        let dim = self.dim();
        if self.vertices[0].dim() != dim {
            return Err("Not a simplex!");
        }

        let mut matrix: na::SMatrix<T, D, D> = na::SMatrix::zeros();
        for i in 0..dim {
            // rows 0..dim-1
            for j in 0..dim {
                // cols 0..dim-1
                matrix[(i, j)] =
                    self.vertices[i + 1].point.coords[j] - self.vertices[0].point.coords[j];
            }
        }

        let a_inv = matrix.try_inverse().ok_or("Singular matrix!")?;

        println!("Matrix A: {:?}", matrix);
        for i in 0..dim {
            println!("Row sum of {}-th row is {}", i, matrix.row(i).row_sum());
        }

        let mut b: na::SMatrix<T, D, 1> = na::SMatrix::zeros();

        for i in 0..dim {
            b[i] = na::distance_squared(
                &na::Point::from(self.vertices[i + 1].point.coords),
                &na::Point::from(self.vertices[0].point.coords),
            );
        }

        println!("Vector b: {:?}", b);

        let solution = a_inv * b;

        let mut solution_array: [T; D] = solution
            .data
            .as_slice()
            .try_into()
            .expect("Failed to convert solution to array");

        for i in 0..dim {
            solution_array[i] /= na::convert::<f64, T>(2.0);
        }

        println!("Solution: {:?}", solution_array);

        let solution_point: Point<f64, D> = Point::<f64, D>::try_from(solution_array)
            .expect("Failed to convert solution array to Point");

        // let solution_point = Point::<f64, D>::try_from(solution_array);
        Ok(Point::<f64, D>::new(solution_point.coords))
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
    pub fn circumsphere_contains(&self, _vertex: Vertex<T, U, D>) -> bool
    where
        T: PartialEq,
        U: PartialEq,
    {
        todo!("Implement circumsphere_contains")
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
}

#[cfg(test)]
mod tests {

    use crate::delaunay_core::point::Point;

    use super::*;

    #[test]
    fn make_cell_with_data() {
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
    fn make_cell_with_data_with_too_many_vertices() {
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
    fn make_cell_without_data() {
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
    fn make_cell_without_data_with_too_many_vertices() {
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

        let circumcenter = cell.circumcenter().unwrap();

        assert_eq!(circumcenter, Point::new([0.5, 0.5, 0.5]));

        // Human readable output for cargo test -- --nocapture
        println!("Circumcenter: {:?}", circumcenter);
    }
}

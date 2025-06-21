//! Data and operations on d-dimensional cells or [simplices](https://en.wikipedia.org/wiki/Simplex).

use super::{
    facet::Facet,
    matrix::invert,
    point::Point,
    utilities::{make_uuid, vec_to_array},
    vertex::Vertex,
};
use na::{ComplexField, Const, OPoint};
use nalgebra as na;
use peroxide::fuga::*;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{collections::HashMap, fmt::Debug, hash::Hash, iter::Sum};
use uuid::Uuid;

#[derive(Builder, Clone, Debug, Default, Deserialize, Serialize)]
#[builder(build_fn(validate = "Self::validate"))]
/// The [Cell] struct represents a d-dimensional
/// [simplex](https://en.wikipedia.org/wiki/Simplex) with vertices, a unique
/// identifier, optional neighbors, and optional data.
///
/// # Properties:
///
/// * `vertices`: A container of vertices. Each [Vertex] has a type T, optional
///   data U, and a constant D representing the number of dimensions.
/// * `uuid`: The `uuid` property is of type [Uuid] and represents a
///   universally unique identifier for a [Cell] in order to identify
///   each instance.
/// * `neighbors`: The `neighbors` property is an optional container of [Uuid]
///   values. It represents the [Uuid]s of the neighboring cells that are connected
///   to the current [Cell], indexed such that the `i-th` neighbor is opposite the
///   `i-th`` [Vertex].
/// * `data`: The `data` property is an optional field that can hold a value of
///   type `V`. It allows storage of additional data associated with the [Cell];
///   the data must implement [Eq], [Hash], [Ord], [PartialEq], and [PartialOrd].
pub struct Cell<T, U, V, const D: usize>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The vertices of the cell.
    pub vertices: Vec<Vertex<T, U, D>>,
    /// The unique identifier of the cell.
    #[builder(setter(skip), default = "make_uuid()")]
    pub uuid: Uuid,
    /// The neighboring cells connected to the current cell.
    #[builder(setter(skip), default = "None")]
    pub neighbors: Option<Vec<Uuid>>,
    /// The optional data associated with the cell.
    #[builder(setter(into, strip_option), default)]
    pub data: Option<V>,
}

impl<T, U, V, const D: usize> CellBuilder<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    fn validate(&self) -> Result<(), CellBuilderError> {
        if self
            .vertices
            .as_ref()
            .expect("Must create a Cell with Vertices!")
            .len()
            > D + 1
        {
            let err = CellBuilderError::ValidationError(
                "Number of vertices must be less than or equal to D + 1!".to_string(),
            );
            Err(err)
        } else {
            Ok(())
        }
    }
}

// Basic implementation block for simpler methods that don't require ComplexField
impl<T, U, V, const D: usize> Cell<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The function returns the number of vertices in the [Cell].
    ///
    /// # Returns:
    ///
    /// The number of vertices in the [Cell].
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3]).build().unwrap();
    /// assert_eq!(cell.number_of_vertices(), 3);
    /// ```
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// The `dim` function returns the dimensionality of the [Cell].
    ///
    /// # Returns:
    ///
    /// The `dim` function returns the dimension, which is calculated by
    /// subtracting 1 from the number of vertices in the [Cell].
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3]).build().unwrap();
    /// assert_eq!(cell.dim(), 2);
    /// ```
    pub fn dim(&self) -> usize {
        self.vertices.len() - 1
    }

    /// The function `contains_vertex` checks if a given vertex is present in
    /// the Cell.
    ///
    /// # Arguments:
    ///
    /// * vertex: The [Vertex] to check.
    ///
    /// # Returns:
    ///
    /// Returns `true` if the given [Vertex] is present in the [Cell], and
    /// `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(1).build().unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).data(2).build().unwrap();
    /// let cell: Cell<f64, i32, &str, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).data("three-one cell").build().unwrap();
    /// assert!(cell.contains_vertex(vertex1));
    /// ```
    pub fn contains_vertex(&self, vertex: Vertex<T, U, D>) -> bool {
        self.vertices.contains(&vertex)
    }

    /// The function `contains_vertex_of` checks if the [Cell] contains any [Vertex] of a given [Cell].
    ///
    /// # Arguments:
    ///
    /// * `cell`: The [Cell] to check.
    ///
    /// # Returns:
    ///
    /// Returns `true` if the given [Cell] has any [Vertex] in common with the [Cell].
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(1).build().unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).data(2).build().unwrap();
    /// let cell: Cell<f64, i32, &str, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).data("three-one cell").build().unwrap();
    /// let vertex5: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(0).build().unwrap();
    /// let cell2: Cell<f64, i32, &str, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex5]).data("one-three cell").build().unwrap();
    /// assert!(cell.contains_vertex_of(&cell2));
    pub fn contains_vertex_of(&self, cell: &Cell<T, U, V, D>) -> bool {
        self.vertices.iter().any(|v| cell.vertices.contains(v))
    }
}

// Advanced implementation block for methods that require ComplexField
impl<T, U, V, const D: usize> Cell<T, U, V, D>
where
    T: Clone + ComplexField<RealField = T> + Copy + Default + PartialEq + PartialOrd + Sum,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    f64: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The function `from_facet_and_vertex` creates a new [Cell] object from a [Facet] and a [Vertex].
    ///
    /// # Arguments:
    ///
    /// * `facet`: The [Facet] to be used to create the [Cell].
    /// * `vertex`: The [Vertex] to be added to the [Cell].
    ///
    /// # Returns:
    ///
    /// A [Result] type containing the new [Cell] or an error message.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::facet::Facet;
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let vertex4 = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).build().unwrap();
    /// let facet = Facet::new(cell.clone(), vertex4).unwrap();
    /// let vertex5 = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).build().unwrap();
    /// let new_cell = Cell::from_facet_and_vertex(facet, vertex5).unwrap();
    /// assert!(new_cell.vertices.contains(&vertex5));
    pub fn from_facet_and_vertex(
        facet: Facet<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<Self, anyhow::Error> {
        let mut vertices = facet.vertices();
        vertices.push(vertex);
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

    /// The function `into_hashmap` converts a [Vec] of cells into a [HashMap],
    /// using the [Cell] [Uuid]s as keys.
    pub fn into_hashmap(cells: Vec<Self>) -> HashMap<Uuid, Self> {
        cells.into_iter().map(|c| (c.uuid, c)).collect()
    }

    /// The function is_valid checks if a [Cell] is valid.
    /// struct.
    ///
    /// # Returns:
    ///
    /// True if the [Cell] is valid; the `Vertices` are correct, the `UUID` is
    /// valid and unique, the `neighbors` contains `UUID`s of neighboring
    /// [Cell]s, and the `neighbors` are indexed such that the index of the
    /// [Vertex] opposite the neighboring cell is the same.
    pub fn is_valid(self) -> bool {
        todo!("Implement is_valid for Cell")
    }

    /// The function `circumcenter` returns the circumcenter of the cell.
    ///
    /// Using the approach from:
    ///
    /// Lévy, Bruno, and Yang Liu.
    /// "Lp Centroidal Voronoi Tessellation and Its Applications."
    /// ACM Transactions on Graphics 29, no. 4 (July 26, 2010): 119:1-119:11.
    /// <https://doi.org/10.1145/1778765.1778856>.
    ///
    /// The circumcenter C of a cell with vertices x_0, x_1, ..., x_n is the
    /// solution to the system:
    ///
    /// C = 1/2 (A^-1*B)
    ///
    /// Where:
    ///
    /// A is a matrix (to be inverted) of the form:
    ///     (x_1-x0) for all coordinates in x1, x0
    ///     (x2-x0) for all coordinates in x2, x0
    ///     ... for all x_n in the cell
    ///
    /// These are the perpendicular bisectors of the edges of the cell.
    ///
    /// And:
    ///
    /// B is a vector of the form:
    ///     (x_1^2-x0^2) for all coordinates in x1, x0
    ///     (x_2^2-x0^2) for all coordinates in x2, x0
    ///     ... for all x_n in the cell
    ///
    /// The resulting vector gives the coordinates of the circumcenter.
    ///
    /// # Returns:
    ///
    /// If the function is successful, it will return an Ok variant containing
    /// the circumcenter as a Point<f64, D> value. If there is an error, it
    /// will return an Err variant containing an error message.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
    /// let cell: Cell<f64, i32, &str, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).data("three-one cell").build().unwrap();
    /// let circumcenter = cell.circumcenter().unwrap();
    /// assert_eq!(circumcenter, Point::new([0.5, 0.5, 0.5]));
    /// ```
    pub fn circumcenter(&self) -> Result<Point<f64, D>, anyhow::Error>
    where
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let dim = self.dim();
        if self.vertices[0].dim() != dim {
            return Err(anyhow::Error::msg("Not a simplex!"));
        }

        let mut matrix = zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                let coords_i_plus_1: [T; D] = (&self.vertices[i + 1]).into();
                let coords_0: [T; D] = (&self.vertices[0]).into();
                matrix[(i, j)] = (coords_i_plus_1[j] - coords_0[j]).into();
            }
        }

        let a_inv = invert(&matrix)?;

        let mut b = zeros(dim, 1);
        for i in 0..dim {
            // Use implicit conversion from vertex to coordinates, then convert to f64
            let coords_i_plus_1: [T; D] = (&self.vertices[i + 1]).into();
            let coords_0: [T; D] = (&self.vertices[0]).into();
            let coords_i_plus_1_f64: [f64; D] = coords_i_plus_1.map(|x| x.into());
            let coords_0_f64: [f64; D] = coords_0.map(|x| x.into());
            b[(i, 0)] = na::distance_squared(
                &na::Point::from(coords_i_plus_1_f64),
                &na::Point::from(coords_0_f64),
            );
        }

        let solution = a_inv * b * 0.5;

        let solution_vec = solution.col(0).to_vec();
        let solution_array = vec_to_array(solution_vec).map_err(anyhow::Error::msg)?;

        let solution_point: Point<f64, D> = Point::<f64, D>::from(solution_array);

        Ok(solution_point)
    }

    /// The function `circumradius` returns the circumradius of the cell.
    /// The circumradius is the distance from the circumcenter to any vertex.
    ///
    /// # Returns:
    ///
    /// If successful, returns an Ok containing the circumradius of the cell,
    /// otherwise returns an Err with an error message.
    fn circumradius(&self) -> Result<T, anyhow::Error>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let circumcenter = self.circumcenter()?;
        // Use implicit conversion from vertex to coordinates, then convert to f64
        let vertex_coords: [T; D] = (&self.vertices[0]).into();
        let vertex_coords_f64: [f64; D] = vertex_coords.map(|x| x.into());
        Ok(na::distance(
            &na::Point::<T, D>::from(circumcenter.coordinates()),
            &na::Point::<T, D>::from(vertex_coords_f64),
        ))
    }

    /// Alternative method that accepts precomputed circumcenter
    fn circumradius_with_center(&self, circumcenter: &Point<f64, D>) -> Result<T, anyhow::Error>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Use implicit conversion from vertex to coordinates, then convert to f64
        let vertex_coords: [T; D] = (&self.vertices[0]).into();
        let vertex_coords_f64: [f64; D] = vertex_coords.map(|x| x.into());
        Ok(na::distance(
            &na::Point::<T, D>::from(circumcenter.coordinates()),
            &na::Point::<T, D>::from(vertex_coords_f64),
        ))
    }

    /// The function `circumsphere_contains` checks if a given vertex is
    /// contained in the circumsphere of the Cell.
    ///
    /// # Arguments:
    ///
    /// * `vertex`: vertex to check.
    ///
    /// # Returns:
    ///
    /// Returns `true` if the given [Vertex] is contained in the circumsphere
    /// of the [Cell], and `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(1).build().unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).data(2).build().unwrap();
    /// let cell: Cell<f64, i32, &str, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).data("three-one cell").build().unwrap();
    /// let origin: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::origin()).build().unwrap();
    /// assert!(cell.circumsphere_contains(origin).unwrap());
    /// ```
    pub fn circumsphere_contains(&self, vertex: Vertex<T, U, D>) -> Result<bool, anyhow::Error>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let circumcenter = self.circumcenter()?;
        let circumradius = self.circumradius_with_center(&circumcenter)?;
        // Use implicit conversion from vertex to coordinates, then convert to f64
        let vertex_coords: [T; D] = (&vertex).into();
        let vertex_coords_f64: [f64; D] = vertex_coords.map(|x| x.into());
        let radius = na::distance(
            &na::Point::<T, D>::from(circumcenter.coordinates()),
            &na::Point::<T, D>::from(vertex_coords_f64),
        );

        Ok(circumradius >= radius)
    }

    /// The function `circumsphere_contains_vertex` checks if a given vertex is
    /// contained in the circumsphere of the Cell using a matrix determinant.
    /// This method is preferred over `circumsphere_contains` as it provides better numerical
    /// stability by using a matrix determinant approach instead of distance calculations,
    /// which can accumulate floating-point errors.
    ///
    /// # Arguments:
    ///
    /// * `vertex`: The [Vertex] to check.
    ///
    /// # Returns:
    ///
    /// Returns `true` if the given [Vertex] is contained in the circumsphere
    /// of the [Cell], and `false` otherwise.
    /// /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(1).build().unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).data(2).build().unwrap();
    /// let cell: Cell<f64, i32, &str, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).data("three-one cell").build().unwrap();
    /// let origin: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::origin()).build().unwrap();
    /// assert!(cell.circumsphere_contains(origin).unwrap());
    /// ```
    pub fn circumsphere_contains_vertex(
        &self,
        vertex: Vertex<T, U, D>,
    ) -> Result<bool, anyhow::Error>
    where
        f64: From<T>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // Setup initial matrix with zeros
        let mut matrix = zeros(D + 1, D + 1);

        // Populate rows with the coordinates of the vertices of the cell
        for (i, v) in self.vertices.iter().enumerate() {
            // Use implicit conversion from vertex to coordinates
            let vertex_coords: [T; D] = v.into();
            for j in 0..D {
                matrix[(i, j)] = vertex_coords[j].into();
            }
            // Add a one to the last column
            matrix[(i, D)] = T::one().into();
        }

        // Add the vertex to the last row of the matrix
        // Use implicit conversion from vertex to coordinates
        let test_vertex_coords: [T; D] = (&vertex).into();
        for j in 0..D {
            matrix[(D, j)] = test_vertex_coords[j].into();
        }
        matrix[(D, D)] = T::one().into();

        // Calculate the determinant of the matrix
        let det = matrix.det();

        // Check if the determinant is positive
        Ok(det > T::zero().into())
    }

    /// The function `facets` returns the [Facet]s of the [Cell].
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// use d_delaunay::delaunay_core::facet::Facet;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(1).build().unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).data(2).build().unwrap();
    /// let cell: Cell<f64, i32, &str, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).data("three-one cell").build().unwrap();
    /// let facets = cell.facets();
    /// assert_eq!(facets.len(), 4);
    pub fn facets(&self) -> Vec<Facet<T, U, V, D>> {
        let mut facets: Vec<Facet<T, U, V, D>> = Vec::new();
        for vertex in self.vertices.iter() {
            facets.push(Facet::new(self.clone(), *vertex).unwrap());
        }

        facets
    }
}

/// Equality of cells is based on equality of sorted vector of vertices.
impl<T, U, V, const D: usize> PartialEq for Cell<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let mut left = self.vertices.clone();
        left.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut right = other.vertices.clone();
        right.sort_by(|a, b| a.partial_cmp(b).unwrap());
        left == right
    }
}

/// Eq implementation for Cell based on equality of sorted vector of vertices.
impl<T, U, V, const D: usize> Eq for Cell<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
}

/// Order of cells is based on lexicographic order of sorted vector of vertices.
impl<T, U, V, const D: usize> PartialOrd for Cell<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let mut left = self.vertices.clone();
        left.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut right = other.vertices.clone();
        right.sort_by(|a, b| a.partial_cmp(b).unwrap());
        left.partial_cmp(&right)
    }
}

/// Custom Hash implementation for Cell
impl<T, U, V, const D: usize> Hash for Cell<T, U, V, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Point<T, D>: Hash, // Add this bound to ensure Point implements Hash
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the UUID first
        self.uuid.hash(state);

        // Sort vertices for consistent hashing regardless of order
        let mut sorted_vertices = self.vertices.clone();
        // The trait bounds on `V` include both `PartialOrd` and `Ord`, which guarantee a total order for the vertices.
        // This ensures that `partial_cmp` will always return `Some(Ordering)`, making the use of `unwrap()` safe here.
        sorted_vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Hash vertices after sorting
        for vertex in &sorted_vertices {
            vertex.hash(state);
        }

        // Hash neighbors and data
        self.neighbors.hash(state);
        self.data.hash(state);
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::delaunay_core::{point::Point, vertex::VertexBuilder};

    #[test]
    fn cell_build() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();

        assert_eq!(cell.vertices, [vertex1, vertex2, vertex3, vertex4]);
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
    #[should_panic(expected = "Must create a Cell with Vertices")]
    fn cell_empty() {
        let _cell: Result<Cell<f64, i32, Option<()>, 3>, CellBuilderError> =
            CellBuilder::default().build();
    }

    #[test]
    fn cell_build_with_too_many_vertices() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([2.0, 2.0, 2.0]))
            .data(3)
            .build()
            .unwrap();
        let cell: Result<Cell<f64, i32, Option<()>, 3>, CellBuilderError> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4, vertex5])
            .build();

        assert!(cell.is_err());

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", cell);
    }

    #[test]
    fn cell_build_with_data() {
        let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, &str, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .data("three-one cell")
            .build()
            .unwrap();

        assert_eq!(cell.vertices, [vertex1, vertex2, vertex3, vertex4]);
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
    fn cell_clone() {
        let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell1: Cell<f64, i32, &str, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .data("three-one cell")
            .build()
            .unwrap();
        let cell2 = cell1.clone();

        assert_eq!(cell1, cell2);
    }

    #[test]
    fn cell_from_facet_and_vertex() {
        let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let facet = Facet::new(cell.clone(), vertex4).unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::origin())
            .build()
            .unwrap();
        let new_cell = Cell::from_facet_and_vertex(facet, vertex5).unwrap();

        assert!(new_cell.vertices.contains(&vertex1));
        assert!(new_cell.vertices.contains(&vertex2));
        assert!(new_cell.vertices.contains(&vertex3));
        assert!(new_cell.vertices.contains(&vertex5));

        // Human readable output for cargo test -- --nocapture
        println!("New Cell: {:?}", new_cell);
    }

    #[test]
    fn cell_into_hashmap() {
        let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
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
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 3);
    }

    #[test]
    fn cell_dim() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        assert_eq!(cell.dim(), 2);
    }

    #[test]
    fn cell_contains_vertex() {
        let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();

        assert!(cell.contains_vertex(vertex1));
        assert!(cell.contains_vertex(vertex2));
        assert!(cell.contains_vertex(vertex3));
        assert!(cell.contains_vertex(vertex4));

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn cell_contains_vertex_of() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, &str, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .data("three-one cell")
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::origin())
            .data(0)
            .build()
            .unwrap();
        let cell2 = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex5])
            .data("one-three cell")
            .build()
            .unwrap();

        assert!(cell.contains_vertex_of(&cell2));

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
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(Vertex::from_points(points))
            .build()
            .unwrap();
        let circumcenter = cell.circumcenter().unwrap();

        assert_eq!(circumcenter, Point::new([0.5, 0.5, 0.5]));

        // Human readable output for cargo test -- --nocapture
        println!("Circumcenter: {:?}", circumcenter);
    }

    #[test]
    fn cell_circumcenter_fail() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let circumcenter = cell.circumcenter();

        assert!(circumcenter.is_err());
    }

    #[test]
    fn cell_circumradius() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let circumradius = cell.circumradius().unwrap();
        let radius: f64 = 3.0_f64.sqrt() / 2.0;

        assert_eq!(circumradius, radius);

        // Human readable output for cargo test -- --nocapture
        println!("Circumradius: {:?}", circumradius);
    }

    #[test]
    fn cell_circumsphere_contains() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(3)
            .build()
            .unwrap();

        assert!(cell.circumsphere_contains(vertex5).unwrap());

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn cell_circumsphere_does_not_contain() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([2.0, 2.0, 2.0]))
            .data(3)
            .build()
            .unwrap();

        assert!(!cell.circumsphere_contains(vertex5).unwrap());

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn cell_facets_contains() {
        let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let cell: Cell<f64, i32, Option<&str>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .data("three-one cell")
            .build()
            .unwrap();
        let facets = cell.facets();

        assert_eq!(facets.len(), 4);
        for facet in facets.iter() {
            // assert!(cell.facets().contains(facet));
            let facet_vertices = facet.vertices();
            assert!(cell.facets().iter().any(|f| f.vertices() == facet_vertices));
        }

        // Human readable output for cargo test -- --nocapture
        println!("Facets: {:?}", facets);
    }

    #[test]
    fn cell_to_and_from_json() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
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

    #[test]
    fn cell_partial_eq() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::origin())
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .build()
            .unwrap();
        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let cell2 = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let cell3 = CellBuilder::default()
            .vertices(vec![vertex4, vertex3, vertex2, vertex1])
            .build()
            .unwrap();
        let cell4 = CellBuilder::default()
            .vertices(vec![vertex5, vertex4, vertex3, vertex2])
            .build()
            .unwrap();

        assert_eq!(cell1, cell2);
        // Two cells with the same vertices but different uuids are equal
        assert_ne!(cell1.uuid, cell2.uuid);
        assert_eq!(cell1.vertices, cell2.vertices);
        assert_eq!(cell2, cell3);
        assert_ne!(cell3, cell4);
    }

    #[test]
    fn cell_partial_ord() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::origin())
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .build()
            .unwrap();
        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();
        let cell2 = CellBuilder::default()
            .vertices(vec![vertex4, vertex3, vertex2, vertex1])
            .build()
            .unwrap();
        let cell3 = CellBuilder::default()
            .vertices(vec![vertex5, vertex4, vertex3, vertex2])
            .build()
            .unwrap();

        assert!(cell1 < cell3);
        assert!(cell2 < cell3);
        assert!(cell3 > cell1);
        assert!(cell3 > cell2);
    }

    #[test]
    fn cell_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let cell2: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        cell1.hash(&mut hasher1);
        cell2.hash(&mut hasher2);

        // Different UUIDs mean different hashes even with same vertices
        assert_ne!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn cell_hash_in_hashmap() {
        use std::collections::HashMap;

        let mut map: HashMap<Cell<i32, Option<()>, Option<()>, 2>, i32> = HashMap::new();

        let vertex1 = VertexBuilder::default()
            .point(Point::new([0, 0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1, 0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0, 1]))
            .build()
            .unwrap();

        let cell1: Cell<i32, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let cell2: Cell<i32, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();

        map.insert(cell1.clone(), 10);
        map.insert(cell2.clone(), 20);

        assert_eq!(map.get(&cell1), Some(&10));
        assert_eq!(map.get(&cell2), Some(&20));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn cell_default() {
        let cell: Cell<f64, Option<()>, Option<()>, 3> = Default::default();

        assert!(cell.vertices.is_empty());
        assert!(cell.uuid.is_nil());
        assert!(cell.neighbors.is_none());
        assert!(cell.data.is_none());
    }

    #[test]
    fn cell_1d() {
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

        assert_eq!(cell.number_of_vertices(), 2);
        assert_eq!(cell.dim(), 1);
        assert!(!cell.uuid.is_nil());
    }

    #[test]
    fn cell_2d() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 3);
        assert_eq!(cell.dim(), 2);
        assert!(!cell.uuid.is_nil());
    }

    #[test]
    fn cell_4d() {
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

        assert_eq!(cell.number_of_vertices(), 5);
        assert_eq!(cell.dim(), 4);
        assert!(!cell.uuid.is_nil());
    }

    #[test]
    fn cell_with_f32() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0f32, 0.0f32]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0f32, 0.0f32]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0f32, 1.0f32]))
            .build()
            .unwrap();

        let cell: Cell<f32, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 3);
        assert_eq!(cell.dim(), 2);
    }

    #[test]
    fn cell_with_integers() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0i32, 0i32, 0i32]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1i32, 0i32, 0i32]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0i32, 1i32, 0i32]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0i32, 0i32, 1i32]))
            .build()
            .unwrap();

        let cell: Cell<i32, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 4);
        assert_eq!(cell.dim(), 3);
    }

    #[test]
    fn cell_single_vertex() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 1);
        assert_eq!(cell.dim(), 0);
    }

    #[test]
    fn cell_uuid_uniqueness() {
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

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let cell2: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        // Same vertices but different UUIDs
        assert_ne!(cell1.uuid, cell2.uuid);
        assert!(!cell1.uuid.is_nil());
        assert!(!cell2.uuid.is_nil());
    }

    #[test]
    fn cell_neighbors_none_by_default() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();

        assert!(cell.neighbors.is_none());
    }

    #[test]
    fn cell_data_none_by_default() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1])
            .build()
            .unwrap();

        assert!(cell.data.is_none());
    }

    #[test]
    fn cell_data_can_be_set() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, i32, 3> = CellBuilder::default()
            .vertices(vec![vertex1])
            .data(42)
            .build()
            .unwrap();

        assert_eq!(cell.data.unwrap(), 42);
    }

    #[test]
    fn cell_into_hashmap_empty() {
        let cells: Vec<Cell<f64, Option<()>, Option<()>, 3>> = Vec::new();
        let hashmap = Cell::into_hashmap(cells);

        assert!(hashmap.is_empty());
    }

    #[test]
    fn cell_into_hashmap_multiple() {
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

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();
        let cell2: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex2, vertex3])
            .build()
            .unwrap();

        let uuid1 = cell1.uuid;
        let uuid2 = cell2.uuid;
        let cells = vec![cell1, cell2];
        let hashmap = Cell::into_hashmap(cells);

        assert_eq!(hashmap.len(), 2);
        assert!(hashmap.contains_key(&uuid1));
        assert!(hashmap.contains_key(&uuid2));
    }

    #[test]
    fn cell_debug_format() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([4.0, 5.0, 6.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, i32, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .data(42)
            .build()
            .unwrap();
        let debug_str = format!("{:?}", cell);

        assert!(debug_str.contains("Cell"));
        assert!(debug_str.contains("vertices"));
        assert!(debug_str.contains("uuid"));
        assert!(debug_str.contains("1.0"));
        assert!(debug_str.contains("2.0"));
        assert!(debug_str.contains("3.0"));
    }

    #[test]
    fn cell_comprehensive_serialization() {
        // Test with different data types and dimensions
        let vertex_2d_1 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0]))
            .build()
            .unwrap();
        let vertex_2d_2 = VertexBuilder::default()
            .point(Point::new([3.0, 4.0]))
            .build()
            .unwrap();
        let vertex_2d_3 = VertexBuilder::default()
            .point(Point::new([5.0, 6.0]))
            .build()
            .unwrap();

        let cell_2d: Cell<f64, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex_2d_1, vertex_2d_2, vertex_2d_3])
            .build()
            .unwrap();
        let serialized_2d = serde_json::to_string(&cell_2d).unwrap();
        let deserialized_2d: Cell<f64, Option<()>, Option<()>, 2> =
            serde_json::from_str(&serialized_2d).unwrap();
        assert_eq!(cell_2d, deserialized_2d);

        let vertex_1d_1 = VertexBuilder::default()
            .point(Point::new([42.0]))
            .build()
            .unwrap();
        let vertex_1d_2 = VertexBuilder::default()
            .point(Point::new([84.0]))
            .build()
            .unwrap();

        let cell_1d: Cell<f64, Option<()>, Option<()>, 1> = CellBuilder::default()
            .vertices(vec![vertex_1d_1, vertex_1d_2])
            .build()
            .unwrap();
        let serialized_1d = serde_json::to_string(&cell_1d).unwrap();
        let deserialized_1d: Cell<f64, Option<()>, Option<()>, 1> =
            serde_json::from_str(&serialized_1d).unwrap();
        assert_eq!(cell_1d, deserialized_1d);
    }

    #[test]
    fn cell_negative_coordinates() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([-1.0, -2.0, -3.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([-4.0, -5.0, -6.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 2);
        assert_eq!(cell.dim(), 1);
    }

    #[test]
    fn cell_large_coordinates() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([1e6, 2e6, 3e6]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([4e6, 5e6, 6e6]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 2);
        assert_eq!(cell.dim(), 1);
    }

    #[test]
    fn cell_small_coordinates() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([1e-6, 2e-6, 3e-6]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([4e-6, 5e-6, 6e-6]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 2);
        assert_eq!(cell.dim(), 1);
    }

    #[test]
    fn cell_ordering_edge_cases() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();
        let cell2: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();

        // Test that equal cells are not less than each other
        assert_ne!(cell1.partial_cmp(&cell2), Some(std::cmp::Ordering::Less));
        assert_ne!(cell2.partial_cmp(&cell1), Some(std::cmp::Ordering::Less));
        assert!(cell1 <= cell2);
        assert!(cell2 <= cell1);
        assert!(cell1 >= cell2);
        assert!(cell2 >= cell1);
    }

    #[test]
    fn cell_eq_trait() {
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

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let cell2: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let cell3: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();

        // Test Eq trait (reflexivity, symmetry) - equality is based on vertices only
        assert_eq!(cell1, cell1); // reflexive
        assert_eq!(cell1, cell2); // same vertices
        assert_eq!(cell2, cell1); // symmetric
        assert_ne!(cell1, cell3); // different vertices
        assert_ne!(cell3, cell1); // symmetric
    }

    #[test]
    fn cell_circumcenter_2d() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([2.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let circumcenter = cell.circumcenter().unwrap();

        // For this triangle, circumcenter should be at (1.0, 0.75)
        assert!((circumcenter.coordinates()[0] - 1.0).abs() < 1e-10);
        assert!((circumcenter.coordinates()[1] - 0.75).abs() < 1e-10);
    }

    #[test]
    fn cell_circumradius_2d() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let circumradius = cell.circumradius().unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert!((circumradius - expected_radius).abs() < 1e-10);
    }

    #[test]
    fn cell_mixed_positive_negative_coordinates() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([1.0, -2.0, 3.0, -4.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([-5.0, 6.0, -7.0, 8.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 4> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 2);
        assert_eq!(cell.dim(), 1);
    }

    #[test]
    fn cell_contains_vertex_false() {
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
            .point(Point::new([2.0, 2.0, 2.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();

        assert!(!cell.contains_vertex(vertex4));
    }

    #[test]
    fn cell_contains_vertex_of_false() {
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
            .point(Point::new([2.0, 2.0, 2.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([3.0, 3.0, 3.0]))
            .build()
            .unwrap();

        let cell1: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .build()
            .unwrap();
        let cell2: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex4, vertex5])
            .build()
            .unwrap();

        assert!(!cell1.contains_vertex_of(&cell2));
    }

    #[test]
    fn cell_validation_max_vertices() {
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .build()
            .unwrap();

        // This should work (3 vertices for 2D)
        let cell: Result<Cell<f64, Option<()>, Option<()>, 2>, CellBuilderError> =
            CellBuilder::default()
                .vertices(vec![vertex1, vertex2, vertex3])
                .build();

        assert!(cell.is_ok());
        assert_eq!(cell.unwrap().number_of_vertices(), 3);
    }
}

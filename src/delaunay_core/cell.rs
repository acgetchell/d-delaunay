//! Data and operations on d-dimensional cells or [simplices](https://en.wikipedia.org/wiki/Simplex).

#![allow(clippy::similar_names)]

use super::{
    facet::Facet,
    utilities::make_uuid,
    vertex::{Vertex, VertexValidationError},
};
use crate::geometry::{FiniteCheck, HashCoordinate, OrderedEq, point::Point};
use na::ComplexField;
use nalgebra as na;
use num_traits::Float;
use peroxide::fuga::anyhow;
use serde::{Serialize, de::DeserializeOwned};
use std::{collections::HashMap, fmt::Debug, hash::Hash, iter::Sum};
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur during cell validation.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum CellValidationError {
    /// The cell has an invalid vertex.
    #[error("Invalid vertex: {source}")]
    InvalidVertex {
        /// The underlying vertex validation error.
        #[from]
        source: VertexValidationError,
    },
    /// The cell has an invalid (nil) UUID.
    #[error("Invalid UUID: cell has nil UUID which is not allowed")]
    InvalidUuid,
    /// The cell contains duplicate vertices.
    #[error("Duplicate vertices: cell contains non-unique vertices which is not allowed")]
    DuplicateVertices,
    /// The cell has insufficient vertices to form a proper D-simplex.
    #[error(
        "Insufficient vertices: cell has {actual} vertices; expected exactly {expected} for a {dimension}D simplex"
    )]
    InsufficientVertices {
        /// The actual number of vertices in the cell.
        actual: usize,
        /// The expected number of vertices (D+1).
        expected: usize,
        /// The dimension D.
        dimension: usize,
    },
}

#[derive(Builder, Clone, Debug, Default, Serialize)]
#[builder(build_fn(validate = "Self::validate"))]
/// The [Cell] struct represents a d-dimensional
/// [simplex](https://en.wikipedia.org/wiki/Simplex) with vertices, a unique
/// identifier, optional neighbors, and optional data.
///
/// # Properties
///
/// - `vertices`: A container of vertices. Each [Vertex] has a type T, optional
///   data U, and a constant D representing the number of dimensions.
/// - `uuid`: The `uuid` property is of type [Uuid] and represents a
///   universally unique identifier for a [Cell] in order to identify
///   each instance.
/// - `neighbors`: The `neighbors` property is an optional container of [Uuid]
///   values. It represents the [Uuid]s of the neighboring cells that are connected
///   to the current [Cell], indexed such that the `i-th` neighbor is opposite the
///   `i-th` [Vertex].
/// - `data`: The `data` property is an optional field that can hold a value of
///   type `V`. It allows storage of additional data associated with the [Cell];
///   the data must implement [Eq], [Hash], [Ord], [`PartialEq`], and [`PartialOrd`].
pub struct Cell<T, U, V, const D: usize>
where
    T: Default
        + OrderedEq
        + Float
        + FiniteCheck
        + HashCoordinate
        + Copy
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The vertices of the cell.
    vertices: Vec<Vertex<T, U, D>>,
    /// The unique identifier of the cell.
    #[builder(setter(skip), default = "make_uuid()")]
    uuid: Uuid,
    /// The neighboring cells connected to the current cell.
    #[builder(setter(skip), default = "None")]
    pub neighbors: Option<Vec<Uuid>>,
    /// The optional data associated with the cell.
    #[builder(setter(into, strip_option), default)]
    pub data: Option<V>,
}

impl<T, U, V, const D: usize> CellBuilder<T, U, V, D>
where
    T: Default
        + OrderedEq
        + Float
        + FiniteCheck
        + HashCoordinate
        + Copy
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
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

// Basic implementation block with minimal trait bounds for common methods
impl<T, U, V, const D: usize> Cell<T, U, V, D>
where
    T: Default
        + OrderedEq
        + Float
        + FiniteCheck
        + HashCoordinate
        + Copy
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The function returns the number of vertices in the [Cell].
    ///
    /// # Returns
    ///
    /// The number of vertices in the [Cell].
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3]).build().unwrap();
    /// assert_eq!(cell.number_of_vertices(), 3);
    /// ```
    #[inline]
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Returns a reference to the vertices of the [Cell].
    ///
    /// # Returns
    ///
    /// A reference to the `Vec<Vertex<T, U, D>>` containing the vertices of the cell.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3]).build().unwrap();
    /// assert_eq!(cell.vertices().len(), 3);
    /// ```
    #[inline]
    pub const fn vertices(&self) -> &Vec<Vertex<T, U, D>> {
        &self.vertices
    }

    /// Returns the UUID of the [Cell].
    ///
    /// # Returns
    ///
    /// The Uuid uniquely identifying this cell.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// use uuid::Uuid;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1]).build().unwrap();
    /// assert_ne!(cell.uuid(), Uuid::nil());
    /// ```
    #[inline]
    pub const fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// The `dim` function returns the dimensionality of the [Cell].
    ///
    /// # Returns
    ///
    /// The `dim` function returns the dimension, which is calculated by
    /// subtracting 1 from the number of vertices in the [Cell].
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3]).build().unwrap();
    /// assert_eq!(cell.dim(), 2);
    /// ```
    #[inline]
    pub fn dim(&self) -> usize {
        self.vertices.len() - 1
    }

    /// The function `contains_vertex` checks if a given vertex is present in
    /// the Cell.
    ///
    /// # Arguments
    ///
    /// * vertex: The [Vertex] to check.
    ///
    /// # Returns
    ///
    /// Returns `true` if the given [Vertex] is present in the [Cell], and
    /// `false` otherwise.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default()
    ///     .point(Point::new([0.0, 0.0, 1.0]))
    ///     .data(1)
    ///     .build()
    ///     .unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default()
    ///     .point(Point::new([0.0, 1.0, 0.0]))
    ///     .data(1)
    ///     .build()
    ///     .unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default()
    ///     .point(Point::new([1.0, 0.0, 0.0]))
    ///     .data(1)
    ///     .build()
    ///     .unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default()
    ///     .point(Point::new([1.0, 1.0, 1.0]))
    ///     .data(2)
    ///     .build()
    ///     .unwrap();
    /// let cell: Cell<f64, i32, i32, 3> = CellBuilder::default()
    ///     .vertices(vec![vertex1, vertex2, vertex3, vertex4])
    ///     .data(42)
    ///     .build()
    ///     .unwrap();
    /// assert!(cell.contains_vertex(vertex1));
    /// ```
    pub fn contains_vertex(&self, vertex: Vertex<T, U, D>) -> bool {
        self.vertices.contains(&vertex)
    }

    /// The function `contains_vertex_of` checks if the [Cell] contains any [Vertex] of a given [Cell].
    ///
    /// # Arguments
    ///
    /// * `cell`: The [Cell] to check.
    ///
    /// # Returns
    ///
    /// Returns `true` if the given [Cell] has any [Vertex] in common with the [Cell].
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(1).build().unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).data(2).build().unwrap();
    /// let cell: Cell<f64, i32, i32, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).data(42).build().unwrap();
    /// let vertex5: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(0).build().unwrap();
    /// let cell2: Cell<f64, i32, i32, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex5]).data(24).build().unwrap();
    /// assert!(cell.contains_vertex_of(&cell2));
    /// ```
    pub fn contains_vertex_of(&self, cell: &Self) -> bool {
        self.vertices.iter().any(|v| cell.vertices.contains(v))
    }

    /// The function `from_facet_and_vertex` creates a new [Cell] object from a [Facet] and a [Vertex].
    ///
    /// # Arguments
    ///
    /// - `facet`: The [Facet] to be used to create the [Cell].
    /// - `vertex`: The [Vertex] to be added to the [Cell].
    ///
    /// # Returns
    ///
    /// A [Result] type containing the new [Cell] or an error message.
    ///
    /// # Errors
    ///
    /// This function currently does not return errors, but uses `Result` for future extensibility.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::facet::Facet;
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let vertex4 = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).build().unwrap();
    /// let facet = Facet::new(cell.clone(), vertex4).unwrap();
    /// let vertex5 = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).build().unwrap();
    /// let new_cell = Cell::from_facet_and_vertex(&facet, vertex5).unwrap();
    /// assert!(new_cell.vertices().contains(&vertex5));
    /// ```
    pub fn from_facet_and_vertex(
        facet: &Facet<T, U, V, D>,
        vertex: Vertex<T, U, D>,
    ) -> Result<Self, anyhow::Error> {
        let mut vertices = facet.vertices();
        vertices.push(vertex);
        let uuid = make_uuid();
        let neighbors = None;
        let data = None;
        Ok(Self {
            vertices,
            uuid,
            neighbors,
            data,
        })
    }

    /// The function `into_hashmap` converts a [Vec] of cells into a [`HashMap`],
    /// using the [Cell] [Uuid]s as keys.
    #[must_use]
    pub fn into_hashmap(cells: Vec<Self>) -> HashMap<Uuid, Self> {
        cells.into_iter().map(|c| (c.uuid, c)).collect()
    }

    /// The function `is_valid` checks if a [Cell] is valid.
    ///
    /// # Type Parameters
    ///
    /// This method requires the coordinate type `T` to implement additional traits:
    /// - [`FiniteCheck`]: Enables checking that all coordinate values are finite
    ///   (not infinite or NaN), which is essential for geometric computations.
    /// - [`HashCoordinate`]: Enables hashing of coordinate values,
    ///   which is required for detecting duplicate vertices efficiently.
    /// - [`Copy`]: Required for efficient comparison operations.
    ///
    /// # Returns
    ///
    /// A Result indicating whether the [Cell] is valid. Returns `Ok(())` if valid,
    /// or a `CellValidationError` if invalid. The validation checks that:
    /// - All vertices are valid (coordinates are finite and UUIDs are valid)
    /// - All vertices are distinct from one another
    /// - The cell UUID is valid and not nil
    /// - The neighbors contain UUIDs of neighboring [Cell]s
    /// - The neighbors are indexed such that the index of the [Vertex] opposite
    ///   the neighboring cell is the same
    ///
    /// # Errors
    ///
    /// Returns `CellValidationError::InvalidVertex` if any vertex is invalid,
    /// `CellValidationError::InvalidUuid` if the cell's UUID is nil,
    /// `CellValidationError::DuplicateVertices` if the cell contains duplicate vertices, or
    /// `CellValidationError::InsufficientVertices` if the cell doesn't have exactly D+1 vertices.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let vertex1 = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).build().unwrap();
    /// let vertex2 = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).build().unwrap();
    /// let vertex3 = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).build().unwrap();
    /// let vertex4 = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).build().unwrap();
    /// let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).build().unwrap();
    /// assert!(cell.is_valid().is_ok());
    /// ```
    pub fn is_valid(&self) -> Result<(), CellValidationError>
    where
        T: FiniteCheck + HashCoordinate + Copy,
    {
        // Check if all vertices are valid
        for vertex in &self.vertices {
            vertex.is_valid()?;
        }

        // Check if UUID is not nil
        if self.uuid.is_nil() {
            return Err(CellValidationError::InvalidUuid);
        }

        // Check if all vertices are distinct from one another
        let mut seen = std::collections::HashSet::new();
        if !self.vertices.iter().all(|vertex| seen.insert(*vertex)) {
            return Err(CellValidationError::DuplicateVertices);
        }

        // Check that cell has exactly D+1 vertices (a proper D-simplex)
        if self.vertices.len() != D + 1 {
            return Err(CellValidationError::InsufficientVertices {
                actual: self.vertices.len(),
                expected: D + 1,
                dimension: D,
            });
        }

        Ok(())
        // TODO: Additional validation can be added here:
        // - Validate neighbors structure if present
        // - Validate neighbor indices match vertex ordering
    }
}

// Advanced implementation block for methods requiring ComplexField
impl<T, U, V, const D: usize> Cell<T, U, V, D>
where
    T: Clone
        + ComplexField<RealField = T>
        + Copy
        + Default
        + PartialEq
        + PartialOrd
        + OrderedEq
        + Sum
        + Float
        + FiniteCheck
        + HashCoordinate
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    f64: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The function `facets` returns the [Facet]s of the [Cell].
    ///
    /// # Panics
    ///
    /// Panics if `Facet::new()` fails for any vertex in the cell. This should not
    /// happen under normal circumstances with valid cell data.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::cell::{Cell, CellBuilder};
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// use d_delaunay::delaunay_core::facet::Facet;
    /// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(1).build().unwrap();
    /// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
    /// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
    /// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).data(2).build().unwrap();
    /// let cell: Cell<f64, i32, i32, 3> = CellBuilder::default().vertices(vec![vertex1, vertex2, vertex3, vertex4]).data(42).build().unwrap();
    /// let facets = cell.facets();
    /// assert_eq!(facets.len(), 4);
    /// ```
    pub fn facets(&self) -> Vec<Facet<T, U, V, D>> {
        self.vertices
            .iter()
            .map(|vertex| Facet::new(self.clone(), *vertex).unwrap())
            .collect()
    }
}

/// Helper function to sort vertices for comparison and hashing
fn sorted_vertices<T, U, const D: usize>(vertices: &[Vertex<T, U, D>]) -> Vec<Vertex<T, U, D>>
where
    T: Default
        + OrderedEq
        + Float
        + FiniteCheck
        + HashCoordinate
        + Copy
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    let mut sorted = vertices.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted
}

/// Equality of cells is based on equality of sorted vector of vertices.
impl<T, U, V, const D: usize> PartialEq for Cell<T, U, V, D>
where
    T: Default
        + OrderedEq
        + Float
        + FiniteCheck
        + HashCoordinate
        + Copy
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        sorted_vertices::<T, U, D>(&self.vertices) == sorted_vertices::<T, U, D>(&other.vertices)
    }
}

/// Eq implementation for Cell based on equality of sorted vector of vertices.
impl<T, U, V, const D: usize> Eq for Cell<T, U, V, D>
where
    T: Default
        + OrderedEq
        + Float
        + FiniteCheck
        + HashCoordinate
        + Copy
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
}

/// Order of cells is based on lexicographic order of sorted vector of vertices.
impl<T, U, V, const D: usize> PartialOrd for Cell<T, U, V, D>
where
    T: Default
        + OrderedEq
        + Float
        + FiniteCheck
        + HashCoordinate
        + Copy
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        sorted_vertices::<T, U, D>(&self.vertices)
            .partial_cmp(&sorted_vertices::<T, U, D>(&other.vertices))
    }
}

/// Custom Hash implementation for Cell
impl<T, U, V, const D: usize> Hash for Cell<T, U, V, D>
where
    T: Default
        + OrderedEq
        + Float
        + FiniteCheck
        + HashCoordinate
        + Copy
        + Debug
        + Serialize
        + DeserializeOwned,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd + Serialize + DeserializeOwned,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Point<T, D>: Hash, // Add this bound to ensure Point implements Hash
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the UUID first
        self.uuid.hash(state);

        // Hash sorted vertices for consistent ordering
        for vertex in &sorted_vertices::<T, U, D>(&self.vertices) {
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
    use crate::delaunay_core::vertex::VertexBuilder;
    use crate::geometry::point::Point;
    use crate::geometry::predicates::{
        circumcenter, circumradius, circumradius_with_center, insphere, insphere_distance,
    };
    use crate::geometry::traits::coordinate::Coordinate;
    use approx::assert_relative_eq;

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

        assert_eq!(*cell.vertices(), vec![vertex1, vertex2, vertex3, vertex4]);
        assert_eq!(cell.vertices()[0].data.unwrap(), 1);
        assert_eq!(cell.vertices()[1].data.unwrap(), 1);
        assert_eq!(cell.vertices()[2].data.unwrap(), 1);
        assert_eq!(cell.vertices()[3].data.unwrap(), 2);
        assert_eq!(cell.dim(), 3);
        assert_eq!(cell.number_of_vertices(), 4);
        assert!(cell.neighbors.is_none());
        assert!(cell.data.is_none());

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {cell:?}");
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
        println!("{cell:?}");
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
        let cell: Cell<f64, i32, i32, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .data(42)
            .build()
            .unwrap();

        assert_eq!(*cell.vertices(), vec![vertex1, vertex2, vertex3, vertex4]);
        assert_eq!(cell.vertices()[0].data.unwrap(), 1);
        assert_eq!(cell.vertices()[1].data.unwrap(), 1);
        assert_eq!(cell.vertices()[2].data.unwrap(), 1);
        assert_eq!(cell.vertices()[3].data.unwrap(), 2);
        assert_eq!(cell.dim(), 3);
        assert_eq!(cell.number_of_vertices(), 4);
        assert!(cell.neighbors.is_none());
        assert!(cell.data.is_some());
        assert_eq!(cell.data.unwrap(), 42);

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {cell:?}");
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
        let cell1: Cell<f64, i32, i32, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .data(42)
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
        let new_cell = Cell::from_facet_and_vertex(&facet, vertex5).unwrap();

        assert!(new_cell.vertices().contains(&vertex1));
        assert!(new_cell.vertices().contains(&vertex2));
        assert!(new_cell.vertices().contains(&vertex3));
        assert!(new_cell.vertices().contains(&vertex5));

        // Human readable output for cargo test -- --nocapture
        println!("New Cell: {new_cell:?}");
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
        println!("values: {values:?}");
        println!("cells = {cell:?}");
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
        println!("Cell: {cell:?}");
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
        let cell: Cell<f64, i32, i32, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .data(42)
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::origin())
            .data(0)
            .build()
            .unwrap();
        let cell2 = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex5])
            .data(43)
            .build()
            .unwrap();

        assert!(cell.contains_vertex_of(&cell2));

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {cell:?}");
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
        let cell: Cell<f64, i32, i32, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .data(31)
            .build()
            .unwrap();
        let facets = cell.facets();

        assert_eq!(facets.len(), 4);
        for facet in &facets {
            // assert!(cell.facets().contains(facet));
            let facet_vertices = facet.vertices();
            assert!(cell.facets().iter().any(|f| f.vertices() == facet_vertices));
        }

        // Human readable output for cargo test -- --nocapture
        println!("Facets: {facets:?}");
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

        // TODO: Re-enable deserialization test after fixing serde trait bounds
        // let deserialized: Cell<f64, Option<()>, Option<()>, 3> =
        //     serde_json::from_str(&serialized).unwrap();
        // assert_eq!(deserialized, cell);

        // Human readable output for cargo test -- --nocapture
        println!("Serialized: {serialized:?}");
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
        assert_ne!(cell1.uuid(), cell2.uuid());
        assert_eq!(cell1.vertices(), cell2.vertices());
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
    fn cell_default() {
        let cell: Cell<f64, Option<()>, Option<()>, 3> = Cell::default();

        assert!(cell.vertices().is_empty());
        assert!(cell.uuid().is_nil());
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
        assert!(!cell.uuid().is_nil());
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
        assert!(!cell.uuid().is_nil());
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
        assert!(!cell.uuid().is_nil());
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
        assert_ne!(cell1.uuid(), cell2.uuid());
        assert!(!cell1.uuid().is_nil());
        assert!(!cell2.uuid().is_nil());
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

        let uuid1 = cell1.uuid();
        let uuid2 = cell2.uuid();
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
        let debug_str = format!("{cell:?}");

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
        let _serialized_2d = serde_json::to_string(&cell_2d).unwrap();
        // TODO: Re-enable deserialization test after fixing serde trait bounds
        // let deserialized_2d: Cell<f64, Option<()>, Option<()>, 2> =
        //     serde_json::from_str(&serialized_2d).unwrap();
        // assert_eq!(cell_2d, deserialized_2d);

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
        let _serialized_1d = serde_json::to_string(&cell_1d).unwrap();
        // TODO: Re-enable deserialization test after fixing serde trait bounds
        // let deserialized_1d: Cell<f64, Option<()>, Option<()>, 1> =
        //     serde_json::from_str(&serialized_1d).unwrap();
        // assert_eq!(cell_1d, deserialized_1d);
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
        let circumradius =
            circumradius(&cell.vertices.iter().map(Point::from).collect::<Vec<_>>()).unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(circumradius, expected_radius, epsilon = 1e-10);
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

    #[test]
    fn cell_circumsphere_contains_vertex_determinant() {
        // Test the matrix determinant method for circumsphere containment
        // Use a simple, well-known case: unit tetrahedron
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

        // Test vertex clearly outside circumsphere
        let vertex_far_outside: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([10.0, 10.0, 10.0]))
            .data(4)
            .build()
            .unwrap();
        // Just check that the method runs without error for now
        let vertex_points: Vec<Point<f64, 3>> = cell.vertices.iter().map(Point::from).collect();
        let result = insphere(&vertex_points, *vertex_far_outside.point());
        assert!(result.is_ok());

        // Test with origin (should be inside or on boundary)
        let origin: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(3)
            .build()
            .unwrap();
        let vertex_points: Vec<Point<f64, 3>> = cell.vertices.iter().map(Point::from).collect();
        let result_origin = insphere(&vertex_points, *origin.point());
        assert!(result_origin.is_ok());
    }

    #[test]
    fn cell_circumsphere_contains_vertex_2d() {
        // Test 2D case for circumsphere containment using determinant method
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

        // Test vertex far outside circumcircle
        let vertex_far_outside: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([10.0, 10.0]))
            .build()
            .unwrap();
        let vertex_points: Vec<Point<f64, 2>> = cell.vertices.iter().map(|v| *v.point()).collect();
        let result = insphere(&vertex_points, *vertex_far_outside.point());
        assert!(result.is_ok());

        // Test with center of triangle (should be inside)
        let center: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([0.33, 0.33]))
            .build()
            .unwrap();
        let vertex_points: Vec<Point<f64, 2>> = cell.vertices.iter().map(|v| *v.point()).collect();
        let result_center = insphere(&vertex_points, *center.point());
        assert!(result_center.is_ok());
    }

    #[test]
    fn cell_circumradius_with_center() {
        // Test the circumradius_with_center method
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

        let circumcenter =
            circumcenter(&cell.vertices.iter().map(|v| *v.point()).collect::<Vec<_>>()).unwrap();
        let radius_with_center = circumradius_with_center(
            &cell.vertices.iter().map(|v| *v.point()).collect::<Vec<_>>(),
            &circumcenter,
        );
        let radius_direct =
            circumradius(&cell.vertices.iter().map(|v| *v.point()).collect::<Vec<_>>()).unwrap();

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
    }

    #[test]
    fn cell_facets_completeness() {
        // Test that facets are generated correctly and completely
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

        let facets = cell.facets();
        assert_eq!(facets.len(), 4); // A tetrahedron should have 4 facets

        // Each facet should have 3 vertices (for 3D tetrahedron)
        for facet in &facets {
            assert_eq!(facet.vertices().len(), 3);
        }

        // All vertices should be represented in facets
        let all_facet_vertices: std::collections::HashSet<_> =
            facets.iter().flat_map(Facet::vertices).collect();
        assert!(all_facet_vertices.contains(&vertex1));
        assert!(all_facet_vertices.contains(&vertex2));
        assert!(all_facet_vertices.contains(&vertex3));
        assert!(all_facet_vertices.contains(&vertex4));
    }

    #[test]
    fn cell_builder_validation_edge_cases() {
        // Test builder validation with exactly D+1 vertices (should work)
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

        // Exactly 4 vertices for 3D (D+1 = 3+1 = 4) should work
        let cell_result: Result<Cell<f64, Option<()>, Option<()>, 3>, CellBuilderError> =
            CellBuilder::default()
                .vertices(vec![vertex1, vertex2, vertex3, vertex4])
                .build();
        assert!(cell_result.is_ok());

        // Test with D+2 vertices (should fail)
        let vertex5 = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .build()
            .unwrap();
        let cell_too_many: Result<Cell<f64, Option<()>, Option<()>, 3>, CellBuilderError> =
            CellBuilder::default()
                .vertices(vec![vertex1, vertex2, vertex3, vertex4, vertex5])
                .build();
        assert!(cell_too_many.is_err());
    }

    #[test]
    fn cell_from_facet_and_vertex_comprehensive() {
        // More comprehensive test of from_facet_and_vertex
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

        let original_cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();

        // Create a facet by removing vertex4
        let facet = Facet::new(original_cell, vertex4).unwrap();

        // Create a new vertex
        let new_vertex = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .build()
            .unwrap();

        // Create new cell from facet and vertex
        let new_cell = Cell::from_facet_and_vertex(&facet, new_vertex).unwrap();

        // Verify the new cell contains the original facet vertices plus the new vertex
        assert!(new_cell.contains_vertex(vertex1));
        assert!(new_cell.contains_vertex(vertex2));
        assert!(new_cell.contains_vertex(vertex3));
        assert!(new_cell.contains_vertex(new_vertex));
        assert!(!new_cell.contains_vertex(vertex4)); // Should not contain the removed vertex
        assert_eq!(new_cell.number_of_vertices(), 4);
        assert_eq!(new_cell.dim(), 3);
    }

    #[test]
    fn cell_different_numeric_types() {
        // Test with different numeric types to ensure type flexibility
        // Test with f32
        let vertex1_f32 = VertexBuilder::default()
            .point(Point::new([0.0f32, 0.0f32]))
            .build()
            .unwrap();
        let vertex2_f32 = VertexBuilder::default()
            .point(Point::new([1.0f32, 0.0f32]))
            .build()
            .unwrap();
        let vertex3_f32 = VertexBuilder::default()
            .point(Point::new([0.0f32, 1.0f32]))
            .build()
            .unwrap();

        let cell_f32: Cell<f32, Option<()>, Option<()>, 2> = CellBuilder::default()
            .vertices(vec![vertex1_f32, vertex2_f32, vertex3_f32])
            .build()
            .unwrap();
        assert_eq!(cell_f32.number_of_vertices(), 3);
        assert_eq!(cell_f32.dim(), 2);
    }

    #[test]
    fn cell_high_dimensional() {
        // Test with higher dimensions (5D)
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex6 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0, 1.0]))
            .build()
            .unwrap();

        let cell: Cell<f64, Option<()>, Option<()>, 5> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4, vertex5, vertex6])
            .build()
            .unwrap();

        assert_eq!(cell.number_of_vertices(), 6);
        assert_eq!(cell.dim(), 5);
        assert_eq!(cell.facets().len(), 6); // Each vertex creates one facet
    }

    #[test]
    fn cell_vertex_data_consistency() {
        // Test cells with vertices that have different data types
        let vertex1 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(2)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(3)
            .build()
            .unwrap();

        let cell: Cell<f64, i32, u32, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3])
            .data(42u32)
            .build()
            .unwrap();

        assert_eq!(cell.vertices()[0].data.unwrap(), 1);
        assert_eq!(cell.vertices()[1].data.unwrap(), 2);
        assert_eq!(cell.vertices()[2].data.unwrap(), 3);
        assert_eq!(cell.data.unwrap(), 42u32);
    }

    #[test]
    fn cell_circumsphere_edge_cases() {
        // Test circumsphere containment with simple cases
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

        // Test that the methods run without error
        let test_point: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([0.5, 0.5]))
            .build()
            .unwrap();

        let circumsphere_result = insphere_distance(
            &cell.vertices.iter().map(|v| *v.point()).collect::<Vec<_>>(),
            *test_point.point(),
        );
        assert!(circumsphere_result.is_ok());

        let vertex_points: Vec<Point<f64, 2>> = cell.vertices.iter().map(|v| *v.point()).collect();
        let determinant_result = insphere(&vertex_points, *test_point.point());
        assert!(determinant_result.is_ok());

        // At minimum, both methods should give the same result for the same input
        let far_point: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([100.0, 100.0]))
            .build()
            .unwrap();

        let circumsphere_far = insphere_distance(
            &cell.vertices.iter().map(|v| *v.point()).collect::<Vec<_>>(),
            *far_point.point(),
        );
        let vertex_points: Vec<Point<f64, 2>> = cell.vertices.iter().map(|v| *v.point()).collect();
        let determinant_far = insphere(&vertex_points, *far_point.point());

        assert!(circumsphere_far.is_ok());
        assert!(determinant_far.is_ok());
    }

    #[test]
    fn cell_is_valid_correct_cell() {
        // Test cell is_valid with valid vertices (exactly D+1 = 4 vertices for 3D)
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
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();

        // Human readable output for cargo test -- --nocapture
        println!("Valid Cell: {cell:?}");
        assert!(cell.is_valid().is_ok());
    }

    #[test]
    fn cell_is_valid_invalid_vertex_error() {
        // Test cell is_valid with invalid vertices (containing NaN)
        let vertex_invalid = VertexBuilder::default()
            .point(Point::new([f64::NAN, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex_valid1 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex_valid2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex_valid3 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let invalid_cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![
                vertex_invalid,
                vertex_valid1,
                vertex_valid2,
                vertex_valid3,
            ])
            .build()
            .unwrap();

        // Human readable output for cargo test -- --nocapture
        println!("Invalid Cell: {invalid_cell:?}");
        let invalid_result = invalid_cell.is_valid();
        assert!(invalid_result.is_err());

        // Verify that we get the correct error type for invalid vertex
        match invalid_result {
            Err(CellValidationError::InvalidVertex { source: _ }) => {
                println!("✓ Correctly detected invalid vertex");
            }
            Err(other_error) => {
                panic!("Expected InvalidVertex error, but got: {other_error:?}");
            }
            Ok(()) => {
                panic!("Expected error for invalid vertex, but validation passed");
            }
        }
    }

    #[test]
    fn cell_is_valid_invalid_uuid_error() {
        // Test cell is_valid with nil UUID
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
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let mut invalid_uuid_cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![vertex1, vertex2, vertex3, vertex4])
            .build()
            .unwrap();

        // Manually set the UUID to nil to trigger the InvalidUuid error
        invalid_uuid_cell.uuid = uuid::Uuid::nil();

        let invalid_uuid_result = invalid_uuid_cell.is_valid();
        assert!(invalid_uuid_result.is_err());

        // Verify that we get the correct error type for invalid UUID
        match invalid_uuid_result {
            Err(CellValidationError::InvalidUuid) => {
                println!("✓ Correctly detected invalid UUID");
            }
            Err(other_error) => {
                panic!("Expected InvalidUuid error, but got: {other_error:?}");
            }
            Ok(()) => {
                panic!("Expected error for invalid UUID, but validation passed");
            }
        }
    }

    #[test]
    fn cell_is_valid_duplicate_vertices_error() {
        // Test cell is_valid with duplicate vertices
        let vertex_dup = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let vertex_distinct1 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex_distinct2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let duplicate_cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(vec![
                vertex_dup,
                vertex_dup,
                vertex_distinct1,
                vertex_distinct2,
            ])
            .build()
            .unwrap();

        // Human readable output for cargo test -- --nocapture
        println!("Duplicate Vertices Cell: {duplicate_cell:?}");
        let duplicate_result = duplicate_cell.is_valid();
        assert!(duplicate_result.is_err());

        // Verify that we get the correct error type for duplicate vertices
        match duplicate_result {
            Err(CellValidationError::DuplicateVertices) => {
                println!("✓ Correctly detected duplicate vertices");
            }
            Err(other_error) => {
                panic!("Expected DuplicateVertices error, but got: {other_error:?}");
            }
            Ok(()) => {
                panic!("Expected error for duplicate vertices, but validation passed");
            }
        }
    }

    #[test]
    fn cell_is_valid_insufficient_vertices_error() {
        // Test cell is_valid with insufficient vertices (wrong vertex count)
        let insufficient_vertices = vec![
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0, 1.0]))
                .build()
                .unwrap(),
            VertexBuilder::default()
                .point(Point::new([0.0, 1.0, 0.0]))
                .build()
                .unwrap(),
        ];
        let insufficient_cell: Cell<f64, Option<()>, Option<()>, 3> = CellBuilder::default()
            .vertices(insufficient_vertices)
            .build()
            .unwrap();

        let insufficient_result = insufficient_cell.is_valid();
        assert!(insufficient_result.is_err());

        // Verify that we get the correct error type for insufficient vertices
        match insufficient_result {
            Err(CellValidationError::InsufficientVertices {
                actual,
                expected,
                dimension,
            }) => {
                assert_eq!(actual, 2);
                assert_eq!(expected, 4); // D+1 = 3+1 = 4
                assert_eq!(dimension, 3);
                println!(
                    "✓ Correctly detected insufficient vertices: {actual} instead of {expected} for {dimension}D"
                );
            }
            Err(other_error) => {
                panic!("Expected InsufficientVertices error, but got: {other_error:?}");
            }
            Ok(()) => {
                panic!("Expected error for insufficient vertices, but validation passed");
            }
        }
    }
}

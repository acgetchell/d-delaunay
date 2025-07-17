//! Data and operations on d-dimensional [vertices](https://en.wikipedia.org/wiki/Vertex_(computer_graphics)).

use super::utilities::make_uuid;
use crate::geometry::point::{OrderedEq, Point, PointValidationError};
use num_traits::Float;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fmt::Debug;
use std::{cmp::Ordering, collections::HashMap, hash::Hash, option::Option};
use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur during vertex validation.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum VertexValidationError {
    /// The vertex has an invalid point.
    #[error("Invalid point: {source}")]
    InvalidPoint {
        /// The underlying point validation error.
        #[from]
        source: PointValidationError,
    },
    /// The vertex has an invalid (nil) UUID.
    #[error("Invalid UUID: vertex has nil UUID which is not allowed")]
    InvalidUuid,
}

#[derive(Builder, Clone, Copy, Debug, Default, Deserialize, Serialize)]
/// The [Vertex] struct represents a vertex in a triangulation with a [Point],
/// a unique identifier, an optional incident cell identifier, and optional
/// data.
///
/// # Properties
///
/// - `point`: A generic [Point] representing the coordinates of
///   the vertex in a D-dimensional space.
/// - `uuid`: A [Uuid] representing a universally unique identifier for the
///   for the [Vertex]. This can be used to uniquely
///   identify the vertex in a graph or any other data structure.
/// - `incident_cell`: The `incident_cell` property is an optional [Uuid] that
///   represents a `Cell` containing the [Vertex]. This is
///   calculated by the `delaunay_core::triangulation_data_structure::Tds`.
/// - `data`: The `data` property is an optional field that can hold any
///   type `U`. It is used to store additional data associated with the vertex.
///
/// The Point<T, D> encapsulates the coordinate type T and provides necessary
/// trait bounds, so the Vertex struct is generic over the coordinate type T,
/// data type U, and dimension D.
///
/// T is intended to be a Float type (f32, f64, etc.) for coordinates.
/// U is intended to be data associated with the vertex, e.g. a string, which
/// implements Eq, Hash, Ord, `PartialEq`, and `PartialOrd`.
pub struct Vertex<T, U, const D: usize>
where
    T: Clone + Copy + Default + Float + PartialEq + PartialOrd + OrderedEq,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The coordinates of the vertex as a D-dimensional Point.
    point: Point<T, D>,
    /// A universally unique identifier for the vertex.
    #[builder(setter(skip), default = "make_uuid()")]
    uuid: Uuid,
    /// The [Uuid] of the `Cell` that the vertex is incident to.
    #[builder(setter(skip), default = "None")]
    pub incident_cell: Option<Uuid>,
    /// Optional data associated with the vertex.
    #[builder(setter(into, strip_option), default)]
    pub data: Option<U>,
}

impl<T, U, const D: usize> Vertex<T, U, D>
where
    T: Clone + Copy + Default + Float + PartialEq + PartialOrd + OrderedEq,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The function `from_points` takes a vector of points and returns a
    /// vector of vertices, using the `new` method.
    ///
    /// # Arguments
    ///
    /// * `points`: `points` is a vector of [Point] objects.
    ///
    /// # Returns
    ///
    /// The function `from_points` returns a `Vec<Vertex<T, U, D>>`, where `T`
    /// is the type of the coordinates of the [Vertex], `U` is the type of the
    /// optional data associated with the [Vertex], and `D` is the
    /// dimensionality of the [Vertex].
    ///
    /// # Panics
    ///
    /// Panics if the `VertexBuilder` fails to build a vertex from any point.
    /// This should not happen under normal circumstances with valid point data.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::geometry::point::Point;
    /// let points = vec![Point::new([1.0, 2.0, 3.0])];
    /// let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points.clone());
    /// assert_eq!(vertices.len(), 1);
    /// assert_eq!(vertices[0].point().coordinates(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_points(points: Vec<Point<T, D>>) -> Vec<Self> {
        points
            .into_iter()
            .map(|p| VertexBuilder::default().point(p).build().unwrap())
            .collect()
    }

    /// The function `into_hashmap` converts a vector of vertices into a
    /// [`HashMap`], using the vertices [Uuid] as the key.
    ///
    /// # Arguments
    ///
    /// * `vertices`: `vertices` is a vector of `Vertex<T, U, D>`.
    ///
    /// # Returns
    ///
    /// The function `into_hashmap` returns a [`HashMap`] with the key type
    /// [Uuid] and the value type [Vertex], i.e. `std::collections::HashMap<Uuid, Vertex<T, U, D>>`.
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::geometry::point::Point;
    /// let points = vec![Point::new([1.0, 2.0]), Point::new([3.0, 4.0])];
    /// let vertices = Vertex::<f64, Option<()>, 2>::from_points(points.clone());
    /// let map: HashMap<_, _> = Vertex::into_hashmap(vertices);
    /// assert_eq!(map.len(), 2);
    /// assert!(map.values().all(|v| v.dim() == 2));
    /// ```
    #[inline]
    #[must_use]
    pub fn into_hashmap(vertices: Vec<Self>) -> HashMap<Uuid, Self> {
        vertices.into_iter().map(|v| (v.uuid(), v)).collect()
    }

    /// Returns the point coordinates of the vertex.
    ///
    /// # Returns
    ///
    /// A reference to the Point representing the vertex's coordinates.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0]);
    /// let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default().point(point).build().unwrap();
    /// let retrieved_point = vertex.point();
    /// assert_eq!(retrieved_point.coordinates(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn point(&self) -> &Point<T, D> {
        &self.point
    }

    /// Returns the UUID of the vertex.
    ///
    /// # Returns
    ///
    /// The Uuid uniquely identifying this vertex.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// use uuid::Uuid;
    /// let point = Point::new([1.0, 2.0, 3.0]);
    /// let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default().point(point).build().unwrap();
    /// let vertex_uuid = vertex.uuid();
    /// // UUID should be valid and unique
    /// assert_ne!(vertex_uuid, Uuid::nil());
    ///
    /// // Creating another vertex should have a different UUID
    /// let another_vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default().point(point).build().unwrap();
    /// assert_ne!(vertex.uuid(), another_vertex.uuid());
    /// ```
    #[inline]
    pub fn uuid(&self) -> Uuid {
        self.uuid
    }

    /// The `dim` function returns the dimensionality of the [Vertex].
    ///
    /// # Returns
    ///
    /// The `dim` function is returning the value of `D`, which the number of
    /// coordinates.
    ///
    /// # Example
    /// ```
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::geometry::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// let vertex: Vertex<f64, Option<()>, 4> = VertexBuilder::default().point(point).build().unwrap();
    /// assert_eq!(vertex.dim(), 4);
    /// ```
    #[inline]
    pub fn dim(&self) -> usize {
        D
    }

    /// The function `is_valid` checks if a [Vertex] is valid.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the [Vertex] is valid, otherwise returns a
    /// `VertexValidationError` indicating the specific validation failure.
    /// A valid vertex has:
    /// - A valid [Point] with finite coordinates (no NaN or infinite values)
    /// - A valid [Uuid] that is not nil
    ///
    /// # Errors
    ///
    /// Returns `VertexValidationError::InvalidPoint` if the point has invalid coordinates,
    /// or `VertexValidationError::InvalidUuid` if the UUID is nil.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder, VertexValidationError};
    /// use d_delaunay::geometry::point::Point;
    /// let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
    ///     .point(Point::new([1.0, 2.0, 3.0]))
    ///     .build()
    ///     .unwrap();
    /// assert!(vertex.is_valid().is_ok());
    ///
    /// let invalid_vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
    ///     .point(Point::new([1.0, f64::NAN, 3.0]))
    ///     .build()
    ///     .unwrap();
    /// match invalid_vertex.is_valid() {
    ///     Err(VertexValidationError::InvalidPoint { .. }) => (), // Expected
    ///     _ => panic!("Expected point validation error"),
    /// }
    /// ```
    pub fn is_valid(self) -> Result<(), VertexValidationError>
    where
        T: crate::geometry::point::FiniteCheck + Copy + Debug,
    {
        // Check if the point is valid (all coordinates are finite)
        self.point.is_valid()?;

        // Check if UUID is not nil
        if self.uuid.is_nil() {
            return Err(VertexValidationError::InvalidUuid);
        }

        Ok(())
        // TODO: Additional validation can be added here:
        // - Validate incident_cell reference if present
        // - Validate data if needed
    }
}

// Implementations with identical trait bounds grouped together
// Group 1: PartialEq, PartialOrd, and From trait implementations
impl<T, U, const D: usize> PartialEq for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Equality of vertices is based on equality of elements in vector of coords.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
        // && self.uuid == other.uuid
        // && self.incident_cell == other.incident_cell
        // && self.data == other.data
    }
}

impl<T, U, const D: usize> PartialOrd for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Order of vertices is based on lexicographic order of elements in vector of coords.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.point.partial_cmp(&other.point)
    }
}

/// Enable implicit conversion from Vertex to coordinate array
/// This allows `vertex.point.coordinates()` to be implicitly converted to `[T; D]`
impl<T, U, const D: usize> From<Vertex<T, U, D>> for [T; D]
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from(vertex: Vertex<T, U, D>) -> [T; D] {
        vertex.point().coordinates()
    }
}

/// Enable implicit conversion from Vertex reference to coordinate array
/// This allows `&vertex` to be implicitly converted to `[T; D]` for coordinate access
impl<T, U, const D: usize> From<&Vertex<T, U, D>> for [T; D]
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from(vertex: &Vertex<T, U, D>) -> [T; D] {
        vertex.point().coordinates()
    }
}

// Group 2: Eq implementation with additional Hash requirement
impl<T, U, const D: usize> Eq for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Vertex<T, U, D>: Hash,
{
    // Generic Eq implementation for Vertex based on point equality
}

impl<T, U, const D: usize> Hash for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Point<T, D>: Hash,
{
    /// Generic Hash implementation for Vertex with any type T where Point<T, D> implements Hash
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.point.hash(state);
        self.uuid.hash(state);
        self.incident_cell.hash(state);
        self.data.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    // Helper function to create a basic vertex with given coordinates
    fn create_vertex<T, U, const D: usize>(coords: [T; D]) -> Vertex<T, U, D>
    where
        T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + Float,
        U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
        [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    {
        VertexBuilder::default()
            .point(Point::new(coords))
            .build()
            .unwrap()
    }

    // Helper function to create a vertex with data
    fn create_vertex_with_data<T, U, const D: usize>(coords: [T; D], data: U) -> Vertex<T, U, D>
    where
        T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + Float,
        U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
        [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    {
        VertexBuilder::default()
            .point(Point::new(coords))
            .data(data)
            .build()
            .unwrap()
    }

    // Helper function to test basic vertex properties
    fn test_basic_vertex_properties<T, U, const D: usize>(
        vertex: &Vertex<T, U, D>,
        expected_coords: [T; D],
        expected_dim: usize,
    ) where
        T: Clone + Copy + Debug + Default + PartialEq + PartialOrd + OrderedEq + Float,
        U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
        [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    {
        assert_eq!(vertex.point().coordinates(), expected_coords);
        assert_eq!(vertex.dim(), expected_dim);
        assert!(!vertex.uuid().is_nil());
        assert!(vertex.incident_cell.is_none());
    }

    #[test]
    fn vertex_default() {
        let vertex: Vertex<f64, Option<()>, 3> = Vertex::default();

        assert_relative_eq!(
            vertex.point().coordinates().as_slice(),
            [0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
        assert!(vertex.uuid().is_nil());
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_none());

        // Human readable output for cargo test -- --nocapture
        println!("{vertex:?}");
    }

    #[test]
    fn vertex_builder() {
        let mut vertex: Vertex<f64, &str, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        assert_relative_eq!(
            vertex.point().coordinates().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
        assert!(!vertex.uuid().is_nil());
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_none());

        // Can mutate later
        vertex.data = Some("3D");
        assert_eq!(vertex.data.unwrap(), "3D");

        // Human readable output for cargo test -- --nocapture
        println!("{vertex:?}");
    }

    #[test]
    fn vertex_builder_with_data() {
        let vertex: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .data(1)
            .build()
            .unwrap();

        assert_relative_eq!(
            vertex.point().coordinates().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
        assert!(!vertex.uuid().is_nil());
        assert!(vertex.incident_cell.is_none());
        assert_eq!(vertex.data.unwrap(), 1);

        // Human readable output for cargo test -- --nocapture
        println!("{vertex:?}");
    }

    #[test]
    fn vertex_copy() {
        let vertex: Vertex<f64, &str, 4> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0, 4.0]))
            .data("4D")
            .build()
            .unwrap();
        let vertex_copy = vertex;

        assert_eq!(vertex, vertex_copy);
        assert_relative_eq!(
            vertex_copy.point().coordinates().as_slice(),
            [1.0, 2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn vertex_from_points() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);

        assert_relative_eq!(
            vertices[0].point().coordinates().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[0].dim(), 3);
        assert_relative_eq!(
            vertices[1].point().coordinates().as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[1].dim(), 3);
        assert_relative_eq!(
            vertices[2].point().coordinates().as_slice(),
            [7.0, 8.0, 9.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[2].dim(), 3);

        // Human readable output for cargo test -- --nocapture
        println!("{vertices:?}");
    }

    #[test]
    fn vertex_into_hashmap() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];
        let mut vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices.clone());
        let mut values: Vec<Vertex<f64, Option<()>, 3>> = hashmap.into_values().collect();

        assert_eq!(values.len(), 3);

        values.sort_by_key(super::Vertex::uuid);
        vertices.sort_by_key(super::Vertex::uuid);

        assert_eq!(values, vertices);

        // Human readable output for cargo test -- --nocapture
        println!("values = {values:?}");
        println!("vertices = {vertices:?}");
    }

    #[test]
    fn vertex_dim() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        assert_eq!(vertex.dim(), 3);
    }

    #[test]
    fn vertex_to_and_from_json() {
        // let vertex: Vertex<f64, Option<()>, 3> = Vertex::new(Point::new([1.0, 2.0, 3.0]));
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let serialized = serde_json::to_string(&vertex).unwrap();

        assert!(serialized.contains("point"));
        assert!(serialized.contains("[1.0,2.0,3.0]"));

        let deserialized: Vertex<f64, Option<()>, 3> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, vertex);

        // Human readable output for cargo test -- --nocapture
        println!("Serialized: {serialized:?}");
    }

    #[test]
    fn vertex_partial_eq() {
        let vertex1: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex3: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 4.0]))
            .build()
            .unwrap();

        assert_eq!(vertex1, vertex2);
        assert_ne!(vertex2, vertex3);
    }

    #[test]
    fn vertex_partial_ord() {
        let vertex1: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 4.0]))
            .build()
            .unwrap();
        let vertex3: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([10.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 10.0]))
            .build()
            .unwrap();

        assert!(vertex1 < vertex2);
        assert!(vertex3 > vertex2);
        assert!(vertex1 < vertex3);
        assert!(vertex1 > vertex4);
    }

    #[test]
    fn vertex_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let vertex1: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex3: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 4.0]))
            .build()
            .unwrap();

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        let mut hasher3 = DefaultHasher::new();

        vertex1.hash(&mut hasher1);
        vertex2.hash(&mut hasher2);
        vertex3.hash(&mut hasher3);

        // Different UUIDs mean different hashes even with same points
        assert_ne!(hasher1.finish(), hasher2.finish());
        assert_ne!(hasher1.finish(), hasher3.finish());
    }

    #[test]
    fn vertex_hash_in_hashmap() {
        use std::collections::HashMap;

        let mut map: HashMap<Vertex<f64, Option<()>, 2>, i32> = HashMap::new();

        let vertex1: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([3.0, 4.0]))
            .build()
            .unwrap();

        map.insert(vertex1, 10);
        map.insert(vertex2, 20);

        assert_eq!(map.get(&vertex1), Some(&10));
        assert_eq!(map.get(&vertex2), Some(&20));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn vertex_clone() {
        let vertex: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .data(42)
            .build()
            .unwrap();
        let cloned_vertex = vertex;

        // Points should be equal but UUIDs should be the same (since we cloned)
        assert_eq!(vertex.point(), cloned_vertex.point());
        assert_eq!(vertex.uuid(), cloned_vertex.uuid());
        assert_eq!(vertex.incident_cell, cloned_vertex.incident_cell);
        assert_eq!(vertex.data, cloned_vertex.data);
        assert_eq!(vertex.dim(), cloned_vertex.dim());
    }

    #[test]
    fn vertex_1d() {
        let vertex: Vertex<f64, Option<()>, 1> = create_vertex([42.0]);
        test_basic_vertex_properties(&vertex, [42.0], 1);
        assert!(vertex.data.is_none());
    }

    #[test]
    fn vertex_2d() {
        let vertex: Vertex<f64, Option<()>, 2> = create_vertex([1.0, 2.0]);
        test_basic_vertex_properties(&vertex, [1.0, 2.0], 2);
        assert!(vertex.data.is_none());
    }

    #[test]
    fn vertex_4d() {
        let vertex: Vertex<f64, Option<()>, 4> = create_vertex([1.0, 2.0, 3.0, 4.0]);
        test_basic_vertex_properties(&vertex, [1.0, 2.0, 3.0, 4.0], 4);
        assert!(vertex.data.is_none());
    }

    #[test]
    fn vertex_5d() {
        let vertex: Vertex<f64, Option<()>, 5> = create_vertex([1.0, 2.0, 3.0, 4.0, 5.0]);
        test_basic_vertex_properties(&vertex, [1.0, 2.0, 3.0, 4.0, 5.0], 5);
        assert!(vertex.data.is_none());
    }

    #[test]
    fn vertex_with_f32() {
        let vertex: Vertex<f32, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([1.5, 2.5]))
            .build()
            .unwrap();

        assert_relative_eq!(
            vertex.point().coordinates().as_slice(),
            [1.5, 2.5].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 2);
        assert!(!vertex.uuid().is_nil());
    }

    #[test]
    fn vertex_with_string_data() {
        let vertex: Vertex<f64, &str, 3> = create_vertex_with_data([1.0, 2.0, 3.0], "test_vertex");
        test_basic_vertex_properties(&vertex, [1.0, 2.0, 3.0], 3);
        assert_eq!(vertex.data.unwrap(), "test_vertex");
    }

    #[test]
    fn vertex_with_numeric_data() {
        let vertex: Vertex<f64, u32, 2> = create_vertex_with_data([5.0, 10.0], 123u32);
        test_basic_vertex_properties(&vertex, [5.0, 10.0], 2);
        assert_eq!(vertex.data.unwrap(), 123u32);
    }

    #[test]
    fn vertex_with_tuple_data() {
        let vertex: Vertex<f64, (i32, i32), 2> = create_vertex_with_data([1.0, 2.0], (42, 84));
        test_basic_vertex_properties(&vertex, [1.0, 2.0], 2);
        assert_eq!(vertex.data.unwrap(), (42, 84));
    }

    #[test]
    fn vertex_debug_format() {
        let vertex: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .data(42)
            .build()
            .unwrap();
        let debug_str = format!("{vertex:?}");

        assert!(debug_str.contains("Vertex"));
        assert!(debug_str.contains("point"));
        assert!(debug_str.contains("uuid"));
        assert!(debug_str.contains("1.0"));
        assert!(debug_str.contains("2.0"));
        assert!(debug_str.contains("3.0"));
    }

    #[test]
    fn vertex_eq_trait() {
        let vertex1: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex3: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 4.0]))
            .build()
            .unwrap();

        // Test Eq trait (reflexivity, symmetry) - equality is based on point only
        assert_eq!(vertex1, vertex1); // reflexive
        assert_eq!(vertex1, vertex2); // same points
        assert_eq!(vertex2, vertex1); // symmetric
        assert_ne!(vertex1, vertex3); // different points
        assert_ne!(vertex3, vertex1); // symmetric
    }

    #[test]
    fn vertex_ordering_edge_cases() {
        let vertex1: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0]))
            .build()
            .unwrap();

        // Test that equal points result in equal ordering
        assert!(vertex1.partial_cmp(&vertex2) != Some(Ordering::Less));
        assert!(vertex2.partial_cmp(&vertex1) != Some(Ordering::Less));
        assert!(matches!(
            vertex1.partial_cmp(&vertex2),
            Some(Ordering::Less | Ordering::Equal)
        ));
        assert!(matches!(
            vertex2.partial_cmp(&vertex1),
            Some(Ordering::Less | Ordering::Equal)
        ));
        assert!(matches!(
            vertex1.partial_cmp(&vertex2),
            Some(Ordering::Greater | Ordering::Equal)
        ));
        assert!(matches!(
            vertex2.partial_cmp(&vertex1),
            Some(Ordering::Greater | Ordering::Equal)
        ));
    }

    #[test]
    fn vertex_comprehensive_serialization() {
        // Test with different data types and dimensions
        let vertex_no_data: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let serialized = serde_json::to_string(&vertex_no_data).unwrap();
        let deserialized: Vertex<f64, Option<()>, 3> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(vertex_no_data, deserialized);

        let vertex_with_data: Vertex<f64, i32, 2> = VertexBuilder::default()
            .point(Point::new([10.5, -5.3]))
            .data(42)
            .build()
            .unwrap();
        let serialized_data = serde_json::to_string(&vertex_with_data).unwrap();
        let deserialized_data: Vertex<f64, i32, 2> =
            serde_json::from_str(&serialized_data).unwrap();
        assert_eq!(vertex_with_data, deserialized_data);

        let vertex_1d: Vertex<f64, Option<()>, 1> = VertexBuilder::default()
            .point(Point::new([42.0]))
            .build()
            .unwrap();
        let serialized_1d = serde_json::to_string(&vertex_1d).unwrap();
        let deserialized_1d: Vertex<f64, Option<()>, 1> =
            serde_json::from_str(&serialized_1d).unwrap();
        assert_eq!(vertex_1d, deserialized_1d);
    }

    #[test]
    fn vertex_negative_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([-1.0, -2.0, -3.0]))
            .build()
            .unwrap();

        assert_relative_eq!(
            vertex.point().coordinates().as_slice(),
            [-1.0, -2.0, -3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
    }

    #[test]
    fn vertex_zero_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();

        let origin_vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::origin())
            .build()
            .unwrap();

        assert_eq!(vertex.point(), origin_vertex.point());
    }

    #[test]
    fn vertex_large_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1e6, 2e6, 3e6]))
            .build()
            .unwrap();

        assert_relative_eq!(
            vertex.point().coordinates().as_slice(),
            [1_000_000.0, 2_000_000.0, 3_000_000.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
    }

    #[test]
    fn vertex_small_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1e-6, 2e-6, 3e-6]))
            .build()
            .unwrap();

        assert_relative_eq!(
            vertex.point().coordinates().as_slice(),
            [0.000_001, 0.000_002, 0.000_003].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
    }

    #[test]
    fn vertex_from_points_empty() {
        let points: Vec<Point<f64, 3>> = Vec::new();
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);

        assert!(vertices.is_empty());
    }

    #[test]
    fn vertex_from_points_single() {
        let points = vec![Point::new([1.0, 2.0, 3.0])];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);

        assert_eq!(vertices.len(), 1);
        assert_relative_eq!(
            vertices[0].point().coordinates().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[0].dim(), 3);
        assert!(!vertices[0].uuid().is_nil());
    }

    #[test]
    fn vertex_into_hashmap_empty() {
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vec::new();
        let hashmap = Vertex::into_hashmap(vertices);

        assert!(hashmap.is_empty());
    }

    #[test]
    fn vertex_into_hashmap_single() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let uuid = vertex.uuid();
        let vertices = vec![vertex];
        let hashmap = Vertex::into_hashmap(vertices);

        assert_eq!(hashmap.len(), 1);
        assert!(hashmap.contains_key(&uuid));
        assert_relative_eq!(
            hashmap.get(&uuid).unwrap().point().coordinates().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn vertex_uuid_uniqueness() {
        let vertex1: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        // Same points but different UUIDs
        assert_ne!(vertex1.uuid(), vertex2.uuid());
        assert!(!vertex1.uuid().is_nil());
        assert!(!vertex2.uuid().is_nil());
    }

    #[test]
    fn vertex_incident_cell_none_by_default() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        assert!(vertex.incident_cell.is_none());
    }

    #[test]
    fn vertex_data_none_by_default() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        assert!(vertex.data.is_none());
    }

    #[test]
    fn vertex_data_can_be_set() {
        let vertex: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .data(42)
            .build()
            .unwrap();

        assert_eq!(vertex.data.unwrap(), 42);
    }

    #[test]
    fn vertex_mixed_positive_negative_coordinates() {
        let vertex: Vertex<f64, Option<()>, 4> = VertexBuilder::default()
            .point(Point::new([1.0, -2.0, 3.0, -4.0]))
            .build()
            .unwrap();

        assert_relative_eq!(
            vertex.point().coordinates().as_slice(),
            [1.0, -2.0, 3.0, -4.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 4);
    }

    #[test]
    fn vertex_implicit_conversion_to_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        // Test implicit conversion from owned vertex
        let coords_owned: [f64; 3] = vertex.into();
        assert_relative_eq!(
            coords_owned.as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Create a new vertex for reference test
        let vertex_ref: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([4.0, 5.0, 6.0]))
            .build()
            .unwrap();

        // Test implicit conversion from vertex reference
        let coords_ref: [f64; 3] = (&vertex_ref).into();
        assert_relative_eq!(
            coords_ref.as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );

        // Verify the original vertex is still available after reference conversion
        assert_relative_eq!(
            vertex_ref.point().coordinates().as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn vertex_is_valid_f64() {
        // Test valid vertex with finite coordinates
        let valid_vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        assert!(valid_vertex.is_valid().is_ok());

        // Test valid vertex with negative coordinates
        let valid_negative: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([-1.0, -2.0, -3.0]))
            .build()
            .unwrap();
        assert!(valid_negative.is_valid().is_ok());

        // Test valid vertex with zero coordinates
        let valid_zero: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        assert!(valid_zero.is_valid().is_ok());

        // Test invalid vertex with NaN coordinate
        let invalid_nan: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, f64::NAN, 3.0]))
            .build()
            .unwrap();
        assert!(invalid_nan.is_valid().is_err());

        // Test invalid vertex with all NaN coordinates
        let invalid_all_nan: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([f64::NAN, f64::NAN, f64::NAN]))
            .build()
            .unwrap();
        assert!(invalid_all_nan.is_valid().is_err());

        // Test invalid vertex with positive infinity
        let invalid_pos_inf: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, f64::INFINITY, 3.0]))
            .build()
            .unwrap();
        assert!(invalid_pos_inf.is_valid().is_err());

        // Test invalid vertex with negative infinity
        let invalid_neg_inf: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, f64::NEG_INFINITY, 3.0]))
            .build()
            .unwrap();
        assert!(invalid_neg_inf.is_valid().is_err());

        // Test invalid vertex with mixed NaN and infinity
        let invalid_mixed: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([f64::NAN, f64::INFINITY, 1.0]))
            .build()
            .unwrap();
        assert!(invalid_mixed.is_valid().is_err());
    }

    #[test]
    fn vertex_is_valid_f32() {
        // Test valid f32 vertex
        let valid_vertex: Vertex<f32, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([1.5f32, 2.5f32]))
            .build()
            .unwrap();
        assert!(valid_vertex.is_valid().is_ok());

        // Test invalid f32 vertex with NaN
        let invalid_nan: Vertex<f32, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([1.0f32, f32::NAN]))
            .build()
            .unwrap();
        assert!(invalid_nan.is_valid().is_err());

        // Test invalid f32 vertex with infinity
        let invalid_inf: Vertex<f32, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([f32::INFINITY, 2.0f32]))
            .build()
            .unwrap();
        assert!(invalid_inf.is_valid().is_err());
    }

    #[test]
    fn vertex_is_valid_different_dimensions() {
        // Test 1D vertex
        let valid_1d: Vertex<f64, Option<()>, 1> = VertexBuilder::default()
            .point(Point::new([42.0]))
            .build()
            .unwrap();
        assert!(valid_1d.is_valid().is_ok());

        let invalid_1d: Vertex<f64, Option<()>, 1> = VertexBuilder::default()
            .point(Point::new([f64::NAN]))
            .build()
            .unwrap();
        assert!(invalid_1d.is_valid().is_err());

        // Test 5D vertex
        let valid_5d: Vertex<f64, Option<()>, 5> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0, 4.0, 5.0]))
            .build()
            .unwrap();
        assert!(valid_5d.is_valid().is_ok());

        let invalid_5d: Vertex<f64, Option<()>, 5> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, f64::NAN, 4.0, 5.0]))
            .build()
            .unwrap();
        assert!(invalid_5d.is_valid().is_err());
    }

    #[test]
    fn vertex_is_valid_uuid_check() {
        // Test that vertex with valid point and UUID is valid
        let valid_vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        assert!(valid_vertex.is_valid().is_ok());
        assert!(!valid_vertex.uuid().is_nil());

        // Test that default vertex (which has nil UUID) is invalid
        let default_vertex: Vertex<f64, Option<()>, 3> = Vertex::default();
        match default_vertex.is_valid() {
            Err(VertexValidationError::InvalidUuid) => (), // Expected
            other => panic!("Expected InvalidUuid error, got: {other:?}"),
        }
        assert!(default_vertex.uuid().is_nil());
        assert!(default_vertex.point().is_valid().is_ok()); // Point itself is valid (zeros)

        // Create a vertex with valid point but manually set nil UUID to test UUID validation
        let invalid_uuid_vertex: Vertex<f64, Option<()>, 3> = Vertex {
            point: Point::new([1.0, 2.0, 3.0]),
            uuid: uuid::Uuid::nil(),
            incident_cell: None,
            data: None,
        };
        match invalid_uuid_vertex.is_valid() {
            Err(VertexValidationError::InvalidUuid) => (), // Expected
            other => panic!("Expected InvalidUuid error, got: {other:?}"),
        }
        assert!(invalid_uuid_vertex.point().is_valid().is_ok()); // Point is valid
        assert!(invalid_uuid_vertex.uuid().is_nil()); // UUID is nil

        // Test vertex with both invalid point and nil UUID (should return point error first)
        let invalid_both: Vertex<f64, Option<()>, 3> = Vertex {
            point: Point::new([f64::NAN, 2.0, 3.0]),
            uuid: uuid::Uuid::nil(),
            incident_cell: None,
            data: None,
        };
        match invalid_both.is_valid() {
            Err(VertexValidationError::InvalidPoint { .. }) => (), // Expected - point checked first
            other => panic!("Expected InvalidPoint error, got: {other:?}"),
        }
        assert!(invalid_both.point().is_valid().is_err()); // Point is invalid
        assert!(invalid_both.uuid().is_nil()); // UUID is nil
    }
}

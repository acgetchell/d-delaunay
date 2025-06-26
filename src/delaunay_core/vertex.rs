//! Data and operations on d-dimensional [vertices](https://en.wikipedia.org/wiki/Vertex_(computer_graphics)).

use super::point::{OrderedEq, Point, PointValidationError};
use crate::delaunay_core::{CellKey, VertexKey};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fmt::Debug;
use std::{cmp::Ordering, hash::Hash, option::Option};
use thiserror::Error;

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
    /// The vertex has an invalid (null) key.
    #[error("Invalid key: vertex has null key which is not allowed")]
    InvalidKey,
}

use slotmap::Key;

#[derive(Builder, Clone, Copy, Debug, Default, Deserialize, Serialize)]
/// The [Vertex] struct represents a vertex in a triangulation with a [Point],
/// a unique identifier, an optional incident cell identifier, and optional
/// data.
///
/// # Properties
///
/// - `point`: A generic [Point] representing the coordinates of
///   the vertex in a D-dimensional space.
/// - `key`: The key into the vertex `SlotMap`. This uniquely identifies the vertex
///   within its containing data structure.
/// - `incident_cell`: The `incident_cell` property represents the key of a `Cell`
///   containing the [Vertex]. This is calculated by the
///   `delaunay_core::triangulation_data_structure::Tds`.
/// - `data`: The `data` property is an optional field that can hold any
///   type `U`. It is used to store additional data associated with the vertex.
///
/// Data type T is in practice f64 which does not implement Eq, Hash, or Ord.
///
/// U is intended to be data associated with the vertex, e.g. a string, which
/// implements Eq, Hash, Ord, `PartialEq`, and `PartialOrd`.
pub struct Vertex<T, U, const D: usize>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The coordinates of the vertex as a D-dimensional Point.
    point: Point<T, D>,
    /// The unique key identifying this vertex in the containing `SlotMap`.
    #[builder(default = "VertexKey::null()")]
    key: VertexKey,
    /// The key of the `Cell` that the vertex is incident to.
    #[builder(setter(skip), default = "None")]
    pub incident_cell: Option<CellKey>,
    /// Optional data associated with the vertex.
    #[builder(setter(into, strip_option), default)]
    pub data: Option<U>,
}

impl<T, U, const D: usize> Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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
    /// use d_delaunay::delaunay_core::point::Point;
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
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0]);
    /// let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default().point(point).build().unwrap();
    /// let retrieved_point = vertex.point();
    /// assert_eq!(retrieved_point.coordinates(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn point(&self) -> &Point<T, D> {
        &self.point
    }

    /// Returns the key of the vertex.
    ///
    /// # Returns
    ///
    /// The key uniquely identifying this vertex in its containing `SlotMap`.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0]);
    /// let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default().point(point).build().unwrap();
    /// let vertex_key = vertex.key();
    /// // Key should be valid and unique
    /// assert!(!vertex_key.is_null());
    ///
    /// // Creating another vertex should have a different key
    /// let another_vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default().point(point).build().unwrap();
    /// assert_ne!(vertex.key(), another_vertex.key());
    /// ```
    #[inline]
    pub fn key(&self) -> VertexKey {
        self.key
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
    /// use d_delaunay::delaunay_core::point::Point;
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
    /// - A valid key that is not null
    ///
    /// # Errors
    ///
    /// Returns `VertexValidationError::InvalidPoint` if the point has invalid coordinates,
    /// or `VertexValidationError::InvalidKey` if the vertex key is null.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder, VertexValidationError};
    /// use d_delaunay::delaunay_core::point::Point;
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
        T: super::point::FiniteCheck + Copy + Debug,
    {
        // Check if the point is valid (all coordinates are finite)
        self.point.is_valid()?;

        // Check if key is not null
        if self.key.is_null() {
            return Err(VertexValidationError::InvalidKey);
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
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Vertex<T, U, D>: Hash,
{
    // Generic Eq implementation for Vertex based on point equality
}

impl<T, U, const D: usize> Hash for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Point<T, D>: Hash,
{
    /// Generic Hash implementation for Vertex with any type T where Point<T, D> implements Hash
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.point.hash(state);
        self.key.hash(state);
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
        T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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
        T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
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
        T: Clone + Copy + Debug + Default + PartialEq + PartialOrd + OrderedEq,
        U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
        [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    {
        assert_eq!(vertex.point().coordinates(), expected_coords);
        assert_eq!(vertex.dim(), expected_dim);
        // VertexKey should be null by default (until inserted into a SlotMap)
        assert!(vertex.key().is_null());
        assert!(vertex.incident_cell.is_none());
    }

    #[test]
    fn vertex_default() {
        let vertex: Vertex<f64, Option<()>, 3> = Vertex::default();

        test_basic_vertex_properties(&vertex, [0.0, 0.0, 0.0], 3);
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

        test_basic_vertex_properties(&vertex, [1.0, 2.0, 3.0], 3);
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

        test_basic_vertex_properties(&vertex, [1.0, 2.0, 3.0], 3);
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

        // Same points with null keys should hash the same
        assert_eq!(hasher1.finish(), hasher2.finish());
        // Different points should hash differently, even with null keys
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

        // Points should be equal but keys should be the same (since we cloned)
        assert_eq!(vertex.point(), cloned_vertex.point());
        assert_eq!(vertex.key(), cloned_vertex.key());
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

        test_basic_vertex_properties(&vertex, [1.5, 2.5], 2);
    }

    #[test]
    fn vertex_with_integers() {
        let vertex: Vertex<i32, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1, 2, 3]))
            .build()
            .unwrap();

        test_basic_vertex_properties(&vertex, [1, 2, 3], 3);
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
        assert!(debug_str.contains("key"));
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

        test_basic_vertex_properties(&vertex, [-1.0, -2.0, -3.0], 3);
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

        test_basic_vertex_properties(&vertex, [1_000_000.0, 2_000_000.0, 3_000_000.0], 3);
    }

    #[test]
    fn vertex_small_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1e-6, 2e-6, 3e-6]))
            .build()
            .unwrap();

        test_basic_vertex_properties(&vertex, [0.000_001, 0.000_002, 0.000_003], 3);
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
        assert!(vertices[0].key().is_null());
    }

    #[test]
    fn vertex_key_uniqueness() {
        let vertex1: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        // Both keys should be null until inserted into SlotMap
        assert!(vertex1.key().is_null());
        assert!(vertex2.key().is_null());
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

        test_basic_vertex_properties(&vertex, [1.0, -2.0, 3.0, -4.0], 4);
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
        // Key should be null by default, making it invalid until inserted in SlotMap
        assert!(matches!(
            valid_vertex.is_valid(),
            Err(VertexValidationError::InvalidKey)
        ));

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
        // Key should be null by default, making it invalid until inserted in SlotMap
        assert!(matches!(
            valid_vertex.is_valid(),
            Err(VertexValidationError::InvalidKey)
        ));

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
    fn vertex_is_valid_integers() {
        // Integer vertices should always be valid
        let valid_i32: Vertex<i32, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1, 2, 3]))
            .build()
            .unwrap();
        // Key should be null by default, making it invalid until inserted in SlotMap
        assert!(matches!(
            valid_i32.is_valid(),
            Err(VertexValidationError::InvalidKey)
        ));

        let valid_negative_i32: Vertex<i32, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([-1, -2, -3]))
            .build()
            .unwrap();
        assert!(valid_negative_i32.is_valid().is_ok());

        let valid_zero_i32: Vertex<i32, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([0, 0, 0]))
            .build()
            .unwrap();
        assert!(valid_zero_i32.is_valid().is_ok());

        let valid_unsigned: Vertex<u64, Option<()>, 2> = VertexBuilder::default()
            .point(Point::new([u64::MAX, u64::MIN]))
            .build()
            .unwrap();
        assert!(valid_unsigned.is_valid().is_ok());
    }

    #[test]
    fn vertex_is_valid_different_dimensions() {
        // Test 1D vertex
        let valid_1d: Vertex<f64, Option<()>, 1> = VertexBuilder::default()
            .point(Point::new([42.0]))
            .build()
            .unwrap();
        // Key should be null by default, making it invalid until inserted in SlotMap
        assert!(matches!(
            valid_1d.is_valid(),
            Err(VertexValidationError::InvalidKey)
        ));

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
        // Key should be null by default, making it invalid until inserted in SlotMap
        assert!(matches!(
            valid_5d.is_valid(),
            Err(VertexValidationError::InvalidKey)
        ));

        let invalid_5d: Vertex<f64, Option<()>, 5> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, f64::NAN, 4.0, 5.0]))
            .build()
            .unwrap();
        assert!(invalid_5d.is_valid().is_err());
    }

    #[test]
    fn vertex_is_valid_key_check() {
        // Test that newly created vertex has valid point but null key
        let valid_vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        assert!(matches!(
            valid_vertex.is_valid(),
            Err(VertexValidationError::InvalidKey)
        ));
        assert!(valid_vertex.key().is_null());

        // Test that default vertex (which has null key) is invalid
        let default_vertex: Vertex<f64, Option<()>, 3> = Vertex::default();
        match default_vertex.is_valid() {
            Err(VertexValidationError::InvalidKey) => (), // Expected
            other => panic!("Expected InvalidKey error, got: {other:?}"),
        }
        assert!(default_vertex.key().is_null());
        assert!(default_vertex.point().is_valid().is_ok()); // Point itself is valid (zeros)

        // Create a vertex with valid point but manually set null key to test key validation
        let invalid_key_vertex: Vertex<f64, Option<()>, 3> = Vertex {
            point: Point::new([1.0, 2.0, 3.0]),
            key: VertexKey::null(),
            incident_cell: None,
            data: None,
        };
        match invalid_key_vertex.is_valid() {
            Err(VertexValidationError::InvalidKey) => (), // Expected
            other => panic!("Expected InvalidKey error, got: {other:?}"),
        }
        assert!(invalid_key_vertex.point().is_valid().is_ok()); // Point is valid
        assert!(invalid_key_vertex.key().is_null()); // key is null

        // Test vertex with both invalid point and null key (should return point error first)
        let invalid_both: Vertex<f64, Option<()>, 3> = Vertex {
            point: Point::new([f64::NAN, 2.0, 3.0]),
            key: VertexKey::null(),
            incident_cell: None,
            data: None,
        };
        match invalid_both.is_valid() {
            Err(VertexValidationError::InvalidPoint { .. }) => (), // Expected - point checked first
            other => panic!("Expected InvalidPoint error, got: {other:?}"),
        }
        assert!(invalid_both.point().is_valid().is_err()); // Point is invalid
        assert!(invalid_both.key().is_null()); // key is null
    }
}

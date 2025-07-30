//! Data and operations on d-dimensional vertices.
//!
//! This module provides the `Vertex` struct which represents a geometric vertex
//! in D-dimensional space with associated metadata including unique identification,
//! incident cell references, and optional user data.
//!
//! # Key Features
//!
//! - **Generic Coordinate Support**: Works with any floating-point type (`f32`, `f64`, etc.)
//!   that implements the `CoordinateScalar` trait
//! - **Unique Identification**: Each vertex has a UUID for consistent identification
//! - **Optional Data Storage**: Supports attaching arbitrary user data of type `U`
//! - **Incident Cell Tracking**: Maintains references to containing cells
//! - **Serialization Support**: Full serde support for persistence
//! - **Builder Pattern**: Convenient vertex construction using `VertexBuilder`
//!
//! # Examples
//!
//! ```rust
//! use d_delaunay::delaunay_core::vertex::Vertex;
//! use d_delaunay::vertex;
//!
//! // Create a simple vertex
//! let vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
//!
//! // Create vertex with data
//! let vertex_with_data: Vertex<f64, i32, 2> = vertex!([1.0, 2.0], 42);
//! ```

// =============================================================================
// IMPORTS
// =============================================================================

use super::{traits::DataType, utilities::make_uuid};
use crate::geometry::{
    point::Point,
    traits::coordinate::{Coordinate, CoordinateScalar, CoordinateValidationError},
};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt::Debug,
    hash::{Hash, Hasher},
};
use thiserror::Error;
use uuid::Uuid;

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Errors that can occur during vertex validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum VertexValidationError {
    /// The vertex has an invalid point.
    #[error("Invalid point: {source}")]
    InvalidPoint {
        /// The underlying point validation error.
        #[from]
        source: CoordinateValidationError,
    },
    /// The vertex has an invalid (nil) UUID.
    #[error("Invalid UUID: vertex has nil UUID which is not allowed")]
    InvalidUuid,
}

// =============================================================================
// CONVENIENCE MACROS AND HELPERS
// =============================================================================

/// Convenience macro for creating vertices with less boilerplate.
///
/// This macro simplifies vertex creation by using the `VertexBuilder` pattern internally
/// and automatically unwrapping the result for convenience. It takes coordinate arrays
/// and optional data, returning a `Vertex` directly.
///
/// # Returns
///
/// Returns `Vertex<T, U, D>` where:
/// - `T` is the coordinate scalar type
/// - `U` is the optional user data type  
/// - `D` is the spatial dimension
///
/// # Panics
///
/// Panics if the `VertexBuilder` fails to construct a valid vertex, which should
/// not happen under normal circumstances with valid input data.
///
/// # Usage
///
/// ```rust
/// use d_delaunay::vertex;
/// use d_delaunay::delaunay_core::vertex::Vertex;
///
/// // Create a vertex without data (explicit type annotation required)
/// let v1: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
///
/// // Create a vertex with data (explicit type annotation required)
/// let v2: Vertex<f64, i32, 2> = vertex!([0.0, 1.0], 42);
/// ```
#[macro_export]
macro_rules! vertex {
    // Pattern 1: Just coordinates - no data
    ($coords:expr) => {
        $crate::delaunay_core::vertex::VertexBuilder::default()
            .point($crate::geometry::point::Point::from($coords))
            .build()
            .expect("Failed to build vertex: invalid coordinates or builder configuration")
    };

    // Pattern 2: Coordinates with data
    ($coords:expr, $data:expr) => {
        $crate::delaunay_core::vertex::VertexBuilder::default()
            .point($crate::geometry::point::Point::from($coords))
            .data($data)
            .build()
            .expect("Failed to build vertex with data: invalid coordinates, data, or builder configuration")
    };
}

// Re-export the macro at the crate level for convenience
pub use crate::vertex;

// =============================================================================
// VERTEX STRUCT DEFINITION
// =============================================================================

#[derive(Builder, Clone, Copy, Debug, Default, Serialize)]
/// The `Vertex` struct represents a vertex in a triangulation with geometric
/// coordinates, unique identification, and optional metadata.
///
/// # Generic Parameters
///
/// * `T` - The scalar coordinate type (typically `f32` or `f64`)
/// * `U` - Optional user data type that implements `DataType`
/// * `D` - The spatial dimension (compile-time constant)
///
/// # Properties
///
/// - **`point`**: A `Point<T, D>` representing the geometric coordinates of the vertex
/// - **`uuid`**: A universally unique identifier for the vertex (auto-generated)
/// - **`incident_cell`**: Optional reference to a containing cell (managed by TDS)
/// - **`data`**: Optional user-defined data associated with the vertex
///
/// # Constraints
///
/// - `T` must implement `CoordinateScalar` (floating-point operations, validation, etc.)
/// - `U` must implement `DataType` (serialization, equality, hashing, etc.)
/// - `[T; D]` must support required serialization and copying operations
///
/// # Usage
///
/// Vertices are typically created using the builder pattern for convenience:
///
/// ```rust
/// use d_delaunay::delaunay_core::vertex::Vertex;
/// use d_delaunay::vertex;
///
/// let vertex: Vertex<f64, i32, 3> = vertex!([1.0, 2.0, 3.0], 42);
/// ```
pub struct Vertex<T, U, const D: usize>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The coordinates of the vertex as a D-dimensional Point.
    pub(crate) point: Point<T, D>,
    /// A universally unique identifier for the vertex.
    #[builder(setter(skip), default = "make_uuid()")]
    pub(crate) uuid: Uuid,
    /// The [Uuid] of the `Cell` that the vertex is incident to.
    #[builder(setter(skip), default = "None")]
    pub incident_cell: Option<Uuid>,
    /// Optional data associated with the vertex.
    #[builder(setter(into, strip_option), default)]
    pub data: Option<U>,
}

// =============================================================================
// DESERIALIZATION IMPLEMENTATION
// =============================================================================

/// Manual implementation of Deserialize for Vertex.
///
/// This custom implementation ensures proper handling of all vertex fields
/// during deserialization, including validation of required fields.
impl<'de, T, U, const D: usize> Deserialize<'de> for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    fn deserialize<De>(deserializer: De) -> Result<Self, De::Error>
    where
        De: serde::Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct VertexVisitor<T, U, const D: usize>
        where
            T: CoordinateScalar,
            U: DataType,
            [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
        {
            _phantom: std::marker::PhantomData<(T, U)>,
        }

        impl<'de, T, U, const D: usize> Visitor<'de> for VertexVisitor<T, U, D>
        where
            T: CoordinateScalar,
            U: DataType,
            [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
        {
            type Value = Vertex<T, U, D>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a Vertex struct")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Vertex<T, U, D>, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut point = None;
                let mut uuid = None;
                let mut incident_cell = None;
                let mut data = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        "point" => {
                            if point.is_some() {
                                return Err(de::Error::duplicate_field("point"));
                            }
                            point = Some(map.next_value()?);
                        }
                        "uuid" => {
                            if uuid.is_some() {
                                return Err(de::Error::duplicate_field("uuid"));
                            }
                            uuid = Some(map.next_value()?);
                        }
                        "incident_cell" => {
                            if incident_cell.is_some() {
                                return Err(de::Error::duplicate_field("incident_cell"));
                            }
                            incident_cell = Some(map.next_value()?);
                        }
                        "data" => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                        _ => {
                            let _ = map.next_value::<serde::de::IgnoredAny>()?;
                        }
                    }
                }

                let point = point.ok_or_else(|| de::Error::missing_field("point"))?;
                let uuid = uuid.ok_or_else(|| de::Error::missing_field("uuid"))?;
                let incident_cell = incident_cell.unwrap_or(None);
                let data = data.unwrap_or(None);

                Ok(Vertex {
                    point,
                    uuid,
                    incident_cell,
                    data,
                })
            }
        }

        const FIELDS: &[&str] = &["point", "uuid", "incident_cell", "data"];
        deserializer.deserialize_struct(
            "Vertex",
            FIELDS,
            VertexVisitor {
                _phantom: std::marker::PhantomData,
            },
        )
    }
}

// =============================================================================
// VERTEX IMPLEMENTATION - CORE METHODS
// =============================================================================

impl<T, U, const D: usize> Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
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
    /// use d_delaunay::geometry::traits::coordinate::Coordinate;
    /// let points = vec![Point::new([1.0, 2.0, 3.0])];
    /// let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points.clone());
    /// assert_eq!(vertices.len(), 1);
    /// assert_eq!(vertices[0].point().to_array(), [1.0, 2.0, 3.0]);
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
    /// use d_delaunay::geometry::traits::coordinate::Coordinate;
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
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::vertex;
    /// use d_delaunay::geometry::traits::coordinate::Coordinate;
    ///
    /// let vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
    /// let retrieved_point = vertex.point();
    /// assert_eq!(retrieved_point.to_array(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub const fn point(&self) -> &Point<T, D> {
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
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::vertex;
    /// use uuid::Uuid;
    ///
    /// let vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
    /// let vertex_uuid = vertex.uuid();
    /// // UUID should be valid and unique
    /// assert_ne!(vertex_uuid, Uuid::nil());
    ///
    /// // Creating another vertex should have a different UUID
    /// let another_vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
    /// assert_ne!(vertex.uuid(), another_vertex.uuid());
    /// ```
    #[inline]
    pub const fn uuid(&self) -> Uuid {
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
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::vertex;
    ///
    /// let vertex: Vertex<f64, Option<()>, 4> = vertex!([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(vertex.dim(), 4);
    /// ```
    #[inline]
    pub const fn dim(&self) -> usize {
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
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexValidationError};
    /// use d_delaunay::vertex;
    ///
    /// let vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
    /// assert!(vertex.is_valid().is_ok());
    ///
    /// let invalid_vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, f64::NAN, 3.0]);
    /// match invalid_vertex.is_valid() {
    ///     Err(VertexValidationError::InvalidPoint { .. }) => (), // Expected
    ///     _ => panic!("Expected point validation error"),
    /// }
    /// ```
    pub fn is_valid(self) -> Result<(), VertexValidationError>
    where
        Point<T, D>: Coordinate<T, D>,
    {
        // Check if the point is valid using the Coordinate trait validation
        self.point
            .validate()
            .map_err(|source| VertexValidationError::InvalidPoint { source })?;

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

// =============================================================================
// STANDARD TRAIT IMPLEMENTATIONS
// =============================================================================
impl<T, U, const D: usize> PartialEq for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Equality of vertices is based on ordered equality of coordinates using the Coordinate trait.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.point.ordered_equals(&other.point)
        // && self.uuid == other.uuid
        // && self.incident_cell == other.incident_cell
        // && self.data == other.data
    }
}

impl<T, U, const D: usize> PartialOrd for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Order of vertices is based on lexicographic order of coordinate arrays.
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.point.to_array().partial_cmp(&other.point.to_array())
    }
}

/// Enable implicit conversion from Vertex to coordinate array
/// This allows `vertex.point.to_array()` to be implicitly converted to `[T; D]`
impl<T, U, const D: usize> From<Vertex<T, U, D>> for [T; D]
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from(vertex: Vertex<T, U, D>) -> [T; D] {
        vertex.point().to_array()
    }
}

/// Enable implicit conversion from Vertex reference to coordinate array
/// This allows `&vertex` to be implicitly converted to `[T; D]` for coordinate access
impl<T, U, const D: usize> From<&Vertex<T, U, D>> for [T; D]
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from(vertex: &Vertex<T, U, D>) -> [T; D] {
        vertex.point().to_array()
    }
}

/// Enable implicit conversion from Vertex reference to Point
/// This allows `&vertex` to be implicitly converted to `Point<T, D>`
impl<T, U, const D: usize> From<&Vertex<T, U, D>> for Point<T, D>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from(vertex: &Vertex<T, U, D>) -> Self {
        *vertex.point()
    }
}

// =============================================================================
// HASHING AND EQUALITY IMPLEMENTATIONS
// =============================================================================
impl<T, U, const D: usize> Eq for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Self: Hash,
{
    // Generic Eq implementation for Vertex based on point equality
}

impl<T, U, const D: usize> Hash for Vertex<T, U, D>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    Point<T, D>: Hash,
{
    /// Generic Hash implementation for Vertex using Coordinate trait for point hashing
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.point.hash_coordinate(state);
        self.uuid.hash(state);
        self.incident_cell.hash(state);
        self.data.hash(state);
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use approx::assert_relative_eq;

    // =============================================================================
    // HELPER FUNCTIONS
    // =============================================================================

    /// Simplified helper function to test basic vertex properties
    fn assert_vertex_properties<T, U, const D: usize>(
        vertex: &Vertex<T, U, D>,
        expected_coords: [T; D],
    ) where
        T: CoordinateScalar,
        U: DataType,
        [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    {
        assert_eq!(vertex.point().to_array(), expected_coords);
        assert_eq!(vertex.dim(), D);
        assert!(!vertex.uuid().is_nil());
        assert!(vertex.incident_cell.is_none());
    }

    // =============================================================================
    // CONVENIENCE MACRO AND HELPER TESTS
    // =============================================================================

    #[test]
    fn test_vertex_macro() {
        // Test new macro syntax without data - no None required!
        let v1: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        assert_relative_eq!(
            v1.point().to_array().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v1.dim(), 3);
        assert!(!v1.uuid().is_nil());
        assert!(v1.data.is_none());

        // Test new macro syntax with data - no Some() required!
        let v2: Vertex<f64, i32, 2> = vertex!([0.0, 1.0], 99);
        assert_relative_eq!(
            v2.point().to_array().as_slice(),
            [0.0, 1.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v2.dim(), 2);
        assert!(!v2.uuid().is_nil());
        assert_eq!(v2.data.unwrap(), 99);

        // Test macro with different data type (using Copy type)
        let v3: Vertex<f64, u32, 4> = vertex!([1.0, 2.0, 3.0, 4.0], 42u32);
        assert_relative_eq!(
            v3.point().to_array().as_slice(),
            [1.0f64, 2.0f64, 3.0f64, 4.0f64].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(v3.dim(), 4);
        assert_eq!(v3.data.unwrap(), 42u32);
    }

    // =============================================================================
    // BASIC VERTEX FUNCTIONALITY
    // =============================================================================

    #[test]
    fn vertex_default() {
        let vertex: Vertex<f64, Option<()>, 3> = Vertex::default();

        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
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
    fn vertex_copy() {
        let vertex: Vertex<f64, u8, 4> = vertex!([1.0, 2.0, 3.0, 4.0], 4u8);
        let vertex_copy = vertex;

        assert_eq!(vertex, vertex_copy);
        assert_relative_eq!(
            vertex_copy.point().to_array().as_slice(),
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
            vertices[0].point().to_array().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[0].dim(), 3);
        assert_relative_eq!(
            vertices[1].point().to_array().as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertices[1].dim(), 3);
        assert_relative_eq!(
            vertices[2].point().to_array().as_slice(),
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
    fn vertex_to_and_from_json() {
        let vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        let serialized = serde_json::to_string(&vertex).unwrap();

        assert!(serialized.contains("point"));
        assert!(serialized.contains("[1.0,2.0,3.0]"));

        // Test deserialization with manual Deserialize implementation
        let deserialized: Vertex<f64, Option<()>, 3> = serde_json::from_str(&serialized).unwrap();

        // Check that deserialized vertex has same point coordinates using approx equality
        assert_relative_eq!(
            deserialized.point().to_array().as_slice(),
            vertex.point().to_array().as_slice(),
            epsilon = f64::EPSILON
        );
        assert_eq!(deserialized.dim(), vertex.dim());
        assert_eq!(deserialized.incident_cell, vertex.incident_cell);
        assert_eq!(deserialized.data, vertex.data);
        // Note: UUID will be different as it's loaded from serialized data
        assert_eq!(deserialized.uuid(), vertex.uuid());

        // Human readable output for cargo test -- --nocapture
        println!("Serialized: {serialized:?}");
    }

    #[test]
    fn vertex_partial_eq() {
        let vertex1: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        let vertex2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        let vertex3: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 4.0]);

        assert_eq!(vertex1, vertex2);
        assert_ne!(vertex2, vertex3);
    }

    #[test]
    fn vertex_partial_ord() {
        let vertex1: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        let vertex2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 4.0]);
        let vertex3: Vertex<f64, Option<()>, 3> = vertex!([10.0, 0.0, 0.0]);
        let vertex4: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 10.0]);

        assert!(vertex1 < vertex2);
        assert!(vertex3 > vertex2);
        assert!(vertex1 < vertex3);
        assert!(vertex1 > vertex4);
    }

    #[test]
    fn vertex_hash() {
        use std::collections::hash_map::DefaultHasher;

        let vertex1: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        let vertex2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        let vertex3: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 4.0]);

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

        let vertex1: Vertex<f64, Option<()>, 2> = vertex!([1.0, 2.0]);
        let vertex2: Vertex<f64, Option<()>, 2> = vertex!([3.0, 4.0]);

        map.insert(vertex1, 10);
        map.insert(vertex2, 20);

        assert_eq!(map.get(&vertex1), Some(&10));
        assert_eq!(map.get(&vertex2), Some(&20));
        assert_eq!(map.len(), 2);
    }

    // =============================================================================
    // DIMENSION-SPECIFIC TESTS
    // =============================================================================

    #[test]
    fn vertex_1d() {
        let vertex: Vertex<f64, Option<()>, 1> = vertex!([42.0]);
        assert_vertex_properties(&vertex, [42.0]);
        assert!(vertex.data.is_none());
    }

    #[test]
    fn vertex_2d() {
        let vertex: Vertex<f64, Option<()>, 2> = vertex!([1.0, 2.0]);
        assert_vertex_properties(&vertex, [1.0, 2.0]);
        assert!(vertex.data.is_none());
    }

    #[test]
    fn vertex_4d() {
        let vertex: Vertex<f64, Option<()>, 4> = vertex!([1.0, 2.0, 3.0, 4.0]);
        assert_vertex_properties(&vertex, [1.0, 2.0, 3.0, 4.0]);
        assert!(vertex.data.is_none());
    }

    #[test]
    fn vertex_5d() {
        let vertex: Vertex<f64, Option<()>, 5> = vertex!([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_vertex_properties(&vertex, [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(vertex.data.is_none());
    }

    // =============================================================================
    // NUMERIC TYPE TESTS
    // =============================================================================

    #[test]
    fn vertex_with_f32() {
        let vertex: Vertex<f32, Option<()>, 2> = vertex!([1.5, 2.5]);

        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [1.5, 2.5].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 2);
        assert!(!vertex.uuid().is_nil());
    }

    // =============================================================================
    // DATA TYPE TESTS
    // =============================================================================

    #[test]
    fn vertex_with_tuple_data() {
        let vertex: Vertex<f64, (i32, i32), 2> = vertex!([1.0, 2.0], (42, 84));
        assert_vertex_properties(&vertex, [1.0, 2.0]);
        assert_eq!(vertex.data.unwrap(), (42, 84));
    }

    #[test]
    fn vertex_debug_format() {
        let vertex: Vertex<f64, i32, 3> = vertex!([1.0, 2.0, 3.0], 42);
        let debug_str = format!("{vertex:?}");

        assert!(debug_str.contains("Vertex"));
        assert!(debug_str.contains("point"));
        assert!(debug_str.contains("uuid"));
        assert!(debug_str.contains("1.0"));
        assert!(debug_str.contains("2.0"));
        assert!(debug_str.contains("3.0"));
    }

    #[test]
    fn vertex_ordering_edge_cases() {
        let vertex1: Vertex<f64, Option<()>, 2> = vertex!([1.0, 2.0]);
        let vertex2: Vertex<f64, Option<()>, 2> = vertex!([1.0, 2.0]);

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

    // =============================================================================
    // COORDINATE VALUE TESTS
    // =============================================================================

    #[test]
    fn vertex_negative_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = vertex!([-1.0, -2.0, -3.0]);

        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [-1.0, -2.0, -3.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
    }

    #[test]
    fn vertex_zero_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);

        let origin_vertex: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);

        assert_eq!(vertex.point(), origin_vertex.point());
    }

    #[test]
    fn vertex_large_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = vertex!([1e6, 2e6, 3e6]);

        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [1_000_000.0, 2_000_000.0, 3_000_000.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
    }

    #[test]
    fn vertex_small_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = vertex!([1e-6, 2e-6, 3e-6]);

        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [0.000_001, 0.000_002, 0.000_003].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 3);
    }

    #[test]
    fn vertex_mixed_positive_negative_coordinates() {
        let vertex: Vertex<f64, Option<()>, 4> = vertex!([1.0, -2.0, 3.0, -4.0]);

        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [1.0, -2.0, 3.0, -4.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(vertex.dim(), 4);
    }

    // =============================================================================
    // COLLECTION OPERATIONS TESTS
    // =============================================================================

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
            vertices[0].point().to_array().as_slice(),
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
        let vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        let uuid = vertex.uuid();
        let vertices = vec![vertex];
        let hashmap = Vertex::into_hashmap(vertices);

        assert_eq!(hashmap.len(), 1);
        assert!(hashmap.contains_key(&uuid));
        assert_relative_eq!(
            hashmap.get(&uuid).unwrap().point().to_array().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
    }

    // =============================================================================
    // VERTEX PROPERTIES TESTS
    // =============================================================================

    #[test]
    fn vertex_uuid_uniqueness() {
        let vertex1: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        let vertex2: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);

        // Same points but different UUIDs
        assert_ne!(vertex1.uuid(), vertex2.uuid());
        assert!(!vertex1.uuid().is_nil());
        assert!(!vertex2.uuid().is_nil());
    }

    // =============================================================================
    // TYPE CONVERSION TESTS
    // =============================================================================

    #[test]
    fn vertex_implicit_conversion_to_coordinates() {
        let vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);

        // Test implicit conversion from owned vertex
        let coords_owned: [f64; 3] = vertex.into();
        assert_relative_eq!(
            coords_owned.as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Create a new vertex for reference test
        let vertex_ref: Vertex<f64, Option<()>, 3> = vertex!([4.0, 5.0, 6.0]);

        // Test implicit conversion from vertex reference
        let coords_ref: [f64; 3] = (&vertex_ref).into();
        assert_relative_eq!(
            coords_ref.as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );

        // Verify the original vertex is still available after reference conversion
        assert_relative_eq!(
            vertex_ref.point().to_array().as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn vertex_implicit_conversion_to_point() {
        let vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);

        // Test implicit conversion from vertex reference to Point
        let point_from_vertex: Point<f64, 3> = (&vertex).into();
        assert_relative_eq!(
            point_from_vertex.to_array().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test that the converted point is equal to the original point
        assert_eq!(point_from_vertex, *vertex.point());

        // Verify the original vertex is still available after conversion
        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );

        // Test with different dimensions
        let vertex_2d: Vertex<f64, Option<()>, 2> = vertex!([10.5, -5.3]);

        let point_2d: Point<f64, 2> = (&vertex_2d).into();
        assert_relative_eq!(
            point_2d.to_array().as_slice(),
            [10.5, -5.3].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(point_2d, *vertex_2d.point());
    }

    // =============================================================================
    // VALIDATION TESTS
    // =============================================================================

    #[test]
    fn vertex_is_valid_f64() {
        // Test valid vertex with finite coordinates
        let valid_vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        assert!(valid_vertex.is_valid().is_ok());

        // Test valid vertex with negative coordinates
        let valid_negative: Vertex<f64, Option<()>, 3> = vertex!([-1.0, -2.0, -3.0]);
        assert!(valid_negative.is_valid().is_ok());

        // Test valid vertex with zero coordinates
        let valid_zero: Vertex<f64, Option<()>, 3> = vertex!([0.0, 0.0, 0.0]);
        assert!(valid_zero.is_valid().is_ok());

        // Test invalid vertex with NaN coordinate
        let invalid_nan: Vertex<f64, Option<()>, 3> = vertex!([1.0, f64::NAN, 3.0]);
        assert!(invalid_nan.is_valid().is_err());

        // Test invalid vertex with all NaN coordinates
        let invalid_all_nan: Vertex<f64, Option<()>, 3> = vertex!([f64::NAN, f64::NAN, f64::NAN]);
        assert!(invalid_all_nan.is_valid().is_err());

        // Test invalid vertex with positive infinity
        let invalid_pos_inf: Vertex<f64, Option<()>, 3> = vertex!([1.0, f64::INFINITY, 3.0]);
        assert!(invalid_pos_inf.is_valid().is_err());

        // Test invalid vertex with negative infinity
        let invalid_neg_inf: Vertex<f64, Option<()>, 3> = vertex!([1.0, f64::NEG_INFINITY, 3.0]);
        assert!(invalid_neg_inf.is_valid().is_err());

        // Test invalid vertex with mixed NaN and infinity
        let invalid_mixed: Vertex<f64, Option<()>, 3> = vertex!([f64::NAN, f64::INFINITY, 1.0]);
        assert!(invalid_mixed.is_valid().is_err());
    }

    #[test]
    fn vertex_is_valid_f32() {
        // Test valid f32 vertex
        let valid_vertex: Vertex<f32, Option<()>, 2> = vertex!([1.5f32, 2.5f32]);
        assert!(valid_vertex.is_valid().is_ok());

        // Test invalid f32 vertex with NaN
        let invalid_nan: Vertex<f32, Option<()>, 2> = vertex!([1.0f32, f32::NAN]);
        assert!(invalid_nan.is_valid().is_err());

        // Test invalid f32 vertex with infinity
        let invalid_inf: Vertex<f32, Option<()>, 2> = vertex!([f32::INFINITY, 2.0f32]);
        assert!(invalid_inf.is_valid().is_err());
    }

    #[test]
    fn vertex_is_valid_different_dimensions() {
        // Test 1D vertex
        let valid_1d: Vertex<f64, Option<()>, 1> = vertex!([42.0]);
        assert!(valid_1d.is_valid().is_ok());

        let invalid_1d: Vertex<f64, Option<()>, 1> = vertex!([f64::NAN]);
        assert!(invalid_1d.is_valid().is_err());

        // Test 5D vertex
        let valid_5d: Vertex<f64, Option<()>, 5> = vertex!([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(valid_5d.is_valid().is_ok());

        let invalid_5d: Vertex<f64, Option<()>, 5> = vertex!([1.0, 2.0, f64::NAN, 4.0, 5.0]);
        assert!(invalid_5d.is_valid().is_err());
    }

    #[test]
    fn vertex_is_valid_uuid_check() {
        // Test that vertex with valid point and UUID is valid
        let valid_vertex: Vertex<f64, Option<()>, 3> = vertex!([1.0, 2.0, 3.0]);
        assert!(valid_vertex.is_valid().is_ok());
        assert!(!valid_vertex.uuid().is_nil());

        // Test that default vertex (which has nil UUID) is invalid
        let default_vertex: Vertex<f64, Option<()>, 3> = Vertex::default();
        match default_vertex.is_valid() {
            Err(VertexValidationError::InvalidUuid) => (), // Expected
            other => panic!("Expected InvalidUuid error, got: {other:?}"),
        }
        assert!(default_vertex.uuid().is_nil());
        assert!(default_vertex.point().validate().is_ok());

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
        assert!(invalid_uuid_vertex.point().validate().is_ok());
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
        assert!(invalid_both.point().validate().is_err());
        assert!(invalid_both.uuid().is_nil()); // UUID is nil
    }

    // =============================================================================
    // ADVANCED DATA TESTS
    // =============================================================================

    #[test]
    fn vertex_from_points_with_str_data() {
        // Test creating vertices from points and then adding Copy data
        let points = vec![Point::new([1.0, 2.0]), Point::new([3.0, 4.0])];
        let mut vertices: Vec<Vertex<f64, u8, 2>> = Vertex::from_points(points);

        // Add Copy u8 data to each vertex
        vertices[0].data = Some(1u8);
        vertices[1].data = Some(2u8);

        assert_eq!(vertices[0].data.unwrap(), 1u8);
        assert_eq!(vertices[1].data.unwrap(), 2u8);
        assert_eq!(vertices.len(), 2);
    }

    #[test]
    fn vertex_hash_with_copy_data() {
        // Test hashing with Copy data
        use std::collections::HashMap;

        let vertex1: Vertex<f64, u16, 2> = vertex!([1.0, 2.0], 999u16);

        let vertex2: Vertex<f64, i32, 2> = vertex!([3.0, 4.0], 42);

        // Test that vertices with Copy data can be used as HashMap keys
        let mut map: HashMap<Vertex<f64, u16, 2>, i32> = HashMap::new();
        map.insert(vertex1, 100);

        let mut map2: HashMap<Vertex<f64, i32, 2>, u8> = HashMap::new();
        map2.insert(vertex2, 255u8);

        assert_eq!(map.len(), 1);
        assert_eq!(map2.len(), 1);
    }

    // =============================================================================
    // DESERIALIZATION ERROR PATH TESTS
    // =============================================================================

    #[test]
    fn test_deserialize_duplicate_point_field() {
        // Test duplicate "point" field error path (line 252)
        let json_with_duplicate_point = r#"{
            "point": [1.0, 2.0, 3.0],
            "point": [4.0, 5.0, 6.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "incident_cell": null,
            "data": null
        }"#;

        let result: Result<Vertex<f64, Option<()>, 3>, _> =
            serde_json::from_str(json_with_duplicate_point);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("duplicate") || error_message.contains("point"));
    }

    #[test]
    fn test_deserialize_duplicate_uuid_field() {
        // Test duplicate "uuid" field error path (line 258)
        let json_with_duplicate_uuid = r#"{
            "point": [1.0, 2.0, 3.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "uuid": "550e8400-e29b-41d4-a716-446655440001", 
            "incident_cell": null,
            "data": null
        }"#;

        let result: Result<Vertex<f64, Option<()>, 3>, _> =
            serde_json::from_str(json_with_duplicate_uuid);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("duplicate") || error_message.contains("uuid"));
    }

    #[test]
    fn test_deserialize_duplicate_incident_cell_field() {
        // Test duplicate "incident_cell" field error path (line 264)
        let json_with_duplicate_incident_cell = r#"{
            "point": [1.0, 2.0, 3.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "incident_cell": null,
            "incident_cell": "550e8400-e29b-41d4-a716-446655440001",
            "data": null
        }"#;

        let result: Result<Vertex<f64, Option<()>, 3>, _> =
            serde_json::from_str(json_with_duplicate_incident_cell);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("duplicate") || error_message.contains("incident_cell"));
    }

    #[test]
    fn test_deserialize_duplicate_data_field() {
        // Test duplicate "data" field error path (line 270)
        let json_with_duplicate_data = r#"{
            "point": [1.0, 2.0, 3.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "incident_cell": null,
            "data": 42,
            "data": 84
        }"#;

        let result: Result<Vertex<f64, i32, 3>, _> = serde_json::from_str(json_with_duplicate_data);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("duplicate") || error_message.contains("data"));
    }

    #[test]
    fn test_deserialize_missing_point_field() {
        // Test missing "point" field error path (line 280)
        let json_without_point = r#"{
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "incident_cell": null,
            "data": null
        }"#;

        let result: Result<Vertex<f64, Option<()>, 3>, _> =
            serde_json::from_str(json_without_point);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("missing") || error_message.contains("point"));
    }

    #[test]
    fn test_deserialize_missing_uuid_field() {
        // Test missing "uuid" field error path (line 281)
        let json_without_uuid = r#"{
            "point": [1.0, 2.0, 3.0],
            "incident_cell": null,
            "data": null
        }"#;

        let result: Result<Vertex<f64, Option<()>, 3>, _> = serde_json::from_str(json_without_uuid);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("missing") || error_message.contains("uuid"));
    }

    #[test]
    fn test_deserialize_unknown_field() {
        // Test unknown field handling (line 275)
        let json_with_unknown_field = r#"{
            "point": [1.0, 2.0, 3.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "incident_cell": null,
            "data": null,
            "unknown_field": "this should be ignored"
        }"#;

        let result: Result<Vertex<f64, Option<()>, 3>, _> =
            serde_json::from_str(json_with_unknown_field);
        // This should succeed - unknown fields are ignored
        assert!(result.is_ok());
        let vertex = result.unwrap();
        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_deserialize_expecting_formatter() {
        // Test the expecting formatter method (line 236)
        // This will trigger the expecting method when serde needs to format an error
        let invalid_json = r#"["not", "a", "vertex", "object"]"#;
        let result: Result<Vertex<f64, Option<()>, 3>, _> = serde_json::from_str(invalid_json);
        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        // The error should mention that it expected a Vertex struct
        assert!(
            error_message.contains("Vertex") || error_message.to_lowercase().contains("struct")
        );
    }

    // =============================================================================
    // VERTEX VALIDATION ERROR TESTS
    // =============================================================================

    #[test]
    fn test_vertex_validation_nil_uuid() {
        // Test validation error for nil UUID (line 503)
        let vertex_with_nil_uuid = Vertex {
            point: Point::new([1.0, 2.0, 3.0]),
            uuid: uuid::Uuid::nil(),
            incident_cell: None,
            data: None::<Option<()>>,
        };

        let validation_result = vertex_with_nil_uuid.is_valid();
        assert!(validation_result.is_err());
        match validation_result.unwrap_err() {
            VertexValidationError::InvalidUuid => (), // Expected
            other @ VertexValidationError::InvalidPoint { .. } => {
                panic!("Expected InvalidUuid error, got: {other:?}")
            }
        }
    }

    // =============================================================================
    // COMPREHENSIVE DESERIALIZATION TESTS
    // =============================================================================

    #[test]
    fn test_comprehensive_deserialization_with_all_fields() {
        // Test successful deserialization with all optional fields present
        let json_with_all_fields = r#"{
            "point": [1.5, 2.5, 3.5],
            "uuid": "550e8400-e29b-41d4-a716-446655440000",
            "incident_cell": "650e8400-e29b-41d4-a716-446655440000",
            "data": 123
        }"#;

        let result: Result<Vertex<f64, i32, 3>, _> = serde_json::from_str(json_with_all_fields);
        assert!(result.is_ok());
        let vertex = result.unwrap();

        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [1.5, 2.5, 3.5].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(
            vertex.uuid().to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
        assert!(vertex.incident_cell.is_some());
        assert_eq!(
            vertex.incident_cell.unwrap().to_string(),
            "650e8400-e29b-41d4-a716-446655440000"
        );
        assert_eq!(vertex.data.unwrap(), 123);
    }

    #[test]
    fn test_deserialization_with_minimal_fields() {
        // Test deserialization with only required fields
        let json_minimal = r#"{
            "point": [10.0, 20.0],
            "uuid": "550e8400-e29b-41d4-a716-446655440000"
        }"#;

        let result: Result<Vertex<f64, Option<()>, 2>, _> = serde_json::from_str(json_minimal);
        assert!(result.is_ok());
        let vertex = result.unwrap();

        assert_relative_eq!(
            vertex.point().to_array().as_slice(),
            [10.0, 20.0].as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(
            vertex.uuid().to_string(),
            "550e8400-e29b-41d4-a716-446655440000"
        );
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_none());
    }

    // =============================================================================
    // ERROR HANDLING EDGE CASES
    // =============================================================================

    #[test]
    fn test_vertex_validation_error_display() {
        // Test error display formatting
        let point_error =
            crate::geometry::traits::coordinate::CoordinateValidationError::InvalidCoordinate {
                coordinate_index: 1,
                coordinate_value: "NaN".to_string(),
                dimension: 3,
            };
        let vertex_error = VertexValidationError::InvalidPoint {
            source: point_error,
        };
        let error_string = format!("{vertex_error}");
        assert!(error_string.contains("Invalid point"));

        let uuid_error = VertexValidationError::InvalidUuid;
        let uuid_error_string = format!("{uuid_error}");
        assert!(uuid_error_string.contains("Invalid UUID"));
    }

    #[test]
    fn test_vertex_validation_error_equality() {
        // Test PartialEq for VertexValidationError
        let error1 = VertexValidationError::InvalidUuid;
        let error2 = VertexValidationError::InvalidUuid;
        assert_eq!(error1, error2);

        let point_error =
            crate::geometry::traits::coordinate::CoordinateValidationError::InvalidCoordinate {
                coordinate_index: 1,
                coordinate_value: "NaN".to_string(),
                dimension: 3,
            };
        let error3 = VertexValidationError::InvalidPoint {
            source: point_error.clone(),
        };
        let error4 = VertexValidationError::InvalidPoint {
            source: point_error,
        };
        assert_eq!(error3, error4);

        assert_ne!(error1, error3);
    }

    // =============================================================================
    // SERIALIZATION ROUNDTRIP TESTS
    // =============================================================================

    #[test]
    fn test_serialization_deserialization_roundtrip() {
        // Test that serialization -> deserialization preserves all data
        let original_vertex: Vertex<f64, char, 4> = vertex!([1.0, 2.0, 3.0, 4.0], 'A');

        // Serialize
        let serialized = serde_json::to_string(&original_vertex).unwrap();

        // Deserialize
        let deserialized_vertex: Vertex<f64, char, 4> = serde_json::from_str(&serialized).unwrap();

        // Verify all fields match
        assert_relative_eq!(
            original_vertex.point().to_array().as_slice(),
            deserialized_vertex.point().to_array().as_slice(),
            epsilon = 1e-9
        );
        assert_eq!(original_vertex.uuid(), deserialized_vertex.uuid());
        assert_eq!(
            original_vertex.incident_cell,
            deserialized_vertex.incident_cell
        );
        assert_eq!(original_vertex.data, deserialized_vertex.data);
    }
}

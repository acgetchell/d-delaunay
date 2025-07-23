//! Coordinate trait for abstracting coordinate storage and operations.
//!
//! This module provides the `Coordinate` trait which unifies all coordinate-related
//! functionality including floating-point operations, hashing, equality comparisons,
//! and validation. This trait abstracts away the specific storage mechanism for
//! coordinates, enabling flexibility between arrays, vectors, or other storage types.
//!
//! # Overview
//!
//! The `Coordinate` trait was created to address the need for abstracting coordinate
//! functionality that was previously spread across individual trait requirements
//! in `Point`, `Vertex`, `Cell`, `Facet`, and `TriangulationDataStructure`. It consolidates
//! all the trait bounds that these structures need:
//!
//! - `Float` for floating-point arithmetic operations
//! - `Hash` for use in hash-based collections like `HashMap` and `HashSet`
//! - `PartialEq` and `Eq` for equality comparisons
//! - `OrderedEq` for NaN-aware equality that treats NaN values as equal to themselves
//! - `FiniteCheck` for validation of coordinate values
//! - `HashCoordinate` for consistent hashing of floating-point values
//! - Serialization traits (`Serialize`, `Deserialize`)
//!
//! # Benefits
//!
//! 1. **Abstraction**: The storage mechanism (arrays, vectors, hash maps, etc.) is
//!    abstracted away, allowing future flexibility in how coordinates are stored.
//!
//! 2. **Trait Consolidation**: All coordinate-related trait bounds are consolidated
//!    into a single trait, simplifying the trait bounds on geometric structures.
//!
//! 3. **Consistent Interface**: All coordinate implementations provide the same
//!    interface regardless of underlying storage mechanism.
//!
//! 4. **Future Extensibility**: New coordinate storage types can be added easily
//!    by implementing the `Coordinate` trait.
//!
//! # Usage with Existing Code
//!
//! The current `Point` structure uses arrays directly. The `Coordinate` trait provides
//! a path for future refactoring where `Point` could be parameterized over different
//! coordinate storage types while maintaining the same API.

use super::{FiniteCheck, HashCoordinate, OrderedEq};
use num_traits::{Float, Zero};
use serde::{Serialize, de::DeserializeOwned};
use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
};

/// A comprehensive trait that encapsulates all coordinate functionality.
///
/// This trait combines all the necessary traits for coordinate types used in
/// geometric computations, providing a single unified interface for coordinate
/// storage and operations. It abstracts the storage mechanism, allowing for
/// different implementations (arrays, vectors, hash maps, etc.) while ensuring
/// consistent behavior.
///
/// # Type Parameters
///
/// * `T` - The scalar type for coordinates (typically f32 or f64)
/// * `const D: usize` - The dimension of the coordinate system
///
/// # Required Functionality
///
/// The trait requires implementors to support:
/// - Floating-point arithmetic operations
/// - Ordered equality comparison (NaN-aware)
/// - Hashing for use in collections
/// - Validation of coordinate values
/// - Serialization/deserialization
/// - Coordinate access and manipulation
/// - Zero/origin creation
///
/// # Examples
///
/// ```ignore
/// // This trait will be implemented for different storage types
/// // For example, with array storage:
/// use d_delaunay::geometry::{ArrayCoordinate, Coordinate};
///
/// // Create coordinates using array storage
/// let coord1: ArrayCoordinate<f64, 3> = ArrayCoordinate::from_array([1.0, 2.0, 3.0]);
/// let coord2: ArrayCoordinate<f64, 3> = ArrayCoordinate::from_array([1.0, 2.0, 3.0]);
///
/// // All coordinate types implement the same trait
/// assert_eq!(coord1.dim(), 3);
/// assert_eq!(coord1.to_array(), [1.0, 2.0, 3.0]);
/// assert_eq!(coord1, coord2);
///
/// // Validate coordinates
/// assert!(coord1.validate().is_ok());
///
/// // Create origin coordinate
/// let origin: ArrayCoordinate<f64, 3> = ArrayCoordinate::origin();
/// assert_eq!(origin.to_array(), [0.0, 0.0, 0.0]);
/// ```
///
/// # Future Storage Implementations
///
/// The trait is designed to support various storage mechanisms:
///
/// ```ignore
/// // Future vector-based coordinate (not yet implemented)
/// struct VectorCoordinate<T> {
///     coords: Vec<T>,
/// }
///
/// // Future hash-based coordinate for sparse dimensions (not yet implemented)
/// struct SparseCoordinate<T, const D: usize> {
///     coords: HashMap<usize, T>,
/// }
///
/// // All would implement the same Coordinate trait
/// ```
pub trait Coordinate<T, const D: usize>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    Self: Copy
        + Clone
        + Default
        + Debug
        + PartialEq
        + Eq
        + Hash
        + PartialOrd
        + Serialize
        + DeserializeOwned
        + Sized,
{
    /// Get the dimensionality of the coordinate system.
    ///
    /// # Returns
    ///
    /// The number of dimensions (D) in the coordinate system.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let coord = SomeCoordinate::new([1.0, 2.0, 3.0]);
    /// assert_eq!(coord.dim(), 3);
    /// ```
    fn dim(&self) -> usize {
        D
    }

    /// Create a new coordinate from an array of scalar values.
    ///
    /// # Arguments
    ///
    /// * `coords` - Array of coordinates of type T with dimension D
    ///
    /// # Returns
    ///
    /// A new coordinate instance with the specified values.
    fn from_array(coords: [T; D]) -> Self;

    /// Create a new coordinate from an array of scalar values.
    /// This is an alias for `from_array` to maintain API compatibility.
    ///
    /// # Arguments
    ///
    /// * `coords` - Array of coordinates of type T with dimension D
    ///
    /// # Returns
    ///
    /// A new coordinate instance with the specified values.
    fn new(coords: [T; D]) -> Self {
        Self::from_array(coords)
    }

    /// Convert the coordinate to an array of scalar values.
    ///
    /// # Returns
    ///
    /// An array containing the coordinate values.
    fn to_array(&self) -> [T; D];

    /// Get a specific coordinate by index.
    ///
    /// # Arguments
    ///
    /// * `index` - The index of the coordinate to retrieve (0-based)
    ///
    /// # Returns
    ///
    /// The coordinate value at the specified index, or None if index is out of bounds.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let coord = SomeCoordinate::new([1.0, 2.0, 3.0]);
    /// assert_eq!(coord.get(0), Some(1.0));
    /// assert_eq!(coord.get(3), None);
    /// ```
    fn get(&self, index: usize) -> Option<T>;

    /// Create a coordinate at the origin (all zeros).
    ///
    /// # Returns
    ///
    /// A new coordinate with all values set to zero.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let origin = SomeCoordinate::origin();
    /// assert_eq!(origin.to_array(), [0.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    fn origin() -> Self
    where
        T: Zero,
    {
        Self::from_array([T::zero(); D])
    }

    /// Validate that all coordinate values are finite.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if all coordinates are finite (not NaN or infinite),
    /// otherwise returns an error describing which coordinate is invalid.
    ///
    /// # Errors
    ///
    /// Returns `CoordinateValidationError::InvalidCoordinate` if any coordinate
    /// is NaN, infinite, or otherwise not finite. The error includes details about
    /// which coordinate index is invalid, its value, and the coordinate dimension.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// let valid = SomeCoordinate::new([1.0, 2.0, 3.0]);
    /// assert!(valid.validate().is_ok());
    ///
    /// let invalid = SomeCoordinate::new([1.0, f64::NAN, 3.0]);
    /// assert!(invalid.validate().is_err());
    /// ```
    fn validate(&self) -> Result<(), CoordinateValidationError>;

    /// Compute the hash of this coordinate.
    ///
    /// This method provides consistent hashing across different coordinate
    /// implementations, including proper handling of special floating-point values.
    ///
    /// # Arguments
    ///
    /// * `state` - The hasher state to write to
    fn hash_coordinate<H: Hasher>(&self, state: &mut H);

    /// Test equality with another coordinate using ordered comparison.
    ///
    /// This method uses ordered comparison semantics that treat NaN values
    /// as equal to themselves, enabling coordinates with NaN to be used in
    /// hash-based collections.
    ///
    /// # Arguments
    ///
    /// * `other` - The other coordinate to compare with
    ///
    /// # Returns
    ///
    /// True if coordinates are equal using ordered comparison.
    fn ordered_equals(&self, other: &Self) -> bool;
}

/// Errors that can occur during coordinate validation.
#[derive(Clone, Debug, thiserror::Error, PartialEq)]
pub enum CoordinateValidationError {
    /// A coordinate value is invalid (NaN or infinite).
    #[error(
        "Invalid coordinate at index {coordinate_index} in dimension {dimension}: {coordinate_value}"
    )]
    InvalidCoordinate {
        /// Index of the invalid coordinate.
        coordinate_index: usize,
        /// Value of the invalid coordinate, as a string.
        coordinate_value: String,
        /// The dimensionality of the coordinate system.
        dimension: usize,
    },
}

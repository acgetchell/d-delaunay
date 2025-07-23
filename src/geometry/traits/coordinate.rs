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
/// ```rust
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

/// Array-based coordinate implementation.
///
/// This provides a concrete implementation of the `Coordinate` trait using
/// fixed-size arrays as the underlying storage mechanism.
#[derive(Clone, Copy, Debug, Default, PartialOrd)]
pub struct ArrayCoordinate<T, const D: usize>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    coords: [T; D],
}

impl<T, const D: usize> ArrayCoordinate<T, D>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Create a new array coordinate.
    ///
    /// # Arguments
    ///
    /// * `coords` - Array of coordinate values
    ///
    /// # Returns
    ///
    /// A new `ArrayCoordinate` instance.
    #[inline]
    pub const fn new(coords: [T; D]) -> Self {
        Self { coords }
    }

    /// Get the raw coordinate array.
    ///
    /// # Returns
    ///
    /// A copy of the internal coordinate array.
    #[inline]
    pub const fn coordinates(&self) -> [T; D] {
        self.coords
    }
}

impl<T, const D: usize> Coordinate<T, D> for ArrayCoordinate<T, D>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from_array(coords: [T; D]) -> Self {
        Self::new(coords)
    }

    #[inline]
    fn to_array(&self) -> [T; D] {
        self.coords
    }

    fn get(&self, index: usize) -> Option<T> {
        self.coords.get(index).copied()
    }

    fn validate(&self) -> Result<(), CoordinateValidationError> {
        for (index, &coord) in self.coords.iter().enumerate() {
            if !coord.is_finite_generic() {
                return Err(CoordinateValidationError::InvalidCoordinate {
                    coordinate_index: index,
                    coordinate_value: format!("{coord:?}"),
                    dimension: D,
                });
            }
        }
        Ok(())
    }

    fn hash_coordinate<H: Hasher>(&self, state: &mut H) {
        for &coord in &self.coords {
            coord.hash_coord(state);
        }
    }

    fn ordered_equals(&self, other: &Self) -> bool {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .all(|(a, b)| a.ordered_eq(b))
    }
}

impl<T, const D: usize> PartialEq for ArrayCoordinate<T, D>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    fn eq(&self, other: &Self) -> bool {
        self.ordered_equals(other)
    }
}

impl<T, const D: usize> Eq for ArrayCoordinate<T, D>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
}

impl<T, const D: usize> Hash for ArrayCoordinate<T, D>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_coordinate(state);
    }
}

impl<T, const D: usize> serde::Serialize for ArrayCoordinate<T, D>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.coords.serialize(serializer)
    }
}

impl<'de, T, const D: usize> serde::Deserialize<'de> for ArrayCoordinate<T, D>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    fn deserialize<DE>(deserializer: DE) -> Result<Self, DE::Error>
    where
        DE: serde::Deserializer<'de>,
    {
        let coords = <[T; D]>::deserialize(deserializer)?;
        Ok(Self::new(coords))
    }
}

// Conversion traits for ArrayCoordinate
impl<T, const D: usize> From<[T; D]> for ArrayCoordinate<T, D>
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from(coords: [T; D]) -> Self {
        Self::new(coords)
    }
}

impl<T, const D: usize> From<ArrayCoordinate<T, D>> for [T; D]
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from(coord: ArrayCoordinate<T, D>) -> [T; D] {
        coord.coordinates()
    }
}

impl<T, const D: usize> From<&ArrayCoordinate<T, D>> for [T; D]
where
    T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn from(coord: &ArrayCoordinate<T, D>) -> [T; D] {
        coord.coordinates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::{HashMap, HashSet};

    #[test]
    fn array_coordinate_creation() {
        let coord = ArrayCoordinate::new([1.0, 2.0, 3.0]);
        assert_eq!(coord.dim(), 3);
        assert_relative_eq!(coord.to_array().as_slice(), [1.0, 2.0, 3.0].as_slice());
    }

    #[test]
    fn array_coordinate_from_array() {
        let coord = ArrayCoordinate::from_array([4.0, 5.0]);
        assert_eq!(coord.dim(), 2);
        assert_relative_eq!(coord.to_array().as_slice(), [4.0, 5.0].as_slice());
    }

    #[test]
    fn array_coordinate_get() {
        let coord = ArrayCoordinate::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(coord.get(0), Some(1.0));
        assert_eq!(coord.get(2), Some(3.0));
        assert_eq!(coord.get(4), None);
    }

    #[test]
    fn array_coordinate_origin() {
        let origin: ArrayCoordinate<f64, 3> = ArrayCoordinate::origin();
        assert_relative_eq!(origin.to_array().as_slice(), [0.0, 0.0, 0.0].as_slice());
    }

    #[test]
    fn array_coordinate_validation() {
        let valid = ArrayCoordinate::new([1.0, 2.0, 3.0]);
        assert!(valid.validate().is_ok());

        let invalid_nan = ArrayCoordinate::new([1.0, f64::NAN, 3.0]);
        assert!(invalid_nan.validate().is_err());

        let invalid_inf = ArrayCoordinate::new([f64::INFINITY, 2.0, 3.0]);
        assert!(invalid_inf.validate().is_err());
    }

    #[test]
    fn array_coordinate_equality() {
        let coord1 = ArrayCoordinate::new([1.0, 2.0, 3.0]);
        let coord2 = ArrayCoordinate::new([1.0, 2.0, 3.0]);
        let coord3 = ArrayCoordinate::new([1.0, 2.0, 4.0]);

        assert_eq!(coord1, coord2);
        assert_ne!(coord1, coord3);

        // Test NaN equality
        let coord_nan1 = ArrayCoordinate::new([f64::NAN, 2.0]);
        let coord_nan2 = ArrayCoordinate::new([f64::NAN, 2.0]);
        assert_eq!(coord_nan1, coord_nan2);
    }

    #[test]
    fn array_coordinate_hash_consistency() {
        let coord1 = ArrayCoordinate::new([1.0, 2.0, 3.0]);
        let coord2 = ArrayCoordinate::new([1.0, 2.0, 3.0]);
        let coord3 = ArrayCoordinate::new([1.0, 2.0, 4.0]);

        let mut map: HashMap<ArrayCoordinate<f64, 3>, &str> = HashMap::new();
        map.insert(coord1, "first");

        assert!(map.contains_key(&coord2));
        assert!(!map.contains_key(&coord3));
    }

    #[test]
    fn array_coordinate_in_collections() {
        let mut set: HashSet<ArrayCoordinate<f64, 2>> = HashSet::new();

        set.insert(ArrayCoordinate::new([1.0, 2.0]));
        set.insert(ArrayCoordinate::new([1.0, 2.0])); // Duplicate
        set.insert(ArrayCoordinate::new([3.0, 4.0]));

        assert_eq!(set.len(), 2);
        assert!(set.contains(&ArrayCoordinate::new([1.0, 2.0])));
        assert!(set.contains(&ArrayCoordinate::new([3.0, 4.0])));
    }

    #[test]
    fn array_coordinate_special_values() {
        // Test with NaN
        let coord_nan = ArrayCoordinate::new([f64::NAN, 1.0]);
        let coord_nan2 = ArrayCoordinate::new([f64::NAN, 1.0]);
        assert_eq!(coord_nan, coord_nan2);

        // Test with infinity
        let coord_inf = ArrayCoordinate::new([f64::INFINITY, 1.0]);
        let coord_inf2 = ArrayCoordinate::new([f64::INFINITY, 1.0]);
        assert_eq!(coord_inf, coord_inf2);

        // Test different special values are not equal
        assert_ne!(coord_nan, coord_inf);
    }

    #[test]
    fn array_coordinate_conversions() {
        let coords = [1.0, 2.0, 3.0];
        let coord = ArrayCoordinate::from(coords);

        // Test conversion back to array
        let back_to_array: [f64; 3] = coord.into();
        assert_relative_eq!(back_to_array.as_slice(), coords.as_slice());

        // Test reference conversion
        let ref_to_array: [f64; 3] = (&coord).into();
        assert_relative_eq!(ref_to_array.as_slice(), coords.as_slice());
    }

    #[test]
    fn array_coordinate_serialization() {
        let coord = ArrayCoordinate::new([1.0, 2.0, 3.0]);

        let serialized = serde_json::to_string(&coord).unwrap();
        let deserialized: ArrayCoordinate<f64, 3> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(coord, deserialized);
    }

    #[test]
    fn coordinate_trait_bounds() {
        // Test that ArrayCoordinate satisfies all the required trait bounds
        fn assert_coordinate_traits<C, T, const D: usize>()
        where
            C: Coordinate<T, D>,
            T: Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Copy + Debug,
        {
            // This function exists purely to test trait bounds at compile time
        }

        assert_coordinate_traits::<ArrayCoordinate<f64, 3>, f64, 3>();
        assert_coordinate_traits::<ArrayCoordinate<f32, 2>, f32, 2>();
    }

    #[test]
    fn validation_error_details() {
        let invalid = ArrayCoordinate::new([1.0, f64::NAN, 3.0]);
        let result = invalid.validate();

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 1);
            assert_eq!(dimension, 3);
            assert!(coordinate_value.contains("NaN"));
        } else {
            panic!("Expected InvalidCoordinate error");
        }
    }

    #[test]
    fn mixed_dimensional_coordinates() {
        // Test different dimensional coordinates
        let coord_1d: ArrayCoordinate<f64, 1> = ArrayCoordinate::new([42.0]);
        let coord_2d: ArrayCoordinate<f64, 2> = ArrayCoordinate::new([1.0, 2.0]);
        let coord_5d: ArrayCoordinate<f64, 5> = ArrayCoordinate::new([1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(coord_1d.dim(), 1);
        assert_eq!(coord_2d.dim(), 2);
        assert_eq!(coord_5d.dim(), 5);

        // Test origins
        let origin_1d: ArrayCoordinate<f64, 1> = ArrayCoordinate::origin();
        let origin_2d: ArrayCoordinate<f64, 2> = ArrayCoordinate::origin();

        assert_relative_eq!(origin_1d.to_array().as_slice(), [0.0].as_slice());
        assert_relative_eq!(origin_2d.to_array().as_slice(), [0.0, 0.0].as_slice());
    }
}

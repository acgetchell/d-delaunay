//! Coordinate trait for abstracting coordinate storage and operations.
//!
//! This module provides the `Coordinate` trait which unifies all coordinate-related
//! functionality including floating-point operations, hashing, equality comparisons,
//! and validation. This trait abstracts away the specific storage mechanism for
//! coordinates, enabling flexibility between arrays, vectors, or other storage types.
//!
//! # Overview
//!
//! The `Coordinate` trait provides a unified abstraction for coordinate operations
//! across different scalar types and storage mechanisms. All geometric structures
//! (`Point`, `Vertex`, `Cell`, `Facet`, and `TriangulationDataStructure`) now use
//! generic type parameters constrained by this trait, enabling support for multiple
//! floating-point precision levels (`f32`, `f64`, etc.). The trait consolidates
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

/// Default tolerance for f32 floating-point comparisons.
///
/// This value is set to 1e-6, which is appropriate for f32 precision and provides
/// a reasonable margin for floating-point comparison errors.
pub const DEFAULT_TOLERANCE_F32: f32 = 1e-6;

/// Default tolerance for f64 floating-point comparisons.
///
/// This value is set to 1e-15, which is appropriate for f64 precision and provides
/// a reasonable margin for floating-point comparison errors.
pub const DEFAULT_TOLERANCE_F64: f64 = 1e-15;

/// Trait alias for the scalar type requirements in coordinate systems.
///
/// This alias captures all the trait bounds required for a scalar type `T` to be used
/// in coordinate systems. It consolidates the requirements from line 116 of the
/// `Coordinate` trait definition to reduce code duplication.
///
/// # Required Traits
///
/// - `Float`: Floating-point arithmetic operations
/// - `OrderedEq`: NaN-aware equality comparison
/// - `HashCoordinate`: Consistent hashing of floating-point values
/// - `FiniteCheck`: Validation of coordinate values
/// - `Default`: Default value construction
/// - `Copy`: Copy semantics for efficient operations
/// - `Debug`: Debug formatting
/// - `Serialize`: Serialization support
/// - `DeserializeOwned`: Deserialization support
///
/// # Usage
///
/// ```rust
/// use d_delaunay::geometry::traits::coordinate::CoordinateScalar;
///
/// fn process_coordinate<T: CoordinateScalar>(value: T) {
///     // T has all the necessary bounds for coordinate operations
/// }
/// ```
pub trait CoordinateScalar:
    Float + OrderedEq + HashCoordinate + FiniteCheck + Default + Debug + Serialize + DeserializeOwned
{
    /// Returns the appropriate default tolerance for this coordinate scalar type.
    ///
    /// This method provides type-specific tolerance values that are appropriate
    /// for floating-point comparisons and geometric computations. The tolerance
    /// values are chosen to account for the precision limitations of each
    /// floating-point type.
    ///
    /// # Returns
    ///
    /// The default tolerance value for this type:
    /// - For `f32`: `1e-6` (appropriate for single precision)
    /// - For `f64`: `1e-15` (appropriate for double precision)
    ///
    /// # Examples
    ///
    /// ```
    /// use d_delaunay::geometry::traits::coordinate::CoordinateScalar;
    ///
    /// // Get appropriate tolerance for f32
    /// let tolerance_f32 = f32::default_tolerance();
    /// assert_eq!(tolerance_f32, 1e-6_f32);
    ///
    /// // Get appropriate tolerance for f64
    /// let tolerance_f64 = f64::default_tolerance();
    /// assert_eq!(tolerance_f64, 1e-15_f64);
    /// ```
    ///
    /// # Usage in Generic Functions
    ///
    /// This method is particularly useful in generic functions that need
    /// appropriate tolerance values for the specific type being used:
    ///
    /// ```
    /// use d_delaunay::geometry::traits::coordinate::CoordinateScalar;
    ///
    /// fn compare_with_tolerance<T: CoordinateScalar>(a: T, b: T) -> bool {
    ///     (a - b).abs() < T::default_tolerance()
    /// }
    /// ```
    fn default_tolerance() -> Self;
}

// Specific implementations for f32 and f64
impl CoordinateScalar for f32 {
    fn default_tolerance() -> Self {
        DEFAULT_TOLERANCE_F32
    }
}

impl CoordinateScalar for f64 {
    fn default_tolerance() -> Self {
        DEFAULT_TOLERANCE_F64
    }
}

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
/// ```
/// use d_delaunay::geometry::{point::Point, traits::coordinate::Coordinate};
///
/// // Create coordinates using Point (which implements Coordinate)
/// let coord1: Point<f64, 3> = Coordinate::new([1.0, 2.0, 3.0]);
/// let coord2: Point<f64, 3> = Coordinate::new([1.0, 2.0, 3.0]);
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
/// let origin: Point<f64, 3> = Point::origin();
/// assert_eq!(origin.to_array(), [0.0, 0.0, 0.0]);
/// ```
///
/// # Future Storage Implementations
///
/// The trait is designed to support various storage mechanisms:
///
/// ```
/// // Example of how future implementations could work:
/// use d_delaunay::geometry::{point::Point, traits::coordinate::Coordinate};
/// use std::collections::HashMap;
///
/// // Current Point implementation uses arrays
/// let point_coord: Point<f64, 2> = Coordinate::new([1.0, 2.0]);
/// assert_eq!(point_coord.dim(), 2);
/// assert_eq!(point_coord.to_array(), [1.0, 2.0]);
///
/// // Future implementations could use other storage types
/// // while maintaining the same Coordinate trait interface
/// ```
pub trait Coordinate<T, const D: usize>
where
    T: CoordinateScalar,
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
    /// ```
    /// use d_delaunay::geometry::{point::Point, traits::coordinate::Coordinate};
    ///
    /// let coord: Point<f64, 3> = Coordinate::new([1.0, 2.0, 3.0]);
    /// assert_eq!(coord.dim(), 3);
    /// ```
    #[must_use]
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
    fn new(coords: [T; D]) -> Self;

    /// Convert the coordinate to an array of scalar values.
    ///
    /// # Returns
    ///
    /// An array containing the coordinate values.
    #[must_use]
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
    /// ```
    /// use d_delaunay::geometry::{point::Point, traits::coordinate::Coordinate};
    ///
    /// let coord: Point<f64, 3> = Coordinate::new([1.0, 2.0, 3.0]);
    /// assert_eq!(coord.get(0), Some(1.0));
    /// assert_eq!(coord.get(3), None);
    /// ```
    #[must_use]
    fn get(&self, index: usize) -> Option<T>;

    /// Create a coordinate at the origin (all zeros).
    ///
    /// # Returns
    ///
    /// A new coordinate with all values set to zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use d_delaunay::geometry::{point::Point, traits::coordinate::Coordinate};
    ///
    /// let origin: Point<f64, 3> = Coordinate::origin();
    /// assert_eq!(origin.to_array(), [0.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    fn origin() -> Self
    where
        T: Zero,
    {
        Self::new([T::zero(); D])
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
    /// ```
    /// use d_delaunay::geometry::{point::Point, traits::coordinate::Coordinate};
    ///
    /// let valid: Point<f64, 3> = Coordinate::new([1.0, 2.0, 3.0]);
    /// assert!(valid.validate().is_ok());
    ///
    /// let invalid: Point<f64, 3> = Coordinate::new([1.0, f64::NAN, 3.0]);
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
    #[must_use]
    fn ordered_equals(&self, other: &Self) -> bool;
}

/// Errors that can occur during coordinate validation.
#[derive(Clone, Debug, thiserror::Error, PartialEq, Eq)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::point::Point;
    use approx::assert_relative_eq;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    // Use the global tolerance constants

    #[test]
    fn coordinate_trait_basic_functionality() {
        // Test through Point implementation of Coordinate trait
        let coord: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);

        // Test dim()
        assert_eq!(coord.dim(), 3);

        // Test to_array()
        assert_relative_eq!(
            coord.to_array().as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );

        // Test get()
        assert_relative_eq!(coord.get(0).unwrap(), 1.0, epsilon = DEFAULT_TOLERANCE_F64);
        assert_relative_eq!(coord.get(1).unwrap(), 2.0, epsilon = DEFAULT_TOLERANCE_F64);
        assert_relative_eq!(coord.get(2).unwrap(), 3.0, epsilon = DEFAULT_TOLERANCE_F64);
        assert_eq!(coord.get(3), None);
        assert_eq!(coord.get(10), None);
    }

    #[test]
    fn coordinate_trait_new() {
        // Test new() method
        let coord1: Point<f64, 2> = Coordinate::new([5.0, 6.0]);
        assert_relative_eq!(
            coord1.to_array().as_slice(),
            [5.0, 6.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );

        // Test multiple creations with new()
        let coord2: Point<f64, 2> = Coordinate::new([5.0, 6.0]);
        assert_relative_eq!(
            coord2.to_array().as_slice(),
            [5.0, 6.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );

        // They should be equal
        assert_eq!(coord1, coord2);
    }

    #[test]
    fn coordinate_trait_origin() {
        // Test origin for different dimensions
        let origin_1d: Point<f64, 1> = Point::origin();
        assert_relative_eq!(
            origin_1d.to_array().as_slice(),
            [0.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );

        let origin_2d: Point<f64, 2> = Point::origin();
        assert_relative_eq!(
            origin_2d.to_array().as_slice(),
            [0.0, 0.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );

        let origin_3d: Point<f64, 3> = Point::origin();
        assert_relative_eq!(
            origin_3d.to_array().as_slice(),
            [0.0, 0.0, 0.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );

        let origin_5d: Point<f64, 5> = Point::origin();
        assert_relative_eq!(
            origin_5d.to_array().as_slice(),
            [0.0, 0.0, 0.0, 0.0, 0.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );
    }

    #[test]
    fn coordinate_trait_origin_different_types() {
        // Test origin with f32
        let origin_f32: Point<f32, 3> = Point::origin();
        assert_relative_eq!(
            origin_f32.to_array().as_slice(),
            [0.0f32, 0.0f32, 0.0f32].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F32
        );

        // Test origin with f64
        let origin_f64: Point<f64, 3> = Point::origin();
        assert_relative_eq!(
            origin_f64.to_array().as_slice(),
            [0.0f64, 0.0f64, 0.0f64].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );
    }

    #[test]
    fn coordinate_trait_validate_valid() {
        // Test validation with valid coordinates
        let valid_coord: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        assert!(valid_coord.validate().is_ok());

        // Test with negative coordinates
        let negative_coord: Point<f64, 3> = Point::new([-1.0, -2.0, -3.0]);
        assert!(negative_coord.validate().is_ok());

        // Test with zero coordinates
        let zero_coord: Point<f64, 3> = Point::new([0.0, 0.0, 0.0]);
        assert!(zero_coord.validate().is_ok());

        // Test with large coordinates
        let large_coord: Point<f64, 3> = Point::new([1e10, 2e10, 3e10]);
        assert!(large_coord.validate().is_ok());

        // Test with small coordinates
        let small_coord: Point<f64, 3> = Point::new([1e-10, 2e-10, 3e-10]);
        assert!(small_coord.validate().is_ok());
    }

    #[test]
    fn coordinate_trait_validate_invalid_nan() {
        // Test validation with NaN coordinates
        let nan_first: Point<f64, 3> = Point::new([f64::NAN, 2.0, 3.0]);
        let result = nan_first.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value: _,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 0);
            assert_eq!(dimension, 3);
        } else {
            panic!("Expected InvalidCoordinate error");
        }

        // Test NaN in middle position
        let nan_middle: Point<f64, 3> = Point::new([1.0, f64::NAN, 3.0]);
        let result = nan_middle.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value: _,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 1);
            assert_eq!(dimension, 3);
        }

        // Test NaN in last position
        let nan_last: Point<f64, 3> = Point::new([1.0, 2.0, f64::NAN]);
        let result = nan_last.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value: _,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 2);
            assert_eq!(dimension, 3);
        }
    }

    #[test]
    fn coordinate_trait_validate_invalid_infinity() {
        // Test validation with positive infinity
        let pos_inf: Point<f64, 2> = Point::new([f64::INFINITY, 2.0]);
        let result = pos_inf.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value: _,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 0);
            assert_eq!(dimension, 2);
        }

        // Test validation with negative infinity
        let neg_inf: Point<f64, 2> = Point::new([1.0, f64::NEG_INFINITY]);
        let result = neg_inf.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value: _,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 1);
            assert_eq!(dimension, 2);
        }
    }

    #[test]
    fn coordinate_trait_validate_first_invalid_reported() {
        // When multiple coordinates are invalid, the first one should be reported
        let multi_invalid: Point<f64, 4> = Point::new([f64::NAN, f64::INFINITY, f64::NAN, 1.0]);
        let result = multi_invalid.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            coordinate_value: _,
            dimension,
        }) = result
        {
            assert_eq!(coordinate_index, 0); // First invalid coordinate
            assert_eq!(dimension, 4);
        }
    }

    #[test]
    fn coordinate_trait_validate_different_dimensions() {
        // Test validation in 1D
        let invalid_1d: Point<f64, 1> = Point::new([f64::NAN]);
        let result = invalid_1d.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate { dimension, .. }) = result {
            assert_eq!(dimension, 1);
        }

        // Test validation in 5D
        let invalid_5d: Point<f64, 5> = Point::new([1.0, 2.0, f64::INFINITY, 4.0, 5.0]);
        let result = invalid_5d.validate();
        assert!(result.is_err());

        if let Err(CoordinateValidationError::InvalidCoordinate {
            coordinate_index,
            dimension,
            ..
        }) = result
        {
            assert_eq!(coordinate_index, 2);
            assert_eq!(dimension, 5);
        }
    }

    #[test]
    fn coordinate_trait_hash_coordinate() {
        // Test hash_coordinate method
        let coord1: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        let coord2: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        let coord3: Point<f64, 3> = Point::new([1.0, 2.0, 4.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        let mut hasher3 = DefaultHasher::new();

        coord1.hash_coordinate(&mut hasher1);
        coord2.hash_coordinate(&mut hasher2);
        coord3.hash_coordinate(&mut hasher3);

        let hash1 = hasher1.finish();
        let hash2 = hasher2.finish();
        let hash3 = hasher3.finish();

        // Same coordinates should have same hash
        assert_eq!(hash1, hash2);

        // Different coordinates should have different hash (with high probability)
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn coordinate_trait_hash_coordinate_special_values() {
        // Test hash_coordinate with special floating-point values
        let nan_coord: Point<f64, 2> = Point::new([f64::NAN, 1.0]);
        let another_nan_coord: Point<f64, 2> = Point::new([f64::NAN, 1.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        nan_coord.hash_coordinate(&mut hasher1);
        another_nan_coord.hash_coordinate(&mut hasher2);

        // NaN coordinates should hash consistently
        assert_eq!(hasher1.finish(), hasher2.finish());

        // Test with infinity
        let inf_coord: Point<f64, 2> = Point::new([f64::INFINITY, 1.0]);
        let another_inf_coord: Point<f64, 2> = Point::new([f64::INFINITY, 1.0]);

        let mut hasher3 = DefaultHasher::new();
        let mut hasher4 = DefaultHasher::new();

        inf_coord.hash_coordinate(&mut hasher3);
        another_inf_coord.hash_coordinate(&mut hasher4);

        // Infinity coordinates should hash consistently
        assert_eq!(hasher3.finish(), hasher4.finish());
    }

    #[test]
    fn coordinate_trait_ordered_equals() {
        // Test ordered_equals with normal values
        let coord1: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        let coord2: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        let coord3: Point<f64, 3> = Point::new([1.0, 2.0, 4.0]);

        assert!(coord1.ordered_equals(&coord2));
        assert!(coord2.ordered_equals(&coord1));
        assert!(!coord1.ordered_equals(&coord3));
        assert!(!coord3.ordered_equals(&coord1));
    }

    #[test]
    fn coordinate_trait_ordered_equals_nan() {
        // Test ordered_equals with NaN values - they should be equal to themselves
        let nan_coord1: Point<f64, 3> = Point::new([f64::NAN, 2.0, 3.0]);
        let nan_coord2: Point<f64, 3> = Point::new([f64::NAN, 2.0, 3.0]);
        let normal_coord: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);

        // NaN coordinates should be equal to themselves using ordered equality
        assert!(nan_coord1.ordered_equals(&nan_coord2));
        assert!(nan_coord2.ordered_equals(&nan_coord1));

        // NaN coordinates should not be equal to normal coordinates
        assert!(!nan_coord1.ordered_equals(&normal_coord));
        assert!(!normal_coord.ordered_equals(&nan_coord1));

        // Multiple NaN coordinates
        let multi_nan1: Point<f64, 3> = Point::new([f64::NAN, f64::NAN, 3.0]);
        let multi_nan2: Point<f64, 3> = Point::new([f64::NAN, f64::NAN, 3.0]);
        assert!(multi_nan1.ordered_equals(&multi_nan2));
    }

    #[test]
    fn coordinate_trait_ordered_equals_infinity() {
        // Test ordered_equals with infinity values
        let inf_coord1: Point<f64, 2> = Point::new([f64::INFINITY, 2.0]);
        let inf_coord2: Point<f64, 2> = Point::new([f64::INFINITY, 2.0]);
        let neg_inf_coord: Point<f64, 2> = Point::new([f64::NEG_INFINITY, 2.0]);

        assert!(inf_coord1.ordered_equals(&inf_coord2));
        assert!(!inf_coord1.ordered_equals(&neg_inf_coord));
        assert!(!neg_inf_coord.ordered_equals(&inf_coord1));
    }

    #[test]
    fn coordinate_trait_ordered_equals_mixed_special_values() {
        // Test ordered_equals with mixed special values
        let mixed1: Point<f64, 4> = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0]);
        let mixed2: Point<f64, 4> = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.0]);
        let mixed3: Point<f64, 4> = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0]);

        assert!(mixed1.ordered_equals(&mixed2));
        assert!(!mixed1.ordered_equals(&mixed3));
    }

    #[test]
    fn coordinate_trait_different_dimensions() {
        // Test that the trait works with different dimensions
        let coord_1d: Point<f64, 1> = Point::new([42.0]);
        assert_eq!(coord_1d.dim(), 1);
        assert_relative_eq!(
            coord_1d.to_array().as_slice(),
            [42.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );
        assert_relative_eq!(
            coord_1d.get(0).unwrap(),
            42.0,
            epsilon = DEFAULT_TOLERANCE_F64
        );
        assert_eq!(coord_1d.get(1), None);

        let coord_2d: Point<f64, 2> = Point::new([1.0, 2.0]);
        assert_eq!(coord_2d.dim(), 2);
        assert_relative_eq!(
            coord_2d.to_array().as_slice(),
            [1.0, 2.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );

        let coord_5d: Point<f64, 5> = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(coord_5d.dim(), 5);
        assert_relative_eq!(
            coord_5d.to_array().as_slice(),
            [1.0, 2.0, 3.0, 4.0, 5.0].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );
        assert_relative_eq!(
            coord_5d.get(4).unwrap(),
            5.0,
            epsilon = DEFAULT_TOLERANCE_F64
        );
        assert_eq!(coord_5d.get(5), None);
    }

    #[test]
    fn coordinate_trait_different_numeric_types() {
        // Test with f32
        let coord_f32: Point<f32, 3> = Point::new([1.5f32, 2.5f32, 3.5f32]);
        assert_eq!(coord_f32.dim(), 3);
        assert_relative_eq!(
            coord_f32.to_array().as_slice(),
            [1.5f32, 2.5f32, 3.5f32].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F32
        );
        assert!(coord_f32.validate().is_ok());

        // Test get() method with f32
        assert_relative_eq!(
            coord_f32.get(0).unwrap(),
            1.5f32,
            epsilon = DEFAULT_TOLERANCE_F32
        );
        assert_relative_eq!(
            coord_f32.get(1).unwrap(),
            2.5f32,
            epsilon = DEFAULT_TOLERANCE_F32
        );
        assert_relative_eq!(
            coord_f32.get(2).unwrap(),
            3.5f32,
            epsilon = DEFAULT_TOLERANCE_F32
        );
        assert_eq!(coord_f32.get(3), None);

        // Test with f64
        let coord_f64: Point<f64, 3> = Point::new([1.5f64, 2.5f64, 3.5f64]);
        assert_eq!(coord_f64.dim(), 3);
        assert_relative_eq!(
            coord_f64.to_array().as_slice(),
            [1.5f64, 2.5f64, 3.5f64].as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );
        assert!(coord_f64.validate().is_ok());

        // Test get() method with f64
        assert_relative_eq!(
            coord_f64.get(0).unwrap(),
            1.5f64,
            epsilon = DEFAULT_TOLERANCE_F64
        );
        assert_relative_eq!(
            coord_f64.get(1).unwrap(),
            2.5f64,
            epsilon = DEFAULT_TOLERANCE_F64
        );
        assert_relative_eq!(
            coord_f64.get(2).unwrap(),
            3.5f64,
            epsilon = DEFAULT_TOLERANCE_F64
        );
        assert_eq!(coord_f64.get(3), None);
    }

    #[test]
    fn coordinate_validation_error_properties() {
        // Test CoordinateValidationError properties
        let error = CoordinateValidationError::InvalidCoordinate {
            coordinate_index: 1,
            coordinate_value: "NaN".to_string(),
            dimension: 3,
        };

        // Test Debug trait
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("InvalidCoordinate"));
        assert!(debug_str.contains("coordinate_index: 1"));
        assert!(debug_str.contains("dimension: 3"));

        // Test Display trait (from Error trait)
        let display_str = format!("{error}");
        assert!(display_str.contains("Invalid coordinate at index 1 in dimension 3: NaN"));

        // Test Clone and PartialEq
        let error_clone = error.clone();
        assert_eq!(error, error_clone);

        let different_error = CoordinateValidationError::InvalidCoordinate {
            coordinate_index: 2,
            coordinate_value: "inf".to_string(),
            dimension: 3,
        };
        assert_ne!(error, different_error);
    }

    #[test]
    fn coordinate_validation_error_different_scenarios() {
        // Test error with different coordinate values and indices
        let scenarios = vec![
            (0, "NaN".to_string(), 1),
            (2, "inf".to_string(), 3),
            (4, "-inf".to_string(), 5),
        ];

        for (index, value, dim) in scenarios {
            let error = CoordinateValidationError::InvalidCoordinate {
                coordinate_index: index,
                coordinate_value: value.clone(),
                dimension: dim,
            };

            let display_str = format!("{error}");
            assert!(display_str.contains(&format!("index {index}")));
            assert!(display_str.contains(&format!("dimension {dim}")));
            assert!(display_str.contains(&value));
        }
    }

    #[test]
    fn coordinate_trait_f32_precision_tests() {
        // Test f32 operations that might have precision issues
        let coord_f32: Point<f32, 3> = Point::new([0.1f32, 0.2f32, 0.3f32]);

        // Test that retrieving values works with appropriate precision
        assert_relative_eq!(
            coord_f32.get(0).unwrap(),
            0.1f32,
            epsilon = f32::EPSILON * 4.0
        );
        assert_relative_eq!(
            coord_f32.get(1).unwrap(),
            0.2f32,
            epsilon = f32::EPSILON * 4.0
        );
        assert_relative_eq!(
            coord_f32.get(2).unwrap(),
            0.3f32,
            epsilon = f32::EPSILON * 4.0
        );

        // Test conversion back to array
        let array = coord_f32.to_array();
        assert_relative_eq!(array[0], 0.1f32, epsilon = f32::EPSILON * 4.0);
        assert_relative_eq!(array[1], 0.2f32, epsilon = f32::EPSILON * 4.0);
        assert_relative_eq!(array[2], 0.3f32, epsilon = f32::EPSILON * 4.0);

        // Test arithmetic operations that might compound floating-point errors
        let computed_sum = 0.1f32 + 0.2f32;
        let coord_sum: Point<f32, 1> = Point::new([computed_sum]);
        assert_relative_eq!(
            coord_sum.get(0).unwrap(),
            0.3f32,
            epsilon = f32::EPSILON * 8.0
        );

        // Test very small f32 values
        let small_coord: Point<f32, 2> = Point::new([f32::EPSILON, f32::EPSILON * 2.0]);
        assert_relative_eq!(
            small_coord.get(0).unwrap(),
            f32::EPSILON,
            epsilon = f32::EPSILON * 2.0
        );
        assert_relative_eq!(
            small_coord.get(1).unwrap(),
            f32::EPSILON * 2.0,
            epsilon = f32::EPSILON * 4.0
        );

        // Test origin with f32
        let origin_f32: Point<f32, 4> = Point::origin();
        for i in 0..4 {
            assert_relative_eq!(origin_f32.get(i).unwrap(), 0.0f32, epsilon = f32::EPSILON);
        }

        // Test validation with f32 edge cases
        let edge_f32: Point<f32, 3> = Point::new([f32::MIN, f32::MAX, 1.0f32]);
        assert!(edge_f32.validate().is_ok());

        // Test f32 coordinates that are close but not exactly equal
        let close_coord1: Point<f32, 2> = Point::new([1.000_000_1_f32, 2.000_000_1_f32]);
        let close_coord2: Point<f32, 2> = Point::new([1.000_000_2_f32, 2.000_000_2_f32]);

        // These should be different when compared exactly, but close with approx
        assert!(close_coord1 != close_coord2);
        assert_relative_eq!(
            close_coord1.to_array().as_slice(),
            close_coord2.to_array().as_slice(),
            epsilon = 1e-6 // Larger epsilon for this comparison
        );
    }

    #[test]
    fn coordinate_trait_f32_special_values() {
        // Test f32 with special floating-point values

        // Test f32 NaN handling
        let nan_f32: Point<f32, 2> = Point::new([f32::NAN, 1.5f32]);
        assert!(nan_f32.validate().is_err());

        // Test f32 infinity handling
        let inf_f32: Point<f32, 2> = Point::new([f32::INFINITY, 1.5f32]);
        assert!(inf_f32.validate().is_err());

        let neg_inf_f32: Point<f32, 2> = Point::new([f32::NEG_INFINITY, 1.5f32]);
        assert!(neg_inf_f32.validate().is_err());

        // Test f32 ordered equality with special values
        let nan_coord1_f32: Point<f32, 2> = Point::new([f32::NAN, 2.0f32]);
        let nan_coord2_f32: Point<f32, 2> = Point::new([f32::NAN, 2.0f32]);
        assert!(nan_coord1_f32.ordered_equals(&nan_coord2_f32));

        // Test f32 hash consistency with special values
        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        nan_coord1_f32.hash_coordinate(&mut hasher1);
        nan_coord2_f32.hash_coordinate(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }

    #[test]
    fn coordinate_scalar_default_tolerance() {
        // Test using tolerance in generic function
        fn test_tolerance<T: CoordinateScalar>(a: T, b: T) -> bool {
            (a - b).abs() < T::default_tolerance()
        }

        // Test that default_tolerance returns the expected values
        assert_relative_eq!(
            f32::default_tolerance(),
            DEFAULT_TOLERANCE_F32,
            epsilon = f32::EPSILON
        );
        assert_relative_eq!(
            f64::default_tolerance(),
            DEFAULT_TOLERANCE_F64,
            epsilon = f64::EPSILON
        );

        // Test that the tolerance values are reasonable
        assert_relative_eq!(f32::default_tolerance(), 1e-6_f32, epsilon = f32::EPSILON);
        assert_relative_eq!(f64::default_tolerance(), 1e-15_f64, epsilon = f64::EPSILON);

        // Test with f32
        let a_f32 = 1.0f32;
        let b_f32 = 1.0f32 + f32::default_tolerance() / 2.0;
        assert!(test_tolerance(a_f32, b_f32));

        // Test with f64
        let a_f64 = 1.0f64;
        let b_f64 = 1.0f64 + f64::default_tolerance() / 2.0;
        assert!(test_tolerance(a_f64, b_f64));

        // Test that tolerance values are different for different types
        assert!(f64::from(f32::default_tolerance()) > f64::default_tolerance());
    }

    #[test]
    fn coordinate_trait_consistency_with_point_methods() {
        // Ensure Coordinate trait methods are consistent with Point's direct methods
        let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]); // Through Coordinate::new
        let point_direct = Point::new([1.0, 2.0, 3.0]); // Through Point::new

        // They should be identical
        assert_eq!(point, point_direct);
        assert_relative_eq!(
            point.to_array().as_slice(),
            point_direct.to_array().as_slice(),
            epsilon = DEFAULT_TOLERANCE_F64
        );

        // Test origin consistency
        let origin_trait: Point<f64, 3> = Point::origin(); // Through Coordinate::origin
        let origin_direct = Point::new([0.0, 0.0, 0.0]); // Direct construction

        assert_eq!(origin_trait, origin_direct);

        // Test consistency with f32 as well
        let point_f32: Point<f32, 2> = Point::new([1.5f32, 2.5f32]);
        let point_f32_direct = Point::new([1.5f32, 2.5f32]);
        assert_eq!(point_f32, point_f32_direct);

        // Test f32 array consistency
        assert_relative_eq!(
            point_f32.to_array().as_slice(),
            point_f32_direct.to_array().as_slice(),
            epsilon = DEFAULT_TOLERANCE_F32
        );
    }
}

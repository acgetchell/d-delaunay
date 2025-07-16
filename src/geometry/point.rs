//! Data and operations on d-dimensional points.
//!
//! # Special Floating-Point Equality Semantics
//!
//! This module implements custom equality semantics for floating-point coordinates
//! that differ from the IEEE 754 standard. Specifically, `NaN` values are treated
//! as equal to themselves to satisfy the requirements of the `Eq` trait and enable
//! Points to be used as keys in hash-based collections.
//!
//! This means that for Points containing floating-point coordinates:
//! - `Point::new([f64::NAN]) == Point::new([f64::NAN])` returns `true`
//! - Points with NaN values can be used as `HashMap` keys
//! - All NaN bit patterns are considered equal
//!
//! If you need standard IEEE 754 equality semantics, compare the coordinates
//! directly instead of using Point equality.

#![allow(clippy::similar_names)]

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use thiserror::Error;

/// A trait representing an abstract vector in D-dimensional space.
///
/// This trait defines the basic operations and properties required for
/// vector storage and manipulation in D-dimensional space. It is used
/// as the underlying storage for points in the library.
///
/// # Type Parameters
///
/// * `T`: The scalar type for coordinates (f32 or f64).
/// * `D`: The dimensionality of the vector.
///
/// # Required Methods
///
/// * `from_array`: Constructs a vector from an array of scalar values.
/// * `as_slice`: Provides a slice view of the vector's coordinates.
pub trait VectorN<T, const D: usize> {
    /// Construct vector from array of scalars.
    fn from_array(arr: [T; D]) -> Self;
    /// Borrow coordinates as slice.
    fn as_slice(&self) -> &[T];
}

// Default implementation using nalgebra SVector for f64
use nalgebra::SVector;
impl<const D: usize> VectorN<f64, D> for SVector<f64, D> {
    fn from_array(arr: [f64; D]) -> Self {
        SVector::<f64, D>::from(arr)
    }
    fn as_slice(&self) -> &[f64] {
        self.as_slice()
    }
}

// Default implementation using nalgebra SVector for f32
impl<const D: usize> VectorN<f32, D> for SVector<f32, D> {
    fn from_array(arr: [f32; D]) -> Self {
        SVector::<f32, D>::from(arr)
    }
    fn as_slice(&self) -> &[f32] {
        self.as_slice()
    }
}

/// Enum representing validation errors for a [`Point`].
#[derive(Error, Debug, PartialEq, Clone)]
pub enum PointValidationError {
    /// A coordinate is invalid (NaN or infinite).
    #[error("Invalid coordinate at index {coordinate_index} in dimension {dimension}: {coordinate_value}")]
    InvalidCoordinate {
        /// Index of the invalid coordinate in the point.
        coordinate_index: usize,
        /// Value of the invalid coordinate, presented as a string.
        coordinate_value: String,
        /// The dimensionality (number of coordinates) of the point.
        dimension: usize,
    },
}

/// A point in D-dimensional space, backed by an abstract vector storage.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct Point<T, V, const D: usize>
where
    T: Copy + PartialEq + PartialOrd + Debug,
    V: VectorN<T, D>,
{
    storage: V,
    _phantom: PhantomData<T>,
}

/// Type alias for 2D point using f64 coordinates
pub type Point2D = Point<f64, SVector<f64, 2>, 2>;
/// Type alias for 3D point using f64 coordinates
pub type Point3D = Point<f64, SVector<f64, 3>, 3>;
/// Type alias for a D-dimensional point using f64 coordinates
///
/// This alias simplifies the creation of points with arbitrary dimensions
/// backed by `nalgebra::SVector` for storage.
///
/// # Example
///
/// ```rust
/// use d_delaunay::geometry::point::PointND;
/// let point: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
/// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
/// ```
pub type PointND<const D: usize> = Point<f64, SVector<f64, D>, D>;

/// Type alias for a D-dimensional point using f32 coordinates
pub type PointF32<const D: usize> = Point<f32, SVector<f32, D>, D>;

/// For backward compatibility, create a default Point type that uses f64 coordinates
pub type PointF64<const D: usize> = Point<f64, SVector<f64, D>, D>;

impl<T, V, const D: usize> Point<T, V, D>
where
    T: Copy + PartialEq + PartialOrd + Debug + Default,
    V: VectorN<T, D>,
{
    /// Construct a new point from an array of coordinates.
    #[inline]
    #[must_use]
    pub fn new(coords: [T; D]) -> Self {
        Self {
            storage: V::from_array(coords),
            _phantom: PhantomData,
        }
    }

    /// Dimensionality of the point.
    #[inline]
    pub fn dim(&self) -> usize {
        D
    }

    /// Returns the coordinates of the point as an array.
    ///
    /// This method provides access to the coordinates of the point, returning an array of
    /// scalar type `T` with a length equal to the point's dimensionality. It creates a copy of the
    /// coordinates, so modifications to the returned array won't affect the original point.
    ///
    /// # Usage
    ///
    /// Use this method for geometric calculations, interfacing with libraries, or debugging.
    ///
    /// # Panics
    ///
    /// Panics if there's a mismatch between the storage's length and `D`, indicating a serious
    /// internal error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use d_delaunay::geometry::point::PointND;
    ///
    /// let point: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
    /// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    pub fn coordinates(&self) -> [T; D] {
        self.storage
            .as_slice()
            .try_into()
            .expect("Vector slice length mismatch")
    }

    /// Origin point with all zeros.
    #[inline]
    #[must_use]
    pub fn origin() -> Self {
        Self::new([T::default(); D])
    }
}

// Implement is_valid for floating-point coordinates only
impl<V, const D: usize> Point<f64, V, D>
where
    V: VectorN<f64, D>,
{
    /// Validates that all coordinates of the point are finite.
    ///
    /// This method checks each coordinate to ensure it is a finite floating-point value.
    /// A coordinate is considered valid if it is not NaN (Not a Number) and not infinite
    /// (positive or negative infinity).
    ///
    /// # Returns
    ///
    /// * `Ok(())` if all coordinates are finite
    /// * `Err(PointValidationError::InvalidCoordinate)` if any coordinate is NaN or infinite
    ///
    /// # Errors
    ///
    /// Returns a [`PointValidationError::InvalidCoordinate`] containing:
    /// - The index of the first invalid coordinate found
    /// - The string representation of the invalid coordinate value
    /// - The dimensionality of the point
    ///
    /// # Performance
    ///
    /// This method has O(D) time complexity where D is the dimensionality of the point,
    /// as it must check each coordinate individually.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use d_delaunay::geometry::point::PointND;
    ///
    /// // Valid point with finite coordinates
    /// let valid_point: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
    /// assert!(valid_point.is_valid().is_ok());
    ///
    /// // Invalid point with NaN coordinate
    /// let invalid_point: PointND<3> = PointND::new([1.0, f64::NAN, 3.0]);
    /// assert!(invalid_point.is_valid().is_err());
    ///
    /// // Invalid point with infinite coordinate
    /// let infinite_point: PointND<3> = PointND::new([1.0, f64::INFINITY, 3.0]);
    /// assert!(infinite_point.is_valid().is_err());
    ///
    /// // Check specific error details
    /// match invalid_point.is_valid() {
    ///     Err(error) => {
    ///         // Error contains the index and value of the invalid coordinate
    ///         println!("Invalid coordinate found: {}", error);
    ///     }
    ///     Ok(_) => unreachable!()
    /// }
    /// ```
    ///
    /// # Use in Delaunay Triangulation
    ///
    /// This validation is particularly important for Delaunay triangulation algorithms,
    /// as NaN or infinite coordinates can cause numerical instability and incorrect
    /// geometric computations. It's recommended to validate points before adding them
    /// to a triangulation.
    pub fn is_valid(&self) -> Result<(), PointValidationError> {
        for (index, &coord) in self.storage.as_slice().iter().enumerate() {
            if !coord.is_finite() {
                return Err(PointValidationError::InvalidCoordinate {
                    coordinate_index: index,
                    coordinate_value: format!("{coord:?}"),
                    dimension: D,
                });
            }
        }
        Ok(())
    }
}

// Implement is_valid for f32 coordinates
impl<V, const D: usize> Point<f32, V, D>
where
    V: VectorN<f32, D>,
{
    /// Validates that all coordinates of the point are finite.
    ///
    /// # Errors
    ///
    /// Returns a [`PointValidationError::InvalidCoordinate`] if any coordinate is NaN or infinite.
    pub fn is_valid(&self) -> Result<(), PointValidationError> {
        for (index, &coord) in self.storage.as_slice().iter().enumerate() {
            if !coord.is_finite() {
                return Err(PointValidationError::InvalidCoordinate {
                    coordinate_index: index,
                    coordinate_value: format!("{coord:?}"),
                    dimension: D,
                });
            }
        }
        Ok(())
    }
}

impl<V, const D: usize> PartialEq for Point<f64, V, D>
where
    V: VectorN<f64, D>,
{
    fn eq(&self, other: &Self) -> bool {
        let self_coords = self.storage.as_slice();
        let other_coords = other.storage.as_slice();

        // Compare coordinates one by one, treating NaN as equal to itself
        for i in 0..D {
            let a = self_coords[i];
            let b = other_coords[i];

            // If both are NaN, they are considered equal
            if a.is_nan() && b.is_nan() {
                continue;
            }

            // If one is NaN and the other is not, they are not equal
            if a.is_nan() || b.is_nan() {
                return false;
            }

            // Use normal f64 equality for non-NaN values
            if a != b {
                return false;
            }
        }

        true
    }
}

impl<V, const D: usize> Eq for Point<f64, V, D> where V: VectorN<f64, D> {}

impl<V, const D: usize> PartialOrd for Point<f64, V, D>
where
    V: VectorN<f64, D>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<V, const D: usize> Ord for Point<f64, V, D>
where
    V: VectorN<f64, D>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare coordinates using OrderedFloat for proper handling of NaN
        let self_coords = self.coordinates();
        let other_coords = other.coordinates();

        for i in 0..D {
            match OrderedFloat(self_coords[i]).cmp(&OrderedFloat(other_coords[i])) {
                std::cmp::Ordering::Equal => {}
                other_ordering => return other_ordering,
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl<V, const D: usize> Hash for Point<f64, V, D>
where
    V: VectorN<f64, D>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &coord in self.storage.as_slice() {
            OrderedFloat(coord).hash(state);
        }
    }
}

// Implement PartialEq for f32 coordinates
impl<V, const D: usize> PartialEq for Point<f32, V, D>
where
    V: VectorN<f32, D>,
{
    fn eq(&self, other: &Self) -> bool {
        let self_coords = self.storage.as_slice();
        let other_coords = other.storage.as_slice();

        // Compare coordinates one by one, treating NaN as equal to itself
        for i in 0..D {
            let a = self_coords[i];
            let b = other_coords[i];

            // If both are NaN, they are considered equal
            if a.is_nan() && b.is_nan() {
                continue;
            }

            // If one is NaN and the other is not, they are not equal
            if a.is_nan() || b.is_nan() {
                return false;
            }

            // Use normal f32 equality for non-NaN values
            if a != b {
                return false;
            }
        }

        true
    }
}

impl<V, const D: usize> Eq for Point<f32, V, D> where V: VectorN<f32, D> {}

impl<V, const D: usize> PartialOrd for Point<f32, V, D>
where
    V: VectorN<f32, D>,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<V, const D: usize> Ord for Point<f32, V, D>
where
    V: VectorN<f32, D>,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare coordinates using OrderedFloat for proper handling of NaN
        let self_coords = self.coordinates();
        let other_coords = other.coordinates();

        for i in 0..D {
            match OrderedFloat(self_coords[i]).cmp(&OrderedFloat(other_coords[i])) {
                std::cmp::Ordering::Equal => {}
                other_ordering => return other_ordering,
            }
        }
        std::cmp::Ordering::Equal
    }
}

impl<V, const D: usize> Hash for Point<f32, V, D>
where
    V: VectorN<f32, D>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &coord in self.storage.as_slice() {
            OrderedFloat(coord).hash(state);
        }
    }
}

/// Helper trait for checking finiteness of coordinates.
pub trait FiniteCheck {
    /// Returns true if the value is finite (not NaN or infinite).
    fn is_finite_generic(&self) -> bool;
}

// Macro for implementing FiniteCheck for floating-point types
macro_rules! impl_finite_check {
    ($($t:ty),*) => {
        $(
            impl FiniteCheck for $t {
                #[inline(always)]
                fn is_finite_generic(&self) -> bool {
                    self.is_finite()
                }
            }
        )*
    };
}

impl_finite_check!(f32, f64);

/// Helper trait for hashing individual coordinates for non-hashable types
/// like f32 and f64.
pub trait HashCoordinate {
    /// Hashes a single coordinate value using the provided hasher.
    ///
    /// This method provides a consistent way to hash coordinate values,
    /// including floating-point types that don't normally implement Hash.
    /// For floating-point types, this uses `OrderedFloat` to ensure consistent
    /// hashing behavior, including proper handling of NaN values.
    ///
    /// # Arguments
    ///
    /// * `state` - The hasher state to write the hash value to
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::geometry::point::HashCoordinate;
    /// use std::collections::hash_map::DefaultHasher;
    /// use std::hash::Hasher;
    ///
    /// let mut hasher = DefaultHasher::new();
    /// let value = 3.14f64;
    /// value.hash_coord(&mut hasher);
    /// let hash_value = hasher.finish();
    /// ```
    fn hash_coord<H: Hasher>(&self, state: &mut H);
}

// Macro for implementing HashCoordinate for floating-point types
macro_rules! impl_hash_coordinate {
    ($($t:ty),*) => {
        $(
            impl HashCoordinate for $t {
                #[inline(always)]
                fn hash_coord<H: Hasher>(&self, state: &mut H) {
                    OrderedFloat(*self).hash(state);
                }
            }
        )*
    };
}

impl_hash_coordinate!(f32, f64);

/// Helper trait for OrderedFloat-based equality comparison that handles NaN properly
pub trait OrderedEq {
    /// Compares two values for equality using ordered comparison semantics.
    ///
    /// This method provides a way to compare floating-point numbers that treats
    /// NaN values as equal to themselves, which is different from the default
    /// floating-point equality comparison where NaN != NaN.
    ///
    /// # Arguments
    ///
    /// * `other` - The other value to compare with
    ///
    /// # Returns
    ///
    /// Returns `true` if the values are equal according to ordered comparison,
    /// `false` otherwise.
    fn ordered_eq(&self, other: &Self) -> bool;
}

// Macro for implementing OrderedEq for floating-point types
macro_rules! impl_ordered_eq {
    ($($t:ty),*) => {
        $(
            impl OrderedEq for $t {
                #[inline(always)]
                fn ordered_eq(&self, other: &Self) -> bool {
                    OrderedFloat(*self) == OrderedFloat(*other)
                }
            }
        )*
    };
}

impl_ordered_eq!(f32, f64);

// Custom PartialEq implementation using OrderedFloat for consistent NaN handling
// impl<T, const D: usize> PartialEq for Point<T, D>
// where
//     T: Clone + Copy + Default + PartialOrd + OrderedEq,
//     [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
// {
//     fn eq(&self, other: &Self) -> bool {
//         self.coords
//             .iter()
//             .zip(other.coords.iter())
//             .all(|(a, b)| a.ordered_eq(b))
//     }
// }

// Manual implementation of Eq for Point using OrderedFloat for proper NaN handling
// impl<T, const D: usize> Eq for Point<T, D>
// where
//     T: Clone + Copy + Default + PartialOrd + OrderedEq,
//     [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
// {
// }

/// From trait implementations for Point conversions
impl<T, V, const D: usize> From<[T; D]> for Point<T, V, D>
where
    T: Clone + Copy + Debug + Default + PartialEq + PartialOrd,
    V: VectorN<T, D>,
{
    /// Create a new [Point] from an array of coordinates of type `T`.
    ///
    /// # Arguments
    ///
    /// * `coords`: An array of type `T` with a length of `D`, representing the
    ///   coordinates of the point.
    ///
    /// # Returns
    ///
    /// The function returns a new instance of the [Point] struct with the
    /// coordinates.
    ///
    /// # Example
    ///
    /// ```rust
    /// use d_delaunay::geometry::point::Point;
    /// let coords = [1.0, 2.0, 3.0];
    /// let point: Point<f64, nalgebra::SVector<f64, 3>, 3> = Point::from(coords);
    /// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    fn from(coords: [T; D]) -> Self {
        Self {
            storage: V::from_array(coords),
            _phantom: PhantomData,
        }
    }
}

/// Implement Into trait for converting Point back to coordinate arrays
impl<V, const D: usize> From<Point<f64, V, D>> for [f64; D]
where
    V: VectorN<f64, D>,
{
    #[inline]
    fn from(point: Point<f64, V, D>) -> Self {
        point.coordinates()
    }
}

/// Implement Into trait for converting Point reference to coordinate arrays
impl<V, const D: usize> From<&Point<f64, V, D>> for [f64; D]
where
    V: VectorN<f64, D>,
{
    #[inline]
    fn from(point: &Point<f64, V, D>) -> Self {
        point.coordinates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::hash_map::DefaultHasher;
    use std::collections::{HashMap, HashSet};
    use std::hash::{Hash, Hasher};

    // Helper function to get hash value for any hashable type
    fn get_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_point_new() {
        let point: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
        assert_relative_eq!(point.coordinates().as_slice(), [1.0, 2.0, 3.0].as_slice());
        assert_eq!(point.dim(), 3);
    }

    #[test]
    fn test_point_default() {
        let point: PointND<4> = PointND::default();
        assert_relative_eq!(
            point.coordinates().as_slice(),
            [0.0, 0.0, 0.0, 0.0].as_slice()
        );
        assert_eq!(point.dim(), 4);
    }

    #[test]
    fn test_point_origin() {
        let point: PointND<2> = PointND::origin();
        assert_relative_eq!(point.coordinates().as_slice(), [0.0, 0.0].as_slice());
        assert_eq!(point.dim(), 2);
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn test_point_copy_and_clone() {
        let point = PointND::new([1.0, 2.0, 3.0]);
        let point_copy = point; // Test Copy trait
        let point_clone = point.clone(); // Test Clone trait

        assert_eq!(point, point_copy);
        assert_eq!(point, point_clone);
        assert_relative_eq!(
            point.coordinates().as_slice(),
            point_copy.coordinates().as_slice()
        );
        assert_relative_eq!(
            point.coordinates().as_slice(),
            point_clone.coordinates().as_slice()
        );
    }

    #[test]
    fn test_point_f32() {
        let point: PointF32<2> = PointF32::new([1.5, 2.5]);
        assert_relative_eq!(point.coordinates().as_slice(), [1.5f32, 2.5f32].as_slice());
        assert_eq!(point.dim(), 2);
    }

    #[test]
    fn test_point_different_dimensions() {
        let point_1d: PointND<1> = PointND::new([42.0]);
        assert_relative_eq!(point_1d.coordinates().as_slice(), [42.0].as_slice());
        assert_eq!(point_1d.dim(), 1);

        let point_2d: Point2D = Point2D::new([1.0, 2.0]);
        assert_relative_eq!(point_2d.coordinates().as_slice(), [1.0, 2.0].as_slice());
        assert_eq!(point_2d.dim(), 2);

        let point_3d: Point3D = Point3D::new([1.0, 2.0, 3.0]);
        assert_relative_eq!(
            point_3d.coordinates().as_slice(),
            [1.0, 2.0, 3.0].as_slice()
        );
        assert_eq!(point_3d.dim(), 3);

        let point_5d: PointND<5> = PointND::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_relative_eq!(
            point_5d.coordinates().as_slice(),
            [1.0, 2.0, 3.0, 4.0, 5.0].as_slice()
        );
        assert_eq!(point_5d.dim(), 5);
    }

    #[test]
    fn test_point_validation_f64() {
        // Valid points
        let valid_point = PointND::new([1.0, 2.0, 3.0]);
        assert!(valid_point.is_valid().is_ok());

        let valid_negative = PointND::new([-1.0, -2.0, -3.0]);
        assert!(valid_negative.is_valid().is_ok());

        let valid_zero = PointND::new([0.0, 0.0]);
        assert!(valid_zero.is_valid().is_ok());

        // Invalid points with NaN
        let invalid_nan = PointND::new([1.0, f64::NAN, 3.0]);
        assert!(invalid_nan.is_valid().is_err());

        let invalid_all_nan = PointND::new([f64::NAN, f64::NAN]);
        assert!(invalid_all_nan.is_valid().is_err());

        // Invalid points with infinity
        let invalid_inf = PointND::new([f64::INFINITY, 2.0]);
        assert!(invalid_inf.is_valid().is_err());

        let invalid_neg_inf = PointND::new([1.0, f64::NEG_INFINITY]);
        assert!(invalid_neg_inf.is_valid().is_err());
    }

    #[test]
    fn test_point_validation_f32() {
        // Valid points
        let valid_point = PointF32::new([1.0, 2.0]);
        assert!(valid_point.is_valid().is_ok());

        // Invalid points
        let invalid_nan = PointF32::new([f32::NAN, 2.0]);
        assert!(invalid_nan.is_valid().is_err());

        let invalid_inf = PointF32::new([f32::INFINITY, 2.0]);
        assert!(invalid_inf.is_valid().is_err());
    }

    #[test]
    fn test_point_validation_error_details() {
        let invalid_point = PointND::new([1.0, f64::NAN, 3.0]);
        let result = invalid_point.is_valid();

        match result {
            Err(PointValidationError::InvalidCoordinate {
                coordinate_index,
                coordinate_value,
                dimension,
            }) => {
                assert_eq!(coordinate_index, 1);
                assert_eq!(dimension, 3);
                assert!(coordinate_value.contains("NaN"));
            }
            _ => panic!("Expected InvalidCoordinate error"),
        }
    }

    #[test]
    fn test_point_equality_normal_values() {
        let point1 = PointND::new([1.0, 2.0, 3.0]);
        let point2 = PointND::new([1.0, 2.0, 3.0]);
        let point3 = PointND::new([1.0, 2.0, 4.0]);

        assert_eq!(point1, point2);
        assert_ne!(point1, point3);
        assert_ne!(point2, point3);
    }

    #[test]
    fn test_point_equality_nan_handling() {
        // Test that NaN values are considered equal to themselves
        let point_nan1 = PointND::new([f64::NAN, 2.0, 3.0]);
        let point_nan2 = PointND::new([f64::NAN, 2.0, 3.0]);
        let point_nan3 = PointND::new([f64::NAN, f64::NAN, f64::NAN]);
        let point_nan4 = PointND::new([f64::NAN, f64::NAN, f64::NAN]);

        assert_eq!(point_nan1, point_nan2);
        assert_eq!(point_nan3, point_nan4);

        // NaN vs normal should not be equal
        let point_normal = PointND::new([1.0, 2.0, 3.0]);
        assert_ne!(point_nan1, point_normal);
    }

    #[test]
    fn test_point_equality_infinity_handling() {
        let point_inf1 = PointND::new([f64::INFINITY, 2.0]);
        let point_inf2 = PointND::new([f64::INFINITY, 2.0]);
        let point_neg_inf = PointND::new([f64::NEG_INFINITY, 2.0]);

        assert_eq!(point_inf1, point_inf2);
        assert_ne!(point_inf1, point_neg_inf);
    }

    #[test]
    fn test_point_ordering() {
        let point1 = PointND::new([1.0, 2.0, 3.0]);
        let point2 = PointND::new([1.0, 2.0, 4.0]);
        let point3 = PointND::new([1.0, 3.0, 0.0]);
        let point4 = PointND::new([2.0, 0.0, 0.0]);

        // Test lexicographic ordering
        assert!(point1 < point2); // Different in last coordinate
        assert!(point1 < point3); // Different in second coordinate
        assert!(point1 < point4); // Different in first coordinate
        assert!(point2 > point1);
        assert!(point3 > point1);
        assert!(point4 > point1);

        // Test ordering with negative values
        let neg_point1 = PointND::new([-1.0, -2.0]);
        let neg_point2 = PointND::new([-1.0, -1.0]);
        assert!(neg_point1 < neg_point2);
    }

    #[test]
    fn test_point_hash_consistency() {
        let point1 = PointND::new([1.0, 2.0, 3.0]);
        let point2 = PointND::new([1.0, 2.0, 3.0]);
        let point3 = PointND::new([1.0, 2.0, 4.0]);

        // Equal points should have equal hashes
        assert_eq!(get_hash(&point1), get_hash(&point2));
        // Different points should have different hashes (with high probability)
        assert_ne!(get_hash(&point1), get_hash(&point3));
    }

    #[test]
    fn test_point_hash_nan_consistency() {
        let point_nan1 = PointND::new([f64::NAN, 2.0]);
        let point_nan2 = PointND::new([f64::NAN, 2.0]);

        // NaN points should hash consistently
        assert_eq!(get_hash(&point_nan1), get_hash(&point_nan2));
    }

    #[test]
    fn test_point_in_hashmap() {
        let mut map: HashMap<PointND<2>, i32> = HashMap::new();

        let point1 = PointND::new([1.0, 2.0]);
        let point2 = PointND::new([3.0, 4.0]);
        let point3 = PointND::new([1.0, 2.0]); // Same as point1

        map.insert(point1, 10);
        map.insert(point2, 20);

        assert_eq!(map.get(&point3), Some(&10));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn test_point_in_hashset() {
        let mut set: HashSet<PointND<3>> = HashSet::new();

        let point1 = PointND::new([1.0, 2.0, 3.0]);
        let point2 = PointND::new([4.0, 5.0, 6.0]);
        let point3 = PointND::new([1.0, 2.0, 3.0]); // Same as point1

        set.insert(point1);
        set.insert(point2);
        set.insert(point3);

        assert_eq!(set.len(), 2); // point1 and point3 should be deduplicated
        assert!(set.contains(&point1));
        assert!(set.contains(&point2));
        assert!(set.contains(&point3));
    }

    #[test]
    fn test_point_from_array() {
        let coords = [1.0, 2.0, 3.0];
        let point: PointND<3> = PointND::from(coords);
        assert_relative_eq!(point.coordinates().as_slice(), coords.as_slice());
    }

    #[test]
    fn test_point_to_array() {
        let point = PointND::new([1.0, 2.0, 3.0]);
        let coords: [f64; 3] = point.into();
        assert_relative_eq!(coords.as_slice(), [1.0, 2.0, 3.0].as_slice());
    }

    #[test]
    fn test_point_ref_to_array() {
        let point = PointND::new([1.0, 2.0, 3.0]);
        let coords: [f64; 3] = (&point).into();
        assert_relative_eq!(coords.as_slice(), [1.0, 2.0, 3.0].as_slice());
        // Point should still be usable
        assert_relative_eq!(point.coordinates().as_slice(), [1.0, 2.0, 3.0].as_slice());
    }

    #[test]
    fn test_point_serialization() {
        let point = PointND::new([1.0, 2.0, 3.0]);
        let serialized = serde_json::to_string(&point).unwrap();
        let deserialized: PointND<3> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(point, deserialized);
    }

    #[test]
    fn test_point_debug_format() {
        let point = PointND::new([1.0, 2.0, 3.0]);
        let debug_str = format!("{point:?}");
        assert!(debug_str.contains("Point"));
        assert!(debug_str.contains("storage"));
    }

    #[test]
    fn test_point_extreme_values() {
        // Test with very large values
        let large_point = PointND::new([f64::MAX, f64::MAX / 2.0]);
        assert!(large_point.is_valid().is_ok());

        // Test with very small values
        let small_point = PointND::new([f64::MIN_POSITIVE, -f64::MIN_POSITIVE]);
        assert!(small_point.is_valid().is_ok());

        // Test with subnormal values
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let subnormal_point = PointND::new([subnormal, -subnormal]);
        assert!(subnormal_point.is_valid().is_ok());
    }

    #[test]
    fn test_point_zero_and_negative_zero() {
        let point_pos_zero = PointND::new([0.0, 0.0]);
        let point_neg_zero = PointND::new([-0.0, -0.0]);
        let point_mixed_zero = PointND::new([0.0, -0.0]);

        // IEEE 754: 0.0 == -0.0
        assert_eq!(point_pos_zero, point_neg_zero);
        assert_eq!(point_pos_zero, point_mixed_zero);
        assert_eq!(point_neg_zero, point_mixed_zero);
    }

    #[test]
    fn test_point_different_nan_patterns() {
        // Test different ways to create NaN
        let nan1 = f64::NAN;
        #[allow(clippy::zero_divided_by_zero)]
        let nan2 = 0.0f64 / 0.0f64;
        let nan3 = f64::INFINITY - f64::INFINITY;

        let point1 = PointND::new([nan1, 1.0]);
        let point2 = PointND::new([nan2, 1.0]);
        let point3 = PointND::new([nan3, 1.0]);

        // All NaN patterns should be considered equal
        assert_eq!(point1, point2);
        assert_eq!(point2, point3);
        assert_eq!(point1, point3);
    }

    #[test]
    fn test_point_high_dimensions() {
        let point_10d: PointND<10> =
            PointND::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(point_10d.dim(), 10);
        assert!(point_10d.is_valid().is_ok());

        let point_20d: PointND<20> = PointND::new([1.0; 20]);
        assert_eq!(point_20d.dim(), 20);
        assert!(point_20d.is_valid().is_ok());
    }

    #[test]
    fn test_point_memory_efficiency() {
        use std::mem;

        // Point should be the same size as its coordinate array
        assert_eq!(mem::size_of::<PointND<3>>(), mem::size_of::<[f64; 3]>());
        assert_eq!(mem::size_of::<PointF32<2>>(), mem::size_of::<[f32; 2]>());
        assert_eq!(mem::size_of::<Point2D>(), mem::size_of::<[f64; 2]>());
        assert_eq!(mem::size_of::<Point3D>(), mem::size_of::<[f64; 3]>());
    }

    #[test]
    fn test_point_send_sync() {
        fn assert_send<T: Send>(_: T) {}
        fn assert_sync<T: Sync>(_: T) {}

        let point = PointND::new([1.0, 2.0, 3.0]);
        assert_send(point);
        assert_sync(point);
    }

    #[test]
    #[allow(clippy::clone_on_copy)]
    fn test_trait_implementations() {
        let point = PointND::new([1.0, 2.0, 3.0]);

        // Test that all expected traits are implemented
        let _: Box<dyn std::fmt::Debug> = Box::new(point);
        let _: Box<dyn Send> = Box::new(point);
        let _: Box<dyn Sync> = Box::new(point);

        // Test traits that are not object-safe by using them directly
        let point2 = point.clone();
        assert_eq!(point, point2);
        assert!(point <= point2);

        // Test that point can be used in collections requiring Hash + Eq
        let mut set = std::collections::HashSet::new();
        set.insert(point);
        assert!(set.contains(&point));
    }

    #[test]
    fn test_finite_check_trait() {
        // Test FiniteCheck trait for floating point types
        assert!(1.0f32.is_finite_generic());
        assert!(1.0f64.is_finite_generic());
        assert!(!f32::NAN.is_finite_generic());
        assert!(!f64::NAN.is_finite_generic());
        assert!(!f32::INFINITY.is_finite_generic());
        assert!(!f64::INFINITY.is_finite_generic());
        assert!(!f32::NEG_INFINITY.is_finite_generic());
        assert!(!f64::NEG_INFINITY.is_finite_generic());
    }

    #[test]
    fn test_hash_coordinate_trait() {
        let mut hasher = DefaultHasher::new();

        // Test HashCoordinate trait
        1.0f32.hash_coord(&mut hasher);
        1.0f64.hash_coord(&mut hasher);
        f32::NAN.hash_coord(&mut hasher);
        f64::NAN.hash_coord(&mut hasher);
        f32::INFINITY.hash_coord(&mut hasher);
        f64::INFINITY.hash_coord(&mut hasher);

        // Should not panic and should produce some hash
        assert!(hasher.finish() > 0);
    }

    #[test]
    fn test_ordered_eq_trait() {
        // Test OrderedEq trait for floating point types
        assert!(1.0f32.ordered_eq(&1.0f32));
        assert!(1.0f64.ordered_eq(&1.0f64));
        assert!(!1.0f32.ordered_eq(&2.0f32));
        assert!(!1.0f64.ordered_eq(&2.0f64));

        // Test NaN equality (should be true with OrderedEq)
        assert!(f32::NAN.ordered_eq(&f32::NAN));
        assert!(f64::NAN.ordered_eq(&f64::NAN));

        // Test infinity values
        assert!(f32::INFINITY.ordered_eq(&f32::INFINITY));
        assert!(f64::INFINITY.ordered_eq(&f64::INFINITY));
        assert!(f32::NEG_INFINITY.ordered_eq(&f32::NEG_INFINITY));
        assert!(f64::NEG_INFINITY.ordered_eq(&f64::NEG_INFINITY));
        assert!(!f32::INFINITY.ordered_eq(&f32::NEG_INFINITY));
        assert!(!f64::INFINITY.ordered_eq(&f64::NEG_INFINITY));
    }

    #[test]
    fn test_edge_cases() {
        // Test 1D point
        let point_1d: PointND<1> = PointND::new([42.0]);
        assert_eq!(point_1d.dim(), 1);
        assert_relative_eq!(point_1d.coordinates().as_slice(), [42.0].as_slice());
        assert!(point_1d.is_valid().is_ok());

        // Test with all same values
        let point_same = PointND::new([5.0, 5.0, 5.0, 5.0]);
        assert_eq!(point_same.dim(), 4);
        assert!(point_same.is_valid().is_ok());
    }

    #[test]
    fn test_point_ordering_edge_cases() {
        use std::cmp::Ordering;

        let point1 = PointND::new([1.0, 2.0]);
        let point2 = PointND::new([1.0, 2.0]);

        // Test that equal points are not less than each other
        assert_ne!(point1.partial_cmp(&point2), Some(Ordering::Less));
        assert_ne!(point2.partial_cmp(&point1), Some(Ordering::Less));
        assert_eq!(point1.partial_cmp(&point2), Some(Ordering::Equal));
        assert!(point1 <= point2);
        assert!(point2 <= point1);
        assert!(point1 >= point2);
        assert!(point2 >= point1);
    }

    #[test]
    fn test_point_validation_first_invalid() {
        // Test that validation returns the FIRST invalid coordinate
        let multi_invalid = PointND::new([1.0, f64::NAN, f64::INFINITY, f64::NAN]);
        let result = multi_invalid.is_valid();

        if let Err(PointValidationError::InvalidCoordinate {
            coordinate_index, ..
        }) = result
        {
            assert_eq!(coordinate_index, 1); // Should be the first invalid (NaN at index 1)
        } else {
            panic!("Expected InvalidCoordinate error");
        }
    }

    #[test]
    fn test_point_error_display() {
        let invalid_point = PointND::new([1.0, f64::NAN, 3.0]);
        let result = invalid_point.is_valid();

        if let Err(error) = result {
            let error_msg = format!("{error}");
            assert!(error_msg.contains("Invalid coordinate at index 1"));
            assert!(error_msg.contains("in dimension 3"));
            assert!(error_msg.contains("NaN"));
        } else {
            panic!("Expected validation error");
        }
    }

    #[test]
    fn test_point_validation_error_clone_eq() {
        let invalid_point = PointND::new([f64::NAN, 2.0]);
        let result1 = invalid_point.is_valid();
        let result2 = invalid_point.is_valid();

        assert!(result1.is_err());
        assert!(result2.is_err());

        let error1 = result1.unwrap_err();
        let error2 = result2.unwrap_err();

        // Test Clone and PartialEq
        let error1_clone = error1.clone();
        assert_eq!(error1, error1_clone);
        assert_eq!(error1, error2);
    }
}

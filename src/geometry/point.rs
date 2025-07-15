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

// impl<T, const D: usize> Hash for Point<T, D>
// where
//     T: HashCoordinate + Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
//     [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
// {
//     #[inline]
//     fn hash<H: Hasher>(&self, state: &mut H) {
//         for &coord in &self.coords {
//             coord.hash_coord(state);
//         }
//     }
// }

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

// Unified macro for implementing OrderedEq
macro_rules! impl_ordered_eq {
    (float: $($t:ty),*) => {
        $(
            impl OrderedEq for $t {
                #[inline(always)]
                fn ordered_eq(&self, other: &Self) -> bool {
                    OrderedFloat(*self) == OrderedFloat(*other)
                }
            }
        )*
    };
    (int: $($t:ty),*) => {
        $(
            impl OrderedEq for $t {
                #[inline(always)]
                fn ordered_eq(&self, other: &Self) -> bool {
                    self == other
                }
            }
        )*
    };
}

impl_ordered_eq!(float: f32, f64);
impl_ordered_eq!(int: i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

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
impl<T, V, const D: usize> From<[T; D]> for Point<f64, V, D>
where
    T: Clone + Copy + Default + Into<f64> + PartialEq + PartialOrd,
    V: VectorN<f64, D>,
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
    /// coordinates converted to f64.
    ///
    /// # Example
    ///
    /// ```rust
    /// use d_delaunay::geometry::point::Point;
    /// let coords = [1, 2, 3];
    /// let point: Point<f64, nalgebra::SVector<f64, 3>, 3> = Point::from(coords);
    /// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    fn from(coords: [T; D]) -> Self {
        // Convert the `coords` array to `[f64; D]`
        let coords_f64: [f64; D] = coords.map(std::convert::Into::into);
        Self {
            storage: V::from_array(coords_f64),
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

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use approx::assert_relative_eq;
//     use std::collections::hash_map::DefaultHasher;
//     use std::hash::{Hash, Hasher};

// Helper function to get hash value for any hashable type
// fn get_hash<T: Hash>(value: &T) -> u64 {
//     let mut hasher = DefaultHasher::new();
//     value.hash(&mut hasher);
//     hasher.finish()
// }

// // Helper function to test basic point properties
// fn test_basic_point_properties<T, const D: usize>(
//     point: &Point<T, D>,
//     expected_coords: [T; D],
//     expected_dim: usize,
// ) where
//     T: Clone + Copy + Debug + Default + PartialEq + PartialOrd + OrderedEq,
//     [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
// {
//     assert_eq!(point.coordinates(), expected_coords);
//     assert_eq!(point.dim(), expected_dim);
// }

// // Helper function to test point equality and hash consistency
// fn test_point_equality_and_hash<T, const D: usize>(
//     point1: Point<T, D>,
//     point2: Point<T, D>,
//     should_be_equal: bool,
// ) where
//     T: HashCoordinate
//         + Clone
//         + Copy
//         + Default
//         + PartialEq
//         + PartialOrd
//         + OrderedEq
//         + std::fmt::Debug,
//     [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
// {
//     if should_be_equal {
//         assert_eq!(point1, point2);
//         assert_eq!(get_hash(&point1), get_hash(&point2));
//     } else {
//         assert_ne!(point1, point2);
//         // Note: Different points may still hash to same value (hash collisions)
//     }
// }

// #[test]
// fn point_default() {
//     let point: Point<f64, 4> = Point::default();

//     let coords = point.coordinates();
//     assert_relative_eq!(
//         coords.as_slice(),
//         [0.0, 0.0, 0.0, 0.0].as_slice(),
//         epsilon = 1e-9
//     );

//     // Human readable output for cargo test -- --nocapture
//     println!("Default: {point:?}");
// }

// #[test]
// fn point_new() {
//     let point = Point::new([1.0, 2.0, 3.0, 4.0]);

//     let coords = point.coordinates();
//     assert_relative_eq!(
//         coords.as_slice(),
//         [1.0, 2.0, 3.0, 4.0].as_slice(),
//         epsilon = 1e-9
//     );

//     // Human readable output for cargo test -- --nocapture
//     println!("Point: {point:?}");
// }

// #[test]
// fn point_copy() {
//     let point = Point::new([1.0, 2.0, 3.0, 4.0]);
//     let point_copy = point;

//     assert_eq!(point, point_copy);
//     let coords1 = point.coordinates();
//     let coords2 = point_copy.coordinates();
//     assert_relative_eq!(coords1.as_slice(), coords2.as_slice(), epsilon = 1e-9);
//     assert_eq!(point.dim(), point_copy.dim());
// }

// #[test]
// fn point_dim() {
//     let point = Point::new([1.0, 2.0, 3.0, 4.0]);

//     assert_eq!(point.dim(), 4);

//     // Human readable output for cargo test -- --nocapture
//     println!("Point: {:?} is {}-D", point, point.dim());
// }

// #[test]
// fn point_origin() {
//     let point: Point<f64, 4> = Point::origin();

//     let coords = point.coordinates();
//     assert_relative_eq!(
//         coords.as_slice(),
//         [0.0, 0.0, 0.0, 0.0].as_slice(),
//         epsilon = 1e-9
//     );

//     // Human readable output for cargo test -- --nocapture
//     println!("Origin: {:?} is {}-D", point, point.dim());
// }

// #[test]
// fn point_serialization() {
//     use serde_test::{assert_tokens, Token};

//     let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
//     assert_tokens(
//         &point,
//         &[
//             Token::Struct {
//                 name: "Point",
//                 len: 1,
//             },
//             Token::Str("coords"),
//             Token::Tuple { len: 3 },
//             Token::F64(1.0),
//             Token::F64(2.0),
//             Token::F64(3.0),
//             Token::TupleEnd,
//             Token::StructEnd,
//         ],
//     );
// }

// #[test]
// fn point_to_and_from_json() {
//     let point: Point<f64, 4> = Point::default();
//     let serialized = serde_json::to_string(&point).unwrap();

//     assert_eq!(serialized, "{\"coords\":[0.0,0.0,0.0,0.0]}");

//     let deserialized: Point<f64, 4> = serde_json::from_str(&serialized).unwrap();

//     assert_eq!(deserialized, point);

//     // Human readable output for cargo test -- --nocapture
//     println!("Serialized: {serialized:?}");
// }

// #[test]
// fn point_from_array() {
//     let coords = [1i32, 2i32, 3i32];
//     let point: Point<f64, 3> = Point::from(coords);

//     let result_coords = point.coordinates();
//     assert_relative_eq!(
//         result_coords.as_slice(),
//         [1.0, 2.0, 3.0].as_slice(),
//         epsilon = 1e-9
//     );
//     assert_eq!(point.dim(), 3);
// }

// #[test]
// fn point_from_array_f32_to_f64() {
//     let coords = [1.5f32, 2.5f32, 3.5f32, 4.5f32];
//     let point: Point<f64, 4> = Point::from(coords);

//     let result_coords = point.coordinates();
//     assert_relative_eq!(
//         result_coords.as_slice(),
//         [1.5, 2.5, 3.5, 4.5].as_slice(),
//         epsilon = 1e-9
//     );
//     assert_eq!(point.dim(), 4);
// }

// #[test]
// fn point_from_array_same_type() {
//     // Test conversion when source and target types are the same
//     let coords_f32 = [1.0f32, 2.0f32, 3.0f32];
//     let point_f32: Point<f32, 3> = Point::from(coords_f32);
//     let result_f32 = point_f32.coordinates();
//     assert_relative_eq!(
//         result_f32.as_slice(),
//         [1.0f32, 2.0f32, 3.0f32].as_slice(),
//         epsilon = 1e-9
//     );

//     let coords_i32 = [1i32, 2i32, 3i32];
//     let point_i32: Point<i32, 3> = Point::from(coords_i32);
//     assert_eq!(point_i32.coordinates(), [1i32, 2i32, 3i32]);
// }

// #[test]
// fn point_from_array_integer_to_integer() {
//     // Test conversion from i32 to i64
//     let coords_i32 = [1i32, 2i32, 3i32];
//     let point_i64: Point<i64, 3> = Point::from(coords_i32);
//     assert_eq!(point_i64.coordinates(), [1i64, 2i64, 3i64]);

//     // Test conversion from u8 to i32
//     let coords_u8 = [10u8, 20u8, 30u8];
//     let point_i32: Point<i32, 3> = Point::from(coords_u8);
//     assert_eq!(point_i32.coordinates(), [10i32, 20i32, 30i32]);

//     // Test conversion from i16 to isize
//     let coords_i16 = [100i16, 200i16];
//     let point_isize: Point<isize, 2> = Point::from(coords_i16);
//     assert_eq!(point_isize.coordinates(), [100isize, 200isize]);
// }

// #[test]
// fn point_from_array_float_to_float() {
//     // Test conversion from f32 to f32 (same type)
//     let coords_f32 = [1.5f32, 2.5f32];
//     let point_f32: Point<f32, 2> = Point::from(coords_f32);
//     let result_f32 = point_f32.coordinates();
//     assert_relative_eq!(
//         result_f32.as_slice(),
//         [1.5f32, 2.5f32].as_slice(),
//         epsilon = 1e-9
//     );

//     // Test conversion from f32 to f64 (safe upcast)
//     let coords_f32 = [1.5f32, 2.5f32];
//     let point_f64: Point<f64, 2> = Point::from(coords_f32);
//     let result_f64 = point_f64.coordinates();
//     assert_relative_eq!(
//         result_f64.as_slice(),
//         [1.5f64, 2.5f64].as_slice(),
//         epsilon = 1e-9
//     );
// }

// #[test]
// fn point_from_array_integer_to_float() {
//     // Test conversion from i32 to f64
//     let coords_i32 = [1i32, 2i32, 3i32];
//     let point_f64: Point<f64, 3> = Point::from(coords_i32);
//     let result_i32_f64 = point_f64.coordinates();
//     assert_relative_eq!(
//         result_i32_f64.as_slice(),
//         [1.0f64, 2.0f64, 3.0f64].as_slice(),
//         epsilon = 1e-9
//     );

//     // Test conversion from u8 to f64
//     let coords_u8 = [10u8, 20u8];
//     let point_f64: Point<f64, 2> = Point::from(coords_u8);
//     let result_u8_f64 = point_f64.coordinates();
//     assert_relative_eq!(
//         result_u8_f64.as_slice(),
//         [10.0f64, 20.0f64].as_slice(),
//         epsilon = 1e-9
//     );
// }

// #[test]
// fn point_hash() {
//     use std::collections::hash_map::DefaultHasher;
//     use std::hash::{Hash, Hasher};

//     let point1 = Point::new([1.0, 2.0, 3.0]);
//     let point2 = Point::new([1.0, 2.0, 3.0]);
//     let point3 = Point::new([1.0, 2.0, 4.0]);

//     let mut hasher1 = DefaultHasher::new();
//     let mut hasher2 = DefaultHasher::new();
//     let mut hasher3 = DefaultHasher::new();

//     point1.hash(&mut hasher1);
//     point2.hash(&mut hasher2);
//     point3.hash(&mut hasher3);

//     // Same points should have same hash
//     assert_eq!(hasher1.finish(), hasher2.finish());
//     // Different points should have different hash (with high probability)
//     assert_ne!(hasher1.finish(), hasher3.finish());
// }

// #[test]
// fn point_hash_in_hashmap() {
//     use std::collections::HashMap;

//     let mut map: HashMap<Point<f64, 2>, i32> = HashMap::new();

//     let point1 = Point::new([1.0, 2.0]);
//     let point2 = Point::new([3.0, 4.0]);
//     let point3 = Point::new([1.0, 2.0]); // Same as point1

//     map.insert(point1, 10);
//     map.insert(point2, 20);

//     assert_eq!(map.get(&point3), Some(&10)); // Should find point1's value
//     assert_eq!(map.len(), 2);
// }

// #[test]
// fn point_partial_eq() {
//     let point1 = Point::new([1.0, 2.0, 3.0]);
//     let point2 = Point::new([1.0, 2.0, 3.0]);
//     let point3 = Point::new([1.0, 2.0, 4.0]);

//     assert_eq!(point1, point2);
//     assert_ne!(point1, point3);
//     assert_ne!(point2, point3);
// }

// #[test]
// fn point_partial_ord() {
//     let point1 = Point::new([1.0, 2.0, 3.0]);
//     let point2 = Point::new([1.0, 2.0, 4.0]);
//     let point3 = Point::new([1.0, 3.0, 0.0]);
//     let point4 = Point::new([2.0, 0.0, 0.0]);

//     // Lexicographic ordering
//     assert!(point1 < point2); // 3.0 < 4.0 in last coordinate
//     assert!(point1 < point3); // 2.0 < 3.0 in second coordinate
//     assert!(point1 < point4); // 1.0 < 2.0 in first coordinate
//     assert!(point2 > point1);
// }

// #[test]
// fn point_from_complex_conversions() {
//     // Test conversion with mixed type arrays
//     let coords_mixed_i32 = [-100i32, 200i32, 300i32];
//     let point_f64: Point<f64, 3> = Point::from(coords_mixed_i32);
//     let mixed_coords = point_f64.coordinates();
//     assert_relative_eq!(
//         mixed_coords.as_slice(),
//         [-100.0f64, 200.0f64, 300.0f64].as_slice(),
//         epsilon = 1e-9
//     );

//     // Test with larger values
//     let coords_large = [10000i32, 20000i32];
//     let point_f64: Point<f64, 2> = Point::from(coords_large);
//     let large_coords = point_f64.coordinates();
//     assert_relative_eq!(
//         large_coords.as_slice(),
//         [10000.0f64, 20000.0f64].as_slice(),
//         epsilon = 1e-9
//     );

//     // Test with very small values
//     // When converting from f32 to f64, there can be small precision
//     // differences due to how floating point numbers are represented in
//     // memory. Use approximate comparison for these small values.
//     let coords_small_f32 = [0.000_001_f32, 0.000_002_f32];
//     let point_f64: Point<f64, 2> = Point::from(coords_small_f32);

//     // Use relative comparison with appropriate epsilon for small floating
//     // point values
//     let expected = [0.000_001_f64, 0.000_002_f64];
//     assert_relative_eq!(
//         point_f64.coordinates().as_slice(),
//         expected.as_slice(),
//         epsilon = 1e-9,
//         max_relative = 1e-7
//     );
// }

// #[test]
// fn point_1d() {
//     let point: Point<f64, 1> = Point::new([42.0]);
//     test_basic_point_properties(&point, [42.0], 1);

//     let origin: Point<f64, 1> = Point::origin();
//     test_basic_point_properties(&origin, [0.0], 1);
// }

// #[test]
// fn point_2d() {
//     let point: Point<f64, 2> = Point::new([1.0, 2.0]);
//     test_basic_point_properties(&point, [1.0, 2.0], 2);

//     let origin: Point<f64, 2> = Point::origin();
//     test_basic_point_properties(&origin, [0.0, 0.0], 2);
// }

// #[test]
// fn point_3d() {
//     let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
//     test_basic_point_properties(&point, [1.0, 2.0, 3.0], 3);

//     let origin: Point<f64, 3> = Point::origin();
//     test_basic_point_properties(&origin, [0.0, 0.0, 0.0], 3);
// }

// #[test]
// fn point_5d() {
//     let point: Point<f64, 5> = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
//     test_basic_point_properties(&point, [1.0, 2.0, 3.0, 4.0, 5.0], 5);

//     let origin: Point<f64, 5> = Point::origin();
//     test_basic_point_properties(&origin, [0.0, 0.0, 0.0, 0.0, 0.0], 5);
// }

// #[test]
// fn point_with_integers() {
//     let point: Point<i32, 3> = Point::new([1, 2, 3]);

//     assert_eq!(point.coordinates(), [1, 2, 3]);
//     assert_eq!(point.dim(), 3);

//     let origin: Point<i32, 3> = Point::origin();
//     assert_eq!(origin.coordinates(), [0, 0, 0]);
// }

// #[test]
// fn point_with_f32() {
//     let point: Point<f32, 2> = Point::new([1.5, 2.5]);

//     let coords = point.coordinates();
//     assert_relative_eq!(coords.as_slice(), [1.5, 2.5].as_slice(), epsilon = 1e-9);
//     assert_eq!(point.dim(), 2);

//     let origin: Point<f32, 2> = Point::origin();
//     let origin_coords = origin.coordinates();
//     assert_relative_eq!(
//         origin_coords.as_slice(),
//         [0.0, 0.0].as_slice(),
//         epsilon = 1e-9
//     );
// }

// #[test]
// fn point_debug_format() {
//     let point = Point::new([1.0, 2.0, 3.0]);
//     let debug_str = format!("{point:?}");

//     assert!(debug_str.contains("Point"));
//     assert!(debug_str.contains("coords"));
//     assert!(debug_str.contains("1.0"));
//     assert!(debug_str.contains("2.0"));
//     assert!(debug_str.contains("3.0"));
// }

// #[test]
// fn point_eq_trait() {
//     let point1 = Point::new([1.0, 2.0, 3.0]);
//     let point2 = Point::new([1.0, 2.0, 3.0]);
//     let point3 = Point::new([1.0, 2.0, 4.0]);

//     // Test Eq trait (transitivity, reflexivity, symmetry)
//     assert_eq!(point1, point1); // reflexive
//     assert_eq!(point1, point2); // symmetric
//     assert_eq!(point2, point1); // symmetric
//     assert_ne!(point1, point3);
//     assert_ne!(point3, point1);
// }

// #[test]
// fn point_comprehensive_serialization() {
//     // Test with different types and dimensions
//     let point_3d = Point::new([1.0, 2.0, 3.0]);
//     let serialized_3d = serde_json::to_string(&point_3d).unwrap();
//     let deserialized_3d: Point<f64, 3> = serde_json::from_str(&serialized_3d).unwrap();
//     assert_eq!(point_3d, deserialized_3d);

//     let point_2d = Point::new([10.5, -5.3]);
//     let serialized_2d = serde_json::to_string(&point_2d).unwrap();
//     let deserialized_2d: Point<f64, 2> = serde_json::from_str(&serialized_2d).unwrap();
//     assert_eq!(point_2d, deserialized_2d);

//     let point_1d = Point::new([42.0]);
//     let serialized_1d = serde_json::to_string(&point_1d).unwrap();
//     let deserialized_1d: Point<f64, 1> = serde_json::from_str(&serialized_1d).unwrap();
//     assert_eq!(point_1d, deserialized_1d);
// }

// #[test]
// fn point_negative_coordinates() {
//     let point = Point::new([-1.0, -2.0, -3.0]);

//     assert_relative_eq!(
//         point.coordinates().as_slice(),
//         [-1.0, -2.0, -3.0].as_slice(),
//         epsilon = 1e-9
//     );
//     assert_eq!(point.dim(), 3);

//     // Test with mixed positive/negative
//     let mixed_point = Point::new([1.0, -2.0, 3.0, -4.0]);
//     assert_relative_eq!(
//         mixed_point.coordinates().as_slice(),
//         [1.0, -2.0, 3.0, -4.0].as_slice(),
//         epsilon = 1e-9
//     );
// }

// #[test]
// fn point_zero_coordinates() {
//     let zero_point = Point::new([0.0, 0.0, 0.0]);
//     let origin: Point<f64, 3> = Point::origin();

//     assert_eq!(zero_point, origin);
//     assert_relative_eq!(
//         zero_point.coordinates().as_slice(),
//         [0.0, 0.0, 0.0].as_slice(),
//         epsilon = 1e-9
//     );
// }

// #[test]
// fn point_large_coordinates() {
//     let large_point = Point::new([1e6, 2e6, 3e6]);

//     let coords = large_point.coordinates();
//     assert_relative_eq!(
//         coords.as_slice(),
//         [1_000_000.0, 2_000_000.0, 3_000_000.0].as_slice(),
//         epsilon = 1e-9
//     );
//     assert_eq!(large_point.dim(), 3);
// }

// #[test]
// fn point_small_coordinates() {
//     let small_point = Point::new([1e-6, 2e-6, 3e-6]);

//     let coords = small_point.coordinates();
//     assert_relative_eq!(
//         coords.as_slice(),
//         [0.000_001, 0.000_002, 0.000_003].as_slice(),
//         epsilon = 1e-9
//     );
//     assert_eq!(small_point.dim(), 3);
// }

// #[test]
// fn point_ordering_edge_cases() {
//     use std::cmp::Ordering;

//     let point1 = Point::new([1.0, 2.0]);
//     let point2 = Point::new([1.0, 2.0]);

//     // Test that equal points are not less than each other
//     assert_ne!(point1.partial_cmp(&point2), Some(Ordering::Less));
//     assert_ne!(point2.partial_cmp(&point1), Some(Ordering::Less));
//     assert!(point1 <= point2);
//     assert!(point2 <= point1);
//     assert!(point1 >= point2);
//     assert!(point2 >= point1);
// }

// #[test]
// fn point_from_different_integer_types() {
//     // Test conversion from different integer types
//     let u8_coords: [u8; 3] = [1, 2, 3];
//     let point_from_u8: Point<f64, 3> = Point::from(u8_coords);
//     assert_relative_eq!(
//         point_from_u8.coordinates().as_slice(),
//         [1.0, 2.0, 3.0].as_slice(),
//         epsilon = 1e-9
//     );

//     let i16_coords: [i16; 2] = [-1, 32767];
//     let point_from_i16: Point<f64, 2> = Point::from(i16_coords);
//     assert_relative_eq!(
//         point_from_i16.coordinates().as_slice(),
//         [-1.0, 32767.0].as_slice()
//     );
// }

// #[test]
// fn point_hash_f32() {
//     use std::collections::HashMap;

//     let mut map: HashMap<Point<f32, 2>, i32> = HashMap::new();

//     let point1 = Point::new([1.5f32, 2.5f32]);
//     let point2 = Point::new([3.5f32, 4.5f32]);
//     let point3 = Point::new([1.5f32, 2.5f32]); // Same as point1

//     map.insert(point1, 10);
//     map.insert(point2, 20);

//     assert_eq!(map.get(&point3), Some(&10)); // Should find point1's value
//     assert_eq!(map.len(), 2);
// }

// #[test]
// fn point_hash_integers() {
//     use std::collections::HashMap;

//     // Test with i32
//     let mut map_i32: HashMap<Point<i32, 3>, &str> = HashMap::new();
//     let point_i32_1 = Point::new([1, 2, 3]);
//     let point_i32_2 = Point::new([4, 5, 6]);
//     let point_i32_3 = Point::new([1, 2, 3]); // Same as point_i32_1

//     map_i32.insert(point_i32_1, "first");
//     map_i32.insert(point_i32_2, "second");

//     assert_eq!(map_i32.get(&point_i32_3), Some(&"first"));
//     assert_eq!(map_i32.len(), 2);

//     // Test with u64
//     let mut map_u64: HashMap<Point<u64, 2>, bool> = HashMap::new();
//     let point_u64_1 = Point::new([100u64, 200u64]);
//     let point_u64_2 = Point::new([300u64, 400u64]);
//     let point_u64_3 = Point::new([100u64, 200u64]); // Same as point_u64_1

//     map_u64.insert(point_u64_1, true);
//     map_u64.insert(point_u64_2, false);

//     assert_eq!(map_u64.get(&point_u64_3), Some(&true));
//     assert_eq!(map_u64.len(), 2);
// }

// #[test]
// fn point_eq_different_types() {
//     // Test Eq for f64
//     let point_f64_1 = Point::new([1.0, 2.0]);
//     let point_f64_2 = Point::new([1.0, 2.0]);
//     let point_f64_3 = Point::new([1.0, 2.1]);

//     assert_eq!(point_f64_1, point_f64_2);
//     assert_ne!(point_f64_1, point_f64_3);

//     // Test Eq for f32
//     let point_f32_1 = Point::new([1.5f32, 2.5f32]);
//     let point_f32_2 = Point::new([1.5f32, 2.5f32]);
//     let point_f32_3 = Point::new([1.5f32, 2.6f32]);

//     assert_eq!(point_f32_1, point_f32_2);
//     assert_ne!(point_f32_1, point_f32_3);

//     // Test Eq for i32
//     let point_i32_1 = Point::new([10, 20]);
//     let point_i32_2 = Point::new([10, 20]);
//     let point_i32_3 = Point::new([10, 21]);

//     assert_eq!(point_i32_1, point_i32_2);
//     assert_ne!(point_i32_1, point_i32_3);
// }

// #[test]
// fn point_hash_consistency_floating_point() {
//     // Test that OrderedFloat provides consistent hashing for NaN-free floats
//     let point1 = Point::new([1.0, 2.0, 3.5]);
//     let point2 = Point::new([1.0, 2.0, 3.5]);
//     test_point_equality_and_hash(point1, point2, true);

//     // Test with f32
//     let point_f32_1 = Point::new([1.5f32, 2.5f32]);
//     let point_f32_2 = Point::new([1.5f32, 2.5f32]);
//     test_point_equality_and_hash(point_f32_1, point_f32_2, true);
// }

// #[test]
// fn point_hash_consistency_integers() {
//     // Test integer hashing consistency
//     let point_i32_1 = Point::new([42, -17, 100]);
//     let point_i32_2 = Point::new([42, -17, 100]);
//     test_point_equality_and_hash(point_i32_1, point_i32_2, true);

//     // Test with u64
//     let point_u64_1 = Point::new([1000u64, 2000u64]);
//     let point_u64_2 = Point::new([1000u64, 2000u64]);
//     test_point_equality_and_hash(point_u64_1, point_u64_2, true);
// }

// #[test]
// fn point_hash_all_primitives() {
//     use std::collections::hash_map::DefaultHasher;
//     use std::hash::{Hash, Hasher};

//     // Function to get hash value for any hashable type
//     fn get_hash<T: Hash>(value: &T) -> u64 {
//         let mut hasher = DefaultHasher::new();
//         value.hash(&mut hasher);
//         hasher.finish()
//     }

//     // Test all primitive integer types
//     let point_i8: Point<i8, 2> = Point::new([1, 2]);
//     let point_i16: Point<i16, 2> = Point::new([1, 2]);
//     let point_i32: Point<i32, 2> = Point::new([1, 2]);
//     let point_i64: Point<i64, 2> = Point::new([1, 2]);
//     let point_u8: Point<u8, 2> = Point::new([1, 2]);
//     let point_u16: Point<u16, 2> = Point::new([1, 2]);
//     let point_u32: Point<u32, 2> = Point::new([1, 2]);
//     let point_u64: Point<u64, 2> = Point::new([1, 2]);
//     let point_usize: Point<usize, 2> = Point::new([1, 2]);
//     let point_isize: Point<isize, 2> = Point::new([1, 2]);

//     // Get hash for each type
//     let _hash_i8 = get_hash(&point_i8);
//     let _hash_i16 = get_hash(&point_i16);
//     let _hash_i32 = get_hash(&point_i32);
//     let _hash_i64 = get_hash(&point_i64);
//     let _hash_u8 = get_hash(&point_u8);
//     let _hash_u16 = get_hash(&point_u16);
//     let _hash_u32 = get_hash(&point_u32);
//     let _hash_u64 = get_hash(&point_u64);
//     let _hash_usize = get_hash(&point_usize);
//     let _hash_isize = get_hash(&point_isize);

//     // Verify that equal points of the same type hash to the same value
//     let point_i32_a: Point<i32, 2> = Point::new([1, 2]);
//     let point_i32_b: Point<i32, 2> = Point::new([1, 2]);
//     assert_eq!(get_hash(&point_i32_a), get_hash(&point_i32_b));

//     // Test points with different values hash differently
//     let point_i32_c: Point<i32, 2> = Point::new([2, 3]);
//     assert_ne!(get_hash(&point_i32_a), get_hash(&point_i32_c));
// }

// #[test]
// fn point_implicit_conversion_to_coordinates() {
//     let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);

//     // Test implicit conversion from owned point
//     let coords_owned: [f64; 3] = point.into();
//     assert_relative_eq!(coords_owned.as_slice(), [1.0, 2.0, 3.0].as_slice());

//     // Create a new point for reference test
//     let point_ref: Point<f64, 3> = Point::new([4.0, 5.0, 6.0]);

//     // Test implicit conversion from point reference
//     let coords_ref: [f64; 3] = (&point_ref).into();
//     assert_relative_eq!(coords_ref.as_slice(), [4.0, 5.0, 6.0].as_slice());

//     // Verify the original point is still available after reference conversion
//     assert_relative_eq!(
//         point_ref.coordinates().as_slice(),
//         [4.0, 5.0, 6.0].as_slice()
//     );
// }

// #[test]
// fn point_is_valid_f64() {
//     // Test valid f64 points
//     let valid_point = Point::new([1.0, 2.0, 3.0]);
//     assert!(valid_point.is_valid().is_ok());

//     let valid_negative = Point::new([-1.0, -2.0, -3.0]);
//     assert!(valid_negative.is_valid().is_ok());

//     let valid_zero = Point::new([0.0, 0.0, 0.0]);
//     assert!(valid_zero.is_valid().is_ok());

//     let valid_mixed = Point::new([1.0, -2.5, 0.0, 42.7]);
//     assert!(valid_mixed.is_valid().is_ok());

//     // Test invalid f64 points with NaN
//     let invalid_nan_single = Point::new([1.0, f64::NAN, 3.0]);
//     assert!(invalid_nan_single.is_valid().is_err());

//     let invalid_nan_all = Point::new([f64::NAN, f64::NAN, f64::NAN]);
//     assert!(invalid_nan_all.is_valid().is_err());

//     let invalid_nan_first = Point::new([f64::NAN, 2.0, 3.0]);
//     assert!(invalid_nan_first.is_valid().is_err());

//     let invalid_nan_last = Point::new([1.0, 2.0, f64::NAN]);
//     assert!(invalid_nan_last.is_valid().is_err());

//     // Test invalid f64 points with infinity
//     let invalid_pos_inf = Point::new([1.0, f64::INFINITY, 3.0]);
//     assert!(invalid_pos_inf.is_valid().is_err());

//     let invalid_neg_inf = Point::new([1.0, f64::NEG_INFINITY, 3.0]);
//     assert!(invalid_neg_inf.is_valid().is_err());

//     let invalid_both_inf = Point::new([f64::INFINITY, f64::NEG_INFINITY]);
//     assert!(invalid_both_inf.is_valid().is_err());

//     // Test mixed invalid cases
//     let invalid_nan_and_inf = Point::new([f64::NAN, f64::INFINITY, 1.0]);
//     assert!(invalid_nan_and_inf.is_valid().is_err());
// }

// #[test]
// fn point_is_valid_f32() {
//     // Test valid f32 points
//     let valid_point = Point::new([1.0f32, 2.0f32, 3.0f32]);
//     assert!(valid_point.is_valid().is_ok());

//     let valid_negative = Point::new([-1.5f32, -2.5f32]);
//     assert!(valid_negative.is_valid().is_ok());

//     let valid_zero = Point::new([0.0f32]);
//     assert!(valid_zero.is_valid().is_ok());

//     // Test invalid f32 points with NaN
//     let invalid_nan = Point::new([1.0f32, f32::NAN]);
//     assert!(invalid_nan.is_valid().is_err());

//     let invalid_all_nan = Point::new([f32::NAN, f32::NAN, f32::NAN, f32::NAN]);
//     assert!(invalid_all_nan.is_valid().is_err());

//     // Test invalid f32 points with infinity
//     let invalid_pos_inf = Point::new([f32::INFINITY, 2.0f32]);
//     assert!(invalid_pos_inf.is_valid().is_err());

//     let invalid_neg_inf = Point::new([1.0f32, f32::NEG_INFINITY]);
//     assert!(invalid_neg_inf.is_valid().is_err());

//     // Test edge cases with very small and large values (but finite)
//     let valid_small = Point::new([f32::MIN_POSITIVE, -f32::MIN_POSITIVE]);
//     assert!(valid_small.is_valid().is_ok());

//     let valid_large = Point::new([f32::MAX, -f32::MAX]);
//     assert!(valid_large.is_valid().is_ok());
// }

// #[test]
// fn point_is_valid_integers() {
//     // All integer types should always be valid (no NaN or infinity)
//     let valid_i32 = Point::new([1i32, 2i32, 3i32]);
//     assert!(valid_i32.is_valid().is_ok());

//     let valid_negative_i32 = Point::new([-1i32, -2i32, -3i32]);
//     assert!(valid_negative_i32.is_valid().is_ok());

//     let valid_zero_i32 = Point::new([0i32, 0i32]);
//     assert!(valid_zero_i32.is_valid().is_ok());

//     let valid_u64 = Point::new([u64::MAX, u64::MIN, 42u64]);
//     assert!(valid_u64.is_valid().is_ok());

//     let valid_i8 = Point::new([i8::MAX, i8::MIN, 0i8, -1i8]);
//     assert!(valid_i8.is_valid().is_ok());

//     let valid_isize = Point::new([isize::MAX, isize::MIN]);
//     assert!(valid_isize.is_valid().is_ok());

//     // Test with various integer types
//     let valid_u8 = Point::new([255u8, 0u8, 128u8]);
//     assert!(valid_u8.is_valid().is_ok());

//     let valid_i16 = Point::new([32767i16, -32768i16, 0i16]);
//     assert!(valid_i16.is_valid().is_ok());

//     let valid_u32 = Point::new([u32::MAX, 0u32, 42u32]);
//     assert!(valid_u32.is_valid().is_ok());
// }

// #[test]
// fn point_is_valid_different_dimensions() {
//     // Test 1D points
//     let valid_1d_f64 = Point::new([42.0]);
//     assert!(valid_1d_f64.is_valid().is_ok());

//     let invalid_1d_nan = Point::new([f64::NAN]);
//     assert!(invalid_1d_nan.is_valid().is_err());

//     let valid_1d_int = Point::new([42i32]);
//     assert!(valid_1d_int.is_valid().is_ok());

//     // Test 2D points
//     let valid_2d = Point::new([1.0, 2.0]);
//     assert!(valid_2d.is_valid().is_ok());

//     let invalid_2d = Point::new([1.0, f64::INFINITY]);
//     assert!(invalid_2d.is_valid().is_err());

//     // Test higher dimensional points
//     let valid_5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
//     assert!(valid_5d.is_valid().is_ok());

//     let invalid_5d = Point::new([1.0, 2.0, f64::NAN, 4.0, 5.0]);
//     assert!(invalid_5d.is_valid().is_err());

//     // Test 10D point
//     let valid_10d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
//     assert!(valid_10d.is_valid().is_ok());

//     let invalid_10d = Point::new([
//         1.0,
//         2.0,
//         3.0,
//         4.0,
//         5.0,
//         f64::NEG_INFINITY,
//         7.0,
//         8.0,
//         9.0,
//         10.0,
//     ]);
//     assert!(invalid_10d.is_valid().is_err());
// }

// #[test]
// fn point_is_valid_edge_cases() {
//     // Test with very small finite values
//     let tiny_valid = Point::new([f64::MIN_POSITIVE, -f64::MIN_POSITIVE, 0.0]);
//     assert!(tiny_valid.is_valid().is_ok());

//     // Test with very large finite values
//     let large_valid = Point::new([f64::MAX, -f64::MAX]);
//     assert!(large_valid.is_valid().is_ok());

//     // Test subnormal numbers (should be valid)
//     let subnormal = f64::MIN_POSITIVE / 2.0;
//     let subnormal_point = Point::new([subnormal, -subnormal]);
//     assert!(subnormal_point.is_valid().is_ok());

//     // Test zero and negative zero
//     let zero_point = Point::new([0.0, -0.0]);
//     assert!(zero_point.is_valid().is_ok());

//     // Mixed valid and invalid in same point should be invalid
//     let mixed_invalid = Point::new([1.0, 2.0, 3.0, f64::NAN, 5.0]);
//     assert!(mixed_invalid.is_valid().is_err());

//     // All coordinates must be valid for point to be valid
//     let one_invalid = Point::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, f64::INFINITY]);
//     assert!(one_invalid.is_valid().is_err());
// }

// #[test]
// fn point_nan_hash_consistency() {
//     use std::collections::hash_map::DefaultHasher;
//     use std::hash::{Hash, Hasher};

//     // Test that OrderedFloat provides consistent hashing for NaN values
//     // Note: Equality comparison for NaN still follows IEEE standard (NaN != NaN)
//     // but hashing uses OrderedFloat which treats all NaN values as equivalent

//     let point_nan1 = Point::new([f64::NAN, 2.0]);
//     let point_nan2 = Point::new([f64::NAN, 2.0]);

//     let mut hasher1 = DefaultHasher::new();
//     let mut hasher2 = DefaultHasher::new();

//     point_nan1.hash(&mut hasher1);
//     point_nan2.hash(&mut hasher2);

//     // NaN points with same structure should hash to same value
//     assert_eq!(hasher1.finish(), hasher2.finish());

//     // Test with f32 NaN
//     let point_f32_nan1 = Point::new([f32::NAN, 1.0f32]);
//     let point_f32_nan2 = Point::new([f32::NAN, 1.0f32]);

//     let mut hasher_f32_1 = DefaultHasher::new();
//     let mut hasher_f32_2 = DefaultHasher::new();

//     point_f32_nan1.hash(&mut hasher_f32_1);
//     point_f32_nan2.hash(&mut hasher_f32_2);

//     assert_eq!(hasher_f32_1.finish(), hasher_f32_2.finish());
// }

// #[test]
// fn point_infinity_hash_consistency() {
//     use std::collections::hash_map::DefaultHasher;
//     use std::hash::{Hash, Hasher};

//     // Test that OrderedFloat provides consistent hashing for infinity values
//     let point_pos_inf1 = Point::new([f64::INFINITY, 2.0]);
//     let point_pos_inf2 = Point::new([f64::INFINITY, 2.0]);

//     let mut hasher1 = DefaultHasher::new();
//     let mut hasher2 = DefaultHasher::new();

//     point_pos_inf1.hash(&mut hasher1);
//     point_pos_inf2.hash(&mut hasher2);

//     // Same infinity points should hash to same value
//     assert_eq!(hasher1.finish(), hasher2.finish());

//     // Test negative infinity
//     let point_neg_inf1 = Point::new([f64::NEG_INFINITY, 2.0]);
//     let point_neg_inf2 = Point::new([f64::NEG_INFINITY, 2.0]);

//     let mut hasher_neg1 = DefaultHasher::new();
//     let mut hasher_neg2 = DefaultHasher::new();

//     point_neg_inf1.hash(&mut hasher_neg1);
//     point_neg_inf2.hash(&mut hasher_neg2);

//     assert_eq!(hasher_neg1.finish(), hasher_neg2.finish());

//     // Positive and negative infinity should hash differently
//     assert_ne!(hasher1.finish(), hasher_neg1.finish());
// }

// #[test]
// fn point_nan_infinity_hash_consistency() {
//     use std::collections::HashMap;

//     // Test that points with NaN can be used as HashMap keys
//     let mut map: HashMap<Point<f64, 2>, i32> = HashMap::new();

//     let point_nan1 = Point::new([f64::NAN, 2.0]);
//     let point_nan2 = Point::new([f64::NAN, 2.0]); // Should be equal to point_nan1
//     let point_inf = Point::new([f64::INFINITY, 2.0]);

//     map.insert(point_nan1, 100);
//     map.insert(point_inf, 200);

//     // Should be able to retrieve using equivalent NaN point
//     assert_eq!(map.get(&point_nan2), Some(&100));
//     assert_eq!(map.len(), 2);

//     // Test with f32
//     let mut map_f32: HashMap<Point<f32, 1>, i32> = HashMap::new();

//     let point_f32_nan1 = Point::new([f32::NAN]);
//     let point_f32_nan2 = Point::new([f32::NAN]);

//     map_f32.insert(point_f32_nan1, 300);
//     assert_eq!(map_f32.get(&point_f32_nan2), Some(&300));
// }

// #[test]
// fn point_nan_equality_comparison() {
//     // Test that NaN == NaN using our OrderedEq implementation
//     // This is different from IEEE 754 standard where NaN != NaN

//     // f64 NaN comparisons
//     let point_nan1 = Point::new([f64::NAN, 2.0, 3.0]);
//     let point_nan2 = Point::new([f64::NAN, 2.0, 3.0]);
//     let point_nan3 = Point::new([f64::NAN, f64::NAN, f64::NAN]);
//     let point_nan4 = Point::new([f64::NAN, f64::NAN, f64::NAN]);

//     // Points with NaN should be equal when all coordinates match
//     assert_eq!(point_nan1, point_nan2);
//     assert_eq!(point_nan3, point_nan4);

//     // Points with different NaN positions should not be equal
//     let point_nan_diff1 = Point::new([f64::NAN, 2.0, 3.0]);
//     let point_nan_diff2 = Point::new([1.0, f64::NAN, 3.0]);
//     assert_ne!(point_nan_diff1, point_nan_diff2);

//     // f32 NaN comparisons
//     let point_f32_nan1 = Point::new([f32::NAN, 1.5f32]);
//     let point_f32_nan2 = Point::new([f32::NAN, 1.5f32]);
//     assert_eq!(point_f32_nan1, point_f32_nan2);

//     // Mixed NaN and normal values
//     let point_mixed1 = Point::new([1.0, f64::NAN, 3.0, 4.0]);
//     let point_mixed2 = Point::new([1.0, f64::NAN, 3.0, 4.0]);
//     let point_mixed3 = Point::new([1.0, f64::NAN, 3.0, 5.0]); // Different last coordinate

//     assert_eq!(point_mixed1, point_mixed2);
//     assert_ne!(point_mixed1, point_mixed3);
// }

// #[test]
// fn point_nan_vs_normal_comparison() {
//     // Test that NaN points are not equal to points with normal values

//     let point_normal = Point::new([1.0, 2.0, 3.0]);
//     let point_nan = Point::new([f64::NAN, 2.0, 3.0]);
//     let point_nan_all = Point::new([f64::NAN, f64::NAN, f64::NAN]);

//     // NaN points should not equal normal points
//     assert_ne!(point_normal, point_nan);
//     assert_ne!(point_normal, point_nan_all);
//     assert_ne!(point_nan, point_normal);
//     assert_ne!(point_nan_all, point_normal);

//     // Test with f32
//     let point_f32_normal = Point::new([1.0f32, 2.0f32]);
//     let point_f32_nan = Point::new([f32::NAN, 2.0f32]);

//     assert_ne!(point_f32_normal, point_f32_nan);
//     assert_ne!(point_f32_nan, point_f32_normal);
// }

// #[test]
// fn point_infinity_comparison() {
//     // Test comparison behavior with infinity values

//     // Positive infinity comparisons
//     let point_pos_inf1 = Point::new([f64::INFINITY, 2.0]);
//     let point_pos_inf2 = Point::new([f64::INFINITY, 2.0]);
//     assert_eq!(point_pos_inf1, point_pos_inf2);

//     // Negative infinity comparisons
//     let point_neg_inf1 = Point::new([f64::NEG_INFINITY, 2.0]);
//     let point_neg_inf2 = Point::new([f64::NEG_INFINITY, 2.0]);
//     assert_eq!(point_neg_inf1, point_neg_inf2);

//     // Positive vs negative infinity should not be equal
//     assert_ne!(point_pos_inf1, point_neg_inf1);

//     // Infinity vs normal values should not be equal
//     let point_normal = Point::new([1.0, 2.0]);
//     assert_ne!(point_pos_inf1, point_normal);
//     assert_ne!(point_neg_inf1, point_normal);

//     // Test with f32
//     let point_f32_pos_inf1 = Point::new([f32::INFINITY]);
//     let point_f32_pos_inf2 = Point::new([f32::INFINITY]);
//     let point_f32_neg_inf = Point::new([f32::NEG_INFINITY]);

//     assert_eq!(point_f32_pos_inf1, point_f32_pos_inf2);
//     assert_ne!(point_f32_pos_inf1, point_f32_neg_inf);
// }

// #[test]
// fn point_nan_infinity_mixed_comparison() {
//     // Test comparisons with mixed NaN and infinity values

//     let point_nan_inf1 = Point::new([f64::NAN, f64::INFINITY, 1.0]);
//     let point_nan_inf2 = Point::new([f64::NAN, f64::INFINITY, 1.0]);
//     let point_nan_inf3 = Point::new([f64::NAN, f64::NEG_INFINITY, 1.0]);

//     // Same NaN/infinity pattern should be equal
//     assert_eq!(point_nan_inf1, point_nan_inf2);

//     // Different infinity signs should not be equal
//     assert_ne!(point_nan_inf1, point_nan_inf3);

//     // Test various combinations
//     let point_all_special = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
//     let point_all_special_copy =
//         Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
//     let point_all_special_diff =
//         Point::new([f64::NAN, f64::NEG_INFINITY, f64::INFINITY, f64::NAN]);

//     assert_eq!(point_all_special, point_all_special_copy);
//     assert_ne!(point_all_special, point_all_special_diff);
// }

// #[test]
// fn point_nan_reflexivity() {
//     // Test that NaN points are equal to themselves (reflexivity)

//     let point_nan = Point::new([f64::NAN, f64::NAN, f64::NAN]);
//     assert_eq!(point_nan, point_nan);

//     let point_mixed = Point::new([1.0, f64::NAN, 3.0, f64::INFINITY]);
//     assert_eq!(point_mixed, point_mixed);

//     // Test with f32
//     let point_f32_nan = Point::new([f32::NAN, f32::NAN]);
//     assert_eq!(point_f32_nan, point_f32_nan);
// }

// #[test]
// fn point_nan_symmetry() {
//     // Test that NaN equality is symmetric (if a == b, then b == a)

//     let point_a = Point::new([f64::NAN, 2.0, f64::INFINITY]);
//     let point_b = Point::new([f64::NAN, 2.0, f64::INFINITY]);

//     assert_eq!(point_a, point_b);
//     assert_eq!(point_b, point_a); // Should be symmetric

//     // Test with f32
//     let point_f32_a = Point::new([f32::NAN, 1.0f32, f32::NEG_INFINITY]);
//     let point_f32_b = Point::new([f32::NAN, 1.0f32, f32::NEG_INFINITY]);

//     assert_eq!(point_f32_a, point_f32_b);
//     assert_eq!(point_f32_b, point_f32_a);
// }

// #[test]
// fn point_nan_transitivity() {
//     // Test that NaN equality is transitive (if a == b and b == c, then a == c)

//     let point_a = Point::new([f64::NAN, 2.0, 3.0]);
//     let point_b = Point::new([f64::NAN, 2.0, 3.0]);
//     let point_c = Point::new([f64::NAN, 2.0, 3.0]);

//     assert_eq!(point_a, point_b);
//     assert_eq!(point_b, point_c);
//     assert_eq!(point_a, point_c); // Should be transitive

//     // Test with complex special values
//     let point_complex_a = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
//     let point_complex_b = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
//     let point_complex_c = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);

//     assert_eq!(point_complex_a, point_complex_b);
//     assert_eq!(point_complex_b, point_complex_c);
//     assert_eq!(point_complex_a, point_complex_c);
// }

// #[test]
// fn point_nan_different_bit_patterns() {
//     // Test that different NaN bit patterns are considered equal
//     // Note: Rust's f64::NAN is a specific bit pattern, but there are many possible NaN values

//     // Create different NaN values
//     let nan1 = f64::NAN;
//     #[allow(clippy::zero_divided_by_zero)]
//     let nan2 = 0.0f64 / 0.0f64; // Another way to create NaN
//     let nan3 = f64::INFINITY - f64::INFINITY; // Yet another way

//     // Verify they are all NaN
//     assert!(nan1.is_nan());
//     assert!(nan2.is_nan());
//     assert!(nan3.is_nan());

//     // Points with different NaN bit patterns should be equal
//     let point1 = Point::new([nan1, 1.0]);
//     let point2 = Point::new([nan2, 1.0]);
//     let point3 = Point::new([nan3, 1.0]);

//     assert_eq!(point1, point2);
//     assert_eq!(point2, point3);
//     assert_eq!(point1, point3);

//     // Test with f32 as well
//     let f32_nan1 = f32::NAN;
//     #[allow(clippy::zero_divided_by_zero)]
//     let f32_nan2 = 0.0f32 / 0.0f32;

//     let point_f32_1 = Point::new([f32_nan1]);
//     let point_f32_2 = Point::new([f32_nan2]);

//     assert_eq!(point_f32_1, point_f32_2);
// }

// #[test]
// fn point_nan_in_different_dimensions() {
//     // Test NaN behavior across different dimensionalities

//     // 1D
//     let point_1d_a = Point::new([f64::NAN]);
//     let point_1d_b = Point::new([f64::NAN]);
//     assert_eq!(point_1d_a, point_1d_b);

//     // 2D
//     let point_2d_a = Point::new([f64::NAN, f64::NAN]);
//     let point_2d_b = Point::new([f64::NAN, f64::NAN]);
//     assert_eq!(point_2d_a, point_2d_b);

//     // 3D
//     let point_3d_a = Point::new([f64::NAN, 1.0, f64::NAN]);
//     let point_3d_b = Point::new([f64::NAN, 1.0, f64::NAN]);
//     assert_eq!(point_3d_a, point_3d_b);

//     // 5D
//     let point_5d_a = Point::new([f64::NAN, 1.0, f64::NAN, f64::INFINITY, f64::NAN]);
//     let point_5d_b = Point::new([f64::NAN, 1.0, f64::NAN, f64::INFINITY, f64::NAN]);
//     assert_eq!(point_5d_a, point_5d_b);

//     // 10D with mixed special values
//     let point_10d_a = Point::new([
//         f64::NAN,
//         1.0,
//         f64::NAN,
//         f64::INFINITY,
//         f64::NEG_INFINITY,
//         0.0,
//         -0.0,
//         f64::NAN,
//         42.0,
//         f64::NAN,
//     ]);
//     let point_10d_b = Point::new([
//         f64::NAN,
//         1.0,
//         f64::NAN,
//         f64::INFINITY,
//         f64::NEG_INFINITY,
//         0.0,
//         -0.0,
//         f64::NAN,
//         42.0,
//         f64::NAN,
//     ]);
//     assert_eq!(point_10d_a, point_10d_b);
// }

// #[test]
// fn point_nan_zero_comparison() {
//     // Test comparison between NaN, positive zero, and negative zero

//     let point_nan = Point::new([f64::NAN, f64::NAN]);
//     let point_pos_zero = Point::new([0.0, 0.0]);
//     let point_neg_zero = Point::new([-0.0, -0.0]);
//     let point_mixed_zero = Point::new([0.0, -0.0]);

//     // NaN should not equal any zero
//     assert_ne!(point_nan, point_pos_zero);
//     assert_ne!(point_nan, point_neg_zero);
//     assert_ne!(point_nan, point_mixed_zero);

//     // Different zeros should be equal (0.0 == -0.0 in IEEE 754)
//     assert_eq!(point_pos_zero, point_neg_zero);
//     assert_eq!(point_pos_zero, point_mixed_zero);
//     assert_eq!(point_neg_zero, point_mixed_zero);

//     // Test with f32
//     let point_f32_nan = Point::new([f32::NAN]);
//     let point_f32_zero = Point::new([0.0f32]);
//     let point_f32_neg_zero = Point::new([-0.0f32]);

//     assert_ne!(point_f32_nan, point_f32_zero);
//     assert_ne!(point_f32_nan, point_f32_neg_zero);
//     assert_eq!(point_f32_zero, point_f32_neg_zero);
// }

// #[test]
// fn finite_check_trait_coverage() {
//     // Test FiniteCheck trait implementations for all numeric types

//     // Test floating point types
//     assert!(1.0f32.is_finite_generic());
//     assert!(1.0f64.is_finite_generic());
//     assert!(!f32::NAN.is_finite_generic());
//     assert!(!f64::NAN.is_finite_generic());
//     assert!(!f32::INFINITY.is_finite_generic());
//     assert!(!f64::INFINITY.is_finite_generic());
//     assert!(!f32::NEG_INFINITY.is_finite_generic());
//     assert!(!f64::NEG_INFINITY.is_finite_generic());

//     // Test edge cases for floating point
//     assert!(f32::MAX.is_finite_generic());
//     assert!(f64::MAX.is_finite_generic());
//     assert!(f32::MIN.is_finite_generic());
//     assert!(f64::MIN.is_finite_generic());
//     assert!(f32::MIN_POSITIVE.is_finite_generic());
//     assert!(f64::MIN_POSITIVE.is_finite_generic());
//     assert!(0.0f32.is_finite_generic());
//     assert!((-0.0f64).is_finite_generic());

//     // Test all integer types (should always be finite)
//     assert!(42i8.is_finite_generic());
//     assert!(42i16.is_finite_generic());
//     assert!(42i32.is_finite_generic());
//     assert!(42i64.is_finite_generic());
//     assert!(42i128.is_finite_generic());
//     assert!(42isize.is_finite_generic());
//     assert!(42u8.is_finite_generic());
//     assert!(42u16.is_finite_generic());
//     assert!(42u32.is_finite_generic());
//     assert!(42u64.is_finite_generic());
//     assert!(42u128.is_finite_generic());
//     assert!(42usize.is_finite_generic());

//     // Test integer edge cases
//     assert!(i8::MAX.is_finite_generic());
//     assert!(i8::MIN.is_finite_generic());
//     assert!(u8::MAX.is_finite_generic());
//     assert!(u8::MIN.is_finite_generic());
//     assert!(i32::MAX.is_finite_generic());
//     assert!(i32::MIN.is_finite_generic());
//     assert!(i64::MAX.is_finite_generic());
//     assert!(i64::MIN.is_finite_generic());
//     assert!(isize::MAX.is_finite_generic());
//     assert!(isize::MIN.is_finite_generic());
//     assert!(usize::MAX.is_finite_generic());
//     assert!(usize::MIN.is_finite_generic());
// }

// #[test]
// fn hash_coordinate_trait_coverage() {
//     use std::collections::hash_map::DefaultHasher;
//     use std::hash::Hasher;

//     // Helper function to get hash for a coordinate
//     fn hash_coord<T: HashCoordinate>(value: &T) -> u64 {
//         let mut hasher = DefaultHasher::new();
//         value.hash_coord(&mut hasher);
//         hasher.finish()
//     }

//     // Test floating point types
//     let hash_f32 = hash_coord(&std::f32::consts::PI);
//     let hash_f64 = hash_coord(&std::f64::consts::PI);
//     assert!(hash_f32 > 0);
//     assert!(hash_f64 > 0);

//     // Test that same values hash to same result
//     assert_eq!(hash_coord(&1.0f32), hash_coord(&1.0f32));
//     assert_eq!(hash_coord(&1.0f64), hash_coord(&1.0f64));

//     // Test NaN hashing consistency
//     assert_eq!(hash_coord(&f32::NAN), hash_coord(&f32::NAN));
//     assert_eq!(hash_coord(&f64::NAN), hash_coord(&f64::NAN));

//     // Test infinity hashing
//     assert_eq!(hash_coord(&f32::INFINITY), hash_coord(&f32::INFINITY));
//     assert_eq!(hash_coord(&f64::INFINITY), hash_coord(&f64::INFINITY));
//     assert_eq!(
//         hash_coord(&f32::NEG_INFINITY),
//         hash_coord(&f32::NEG_INFINITY)
//     );
//     assert_eq!(
//         hash_coord(&f64::NEG_INFINITY),
//         hash_coord(&f64::NEG_INFINITY)
//     );

//     // Test that different special values hash differently
//     assert_ne!(hash_coord(&f64::INFINITY), hash_coord(&f64::NEG_INFINITY));

//     // Test all integer types
//     assert_eq!(hash_coord(&42i8), hash_coord(&42i8));
//     assert_eq!(hash_coord(&42i16), hash_coord(&42i16));
//     assert_eq!(hash_coord(&42i32), hash_coord(&42i32));
//     assert_eq!(hash_coord(&42i64), hash_coord(&42i64));
//     assert_eq!(hash_coord(&42i128), hash_coord(&42i128));
//     assert_eq!(hash_coord(&42isize), hash_coord(&42isize));
//     assert_eq!(hash_coord(&42u8), hash_coord(&42u8));
//     assert_eq!(hash_coord(&42u16), hash_coord(&42u16));
//     assert_eq!(hash_coord(&42u32), hash_coord(&42u32));
//     assert_eq!(hash_coord(&42u64), hash_coord(&42u64));
//     assert_eq!(hash_coord(&42u128), hash_coord(&42u128));
//     assert_eq!(hash_coord(&42usize), hash_coord(&42usize));

//     // Test integer edge cases
//     assert_eq!(hash_coord(&i8::MAX), hash_coord(&i8::MAX));
//     assert_eq!(hash_coord(&i8::MIN), hash_coord(&i8::MIN));
//     assert_eq!(hash_coord(&u64::MAX), hash_coord(&u64::MAX));
//     assert_eq!(hash_coord(&u64::MIN), hash_coord(&u64::MIN));
// }

// #[test]
// fn ordered_eq_trait_coverage() {
//     // Test OrderedEq trait implementations

//     // Test floating point types with normal values
//     assert!(1.0f32.ordered_eq(&1.0f32));
//     assert!(1.0f64.ordered_eq(&1.0f64));
//     assert!(!1.0f32.ordered_eq(&2.0f32));
//     assert!(!1.0f64.ordered_eq(&2.0f64));

//     // Test NaN equality (should be true with OrderedEq)
//     assert!(f32::NAN.ordered_eq(&f32::NAN));
//     assert!(f64::NAN.ordered_eq(&f64::NAN));

//     // Test infinity values
//     assert!(f32::INFINITY.ordered_eq(&f32::INFINITY));
//     assert!(f64::INFINITY.ordered_eq(&f64::INFINITY));
//     assert!(f32::NEG_INFINITY.ordered_eq(&f32::NEG_INFINITY));
//     assert!(f64::NEG_INFINITY.ordered_eq(&f64::NEG_INFINITY));
//     assert!(!f32::INFINITY.ordered_eq(&f32::NEG_INFINITY));
//     assert!(!f64::INFINITY.ordered_eq(&f64::NEG_INFINITY));

//     // Test zero comparisons
//     assert!(0.0f32.ordered_eq(&(-0.0f32)));
//     assert!(0.0f64.ordered_eq(&(-0.0f64)));

//     // Test integer types
//     assert!(42i8.ordered_eq(&42i8));
//     assert!(42i16.ordered_eq(&42i16));
//     assert!(42i32.ordered_eq(&42i32));
//     assert!(42i64.ordered_eq(&42i64));
//     assert!(42i128.ordered_eq(&42i128));
//     assert!(42isize.ordered_eq(&42isize));
//     assert!(42u8.ordered_eq(&42u8));
//     assert!(42u16.ordered_eq(&42u16));
//     assert!(42u32.ordered_eq(&42u32));
//     assert!(42u64.ordered_eq(&42u64));
//     assert!(42u128.ordered_eq(&42u128));
//     assert!(42usize.ordered_eq(&42usize));

//     // Test integer inequality
//     assert!(!1i32.ordered_eq(&2i32));
//     assert!(!100u64.ordered_eq(&200u64));

//     // Test edge cases for integers
//     assert!(i8::MAX.ordered_eq(&i8::MAX));
//     assert!(i8::MIN.ordered_eq(&i8::MIN));
//     assert!(!i8::MAX.ordered_eq(&i8::MIN));
//     assert!(u32::MAX.ordered_eq(&u32::MAX));
//     assert!(u32::MIN.ordered_eq(&u32::MIN));
// }

// #[test]
// #[allow(clippy::cast_precision_loss)]
// fn point_extreme_dimensions() {
//     // Test with high dimensional points (limited by serde trait implementations)

//     // Test 20D point
//     let coords_20d = [1.0; 20];
//     let point_20d = Point::new(coords_20d);
//     assert_eq!(point_20d.dim(), 20);
//     assert_relative_eq!(point_20d.coordinates().as_slice(), coords_20d.as_slice());
//     assert!(point_20d.is_valid().is_ok());

//     // Test 25D point
//     let coords_25d = [2.5; 25];
//     let point_25d = Point::new(coords_25d);
//     assert_eq!(point_25d.dim(), 25);
//     assert_relative_eq!(point_25d.coordinates().as_slice(), coords_25d.as_slice());
//     assert!(point_25d.is_valid().is_ok());

//     // Test 32D point with mixed values (max supported by std traits)
//     let mut coords_32d = [0.0; 32];
//     for (i, coord) in coords_32d.iter_mut().enumerate() {
//         *coord = i as f64;
//     }
//     let point_32d = Point::new(coords_32d);
//     assert_eq!(point_32d.dim(), 32);
//     assert_relative_eq!(point_32d.coordinates().as_slice(), coords_32d.as_slice());
//     assert!(point_32d.is_valid().is_ok());

//     // Test high dimensional point with NaN
//     let mut coords_with_nan = [1.0; 25];
//     coords_with_nan[12] = f64::NAN;
//     let point_with_nan = Point::new(coords_with_nan);
//     assert!(point_with_nan.is_valid().is_err());

//     // Test equality for high dimensional points
//     let point_20d_copy = Point::new([1.0; 20]);
//     assert_eq!(point_20d, point_20d_copy);

//     // Test with 30D points
//     let coords_30d_a = [std::f64::consts::PI; 30];
//     let coords_30d_b = [std::f64::consts::PI; 30];
//     let point_30d_a = Point::new(coords_30d_a);
//     let point_30d_b = Point::new(coords_30d_b);
//     assert_eq!(point_30d_a, point_30d_b);
//     assert!(point_30d_a.is_valid().is_ok());
// }

// #[test]
// fn point_boundary_numeric_values() {
//     // Test with extreme numeric values

//     // Test with very large f64 values
//     let large_point = Point::new([f64::MAX, f64::MAX / 2.0, 1e308]);
//     assert!(large_point.is_valid().is_ok());
//     assert_relative_eq!(large_point.coordinates()[0], f64::MAX);

//     // Test with very small f64 values
//     let small_point = Point::new([f64::MIN, f64::MIN_POSITIVE, 1e-308]);
//     assert!(small_point.is_valid().is_ok());

//     // Test with subnormal numbers
//     let subnormal = f64::MIN_POSITIVE / 2.0;
//     let subnormal_point = Point::new([subnormal, -subnormal, 0.0]);
//     assert!(subnormal_point.is_valid().is_ok());

//     // Test with integer extremes
//     let extreme_int_point = Point::new([i64::MAX, i64::MIN, 0i64]);
//     assert!(extreme_int_point.is_valid().is_ok());
//     assert_eq!(extreme_int_point.coordinates(), [i64::MAX, i64::MIN, 0i64]);

//     // Test with unsigned integer extremes
//     let extreme_uint_point = Point::new([u64::MAX, u64::MIN, 42u64]);
//     assert!(extreme_uint_point.is_valid().is_ok());
//     assert_eq!(
//         extreme_uint_point.coordinates(),
//         [u64::MAX, u64::MIN, 42u64]
//     );

//     // Test f32 extremes
//     let extreme_f32_point = Point::new([f32::MAX, f32::MIN, f32::MIN_POSITIVE]);
//     assert!(extreme_f32_point.is_valid().is_ok());
// }

// #[test]
// #[allow(clippy::redundant_clone)]
// fn point_clone_and_copy_semantics() {
//     // Test that Point correctly implements Clone and Copy

//     let original = Point::new([1.0, 2.0, 3.0]);

//     // Test explicit cloning
//     let cloned = original.clone();
//     assert_relative_eq!(
//         original.coordinates().as_slice(),
//         cloned.coordinates().as_slice()
//     );

//     // Test copy semantics (should work implicitly)
//     let copied = original; // This should copy, not move
//     assert_eq!(original, copied);

//     // Original should still be accessible after copy
//     assert_eq!(original.dim(), 3);
//     assert_eq!(copied.dim(), 3);

//     // Test with different types
//     let int_point = Point::new([1i32, 2i32]);
//     let int_copied = int_point;
//     assert_eq!(int_point, int_copied);

//     // Test with f32
//     let f32_point = Point::new([1.5f32, 2.5f32, 3.5f32, 4.5f32]);
//     let f32_copied = f32_point;
//     assert_eq!(f32_point, f32_copied);
// }

// #[test]
// fn point_partial_ord_comprehensive() {
//     use std::cmp::Ordering;

//     // Test lexicographic ordering in detail
//     let point_a = Point::new([1.0, 2.0, 3.0]);
//     let point_b = Point::new([1.0, 2.0, 4.0]); // Greater in last coordinate
//     let point_c = Point::new([1.0, 3.0, 0.0]); // Greater in second coordinate
//     let point_d = Point::new([2.0, 0.0, 0.0]); // Greater in first coordinate

//     // Test all comparison operators
//     assert!(point_a < point_b);
//     assert!(point_b > point_a);
//     assert!(point_a <= point_b);
//     assert!(point_b >= point_a);

//     assert!(point_a < point_c);
//     assert!(point_a < point_d);
//     assert!(point_c < point_d);

//     // Test partial_cmp directly
//     assert_eq!(point_a.partial_cmp(&point_b), Some(Ordering::Less));
//     assert_eq!(point_b.partial_cmp(&point_a), Some(Ordering::Greater));
//     assert_eq!(point_a.partial_cmp(&point_a), Some(Ordering::Equal));

//     // Test with negative numbers
//     let neg_point_a = Point::new([-1.0, -2.0]);
//     let neg_point_b = Point::new([-1.0, -1.0]);
//     assert!(neg_point_a < neg_point_b); // -2.0 < -1.0

//     // Test with mixed positive/negative
//     let mixed_a = Point::new([-1.0, 2.0]);
//     let mixed_b = Point::new([1.0, -2.0]);
//     assert!(mixed_a < mixed_b); // -1.0 < 1.0

//     // Test with zeros
//     let zero_a = Point::new([0.0, 0.0]);
//     let zero_b = Point::new([0.0, 0.0]);
//     assert_eq!(zero_a.partial_cmp(&zero_b), Some(Ordering::Equal));

//     // Test with special float values (where defined)
//     let inf_point = Point::new([f64::INFINITY]);
//     let normal_point = Point::new([1.0]);
//     // Note: PartialOrd with NaN/Infinity may have special behavior
//     assert!(normal_point < inf_point);
// }

// #[test]
// fn point_memory_layout_and_size() {
//     use std::mem;

//     // Test that Point has the expected memory layout
//     // Point should be the same size as its coordinate array

//     assert_eq!(mem::size_of::<Point<f64, 3>>(), mem::size_of::<[f64; 3]>());
//     assert_eq!(mem::size_of::<Point<f32, 4>>(), mem::size_of::<[f32; 4]>());
//     assert_eq!(mem::size_of::<Point<i32, 2>>(), mem::size_of::<[i32; 2]>());
//     assert_eq!(mem::size_of::<Point<u64, 5>>(), mem::size_of::<[u64; 5]>());

//     // Test alignment
//     assert_eq!(
//         mem::align_of::<Point<f64, 3>>(),
//         mem::align_of::<[f64; 3]>()
//     );
//     assert_eq!(
//         mem::align_of::<Point<i32, 4>>(),
//         mem::align_of::<[i32; 4]>()
//     );

//     // Test with different dimensions
//     assert_eq!(mem::size_of::<Point<f64, 1>>(), 8); // 1 * 8 bytes
//     assert_eq!(mem::size_of::<Point<f64, 2>>(), 16); // 2 * 8 bytes
//     assert_eq!(mem::size_of::<Point<f64, 10>>(), 80); // 10 * 8 bytes

//     assert_eq!(mem::size_of::<Point<f32, 1>>(), 4); // 1 * 4 bytes
//     assert_eq!(mem::size_of::<Point<f32, 2>>(), 8); // 2 * 4 bytes

//     assert_eq!(mem::size_of::<Point<i32, 1>>(), 4); // 1 * 4 bytes
//     assert_eq!(mem::size_of::<Point<i32, 3>>(), 12); // 3 * 4 bytes
// }

// #[test]
// fn point_zero_dimensional() {
//     // Test 0-dimensional points (edge case)
//     let point_0d: Point<f64, 0> = Point::new([]);
//     assert_eq!(point_0d.dim(), 0);
//     assert_relative_eq!(
//         point_0d.coordinates().as_slice(),
//         ([] as [f64; 0]).as_slice()
//     );
//     assert!(point_0d.is_valid().is_ok());

//     // Test equality for 0D points
//     let point_0d_2: Point<f64, 0> = Point::new([]);
//     assert_eq!(point_0d, point_0d_2);

//     // Test hashing for 0D points
//     let hash_0d = get_hash(&point_0d);
//     let hash_0d_2 = get_hash(&point_0d_2);
//     assert_eq!(hash_0d, hash_0d_2);

//     // Test with different types
//     let point_0d_int: Point<i32, 0> = Point::new([]);
//     assert_eq!(point_0d_int.dim(), 0);
//     assert!(point_0d_int.is_valid().is_ok());

//     // Test origin for 0D
//     let origin_0d: Point<f64, 0> = Point::origin();
//     assert_eq!(origin_0d, point_0d);
// }

// #[test]
// fn point_serialization_edge_cases() {
//     // Test serialization with special floating point values

//     // Test with NaN
//     let point_with_nan = Point::new([f64::NAN, 1.0, 2.0]);
//     let serialized_nan = serde_json::to_string(&point_with_nan).unwrap();
//     // NaN serializes as null in JSON
//     assert!(serialized_nan.contains("null"));

//     // Test with infinity
//     let point_with_inf = Point::new([f64::INFINITY, 1.0]);
//     let serialized_inf = serde_json::to_string(&point_with_inf).unwrap();
//     // Infinity serializes as null in JSON
//     assert!(serialized_inf.contains("null"));

//     // Test with negative infinity
//     let point_with_neg_inf = Point::new([f64::NEG_INFINITY, 1.0]);
//     let serialized_neg_inf = serde_json::to_string(&point_with_neg_inf).unwrap();
//     assert!(serialized_neg_inf.contains("null"));

//     // Test with very large numbers
//     let point_large = Point::new([1e100, -1e100, 0.0]);
//     let serialized_large = serde_json::to_string(&point_large).unwrap();
//     let deserialized_large: Point<f64, 3> = serde_json::from_str(&serialized_large).unwrap();
//     assert_eq!(point_large, deserialized_large);

//     // Test with very small numbers
//     let point_small = Point::new([1e-100, -1e-100, 0.0]);
//     let serialized_small = serde_json::to_string(&point_small).unwrap();
//     let deserialized_small: Point<f64, 3> = serde_json::from_str(&serialized_small).unwrap();
//     assert_eq!(point_small, deserialized_small);
// }

// #[test]
// fn point_conversion_edge_cases() {
//     // Test edge cases in type conversions

//     // Test conversion with potential precision loss (should still work)
//     let precise_coords = [1.000_000_000_000_001_f64, 2.000_000_000_000_002_f64];
//     let point_precise: Point<f64, 2> = Point::from(precise_coords);
//     assert_relative_eq!(
//         point_precise.coordinates().as_slice(),
//         precise_coords.as_slice()
//     );

//     // Test conversion from array reference
//     let coords_ref = &[1.0f32, 2.0f32, 3.0f32];
//     let point_from_ref: Point<f64, 3> = Point::from(*coords_ref);
//     assert_relative_eq!(
//         point_from_ref.coordinates().as_slice(),
//         [1.0f64, 2.0f64, 3.0f64].as_slice()
//     );

//     // Test conversion to array with different methods
//     let point = Point::new([1.0, 2.0, 3.0]);

//     // Using Into trait
//     let coords_into: [f64; 3] = point.into();
//     assert_relative_eq!(coords_into.as_slice(), [1.0, 2.0, 3.0].as_slice());

//     // Using From trait with reference
//     let point_ref = Point::new([4.0, 5.0]);
//     let coords_from_ref: [f64; 2] = (&point_ref).into();
//     assert_relative_eq!(coords_from_ref.as_slice(), [4.0, 5.0].as_slice());

//     // Verify original point is still usable after reference conversion
//     assert_relative_eq!(point_ref.coordinates().as_slice(), [4.0, 5.0].as_slice());
// }

// #[test]
// fn point_hash_distribution_basic() {
//     use std::collections::HashSet;

//     // Test that different points generally produce different hashes
//     // (This is a probabilistic test, not a guarantee)

//     let mut hashes = HashSet::new();

//     // Generate a variety of points and collect their hashes
//     for i in 0..100 {
//         let point = Point::new([f64::from(i), f64::from(i * 2)]);
//         let hash = get_hash(&point);
//         hashes.insert(hash);
//     }

//     // We should have close to 100 unique hashes (allowing for some collisions)
//     assert!(
//         hashes.len() > 90,
//         "Hash distribution seems poor: {} unique hashes out of 100",
//         hashes.len()
//     );

//     // Test with negative values
//     for i in -50..50 {
//         let point = Point::new([f64::from(i), f64::from(i * 3), f64::from(i * 5)]);
//         let hash = get_hash(&point);
//         hashes.insert(hash);
//     }

//     // Should have even more unique hashes now
//     assert!(
//         hashes.len() > 140,
//         "Hash distribution with negatives: {} unique hashes",
//         hashes.len()
//     );
// }

// #[test]
// fn point_validation_error_details() {
//     // Test PointValidationError with specific error details

//     // Test invalid coordinate at specific index
//     let invalid_point = Point::new([1.0, f64::NAN, 3.0]);
//     let result = invalid_point.is_valid();
//     assert!(result.is_err());

//     if let Err(PointValidationError::InvalidCoordinate {
//         coordinate_index,
//         coordinate_value,
//         dimension,
//     }) = result
//     {
//         assert_eq!(coordinate_index, 1);
//         assert_eq!(dimension, 3);
//         assert!(coordinate_value.contains("NaN"));
//     } else {
//         panic!("Expected InvalidCoordinate error");
//     }

//     // Test with infinity at different positions
//     let inf_point = Point::new([f64::INFINITY, 2.0, 3.0, 4.0]);
//     let result = inf_point.is_valid();
//     if let Err(PointValidationError::InvalidCoordinate {
//         coordinate_index,
//         coordinate_value,
//         dimension,
//     }) = result
//     {
//         assert_eq!(coordinate_index, 0);
//         assert_eq!(dimension, 4);
//         assert!(coordinate_value.contains("inf"));
//     } else {
//         panic!("Expected InvalidCoordinate error");
//     }

//     // Test with negative infinity at last position
//     let neg_inf_point = Point::new([1.0, 2.0, f64::NEG_INFINITY]);
//     let result = neg_inf_point.is_valid();
//     if let Err(PointValidationError::InvalidCoordinate {
//         coordinate_index,
//         coordinate_value,
//         dimension,
//     }) = result
//     {
//         assert_eq!(coordinate_index, 2);
//         assert_eq!(dimension, 3);
//         assert!(coordinate_value.contains("inf"));
//     }

//     // Test f32 validation errors
//     let invalid_f32_point = Point::new([1.0f32, f32::NAN, 3.0f32]);
//     let result = invalid_f32_point.is_valid();
//     if let Err(PointValidationError::InvalidCoordinate {
//         coordinate_index,
//         coordinate_value,
//         dimension,
//     }) = result
//     {
//         assert_eq!(coordinate_index, 1);
//         assert_eq!(dimension, 3);
//         assert!(coordinate_value.contains("NaN"));
//     }
// }

// #[test]
// fn point_validation_error_display() {
//     // Test error message formatting
//     let invalid_point = Point::new([1.0, f64::NAN, 3.0]);
//     let result = invalid_point.is_valid();

//     if let Err(error) = result {
//         let error_msg = format!("{error}");
//         assert!(error_msg.contains("Invalid coordinate at index 1"));
//         assert!(error_msg.contains("in dimension 3"));
//         assert!(error_msg.contains("NaN"));
//     } else {
//         panic!("Expected validation error");
//     }

//     // Test with infinity
//     let inf_point = Point::new([f64::INFINITY]);
//     let result = inf_point.is_valid();

//     if let Err(error) = result {
//         let error_msg = format!("{error}");
//         assert!(error_msg.contains("Invalid coordinate at index 0"));
//         assert!(error_msg.contains("in dimension 1"));
//         assert!(error_msg.contains("inf"));
//     }
// }

// #[test]
// fn point_validation_error_clone_and_eq() {
//     // Test that PointValidationError can be cloned and compared
//     let invalid_point = Point::new([f64::NAN, 2.0]);
//     let result1 = invalid_point.is_valid();
//     let result2 = invalid_point.is_valid();

//     assert!(result1.is_err());
//     assert!(result2.is_err());

//     let error1 = result1.unwrap_err();
//     let error2 = result2.unwrap_err();

//     // Test Clone
//     let error1_clone = error1.clone();
//     assert_eq!(error1, error1_clone);

//     // Test PartialEq
//     assert_eq!(error1, error2);

//     // Test Debug
//     let debug_output = format!("{error1:?}");
//     assert!(debug_output.contains("InvalidCoordinate"));
//     assert!(debug_output.contains("coordinate_index"));
//     assert!(debug_output.contains("dimension"));
// }

// #[test]
// fn point_validation_all_coordinate_types() {
//     // Test validation with different coordinate types

//     // All integer types should always be valid
//     let int_types_valid = [
//         Point::new([1i8, 2i8]).is_valid().is_ok(),
//         Point::new([1i16, 2i16]).is_valid().is_ok(),
//         Point::new([1i32, 2i32]).is_valid().is_ok(),
//         Point::new([1i64, 2i64]).is_valid().is_ok(),
//         Point::new([1i128, 2i128]).is_valid().is_ok(),
//         Point::new([1isize, 2isize]).is_valid().is_ok(),
//         Point::new([1u8, 2u8]).is_valid().is_ok(),
//         Point::new([1u16, 2u16]).is_valid().is_ok(),
//         Point::new([1u32, 2u32]).is_valid().is_ok(),
//         Point::new([1u64, 2u64]).is_valid().is_ok(),
//         Point::new([1u128, 2u128]).is_valid().is_ok(),
//         Point::new([1usize, 2usize]).is_valid().is_ok(),
//     ];

//     for valid in int_types_valid {
//         assert!(valid, "Integer type should always be valid");
//     }

//     // Floating point types can be invalid
//     assert!(Point::new([1.0f32, 2.0f32]).is_valid().is_ok());
//     assert!(Point::new([1.0f64, 2.0f64]).is_valid().is_ok());
//     assert!(Point::new([f32::NAN, 2.0f32]).is_valid().is_err());
//     assert!(Point::new([f64::NAN, 2.0f64]).is_valid().is_err());
// }

// #[test]
// fn point_validation_first_invalid_coordinate() {
//     // Test that validation returns the FIRST invalid coordinate found
//     let multi_invalid = Point::new([1.0, f64::NAN, f64::INFINITY, f64::NAN]);
//     let result = multi_invalid.is_valid();

//     if let Err(PointValidationError::InvalidCoordinate {
//         coordinate_index, ..
//     }) = result
//     {
//         // Should return the first invalid coordinate (index 1, not 2 or 3)
//         assert_eq!(coordinate_index, 1);
//     } else {
//         panic!("Expected InvalidCoordinate error");
//     }

//     // Test with invalid at index 0
//     let first_invalid = Point::new([f64::INFINITY, f64::NAN, 3.0]);
//     let result = first_invalid.is_valid();

//     if let Err(PointValidationError::InvalidCoordinate {
//         coordinate_index, ..
//     }) = result
//     {
//         assert_eq!(coordinate_index, 0);
//     }
// }

// #[test]
// #[allow(clippy::redundant_clone)]
// fn point_trait_completeness() {
//     // Helper functions for compile-time trait checks
//     fn assert_send<T: Send>(_: T) {}
//     fn assert_sync<T: Sync>(_: T) {}

//     // Test that Point implements all expected traits

//     let point = Point::new([1.0, 2.0, 3.0]);

//     // Test Debug trait
//     let debug_output = format!("{point:?}");
//     assert!(!debug_output.is_empty());
//     assert!(debug_output.contains("Point"));

//     // Test Default trait
//     let default_point: Point<f64, 3> = Point::default();
//     assert_relative_eq!(
//         default_point.coordinates().as_slice(),
//         [0.0, 0.0, 0.0].as_slice()
//     );

//     // Test PartialOrd trait (ordering)
//     let point_smaller = Point::new([1.0, 2.0, 2.9]);
//     assert!(point_smaller < point);

//     // Test that Send and Sync are implemented (compile-time check)
//     assert_send(point);
//     assert_sync(point);

//     // Test Clone and Copy
//     #[allow(clippy::redundant_clone)]
//     let cloned = point.clone();
//     let copied = point;

//     // Verify copy worked by using the copied value
//     assert_eq!(copied.dim(), cloned.dim());

//     // Test that point can be used in collections requiring Hash + Eq
//     let mut set = std::collections::HashSet::new();
//     set.insert(point);
//     assert!(set.contains(&point));
// }
// }

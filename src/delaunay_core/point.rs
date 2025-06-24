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
//! - Points with NaN values can be used as HashMap keys
//! - All NaN bit patterns are considered equal
//!
//! If you need standard IEEE 754 equality semantics, compare the coordinates
//! directly instead of using Point equality.

use ordered_float::OrderedFloat;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialOrd, Serialize)]
/// The [Point] struct represents a point in a D-dimensional space, where the
/// coordinates are of type `T`.
///
/// # Properties:
///
/// * `coords`: `coords` is a private property of the [Point]. It is an array of
///   type `T` with a length of `D`. The type `T` is a generic type parameter,
///   which means it can be any type. The length `D` is a constant unsigned
///   integer known at compile time.
///
/// Points are intended to be immutable once created, so the `coords` field is
/// private to prevent modification after instantiation.
pub struct Point<T, const D: usize>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The coordinates of the point.
    coords: [T; D],
}

impl<T, const D: usize> Point<T, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The function `new` creates a new instance of a [Point] with the given
    /// coordinates.
    ///
    /// # Arguments:
    ///
    /// * `coords`: The `coords` parameter is an array of type `T` with a
    ///   length of `D`.
    ///
    /// # Returns:
    ///
    /// The `new` function returns an instance of the [Point].
    ///
    /// # Example:
    ///
    /// ```rust
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn new(coords: [T; D]) -> Self {
        Self { coords }
    }

    /// The `dim` function returns the dimensionality of the [Point].
    ///
    /// # Returns:
    ///
    /// The `dim` function returns the value of `D`, which the number of
    /// coordinates.
    ///
    /// # Example:
    ///
    /// ```rust
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(point.dim(), 4);
    /// ```
    #[must_use]
    #[inline]
    pub fn dim(&self) -> usize {
        D
    }

    /// Returns a copy of the coordinates of the point.
    ///
    /// # Returns:
    ///
    /// The `coordinates` function returns a copy of the coordinates array.
    ///
    /// # Example:
    ///
    /// ```rust
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn coordinates(&self) -> [T; D] {
        self.coords
    }

    /// The `origin` function returns the origin [Point].
    ///
    /// # Returns:
    ///
    /// The `origin()` function returns a D-dimensional origin point
    /// in Cartesian coordinates.
    ///
    /// # Example:
    ///
    /// ```rust
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point: Point<f64, 4> = Point::origin();
    /// assert_eq!(point.coordinates(), [0.0, 0.0, 0.0, 0.0]);
    /// ```
    #[must_use]
    pub fn origin() -> Self
    where
        T: num_traits::Zero + Copy,
    {
        Self::new([T::zero(); D])
    }

    /// Check if all coordinates are finite (no NaN or infinite values).
    /// The Rust type system guarantees that the number of coordinates
    /// matches the dimensionality `D`.
    ///
    /// # Returns:
    ///
    /// The `is_valid()` function returns a boolean indicating whether the
    /// point is valid, meaning all coordinates are finite.
    ///
    /// # Example:
    ///
    /// ```rust
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0]);
    /// assert!(point.is_valid());
    /// let invalid_point = Point::new([1.0, f64::NAN, 3.0]);
    /// assert!(!invalid_point.is_valid());
    /// ```
    pub fn is_valid(&self) -> bool
    where
        T: FiniteCheck + Copy,
    {
        // Verify all coordinates are finite
        self.coords.iter().all(|&coord| coord.is_finite_generic())
    }
}

/// Helper trait for checking finiteness of coordinates.
pub trait FiniteCheck {
    /// Returns true if the value is finite (not NaN or infinite).
    fn is_finite_generic(&self) -> bool;
}

// Unified macro for implementing FiniteCheck
macro_rules! impl_finite_check {
    (float: $($t:ty),*) => {
        $(
            impl FiniteCheck for $t {
                #[inline(always)]
                fn is_finite_generic(&self) -> bool {
                    self.is_finite()
                }
            }
        )*
    };
    (int: $($t:ty),*) => {
        $(
            impl FiniteCheck for $t {
                #[inline(always)]
                fn is_finite_generic(&self) -> bool {
                    true
                }
            }
        )*
    };
}

impl_finite_check!(float: f32, f64);
impl_finite_check!(int: i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

/// Helper trait for hashing individual coordinates for non-hashable types
/// like f32 and f64.
pub trait HashCoordinate {
    /// Hashes a single coordinate value using the provided hasher.
    ///
    /// This method provides a consistent way to hash coordinate values,
    /// including floating-point types that don't normally implement Hash.
    /// For floating-point types, this uses OrderedFloat to ensure consistent
    /// hashing behavior, including proper handling of NaN values.
    ///
    /// # Arguments
    ///
    /// * `state` - The hasher state to write the hash value to
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::point::HashCoordinate;
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

// Unified macro for implementing HashCoordinate
macro_rules! impl_hash_coordinate {
    (float: $($t:ty),*) => {
        $(
            impl HashCoordinate for $t {
                #[inline(always)]
                fn hash_coord<H: Hasher>(&self, state: &mut H) {
                    OrderedFloat(*self).hash(state);
                }
            }
        )*
    };
    (int: $($t:ty),*) => {
        $(
            impl HashCoordinate for $t {
                #[inline(always)]
                fn hash_coord<H: Hasher>(&self, state: &mut H) {
                    self.hash(state);
                }
            }
        )*
    };
}

impl_hash_coordinate!(float: f32, f64);
impl_hash_coordinate!(int: i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl<T, const D: usize> Hash for Point<T, D>
where
    T: HashCoordinate + Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        for &coord in &self.coords {
            coord.hash_coord(state);
        }
    }
}

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
    /// # Returns:
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
impl<T, const D: usize> PartialEq for Point<T, D>
where
    T: Clone + Copy + Default + PartialOrd + OrderedEq,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    fn eq(&self, other: &Self) -> bool {
        self.coords
            .iter()
            .zip(other.coords.iter())
            .all(|(a, b)| a.ordered_eq(b))
    }
}

// Manual implementation of Eq for Point using OrderedFloat for proper NaN handling
impl<T, const D: usize> Eq for Point<T, D>
where
    T: Clone + Copy + Default + PartialOrd + OrderedEq,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
}

/// From trait implementations for Point conversions
impl<T, U, const D: usize> From<[T; D]> for Point<U, D>
where
    T: Clone + Copy + Default + Into<U> + PartialEq + PartialOrd,
    U: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [U; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// Create a new [Point] from an array of coordinates of type `T`.
    ///
    /// # Arguments:
    ///
    /// * `coords`: An array of type `T` with a length of `D`, representing the
    ///   coordinates of the point.
    ///
    /// # Returns:
    ///
    /// The function returns a new instance of the [Point] struct with the
    /// coordinates converted to type `U`.
    ///
    /// # Example:
    ///
    /// ```rust
    /// use d_delaunay::delaunay_core::point::Point;
    /// let coords = [1, 2, 3];
    /// let point: Point<f64, 3> = Point::from(coords);
    /// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
    /// ```
    #[inline]
    fn from(coords: [T; D]) -> Self {
        // Convert the `coords` array to `[U; D]`
        let coords_u: [U; D] = coords.map(std::convert::Into::into);
        Self { coords: coords_u }
    }
}

/// Enable conversions from Point to coordinate arrays
/// This allows `point` and `&point` to be implicitly converted to `[T; D]`
impl<T, const D: usize> From<Point<T, D>> for [T; D]
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// # Example:
    ///
    /// ```rust
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0]);
    /// let coords: [f64; 2] = point.into();
    /// assert_eq!(coords, [1.0, 2.0]);
    /// ```
    #[inline]
    fn from(point: Point<T, D>) -> [T; D] {
        point.coordinates()
    }
}

impl<T, const D: usize> From<&Point<T, D>> for [T; D]
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// # Example:
    ///
    /// ```rust
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([3, 4]);
    /// let coords: [i32; 2] = (&point).into();
    /// assert_eq!(coords, [3, 4]);
    /// ```
    #[inline]
    fn from(point: &Point<T, D>) -> [T; D] {
        point.coordinates()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Helper function to get hash value for any hashable type
    fn get_hash<T: Hash>(value: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    // Helper function to test basic point properties
    fn test_basic_point_properties<T, const D: usize>(
        point: &Point<T, D>,
        expected_coords: [T; D],
        expected_dim: usize,
    ) where
        T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq + std::fmt::Debug,
        [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    {
        assert_eq!(point.coordinates(), expected_coords);
        assert_eq!(point.dim(), expected_dim);
    }

    // Helper function to test point equality and hash consistency
    fn test_point_equality_and_hash<T, const D: usize>(
        point1: Point<T, D>,
        point2: Point<T, D>,
        should_be_equal: bool,
    ) where
        T: HashCoordinate
            + Clone
            + Copy
            + Default
            + PartialEq
            + PartialOrd
            + OrderedEq
            + std::fmt::Debug,
        [T; D]: Copy + Default + serde::de::DeserializeOwned + serde::Serialize + Sized,
    {
        if should_be_equal {
            assert_eq!(point1, point2);
            assert_eq!(get_hash(&point1), get_hash(&point2));
        } else {
            assert_ne!(point1, point2);
            // Note: Different points may still hash to same value (hash collisions)
        }
    }

    #[test]
    fn point_default() {
        let point: Point<f64, 4> = Default::default();

        assert_eq!(point.coordinates(), [0.0, 0.0, 0.0, 0.0]);

        // Human readable output for cargo test -- --nocapture
        println!("Default: {point:?}");
    }

    #[test]
    fn point_new() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);

        assert_eq!(point.coordinates(), [1.0, 2.0, 3.0, 4.0]);

        // Human readable output for cargo test -- --nocapture
        println!("Point: {point:?}");
    }

    #[test]
    fn point_copy() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);
        let point_copy = point;

        assert_eq!(point, point_copy);
        assert_eq!(point.coordinates(), point_copy.coordinates());
        assert_eq!(point.dim(), point_copy.dim());
    }

    #[test]
    fn point_dim() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);

        assert_eq!(point.dim(), 4);

        // Human readable output for cargo test -- --nocapture
        println!("Point: {:?} is {}-D", point, point.dim());
    }

    #[test]
    fn point_origin() {
        let point: Point<f64, 4> = Point::origin();

        assert_eq!(point.coordinates(), [0.0, 0.0, 0.0, 0.0]);

        // Human readable output for cargo test -- --nocapture
        println!("Origin: {:?} is {}-D", point, point.dim());
    }

    #[test]
    fn point_serialization() {
        use serde_test::{assert_tokens, Token};

        let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        assert_tokens(
            &point,
            &[
                Token::Struct {
                    name: "Point",
                    len: 1,
                },
                Token::Str("coords"),
                Token::Tuple { len: 3 },
                Token::F64(1.0),
                Token::F64(2.0),
                Token::F64(3.0),
                Token::TupleEnd,
                Token::StructEnd,
            ],
        );
    }

    #[test]
    fn point_to_and_from_json() {
        let point: Point<f64, 4> = Default::default();
        let serialized = serde_json::to_string(&point).unwrap();

        assert_eq!(serialized, "{\"coords\":[0.0,0.0,0.0,0.0]}");

        let deserialized: Point<f64, 4> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, point);

        // Human readable output for cargo test -- --nocapture
        println!("Serialized: {serialized:?}");
    }

    #[test]
    fn point_from_array() {
        let coords = [1i32, 2i32, 3i32];
        let point: Point<f64, 3> = Point::from(coords);

        assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
        assert_eq!(point.dim(), 3);
    }

    #[test]
    fn point_from_array_f32_to_f64() {
        let coords = [1.5f32, 2.5f32, 3.5f32, 4.5f32];
        let point: Point<f64, 4> = Point::from(coords);

        assert_eq!(point.coordinates(), [1.5, 2.5, 3.5, 4.5]);
        assert_eq!(point.dim(), 4);
    }

    #[test]
    fn point_from_array_same_type() {
        // Test conversion when source and target types are the same
        let coords_f32 = [1.0f32, 2.0f32, 3.0f32];
        let point_f32: Point<f32, 3> = Point::from(coords_f32);
        assert_eq!(point_f32.coordinates(), [1.0f32, 2.0f32, 3.0f32]);

        let coords_i32 = [1i32, 2i32, 3i32];
        let point_i32: Point<i32, 3> = Point::from(coords_i32);
        assert_eq!(point_i32.coordinates(), [1i32, 2i32, 3i32]);
    }

    #[test]
    fn point_from_array_integer_to_integer() {
        // Test conversion from i32 to i64
        let coords_i32 = [1i32, 2i32, 3i32];
        let point_i64: Point<i64, 3> = Point::from(coords_i32);
        assert_eq!(point_i64.coordinates(), [1i64, 2i64, 3i64]);

        // Test conversion from u8 to i32
        let coords_u8 = [10u8, 20u8, 30u8];
        let point_i32: Point<i32, 3> = Point::from(coords_u8);
        assert_eq!(point_i32.coordinates(), [10i32, 20i32, 30i32]);

        // Test conversion from i16 to isize
        let coords_i16 = [100i16, 200i16];
        let point_isize: Point<isize, 2> = Point::from(coords_i16);
        assert_eq!(point_isize.coordinates(), [100isize, 200isize]);
    }

    #[test]
    fn point_from_array_float_to_float() {
        // Test conversion from f32 to f32 (same type)
        let coords_f32 = [1.5f32, 2.5f32];
        let point_f32: Point<f32, 2> = Point::from(coords_f32);
        assert_eq!(point_f32.coordinates(), [1.5f32, 2.5f32]);

        // Test conversion from f32 to f64 (safe upcast)
        let coords_f32 = [1.5f32, 2.5f32];
        let point_f64: Point<f64, 2> = Point::from(coords_f32);
        assert_eq!(point_f64.coordinates(), [1.5f64, 2.5f64]);
    }

    #[test]
    fn point_from_array_integer_to_float() {
        // Test conversion from i32 to f64
        let coords_i32 = [1i32, 2i32, 3i32];
        let point_f64: Point<f64, 3> = Point::from(coords_i32);
        assert_eq!(point_f64.coordinates(), [1.0f64, 2.0f64, 3.0f64]);

        // Test conversion from u8 to f64
        let coords_u8 = [10u8, 20u8];
        let point_f64: Point<f64, 2> = Point::from(coords_u8);
        assert_eq!(point_f64.coordinates(), [10.0f64, 20.0f64]);
    }

    #[test]
    fn point_hash() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);
        let point3 = Point::new([1.0, 2.0, 4.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        let mut hasher3 = DefaultHasher::new();

        point1.hash(&mut hasher1);
        point2.hash(&mut hasher2);
        point3.hash(&mut hasher3);

        // Same points should have same hash
        assert_eq!(hasher1.finish(), hasher2.finish());
        // Different points should have different hash (with high probability)
        assert_ne!(hasher1.finish(), hasher3.finish());
    }

    #[test]
    fn point_hash_in_hashmap() {
        use std::collections::HashMap;

        let mut map: HashMap<Point<f64, 2>, i32> = HashMap::new();

        let point1 = Point::new([1.0, 2.0]);
        let point2 = Point::new([3.0, 4.0]);
        let point3 = Point::new([1.0, 2.0]); // Same as point1

        map.insert(point1, 10);
        map.insert(point2, 20);

        assert_eq!(map.get(&point3), Some(&10)); // Should find point1's value
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn point_partial_eq() {
        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);
        let point3 = Point::new([1.0, 2.0, 4.0]);

        assert_eq!(point1, point2);
        assert_ne!(point1, point3);
        assert_ne!(point2, point3);
    }

    #[test]
    fn point_partial_ord() {
        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 4.0]);
        let point3 = Point::new([1.0, 3.0, 0.0]);
        let point4 = Point::new([2.0, 0.0, 0.0]);

        // Lexicographic ordering
        assert!(point1 < point2); // 3.0 < 4.0 in last coordinate
        assert!(point1 < point3); // 2.0 < 3.0 in second coordinate
        assert!(point1 < point4); // 1.0 < 2.0 in first coordinate
        assert!(point2 > point1);
    }

    #[test]
    fn point_from_complex_conversions() {
        // Test conversion with mixed type arrays
        let coords_mixed_i32 = [-100i32, 200i32, 300i32];
        let point_f64: Point<f64, 3> = Point::from(coords_mixed_i32);
        assert_eq!(point_f64.coordinates(), [-100.0f64, 200.0f64, 300.0f64]);

        // Test with larger values
        let coords_large = [10000i32, 20000i32];
        let point_f64: Point<f64, 2> = Point::from(coords_large);
        assert_eq!(point_f64.coordinates(), [10000.0f64, 20000.0f64]);

        // Test with very small values
        // When converting from f32 to f64, there can be small precision
        // differences due to how floating point numbers are represented in
        // memory. Use approximate comparison for these small values.
        let coords_small_f32 = [0.000001f32, 0.000002f32];
        let point_f64: Point<f64, 2> = Point::from(coords_small_f32);

        // Use relative comparison with appropriate epsilon for small floating
        // point values
        let expected = [0.000001f64, 0.000002f64];
        for (&actual, &expected) in point_f64.coordinates().iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6, max_relative = 1e-6);
        }
    }

    #[test]
    fn point_1d() {
        let point: Point<f64, 1> = Point::new([42.0]);
        test_basic_point_properties(&point, [42.0], 1);

        let origin: Point<f64, 1> = Point::origin();
        test_basic_point_properties(&origin, [0.0], 1);
    }

    #[test]
    fn point_2d() {
        let point: Point<f64, 2> = Point::new([1.0, 2.0]);
        test_basic_point_properties(&point, [1.0, 2.0], 2);

        let origin: Point<f64, 2> = Point::origin();
        test_basic_point_properties(&origin, [0.0, 0.0], 2);
    }

    #[test]
    fn point_3d() {
        let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);
        test_basic_point_properties(&point, [1.0, 2.0, 3.0], 3);

        let origin: Point<f64, 3> = Point::origin();
        test_basic_point_properties(&origin, [0.0, 0.0, 0.0], 3);
    }

    #[test]
    fn point_5d() {
        let point: Point<f64, 5> = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        test_basic_point_properties(&point, [1.0, 2.0, 3.0, 4.0, 5.0], 5);

        let origin: Point<f64, 5> = Point::origin();
        test_basic_point_properties(&origin, [0.0, 0.0, 0.0, 0.0, 0.0], 5);
    }

    #[test]
    fn point_with_integers() {
        let point: Point<i32, 3> = Point::new([1, 2, 3]);

        assert_eq!(point.coordinates(), [1, 2, 3]);
        assert_eq!(point.dim(), 3);

        let origin: Point<i32, 3> = Point::origin();
        assert_eq!(origin.coordinates(), [0, 0, 0]);
    }

    #[test]
    fn point_with_f32() {
        let point: Point<f32, 2> = Point::new([1.5, 2.5]);

        assert_eq!(point.coordinates(), [1.5, 2.5]);
        assert_eq!(point.dim(), 2);

        let origin: Point<f32, 2> = Point::origin();
        assert_eq!(origin.coordinates(), [0.0, 0.0]);
    }

    #[test]
    fn point_debug_format() {
        let point = Point::new([1.0, 2.0, 3.0]);
        let debug_str = format!("{point:?}");

        assert!(debug_str.contains("Point"));
        assert!(debug_str.contains("coords"));
        assert!(debug_str.contains("1.0"));
        assert!(debug_str.contains("2.0"));
        assert!(debug_str.contains("3.0"));
    }

    #[test]
    fn point_eq_trait() {
        let point1 = Point::new([1.0, 2.0, 3.0]);
        let point2 = Point::new([1.0, 2.0, 3.0]);
        let point3 = Point::new([1.0, 2.0, 4.0]);

        // Test Eq trait (transitivity, reflexivity, symmetry)
        assert_eq!(point1, point1); // reflexive
        assert_eq!(point1, point2); // symmetric
        assert_eq!(point2, point1); // symmetric
        assert_ne!(point1, point3);
        assert_ne!(point3, point1);
    }

    #[test]
    fn point_comprehensive_serialization() {
        // Test with different types and dimensions
        let point_3d = Point::new([1.0, 2.0, 3.0]);
        let serialized_3d = serde_json::to_string(&point_3d).unwrap();
        let deserialized_3d: Point<f64, 3> = serde_json::from_str(&serialized_3d).unwrap();
        assert_eq!(point_3d, deserialized_3d);

        let point_2d = Point::new([10.5, -5.3]);
        let serialized_2d = serde_json::to_string(&point_2d).unwrap();
        let deserialized_2d: Point<f64, 2> = serde_json::from_str(&serialized_2d).unwrap();
        assert_eq!(point_2d, deserialized_2d);

        let point_1d = Point::new([42.0]);
        let serialized_1d = serde_json::to_string(&point_1d).unwrap();
        let deserialized_1d: Point<f64, 1> = serde_json::from_str(&serialized_1d).unwrap();
        assert_eq!(point_1d, deserialized_1d);
    }

    #[test]
    fn point_negative_coordinates() {
        let point = Point::new([-1.0, -2.0, -3.0]);

        assert_eq!(point.coordinates(), [-1.0, -2.0, -3.0]);
        assert_eq!(point.dim(), 3);

        // Test with mixed positive/negative
        let mixed_point = Point::new([1.0, -2.0, 3.0, -4.0]);
        assert_eq!(mixed_point.coordinates(), [1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn point_zero_coordinates() {
        let zero_point = Point::new([0.0, 0.0, 0.0]);
        let origin: Point<f64, 3> = Point::origin();

        assert_eq!(zero_point, origin);
        assert_eq!(zero_point.coordinates(), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn point_large_coordinates() {
        let large_point = Point::new([1e6, 2e6, 3e6]);

        assert_eq!(large_point.coordinates(), [1000000.0, 2000000.0, 3000000.0]);
        assert_eq!(large_point.dim(), 3);
    }

    #[test]
    fn point_small_coordinates() {
        let small_point = Point::new([1e-6, 2e-6, 3e-6]);

        assert_eq!(small_point.coordinates(), [0.000001, 0.000002, 0.000003]);
        assert_eq!(small_point.dim(), 3);
    }

    #[test]
    fn point_ordering_edge_cases() {
        use std::cmp::Ordering;

        let point1 = Point::new([1.0, 2.0]);
        let point2 = Point::new([1.0, 2.0]);

        // Test that equal points are not less than each other
        assert_ne!(point1.partial_cmp(&point2), Some(Ordering::Less));
        assert_ne!(point2.partial_cmp(&point1), Some(Ordering::Less));
        assert!(point1 <= point2);
        assert!(point2 <= point1);
        assert!(point1 >= point2);
        assert!(point2 >= point1);
    }

    #[test]
    fn point_from_different_integer_types() {
        // Test conversion from different integer types
        let u8_coords: [u8; 3] = [1, 2, 3];
        let point_from_u8: Point<f64, 3> = Point::from(u8_coords);
        assert_eq!(point_from_u8.coordinates(), [1.0, 2.0, 3.0]);

        let i16_coords: [i16; 2] = [-1, 32767];
        let point_from_i16: Point<f64, 2> = Point::from(i16_coords);
        assert_eq!(point_from_i16.coordinates(), [-1.0, 32767.0]);
    }

    #[test]
    fn point_hash_f32() {
        use std::collections::HashMap;

        let mut map: HashMap<Point<f32, 2>, i32> = HashMap::new();

        let point1 = Point::new([1.5f32, 2.5f32]);
        let point2 = Point::new([3.5f32, 4.5f32]);
        let point3 = Point::new([1.5f32, 2.5f32]); // Same as point1

        map.insert(point1, 10);
        map.insert(point2, 20);

        assert_eq!(map.get(&point3), Some(&10)); // Should find point1's value
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn point_hash_integers() {
        use std::collections::HashMap;

        // Test with i32
        let mut map_i32: HashMap<Point<i32, 3>, &str> = HashMap::new();
        let point_i32_1 = Point::new([1, 2, 3]);
        let point_i32_2 = Point::new([4, 5, 6]);
        let point_i32_3 = Point::new([1, 2, 3]); // Same as point_i32_1

        map_i32.insert(point_i32_1, "first");
        map_i32.insert(point_i32_2, "second");

        assert_eq!(map_i32.get(&point_i32_3), Some(&"first"));
        assert_eq!(map_i32.len(), 2);

        // Test with u64
        let mut map_u64: HashMap<Point<u64, 2>, bool> = HashMap::new();
        let point_u64_1 = Point::new([100u64, 200u64]);
        let point_u64_2 = Point::new([300u64, 400u64]);
        let point_u64_3 = Point::new([100u64, 200u64]); // Same as point_u64_1

        map_u64.insert(point_u64_1, true);
        map_u64.insert(point_u64_2, false);

        assert_eq!(map_u64.get(&point_u64_3), Some(&true));
        assert_eq!(map_u64.len(), 2);
    }

    #[test]
    fn point_eq_different_types() {
        // Test Eq for f64
        let point_f64_1 = Point::new([1.0, 2.0]);
        let point_f64_2 = Point::new([1.0, 2.0]);
        let point_f64_3 = Point::new([1.0, 2.1]);

        assert_eq!(point_f64_1, point_f64_2);
        assert_ne!(point_f64_1, point_f64_3);

        // Test Eq for f32
        let point_f32_1 = Point::new([1.5f32, 2.5f32]);
        let point_f32_2 = Point::new([1.5f32, 2.5f32]);
        let point_f32_3 = Point::new([1.5f32, 2.6f32]);

        assert_eq!(point_f32_1, point_f32_2);
        assert_ne!(point_f32_1, point_f32_3);

        // Test Eq for i32
        let point_i32_1 = Point::new([10, 20]);
        let point_i32_2 = Point::new([10, 20]);
        let point_i32_3 = Point::new([10, 21]);

        assert_eq!(point_i32_1, point_i32_2);
        assert_ne!(point_i32_1, point_i32_3);
    }

    #[test]
    fn point_hash_consistency_floating_point() {
        // Test that OrderedFloat provides consistent hashing for NaN-free floats
        let point1 = Point::new([1.0, 2.0, 3.5]);
        let point2 = Point::new([1.0, 2.0, 3.5]);
        test_point_equality_and_hash(point1, point2, true);

        // Test with f32
        let point_f32_1 = Point::new([1.5f32, 2.5f32]);
        let point_f32_2 = Point::new([1.5f32, 2.5f32]);
        test_point_equality_and_hash(point_f32_1, point_f32_2, true);
    }

    #[test]
    fn point_hash_consistency_integers() {
        // Test integer hashing consistency
        let point_i32_1 = Point::new([42, -17, 100]);
        let point_i32_2 = Point::new([42, -17, 100]);
        test_point_equality_and_hash(point_i32_1, point_i32_2, true);

        // Test with u64
        let point_u64_1 = Point::new([1000u64, 2000u64]);
        let point_u64_2 = Point::new([1000u64, 2000u64]);
        test_point_equality_and_hash(point_u64_1, point_u64_2, true);
    }

    #[test]
    fn point_hash_all_primitives() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Function to get hash value for any hashable type
        fn get_hash<T: Hash>(value: &T) -> u64 {
            let mut hasher = DefaultHasher::new();
            value.hash(&mut hasher);
            hasher.finish()
        }

        // Test all primitive integer types
        let point_i8: Point<i8, 2> = Point::new([1, 2]);
        let point_i16: Point<i16, 2> = Point::new([1, 2]);
        let point_i32: Point<i32, 2> = Point::new([1, 2]);
        let point_i64: Point<i64, 2> = Point::new([1, 2]);
        let point_u8: Point<u8, 2> = Point::new([1, 2]);
        let point_u16: Point<u16, 2> = Point::new([1, 2]);
        let point_u32: Point<u32, 2> = Point::new([1, 2]);
        let point_u64: Point<u64, 2> = Point::new([1, 2]);
        let point_usize: Point<usize, 2> = Point::new([1, 2]);
        let point_isize: Point<isize, 2> = Point::new([1, 2]);

        // Get hash for each type
        let _hash_i8 = get_hash(&point_i8);
        let _hash_i16 = get_hash(&point_i16);
        let _hash_i32 = get_hash(&point_i32);
        let _hash_i64 = get_hash(&point_i64);
        let _hash_u8 = get_hash(&point_u8);
        let _hash_u16 = get_hash(&point_u16);
        let _hash_u32 = get_hash(&point_u32);
        let _hash_u64 = get_hash(&point_u64);
        let _hash_usize = get_hash(&point_usize);
        let _hash_isize = get_hash(&point_isize);

        // Verify that equal points of the same type hash to the same value
        let point_i32_a: Point<i32, 2> = Point::new([1, 2]);
        let point_i32_b: Point<i32, 2> = Point::new([1, 2]);
        assert_eq!(get_hash(&point_i32_a), get_hash(&point_i32_b));

        // Test points with different values hash differently
        let point_i32_c: Point<i32, 2> = Point::new([2, 3]);
        assert_ne!(get_hash(&point_i32_a), get_hash(&point_i32_c));
    }

    #[test]
    fn point_implicit_conversion_to_coordinates() {
        let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);

        // Test implicit conversion from owned point
        let coords_owned: [f64; 3] = point.into();
        assert_eq!(coords_owned, [1.0, 2.0, 3.0]);

        // Create a new point for reference test
        let point_ref: Point<f64, 3> = Point::new([4.0, 5.0, 6.0]);

        // Test implicit conversion from point reference
        let coords_ref: [f64; 3] = (&point_ref).into();
        assert_eq!(coords_ref, [4.0, 5.0, 6.0]);

        // Verify the original point is still available after reference conversion
        assert_eq!(point_ref.coordinates(), [4.0, 5.0, 6.0]);
    }

    #[test]
    fn point_is_valid_f64() {
        // Test valid f64 points
        let valid_point = Point::new([1.0, 2.0, 3.0]);
        assert!(valid_point.is_valid());

        let valid_negative = Point::new([-1.0, -2.0, -3.0]);
        assert!(valid_negative.is_valid());

        let valid_zero = Point::new([0.0, 0.0, 0.0]);
        assert!(valid_zero.is_valid());

        let valid_mixed = Point::new([1.0, -2.5, 0.0, 42.7]);
        assert!(valid_mixed.is_valid());

        // Test invalid f64 points with NaN
        let invalid_nan_single = Point::new([1.0, f64::NAN, 3.0]);
        assert!(!invalid_nan_single.is_valid());

        let invalid_nan_all = Point::new([f64::NAN, f64::NAN, f64::NAN]);
        assert!(!invalid_nan_all.is_valid());

        let invalid_nan_first = Point::new([f64::NAN, 2.0, 3.0]);
        assert!(!invalid_nan_first.is_valid());

        let invalid_nan_last = Point::new([1.0, 2.0, f64::NAN]);
        assert!(!invalid_nan_last.is_valid());

        // Test invalid f64 points with infinity
        let invalid_pos_inf = Point::new([1.0, f64::INFINITY, 3.0]);
        assert!(!invalid_pos_inf.is_valid());

        let invalid_neg_inf = Point::new([1.0, f64::NEG_INFINITY, 3.0]);
        assert!(!invalid_neg_inf.is_valid());

        let invalid_both_inf = Point::new([f64::INFINITY, f64::NEG_INFINITY]);
        assert!(!invalid_both_inf.is_valid());

        // Test mixed invalid cases
        let invalid_nan_and_inf = Point::new([f64::NAN, f64::INFINITY, 1.0]);
        assert!(!invalid_nan_and_inf.is_valid());
    }

    #[test]
    fn point_is_valid_f32() {
        // Test valid f32 points
        let valid_point = Point::new([1.0f32, 2.0f32, 3.0f32]);
        assert!(valid_point.is_valid());

        let valid_negative = Point::new([-1.5f32, -2.5f32]);
        assert!(valid_negative.is_valid());

        let valid_zero = Point::new([0.0f32]);
        assert!(valid_zero.is_valid());

        // Test invalid f32 points with NaN
        let invalid_nan = Point::new([1.0f32, f32::NAN]);
        assert!(!invalid_nan.is_valid());

        let invalid_all_nan = Point::new([f32::NAN, f32::NAN, f32::NAN, f32::NAN]);
        assert!(!invalid_all_nan.is_valid());

        // Test invalid f32 points with infinity
        let invalid_pos_inf = Point::new([f32::INFINITY, 2.0f32]);
        assert!(!invalid_pos_inf.is_valid());

        let invalid_neg_inf = Point::new([1.0f32, f32::NEG_INFINITY]);
        assert!(!invalid_neg_inf.is_valid());

        // Test edge cases with very small and large values (but finite)
        let valid_small = Point::new([f32::MIN_POSITIVE, -f32::MIN_POSITIVE]);
        assert!(valid_small.is_valid());

        let valid_large = Point::new([f32::MAX, -f32::MAX]);
        assert!(valid_large.is_valid());
    }

    #[test]
    fn point_is_valid_integers() {
        // All integer types should always be valid (no NaN or infinity)
        let valid_i32 = Point::new([1i32, 2i32, 3i32]);
        assert!(valid_i32.is_valid());

        let valid_negative_i32 = Point::new([-1i32, -2i32, -3i32]);
        assert!(valid_negative_i32.is_valid());

        let valid_zero_i32 = Point::new([0i32, 0i32]);
        assert!(valid_zero_i32.is_valid());

        let valid_u64 = Point::new([u64::MAX, u64::MIN, 42u64]);
        assert!(valid_u64.is_valid());

        let valid_i8 = Point::new([i8::MAX, i8::MIN, 0i8, -1i8]);
        assert!(valid_i8.is_valid());

        let valid_isize = Point::new([isize::MAX, isize::MIN]);
        assert!(valid_isize.is_valid());

        // Test with various integer types
        let valid_u8 = Point::new([255u8, 0u8, 128u8]);
        assert!(valid_u8.is_valid());

        let valid_i16 = Point::new([32767i16, -32768i16, 0i16]);
        assert!(valid_i16.is_valid());

        let valid_u32 = Point::new([u32::MAX, 0u32, 42u32]);
        assert!(valid_u32.is_valid());
    }

    #[test]
    fn point_is_valid_different_dimensions() {
        // Test 1D points
        let valid_1d_f64 = Point::new([42.0]);
        assert!(valid_1d_f64.is_valid());

        let invalid_1d_nan = Point::new([f64::NAN]);
        assert!(!invalid_1d_nan.is_valid());

        let valid_1d_int = Point::new([42i32]);
        assert!(valid_1d_int.is_valid());

        // Test 2D points
        let valid_2d = Point::new([1.0, 2.0]);
        assert!(valid_2d.is_valid());

        let invalid_2d = Point::new([1.0, f64::INFINITY]);
        assert!(!invalid_2d.is_valid());

        // Test higher dimensional points
        let valid_5d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(valid_5d.is_valid());

        let invalid_5d = Point::new([1.0, 2.0, f64::NAN, 4.0, 5.0]);
        assert!(!invalid_5d.is_valid());

        // Test 10D point
        let valid_10d = Point::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert!(valid_10d.is_valid());

        let invalid_10d = Point::new([
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            f64::NEG_INFINITY,
            7.0,
            8.0,
            9.0,
            10.0,
        ]);
        assert!(!invalid_10d.is_valid());
    }

    #[test]
    fn point_is_valid_edge_cases() {
        // Test with very small finite values
        let tiny_valid = Point::new([f64::MIN_POSITIVE, -f64::MIN_POSITIVE, 0.0]);
        assert!(tiny_valid.is_valid());

        // Test with very large finite values
        let large_valid = Point::new([f64::MAX, -f64::MAX]);
        assert!(large_valid.is_valid());

        // Test subnormal numbers (should be valid)
        let subnormal = f64::MIN_POSITIVE / 2.0;
        let subnormal_point = Point::new([subnormal, -subnormal]);
        assert!(subnormal_point.is_valid());

        // Test zero and negative zero
        let zero_point = Point::new([0.0, -0.0]);
        assert!(zero_point.is_valid());

        // Mixed valid and invalid in same point should be invalid
        let mixed_invalid = Point::new([1.0, 2.0, 3.0, f64::NAN, 5.0]);
        assert!(!mixed_invalid.is_valid());

        // All coordinates must be valid for point to be valid
        let one_invalid = Point::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, f64::INFINITY]);
        assert!(!one_invalid.is_valid());
    }

    #[test]
    fn point_nan_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Test that OrderedFloat provides consistent hashing for NaN values
        // Note: Equality comparison for NaN still follows IEEE standard (NaN != NaN)
        // but hashing uses OrderedFloat which treats all NaN values as equivalent

        let point_nan1 = Point::new([f64::NAN, 2.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        point_nan1.hash(&mut hasher1);
        point_nan2.hash(&mut hasher2);

        // NaN points with same structure should hash to same value
        assert_eq!(hasher1.finish(), hasher2.finish());

        // Test with f32 NaN
        let point_f32_nan1 = Point::new([f32::NAN, 1.0f32]);
        let point_f32_nan2 = Point::new([f32::NAN, 1.0f32]);

        let mut hasher_f32_1 = DefaultHasher::new();
        let mut hasher_f32_2 = DefaultHasher::new();

        point_f32_nan1.hash(&mut hasher_f32_1);
        point_f32_nan2.hash(&mut hasher_f32_2);

        assert_eq!(hasher_f32_1.finish(), hasher_f32_2.finish());
    }

    #[test]
    fn point_infinity_hash_consistency() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Test that OrderedFloat provides consistent hashing for infinity values
        let point_pos_inf1 = Point::new([f64::INFINITY, 2.0]);
        let point_pos_inf2 = Point::new([f64::INFINITY, 2.0]);

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();

        point_pos_inf1.hash(&mut hasher1);
        point_pos_inf2.hash(&mut hasher2);

        // Same infinity points should hash to same value
        assert_eq!(hasher1.finish(), hasher2.finish());

        // Test negative infinity
        let point_neg_inf1 = Point::new([f64::NEG_INFINITY, 2.0]);
        let point_neg_inf2 = Point::new([f64::NEG_INFINITY, 2.0]);

        let mut hasher_neg1 = DefaultHasher::new();
        let mut hasher_neg2 = DefaultHasher::new();

        point_neg_inf1.hash(&mut hasher_neg1);
        point_neg_inf2.hash(&mut hasher_neg2);

        assert_eq!(hasher_neg1.finish(), hasher_neg2.finish());

        // Positive and negative infinity should hash differently
        assert_ne!(hasher1.finish(), hasher_neg1.finish());
    }

    #[test]
    fn point_nan_infinity_hash_consistency() {
        use std::collections::HashMap;

        // Test that points with NaN can be used as HashMap keys
        let mut map: HashMap<Point<f64, 2>, i32> = HashMap::new();

        let point_nan1 = Point::new([f64::NAN, 2.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0]); // Should be equal to point_nan1
        let point_inf = Point::new([f64::INFINITY, 2.0]);

        map.insert(point_nan1, 100);
        map.insert(point_inf, 200);

        // Should be able to retrieve using equivalent NaN point
        assert_eq!(map.get(&point_nan2), Some(&100));
        assert_eq!(map.len(), 2);

        // Test with f32
        let mut map_f32: HashMap<Point<f32, 1>, i32> = HashMap::new();

        let point_f32_nan1 = Point::new([f32::NAN]);
        let point_f32_nan2 = Point::new([f32::NAN]);

        map_f32.insert(point_f32_nan1, 300);
        assert_eq!(map_f32.get(&point_f32_nan2), Some(&300));
    }

    #[test]
    fn point_nan_equality_comparison() {
        // Test that NaN == NaN using our OrderedEq implementation
        // This is different from IEEE 754 standard where NaN != NaN

        // f64 NaN comparisons
        let point_nan1 = Point::new([f64::NAN, 2.0, 3.0]);
        let point_nan2 = Point::new([f64::NAN, 2.0, 3.0]);
        let point_nan3 = Point::new([f64::NAN, f64::NAN, f64::NAN]);
        let point_nan4 = Point::new([f64::NAN, f64::NAN, f64::NAN]);

        // Points with NaN should be equal when all coordinates match
        assert_eq!(point_nan1, point_nan2);
        assert_eq!(point_nan3, point_nan4);

        // Points with different NaN positions should not be equal
        let point_nan_diff1 = Point::new([f64::NAN, 2.0, 3.0]);
        let point_nan_diff2 = Point::new([1.0, f64::NAN, 3.0]);
        assert_ne!(point_nan_diff1, point_nan_diff2);

        // f32 NaN comparisons
        let point_f32_nan1 = Point::new([f32::NAN, 1.5f32]);
        let point_f32_nan2 = Point::new([f32::NAN, 1.5f32]);
        assert_eq!(point_f32_nan1, point_f32_nan2);

        // Mixed NaN and normal values
        let point_mixed1 = Point::new([1.0, f64::NAN, 3.0, 4.0]);
        let point_mixed2 = Point::new([1.0, f64::NAN, 3.0, 4.0]);
        let point_mixed3 = Point::new([1.0, f64::NAN, 3.0, 5.0]); // Different last coordinate

        assert_eq!(point_mixed1, point_mixed2);
        assert_ne!(point_mixed1, point_mixed3);
    }

    #[test]
    fn point_nan_vs_normal_comparison() {
        // Test that NaN points are not equal to points with normal values

        let point_normal = Point::new([1.0, 2.0, 3.0]);
        let point_nan = Point::new([f64::NAN, 2.0, 3.0]);
        let point_nan_all = Point::new([f64::NAN, f64::NAN, f64::NAN]);

        // NaN points should not equal normal points
        assert_ne!(point_normal, point_nan);
        assert_ne!(point_normal, point_nan_all);
        assert_ne!(point_nan, point_normal);
        assert_ne!(point_nan_all, point_normal);

        // Test with f32
        let point_f32_normal = Point::new([1.0f32, 2.0f32]);
        let point_f32_nan = Point::new([f32::NAN, 2.0f32]);

        assert_ne!(point_f32_normal, point_f32_nan);
        assert_ne!(point_f32_nan, point_f32_normal);
    }

    #[test]
    fn point_infinity_comparison() {
        // Test comparison behavior with infinity values

        // Positive infinity comparisons
        let point_pos_inf1 = Point::new([f64::INFINITY, 2.0]);
        let point_pos_inf2 = Point::new([f64::INFINITY, 2.0]);
        assert_eq!(point_pos_inf1, point_pos_inf2);

        // Negative infinity comparisons
        let point_neg_inf1 = Point::new([f64::NEG_INFINITY, 2.0]);
        let point_neg_inf2 = Point::new([f64::NEG_INFINITY, 2.0]);
        assert_eq!(point_neg_inf1, point_neg_inf2);

        // Positive vs negative infinity should not be equal
        assert_ne!(point_pos_inf1, point_neg_inf1);

        // Infinity vs normal values should not be equal
        let point_normal = Point::new([1.0, 2.0]);
        assert_ne!(point_pos_inf1, point_normal);
        assert_ne!(point_neg_inf1, point_normal);

        // Test with f32
        let point_f32_pos_inf1 = Point::new([f32::INFINITY]);
        let point_f32_pos_inf2 = Point::new([f32::INFINITY]);
        let point_f32_neg_inf = Point::new([f32::NEG_INFINITY]);

        assert_eq!(point_f32_pos_inf1, point_f32_pos_inf2);
        assert_ne!(point_f32_pos_inf1, point_f32_neg_inf);
    }

    #[test]
    fn point_nan_infinity_mixed_comparison() {
        // Test comparisons with mixed NaN and infinity values

        let point_nan_inf1 = Point::new([f64::NAN, f64::INFINITY, 1.0]);
        let point_nan_inf2 = Point::new([f64::NAN, f64::INFINITY, 1.0]);
        let point_nan_inf3 = Point::new([f64::NAN, f64::NEG_INFINITY, 1.0]);

        // Same NaN/infinity pattern should be equal
        assert_eq!(point_nan_inf1, point_nan_inf2);

        // Different infinity signs should not be equal
        assert_ne!(point_nan_inf1, point_nan_inf3);

        // Test various combinations
        let point_all_special = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
        let point_all_special_copy =
            Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
        let point_all_special_diff =
            Point::new([f64::NAN, f64::NEG_INFINITY, f64::INFINITY, f64::NAN]);

        assert_eq!(point_all_special, point_all_special_copy);
        assert_ne!(point_all_special, point_all_special_diff);
    }

    #[test]
    fn point_nan_reflexivity() {
        // Test that NaN points are equal to themselves (reflexivity)

        let point_nan = Point::new([f64::NAN, f64::NAN, f64::NAN]);
        assert_eq!(point_nan, point_nan);

        let point_mixed = Point::new([1.0, f64::NAN, 3.0, f64::INFINITY]);
        assert_eq!(point_mixed, point_mixed);

        // Test with f32
        let point_f32_nan = Point::new([f32::NAN, f32::NAN]);
        assert_eq!(point_f32_nan, point_f32_nan);
    }

    #[test]
    fn point_nan_symmetry() {
        // Test that NaN equality is symmetric (if a == b, then b == a)

        let point_a = Point::new([f64::NAN, 2.0, f64::INFINITY]);
        let point_b = Point::new([f64::NAN, 2.0, f64::INFINITY]);

        assert_eq!(point_a, point_b);
        assert_eq!(point_b, point_a); // Should be symmetric

        // Test with f32
        let point_f32_a = Point::new([f32::NAN, 1.0f32, f32::NEG_INFINITY]);
        let point_f32_b = Point::new([f32::NAN, 1.0f32, f32::NEG_INFINITY]);

        assert_eq!(point_f32_a, point_f32_b);
        assert_eq!(point_f32_b, point_f32_a);
    }

    #[test]
    fn point_nan_transitivity() {
        // Test that NaN equality is transitive (if a == b and b == c, then a == c)

        let point_a = Point::new([f64::NAN, 2.0, 3.0]);
        let point_b = Point::new([f64::NAN, 2.0, 3.0]);
        let point_c = Point::new([f64::NAN, 2.0, 3.0]);

        assert_eq!(point_a, point_b);
        assert_eq!(point_b, point_c);
        assert_eq!(point_a, point_c); // Should be transitive

        // Test with complex special values
        let point_complex_a = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let point_complex_b = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);
        let point_complex_c = Point::new([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0]);

        assert_eq!(point_complex_a, point_complex_b);
        assert_eq!(point_complex_b, point_complex_c);
        assert_eq!(point_complex_a, point_complex_c);
    }

    #[test]
    fn point_nan_different_bit_patterns() {
        // Test that different NaN bit patterns are considered equal
        // Note: Rust's f64::NAN is a specific bit pattern, but there are many possible NaN values

        // Create different NaN values
        let nan1 = f64::NAN;
        #[allow(clippy::zero_divided_by_zero)]
        let nan2 = 0.0f64 / 0.0f64; // Another way to create NaN
        let nan3 = f64::INFINITY - f64::INFINITY; // Yet another way

        // Verify they are all NaN
        assert!(nan1.is_nan());
        assert!(nan2.is_nan());
        assert!(nan3.is_nan());

        // Points with different NaN bit patterns should be equal
        let point1 = Point::new([nan1, 1.0]);
        let point2 = Point::new([nan2, 1.0]);
        let point3 = Point::new([nan3, 1.0]);

        assert_eq!(point1, point2);
        assert_eq!(point2, point3);
        assert_eq!(point1, point3);

        // Test with f32 as well
        let f32_nan1 = f32::NAN;
        #[allow(clippy::zero_divided_by_zero)]
        let f32_nan2 = 0.0f32 / 0.0f32;

        let point_f32_1 = Point::new([f32_nan1]);
        let point_f32_2 = Point::new([f32_nan2]);

        assert_eq!(point_f32_1, point_f32_2);
    }

    #[test]
    fn point_nan_in_different_dimensions() {
        // Test NaN behavior across different dimensionalities

        // 1D
        let point_1d_a = Point::new([f64::NAN]);
        let point_1d_b = Point::new([f64::NAN]);
        assert_eq!(point_1d_a, point_1d_b);

        // 2D
        let point_2d_a = Point::new([f64::NAN, f64::NAN]);
        let point_2d_b = Point::new([f64::NAN, f64::NAN]);
        assert_eq!(point_2d_a, point_2d_b);

        // 3D
        let point_3d_a = Point::new([f64::NAN, 1.0, f64::NAN]);
        let point_3d_b = Point::new([f64::NAN, 1.0, f64::NAN]);
        assert_eq!(point_3d_a, point_3d_b);

        // 5D
        let point_5d_a = Point::new([f64::NAN, 1.0, f64::NAN, f64::INFINITY, f64::NAN]);
        let point_5d_b = Point::new([f64::NAN, 1.0, f64::NAN, f64::INFINITY, f64::NAN]);
        assert_eq!(point_5d_a, point_5d_b);

        // 10D with mixed special values
        let point_10d_a = Point::new([
            f64::NAN,
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            f64::NAN,
            42.0,
            f64::NAN,
        ]);
        let point_10d_b = Point::new([
            f64::NAN,
            1.0,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            f64::NAN,
            42.0,
            f64::NAN,
        ]);
        assert_eq!(point_10d_a, point_10d_b);
    }

    #[test]
    fn point_nan_zero_comparison() {
        // Test comparison between NaN, positive zero, and negative zero

        let point_nan = Point::new([f64::NAN, f64::NAN]);
        let point_pos_zero = Point::new([0.0, 0.0]);
        let point_neg_zero = Point::new([-0.0, -0.0]);
        let point_mixed_zero = Point::new([0.0, -0.0]);

        // NaN should not equal any zero
        assert_ne!(point_nan, point_pos_zero);
        assert_ne!(point_nan, point_neg_zero);
        assert_ne!(point_nan, point_mixed_zero);

        // Different zeros should be equal (0.0 == -0.0 in IEEE 754)
        assert_eq!(point_pos_zero, point_neg_zero);
        assert_eq!(point_pos_zero, point_mixed_zero);
        assert_eq!(point_neg_zero, point_mixed_zero);

        // Test with f32
        let point_f32_nan = Point::new([f32::NAN]);
        let point_f32_zero = Point::new([0.0f32]);
        let point_f32_neg_zero = Point::new([-0.0f32]);

        assert_ne!(point_f32_nan, point_f32_zero);
        assert_ne!(point_f32_nan, point_f32_neg_zero);
        assert_eq!(point_f32_zero, point_f32_neg_zero);
    }
}

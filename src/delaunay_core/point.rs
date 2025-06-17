//! Data and operations on d-dimensional points.

use ordered_float::OrderedFloat;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::hash::{Hash, Hasher};

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
/// The [Point] struct represents a point in a D-dimensional space, where the
/// coordinates are of type `T`.
///
/// # Properties:
///
/// * `coords`: `coords` is a public property of the [Point]. It is an array of
///   type `T` with a length of `D`. The type `T` is a generic type parameter,
///   which means it can be any type. The length `D` is a constant unsigned
///   integer known at compile time.
pub struct Point<T, const D: usize>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The coordinates of the point.
    pub coords: [T; D],
}

impl<T, const D: usize> From<[T; D]> for Point<f64, D>
where
    T: Clone + Copy + Default + Into<f64> + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    f64: From<T>,
{
    fn from(coords: [T; D]) -> Self {
        // Convert the `coords` array to `[f64; D]`
        let coords_f64: [f64; D] = coords.map(|coord| coord.into());
        Self { coords: coords_f64 }
    }
}

impl<T, const D: usize> Point<T, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
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
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(point.coords, [1.0, 2.0, 3.0, 4.0]);
    /// ```
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
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(point.dim(), 4);
    /// ```
    pub fn dim(&self) -> usize {
        D
    }

    /// The `origin` function returns the origin [Point].
    ///
    /// # Returns:
    ///
    /// The `origin()` function returns a D-dimensional origin point
    /// in Cartesian coordinates.
    ///
    /// # Example
    /// ```
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point: Point<f64, 4> = Point::origin();
    /// assert_eq!(point.coords, [0.0, 0.0, 0.0, 0.0]);
    /// ```
    pub fn origin() -> Self
    where
        T: num_traits::Zero + Copy,
    {
        Self::new([T::zero(); D])
    }
}

impl<T, const D: usize> Eq for Point<T, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
}

impl<T, const D: usize> Hash for Point<T, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    OrderedFloat<f64>: From<T>,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        for val in &self.coords {
            OrderedFloat::<f64>::from(*val).hash(state);
        }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn point_default() {
        let point: Point<f64, 4> = Default::default();

        assert_eq!(point.coords, [0.0, 0.0, 0.0, 0.0]);

        // Human readable output for cargo test -- --nocapture
        println!("Default: {:?}", point);
    }

    #[test]
    fn point_new() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);

        assert_eq!(point.coords, [1.0, 2.0, 3.0, 4.0]);

        // Human readable output for cargo test -- --nocapture
        println!("Point: {:?}", point);
    }

    #[test]
    fn point_copy() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);
        let point_copy = point;

        assert_eq!(point, point_copy);
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

        assert_eq!(point.coords, [0.0, 0.0, 0.0, 0.0]);

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
        println!("Serialized: {:?}", serialized);
    }

    #[test]
    fn point_from_array() {
        let coords = [1i32, 2i32, 3i32];
        let point: Point<f64, 3> = Point::from(coords);

        assert_eq!(point.coords, [1.0, 2.0, 3.0]);
        assert_eq!(point.dim(), 3);
    }

    #[test]
    fn point_from_array_f32_to_f64() {
        let coords = [1.5f32, 2.5f32, 3.5f32, 4.5f32];
        let point: Point<f64, 4> = Point::from(coords);

        assert_eq!(point.coords, [1.5, 2.5, 3.5, 4.5]);
        assert_eq!(point.dim(), 4);
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
    fn point_clone() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);
        let cloned_point = point;

        assert_eq!(point, cloned_point);
        assert_eq!(point.coords, cloned_point.coords);
        assert_eq!(point.dim(), cloned_point.dim());
    }

    #[test]
    fn point_1d() {
        let point: Point<f64, 1> = Point::new([42.0]);

        assert_eq!(point.coords, [42.0]);
        assert_eq!(point.dim(), 1);

        let origin: Point<f64, 1> = Point::origin();
        assert_eq!(origin.coords, [0.0]);
    }

    #[test]
    fn point_2d() {
        let point: Point<f64, 2> = Point::new([1.0, 2.0]);

        assert_eq!(point.coords, [1.0, 2.0]);
        assert_eq!(point.dim(), 2);

        let origin: Point<f64, 2> = Point::origin();
        assert_eq!(origin.coords, [0.0, 0.0]);
    }

    #[test]
    fn point_3d() {
        let point: Point<f64, 3> = Point::new([1.0, 2.0, 3.0]);

        assert_eq!(point.coords, [1.0, 2.0, 3.0]);
        assert_eq!(point.dim(), 3);

        let origin: Point<f64, 3> = Point::origin();
        assert_eq!(origin.coords, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn point_5d() {
        let point: Point<f64, 5> = Point::new([1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(point.coords, [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(point.dim(), 5);

        let origin: Point<f64, 5> = Point::origin();
        assert_eq!(origin.coords, [0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn point_with_integers() {
        let point: Point<i32, 3> = Point::new([1, 2, 3]);

        assert_eq!(point.coords, [1, 2, 3]);
        assert_eq!(point.dim(), 3);

        let origin: Point<i32, 3> = Point::origin();
        assert_eq!(origin.coords, [0, 0, 0]);
    }

    #[test]
    fn point_with_f32() {
        let point: Point<f32, 2> = Point::new([1.5, 2.5]);

        assert_eq!(point.coords, [1.5, 2.5]);
        assert_eq!(point.dim(), 2);

        let origin: Point<f32, 2> = Point::origin();
        assert_eq!(origin.coords, [0.0, 0.0]);
    }

    #[test]
    fn point_debug_format() {
        let point = Point::new([1.0, 2.0, 3.0]);
        let debug_str = format!("{:?}", point);

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

        assert_eq!(point.coords, [-1.0, -2.0, -3.0]);
        assert_eq!(point.dim(), 3);

        // Test with mixed positive/negative
        let mixed_point = Point::new([1.0, -2.0, 3.0, -4.0]);
        assert_eq!(mixed_point.coords, [1.0, -2.0, 3.0, -4.0]);
    }

    #[test]
    fn point_zero_coordinates() {
        let zero_point = Point::new([0.0, 0.0, 0.0]);
        let origin: Point<f64, 3> = Point::origin();

        assert_eq!(zero_point, origin);
        assert_eq!(zero_point.coords, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn point_large_coordinates() {
        let large_point = Point::new([1e6, 2e6, 3e6]);

        assert_eq!(large_point.coords, [1000000.0, 2000000.0, 3000000.0]);
        assert_eq!(large_point.dim(), 3);
    }

    #[test]
    fn point_small_coordinates() {
        let small_point = Point::new([1e-6, 2e-6, 3e-6]);

        assert_eq!(small_point.coords, [0.000001, 0.000002, 0.000003]);
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
        assert_eq!(point_from_u8.coords, [1.0, 2.0, 3.0]);

        let i16_coords: [i16; 2] = [-1, 32767];
        let point_from_i16: Point<f64, 2> = Point::from(i16_coords);
        assert_eq!(point_from_i16.coords, [-1.0, 32767.0]);
    }
}

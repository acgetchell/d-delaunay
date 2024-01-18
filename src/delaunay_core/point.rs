//! Data and operations on d-dimensional points.

use serde::{de::DeserializeOwned, Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, PartialOrd, Serialize)]
/// The `Point` struct represents a point in a D-dimensional space, where the
/// coordinates are of type `T`.
///
/// # Properties:
///
/// * `coords`: `coords` is a public property of the `Point`. It is an array of
/// type `T` with a length of `D`. The type `T` is a generic type parameter,
/// which means it can be any type. The length `D` is a constant unsigned
/// integer, which means it cannot be changed and is known at compile time.
pub struct Point<T, const D: usize>
where
    T: PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + PartialEq + Serialize + Sized,
{
    /// The coordinates of the point.
    pub coords: [T; D],
}

impl<T, const D: usize> From<[T; D]> for Point<f64, D>
where
    T: Into<f64>,
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
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
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
{
    /// The function `new` creates a new instance of a `Point` with the given
    /// coordinates.
    ///
    /// # Arguments:
    ///
    /// * `coords`: The `coords` parameter is an array of type `T` with a
    /// length of `D`.
    ///
    /// # Returns:
    ///
    /// The `new` function returns an instance of the `Point`.
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

    /// The `dim` function returns the dimensionality of the `Point`.
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

    /// The `origin` function returns the origin Point.
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

#[cfg(test)]
mod tests {

    use super::*;

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
    fn point_default_trait() {
        let point: Point<f64, 4> = Default::default();

        assert_eq!(point.coords, [0.0, 0.0, 0.0, 0.0]);

        // Human readable output for cargo test -- --nocapture
        println!("Default: {:?} is {}-D", point, point.dim());
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
}

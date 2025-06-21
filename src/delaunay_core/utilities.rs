//! Utility functions

use super::vertex::Vertex;
use serde::{de::DeserializeOwned, Serialize};
use std::{cmp::Ordering, collections::HashMap, hash::Hash};
use uuid::Uuid;

/// The function `make_uuid` generates a version 4 [Uuid].
///
/// # Returns:
///
/// a randomly generated [Uuid] (Universally Unique Identifier) using the
/// `new_v4` method from the [Uuid] struct.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::utilities::make_uuid;
/// let uuid = make_uuid();
/// assert_eq!(uuid.get_version_num(), 4);
/// ```
pub fn make_uuid() -> Uuid {
    Uuid::new_v4()
}

/// The function `find_extreme_coordinates` takes a [HashMap] of vertices and
/// returns the minimum or maximum coordinates based on the specified
/// ordering.
///
/// # Arguments:
///
/// * `vertices`: A [HashMap] containing [Vertex] objects, where the key is a
///   [Uuid] and the value is a [Vertex].
/// * `ordering`: The `ordering` parameter is of type [Ordering] and is used to
///   specify whether the function should find the minimum or maximum
///   coordinates. [Ordering] is an enum with three possible values: `Less`,
///   `Equal`, and `Greater`.
///
/// # Returns:
///
/// an array of type `T` with length `D` containing the minimum or maximum
/// coordinate for each dimension.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::utilities::find_extreme_coordinates;
/// use d_delaunay::delaunay_core::vertex::Vertex;
/// use d_delaunay::delaunay_core::point::Point;
/// use std::collections::HashMap;
/// use std::cmp::Ordering;
/// let points = vec![
///     Point::new([-1.0, 2.0, 3.0]),
///     Point::new([4.0, -5.0, 6.0]),
///     Point::new([7.0, 8.0, -9.0]),
/// ];
/// let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
/// let hashmap = Vertex::into_hashmap(vertices);
/// let min_coords = find_extreme_coordinates(hashmap, Ordering::Less);
/// assert_eq!(min_coords, [-1.0, -5.0, -9.0]);
/// ```
/// Returns the extreme coordinates (minimum or maximum per dimension) among all vertices based on the specified ordering.
///
/// If the input map is empty, returns an array of default values. For Ordering::Less, finds the minimum coordinate in each dimension; for Ordering::Greater, finds the maximum. For Ordering::Equal, returns the coordinates of the first vertex found.
///
/// # Parameters
/// - vertices: A map of vertices keyed by UUID.
/// - ordering: Determines whether to find minimum, maximum, or simply return the first vertex coordinates.
///
/// # Returns
/// An array containing the extreme coordinates per dimension according to the specified ordering. If no vertices are present, returns an array of default values.
pub fn find_extreme_coordinates<T, U, const D: usize>(
    vertices: HashMap<Uuid, Vertex<T, U, D>>,
    ordering: Ordering,
) -> [T; D]
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
{
    if vertices.is_empty() {
        return [Default::default(); D];
    }

    // Initialize with the first vertex's coordinates using implicit conversion
    let mut extreme_coords: [T; D] = vertices.values().next().unwrap().into();

    // Compare with remaining vertices
    for vertex in vertices.values().skip(1) {
        let vertex_coords: [T; D] = vertex.into();
        for (i, coord) in vertex_coords.iter().enumerate() {
            match ordering {
                Ordering::Less => {
                    if *coord < extreme_coords[i] {
                        extreme_coords[i] = *coord;
                    }
                }
                Ordering::Greater => {
                    if *coord > extreme_coords[i] {
                        extreme_coords[i] = *coord;
                    }
                }
                Ordering::Equal => {
                    // For Equal ordering, return the first vertex's coordinates
                    // This behavior maintains backward compatibility
                }
            }
        }
    }

    extreme_coords
}

/// The function `vec_to_array` converts a [Vec] to an array of f64
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::utilities::vec_to_array;
/// let vec = vec![1.0, 2.0, 3.0];
/// let array = vec_to_array::<3>(vec).unwrap();
/// assert_eq!(array, [1.0, 2.0, 3.0]);
/// ```
pub fn vec_to_array<const D: usize>(vec: Vec<f64>) -> Result<[f64; D], anyhow::Error> {
    if vec.len() != D {
        return Err(anyhow::Error::msg(
            "Vector length does not match array dimension!",
        ));
    }
    let array: [f64; D] = std::array::from_fn(|i| vec[i]);

    Ok(array)
}

#[cfg(test)]
mod tests {

    use crate::delaunay_core::point::Point;

    use super::*;

    #[test]
    fn utilities_uuid() {
        let uuid = make_uuid();

        assert_eq!(uuid.get_version_num(), 4);
        assert_ne!(uuid, make_uuid());

        // Human readable output for cargo test -- --nocapture
        println!("make_uuid = {:?}", uuid);
        println!("uuid version: {:?}\n", uuid.get_version_num());
    }

    #[test]
    fn utilities_find_min_coordinate() {
        let points = vec![
            Point::new([-1.0, 2.0, 3.0]),
            Point::new([4.0, -5.0, 6.0]),
            Point::new([7.0, 8.0, -9.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices);
        let min_coords = find_extreme_coordinates(hashmap, Ordering::Less);

        assert_eq!(min_coords, [-1.0, -5.0, -9.0]);

        // Human readable output for cargo test -- --nocapture
        println!("min_coords = {:?}", min_coords);
    }

    #[test]
    fn utilities_find_max_coordinate() {
        let points = vec![
            Point::new([-1.0, 2.0, 3.0]),
            Point::new([4.0, -5.0, 6.0]),
            Point::new([7.0, 8.0, -9.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices);
        let max_coords = find_extreme_coordinates(hashmap, Ordering::Greater);

        assert_eq!(max_coords, [7.0, 8.0, 6.0]);

        // Human readable output for cargo test -- --nocapture
        println!("max_coords = {:?}", max_coords);
    }

    #[test]
    fn utilities_vec_to_array_success() {
        let vec = vec![1.0, 2.0, 3.0];
        let result = vec_to_array::<3>(vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn utilities_vec_to_array_wrong_length() {
        let vec = vec![1.0, 2.0]; // Length 2, but expecting 3
        let result = vec_to_array::<3>(vec);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Vector length does not match array dimension!"
        );
    }

    #[test]
    fn utilities_vec_to_array_empty() {
        let vec: Vec<f64> = vec![];
        let result = vec_to_array::<3>(vec);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Vector length does not match array dimension!"
        );
    }

    #[test]
    fn utilities_vec_to_array_too_long() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Length 5, but expecting 3
        let result = vec_to_array::<3>(vec);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Vector length does not match array dimension!"
        );
    }

    #[test]
    fn utilities_vec_to_array_1d() {
        let vec = vec![42.0];
        let result = vec_to_array::<1>(vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array, [42.0]);
    }

    #[test]
    fn utilities_vec_to_array_large_dimension() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = vec_to_array::<10>(vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn utilities_vec_to_array_negative_values() {
        let vec = vec![-1.0, -2.0, -3.0];
        let result = vec_to_array::<3>(vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array, [-1.0, -2.0, -3.0]);
    }

    #[test]
    fn utilities_vec_to_array_zero() {
        let vec = vec![0.0, 0.0, 0.0];
        let result = vec_to_array::<3>(vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_eq!(array, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn utilities_find_extreme_coordinates_empty() {
        let empty_hashmap: HashMap<Uuid, Vertex<f64, Option<()>, 3>> = HashMap::new();
        let min_coords = find_extreme_coordinates(empty_hashmap.clone(), Ordering::Less);
        let max_coords = find_extreme_coordinates(empty_hashmap, Ordering::Greater);

        // With empty hashmap, should return default values [0.0, 0.0, 0.0]
        assert_eq!(min_coords, [0.0, 0.0, 0.0]);
        assert_eq!(max_coords, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn utilities_find_extreme_coordinates_single_point() {
        let points = vec![Point::new([5.0, -3.0, 7.0])];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(hashmap.clone(), Ordering::Less);
        let max_coords = find_extreme_coordinates(hashmap, Ordering::Greater);

        // With single point, min and max should be the same
        assert_eq!(min_coords, [5.0, -3.0, 7.0]);
        assert_eq!(max_coords, [5.0, -3.0, 7.0]);
    }

    #[test]
    fn utilities_find_extreme_coordinates_equal_ordering() {
        let points = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0])];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices);

        // Using Ordering::Equal should return the first vertex's coordinates unchanged
        let coords = find_extreme_coordinates(hashmap, Ordering::Equal);
        // The first vertex in the iteration (order is not guaranteed in HashMap)
        // but the result should be one of the input coordinates
        assert!(coords == [1.0, 2.0, 3.0] || coords == [4.0, 5.0, 6.0]);
    }

    #[test]
    fn utilities_find_extreme_coordinates_2d() {
        let points = vec![
            Point::new([1.0, 4.0]),
            Point::new([3.0, 2.0]),
            Point::new([2.0, 5.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 2>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(hashmap.clone(), Ordering::Less);
        let max_coords = find_extreme_coordinates(hashmap, Ordering::Greater);

        assert_eq!(min_coords, [1.0, 2.0]);
        assert_eq!(max_coords, [3.0, 5.0]);
    }

    #[test]
    fn utilities_find_extreme_coordinates_1d() {
        let points = vec![Point::new([10.0]), Point::new([-5.0]), Point::new([3.0])];
        let vertices: Vec<Vertex<f64, Option<()>, 1>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(hashmap.clone(), Ordering::Less);
        let max_coords = find_extreme_coordinates(hashmap, Ordering::Greater);

        assert_eq!(min_coords, [-5.0]);
        assert_eq!(max_coords, [10.0]);
    }

    #[test]
    fn utilities_find_extreme_coordinates_with_typed_data() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, -1.0, 2.0]),
            Point::new([-2.0, 5.0, 1.0]),
        ];
        let vertices: Vec<Vertex<f64, i32, 3>> = points
            .into_iter()
            .enumerate()
            .map(|(i, point)| {
                use crate::delaunay_core::vertex::VertexBuilder;
                VertexBuilder::default()
                    .point(point)
                    .data(i as i32)
                    .build()
                    .unwrap()
            })
            .collect();
        let hashmap = Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(hashmap.clone(), Ordering::Less);
        let max_coords = find_extreme_coordinates(hashmap, Ordering::Greater);

        assert_eq!(min_coords, [-2.0, -1.0, 1.0]);
        assert_eq!(max_coords, [4.0, 5.0, 3.0]);
    }

    #[test]
    fn utilities_find_extreme_coordinates_identical_points() {
        let points = vec![
            Point::new([2.0, 3.0, 4.0]),
            Point::new([2.0, 3.0, 4.0]),
            Point::new([2.0, 3.0, 4.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(hashmap.clone(), Ordering::Less);
        let max_coords = find_extreme_coordinates(hashmap, Ordering::Greater);

        // All points are identical, so min and max should be the same
        assert_eq!(min_coords, [2.0, 3.0, 4.0]);
        assert_eq!(max_coords, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn utilities_find_extreme_coordinates_large_numbers() {
        let points = vec![
            Point::new([1e6, -1e6, 1e12]),
            Point::new([-1e9, 1e3, -1e15]),
            Point::new([1e15, 1e9, 1e6]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(hashmap.clone(), Ordering::Less);
        let max_coords = find_extreme_coordinates(hashmap, Ordering::Greater);

        assert_eq!(min_coords, [-1e9, -1e6, -1e15]);
        assert_eq!(max_coords, [1e15, 1e9, 1e12]);
    }

    #[test]
    fn utilities_make_uuid_uniqueness() {
        let uuid1 = make_uuid();
        let uuid2 = make_uuid();
        let uuid3 = make_uuid();

        // All UUIDs should be different
        assert_ne!(uuid1, uuid2);
        assert_ne!(uuid1, uuid3);
        assert_ne!(uuid2, uuid3);

        // All should be version 4
        assert_eq!(uuid1.get_version_num(), 4);
        assert_eq!(uuid2.get_version_num(), 4);
        assert_eq!(uuid3.get_version_num(), 4);
    }

    #[test]
    fn utilities_make_uuid_format() {
        let uuid = make_uuid();
        let uuid_string = uuid.to_string();

        // UUID should have proper format: 8-4-4-4-12 characters
        assert_eq!(uuid_string.len(), 36); // Including hyphens
        assert_eq!(uuid_string.chars().filter(|&c| c == '-').count(), 4);

        // Should be valid hyphenated format
        let parts: Vec<&str> = uuid_string.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
    }
}

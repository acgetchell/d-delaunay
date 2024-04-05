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
/// [Uuid] and the value is a [Vertex].
/// * `ordering`: The `ordering` parameter is of type [Ordering] and is used to
/// specify whether the function should find the minimum or maximum
/// coordinates. [Ordering] is an enum with three possible values: `Less`,
/// `Equal`, and `Greater`.
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
pub fn find_extreme_coordinates<T, U, const D: usize>(
    vertices: HashMap<Uuid, Vertex<T, U, D>>,
    ordering: Ordering,
) -> [T; D]
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
{
    let mut min_coords = [Default::default(); D];

    for vertex in vertices.values() {
        for (i, coord) in vertex.point.coords.iter().enumerate() {
            if coord.partial_cmp(&min_coords[i]) == Some(ordering) {
                min_coords[i] = *coord;
            }
        }
    }

    min_coords
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
}

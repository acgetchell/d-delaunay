//! Utility functions

use std::collections::HashMap;

use uuid::Uuid;

use super::vertex::Vertex;

/// The function `make_uuid` generates a version 4 UUID in Rust.
///
/// # Returns:
///
/// a randomly generated UUID (Universally Unique Identifier) using the `new_v4` method from the `Uuid`
/// struct.
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

/// The function `find_min_coordinate` takes a hashmap of vertices and returns an array containing the
/// minimum coordinate for each dimension.
///
/// Arguments:
///
/// * `vertices`: A HashMap containing vertices, where the key is a Uuid and the value is a Vertex
/// struct.
///
/// Returns:
///
/// an array of type `T` with length `D`.
pub fn find_min_coordinate<T, U, const D: usize>(vertices: HashMap<Uuid, Vertex<T, U, D>>) -> [T; D]
where
    T: Copy + Default + PartialOrd,
{
    let mut min_coords = [Default::default(); D];

    for vertex in vertices.values() {
        for (i, coord) in vertex.point.coords.iter().enumerate() {
            if *coord < min_coords[i] {
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
        let hashmap = Vertex::into_hashmap(vertices.clone());

        let min_coords = find_min_coordinate(hashmap);

        assert_eq!(min_coords, [-1.0, -5.0, -9.0]);

        // Human readable output for cargo test -- --nocapture
        println!("min_coords = {:?}", min_coords);
    }
}

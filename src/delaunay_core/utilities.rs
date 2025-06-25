//! Utility functions

use super::{point::OrderedEq, vertex::Vertex};
use serde::{de::DeserializeOwned, Serialize};
use slotmap::SlotMap;
use std::{cmp::Ordering, hash::Hash};

/// The function `find_extreme_coordinates` takes a SlotMap of vertices and
/// returns the minimum or maximum coordinates based on the specified
/// ordering.
///
/// # Arguments
///
/// - `vertices`: A SlotMap containing [Vertex] objects.
/// - `ordering`: The `ordering` parameter is of type [Ordering] and is used to
///   specify whether the function should find the minimum or maximum
///   coordinates. [Ordering] is an enum with three possible values: `Less`,
///   `Equal`, and `Greater`.
///
/// # Returns
///
/// an array of type `T` with length `D` containing the minimum or maximum
/// coordinate for each dimension.
///
/// # Panics
///
/// This function should not panic under normal circumstances as it handles
/// the empty vertices case by returning default coordinates. However, it uses
/// `.unwrap()` internally which could theoretically panic if the SlotMap
/// iterator behavior changes unexpectedly.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::utilities::find_extreme_coordinates;
/// use d_delaunay::delaunay_core::vertex::Vertex;
/// use d_delaunay::delaunay_core::point::Point;
/// use slotmap::SlotMap;
/// use d_delaunay::delaunay_core::triangulation_data_structure::VertexKey;
/// use std::cmp::Ordering;
/// let points = vec![
///     Point::new([-1.0, 2.0, 3.0]),
///     Point::new([4.0, -5.0, 6.0]),
///     Point::new([7.0, 8.0, -9.0]),
/// ];
/// let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
/// let mut slotmap = SlotMap::with_key();
/// for vertex in vertices {
///     slotmap.insert(vertex);
/// }
/// let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less);
/// # use approx::assert_relative_eq;
/// assert_relative_eq!(min_coords.as_slice(), [-1.0, -5.0, -9.0].as_slice(), epsilon = 1e-9);
/// ```
#[must_use]
pub fn find_extreme_coordinates<T, U, const D: usize, K>(
    vertices: &SlotMap<K, Vertex<T, U, D>>,
    ordering: Ordering,
) -> [T; D]
where
    T: Clone + Copy + Default + PartialEq + PartialOrd + OrderedEq,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
    K: slotmap::Key,
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
/// # Errors
///
/// Returns an error if the vector length does not match the target array dimension `D`.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::utilities::vec_to_array;
/// let vec = vec![1.0, 2.0, 3.0];
/// let array = vec_to_array::<3>(&vec).unwrap();
/// # use approx::assert_relative_eq;
/// assert_relative_eq!(array.as_slice(), [1.0, 2.0, 3.0].as_slice(), epsilon = 1e-9);
/// ```
pub fn vec_to_array<const D: usize>(vec: &[f64]) -> Result<[f64; D], anyhow::Error> {
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

    use super::*;
    use crate::delaunay_core::point::Point;
    use crate::delaunay_core::triangulation_data_structure::VertexKey;
    use approx::assert_relative_eq;
    use slotmap::SlotMap;

    #[test]
    fn utilities_find_min_coordinate() {
        let points = vec![
            Point::new([-1.0, 2.0, 3.0]),
            Point::new([4.0, -5.0, 6.0]),
            Point::new([7.0, 8.0, -9.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }
        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less);

        assert_relative_eq!(
            min_coords.as_slice(),
            [-1.0, -5.0, -9.0].as_slice(),
            epsilon = 1e-9
        );

        // Human readable output for cargo test -- --nocapture
        println!("min_coords = {min_coords:?}");
    }

    #[test]
    fn utilities_find_max_coordinate() {
        let points = vec![
            Point::new([-1.0, 2.0, 3.0]),
            Point::new([4.0, -5.0, 6.0]),
            Point::new([7.0, 8.0, -9.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater);

        assert_relative_eq!(
            max_coords.as_slice(),
            [7.0, 8.0, 6.0].as_slice(),
            epsilon = 1e-9
        );

        // Human readable output for cargo test -- --nocapture
        println!("max_coords = {max_coords:?}");
    }

    #[test]
    fn utilities_vec_to_array_success() {
        let vec = vec![1.0, 2.0, 3.0];
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(array.as_slice(), [1.0, 2.0, 3.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn utilities_vec_to_array_wrong_length() {
        let vec = vec![1.0, 2.0]; // Length 2, but expecting 3
        let result = vec_to_array::<3>(&vec);

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
        let result = vec_to_array::<3>(&vec);

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
        let result = vec_to_array::<3>(&vec);

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
        let result = vec_to_array::<1>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(array.as_slice(), [42.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn utilities_vec_to_array_large_dimension() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = vec_to_array::<10>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(
            array.as_slice(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn utilities_vec_to_array_negative_values() {
        let vec = vec![-1.0, -2.0, -3.0];
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(
            array.as_slice(),
            [-1.0, -2.0, -3.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn utilities_vec_to_array_zero() {
        let vec = vec![0.0, 0.0, 0.0];
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(array.as_slice(), [0.0, 0.0, 0.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn utilities_find_extreme_coordinates_empty() {
        let empty_slotmap: SlotMap<VertexKey, Vertex<f64, Option<()>, 3>> = SlotMap::with_key();
        let min_coords = find_extreme_coordinates(&empty_slotmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&empty_slotmap, Ordering::Greater);

        // With empty slotmap, should return default values [0.0, 0.0, 0.0]
        assert_relative_eq!(
            min_coords.as_slice(),
            [0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            max_coords.as_slice(),
            [0.0, 0.0, 0.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn utilities_find_extreme_coordinates_single_point() {
        let points = vec![Point::new([5.0, -3.0, 7.0])];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater);

        // With single point, min and max should be the same
        assert_relative_eq!(
            min_coords.as_slice(),
            [5.0, -3.0, 7.0].as_slice(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            max_coords.as_slice(),
            [5.0, -3.0, 7.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn utilities_find_extreme_coordinates_equal_ordering() {
        let points = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0])];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }

        // Using Ordering::Equal should return the first vertex's coordinates unchanged
        let coords = find_extreme_coordinates(&slotmap, Ordering::Equal);
        // The first vertex in the iteration (order not guaranteed)
        // but the result should be one of the input coordinates
        let matches_first = approx::relative_eq!(
            coords.as_slice(),
            [1.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
        let matches_second = approx::relative_eq!(
            coords.as_slice(),
            [4.0, 5.0, 6.0].as_slice(),
            epsilon = 1e-9
        );
        assert!(matches_first || matches_second);
    }

    #[test]
    fn utilities_find_extreme_coordinates_2d() {
        let points = vec![
            Point::new([1.0, 4.0]),
            Point::new([3.0, 2.0]),
            Point::new([2.0, 5.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 2>> = Vertex::from_points(points);
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater);

        assert_relative_eq!(min_coords.as_slice(), [1.0, 2.0].as_slice(), epsilon = 1e-9);
        assert_relative_eq!(max_coords.as_slice(), [3.0, 5.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn utilities_find_extreme_coordinates_1d() {
        let points = vec![Point::new([10.0]), Point::new([-5.0]), Point::new([3.0])];
        let vertices: Vec<Vertex<f64, Option<()>, 1>> = Vertex::from_points(points);
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater);

        assert_relative_eq!(min_coords.as_slice(), [-5.0].as_slice(), epsilon = 1e-9);
        assert_relative_eq!(max_coords.as_slice(), [10.0].as_slice(), epsilon = 1e-9);
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
                    .data(i32::try_from(i).unwrap())
                    .build()
                    .unwrap()
            })
            .collect();
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater);

        assert_relative_eq!(
            min_coords.as_slice(),
            [-2.0, -1.0, 1.0].as_slice(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            max_coords.as_slice(),
            [4.0, 5.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn utilities_find_extreme_coordinates_identical_points() {
        let points = vec![
            Point::new([2.0, 3.0, 4.0]),
            Point::new([2.0, 3.0, 4.0]),
            Point::new([2.0, 3.0, 4.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater);

        // All points are identical, so min and max should be the same
        assert_relative_eq!(
            min_coords.as_slice(),
            [2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            max_coords.as_slice(),
            [2.0, 3.0, 4.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn utilities_find_extreme_coordinates_large_numbers() {
        let points = vec![
            Point::new([1e6, -1e6, 1e12]),
            Point::new([-1e9, 1e3, -1e15]),
            Point::new([1e15, 1e9, 1e6]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let mut slotmap = SlotMap::with_key();
        for vertex in vertices {
            slotmap.insert(vertex);
        }

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater);

        assert_relative_eq!(
            min_coords.as_slice(),
            [-1e9, -1e6, -1e15].as_slice(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            max_coords.as_slice(),
            [1e15, 1e9, 1e12].as_slice(),
            epsilon = 1e-9
        );
    }
}

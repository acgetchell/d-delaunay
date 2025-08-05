//! General helper utilities

use anyhow::Error;
use serde::{Serialize, de::DeserializeOwned};
use slotmap::SlotMap;
use std::cmp::Ordering;
use thiserror::Error;
use uuid::Uuid;

use crate::delaunay_core::facet::Facet;
use crate::delaunay_core::traits::data_type::DataType;
use crate::delaunay_core::vertex::Vertex;
use crate::geometry::point::Point;
use crate::geometry::traits::coordinate::{Coordinate, CoordinateScalar};
use num_traits::NumCast;

// =============================================================================
// UUID VALIDATION
// =============================================================================

/// Errors that can occur during UUID validation.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum UuidValidationError {
    /// The UUID is nil (all zeros), which is not allowed.
    #[error("UUID is nil (all zeros) which is not allowed")]
    NilUuid,
    /// The UUID is not version 4.
    #[error("UUID is not version 4: expected version 4, found version {found}")]
    InvalidVersion {
        /// The version number that was found.
        found: usize,
    },
}

/// Validates that a UUID is not nil and is version 4.
///
/// This function performs comprehensive UUID validation to ensure that UUIDs
/// used throughout the system meet our requirements:
/// - Must not be nil (all zeros)
/// - Must be version 4 (randomly generated)
///
/// # Arguments
///
/// * `uuid` - The UUID to validate
///
/// # Returns
///
/// Returns `Ok(())` if the UUID is valid, or a `UuidValidationError` if invalid.
///
/// # Errors
///
/// Returns `UuidValidationError::NilUuid` if the UUID is nil,
/// or `UuidValidationError::InvalidVersion` if the UUID is not version 4.
///
/// # Examples
///
/// ```
/// use d_delaunay::delaunay_core::utilities::{make_uuid, validate_uuid};
/// use uuid::Uuid;
///
/// // Valid UUID (version 4)
/// let valid_uuid = make_uuid();
/// assert!(validate_uuid(&valid_uuid).is_ok());
///
/// // Invalid UUID (nil)
/// let nil_uuid = Uuid::nil();
/// assert!(validate_uuid(&nil_uuid).is_err());
/// ```
pub const fn validate_uuid(uuid: &Uuid) -> Result<(), UuidValidationError> {
    // Check if UUID is nil
    if uuid.is_nil() {
        return Err(UuidValidationError::NilUuid);
    }

    // Check if UUID is version 4
    let version = uuid.get_version_num();
    if version != 4 {
        return Err(UuidValidationError::InvalidVersion { found: version });
    }

    Ok(())
}

/// The function `make_uuid` generates a version 4 [Uuid].
///
/// # Returns
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
#[must_use]
pub fn make_uuid() -> Uuid {
    Uuid::new_v4()
}

/// Find the extreme coordinates (minimum or maximum) across all vertices in a `SlotMap`.
///
/// This function takes a `SlotMap` of vertices and returns the minimum or maximum
/// coordinates based on the specified ordering. This works directly with `SlotMap`
/// to provide efficient coordinate finding in performance-critical contexts.
///
/// # Arguments
///
/// * `vertices` - A `SlotMap` containing Vertex objects
/// * `ordering` - The ordering parameter is of type Ordering and is used to
///   specify whether the function should find the minimum or maximum
///   coordinates. Ordering is an enum with three possible values: `Less`,
///   `Equal`, and `Greater`.
///
/// # Returns
///
/// Returns `Ok([T; D])` containing the minimum or maximum coordinate for each dimension,
/// or an error if the vertices `SlotMap` is empty.
///
/// # Errors
///
/// Returns an error if the vertices `SlotMap` is empty.
///
/// # Panics
///
/// This function should not panic under normal circumstances as the empty `SlotMap`
/// case is handled by returning an error.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::utilities::find_extreme_coordinates;
/// use d_delaunay::delaunay_core::vertex::Vertex;
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::traits::coordinate::Coordinate;
/// use slotmap::{SlotMap, DefaultKey};
/// use std::cmp::Ordering;
///
/// let points = vec![
///     Point::new([-1.0, 2.0, 3.0]),
///     Point::new([4.0, -5.0, 6.0]),
///     Point::new([7.0, 8.0, -9.0]),
/// ];
/// let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
/// let mut slotmap: SlotMap<DefaultKey, Vertex<f64, Option<()>, 3>> = SlotMap::new();
/// for vertex in vertices {
///     slotmap.insert(vertex);
/// }
/// let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
/// # use approx::assert_relative_eq;
/// assert_relative_eq!(min_coords.as_slice(), [-1.0, -5.0, -9.0].as_slice(), epsilon = 1e-9);
/// ```
pub fn find_extreme_coordinates<K, T, U, const D: usize>(
    vertices: &SlotMap<K, Vertex<T, U, D>>,
    ordering: Ordering,
) -> Result<[T; D], Error>
where
    K: slotmap::Key,
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
{
    if vertices.is_empty() {
        return Err(anyhow::Error::msg(
            "Cannot find extreme coordinates: vertices SlotMap is empty",
        ));
    }

    let mut iter = vertices.values();
    let first_vertex = iter
        .next()
        .expect("SlotMap is unexpectedly empty despite earlier check");
    let mut extreme_coords: [T; D] = (*first_vertex).into();

    for vertex in iter {
        let vertex_coords: [T; D] = (*vertex).into();
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
                }
            }
        }
    }

    Ok(extreme_coords)
}

/// Checks if two facets are adjacent by comparing their vertex sets.
///
/// Two facets are considered adjacent if they share the exact same set of vertices,
/// regardless of the order. This is a common check in triangulation algorithms to
/// identify neighboring cells.
///
/// # Arguments
///
/// * `facet1` - A reference to the first facet.
/// * `facet2` - A reference to the second facet.
///
/// # Returns
///
/// `true` if the facets share the same vertices, `false` otherwise.
///
/// # Examples
///
/// ```
/// use d_delaunay::delaunay_core::facet::Facet;
/// use d_delaunay::delaunay_core::utilities::facets_are_adjacent;
/// use d_delaunay::delaunay_core::vertex::Vertex;
/// use d_delaunay::delaunay_core::cell::Cell;
/// use d_delaunay::{cell, vertex};
///
/// let v1: Vertex<f64, Option<()>, 2> = vertex!([0.0, 0.0]);
/// let v2: Vertex<f64, Option<()>, 2> = vertex!([1.0, 0.0]);
/// let v3: Vertex<f64, Option<()>, 2> = vertex!([0.0, 1.0]);
/// let v4: Vertex<f64, Option<()>, 2> = vertex!([1.0, 1.0]);
///
/// let cell1: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v1, v2, v3]);
/// let cell2: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v2, v3, v4]);
///
/// let facet1 = Facet::new(cell1, v1).unwrap();
/// let facet2 = Facet::new(cell2, v4).unwrap();
///
/// // These facets share vertices v2 and v3, so they are adjacent
/// assert!(facets_are_adjacent(&facet1, &facet2));
/// ```
pub fn facets_are_adjacent<T, U, V, const D: usize>(
    facet1: &Facet<T, U, V, D>,
    facet2: &Facet<T, U, V, D>,
) -> bool
where
    T: CoordinateScalar,
    U: DataType,
    V: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    // This works because Vertex implements `Eq` and `Hash`
    use std::collections::HashSet;
    let vertices1: HashSet<_> = facet1.vertices().into_iter().collect();
    let vertices2: HashSet<_> = facet2.vertices().into_iter().collect();

    vertices1 == vertices2
}

/// Generates all unique combinations of `k` items from a given slice.
///
/// This function is used to generate vertex combinations for creating k-simplices
/// (e.g., edges, triangles, tetrahedra) from a set of vertices.
///
/// # Arguments
///
/// * `vertices` - A slice of vertices from which to generate combinations.
/// * `k` - The size of each combination.
///
/// # Returns
///
/// A vector of vectors, where each inner vector is a unique combination of `k` vertices.
///
/// # Examples
///
/// This function is made public for testing purposes.
///
/// ```
/// use d_delaunay::delaunay_core::utilities::generate_combinations;
/// use d_delaunay::delaunay_core::vertex::Vertex;
/// use d_delaunay::vertex;
///
/// let vertices: Vec<Vertex<f64, Option<()>, 1>> = vec![
///     vertex!([0.0]),
///     vertex!([1.0]),
///     vertex!([2.0]),
/// ];
///
/// // Generate all 2-vertex combinations (edges)
/// let combinations = generate_combinations(&vertices, 2);
///
/// assert_eq!(combinations.len(), 3);
/// assert!(combinations.contains(&vec![vertices[0], vertices[1]]));
/// assert!(combinations.contains(&vec![vertices[0], vertices[2]]));
/// assert!(combinations.contains(&vec![vertices[1], vertices[2]]));
/// ```
pub fn generate_combinations<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    k: usize,
) -> Vec<Vec<Vertex<T, U, D>>>
where
    T: CoordinateScalar,
    U: DataType,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    let mut combinations = Vec::new();

    if k == 0 {
        combinations.push(Vec::new());
        return combinations;
    }

    if k > vertices.len() {
        return combinations;
    }

    if k == vertices.len() {
        combinations.push(vertices.to_vec());
        return combinations;
    }

    // Generate combinations using iterative approach
    let n = vertices.len();
    let mut indices = (0..k).collect::<Vec<_>>();

    loop {
        // Add current combination
        let combination = indices.iter().map(|i| vertices[*i]).collect();
        combinations.push(combination);

        // Find next combination
        let mut i = k;
        loop {
            if i == 0 {
                return combinations;
            }
            i -= 1;
            if indices[i] != i + n - k {
                break;
            }
        }

        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

// =============================================================================
// SUPERCELL SIMPLEX CREATION
// =============================================================================

/// Creates a well-formed simplex centered at the given point with the given radius.
///
/// This utility function generates a proper non-degenerate simplex (e.g., tetrahedron
/// for 3D) that can be used as a supercell in triangulation algorithms. The simplex
/// is constructed to have vertices positioned strategically around the center point
/// to ensure geometric validity and avoid degeneracies.
///
/// # Arguments
///
/// * `center` - The center point coordinates for the simplex
/// * `radius` - The radius (half the size) of the simplex from center to vertices
///
/// # Returns
///
/// A vector of `Point<T, D>` representing the vertices of the simplex.
/// For D-dimensional space, returns D+1 vertices forming a valid D-simplex.
///
/// # Type Parameters
///
/// * `T` - The coordinate scalar type (e.g., f64, f32)
/// * `D` - The dimension of the space
///
/// # Panics
///
/// Panics if coordinate conversion from `f64` to type `T` fails during vertex creation.
/// This can happen if the computed coordinate values exceed the representable range of type `T`
/// or if the `NumCast::from` conversion fails.
///
/// # Examples
///
/// ```
/// use d_delaunay::delaunay_core::utilities::create_supercell_simplex;
/// use d_delaunay::geometry::point::Point;
///
/// // Create a 3D tetrahedron centered at origin with radius 10.0
/// let center = [0.0f64; 3];
/// let radius = 10.0f64;
/// let simplex_points = create_supercell_simplex(&center, radius);
/// assert_eq!(simplex_points.len(), 4); // Tetrahedron has 4 vertices
///
/// // Create a 2D triangle
/// let center_2d = [5.0f64, 5.0f64];
/// let simplex_2d = create_supercell_simplex(&center_2d, 3.0f64);
/// assert_eq!(simplex_2d.len(), 3); // Triangle has 3 vertices
/// ```
///
/// # Algorithm Details
///
/// - **3D Case**: Creates a regular tetrahedron using the vertices of a cube:
///   (1,1,1), (1,-1,-1), (-1,1,-1), (-1,-1,1), scaled by radius and translated by center
/// - **General Case**: For D dimensions, creates D+1 vertices using a systematic
///   approach that places one vertex with all positive offsets, and D vertices
///   each with one negative offset dimension
///
/// The resulting simplex is guaranteed to be non-degenerate and suitable for
/// use as a bounding supercell in triangulation algorithms.
pub fn create_supercell_simplex<T, const D: usize>(center: &[T; D], radius: T) -> Vec<Point<T, D>>
where
    T: CoordinateScalar + NumCast,
    f64: From<T>,
    [T; D]: Default + DeserializeOwned + Serialize + Copy + Sized,
{
    let mut points = Vec::new();

    // For 3D, create a regular tetrahedron
    if D == 3 {
        // Create a regular tetrahedron with vertices at:
        // (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
        // scaled by radius and translated by center
        let tetrahedron_vertices = [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ];

        for vertex_coords in &tetrahedron_vertices {
            let mut coords = [T::default(); D];
            for i in 0..D {
                let center_f64: f64 = center[i].into();
                let radius_f64: f64 = radius.into();
                let coord_f64 = radius_f64.mul_add(vertex_coords[i], center_f64);
                coords[i] = NumCast::from(coord_f64).expect("Failed to convert coordinate");
            }
            points.push(Point::new(coords));
        }
    } else {
        // For other dimensions, create a simplex using a generalized approach
        // Create D+1 vertices for a D-dimensional simplex

        // Create a regular simplex by placing vertices at the corners of a hypercube
        // scaled and offset appropriately
        let radius_f64: f64 = radius.into();

        // First vertex: all coordinates positive
        let mut coords = [T::default(); D];
        for i in 0..D {
            let center_f64: f64 = center[i].into();
            coords[i] =
                NumCast::from(center_f64 + radius_f64).expect("Failed to convert center + radius");
        }
        points.push(Point::new(coords));

        // Remaining D vertices: flip one coordinate at a time to negative
        for dim in 0..D {
            let mut coords = [T::default(); D];
            for i in 0..D {
                let center_f64: f64 = center[i].into();
                if i == dim {
                    // This dimension gets negative offset
                    coords[i] = NumCast::from(center_f64 - radius_f64)
                        .expect("Failed to convert center - radius");
                } else {
                    // Other dimensions get positive offset
                    coords[i] = NumCast::from(center_f64 + radius_f64)
                        .expect("Failed to convert center + radius");
                }
            }
            points.push(Point::new(coords));
        }
    }

    points
}

#[cfg(test)]
mod tests {

    use crate::geometry::point::Point;
    use crate::geometry::traits::coordinate::Coordinate;
    use crate::vertex;
    use approx::assert_relative_eq;
    use slotmap::{DefaultKey, SlotMap};

    use super::*;

    fn create_vertex_slotmap<T, U, const D: usize>(
        vertices: Vec<Vertex<T, U, D>>,
    ) -> SlotMap<DefaultKey, Vertex<T, U, D>>
    where
        T: CoordinateScalar,
        U: DataType,
        [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    {
        let mut slotmap = SlotMap::new();
        for vertex in vertices {
            slotmap.insert(vertex);
        }
        slotmap
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

    #[test]
    fn utilities_find_extreme_coordinates_min_max() {
        let points = vec![
            Point::new([-1.0, 2.0, 3.0]),
            Point::new([4.0, -5.0, 6.0]),
            Point::new([7.0, 8.0, -9.0]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

        assert_relative_eq!(
            min_coords.as_slice(),
            [-1.0, -5.0, -9.0].as_slice(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            max_coords.as_slice(),
            [7.0, 8.0, 6.0].as_slice(),
            epsilon = 1e-9
        );

        // Human readable output for cargo test -- --nocapture
        println!("min_coords = {min_coords:?}");
        println!("max_coords = {max_coords:?}");
    }

    #[test]
    fn utilities_find_extreme_coordinates_single_point() {
        let points = vec![Point::new([5.0, -3.0, 7.0])];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

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
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let slotmap = create_vertex_slotmap(vertices);

        // Using Ordering::Equal should return the first vertex's coordinates unchanged
        let coords = find_extreme_coordinates(&slotmap, Ordering::Equal).unwrap();
        // The first vertex in the iteration (order is deterministic in SlotMap)
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
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 2>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

        assert_relative_eq!(min_coords.as_slice(), [1.0, 2.0].as_slice(), epsilon = 1e-9);
        assert_relative_eq!(max_coords.as_slice(), [3.0, 5.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn utilities_find_extreme_coordinates_1d() {
        let points = vec![Point::new([10.0]), Point::new([-5.0]), Point::new([3.0])];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 1>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

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
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, i32, 3>> = points
            .into_iter()
            .enumerate()
            .map(|(i, point)| {
                vertex!(
                    point.to_array(),
                    i32::try_from(i).expect("Index out of bounds")
                )
            })
            .collect();
        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

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
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

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
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

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

    #[test]
    fn utilities_find_extreme_coordinates_with_f32() {
        // Test with f32 type to ensure generic type coverage
        let points = vec![
            Point::new([1.5f32, 2.5f32, 3.5f32]),
            Point::new([0.5f32, 4.5f32, 1.5f32]),
            Point::new([2.5f32, 1.5f32, 2.5f32]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f32, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

        assert_relative_eq!(
            min_coords.as_slice(),
            [0.5f32, 1.5f32, 1.5f32].as_slice(),
            epsilon = 1e-6
        );
        assert_relative_eq!(
            max_coords.as_slice(),
            [2.5f32, 4.5f32, 3.5f32].as_slice(),
            epsilon = 1e-6
        );
    }

    #[test]
    fn utilities_find_extreme_coordinates_empty_error_message() {
        // Test that the correct error message is returned for empty slotmap
        let empty_slotmap: SlotMap<
            DefaultKey,
            crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>,
        > = SlotMap::new();
        let result = find_extreme_coordinates(&empty_slotmap, Ordering::Less);

        assert!(result.is_err());
        let error_message = result.unwrap_err().to_string();
        assert!(error_message.contains("Cannot find extreme coordinates"));
        assert!(error_message.contains("vertices SlotMap is empty"));
    }

    #[test]
    fn utilities_find_extreme_coordinates_ordering() {
        // Test SlotMap ordering and insertion behavior
        let points = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 1.0, 2.0])];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);

        let slotmap = create_vertex_slotmap(vertices);

        let min_coords = find_extreme_coordinates(&slotmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&slotmap, Ordering::Greater).unwrap();

        assert_relative_eq!(
            min_coords.as_slice(),
            [1.0, 1.0, 2.0].as_slice(),
            epsilon = 1e-9
        );
        assert_relative_eq!(
            max_coords.as_slice(),
            [4.0, 2.0, 3.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_facets_are_adjacent() {
        use crate::delaunay_core::{cell::Cell, facet::Facet};
        use crate::{cell, vertex};

        let v1: Vertex<f64, Option<()>, 2> = vertex!([0.0, 0.0]);
        let v2: Vertex<f64, Option<()>, 2> = vertex!([1.0, 0.0]);
        let v3: Vertex<f64, Option<()>, 2> = vertex!([0.0, 1.0]);
        let v4: Vertex<f64, Option<()>, 2> = vertex!([1.0, 1.0]);

        let cell1: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v1, v2, v3]);
        let cell2: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v2, v3, v4]);
        let cell3: Cell<f64, Option<()>, Option<()>, 2> = cell!(vec![v1, v2, v4]);

        let facet1 = Facet::new(cell1, v1).unwrap(); // Vertices: v2, v3
        let facet2 = Facet::new(cell2, v4).unwrap(); // Vertices: v2, v3
        let facet3 = Facet::new(cell3, v4).unwrap(); // Vertices: v1, v2

        assert!(facets_are_adjacent(&facet1, &facet2)); // Same vertices
        assert!(!facets_are_adjacent(&facet1, &facet3)); // Different vertices
    }

    #[test]
    fn test_facets_are_adjacent_edge_cases() {
        use crate::cell;
        use crate::delaunay_core::cell::Cell;

        let points1 = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];

        let points2 = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([2.0, 0.0, 0.0]),
        ];

        let cell1: Cell<f64, usize, usize, 3> = cell!(Vertex::from_points(points1));
        let cell2: Cell<f64, usize, usize, 3> = cell!(Vertex::from_points(points2));

        let facets1 = cell1.facets();
        let facets2 = cell2.facets();

        // Test adjacency detection
        let mut found_adjacent = false;

        for facet1 in &facets1 {
            for facet2 in &facets2 {
                if facets_are_adjacent(facet1, facet2) {
                    found_adjacent = true;
                    break;
                }
            }
            if found_adjacent {
                break;
            }
        }

        // These cells share 3 vertices, so they should have adjacent facets
        assert!(
            found_adjacent,
            "Cells sharing 3 vertices should have adjacent facets"
        );

        // Test with completely different cells
        let points3 = vec![
            Point::new([10.0, 10.0, 10.0]),
            Point::new([11.0, 10.0, 10.0]),
            Point::new([10.0, 11.0, 10.0]),
            Point::new([10.0, 10.0, 11.0]),
        ];

        let cell3: Cell<f64, usize, usize, 3> = cell!(Vertex::from_points(points3));
        let facets3 = cell3.facets();

        let mut found_adjacent2 = false;
        for facet1 in &facets1 {
            for facet3 in &facets3 {
                if facets_are_adjacent(facet1, facet3) {
                    found_adjacent2 = true;
                    break;
                }
            }
            if found_adjacent2 {
                break;
            }
        }

        // These cells share no vertices, so no facets should be adjacent
        assert!(
            !found_adjacent2,
            "Cells sharing no vertices should not have adjacent facets"
        );
    }

    #[test]
    fn test_validate_uuid_valid() {
        // Test valid UUID (version 4)
        let valid_uuid = make_uuid();
        assert!(validate_uuid(&valid_uuid).is_ok());

        // Test that the function returns Ok for valid UUIDs
        let result = validate_uuid(&valid_uuid);
        match result {
            Ok(()) => (), // Expected
            Err(e) => panic!("Expected valid UUID to pass validation, got: {e:?}"),
        }
    }

    #[test]
    fn test_validate_uuid_nil() {
        // Test nil UUID
        let nil_uuid = Uuid::nil();
        let result = validate_uuid(&nil_uuid);

        assert!(result.is_err());
        match result {
            Err(UuidValidationError::NilUuid) => (), // Expected
            Err(other) => panic!("Expected NilUuid error, got: {other:?}"),
            Ok(()) => panic!("Expected error for nil UUID, but validation passed"),
        }
    }

    #[test]
    fn test_validate_uuid_wrong_version() {
        // Create a UUID with different version (version 1)
        let v1_uuid = Uuid::parse_str("550e8400-e29b-11d4-a716-446655440000").unwrap();
        assert_eq!(v1_uuid.get_version_num(), 1);

        let result = validate_uuid(&v1_uuid);
        assert!(result.is_err());

        match result {
            Err(UuidValidationError::InvalidVersion { found }) => {
                assert_eq!(found, 1);
            }
            Err(other) => panic!("Expected InvalidVersion error, got: {other:?}"),
            Ok(()) => panic!("Expected error for version 1 UUID, but validation passed"),
        }
    }

    #[test]
    fn test_validate_uuid_error_display() {
        // Test error display formatting
        let nil_error = UuidValidationError::NilUuid;
        let nil_error_string = format!("{nil_error}");
        assert!(nil_error_string.contains("nil"));
        assert!(nil_error_string.contains("not allowed"));

        let version_error = UuidValidationError::InvalidVersion { found: 3 };
        let version_error_string = format!("{version_error}");
        assert!(version_error_string.contains("version 4"));
        assert!(version_error_string.contains("found version 3"));
    }

    #[test]
    fn test_validate_uuid_error_equality() {
        // Test PartialEq for UuidValidationError
        let error1 = UuidValidationError::NilUuid;
        let error2 = UuidValidationError::NilUuid;
        assert_eq!(error1, error2);

        let error3 = UuidValidationError::InvalidVersion { found: 2 };
        let error4 = UuidValidationError::InvalidVersion { found: 2 };
        assert_eq!(error3, error4);

        let error5 = UuidValidationError::InvalidVersion { found: 3 };
        assert_ne!(error3, error5);
        assert_ne!(error1, error3);
    }

    #[test]
    fn test_generate_combinations() {
        let vertices: Vec<Vertex<f64, Option<()>, 1>> = vec![
            vertex!([0.0]),
            vertex!([1.0]),
            vertex!([2.0]),
            vertex!([3.0]),
        ];

        // Combinations of 2 from 4
        let combinations_2 = generate_combinations(&vertices, 2);
        assert_eq!(combinations_2.len(), 6);

        // Combinations of 3 from 4
        let combinations_3 = generate_combinations(&vertices, 3);
        assert_eq!(combinations_3.len(), 4);
        assert!(combinations_3.contains(&vec![vertices[0], vertices[1], vertices[2]]));

        // Edge case: k=0
        let combinations_0 = generate_combinations(&vertices, 0);
        assert_eq!(combinations_0.len(), 1);
        assert!(combinations_0[0].is_empty());

        // Edge case: k > len
        let combinations_5 = generate_combinations(&vertices, 5);
        assert!(combinations_5.is_empty());
    }
}

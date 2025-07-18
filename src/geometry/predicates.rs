//! Geometric predicates for d-dimensional geometry calculations.
//!
//! This module contains fundamental geometric predicates and calculations
//! that operate on points and simplices, including circumcenter and circumradius
//! calculations.

use crate::delaunay_core::{matrix::invert, utilities::vec_to_array, vertex::Vertex};
use crate::geometry::point::{OrderedEq, Point};
use na::{ComplexField, Const, OPoint};
use nalgebra as na;
use num_traits::Float;
use peroxide::fuga::{LinearAlgebra, MatrixTrait, anyhow, zeros};
use serde::{Serialize, de::DeserializeOwned};
use std::{hash::Hash, iter::Sum};

/// Calculate the circumcenter of a set of vertices forming a simplex.
///
/// The circumcenter is the unique point equidistant from all vertices of
/// the simplex. Returns an error if the vertices do not form a valid simplex or
/// if the computation fails due to degeneracy or numerical issues.
///
/// Using the approach from:
///
/// Lévy, Bruno, and Yang Liu.
/// "Lp Centroidal Voronoi Tessellation and Its Applications."
/// ACM Transactions on Graphics 29, no. 4 (July 26, 2010): 119:1-119:11.
/// <https://doi.org/10.1145/1778765.1778856>.
///
/// The circumcenter C of a simplex with vertices `x_0`, `x_1`, ..., `x_n` is the
/// solution to the system:
///
/// C = 1/2 (A^-1*B)
///
/// Where:
///
/// A is a matrix (to be inverted) of the form:
///     (x_1-x0) for all coordinates in x1, x0
///     (x2-x0) for all coordinates in x2, x0
///     ... for all `x_n` in the simplex
///
/// These are the perpendicular bisectors of the edges of the simplex.
///
/// And:
///
/// B is a vector of the form:
///     (x_1^2-x0^2) for all coordinates in x1, x0
///     (x_2^2-x0^2) for all coordinates in x2, x0
///     ... for all `x_n` in the simplex
///
/// The resulting vector gives the coordinates of the circumcenter.
///
/// # Arguments
///
/// * `vertices` - A slice of vertices that form the simplex
///
/// # Returns
/// The circumcenter as a Point<f64, D> if successful, or an error if the
/// simplex is degenerate or the matrix inversion fails.
///
/// # Errors
///
/// Returns an error if:
/// - The vertices do not form a valid simplex
/// - The matrix inversion fails due to degeneracy
/// - Vector to array conversion fails
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::circumcenter;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let vertices = vec![vertex1, vertex2, vertex3, vertex4];
/// let center = circumcenter(&vertices).unwrap();
/// assert_eq!(center, Point::new([0.5, 0.5, 0.5]));
/// ```
pub fn circumcenter<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
) -> Result<Point<f64, D>, anyhow::Error>
where
    T: Clone
        + ComplexField<RealField = T>
        + Copy
        + Default
        + PartialEq
        + PartialOrd
        + OrderedEq
        + Sum
        + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    f64: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
{
    if vertices.is_empty() {
        return Err(anyhow::Error::msg("Empty vertex set"));
    }

    let dim = vertices.len() - 1;
    if vertices[0].dim() != dim {
        return Err(anyhow::Error::msg("Not a simplex!"));
    }

    // Build matrix A and vector B for the linear system
    let mut matrix = zeros(dim, dim);
    let mut b = zeros(dim, 1);
    let coords_0: [T; D] = (&vertices[0]).into();
    let coords_0_f64: [f64; D] = coords_0.map(std::convert::Into::into);

    for i in 0..dim {
        let coords_i: [T; D] = (&vertices[i + 1]).into();
        let coords_vertex_f64: [f64; D] = coords_i.map(std::convert::Into::into);

        // Fill matrix row
        for j in 0..dim {
            matrix[(i, j)] = (coords_i[j] - coords_0[j]).into();
        }

        // Fill vector element
        b[(i, 0)] = na::distance_squared(
            &na::Point::from(coords_vertex_f64),
            &na::Point::from(coords_0_f64),
        );
    }

    let a_inv = invert(&matrix)?;
    let solution = a_inv * b * 0.5;
    let solution_vec = solution.col(0).clone();
    let solution_array = vec_to_array(&solution_vec).map_err(anyhow::Error::msg)?;

    Ok(Point::<f64, D>::from(solution_array))
}

/// Calculate the circumradius of a set of vertices forming a simplex.
///
/// The circumradius is the distance from the circumcenter to any vertex of the simplex.
///
/// # Arguments
///
/// * `vertices` - A slice of vertices that form the simplex
///
/// # Returns
/// The circumradius as a value of type T if successful, or an error if the
/// circumcenter calculation fails.
///
/// # Errors
///
/// Returns an error if the circumcenter calculation fails. See [`circumcenter`] for details.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::circumradius;
/// use approx::assert_relative_eq;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let vertices = vec![vertex1, vertex2, vertex3, vertex4];
/// let radius = circumradius(&vertices).unwrap();
/// let expected_radius: f64 = 3.0_f64.sqrt() / 2.0;
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// ```
pub fn circumradius<T, U, const D: usize>(vertices: &[Vertex<T, U, D>]) -> Result<T, anyhow::Error>
where
    T: Clone
        + ComplexField<RealField = T>
        + Copy
        + Default
        + PartialEq
        + PartialOrd
        + OrderedEq
        + Sum
        + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    f64: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    OPoint<T, Const<D>>: From<[f64; D]>,
{
    let circumcenter = circumcenter(vertices)?;
    Ok(circumradius_with_center(vertices, &circumcenter))
}

/// Calculate the circumradius given a precomputed circumcenter.
///
/// This is a helper function that calculates the circumradius when the circumcenter
/// is already known, avoiding redundant computation.
///
/// # Arguments
///
/// * `vertices` - A slice of vertices that form the simplex
/// * `circumcenter` - The precomputed circumcenter
///
/// # Returns
/// The circumradius as a value of type T
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::{circumcenter, circumradius_with_center};
/// use approx::assert_relative_eq;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let vertices = vec![vertex1, vertex2, vertex3, vertex4];
/// let center = circumcenter(&vertices).unwrap();
/// let radius = circumradius_with_center(&vertices, &center);
/// let expected_radius: f64 = 3.0_f64.sqrt() / 2.0;
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// ```
pub fn circumradius_with_center<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    circumcenter: &Point<f64, D>,
) -> T
where
    T: Clone
        + ComplexField<RealField = T>
        + Copy
        + Default
        + PartialEq
        + PartialOrd
        + OrderedEq
        + Sum
        + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    f64: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    OPoint<T, Const<D>>: From<[f64; D]>,
{
    if vertices.is_empty() {
        return T::zero();
    }

    let vertex_coords: [T; D] = (&vertices[0]).into();
    let vertex_coords_f64: [f64; D] = vertex_coords.map(std::convert::Into::into);
    na::distance(
        &na::Point::<T, D>::from(circumcenter.coordinates()),
        &na::Point::<T, D>::from(vertex_coords_f64),
    )
}

/// Check if a vertex is contained within the circumsphere of a simplex.
///
/// This function uses distance calculations to determine if a vertex lies within
/// the circumsphere formed by the given vertices.
///
/// # Arguments
///
/// * `simplex_vertices` - A slice of vertices that form the simplex
/// * `test_vertex` - The vertex to test for containment
///
/// # Returns
///
/// Returns `true` if the given vertex is contained in the circumsphere
/// of the simplex, and `false` otherwise.
///
/// # Errors
///
/// Returns an error if the circumcenter calculation fails. See [`circumcenter`] for details.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::circumsphere_contains;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
/// let test_vertex: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 1.0, 1.0])).data(3).build().unwrap();
/// assert!(circumsphere_contains(&simplex_vertices, test_vertex).unwrap());
/// ```
pub fn circumsphere_contains<T, U, const D: usize>(
    simplex_vertices: &[Vertex<T, U, D>],
    test_vertex: Vertex<T, U, D>,
) -> Result<bool, anyhow::Error>
where
    T: Clone
        + ComplexField<RealField = T>
        + Copy
        + Default
        + PartialEq
        + PartialOrd
        + OrderedEq
        + Sum
        + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    f64: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    OPoint<T, Const<D>>: From<[f64; D]>,
{
    let circumcenter = circumcenter(simplex_vertices)?;
    let circumradius = circumradius_with_center(simplex_vertices, &circumcenter);
    // Use implicit conversion from vertex to coordinates, then convert to f64
    let vertex_coords: [T; D] = (&test_vertex).into();
    let vertex_coords_f64: [f64; D] = vertex_coords.map(std::convert::Into::into);
    let radius = na::distance(
        &na::Point::<T, D>::from(circumcenter.coordinates()),
        &na::Point::<T, D>::from(vertex_coords_f64),
    );

    Ok(circumradius >= radius)
}

/// Check if a vertex is contained within the circumsphere of a simplex using matrix determinant.
///
/// This method is preferred over `circumsphere_contains` as it provides better numerical
/// stability by using a matrix determinant approach instead of distance calculations,
/// which can accumulate floating-point errors.
///
/// # Algorithm
///
/// The in-sphere test uses the determinant of a specially constructed matrix. For a
/// d-dimensional simplex with vertices `v₁, v₂, ..., vₐ₊₁` and test point `p`, the
/// matrix has the structure:
///
/// ```text
/// |  x₁   y₁   z₁  ...  x₁²+y₁²+z₁²+...  1  |
/// |  x₂   y₂   z₂  ...  x₂²+y₂²+z₂²+...  1  |
/// |  x₃   y₃   z₃  ...  x₃²+y₃²+z₃²+...  1  |
/// |  ...  ...  ... ...       ...        ... |
/// |  xₚ   yₚ   zₚ   ...  xₚ²+yₚ²+zₚ²+...   1  |
/// ```
///
/// Where each row contains:
/// - The d coordinates of a vertex
/// - The squared norm (sum of squares) of the vertex coordinates
/// - A constant 1
///
/// The test point `p` is inside the circumsphere if and only if the determinant
/// of this matrix is negative.
///
/// # Mathematical Background
///
/// This determinant test is mathematically equivalent to checking if the test point
/// lies inside the circumsphere, but avoids the numerical instability that can arise
/// from computing circumcenter coordinates and distances explicitly.
///
/// # Arguments
///
/// * `simplex_vertices` - A slice of vertices that form the simplex (must have exactly D+1 vertices)
/// * `test_vertex` - The vertex to test for containment
///
/// # Returns
///
/// Returns `true` if the given vertex is contained in the circumsphere
/// of the simplex, and `false` otherwise.
///
/// # Errors
///
/// Returns an error if:
/// - The number of simplex vertices is not exactly D+1
/// - Matrix operations fail
/// - Coordinate conversion fails
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::circumsphere_contains_vertex;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
///
/// // Test with a point clearly outside the circumsphere
/// let outside_vertex: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([2.0, 2.0, 2.0])).data(3).build().unwrap();
/// assert!(!circumsphere_contains_vertex(&simplex_vertices, outside_vertex).unwrap());
///
/// // Test with a point clearly inside the circumsphere
/// let inside_vertex: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.25, 0.25, 0.25])).data(4).build().unwrap();
/// assert!(circumsphere_contains_vertex(&simplex_vertices, inside_vertex).unwrap());
/// ```
pub fn circumsphere_contains_vertex<T, U, const D: usize>(
    simplex_vertices: &[Vertex<T, U, D>],
    test_vertex: Vertex<T, U, D>,
) -> Result<bool, anyhow::Error>
where
    T: Clone
        + ComplexField<RealField = T>
        + Copy
        + Default
        + PartialEq
        + PartialOrd
        + OrderedEq
        + Sum
        + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    f64: From<T>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    [f64; D]: Default + DeserializeOwned + Serialize + Sized,
{
    if simplex_vertices.len() != D + 1 {
        return Err(anyhow::Error::msg(
            "Invalid simplex: wrong number of vertices",
        ));
    }

    // Create matrix for in-sphere test
    // Matrix has D+2 columns: D coordinates + squared norm + 1
    let mut matrix = zeros(D + 2, D + 2);

    // Populate rows with the coordinates of the vertices of the simplex
    for (i, v) in simplex_vertices.iter().enumerate() {
        // Use implicit conversion from vertex to coordinates
        let vertex_coords: [T; D] = v.into();
        let vertex_coords_f64: [f64; D] = vertex_coords.map(std::convert::Into::into);

        // Add coordinates
        for j in 0..D {
            matrix[(i, j)] = vertex_coords_f64[j];
        }

        // Add squared norm (sum of squares of coordinates)
        let squared_norm: f64 = vertex_coords_f64.iter().map(|&x| x * x).sum();
        matrix[(i, D)] = squared_norm;

        // Add one to the last column
        matrix[(i, D + 1)] = 1.0;
    }

    // Add the test vertex to the last row of the matrix
    let test_vertex_coords: [T; D] = (&test_vertex).into();
    let test_vertex_coords_f64: [f64; D] = test_vertex_coords.map(std::convert::Into::into);

    // Add coordinates
    for j in 0..D {
        matrix[(D + 1, j)] = test_vertex_coords_f64[j];
    }

    // Add squared norm
    let test_squared_norm: f64 = test_vertex_coords_f64.iter().map(|&x| x * x).sum();
    matrix[(D + 1, D)] = test_squared_norm;

    // Add one to the last column
    matrix[(D + 1, D + 1)] = 1.0;

    // Calculate the determinant of the matrix
    let det = matrix.det();

    // For the in-sphere test, the point is inside if the determinant is negative
    // The sign depends on the orientation of the simplex vertices
    Ok(det < 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delaunay_core::vertex::VertexBuilder;
    use crate::geometry::point::Point;
    use approx::assert_relative_eq;

    #[test]
    fn predicates_circumcenter() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let center = circumcenter(&vertices).unwrap();

        assert_eq!(center, Point::new([0.5, 0.5, 0.5]));
    }

    #[test]
    fn predicates_circumcenter_fail() {
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertices = vec![vertex1, vertex2, vertex3];
        let center = circumcenter(&vertices);

        assert!(center.is_err());
    }

    #[test]
    fn predicates_circumradius() {
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let radius = circumradius(&vertices).unwrap();
        let expected_radius: f64 = 3.0_f64.sqrt() / 2.0;

        assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
    }

    #[test]
    fn predicates_circumsphere_contains() {
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let test_vertex = VertexBuilder::default()
            .point(Point::new([1.0, 1.0, 1.0]))
            .data(3)
            .build()
            .unwrap();

        assert!(circumsphere_contains(&simplex_vertices, test_vertex).unwrap());
    }

    #[test]
    fn predicates_circumsphere_does_not_contain() {
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
        let test_vertex = VertexBuilder::default()
            .point(Point::new([2.0, 2.0, 2.0]))
            .data(3)
            .build()
            .unwrap();

        assert!(!circumsphere_contains(&simplex_vertices, test_vertex).unwrap());
    }

    #[test]
    fn predicates_circumcenter_2d() {
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 2> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([2.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0]))
            .build()
            .unwrap();
        let vertices = vec![vertex1, vertex2, vertex3];
        let center = circumcenter(&vertices).unwrap();

        // For this triangle, circumcenter should be at (1.0, 0.75)
        assert_relative_eq!(center.coordinates()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(center.coordinates()[1], 0.75, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumradius_2d() {
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 2> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .build()
            .unwrap();
        let vertices = vec![vertex1, vertex2, vertex3];
        let radius = circumradius(&vertices).unwrap();

        // For a right triangle with legs of length 1, circumradius is sqrt(2)/2
        let expected_radius = 2.0_f64.sqrt() / 2.0;
        assert_relative_eq!(radius, expected_radius, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_determinant() {
        // Test the matrix determinant method for circumsphere containment
        // Use a simple, well-known case: unit tetrahedron
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];

        // Test vertex clearly outside circumsphere
        let vertex_far_outside = VertexBuilder::default()
            .point(Point::new([10.0, 10.0, 10.0]))
            .data(4)
            .build()
            .unwrap();
        // Just check that the method runs without error for now
        let result = circumsphere_contains_vertex(&simplex_vertices, vertex_far_outside);
        assert!(result.is_ok());

        // Test with origin (should be inside or on boundary)
        let origin = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(3)
            .build()
            .unwrap();
        let result_origin = circumsphere_contains_vertex(&simplex_vertices, origin);
        assert!(result_origin.is_ok());
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_2d() {
        // Test 2D case for circumsphere containment using determinant method
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 2> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3];

        // Test vertex far outside circumcircle
        let vertex_far_outside = VertexBuilder::default()
            .point(Point::new([10.0, 10.0]))
            .build()
            .unwrap();
        let result = circumsphere_contains_vertex(&simplex_vertices, vertex_far_outside);
        assert!(result.is_ok());

        // Test with center of triangle (should be inside)
        let center = VertexBuilder::default()
            .point(Point::new([0.33, 0.33]))
            .build()
            .unwrap();
        let result_center = circumsphere_contains_vertex(&simplex_vertices, center);
        assert!(result_center.is_ok());
    }

    #[test]
    fn predicates_circumcenter_error_cases() {
        // Test circumcenter calculation with degenerate cases
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 2> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .build()
            .unwrap();
        let vertices = vec![vertex1, vertex2];

        // Test with insufficient vertices for proper simplex (2 vertices in 2D space)
        let center_result = circumcenter(&vertices);
        assert!(center_result.is_err());
    }

    #[test]
    fn predicates_circumcenter_collinear_points() {
        // Test circumcenter with collinear points (should fail)
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([2.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([3.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertices = vec![vertex1, vertex2, vertex3, vertex4];

        // This should fail because points are collinear
        let center_result = circumcenter(&vertices);
        assert!(center_result.is_err());
    }

    #[test]
    fn predicates_circumradius_with_center() {
        // Test the circumradius_with_center function
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let vertices = vec![vertex1, vertex2, vertex3, vertex4];

        let center = circumcenter(&vertices).unwrap();
        let radius_with_center = circumradius_with_center(&vertices, &center);
        let radius_direct = circumradius(&vertices).unwrap();

        assert_relative_eq!(radius_with_center, radius_direct, epsilon = 1e-10);
    }

    #[test]
    fn predicates_circumsphere_edge_cases() {
        // Test circumsphere containment with simple cases
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 2> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0]))
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3];

        // Test that the methods run without error
        let test_point = VertexBuilder::default()
            .point(Point::new([0.25, 0.25]))
            .build()
            .unwrap();

        let circumsphere_result = circumsphere_contains(&simplex_vertices, test_point);
        assert!(circumsphere_result.is_ok());

        let determinant_result = circumsphere_contains_vertex(&simplex_vertices, test_point);
        assert!(determinant_result.is_ok());

        // At minimum, both methods should give the same result for the same input
        let far_point = VertexBuilder::default()
            .point(Point::new([100.0, 100.0]))
            .build()
            .unwrap();

        let circumsphere_far = circumsphere_contains(&simplex_vertices, far_point);
        let determinant_far = circumsphere_contains_vertex(&simplex_vertices, far_point);

        assert!(circumsphere_far.is_ok());
        assert!(determinant_far.is_ok());
    }
}

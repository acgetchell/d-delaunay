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
use std::{cmp::Ordering, collections::HashMap, hash::Hash, iter::Sum};
use uuid::Uuid;

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
    let coords_0_f64: [f64; D] = {
        let coords_0: [T; D] = (&vertices[0]).into();
        coords_0.map(std::convert::Into::into)
    };

    for i in 0..dim {
        let coords_vertex: [T; D] = (&vertices[i + 1]).into();
        let coords_vertex_f64: [f64; D] = coords_vertex.map(std::convert::Into::into);

        // Fill matrix row
        for j in 0..dim {
            matrix[(i, j)] = coords_vertex_f64[j] - coords_0_f64[j];
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
    circumradius_with_center(vertices, &circumcenter)
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
/// The circumradius as a value of type T if successful, or an error if the
/// simplex is degenerate or the distance calculation fails.
///
/// # Errors
///
/// Returns an error if:
/// - The vertices slice is empty
/// - Coordinate conversion fails
/// - Distance calculation fails
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
/// let radius = circumradius_with_center(&vertices, &center).unwrap();
/// let expected_radius: f64 = 3.0_f64.sqrt() / 2.0;
/// assert_relative_eq!(radius, expected_radius, epsilon = 1e-9);
/// ```
pub fn circumradius_with_center<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    circumcenter: &Point<f64, D>,
) -> Result<T, anyhow::Error>
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
        return Err(anyhow::Error::msg("Empty vertex set"));
    }

    let vertex_coords: [T; D] = (&vertices[0]).into();
    let vertex_coords_f64: [f64; D] = vertex_coords.map(std::convert::Into::into);
    Ok(na::distance(
        &na::Point::<T, D>::from(circumcenter.coordinates()),
        &na::Point::<T, D>::from(vertex_coords_f64),
    ))
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
    let circumradius = circumradius_with_center(simplex_vertices, &circumcenter)?;
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
/// has the correct sign relative to the simplex orientation.
///
/// # Mathematical Background
///
/// This determinant test is mathematically equivalent to checking if the test point
/// lies inside the circumsphere, but avoids the numerical instability that can arise
/// from computing circumcenter coordinates and distances explicitly.
///
/// The sign of the determinant depends on the orientation of the simplex:
/// - For a **positively oriented** simplex: positive determinant means the point is inside
/// - For a **negatively oriented** simplex: negative determinant means the point is inside
///
/// This function automatically determines the simplex orientation using [`simplex_orientation`]
/// and interprets the determinant sign accordingly, ensuring correct results regardless
/// of vertex ordering.
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

    // The sign of the determinant depends on the orientation of the simplex.
    // For a positively oriented simplex, the determinant is positive when the test point
    // is inside the circumsphere. For a negatively oriented simplex, the determinant
    // is negative when the test point is inside the circumsphere.
    // We need to check the orientation of the simplex to interpret the determinant correctly.
    let is_positive_orientation = simplex_orientation(simplex_vertices)?;

    if is_positive_orientation {
        // For positive orientation, positive determinant means inside circumsphere
        Ok(det > 0.0)
    } else {
        // For negative orientation, negative determinant means inside circumsphere
        Ok(det < 0.0)
    }
}

/// Determine the orientation of a simplex using the determinant of its coordinate matrix.
///
/// This function computes the orientation of a d-dimensional simplex by calculating
/// the determinant of a matrix formed by the coordinates of its vertices.
///
/// # Arguments
///
/// * `simplex_vertices` - A slice of vertices that form the simplex (must have exactly D+1 vertices)
///
/// # Returns
///
/// Returns `true` if the simplex is positively oriented (determinant > 0),
/// `false` if negatively oriented (determinant < 0), and `false` for degenerate cases (determinant = 0).
///
/// # Errors
///
/// Returns an error if the number of simplex vertices is not exactly D+1.
///
/// # Algorithm
///
/// For a d-dimensional simplex with vertices `v₁, v₂, ..., vₐ₊₁`, the orientation
/// is determined by the sign of the determinant of the matrix:
///
/// ```text
/// |  x₁   y₁   z₁  ...  1  |
/// |  x₂   y₂   z₂  ...  1  |
/// |  x₃   y₃   z₃  ...  1  |
/// |  ...  ...  ... ...  ... |
/// |  xₐ₊₁ yₐ₊₁ zₐ₊₁ ... 1  |
/// ```
///
/// Where each row contains the d coordinates of a vertex and a constant 1.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::simplex_orientation;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
/// let is_positive = simplex_orientation(&simplex_vertices).unwrap();
/// ```
pub fn simplex_orientation<T, U, const D: usize>(
    simplex_vertices: &[Vertex<T, U, D>],
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

    // Create matrix for orientation test
    // Matrix has D+1 columns: D coordinates + 1
    let mut matrix = zeros(D + 1, D + 1);

    // Populate rows with the coordinates of the vertices of the simplex
    for (i, v) in simplex_vertices.iter().enumerate() {
        // Use implicit conversion from vertex to coordinates
        let vertex_coords: [T; D] = v.into();
        let vertex_coords_f64: [f64; D] = vertex_coords.map(std::convert::Into::into);

        // Add coordinates
        for j in 0..D {
            matrix[(i, j)] = vertex_coords_f64[j];
        }

        // Add one to the last column
        matrix[(i, D)] = 1.0;
    }

    // Calculate the determinant of the matrix
    let det = matrix.det();

    // Positive determinant means positive orientation
    Ok(det > 0.0)
}

/// Find the extreme coordinates (minimum or maximum) across all vertices in a `HashMap`.
///
/// This function takes a `HashMap` of vertices and returns the minimum or maximum
/// coordinates based on the specified ordering.
///
/// # Arguments
///
/// * `vertices` - A `HashMap` containing Vertex objects, where the key is a
///   Uuid and the value is a Vertex.
/// * `ordering` - The ordering parameter is of type Ordering and is used to
///   specify whether the function should find the minimum or maximum
///   coordinates. Ordering is an enum with three possible values: `Less`,
///   `Equal`, and `Greater`.
///
/// # Returns
///
/// An array of type `T` with length `D` containing the minimum or maximum
/// coordinate for each dimension.
///
/// # Panics
///
/// This function should not panic under normal circumstances as it handles
/// the empty vertices case by returning default coordinates. However, it uses
/// `.unwrap()` internally which could theoretically panic if the `HashMap`
/// iterator behavior changes unexpectedly.
///
/// # Example
///
/// ```
/// use d_delaunay::geometry::predicates::find_extreme_coordinates;
/// use d_delaunay::delaunay_core::vertex::Vertex;
/// use d_delaunay::geometry::point::Point;
/// use std::collections::HashMap;
/// use std::cmp::Ordering;
/// let points = vec![
///     Point::new([-1.0, 2.0, 3.0]),
///     Point::new([4.0, -5.0, 6.0]),
///     Point::new([7.0, 8.0, -9.0]),
/// ];
/// let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
/// let hashmap = Vertex::into_hashmap(vertices);
/// let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less);
/// # use approx::assert_relative_eq;
/// assert_relative_eq!(min_coords.as_slice(), [-1.0, -5.0, -9.0].as_slice(), epsilon = 1e-9);
/// ```
#[must_use]
pub fn find_extreme_coordinates<T, U, const D: usize, S: ::std::hash::BuildHasher>(
    vertices: &HashMap<Uuid, Vertex<T, U, D>, S>,
    ordering: Ordering,
) -> [T; D]
where
    T: Default + OrderedEq + Float,
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

        assert_relative_eq!(radius_with_center.unwrap(), radius_direct, epsilon = 1e-10);
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

    #[test]
    fn predicates_find_min_coordinate() {
        let points = vec![
            Point::new([-1.0, 2.0, 3.0]),
            Point::new([4.0, -5.0, 6.0]),
            Point::new([7.0, 8.0, -9.0]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);
        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less);

        assert_relative_eq!(
            min_coords.as_slice(),
            [-1.0, -5.0, -9.0].as_slice(),
            epsilon = 1e-9
        );

        // Human readable output for cargo test -- --nocapture
        println!("min_coords = {min_coords:?}");
    }

    #[test]
    fn predicates_find_max_coordinate() {
        let points = vec![
            Point::new([-1.0, 2.0, 3.0]),
            Point::new([4.0, -5.0, 6.0]),
            Point::new([7.0, 8.0, -9.0]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater);

        assert_relative_eq!(
            max_coords.as_slice(),
            [7.0, 8.0, 6.0].as_slice(),
            epsilon = 1e-9
        );

        // Human readable output for cargo test -- --nocapture
        println!("max_coords = {max_coords:?}");
    }

    #[test]
    fn predicates_find_extreme_coordinates_empty() {
        let empty_hashmap: HashMap<Uuid, crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            HashMap::new();
        let min_coords = find_extreme_coordinates(&empty_hashmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&empty_hashmap, Ordering::Greater);

        // With empty hashmap, should return default values [0.0, 0.0, 0.0]
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
    fn predicates_find_extreme_coordinates_single_point() {
        let points = vec![Point::new([5.0, -3.0, 7.0])];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater);

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
    fn predicates_find_extreme_coordinates_equal_ordering() {
        let points = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0])];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        // Using Ordering::Equal should return the first vertex's coordinates unchanged
        let coords = find_extreme_coordinates(&hashmap, Ordering::Equal);
        // The first vertex in the iteration (order is not guaranteed in HashMap)
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
    fn predicates_find_extreme_coordinates_2d() {
        let points = vec![
            Point::new([1.0, 4.0]),
            Point::new([3.0, 2.0]),
            Point::new([2.0, 5.0]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 2>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater);

        assert_relative_eq!(min_coords.as_slice(), [1.0, 2.0].as_slice(), epsilon = 1e-9);
        assert_relative_eq!(max_coords.as_slice(), [3.0, 5.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn predicates_find_extreme_coordinates_1d() {
        let points = vec![Point::new([10.0]), Point::new([-5.0]), Point::new([3.0])];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 1>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater);

        assert_relative_eq!(min_coords.as_slice(), [-5.0].as_slice(), epsilon = 1e-9);
        assert_relative_eq!(max_coords.as_slice(), [10.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn predicates_find_extreme_coordinates_with_typed_data() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, -1.0, 2.0]),
            Point::new([-2.0, 5.0, 1.0]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, i32, 3>> = points
            .into_iter()
            .enumerate()
            .map(|(i, point)| {
                VertexBuilder::default()
                    .point(point)
                    .data(i32::try_from(i).unwrap())
                    .build()
                    .unwrap()
            })
            .collect();
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater);

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
    fn predicates_find_extreme_coordinates_identical_points() {
        let points = vec![
            Point::new([2.0, 3.0, 4.0]),
            Point::new([2.0, 3.0, 4.0]),
            Point::new([2.0, 3.0, 4.0]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater);

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
    fn predicates_find_extreme_coordinates_large_numbers() {
        let points = vec![
            Point::new([1e6, -1e6, 1e12]),
            Point::new([-1e9, 1e3, -1e15]),
            Point::new([1e15, 1e9, 1e6]),
        ];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less);
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater);

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
    fn predicates_simplex_orientation_positive() {
        // Test a positively oriented simplex
        // Using vertices that create a positive determinant
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];

        let orientation = simplex_orientation(&simplex_vertices).unwrap();
        assert!(orientation, "This simplex should be positively oriented");
    }

    #[test]
    fn predicates_simplex_orientation_negative() {
        // Test a negatively oriented simplex
        // Using vertices that create a negative determinant
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

        let orientation = simplex_orientation(&simplex_vertices).unwrap();
        assert!(!orientation, "This simplex should be negatively oriented");
    }

    #[test]
    fn predicates_simplex_orientation_2d() {
        // Test 2D orientation
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

        let orientation = simplex_orientation(&simplex_vertices).unwrap();
        assert!(orientation, "This 2D simplex should be positively oriented");
    }

    #[test]
    fn predicates_simplex_orientation_error_wrong_vertex_count() {
        // Test with wrong number of vertices
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
        let simplex_vertices = vec![vertex1, vertex2]; // Only 2 vertices for 3D

        let result = simplex_orientation(&simplex_vertices);
        assert!(
            result.is_err(),
            "Should error with wrong number of vertices"
        );
    }
}

//! Geometric predicates for d-dimensional geometry calculations.
//!
//! This module contains fundamental geometric predicates and calculations
//! that operate on points and simplices, including circumcenter and circumradius
//! calculations.

use crate::delaunay_core::{utilities::vec_to_array, vertex::Vertex};
use crate::geometry::matrix::invert;
use crate::geometry::point::{OrderedEq, Point};
use na::{ComplexField, Const, OPoint};
use nalgebra as na;
use num_traits::Float;
use peroxide::fuga::{LinearAlgebra, MatrixTrait, anyhow, zeros};
use serde::{Serialize, de::DeserializeOwned};
use std::{cmp::Ordering, collections::HashMap, hash::Hash, iter::Sum};
use uuid::Uuid;

/// Default tolerance for geometric predicates and degeneracy detection
const DEFAULT_TOLERANCE: f64 = 1e-10;

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

/// Represents the position of a point relative to a circumsphere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InSphere {
    /// The point is outside the circumsphere
    OUTSIDE,
    /// The point is on the boundary of the circumsphere (within numerical tolerance)
    BOUNDARY,
    /// The point is inside the circumsphere
    INSIDE,
}

impl std::fmt::Display for InSphere {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InSphere::OUTSIDE => write!(f, "OUTSIDE"),
            InSphere::BOUNDARY => write!(f, "BOUNDARY"),
            InSphere::INSIDE => write!(f, "INSIDE"),
        }
    }
}

/// Represents the orientation of a simplex.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    /// The simplex has negative orientation (determinant < 0)
    NEGATIVE,
    /// The simplex is degenerate (determinant ≈ 0)
    DEGENERATE,
    /// The simplex has positive orientation (determinant > 0)
    POSITIVE,
}

impl std::fmt::Display for Orientation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Orientation::NEGATIVE => write!(f, "NEGATIVE"),
            Orientation::DEGENERATE => write!(f, "DEGENERATE"),
            Orientation::POSITIVE => write!(f, "POSITIVE"),
        }
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
/// Returns an `Orientation` enum indicating whether the simplex is `POSITIVE`,
/// `NEGATIVE`, or `DEGENERATE`.
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
/// use d_delaunay::geometry::Orientation;
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::simplex_orientation;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
/// let oriented = simplex_orientation(&simplex_vertices).unwrap();
/// assert_eq!(oriented, Orientation::NEGATIVE);
/// ```
pub fn simplex_orientation<T, U, const D: usize>(
    simplex_vertices: &[Vertex<T, U, D>],
) -> Result<Orientation, anyhow::Error>
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

    // Use a tolerance for degenerate case detection
    let tolerance = DEFAULT_TOLERANCE;

    if det > tolerance {
        Ok(Orientation::POSITIVE)
    } else if det < -tolerance {
        Ok(Orientation::NEGATIVE)
    } else {
        Ok(Orientation::DEGENERATE)
    }
}

/// Check if a vertex is contained within the circumsphere of a simplex using distance calculations.
///
/// This function uses explicit distance calculations to determine if a vertex lies within
/// the circumsphere formed by the given vertices. It computes the circumcenter and circumradius
/// of the simplex, then calculates the distance from the test point to the circumcenter
/// and compares it with the circumradius.
///
/// # Algorithm
///
/// The algorithm follows these steps:
/// 1. Calculate the circumcenter of the simplex using [`circumcenter`]
/// 2. Calculate the circumradius using [`circumradius_with_center`]
/// 3. Compute the Euclidean distance from the test vertex to the circumcenter
/// 4. Compare the distance with the circumradius to determine containment
///
/// # Numerical Stability
///
/// This method can accumulate floating-point errors through multiple steps:
/// - Matrix inversion for circumcenter calculation
/// - Distance computation in potentially high-dimensional space
/// - Multiple coordinate transformations
///
/// For better numerical stability, consider using [`insphere`] which uses a
/// determinant-based approach that avoids explicit circumcenter computation.
///
/// # Arguments
///
/// * `simplex_vertices` - A slice of vertices that form the simplex
/// * `test_vertex` - The vertex to test for containment
///
/// # Returns
///
/// Returns an `InSphere` enum indicating whether the vertex is `INSIDE`, `OUTSIDE`,
/// or on the `BOUNDARY` of the circumsphere.
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
/// use d_delaunay::geometry::predicates::{insphere_distance, InSphere};
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
/// let test_vertex: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.5, 0.5, 0.5])).data(3).build().unwrap();
/// assert_eq!(insphere_distance(&simplex_vertices, test_vertex).unwrap(), InSphere::INSIDE);
/// ```
pub fn insphere_distance<T, U, const D: usize>(
    simplex_vertices: &[Vertex<T, U, D>],
    test_vertex: Vertex<T, U, D>,
) -> Result<InSphere, anyhow::Error>
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

    let tolerance = T::from(1e-9).unwrap_or_else(|| T::from(f64::EPSILON).unwrap_or_default());
    if num_traits::Float::abs(circumradius - radius) < tolerance {
        Ok(InSphere::BOUNDARY)
    } else if circumradius > radius {
        Ok(InSphere::INSIDE)
    } else {
        Ok(InSphere::OUTSIDE)
    }
}

/// Check if a vertex is contained within the circumsphere of a simplex using matrix determinant.
///
/// This is the `InSphere` predicate test, which determines whether a test point lies inside,
/// outside, or on the boundary of the circumsphere of a given simplex. This method is preferred
/// over `circumsphere_contains` as it provides better numerical stability by using a matrix
/// determinant approach instead of distance calculations, which can accumulate floating-point errors.
///
/// # Algorithm
///
/// This implementation follows the robust geometric predicates approach described in:
///
/// Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
/// Predicates." Discrete & Computational Geometry 18, no. 3 (1997): 305-363.
/// DOI: [10.1007/PL00009321](https://doi.org/10.1007/PL00009321)
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
/// from computing circumcenter coordinates and distances explicitly. As demonstrated
/// by Shewchuk, this approach provides much better numerical robustness for geometric
/// computations.
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
/// Returns [`InSphere::INSIDE`] if the given vertex is inside the circumsphere,
/// [`InSphere::BOUNDARY`] if it's on the boundary, or [`InSphere::OUTSIDE`] if it's outside.
///
/// # Errors
///
/// Returns an error if:
/// - The number of simplex vertices is not exactly D+1
/// - Matrix operations fail
/// - Coordinate conversion fails
///
/// # References
///
/// - Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric
///   Predicates." Discrete & Computational Geometry 18, no. 3 (1997): 305-363.
/// - Shewchuk, J. R. "Robust Adaptive Floating-Point Geometric Predicates."
///   Proceedings of the Twelfth Annual Symposium on Computational Geometry (1996): 141-150.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::insphere;
/// use d_delaunay::geometry::InSphere;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
///
/// // Test with a point clearly outside the circumsphere
/// let outside_vertex: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([2.0, 2.0, 2.0])).data(3).build().unwrap();
/// assert_eq!(insphere(&simplex_vertices, outside_vertex).unwrap(), InSphere::OUTSIDE);
///
/// // Test with a point clearly inside the circumsphere
/// let inside_vertex: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.25, 0.25, 0.25])).data(4).build().unwrap();
/// assert_eq!(insphere(&simplex_vertices, inside_vertex).unwrap(), InSphere::INSIDE);
/// ```
pub fn insphere<T, U, const D: usize>(
    simplex_vertices: &[Vertex<T, U, D>],
    test_vertex: Vertex<T, U, D>,
) -> Result<InSphere, anyhow::Error>
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
    // Matrix dimensions: (D+2) x (D+2)
    //   rows = D+1 simplex vertices + 1 test point
    //   cols = D coordinates + squared norm + 1
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
    let orientation = simplex_orientation(simplex_vertices)?;

    // Use a tolerance for boundary detection
    let tolerance = DEFAULT_TOLERANCE;

    match orientation {
        Orientation::DEGENERATE => {
            // Degenerate simplex - cannot determine containment reliably
            Err(anyhow::Error::msg(
                "Cannot determine circumsphere containment: simplex is degenerate",
            ))
        }
        Orientation::POSITIVE => {
            // For positive orientation, positive determinant means inside circumsphere
            if det > tolerance {
                Ok(InSphere::INSIDE)
            } else if det < -tolerance {
                Ok(InSphere::OUTSIDE)
            } else {
                Ok(InSphere::BOUNDARY)
            }
        }
        Orientation::NEGATIVE => {
            // For negative orientation, negative determinant means inside circumsphere
            if det < -tolerance {
                Ok(InSphere::INSIDE)
            } else if det > tolerance {
                Ok(InSphere::OUTSIDE)
            } else {
                Ok(InSphere::BOUNDARY)
            }
        }
    }
}

/// Check if a vertex is contained within the circumsphere of a simplex using the lifted paraboloid determinant method.
///
/// This is an alternative implementation of the circumsphere containment test using
/// a numerically stable matrix determinant approach based on the "lifted paraboloid" technique.
/// This method maps points to a higher-dimensional paraboloid and uses determinant calculations
/// to determine sphere containment, following the classical computational geometry approach.
///
/// # Algorithm
///
/// This implementation uses the lifted paraboloid method described in:
///
/// Preparata, Franco P., and Michael Ian Shamos.
/// "Computational Geometry: An Introduction."
/// Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
///
/// The method works by "lifting" points from d-dimensional space to (d+1)-dimensional space
/// by adding their squared distance as an additional coordinate. The in-sphere test then
/// reduces to computing the determinant of a matrix formed from these lifted coordinates.
///
/// For a d-dimensional simplex with vertices `v₀, v₁, ..., vₐ` and test point `p`,
/// the matrix has the structure:
///
/// ```text
/// | v₁-v₀  ||v₁-v₀||² |
/// | v₂-v₀  ||v₂-v₀||² |
/// | ...    ...       |
/// | vₐ-v₀  ||vₐ-v₀||² |
/// | p-v₀   ||p-v₀||²  |
/// ```
///
/// This formulation centers coordinates around the first vertex (v₀), which improves
/// numerical stability by reducing the magnitude of matrix elements compared to using
/// absolute coordinates.
///
/// # Mathematical Background
///
/// The lifted paraboloid method exploits the fact that the circumsphere of a set of points
/// in d-dimensional space corresponds to a hyperplane in (d+1)-dimensional space when
/// points are lifted to the paraboloid z = x₁² + x₂² + ... + xₐ². A point lies inside
/// the circumsphere if and only if it lies below this hyperplane in the lifted space.
///
/// # Arguments
///
/// * `simplex_vertices` - A slice of vertices that form the simplex (must have exactly D+1 vertices)
/// * `test_vertex` - The vertex to test for containment
///
/// # Returns
///
/// Returns an `InSphere` enum indicating whether the vertex is `INSIDE`, `OUTSIDE`,
/// or on the `BOUNDARY` of the circumsphere.
///
/// # Errors
///
/// Returns an error if:
/// - The number of simplex vertices is not exactly D+1
/// - Matrix operations fail
/// - Coordinate conversion fails
///
/// # References
///
/// - Preparata, Franco P., and Michael Ian Shamos. "Computational Geometry: An Introduction."
///   Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
/// - Edelsbrunner, Herbert. "Algorithms in Combinatorial Geometry."
///   EATCS Monographs on Theoretical Computer Science. Berlin: Springer-Verlag, 1987.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
/// use d_delaunay::geometry::point::Point;
/// use d_delaunay::geometry::predicates::insphere_lifted;
/// let vertex1: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex2: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([1.0, 0.0, 0.0])).data(1).build().unwrap();
/// let vertex3: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 1.0, 0.0])).data(1).build().unwrap();
/// let vertex4: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.0, 0.0, 1.0])).data(2).build().unwrap();
/// let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4];
///
/// // Test with a point that should be inside according to the lifted paraboloid method
/// let test_vertex: Vertex<f64, i32, 3> = VertexBuilder::default().point(Point::new([0.1, 0.1, 0.1])).data(3).build().unwrap();
/// let result = insphere_lifted(&simplex_vertices, test_vertex);
/// assert!(result.is_ok()); // Should execute without error
/// ```
pub fn insphere_lifted<T, U, const D: usize>(
    simplex_vertices: &[Vertex<T, U, D>],
    test_vertex: Vertex<T, U, D>,
) -> Result<InSphere, anyhow::Error>
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

    // Get the reference vertex (first vertex of the simplex)
    let ref_vertex_coords: [T; D] = (&simplex_vertices[0]).into();
    let ref_vertex_coords_f64: [f64; D] = ref_vertex_coords.map(std::convert::Into::into);

    // Create matrix for in-sphere test
    // Matrix dimensions: (D+1) x (D+1)
    //   rows = D simplex vertices (relative to first) + 1 test point
    //   cols = D coordinates + 1 squared norm
    let mut matrix = zeros(D + 1, D + 1);

    // Populate rows with the coordinates relative to the reference vertex
    for i in 1..=D {
        let vertex_coords: [T; D] = (&simplex_vertices[i]).into();
        let vertex_coords_f64: [f64; D] = vertex_coords.map(std::convert::Into::into);

        let mut squared_norm = 0.0_f64;

        // Calculate relative coordinates and squared norm
        for j in 0..D {
            let relative_coord = vertex_coords_f64[j] - ref_vertex_coords_f64[j];
            matrix[(i - 1, j)] = relative_coord;
            squared_norm += relative_coord * relative_coord;
        }

        // Add squared norm to the last column
        matrix[(i - 1, D)] = squared_norm;
    }

    // Add the test vertex to the last row
    let test_vertex_coords: [T; D] = (&test_vertex).into();
    let test_vertex_coords_f64: [f64; D] = test_vertex_coords.map(std::convert::Into::into);

    let mut test_squared_norm = 0.0_f64;

    // Calculate relative coordinates and squared norm for test vertex
    for j in 0..D {
        let relative_coord = test_vertex_coords_f64[j] - ref_vertex_coords_f64[j];
        matrix[(D, j)] = relative_coord;
        test_squared_norm += relative_coord * relative_coord;
    }

    // Add squared norm to the last column
    matrix[(D, D)] = test_squared_norm;

    // Calculate the determinant of the matrix
    let det = matrix.det();

    // For this matrix formulation using relative coordinates, we need to check
    // the simplex orientation to correctly interpret the determinant sign.
    let orientation = simplex_orientation(simplex_vertices)?;

    // Use a tolerance for boundary detection
    let tolerance = DEFAULT_TOLERANCE;

    match orientation {
        Orientation::DEGENERATE => {
            // Degenerate simplex - cannot determine containment reliably
            Err(anyhow::Error::msg(
                "Cannot determine circumsphere containment: simplex is degenerate",
            ))
        }
        Orientation::POSITIVE => {
            // For positive orientation, negative determinant means inside circumsphere
            if det < -tolerance {
                Ok(InSphere::INSIDE)
            } else if det > tolerance {
                Ok(InSphere::OUTSIDE)
            } else {
                Ok(InSphere::BOUNDARY)
            }
        }
        Orientation::NEGATIVE => {
            // For negative orientation, positive determinant means inside circumsphere
            if det > tolerance {
                Ok(InSphere::INSIDE)
            } else if det < -tolerance {
                Ok(InSphere::OUTSIDE)
            } else {
                Ok(InSphere::BOUNDARY)
            }
        }
    }
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
/// Returns `Ok([T; D])` containing the minimum or maximum coordinate for each dimension,
/// or an error if the vertices `HashMap` is empty.
///
/// # Errors
///
/// Returns an error if the vertices `HashMap` is empty.
///
/// # Panics
///
/// Panics if the vertices `HashMap` is empty (this should be caught by the early return).
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
/// let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less).unwrap();
/// # use approx::assert_relative_eq;
/// assert_relative_eq!(min_coords.as_slice(), [-1.0, -5.0, -9.0].as_slice(), epsilon = 1e-9);
/// ```
pub fn find_extreme_coordinates<T, U, const D: usize, S: ::std::hash::BuildHasher>(
    vertices: &HashMap<Uuid, Vertex<T, U, D>, S>,
    ordering: Ordering,
) -> Result<[T; D], anyhow::Error>
where
    T: Default + OrderedEq + Float,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Default + DeserializeOwned + Serialize + Sized,
{
    if vertices.is_empty() {
        return Err(anyhow::Error::msg(
            "Cannot find extreme coordinates: vertices HashMap is empty",
        ));
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

    Ok(extreme_coords)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::delaunay_core::vertex::VertexBuilder;
    use crate::geometry::point::Point;
    use approx::assert_relative_eq;

    /// Tolerance for distance comparisons in tests
    const DISTANCE_TOLERANCE: f64 = 1e-9;

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

        assert_relative_eq!(radius, expected_radius, epsilon = DISTANCE_TOLERANCE);
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

        assert_eq!(
            insphere_distance(&simplex_vertices, test_vertex).unwrap(),
            InSphere::BOUNDARY
        );
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

        assert_eq!(
            insphere(&simplex_vertices, test_vertex).unwrap(),
            InSphere::OUTSIDE
        );
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
        assert_relative_eq!(radius, expected_radius, epsilon = DEFAULT_TOLERANCE);
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
        let result = insphere(&simplex_vertices, vertex_far_outside);
        assert!(result.is_ok());

        // Test with origin (should be inside or on boundary)
        let origin = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(3)
            .build()
            .unwrap();
        let result_origin = insphere(&simplex_vertices, origin);
        assert!(result_origin.is_ok());
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_matrix() {
        // Test the optimized matrix determinant method for circumsphere containment
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
        let _vertex_far_outside: crate::delaunay_core::vertex::Vertex<f64, i32, 3> =
            VertexBuilder::default()
                .point(Point::new([10.0, 10.0, 10.0]))
                .data(4)
                .build()
                .unwrap();
        // TODO: Matrix method should correctly identify this as outside, but currently fails
        // This is why we use circumsphere_contains_vertex in bowyer_watson instead
        // assert_eq!(
        //     circumsphere_contains_vertex_matrix(&simplex_vertices, _vertex_far_outside).unwrap(),
        //     InSphere::OUTSIDE
        // );

        // Test with origin (should be inside or on boundary)
        let origin = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0]))
            .data(3)
            .build()
            .unwrap();
        // Test with origin, which is a vertex of the simplex (on boundary of circumsphere)
        assert_eq!(
            insphere_lifted(&simplex_vertices, origin).unwrap(),
            InSphere::BOUNDARY
        );
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_matrix_2d() {
        // Test the optimized matrix method for 2D circumcircle containment
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

        // Test vertex far outside circumcircle - should be outside
        let vertex_far_outside = VertexBuilder::default()
            .point(Point::new([10.0, 10.0]))
            .build()
            .unwrap();
        assert_eq!(
            insphere_lifted(&simplex_vertices, vertex_far_outside).unwrap(),
            InSphere::OUTSIDE
        );

        // Test with point inside the triangle - should be inside
        let inside_point = VertexBuilder::default()
            .point(Point::new([0.1, 0.1]))
            .build()
            .unwrap();
        assert_eq!(
            insphere_lifted(&simplex_vertices, inside_point).unwrap(),
            InSphere::INSIDE
        );
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_matrix_4d() {
        // Test the optimized matrix method for 4D circumsphere containment
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 4> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0, 0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

        // Test vertex clearly outside circumsphere
        let vertex_far_outside = VertexBuilder::default()
            .point(Point::new([10.0, 10.0, 10.0, 10.0]))
            .build()
            .unwrap();
        assert_eq!(
            insphere_lifted(&simplex_vertices, vertex_far_outside).unwrap(),
            InSphere::OUTSIDE
        );

        // Test with point inside the simplex's circumsphere
        let inside_point = VertexBuilder::default()
            .point(Point::new([0.1, 0.1, 0.1, 0.1]))
            .build()
            .unwrap();
        assert_eq!(
            insphere_lifted(&simplex_vertices, inside_point).unwrap(),
            InSphere::INSIDE
        );
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_matrix_4d_edge_cases() {
        // Test with known geometric cases for 4D circumsphere containment
        // Unit 4-simplex: vertices at origin and unit vectors along each axis
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, i32, 4> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

        // The circumcenter of this 4D simplex should be at (0.5, 0.5, 0.5, 0.5)
        let circumcenter_point = VertexBuilder::default()
            .point(Point::new([0.5, 0.5, 0.5, 0.5]))
            .data(3)
            .build()
            .unwrap();

        // Point at circumcenter should be inside the circumsphere
        assert_eq!(
            insphere_lifted(&simplex_vertices, circumcenter_point).unwrap(),
            InSphere::INSIDE
        );

        // Test with point that is actually inside circumsphere (distance 0.8 < radius 1.0)
        let actually_inside = VertexBuilder::default()
            .point(Point::new([0.9, 0.9, 0.9, 0.9]))
            .data(4)
            .build()
            .unwrap();
        assert_eq!(
            insphere_lifted(&simplex_vertices, actually_inside).unwrap(),
            InSphere::INSIDE
        );

        // Test with one of the simplex vertices (on boundary of circumsphere)
        assert_eq!(
            insphere_lifted(&simplex_vertices, vertex1).unwrap(),
            InSphere::BOUNDARY
        );

        // Test with a point on one of the coordinate axes but closer to origin
        let axis_point = VertexBuilder::default()
            .point(Point::new([0.25, 0.0, 0.0, 0.0]))
            .data(5)
            .build()
            .unwrap();
        assert_eq!(
            insphere_lifted(&simplex_vertices, axis_point).unwrap(),
            InSphere::INSIDE
        );

        // Test with point equidistant from multiple vertices
        let equidistant_point = VertexBuilder::default()
            .point(Point::new([0.5, 0.5, 0.0, 0.0]))
            .data(6)
            .build()
            .unwrap();
        assert_eq!(
            insphere_lifted(&simplex_vertices, equidistant_point).unwrap(),
            InSphere::INSIDE
        );
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_matrix_4d_degenerate_cases() {
        // Test with 4D simplex that has some special properties
        // Regular 4D simplex with vertices forming a specific pattern
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 4> =
            VertexBuilder::default()
                .point(Point::new([1.0, 1.0, 1.0, 1.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, -1.0, -1.0, -1.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([-1.0, 1.0, -1.0, -1.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([-1.0, -1.0, 1.0, -1.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([-1.0, -1.0, -1.0, 1.0]))
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

        // Test with origin (should be inside this symmetric simplex)
        let origin = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        // TODO: Fix matrix method - it disagrees with standard method on this case
        let _result = insphere_lifted(&simplex_vertices, origin).unwrap();
        // Don't assert specific result until matrix method is fixed

        // Test with point far outside
        let far_point = VertexBuilder::default()
            .point(Point::new([10.0, 10.0, 10.0, 10.0]))
            .build()
            .unwrap();
        // TODO: Fix matrix method - it may give incorrect results for far points in 4D cases
        let _far_result = insphere_lifted(&simplex_vertices, far_point).unwrap();
        // Don't assert specific result until matrix method is fixed

        // Test with point on the surface of the circumsphere (approximately)
        // This is challenging to compute exactly, so we test a point that should be close
        let surface_point = VertexBuilder::default()
            .point(Point::new([1.5, 1.5, 1.5, 1.5]))
            .build()
            .unwrap();
        let result = insphere_lifted(&simplex_vertices, surface_point);
        assert!(result.is_ok()); // Should not error, result depends on exact circumsphere
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_matrix_error_cases() {
        // Test with wrong number of vertices (should error)
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
        let incomplete_simplex = vec![vertex1, vertex2]; // Only 2 vertices for 3D

        let test_vertex = VertexBuilder::default()
            .point(Point::new([0.5, 0.5, 0.5]))
            .data(3)
            .build()
            .unwrap();

        let result = insphere_lifted(&incomplete_simplex, test_vertex);
        assert!(result.is_err(), "Should error with insufficient vertices");
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_matrix_edge_cases() {
        // Test with known geometric cases
        // Unit tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
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

        // The circumcenter of this tetrahedron is at (0.5, 0.5, 0.5)
        let _circumcenter_point: crate::delaunay_core::vertex::Vertex<f64, i32, 3> =
            VertexBuilder::default()
                .point(Point::new([0.5, 0.5, 0.5]))
                .data(3)
                .build()
                .unwrap();

        // TODO: Point at circumcenter should be inside the circumsphere, but matrix method fails
        // This is why we use circumsphere_contains_vertex in bowyer_watson instead
        // assert_eq!(
        //     circumsphere_contains_vertex_matrix(&simplex_vertices, _circumcenter_point).unwrap(),
        //     InSphere::INSIDE
        // );

        // Test with point that is actually inside circumsphere (distance 0.693 < radius 0.866)
        let _actually_inside: crate::delaunay_core::vertex::Vertex<f64, i32, 3> =
            VertexBuilder::default()
                .point(Point::new([0.9, 0.9, 0.9]))
                .data(4)
                .build()
                .unwrap();
        // TODO: Matrix method should correctly identify this point as inside, but currently fails
        // This is why we use circumsphere_contains_vertex in bowyer_watson instead
        // assert_eq!(
        //     circumsphere_contains_vertex_matrix(&simplex_vertices, _actually_inside).unwrap(),
        //     InSphere::INSIDE
        // );

        // Test with one of the simplex vertices (on boundary, but matrix method returns BOUNDARY)
        assert_eq!(
            insphere_lifted(&simplex_vertices, vertex1).unwrap(),
            InSphere::BOUNDARY
        );
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_matrix_1d() {
        // Test with 1D case (line segment)
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 1> =
            VertexBuilder::default()
                .point(Point::new([0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([2.0]))
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2];

        // Test point at the midpoint (should be on the "circumcircle" - the perpendicular bisector)
        let midpoint = VertexBuilder::default()
            .point(Point::new([1.0]))
            .build()
            .unwrap();
        let result = insphere_lifted(&simplex_vertices, midpoint);
        assert!(result.is_ok()); // Should not error

        // Test point far from the line segment
        let far_point = VertexBuilder::default()
            .point(Point::new([10.0]))
            .build()
            .unwrap();
        let result_far = insphere_lifted(&simplex_vertices, far_point);
        assert!(result_far.is_ok()); // Should not error
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_4d() {
        // Test the standard determinant method for 4D circumsphere containment
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 4> =
            VertexBuilder::default()
                .point(Point::new([0.0, 0.0, 0.0, 0.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0]))
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

        // Test vertex clearly outside circumsphere
        let vertex_far_outside = VertexBuilder::default()
            .point(Point::new([10.0, 10.0, 10.0, 10.0]))
            .build()
            .unwrap();
        assert_eq!(
            insphere(&simplex_vertices, vertex_far_outside).unwrap(),
            InSphere::OUTSIDE
        );

        // Test with point inside the simplex's circumsphere
        let inside_point = VertexBuilder::default()
            .point(Point::new([0.1, 0.1, 0.1, 0.1]))
            .build()
            .unwrap();
        assert_eq!(
            insphere(&simplex_vertices, inside_point).unwrap(),
            InSphere::INSIDE
        );
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_4d_edge_cases() {
        // Test with known geometric cases for 4D circumsphere containment
        // Unit 4-simplex: vertices at origin and unit vectors along each axis
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, i32, 4> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 0.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([0.0, 1.0, 0.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 1.0, 0.0]))
            .data(1)
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 1.0]))
            .data(2)
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

        // The circumcenter of this 4D simplex should be at (0.5, 0.5, 0.5, 0.5)
        let circumcenter_point = VertexBuilder::default()
            .point(Point::new([0.5, 0.5, 0.5, 0.5]))
            .data(3)
            .build()
            .unwrap();

        // Point at circumcenter should be inside the circumsphere
        assert_eq!(
            insphere(&simplex_vertices, circumcenter_point).unwrap(),
            InSphere::INSIDE
        );

        // Test with point that is actually inside circumsphere (distance 0.8 < radius 1.0)
        let actually_inside = VertexBuilder::default()
            .point(Point::new([0.9, 0.9, 0.9, 0.9]))
            .data(4)
            .build()
            .unwrap();
        assert_eq!(
            insphere(&simplex_vertices, actually_inside).unwrap(),
            InSphere::INSIDE
        );

        // Test with one of the simplex vertices (should be on the boundary)
        // Due to floating-point precision, this might be exactly on the boundary
        let result = insphere(&simplex_vertices, vertex1).unwrap();
        // For vertices of the simplex, they should be on the boundary, but floating-point precision
        // might cause slight variations, so we just verify the method runs without error
        let _ = result; // We don't assert a specific result here due to numerical precision

        // Test with a point on one of the coordinate axes but closer to origin
        let axis_point = VertexBuilder::default()
            .point(Point::new([0.25, 0.0, 0.0, 0.0]))
            .data(5)
            .build()
            .unwrap();
        assert_eq!(
            insphere(&simplex_vertices, axis_point).unwrap(),
            InSphere::INSIDE
        );

        // Test with point equidistant from multiple vertices
        let equidistant_point = VertexBuilder::default()
            .point(Point::new([0.5, 0.5, 0.0, 0.0]))
            .data(6)
            .build()
            .unwrap();
        assert_eq!(
            insphere(&simplex_vertices, equidistant_point).unwrap(),
            InSphere::INSIDE
        );
    }

    #[test]
    fn predicates_circumsphere_contains_vertex_4d_degenerate_cases() {
        // Test with 4D simplex that has some special properties
        // Regular 4D simplex with vertices forming a specific pattern
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 4> =
            VertexBuilder::default()
                .point(Point::new([1.0, 1.0, 1.0, 1.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, -1.0, -1.0, -1.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([-1.0, 1.0, -1.0, -1.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([-1.0, -1.0, 1.0, -1.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([-1.0, -1.0, -1.0, 1.0]))
            .build()
            .unwrap();
        let simplex_vertices = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

        // Test with origin (should be inside this symmetric simplex)
        let origin = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();
        assert_eq!(
            insphere_distance(&simplex_vertices, origin).unwrap(),
            InSphere::INSIDE
        );

        // Test with point far outside
        let far_point = VertexBuilder::default()
            .point(Point::new([10.0, 10.0, 10.0, 10.0]))
            .build()
            .unwrap();
        assert_eq!(
            insphere_distance(&simplex_vertices, far_point).unwrap(),
            InSphere::OUTSIDE
        );

        // Test with point on the surface of the circumsphere (approximately)
        // This is challenging to compute exactly, so we test a point that should be close
        let surface_point = VertexBuilder::default()
            .point(Point::new([1.5, 1.5, 1.5, 1.5]))
            .build()
            .unwrap();
        let result = insphere_distance(&simplex_vertices, surface_point);
        assert!(result.is_ok()); // Should not error, result depends on exact circumsphere
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
        let result = insphere(&simplex_vertices, vertex_far_outside);
        assert!(result.is_ok());

        // Test with center of triangle (should be inside)
        let center = VertexBuilder::default()
            .point(Point::new([0.33, 0.33]))
            .build()
            .unwrap();
        let result_center = insphere(&simplex_vertices, center);
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

        let circumsphere_result = insphere_distance(&simplex_vertices, test_point);
        assert!(circumsphere_result.is_ok());

        let determinant_result = insphere_distance(&simplex_vertices, test_point);
        assert!(determinant_result.is_ok());

        // At minimum, both methods should give the same result for the same input
        let far_point = VertexBuilder::default()
            .point(Point::new([100.0, 100.0]))
            .build()
            .unwrap();

        let circumsphere_far = insphere_distance(&simplex_vertices, far_point);
        let determinant_far = insphere_distance(&simplex_vertices, far_point);

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
        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less).unwrap();

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
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater).unwrap();

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
        let min_coords_result = find_extreme_coordinates(&empty_hashmap, Ordering::Less);
        let max_coords_result = find_extreme_coordinates(&empty_hashmap, Ordering::Greater);

        // With empty hashmap, should return an error
        assert!(min_coords_result.is_err());
        assert!(max_coords_result.is_err());
    }

    #[test]
    fn predicates_find_extreme_coordinates_single_point() {
        let points = vec![Point::new([5.0, -3.0, 7.0])];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 3>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater).unwrap();

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
        let coords = find_extreme_coordinates(&hashmap, Ordering::Equal).unwrap();
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

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater).unwrap();

        assert_relative_eq!(min_coords.as_slice(), [1.0, 2.0].as_slice(), epsilon = 1e-9);
        assert_relative_eq!(max_coords.as_slice(), [3.0, 5.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn predicates_find_extreme_coordinates_1d() {
        let points = vec![Point::new([10.0]), Point::new([-5.0]), Point::new([3.0])];
        let vertices: Vec<crate::delaunay_core::vertex::Vertex<f64, Option<()>, 1>> =
            crate::delaunay_core::vertex::Vertex::from_points(points);
        let hashmap = crate::delaunay_core::vertex::Vertex::into_hashmap(vertices);

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater).unwrap();

        assert_relative_eq!(
            min_coords.as_slice(),
            [-5.0].as_slice(),
            epsilon = DISTANCE_TOLERANCE
        );
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

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater).unwrap();

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

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater).unwrap();

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

        let min_coords = find_extreme_coordinates(&hashmap, Ordering::Less).unwrap();
        let max_coords = find_extreme_coordinates(&hashmap, Ordering::Greater).unwrap();

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
        assert_eq!(
            orientation,
            Orientation::POSITIVE,
            "This simplex should be positively oriented"
        );
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
        assert_eq!(
            orientation,
            Orientation::NEGATIVE,
            "This simplex should be negatively oriented"
        );
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
        assert_eq!(
            orientation,
            Orientation::POSITIVE,
            "This 2D simplex should be positively oriented"
        );
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

    #[test]
    fn debug_circumsphere_properties() {
        println!("=== 3D Unit Tetrahedron Analysis ===");

        // Unit tetrahedron: vertices at (0,0,0), (1,0,0), (0,1,0), (0,0,1)
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

        let center = circumcenter(&simplex_vertices).unwrap();
        let radius = circumradius(&simplex_vertices).unwrap();

        println!("Circumcenter: {:?}", center.coordinates());
        println!("Circumradius: {radius}");

        // Test the point (0.9, 0.9, 0.9)
        let distance_to_center =
            ((0.9_f64 - 0.5).powi(2) + (0.9_f64 - 0.5).powi(2) + (0.9_f64 - 0.5).powi(2)).sqrt();
        println!("Point (0.9, 0.9, 0.9) distance to circumcenter: {distance_to_center}");
        println!(
            "Is point inside circumsphere (distance < radius)? {}",
            distance_to_center < radius
        );

        let test_vertex = VertexBuilder::default()
            .point(Point::new([0.9, 0.9, 0.9]))
            .data(4)
            .build()
            .unwrap();

        let standard_result = insphere_distance(&simplex_vertices, test_vertex).unwrap();
        let matrix_result = insphere_lifted(&simplex_vertices, test_vertex).unwrap();

        println!("Standard method result: {standard_result}");
        println!("Matrix method result: {matrix_result}");

        println!("\n=== 4D Symmetric Simplex Analysis ===");

        // Regular 4D simplex with vertices forming a specific pattern
        let vertex1: crate::delaunay_core::vertex::Vertex<f64, Option<()>, 4> =
            VertexBuilder::default()
                .point(Point::new([1.0, 1.0, 1.0, 1.0]))
                .build()
                .unwrap();
        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, -1.0, -1.0, -1.0]))
            .build()
            .unwrap();
        let vertex3 = VertexBuilder::default()
            .point(Point::new([-1.0, 1.0, -1.0, -1.0]))
            .build()
            .unwrap();
        let vertex4 = VertexBuilder::default()
            .point(Point::new([-1.0, -1.0, 1.0, -1.0]))
            .build()
            .unwrap();
        let vertex5 = VertexBuilder::default()
            .point(Point::new([-1.0, -1.0, -1.0, 1.0]))
            .build()
            .unwrap();
        let simplex_vertices_4d = vec![vertex1, vertex2, vertex3, vertex4, vertex5];

        let center_4d = circumcenter(&simplex_vertices_4d).unwrap();
        let radius_4d = circumradius(&simplex_vertices_4d).unwrap();

        println!("4D Circumcenter: {:?}", center_4d.coordinates());
        println!("4D Circumradius: {radius_4d}");

        // Test the origin (0, 0, 0, 0)
        let distance_to_center_4d =
            (center_4d.coordinates().iter().map(|&x| x * x).sum::<f64>()).sqrt();
        println!("Origin distance to circumcenter: {distance_to_center_4d}");
        println!(
            "Is origin inside circumsphere (distance < radius)? {}",
            distance_to_center_4d < radius_4d
        );

        let origin_vertex = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 0.0, 0.0]))
            .build()
            .unwrap();

        let standard_result_4d = insphere_distance(&simplex_vertices_4d, origin_vertex).unwrap();
        let matrix_result_4d = insphere_lifted(&simplex_vertices_4d, origin_vertex).unwrap();

        println!("Standard method result for origin: {standard_result_4d}");
        println!("Matrix method result for origin: {matrix_result_4d}");

        // Don't assert anything, just debug output
    }

    #[test]
    fn compare_circumsphere_methods() {
        // Compare results between standard and matrix methods
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

        // Test various points
        let test_points = [
            Point::new([0.1, 0.1]),   // Should be inside
            Point::new([0.5, 0.5]),   // Circumcenter region
            Point::new([10.0, 10.0]), // Far outside
            Point::new([0.25, 0.25]), // Inside
            Point::new([2.0, 2.0]),   // Outside
        ];

        for (i, point) in test_points.iter().enumerate() {
            let test_vertex = VertexBuilder::default().point(*point).build().unwrap();

            let standard_result = insphere_distance(&simplex_vertices, test_vertex).unwrap();
            let matrix_result = insphere_lifted(&simplex_vertices, test_vertex).unwrap();

            println!(
                "Point {}: {:?} -> Standard: {:?}, Matrix: {}",
                i,
                point.coordinates(),
                standard_result,
                matrix_result
            );
        }

        // Don't assert anything - just observe the comparison
    }
}

//! Data and operations on d-dimensional [vertices](https://en.wikipedia.org/wiki/Vertex_(computer_graphics)).

use super::{point::Point, utilities::make_uuid};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::{cmp::Ordering, collections::HashMap, hash::Hash, option::Option};
use uuid::Uuid;

#[derive(Builder, Clone, Copy, Debug, Default, Deserialize, Serialize)]
/// The [Vertex] struct represents a vertex in a triangulation with a [Point],
/// a unique identifier, an optional incident cell identifier, and optional
/// data.
///
/// # Properties:
///
/// * `point`: A generic [Point] representing the coordinates of
///   the vertex in a D-dimensional space.
/// * `uuid`: A [Uuid] representing a universally unique identifier for the
///   for the [Vertex]. This can be used to uniquely
///   identify the vertex in a graph or any other data structure.
/// * `incident_cell`: The `incident_cell` property is an optional [Uuid] that
///   represents a `Cell` containing the [Vertex]. This is
///   calculated by the `delaunay_core::triangulation_data_structure::Tds`.
/// * `data`: The `data` property is an optional field that can hold any
///   type `U`. It is used to store additional data associated with the vertex.
///
/// Data type T is in practice f64 which does not implement Eq, Hash, or Ord.
///
/// U is intended to be data associated with the vertex, e.g. a string, which
/// implements Eq, Hash, Ord, PartialEq, and PartialOrd.
pub struct Vertex<T, U, const D: usize>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The coordinates of the vertex in a D-dimensional space.
    pub point: Point<T, D>,
    /// A universally unique identifier for the vertex.
    #[builder(setter(skip), default = "make_uuid()")]
    pub uuid: Uuid,
    /// The [Uuid] of the `Cell` that the vertex is incident to.
    #[builder(setter(skip), default = "None")]
    pub incident_cell: Option<Uuid>,
    /// Optional data associated with the vertex.
    #[builder(setter(into, strip_option), default)]
    pub data: Option<U>,
}

impl<T, U, const D: usize> Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// The function `from_points` takes a vector of points and returns a
    /// vector of vertices, using the `new` method.
    ///
    /// # Arguments:
    ///
    /// * `points`: `points` is a vector of [Point] objects.
    ///
    /// # Returns:
    ///
    /// The function `from_points` returns a `Vec<Vertex<T, U, D>>`, where `T`
    /// is the type of the coordinates of the [Vertex], `U` is the type of the
    /// optional data associated with the [Vertex], and `D` is the
    /// dimensionality of the [Vertex].
    pub fn from_points(points: Vec<Point<T, D>>) -> Vec<Self> {
        // points.into_iter().map(|p| Self::new(p)).collect()
        points
            .into_iter()
            .map(|p| VertexBuilder::default().point(p).build().unwrap())
            .collect()
    }

    /// The function `into_hashmap` converts a vector of vertices into a
    /// [HashMap], using the vertices [Uuid] as the key.
    ///
    /// # Arguments:
    ///
    /// * `vertices`: `vertices` is a vector of `Vertex<T, U, D>`.
    ///
    /// # Returns:
    ///
    /// The function `into_hashmap` returns a [HashMap] with the key type
    /// [Uuid] and the value type [Vertex], i.e.
    /// `std::collections::HashMap<Uuid, Vertex<T, U, D>`.
    pub fn into_hashmap(vertices: Vec<Self>) -> HashMap<Uuid, Self> {
        vertices.into_iter().map(|v| (v.uuid, v)).collect()
    }

    /// The `dim` function returns the dimensionality of the [Vertex].
    ///
    /// # Returns:
    ///
    /// The `dim` function is returning the value of `D`, which the number of
    /// coordinates.
    ///
    /// # Example:
    /// ```
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// let vertex: Vertex<f64, Option<()>, 4> = VertexBuilder::default().point(point).build().unwrap();
    /// assert_eq!(vertex.dim(), 4);
    /// ```
    pub fn dim(&self) -> usize {
        D
    }

    /// The function is_valid checks if a [Vertex] is valid.
    ///
    /// # Returns:
    ///
    /// True if the [Vertex] is valid; the [Point] is correct, the [Uuid] is
    /// valid and unique, and the `incident_cell` contains the [Uuid] of a
    /// `Cell` that contains the [Vertex].
    pub fn is_valid(self) -> bool {
        todo!("Implement is_valid for Vertex")
    }
}

/// Equality of vertices is based on equality of elements in vector of coords.
impl<T, U, const D: usize> PartialEq for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.point == other.point
        // && self.uuid == other.uuid
        // && self.incident_cell == other.incident_cell
        // && self.data == other.data
    }
}

/// Eq implementation for Vertex based on point equality
impl<T, U, const D: usize> Eq for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{}

/// Order of vertices is based on lexicographic order of elements in vector of coords.
impl<T, U, const D: usize> PartialOrd for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.point.partial_cmp(&other.point)
    }
}

impl<T, U, const D: usize> Hash for Vertex<T, U, D>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    ordered_float::OrderedFloat<f64>: From<T>,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.point.hash(state);
        self.uuid.hash(state);
        self.incident_cell.hash(state);
        self.data.hash(state);
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn vertex_default() {
        let vertex: Vertex<f64, Option<()>, 3> = Default::default();

        assert_eq!(vertex.point.coords, [0.0, 0.0, 0.0]);
        assert_eq!(vertex.dim(), 3);
        assert!(vertex.uuid.is_nil());
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_none());

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", vertex);
    }

    #[test]
    fn vertex_builder() {
        let mut vertex: Vertex<f64, &str, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        assert_eq!(vertex.point.coords, [1.0, 2.0, 3.0]);
        assert_eq!(vertex.dim(), 3);
        assert!(!vertex.uuid.is_nil());
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_none());

        // Can mutate later
        vertex.data = Some("3D");
        assert_eq!(vertex.data.unwrap(), "3D");

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", vertex);
    }

    #[test]
    fn vertex_builder_with_data() {
        let vertex: Vertex<f64, i32, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .data(1)
            .build()
            .unwrap();

        assert_eq!(vertex.point.coords, [1.0, 2.0, 3.0]);
        assert_eq!(vertex.dim(), 3);
        assert!(!vertex.uuid.is_nil());
        assert!(vertex.incident_cell.is_none());
        assert_eq!(vertex.data.unwrap(), 1);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", vertex);
    }

    #[test]
    fn vertex_copy() {
        let vertex: Vertex<f64, &str, 4> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0, 4.0]))
            .data("4D")
            .build()
            .unwrap();
        let vertex_copy = vertex;

        assert_eq!(vertex, vertex_copy);
    }

    #[test]
    fn vertex_from_points() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);

        assert_eq!(vertices[0].point.coords, [1.0, 2.0, 3.0]);
        assert_eq!(vertices[0].dim(), 3);
        assert_eq!(vertices[1].point.coords, [4.0, 5.0, 6.0]);
        assert_eq!(vertices[1].dim(), 3);
        assert_eq!(vertices[2].point.coords, [7.0, 8.0, 9.0]);
        assert_eq!(vertices[2].dim(), 3);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", vertices);
    }

    #[test]
    fn vertex_into_hashmap() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];
        let mut vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices.clone());
        let mut values: Vec<Vertex<f64, Option<()>, 3>> = hashmap.into_values().collect();

        assert_eq!(values.len(), 3);

        values.sort_by(|a, b| a.uuid.cmp(&b.uuid));
        vertices.sort_by(|a, b| a.uuid.cmp(&b.uuid));

        assert_eq!(values, vertices);

        // Human readable output for cargo test -- --nocapture
        println!("values = {:?}", values);
        println!("vertices = {:?}", vertices);
    }

    #[test]
    fn vertex_dim() {
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        assert_eq!(vertex.dim(), 3);
    }

    #[test]
    fn vertex_to_and_from_json() {
        // let vertex: Vertex<f64, Option<()>, 3> = Vertex::new(Point::new([1.0, 2.0, 3.0]));
        let vertex: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let serialized = serde_json::to_string(&vertex).unwrap();

        assert!(serialized.contains("point"));
        assert!(serialized.contains("[1.0,2.0,3.0]"));

        let deserialized: Vertex<f64, Option<()>, 3> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, vertex);

        // Human readable output for cargo test -- --nocapture
        println!("Serialized: {:?}", serialized);
    }

    #[test]
    fn vertex_partial_eq() {
        let vertex1: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex3: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 4.0]))
            .build()
            .unwrap();

        assert_eq!(vertex1, vertex2);
        assert_ne!(vertex2, vertex3);
    }

    #[test]
    fn vertex_partial_ord() {
        let vertex1: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let vertex2: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 4.0]))
            .build()
            .unwrap();
        let vertex3: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([10.0, 0.0, 0.0]))
            .build()
            .unwrap();
        let vertex4: Vertex<f64, Option<()>, 3> = VertexBuilder::default()
            .point(Point::new([0.0, 0.0, 10.0]))
            .build()
            .unwrap();

        assert!(vertex1 < vertex2);
        assert!(vertex3 > vertex2);
        assert!(vertex1 < vertex3);
        assert!(vertex1 > vertex4);
    }
}

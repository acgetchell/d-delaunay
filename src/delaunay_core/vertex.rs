//! Data and operations on d-dimensional [vertices](https://en.wikipedia.org/wiki/Vertex_(computer_graphics)).

use uuid::Uuid;

use std::option::Option;

use super::{point::Point, utilities::make_uuid};

#[derive(Debug, PartialEq, Clone, Copy)]
/// The `Vertex` struct represents a vertex in a triangulation with a `Point`, a unique
/// identifier, an optional incident cell identifier, and optional data.
///
/// # Properties:
///
/// * `point`: A generic type `Point<T, D>` representing the coordinates of the vertex in a
/// D-dimensional space.
/// * `uuid`: The `uuid` property is of type `Uuid` and represents a universally unique identifier for
/// the vertex. This can be used to uniquely identify the vertex in a graph or any other data structure.
/// * `incident_cell`: The `incident_cell` property is an optional `Uuid` that represents the UUID of
/// the cell that the vertex is incident to. This is calculated by the `delaunay_core::triangulation_data_structure::Tds`.
/// * `data`: The `data` property is an optional field that can hold any type `U`. It is used to store
/// additional data associated with the vertex.
pub struct Vertex<T, U, const D: usize> {
    /// The coordinates of the vertex in a D-dimensional space.
    pub point: Point<T, D>,
    /// A universally unique identifier for the vertex.
    pub uuid: Uuid,
    /// The UUID of the cell that the vertex is incident to.
    pub incident_cell: Option<Uuid>,
    /// Optional data associated with the vertex.
    pub data: Option<U>,
}

impl<T, U, const D: usize> Vertex<T, U, D> {
    /// The function `new_with_data` creates a new instance of a `Vertex` with the given point and data, and
    /// assigns a unique identifier to it.
    ///
    /// # Arguments:
    ///
    /// * `point`: A generic type `Point<T, D>` representing a point in a multi-dimensional space, where
    /// the coordinates are of type `T` and the dimensionality is `D`.
    /// * `data`: The `data` parameter is of type `U`.
    ///
    /// # Returns:
    ///
    /// The `new_with_data` function returns an instance of the `Vertex`.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// let vertex = Vertex::new_with_data(point, "4D");
    /// assert_eq!(vertex.point.coords, [1.0, 2.0, 3.0, 4.0]);
    /// assert_eq!(vertex.data.unwrap(), "4D");
    /// ```
    pub fn new_with_data(point: Point<T, D>, data: U) -> Self {
        let uuid = make_uuid();
        let incident_cell = None;
        let data = Some(data);
        Self {
            point,
            uuid,
            incident_cell,
            data,
        }
    }

    /// The function creates a new instance of a `Vertex`.
    ///
    /// # Arguments:
    ///
    /// * `point`: A generic type `Point<T, D>`, representing a point in a multi-dimensional space, where
    /// the coordinates are of type `T` and the dimensionality is `D`.
    ///
    /// # Returns:
    ///
    /// The `new` function returns an instance of the `Vertex`.
    ///
    /// # Example
    ///
    /// ```
    /// use d_delaunay::delaunay_core::vertex::Vertex;
    /// use d_delaunay::delaunay_core::point::Point;
    /// let point = Point::new([1.0, 2.0, 3.0, 4.0]);
    /// let vertex: Vertex<f64, Option<()>, 4> = Vertex::new(point);
    /// assert_eq!(vertex.point.coords, [1.0, 2.0, 3.0, 4.0]);
    /// ```
    pub fn new(point: Point<T, D>) -> Self {
        let uuid = make_uuid();
        let incident_cell = None;
        let data = None;
        Self {
            point,
            uuid,
            incident_cell,
            data,
        }
    }

    /// The function `from_points` takes a vector of points and returns a vector of vertices, using the `new` method.
    ///
    /// # Arguments:
    ///
    /// * `points`: `points` is a vector of `Point<T, D>` objects.
    ///
    /// # Returns:
    ///
    /// The function `from_points` returns a `Vec<Vertex<T, U, D>>`, where `T` is the type of the coordinates of the
    /// `Vertex`, `U` is the type of the optional data associated with the `Vertex`, and `D` is the dimensionality
    /// of the `Vertex`.
    pub fn from_points(points: Vec<Point<T, D>>) -> Vec<Self> {
        points.into_iter().map(|p| Self::new(p)).collect()
    }

    /// The function `into_hashmap` converts a vector of vertices into a hashmap, using the vectors' UUID
    /// as the key.
    ///
    /// # Arguments:
    ///
    /// * `vertices`: `vertices` is a vector of `Vertex<T, U, D>`.
    ///
    /// # Returns:
    ///
    /// The function `into_hashmap` returns a `std::collections::HashMap<Uuid, Vertex<T, U, D>`.
    pub fn into_hashmap(vertices: Vec<Self>) -> std::collections::HashMap<Uuid, Self> {
        vertices.into_iter().map(|v| (v.uuid, v)).collect()
    }

    /// The `dim` function returns the dimensionality of the `Vertex`.
    ///
    /// # Returns:
    ///
    /// The `dim` function is returning the value of `D`, which the number of coordinates.
    pub fn dim(&self) -> usize {
        D
    }

    pub fn is_valid(self) -> bool {
        todo!("Implement is_valid for Vertex")
    }
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn make_vertex_with_data() {
        let vertex = Vertex::new_with_data(Point::new([1.0, 2.0, 3.0, 4.0]), "4D");
        println!("{:?}", vertex);
        assert_eq!(vertex.point.coords[0], 1.0);
        assert_eq!(vertex.point.coords[1], 2.0);
        assert_eq!(vertex.point.coords[2], 3.0);
        assert_eq!(vertex.point.coords[3], 4.0);
        assert_eq!(vertex.dim(), 4);
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_some());
        assert_eq!(vertex.data.unwrap(), "4D");
    }

    #[test]
    fn make_vertex_without_data() {
        let vertex: Vertex<f64, Option<()>, 3> = Vertex::new(Point::new([1.0, 2.0, 3.0]));
        println!("{:?}", vertex);
        assert_eq!(vertex.point.coords[0], 1.0);
        assert_eq!(vertex.point.coords[1], 2.0);
        assert_eq!(vertex.point.coords[2], 3.0);
        assert_eq!(vertex.dim(), 3);
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_none());
    }

    #[test]
    fn make_vertices_from_points() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];
        let vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);

        assert_eq!(vertices.len(), 3);
        assert_eq!(vertices[0].point.coords[0], 1.0);
        assert_eq!(vertices[0].point.coords[1], 2.0);
        assert_eq!(vertices[0].point.coords[2], 3.0);
        assert_eq!(vertices[0].dim(), 3);
        assert_eq!(vertices[1].point.coords[0], 4.0);
        assert_eq!(vertices[1].point.coords[1], 5.0);
        assert_eq!(vertices[1].point.coords[2], 6.0);
        assert_eq!(vertices[1].dim(), 3);
        assert_eq!(vertices[2].point.coords[0], 7.0);
        assert_eq!(vertices[2].point.coords[1], 8.0);
        assert_eq!(vertices[2].point.coords[2], 9.0);
        assert_eq!(vertices[2].dim(), 3);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", vertices);
    }

    #[test]
    fn make_hashmap_from_vec() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];
        let mut vertices: Vec<Vertex<f64, Option<()>, 3>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices.clone());

        assert_eq!(hashmap.len(), 3);
        for (key, val) in hashmap.iter() {
            assert_eq!(*key, val.uuid);
        }

        let mut values: Vec<Vertex<f64, Option<()>, 3>> = hashmap.into_values().collect();
        assert_eq!(values.len(), 3);

        values.sort_by(|a, b| a.uuid.cmp(&b.uuid));
        vertices.sort_by(|a, b| a.uuid.cmp(&b.uuid));
        assert_eq!(values, vertices);

        // Human readable output for cargo test -- --nocapture
        println!("values = {:?}", values);
        println!("vertices = {:?}", vertices);
    }
}

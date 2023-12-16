use uuid::Uuid;

use std::option::Option;

use super::{point::Point, utilities::make_uuid};

#[derive(Debug, PartialEq, Clone)]
pub struct Vertex<T, U> {
    pub point: Point<T>,
    pub uuid: Uuid,
    pub incident_cell: Option<Uuid>,
    pub data: Option<U>,
}

impl<T, U> Vertex<T, U> {
    pub fn new_with_data(point: Point<T>, data: U) -> Self {
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

    pub fn new(point: Point<T>) -> Self {
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

    pub fn from_points(points: Vec<Point<T>>) -> Vec<Self> {
        points.into_iter().map(|p| Self::new(p)).collect()
    }

    pub fn into_hashmap(vertices: Vec<Self>) -> std::collections::HashMap<Uuid, Self> {
        vertices.into_iter().map(|v| (v.uuid, v)).collect()
    }
}
#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn make_vertex_with_data() {
        let vertex = Vertex::new_with_data(Point::new(1.0, 2.0, 3.0), 3);
        println!("{:?}", vertex);
        assert_eq!(vertex.point.x, 1.0);
        assert_eq!(vertex.point.y, 2.0);
        assert_eq!(vertex.point.z, 3.0);
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_some());
        assert_eq!(vertex.data.unwrap(), 3);
    }

    #[test]
    fn make_vertex_without_data() {
        let vertex: Vertex<f64, Option<()>> = Vertex::new(Point::new(1.0, 2.0, 3.0));
        println!("{:?}", vertex);
        assert_eq!(vertex.point.x, 1.0);
        assert_eq!(vertex.point.y, 2.0);
        assert_eq!(vertex.point.z, 3.0);
        assert!(vertex.incident_cell.is_none());
        assert!(vertex.data.is_none());
    }

    #[test]
    fn make_vertices_from_points() {
        let points = vec![
            Point::new(1.0, 2.0, 3.0),
            Point::new(4.0, 5.0, 6.0),
            Point::new(7.0, 8.0, 9.0),
        ];
        let vertices: Vec<Vertex<f64, Option<()>>> = Vertex::from_points(points);
        println!("{:?}", vertices);
        assert_eq!(vertices.len(), 3);
        assert_eq!(vertices[0].point.x, 1.0);
        assert_eq!(vertices[0].point.y, 2.0);
        assert_eq!(vertices[0].point.z, 3.0);
        assert_eq!(vertices[1].point.x, 4.0);
        assert_eq!(vertices[1].point.y, 5.0);
        assert_eq!(vertices[1].point.z, 6.0);
        assert_eq!(vertices[2].point.x, 7.0);
        assert_eq!(vertices[2].point.y, 8.0);
        assert_eq!(vertices[2].point.z, 9.0);
    }

    #[test]
    fn make_hashmap_from_vec() {
        let points = vec![
            Point::new(1.0, 2.0, 3.0),
            Point::new(4.0, 5.0, 6.0),
            Point::new(7.0, 8.0, 9.0),
        ];
        let mut vertices: Vec<Vertex<f64, Option<()>>> = Vertex::from_points(points);
        let hashmap = Vertex::into_hashmap(vertices.clone());
        println!("{:?}", hashmap);
        assert_eq!(hashmap.len(), 3);
        for (key, val) in hashmap.iter() {
            assert_eq!(*key, val.uuid);
        }

        let mut values: Vec<Vertex<f64, Option<()>>> = hashmap.into_values().collect();
        assert_eq!(values.len(), 3);

        values.sort_by(|a, b| a.uuid.cmp(&b.uuid));
        vertices.sort_by(|a, b| a.uuid.cmp(&b.uuid));
        assert_eq!(values, vertices);

        // Human readable output for cargo test -- --nocapture
        println!("values = {:?}", values);
        println!("vertices = {:?}", vertices);
    }
}

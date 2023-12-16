use uuid::Uuid;

use std::option::Option;

use super::{point::Point, utilities::make_uuid};

#[derive(Debug)]
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
}

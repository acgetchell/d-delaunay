use uuid::Uuid;

use std::option::Option;

use super::{point::Point, utilities::make_uuid};

#[derive(Debug)]
pub struct Vertex<T, U> {
    pub point: Point<T>,
    pub uuid: Uuid,
    pub incident_cell: Option<Uuid>,
    pub data: U,
}

impl<T, U> Vertex<T, U> {
    pub fn new(point: Point<T>, data: U) -> Self {
        let uuid = make_uuid();
        let incident_cell = None;
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
    fn make_vertex() {
        let vertex = Vertex::new(Point::new(1.0, 2.0, 3.0), 3);
        println!("{:?}", vertex);
        assert_eq!(vertex.point.x, 1.0);
        assert_eq!(vertex.point.y, 2.0);
        assert_eq!(vertex.point.z, 3.0);
        assert_eq!(vertex.uuid.get_version_num(), 4);
        println!("uuid version: {:?}\n", vertex.uuid.get_version_num());
        assert_ne!(vertex.uuid, make_uuid());
        assert!(vertex.incident_cell.is_none());
        assert_eq!(vertex.data, 3);
    }
}

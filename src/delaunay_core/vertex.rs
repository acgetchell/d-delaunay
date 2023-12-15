use uuid::Uuid;

use super::{point::Point, utilities::make_uuid};

#[allow(dead_code)]
#[derive(Debug)]
pub struct Vertex<T, U> {
    pub point: Point<T>,
    pub uuid: Uuid,
    pub data: U,
}

#[allow(dead_code)]
impl<T, U> Vertex<T, U> {
    pub fn new(point: Point<T>, data: U) -> Self {
        let uuid = make_uuid();
        Self { point, uuid, data }
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
        assert_ne!(vertex.uuid, make_uuid());
        assert_eq!(vertex.data, 3);
    }
}

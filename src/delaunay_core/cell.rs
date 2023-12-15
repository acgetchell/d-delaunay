use uuid::Uuid;

use super::{utilities::make_uuid, vertex::Vertex};

#[allow(dead_code)]
#[derive(Debug)]
struct Cell<T, U, V> {
    pub vertices: Vec<Vertex<U, V>>,
    uuid: Uuid,
    pub data: T,
}

#[allow(dead_code)]
impl<T, U, V> Cell<T, U, V> {
    pub fn new(vertices: Vec<Vertex<U, V>>, data: T) -> Self {
        let uuid = make_uuid();
        Cell {
            vertices,
            uuid,
            data,
        }
    }

    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }
}

#[cfg(test)]
mod tests {

    use crate::delaunay_core::point::Point;

    use super::*;

    #[test]
    fn make_cell() {
        let vertex1 = Vertex::new(Point::new(1.0, 2.0, 3.0), 3);
        let cell = Cell::new(vec![vertex1], 10);
        println!("{:?}", cell);
        assert_eq!(cell.vertices[0].point.x, 1.0);
        assert_eq!(cell.vertices[0].point.y, 2.0);
        assert_eq!(cell.vertices[0].point.z, 3.0);
        assert_ne!(cell.vertices[0].uuid, make_uuid());
        assert_eq!(cell.vertices[0].data, 3);
        assert_eq!(cell.number_of_vertices(), 1);
        assert_ne!(cell.uuid, make_uuid());
        assert_ne!(cell.uuid, cell.vertices[0].uuid);
        assert_eq!(cell.data, 10);
    }
}

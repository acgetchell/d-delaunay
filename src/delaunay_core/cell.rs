use uuid::Uuid;

use super::{utilities::make_uuid, vertex::Vertex};

#[derive(Debug)]
pub struct Cell<T, U, V> {
    pub vertices: Vec<Vertex<T, U>>,
    pub uuid: Uuid,
    pub neighbors: Option<Vec<Uuid>>,
    pub data: Option<V>,
}

impl<T, U, V> Cell<T, U, V> {
    pub fn new_with_data(vertices: Vec<Vertex<T, U>>, data: V) -> Self {
        let uuid = make_uuid();
        let neighbors = None;
        let data = Some(data);
        Cell {
            vertices,
            uuid,
            neighbors,
            data,
        }
    }

    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn new(vertices: Vec<Vertex<T, U>>) -> Self {
        let uuid = make_uuid();
        let neighbors = None;
        let data = None;
        Cell {
            vertices,
            uuid,
            neighbors,
            data,
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::delaunay_core::point::Point;

    use super::*;

    #[test]
    fn make_cell_with_data() {
        let vertex1 = Vertex::new_with_data(Point::new(1.0, 2.0, 3.0), 3);
        let cell = Cell::new_with_data(vec![vertex1], 10);

        assert_eq!(cell.vertices[0].point.x, 1.0);
        assert_eq!(cell.vertices[0].point.y, 2.0);
        assert_eq!(cell.vertices[0].point.z, 3.0);
        assert_eq!(cell.vertices[0].data, Some(3));
        assert_eq!(cell.number_of_vertices(), 1);
        assert!(cell.neighbors.is_none());
        assert!(cell.data.is_some());
        assert_eq!(cell.data.unwrap(), 10);

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }

    #[test]
    fn make_cell_without_data() {
        let vertex1 = Vertex::new_with_data(Point::new(1.0, 2.0, 3.0), 3);
        let cell: Cell<f64, i32, Option<()>> = Cell::new(vec![vertex1]);

        assert_eq!(cell.vertices[0].point.x, 1.0);
        assert_eq!(cell.vertices[0].point.y, 2.0);
        assert_eq!(cell.vertices[0].point.z, 3.0);
        assert_eq!(cell.vertices[0].data, Some(3));
        assert_eq!(cell.number_of_vertices(), 1);
        assert!(cell.neighbors.is_none());
        assert!(cell.data.is_none());

        // Human readable output for cargo test -- --nocapture
        println!("Cell: {:?}", cell);
    }
}

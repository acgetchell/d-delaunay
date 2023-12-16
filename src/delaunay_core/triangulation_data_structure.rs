use super::{cell::Cell, point::Point, vertex::Vertex};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug)]
pub struct Tds<T, U, V> {
    pub vertices: HashMap<Uuid, Vertex<U, V>>,
    pub cells: HashMap<Uuid, Cell<T, U, V>>,
}

impl<T, U, V> Tds<T, U, V> {
    pub fn new(points: Vec<Point<U>>) -> Self {
        let vertices = Vertex::into_hashmap(Vertex::from_points(points));
        let cells = HashMap::new();
        Self { vertices, cells }
    }
}

pub fn hello() -> i32 {
    println!("Hello, world!");
    1
}

#[cfg(test)]
mod tests {

    use crate::delaunay_core::triangulation_data_structure;

    use super::*;

    #[test]
    fn make_tds() {
        let points = vec![
            Point::new(1.0, 2.0, 3.0),
            Point::new(4.0, 5.0, 6.0),
            Point::new(7.0, 8.0, 9.0),
            Point::new(10.0, 11.0, 12.0),
        ];
        let tds: triangulation_data_structure::Tds<usize, f64, usize> = Tds::new(points);

        assert_eq!(tds.vertices.len(), 4);
        assert_eq!(tds.cells.len(), 0);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", tds);
    }
}

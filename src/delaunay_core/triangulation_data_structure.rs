use super::{cell::Cell, point::Point, vertex::Vertex};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug)]
pub struct Tds<T, U, V> {
    pub vertices: HashMap<Uuid, Vertex<U, V>>,
    pub cells: HashMap<Uuid, Cell<T, U, V>>,
}

impl<T, U, V> Tds<T, U, V> {
    pub fn new(_points: Vec<Point<U>>) -> Self {
        let vertices = HashMap::new();
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
        let tds: triangulation_data_structure::Tds<usize, f64, usize> =
            Tds::new(vec![Point::new(1.0, 2.0, 3.0)]);
        println!("{:?}", tds);
        assert_eq!(tds.vertices.len(), 0);
        assert_eq!(tds.cells.len(), 0);
    }
}

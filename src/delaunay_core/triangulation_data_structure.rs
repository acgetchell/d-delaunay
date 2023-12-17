use super::{cell::Cell, point::Point, vertex::Vertex};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug)]
pub struct Tds<T, U, V, const D: usize> {
    pub vertices: HashMap<Uuid, Vertex<T, U, D>>,
    pub cells: HashMap<Uuid, Cell<T, U, V, D>>,
}

impl<T, U, V, const D: usize> Tds<T, U, V, D> {
    pub fn new(points: Vec<Point<T, D>>) -> Self {
        let vertices = Vertex::into_hashmap(Vertex::from_points(points));
        let cells = HashMap::new();
        Self { vertices, cells }
    }
}

pub fn start() -> i32 {
    println!("Starting ...");
    1
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn make_tds() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];

        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.vertices.len(), 4);
        assert_eq!(tds.cells.len(), 0);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", tds);
    }
}

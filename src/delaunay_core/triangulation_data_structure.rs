use super::{cell::Cell, point::Point, vertex::Vertex};
use std::cmp::PartialEq;
use std::{cmp::min, collections::HashMap};
use uuid::Uuid;

#[derive(Debug, Clone)]
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

    pub fn add(&mut self, vertex: Vertex<T, U, D>) -> Result<(), &'static str>
    where
        T: PartialEq,
    {
        // Don't add if vertex with that point already exists
        for val in self.vertices.values() {
            if val.point.coords == vertex.point.coords {
                return Err("Vertex already exists");
            }
        }

        let result = self.vertices.insert(vertex.uuid, vertex);

        // Hashmap::insert returns the old value if the key already exists and updates it with the new value
        match result {
            Some(_) => Err("Uuid already exists"),
            None => Ok(()),
        }
    }

    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    pub fn dim(&self) -> i32 {
        let len = self.number_of_vertices() as i32;

        min(len - 1, D as i32)
    }

    pub fn number_of_cells(&self) -> usize {
        self.cells.len()
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
    fn tds_new() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];

        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), 3);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", tds);
    }

    #[test]
    fn tds_add_dim() {
        let points: Vec<Point<f64, 3>> = Vec::new();

        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), -1);

        let new_vertex1: Vertex<f64, usize, 3> = Vertex::new(Point::new([1.0, 2.0, 3.0]));
        let _ = tds.add(new_vertex1);
        assert_eq!(tds.number_of_vertices(), 1);
        assert_eq!(tds.dim(), 0);

        let new_vertex2: Vertex<f64, usize, 3> = Vertex::new(Point::new([4.0, 5.0, 6.0]));
        let _ = tds.add(new_vertex2);
        assert_eq!(tds.number_of_vertices(), 2);
        assert_eq!(tds.dim(), 1);

        let new_vertex3: Vertex<f64, usize, 3> = Vertex::new(Point::new([7.0, 8.0, 9.0]));
        let _ = tds.add(new_vertex3);
        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.dim(), 2);

        let new_vertex4: Vertex<f64, usize, 3> = Vertex::new(Point::new([10.0, 11.0, 12.0]));
        let _ = tds.add(new_vertex4);
        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.dim(), 3);

        let new_vertex5: Vertex<f64, usize, 3> = Vertex::new(Point::new([13.0, 14.0, 15.0]));
        let _ = tds.add(new_vertex5);
        assert_eq!(tds.number_of_vertices(), 5);
        assert_eq!(tds.dim(), 3);
    }

    #[test]
    fn tds_no_add() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];

        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.cells.len(), 0);
        assert_eq!(tds.dim(), 3);

        let new_vertex1: Vertex<f64, usize, 3> = Vertex::new(Point::new([1.0, 2.0, 3.0]));
        let result = tds.add(new_vertex1);
        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.dim(), 3);

        assert!(result.is_err());
    }
}

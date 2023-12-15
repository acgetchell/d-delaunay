use crate::delaunay_core::vertex::Vertex;

#[allow(dead_code)]
#[derive(Debug)]
struct Cell<T, U, V> {
    pub vertices: Vec<Vertex<U, V>>,
    index: usize,
    pub data: T,
}

#[allow(dead_code)]
impl<T, U, V> Cell<T, U, V> {
    pub fn new(vertices: Vec<Vertex<U, V>>, index: usize, data: T) -> Self {
        Cell {
            vertices,
            index,
            data,
        }
    }
}

mod tests {

    #[test]
    fn make_cell() {
        use crate::delaunay_core::cell::Cell;
        use crate::delaunay_core::vertex::Vertex;
        let vertex1 = Vertex::new(
            crate::delaunay_core::vertex::Point::new(1.0, 2.0, 3.0),
            0,
            3,
        );
        let cell = Cell::new(vec![vertex1], 5, 10);
        println!("{:?}", cell);
        assert_eq!(cell.vertices[0].point.x, 1.0);
        assert_eq!(cell.vertices[0].point.y, 2.0);
        assert_eq!(cell.vertices[0].point.z, 3.0);
        assert_eq!(cell.vertices[0].index, 0);
        assert_eq!(cell.vertices[0].data, 3);
        assert_eq!(cell.index, 5);
        assert_eq!(cell.data, 10);
    }
}

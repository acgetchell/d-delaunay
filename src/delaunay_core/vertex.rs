#[allow(dead_code)]
#[derive(Debug)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[allow(dead_code)]
impl<T> Point<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Vertex<T, U> {
    pub point: Point<T>,
    pub index: usize,
    pub data: U,
}

#[allow(dead_code)]
impl<T, U> Vertex<T, U> {
    pub fn new(point: Point<T>, index: usize, data: U) -> Self {
        Self { point, index, data }
    }
}

mod tests {

    #[test]
    fn make_vertex() {
        use crate::delaunay_core::vertex::Vertex;

        let vertex = Vertex::new(
            crate::delaunay_core::vertex::Point::new(1.0, 2.0, 3.0),
            0,
            3,
        );
        println!("{:?}", vertex);
        assert_eq!(vertex.point.x, 1.0);
        assert_eq!(vertex.point.y, 2.0);
        assert_eq!(vertex.point.z, 3.0);
        assert_eq!(vertex.index, 0);
        assert_eq!(vertex.data, 3);
    }
}

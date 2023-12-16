#[derive(Debug, PartialEq, Clone)]
pub struct Point<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl<T> Point<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn make_point() {
        let point = Point::new(1.0, 2.0, 3.0);

        assert_eq!(point.x, 1.0);
        assert_eq!(point.y, 2.0);
        assert_eq!(point.z, 3.0);

        // Human readable output for cargo test -- --nocapture
        println!("Point: {:?}", point);
    }
}

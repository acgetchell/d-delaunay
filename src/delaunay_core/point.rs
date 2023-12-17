#[derive(Debug, PartialEq, Clone)]
pub struct Point<T, const D: usize> {
    pub coords: [T; D],
}

impl<T: Clone, const D: usize> Point<T, D> {
    pub fn new(coords: [T; D]) -> Self {
        Self { coords }
    }

    pub fn dim(&self) -> usize {
        D
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn make_point() {
        let point = Point::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(point.coords[0], 1.0);
        assert_eq!(point.coords[1], 2.0);
        assert_eq!(point.coords[2], 3.0);
        assert_eq!(point.dim(), 4);

        // Human readable output for cargo test -- --nocapture
        println!("Point: {:?} is {}-D", point, point.dim());
    }
}

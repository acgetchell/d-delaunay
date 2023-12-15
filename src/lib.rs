mod delaunay_core {
    pub mod cell;
    pub mod point;
    pub mod triangulation_data_structure;
    pub mod utilities;
    pub mod vertex;
}

#[cfg(test)]
mod tests {
    use crate::delaunay_core::triangulation_data_structure::tds;

    #[test]
    fn it_works() {
        let result = tds();
        assert_eq!(result, 1);
    }
}

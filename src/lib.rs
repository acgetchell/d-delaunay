mod delaunay_core {
    pub mod triangulation_data_structure;
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

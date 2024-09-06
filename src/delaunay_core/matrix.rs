#![allow(missing_docs)]
// use peroxide::fuga::*;


// pub struct Matrix {
//     pub matrix: peroxide::prelude::Matrix,
// }

// impl Default for Matrix {
//     fn default() -> Self {
//         Self {
//             matrix: peroxide::prelude::Matrix::default()
//         }
//     }
// }

#[cfg(test)]
mod tests {
    
        use peroxide::fuga::*;
        use peroxide::c;

        // use super::*;
    
        #[test]
        fn matrix_default() {
            let matrix: Matrix = Default::default();
    
            assert_eq!(matrix.data, vec![0.0; 0]);
    
            // Human readable output for cargo test -- --nocapture
            matrix.print();
        }
    
        #[test]
        fn matrix_new() {
            let matrix = matrix(c!(1,2,3,4), 2, 2, Row);
    
            assert_eq!(matrix.data, vec![1.0, 2.0, 3.0, 4.0]);
    
            // Human readable output for cargo test -- --nocapture
            matrix.print();
        }
    
        #[test]
        fn matrix_copy() {
            let matrix = matrix(c!(1,2,3,4), 2, 2, Row);
            let matrix_copy = matrix.clone();
            
            assert_eq!(matrix, matrix_copy);
        }
    
        #[test]
        fn matrix_dim() {
            let matrix = matrix(c!(1,2,3,4), 2, 2, Row);
    
            assert_eq!(matrix.col, 2);
            assert_eq!(matrix.row, 2);
    
            // Human readable output for cargo test -- --nocapture
            matrix.print();
        }
    
        #[test]
        fn matrix_identity() {
            let matrix: Matrix = eye(3);
    
            assert_eq!(matrix.data, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    
            // Human readable output for cargo test -- --nocapture
            matrix.print();
        }
    
        // #[test]
        // fn matrix_serialization() {
        //     let matrix = matrix(c!(1,2,3,4), 2, 2, Row);
        //     let serialized = serde_json::to_string(&matrix).unwrap();
        //     let deserialized: Matrix = serde_json::from_str(&serialized).unwrap();
    
        //     assert_eq!(matrix, deserialized);
    
        //     // Human readable output for cargo test -- --nocapture
        //     println!("Matrix: {:?}", matrix);
        //     println!("Serialized: {}", serialized);
        //     println!("Deserialized: {:?}", deserialized);
        // }

}
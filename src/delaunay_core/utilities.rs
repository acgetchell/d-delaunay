//! Utility functions

use uuid::Uuid;

/// The function `make_uuid` generates a version 4 [Uuid].
///
/// # Returns
///
/// a randomly generated [Uuid] (Universally Unique Identifier) using the
/// `new_v4` method from the [Uuid] struct.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::utilities::make_uuid;
/// let uuid = make_uuid();
/// assert_eq!(uuid.get_version_num(), 4);
/// ```
#[must_use]
pub fn make_uuid() -> Uuid {
    Uuid::new_v4()
}

/// The function `vec_to_array` converts a [Vec] to an array of f64
///
/// # Errors
///
/// Returns an error if the vector length does not match the target array dimension `D`.
///
/// # Example
///
/// ```
/// use d_delaunay::delaunay_core::utilities::vec_to_array;
/// let vec = vec![1.0, 2.0, 3.0];
/// let array = vec_to_array::<3>(&vec).unwrap();
/// # use approx::assert_relative_eq;
/// assert_relative_eq!(array.as_slice(), [1.0, 2.0, 3.0].as_slice(), epsilon = 1e-9);
/// ```
pub fn vec_to_array<const D: usize>(vec: &[f64]) -> Result<[f64; D], anyhow::Error> {
    if vec.len() != D {
        return Err(anyhow::Error::msg(
            "Vector length does not match array dimension!",
        ));
    }
    let array: [f64; D] = std::array::from_fn(|i| vec[i]);

    Ok(array)
}

#[cfg(test)]
mod tests {

    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn utilities_uuid() {
        let uuid = make_uuid();

        assert_eq!(uuid.get_version_num(), 4);
        assert_ne!(uuid, make_uuid());

        // Human readable output for cargo test -- --nocapture
        println!("make_uuid = {uuid:?}");
        println!("uuid version: {:?}\n", uuid.get_version_num());
    }

    #[test]
    fn utilities_vec_to_array_success() {
        let vec = vec![1.0, 2.0, 3.0];
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(array.as_slice(), [1.0, 2.0, 3.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn utilities_vec_to_array_wrong_length() {
        let vec = vec![1.0, 2.0]; // Length 2, but expecting 3
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Vector length does not match array dimension!"
        );
    }

    #[test]
    fn utilities_vec_to_array_empty() {
        let vec: Vec<f64> = vec![];
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Vector length does not match array dimension!"
        );
    }

    #[test]
    fn utilities_vec_to_array_too_long() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Length 5, but expecting 3
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_err());
        let error = result.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Vector length does not match array dimension!"
        );
    }

    #[test]
    fn utilities_vec_to_array_1d() {
        let vec = vec![42.0];
        let result = vec_to_array::<1>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(array.as_slice(), [42.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn utilities_vec_to_array_large_dimension() {
        let vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = vec_to_array::<10>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(
            array.as_slice(),
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn utilities_vec_to_array_negative_values() {
        let vec = vec![-1.0, -2.0, -3.0];
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(
            array.as_slice(),
            [-1.0, -2.0, -3.0].as_slice(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn utilities_vec_to_array_zero() {
        let vec = vec![0.0, 0.0, 0.0];
        let result = vec_to_array::<3>(&vec);

        assert!(result.is_ok());
        let array = result.unwrap();
        assert_relative_eq!(array.as_slice(), [0.0, 0.0, 0.0].as_slice(), epsilon = 1e-9);
    }

    #[test]
    fn utilities_make_uuid_uniqueness() {
        let uuid1 = make_uuid();
        let uuid2 = make_uuid();
        let uuid3 = make_uuid();

        // All UUIDs should be different
        assert_ne!(uuid1, uuid2);
        assert_ne!(uuid1, uuid3);
        assert_ne!(uuid2, uuid3);

        // All should be version 4
        assert_eq!(uuid1.get_version_num(), 4);
        assert_eq!(uuid2.get_version_num(), 4);
        assert_eq!(uuid3.get_version_num(), 4);
    }

    #[test]
    fn utilities_make_uuid_format() {
        let uuid = make_uuid();
        let uuid_string = uuid.to_string();

        // UUID should have proper format: 8-4-4-4-12 characters
        assert_eq!(uuid_string.len(), 36); // Including hyphens
        assert_eq!(uuid_string.chars().filter(|&c| c == '-').count(), 4);

        // Should be valid hyphenated format
        let parts: Vec<&str> = uuid_string.split('-').collect();
        assert_eq!(parts.len(), 5);
        assert_eq!(parts[0].len(), 8);
        assert_eq!(parts[1].len(), 4);
        assert_eq!(parts[2].len(), 4);
        assert_eq!(parts[3].len(), 4);
        assert_eq!(parts[4].len(), 12);
    }
}

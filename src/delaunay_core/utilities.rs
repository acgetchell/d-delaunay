//! Utility functions

use uuid::Uuid;

/// The function `make_uuid` generates a version 4 UUID in Rust.
///
/// # Returns:
///
/// a randomly generated UUID (Universally Unique Identifier) using the `new_v4` method from the `Uuid`
/// struct.
pub fn make_uuid() -> Uuid {
    Uuid::new_v4()
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_uuid() {
        let uuid = make_uuid();

        assert_eq!(uuid.get_version_num(), 4);
        assert_ne!(uuid, make_uuid());

        // Human readable output for cargo test -- --nocapture
        println!("make_uuid = {:?}", uuid);
        println!("uuid version: {:?}\n", uuid.get_version_num());
    }
}

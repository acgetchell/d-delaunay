//! Example demonstrating generic scalar type support in Point struct.
//!
//! This example shows how to use the Point struct with different scalar types
//! (f32 and f64) and demonstrates the various type aliases available.

use d_delaunay::geometry::point::{Point, PointF32, PointF64, PointND};
use nalgebra::SVector;

fn main() {
    // Test f64 support (default)
    let point_f64: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
    println!("f64 point: {:?}", point_f64);
    println!("f64 coordinates: {:?}", point_f64.coordinates());

    // Test f32 support
    let point_f32: PointF32<3> = PointF32::new([1.0f32, 2.0f32, 3.0f32]);
    println!("f32 point: {:?}", point_f32);
    println!("f32 coordinates: {:?}", point_f32.coordinates());

    // Test using the generic Point directly
    let point_generic_f64: Point<f64, SVector<f64, 2>, 2> = Point::new([4.0, 5.0]);
    println!("Generic f64 point: {:?}", point_generic_f64);

    let point_generic_f32: Point<f32, SVector<f32, 2>, 2> = Point::new([4.0f32, 5.0f32]);
    println!("Generic f32 point: {:?}", point_generic_f32);

    // Test validation
    println!("f64 point validation: {:?}", point_f64.is_valid());
    println!("f32 point validation: {:?}", point_f32.is_valid());

    // Test NaN validation
    let nan_f64: PointND<2> = PointND::new([1.0, f64::NAN]);
    println!("NaN f64 point validation: {:?}", nan_f64.is_valid());

    let nan_f32: PointF32<2> = PointF32::new([1.0f32, f32::NAN]);
    println!("NaN f32 point validation: {:?}", nan_f32.is_valid());

    // Test type aliases
    let point_f64_alias: PointF64<3> = PointF64::new([1.0, 2.0, 3.0]);
    println!("f64 alias point: {:?}", point_f64_alias);

    println!("All tests completed successfully!");
}

#[cfg(test)]
mod tests {
    use d_delaunay::geometry::point::{Point, PointF32, PointF64, PointND, PointValidationError};
    use nalgebra::SVector;

    #[test]
    fn test_f64_point_creation() {
        let point: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
        assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
        assert!(point.is_valid().is_ok());
    }

    #[test]
    fn test_f32_point_creation() {
        let point: PointF32<3> = PointF32::new([1.0f32, 2.0f32, 3.0f32]);
        assert_eq!(point.coordinates(), [1.0f32, 2.0f32, 3.0f32]);
        assert!(point.is_valid().is_ok());
    }

    #[test]
    fn test_generic_point_creation() {
        let point_f64: Point<f64, SVector<f64, 2>, 2> = Point::new([4.0, 5.0]);
        assert_eq!(point_f64.coordinates(), [4.0, 5.0]);
        assert!(point_f64.is_valid().is_ok());

        let point_f32: Point<f32, SVector<f32, 2>, 2> = Point::new([4.0f32, 5.0f32]);
        assert_eq!(point_f32.coordinates(), [4.0f32, 5.0f32]);
        assert!(point_f32.is_valid().is_ok());
    }

    #[test]
    fn test_type_aliases() {
        let point_f64: PointF64<3> = PointF64::new([1.0, 2.0, 3.0]);
        let point_nd: PointND<3> = PointND::new([1.0, 2.0, 3.0]);

        // Both should be equivalent
        assert_eq!(point_f64.coordinates(), point_nd.coordinates());
        assert_eq!(point_f64.is_valid(), point_nd.is_valid());
    }

    #[test]
    fn test_different_dimensions() {
        // Test 1D
        let point_1d: PointND<1> = PointND::new([42.0]);
        assert_eq!(point_1d.coordinates(), [42.0]);
        assert!(point_1d.is_valid().is_ok());

        // Test 2D
        let point_2d: PointND<2> = PointND::new([1.0, 2.0]);
        assert_eq!(point_2d.coordinates(), [1.0, 2.0]);
        assert!(point_2d.is_valid().is_ok());

        // Test 4D
        let point_4d: PointND<4> = PointND::new([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(point_4d.coordinates(), [1.0, 2.0, 3.0, 4.0]);
        assert!(point_4d.is_valid().is_ok());

        // Test 5D
        let point_5d: PointND<5> = PointND::new([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(point_5d.coordinates(), [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(point_5d.is_valid().is_ok());
    }

    #[test]
    fn test_f64_validation() {
        // Valid point
        let valid_point: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
        assert!(valid_point.is_valid().is_ok());

        // NaN point
        let nan_point: PointND<2> = PointND::new([1.0, f64::NAN]);
        assert!(nan_point.is_valid().is_err());
        if let Err(e) = nan_point.is_valid() {
            assert!(matches!(e, PointValidationError::InvalidCoordinate { .. }));
        }

        // Infinity point
        let inf_point: PointND<2> = PointND::new([1.0, f64::INFINITY]);
        assert!(inf_point.is_valid().is_err());
        if let Err(e) = inf_point.is_valid() {
            assert!(matches!(e, PointValidationError::InvalidCoordinate { .. }));
        }

        // Negative infinity point
        let neg_inf_point: PointND<2> = PointND::new([1.0, f64::NEG_INFINITY]);
        assert!(neg_inf_point.is_valid().is_err());
        if let Err(e) = neg_inf_point.is_valid() {
            assert!(matches!(e, PointValidationError::InvalidCoordinate { .. }));
        }
    }

    #[test]
    fn test_f32_validation() {
        // Valid point
        let valid_point: PointF32<3> = PointF32::new([1.0f32, 2.0f32, 3.0f32]);
        assert!(valid_point.is_valid().is_ok());

        // NaN point
        let nan_point: PointF32<2> = PointF32::new([1.0f32, f32::NAN]);
        assert!(nan_point.is_valid().is_err());
        if let Err(e) = nan_point.is_valid() {
            assert!(matches!(e, PointValidationError::InvalidCoordinate { .. }));
        }

        // Infinity point
        let inf_point: PointF32<2> = PointF32::new([1.0f32, f32::INFINITY]);
        assert!(inf_point.is_valid().is_err());
        if let Err(e) = inf_point.is_valid() {
            assert!(matches!(e, PointValidationError::InvalidCoordinate { .. }));
        }

        // Negative infinity point
        let neg_inf_point: PointF32<2> = PointF32::new([1.0f32, f32::NEG_INFINITY]);
        assert!(neg_inf_point.is_valid().is_err());
        if let Err(e) = neg_inf_point.is_valid() {
            assert!(matches!(e, PointValidationError::InvalidCoordinate { .. }));
        }
    }

    #[test]
    fn test_edge_values() {
        // Test with zero coordinates
        let zero_f64: PointND<3> = PointND::new([0.0, 0.0, 0.0]);
        assert!(zero_f64.is_valid().is_ok());
        assert_eq!(zero_f64.coordinates(), [0.0, 0.0, 0.0]);

        let zero_f32: PointF32<3> = PointF32::new([0.0f32, 0.0f32, 0.0f32]);
        assert!(zero_f32.is_valid().is_ok());
        assert_eq!(zero_f32.coordinates(), [0.0f32, 0.0f32, 0.0f32]);

        // Test with negative coordinates
        let neg_f64: PointND<3> = PointND::new([-1.0, -2.0, -3.0]);
        assert!(neg_f64.is_valid().is_ok());
        assert_eq!(neg_f64.coordinates(), [-1.0, -2.0, -3.0]);

        let neg_f32: PointF32<3> = PointF32::new([-1.0f32, -2.0f32, -3.0f32]);
        assert!(neg_f32.is_valid().is_ok());
        assert_eq!(neg_f32.coordinates(), [-1.0f32, -2.0f32, -3.0f32]);

        // Test with very small values
        let small_f64: PointND<2> = PointND::new([1e-300, 2e-300]);
        assert!(small_f64.is_valid().is_ok());
        assert_eq!(small_f64.coordinates(), [1e-300, 2e-300]);

        let small_f32: PointF32<2> = PointF32::new([1e-30f32, 2e-30f32]);
        assert!(small_f32.is_valid().is_ok());
        assert_eq!(small_f32.coordinates(), [1e-30f32, 2e-30f32]);

        // Test with very large values
        let large_f64: PointND<2> = PointND::new([1e300, 2e300]);
        assert!(large_f64.is_valid().is_ok());
        assert_eq!(large_f64.coordinates(), [1e300, 2e300]);

        let large_f32: PointF32<2> = PointF32::new([1e30f32, 2e30f32]);
        assert!(large_f32.is_valid().is_ok());
        assert_eq!(large_f32.coordinates(), [1e30f32, 2e30f32]);
    }

    #[test]
    fn test_precision_differences() {
        // Test that f32 and f64 maintain their respective precision
        let f64_point: PointND<1> = PointND::new([1.23456789012345]);
        let f32_point: PointF32<1> = PointF32::new([1.23456789012345f32]);

        // f64 should maintain more precision
        assert_eq!(f64_point.coordinates()[0], 1.23456789012345);
        // f32 will have less precision due to its nature
        assert_eq!(f32_point.coordinates()[0], 1.2345679f32);
    }

    #[test]
    fn test_mixed_scalar_operations() {
        // Test that we can create points with different scalar types
        let points_f64: Vec<PointND<2>> = vec![
            PointND::new([1.0, 2.0]),
            PointND::new([3.0, 4.0]),
            PointND::new([5.0, 6.0]),
        ];

        let points_f32: Vec<PointF32<2>> = vec![
            PointF32::new([1.0f32, 2.0f32]),
            PointF32::new([3.0f32, 4.0f32]),
            PointF32::new([5.0f32, 6.0f32]),
        ];

        // All points should be valid
        for point in &points_f64 {
            assert!(point.is_valid().is_ok());
        }

        for point in &points_f32 {
            assert!(point.is_valid().is_ok());
        }
    }

    #[test]
    fn test_coordinate_access() {
        let point_f64: PointND<4> = PointND::new([1.0, 2.0, 3.0, 4.0]);
        let coords_f64 = point_f64.coordinates();
        assert_eq!(coords_f64.len(), 4);
        assert_eq!(coords_f64[0], 1.0);
        assert_eq!(coords_f64[1], 2.0);
        assert_eq!(coords_f64[2], 3.0);
        assert_eq!(coords_f64[3], 4.0);

        let point_f32: PointF32<4> = PointF32::new([1.0f32, 2.0f32, 3.0f32, 4.0f32]);
        let coords_f32 = point_f32.coordinates();
        assert_eq!(coords_f32.len(), 4);
        assert_eq!(coords_f32[0], 1.0f32);
        assert_eq!(coords_f32[1], 2.0f32);
        assert_eq!(coords_f32[2], 3.0f32);
        assert_eq!(coords_f32[3], 4.0f32);
    }

    #[test]
    fn test_debug_display() {
        let point_f64: PointND<2> = PointND::new([1.0, 2.0]);
        let debug_str = format!("{:?}", point_f64);
        assert!(debug_str.contains("Point"));

        let point_f32: PointF32<2> = PointF32::new([1.0f32, 2.0f32]);
        let debug_str = format!("{:?}", point_f32);
        assert!(debug_str.contains("Point"));
    }

    #[test]
    fn test_consistency_across_types() {
        // Test that equivalent values in f32 and f64 behave consistently
        let f64_point: PointND<3> = PointND::new([1.0, 2.0, 3.0]);
        let f32_point: PointF32<3> = PointF32::new([1.0f32, 2.0f32, 3.0f32]);

        // Both should be valid
        assert!(f64_point.is_valid().is_ok());
        assert!(f32_point.is_valid().is_ok());

        // Coordinates should be equivalent (within f32 precision)
        let f64_coords = f64_point.coordinates();
        let f32_coords = f32_point.coordinates();

        for i in 0..3 {
            assert_eq!(f64_coords[i] as f32, f32_coords[i]);
        }
    }

    #[test]
    fn test_large_dimensions() {
        // Test with larger dimensions to ensure the generic system works
        let point_10d: PointND<10> =
            PointND::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        assert!(point_10d.is_valid().is_ok());
        assert_eq!(point_10d.coordinates().len(), 10);

        let point_f32_10d: PointF32<10> = PointF32::new([
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32, 9.0f32, 10.0f32,
        ]);
        assert!(point_f32_10d.is_valid().is_ok());
        assert_eq!(point_f32_10d.coordinates().len(), 10);
    }
}

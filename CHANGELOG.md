# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-07-25

### Changed

- **BREAKING**: Generalized coordinate types from `f64` to generic type `T`
  - All geometric structures (`Point`, `Vertex`, `Cell`, `Facet`, `TriangulationDataStructure`) now accept a generic floating-point type `T`
    instead of being hardcoded to `f64`
  - The generic type `T` must implement `CoordinateScalar`, which includes traits for floating-point operations, hashing, equality, validation, and serialization
  - This change enables support for `f32`, `f64`, and other floating-point types that satisfy the coordinate requirements
  - Examples now demonstrate usage with both `f32` and `f64` coordinate types
  - All tests continue to pass with the new generic implementation
- **Point Implementation Improvements**:
  - Replaced `Point::from` usage with `Point::new` in tests for consistency and directness
  - Enhanced Point implementation to leverage the `Coordinate` trait more effectively
  - Improved code style and idiomatic Rust patterns throughout Point implementation
  - Updated Point struct to use `Self::new(coords_u)` instead of `Point::new(coords_u)` for consistency

### Added

- `CoordinateScalar` trait alias that captures all required trait bounds for coordinate scalar types
- Support for multiple floating-point precision levels (`f32`, `f64`, etc.)
- Enhanced documentation explaining the generic coordinate system
- Validation and finite checking for generic coordinate types

### Fixed

- Resolved clippy warnings related to redundant closure usage in Point coordinate mapping
- Fixed clippy warnings by replacing redundant closures (`|x| x.into()`) with direct method references (`std::convert::Into::into`)
- Improved code clarity and maintainability in Point conversion implementations

### Technical Details

- The `Coordinate` trait provides a unified interface for coordinate operations across different storage mechanisms
- All coordinate-related trait bounds are now consolidated into the `CoordinateScalar` trait for cleaner type signatures
- Custom equality semantics for floating-point coordinates handle NaN values consistently across different precision levels
- Hash implementations ensure consistent behavior for coordinates used as keys in collections
- Enhanced Point coordinate type conversions with better error handling and validation
- Improved Point trait implementations for better integration with the Coordinate trait system

## [0.2.0] - 2024-12-19

### Initial Release

- Initial release with d-dimensional Delaunay triangulation support
- Serialization/Deserialization capabilities
- CGAL-inspired API design

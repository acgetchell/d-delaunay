# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-07-26

### Changed

- **Distance Calculation Consistency**: Standardized all distance calculations to use established utility functions
  - Updated `circumcenter` function to use `squared_norm` for squared distance calculations instead of manual coordinate squaring
  - Replaced manual distance calculations in tests with cleaner `hypot` function calls
  - Simplified complex manual distance computations in debug test functions
- **Code Quality Improvements**:
  - Eliminated redundant manual distance calculation code throughout the codebase
  - Improved code readability and maintainability by using consistent utility functions
  - Enhanced numerical stability by leveraging optimized `hypot` implementation

### Performance

- **Measurable Performance Gains**: Consistent use of optimized functions improved performance across all methods
  - `insphere_distance`: +2.8% improvement (1,505 ns → 1,463 ns)
  - `insphere_lifted`: +1.7% improvement in batch operations (661 μs → 650 μs)
  - `insphere`: +1.3% improvement in batch operations (822 μs → 811 μs)
  - Updated benchmark results in `benches/README.md` with version 0.3.0 vs 0.3.1 comparison
- **Cumulative Improvements**: Total performance gains from v0.2.0 to v0.3.1:
  - `insphere_distance`: +4.0% faster overall
  - `insphere_lifted`: +2.7% faster in batch operations

### Technical Details

- The `hypot` function provides numerically stable Euclidean distance calculations with better overflow/underflow handling
- The `squared_norm` function efficiently computes squared distances using generic arithmetic operations
- All distance calculations now benefit from the same optimized, well-tested implementations
- Test code clarity improved by eliminating manual coordinate arithmetic in favor of established utility functions

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

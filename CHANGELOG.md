# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2025-07-29

### Added

- **Complete Test Coverage for Coordinate Traits**: Achieved 100% test coverage for `src/geometry/traits/coordinate.rs`
  - Added comprehensive tests for hash collision edge cases and coordinate distribution
  - Added complete coverage of `CoordinateScalar` trait implementations and tolerance methods
  - Added extensive edge case testing for coordinate patterns, extreme dimensions, and boundary values
  - Added tests for mixed special floating-point values (NaN, infinity, negative zero)
  - Enhanced validation testing for different dimensions and error scenarios
  - Coverage improved from 99.69% (7/8 lines) to 100% (8/8 lines)
- **Enhanced Testing Framework**: Comprehensive test suite improvements across core data structures
  - Added robust testing for `Cell`, `Facet`, `Vertex`, and utility functions
  - Improved test coverage for serialization/deserialization edge cases
  - Enhanced error handling validation in geometric predicates
  - Added comprehensive validation tests for data structure integrity
- **New Macros `vertex!` and `cell!`**: Simplified creation of vertices and cells for ease of use
  - Introduced `vertex!` macro to facilitate vertex instantiation with concise syntax
  - Introduced `cell!` macro to streamline cell creation in tests and examples

### Changed

- **Coordinate Trait Simplification**: Streamlined coordinate trait implementation
  - Reduced `coordinate.rs` from 1,036 lines to 626 lines while maintaining full functionality
  - Consolidated redundant test cases and improved test organization
  - Enhanced code clarity and maintainability in coordinate operations
- **Facet Structure Enhancements**: Improved facet implementation and testing
  - Enhanced facet creation, validation, and serialization capabilities
  - Simplified facet example usage with improved macro support
  - Strengthened error handling and edge case coverage
- **Cell Structure Improvements**: Optimized cell data structure implementation
  - Streamlined cell creation and validation processes
  - Enhanced circumsphere and geometric predicate implementations
  - Improved memory efficiency and code organization
- **Testing Framework Refactoring**: Consolidated and enhanced testing infrastructure
  - Removed redundant test utilities and consolidated common testing patterns
  - Improved test readability and maintainability across all modules
  - Enhanced error message validation and edge case testing

### Fixed

- **Clippy Compliance**: Resolved all clippy pedantic warnings across the codebase
  - Fixed unreadable literal warnings by adding separators to long floating-point numbers
  - Replaced strict floating-point comparisons (`assert_eq!`) with relative comparisons (`assert_relative_eq!`)
  - Added appropriate `#[allow(clippy::cast_possible_truncation)]` annotations for intentional type conversions
  - Fixed floating-point array comparisons to use proper epsilon-based comparison methods
  - Enhanced code readability and maintainability following Rust best practices
- **Variable Naming**: Fixed clippy `similar_names` warnings by renaming variables for better clarity
  - Renamed `coord_10d` to `coord_10_d` and `origin_10d` to `origin_10_d` to avoid similar name confusion
- **Code Organization**: Eliminated redundant code and improved structure consistency
  - Removed duplicate test implementations and consolidated similar functionality
  - Improved error handling patterns across data structures
  - Enhanced documentation and code comments for better maintainability

### Performance

- **Reduced Code Complexity**: Significant reduction in lines of code while maintaining functionality
  - `coordinate.rs`: Reduced by 410 lines (40% reduction) through better organization
  - `cell.rs`: Streamlined by 354 lines through code consolidation
  - `facet.rs`: Optimized structure while adding 284 lines of comprehensive tests
  - Overall improvement in compilation times and code maintainability

### Technical Details

- All 26 coordinate trait tests pass with comprehensive coverage of:
  - Basic coordinate functionality across multiple dimensions and types
  - Special floating-point value handling (NaN, infinity, subnormal numbers)
  - Hash collision resistance and distribution testing
  - Ordered equality semantics for floating-point coordinates
  - Validation error handling and reporting
  - Tolerance method implementations for different precision levels
- Enhanced data structure validation ensures geometric consistency across all operations
- Improved floating-point comparison practices throughout the entire test suite
- Strengthened serialization/deserialization robustness with comprehensive error handling
- All geometric predicates maintain numerical stability with enhanced edge case coverage

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

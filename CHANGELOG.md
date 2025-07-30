# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Prelude module for simplified imports
  - Consolidates commonly used types and macros
  - Reduces boilerplate in examples and user code
  - Re-exports vertex! and cell! macros for convenience
- Enhanced triangulation data structure documentation
  - Comprehensive module-level documentation with examples
  - Detailed complexity analysis for algorithms
  - Usage examples for different dimensional triangulations

### Changed

- Streamlined supercell removal process
  - Simplified `remove_cells_containing_supercell_vertices` method signature
  - Removed redundant supercell parameter for cleaner API
  - Eliminated duplicate calls to `remove_duplicate_cells` for better performance
- Improved error handling in supercell creation
  - Replaced unwrap() calls with proper error propagation
  - Added informative error messages for triangulation failures
  - Enhanced robustness during triangulation initialization
- Triangulation data structure refactoring
  - Cleaner separation of concerns in Bowyer-Watson algorithm
  - Improved code organization and maintainability
  - Enhanced documentation with algorithmic complexity analysis

### Performance

- Optimized supercell removal logic
  - Reduced redundant operations in cleanup phase
  - Streamlined cell filtering and removal process
- Improved boundary facet detection algorithms
  - Reduced complexity from O(N²·F²) to O(N·F) using HashMap-based approach
  - Single-pass facet-to-cells mapping for efficient boundary identification
  - Significant performance gains for large triangulations (2000+ cells)

### Fixed

- Eliminated unnecessary cloning in supercell insertion
- Corrected method signatures for consistency
- Improved memory efficiency in cell removal operations

## [0.3.2] - 2025-07-29

### Added

- Complete test coverage for coordinate traits (100% coverage)
  - Comprehensive tests for hash collision edge cases and coordinate distribution
  - Full coverage of `CoordinateScalar` trait implementations and tolerance methods
  - Edge case testing for coordinate patterns, extreme dimensions, and boundary values
  - Tests for mixed special floating-point values (NaN, infinity, negative zero)
  - Validation testing for different dimensions and error scenarios
- Enhanced testing framework
  - Robust testing for `Cell`, `Facet`, `Vertex`, and utility functions
  - Coverage for serialization/deserialization edge cases
  - Error handling validation in geometric predicates
  - Validation tests for data structure integrity
- New macros `vertex!` and `cell!`
  - Simplified creation of vertices and cells
  - `vertex!` macro for vertex instantiation
  - `cell!` macro for cell creation in tests and examples

### Changed

- Coordinate trait simplification
  - Reduced lines in `coordinate.rs` while maintaining functionality
  - Consolidated redundant test cases and improved test organization
  - Enhanced code clarity and maintainability
- Facet structure enhancements
  - Improved facet creation, validation, and serialization
  - Simplified example usage with macro support
  - Strengthened error handling and edge case coverage
- Cell structure improvements
  - Optimized cell creation and validation processes
  - Enhanced circumsphere and geometric predicates
  - Improved memory efficiency and code organization
- Testing framework refactoring
  - Removed redundant test utilities and consolidated testing patterns
  - Improved test readability and maintainability
  - Enhanced error message validation

### Fixed

- Clippy compliance
  - Resolved all clippy pedantic warnings
  - Fixed unreadable literal warnings with separators
  - Replaced float comparisons with relative comparisons
  - Annotated intentional type conversions
  - Fixed floating-point array comparisons
  - Improved code readability and maintainability
- Variable naming
  - Renamed variables to avoid similar name confusion
- Code organization
  - Eliminated redundant code
  - Improved error handling patterns
  - Enhanced documentation and code comments

### Performance

- Reduced code complexity
  - Significant reduction in lines while maintaining functionality
  - Overall improvement in compilation times and maintainability

### Technical Details

- All 26 coordinate trait tests pass
  - Comprehensive coverage of functionality
  - Handling of special floating-point values
  - Hash collision resistance and distribution testing
  - Ordered equality semantics for floating-point coordinates
  - Validation error handling
  - Tolerance method implementations for precision levels
- Enhanced data structure validation ensures geometric consistency
- Improved floating-point comparison practices
- Strengthened serialization/deserialization robustness
- All geometric predicates maintain numerical stability

## [0.3.1] - 2025-07-26

### Changed

- Distance calculation consistency
  - Standardized all distance calculations by using utility functions
- Code quality improvements

### Performance

- Measurable performance gains
  - Updated benchmarks to reflect improvements
- Cumulative improvements

### Technical Details

- The `hypot` function provides stable Euclidean distance calculations
- Improved test code clarity

## [0.3.0] - 2025-07-25

### Changed

- Generalized coordinate types
  - Coordinate structures now accept a generic floating-point type instead of being hardcoded
- Point implementation improvements

### Added

- `CoordinateScalar` trait alias
- Support for multiple floating-point precision levels

### Fixed

- Resolved clippy warnings
- Improved code clarity in Point conversion implementations

### Technical Details

- The `Coordinate` trait provides a unified interface

## [0.2.0] - 2024-12-19

### Initial Release

- Initial release with Delaunay triangulation support
- Serialization/Deserialization capabilities
- CGAL-inspired API design

[Unreleased]: https://github.com/acgetchell/d-delaunay/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/acgetchell/d-delaunay/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/acgetchell/d-delaunay/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/acgetchell/d-delaunay/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/acgetchell/d-delaunay/releases/tag/v0.2.0

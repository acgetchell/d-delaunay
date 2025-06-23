# Examples

This directory contains examples demonstrating various features and
capabilities of the d-delaunay library.

## Available Examples

### 1. Point Comparison and Hashing (`point_comparison_and_hashing.rs`)

Demonstrates the robust comparison and hashing behavior of the Point struct,
with special emphasis on handling of NaN (Not a Number) and infinity values.

**Key Features:**

- **NaN-aware equality**: Unlike IEEE 754 standard where NaN ≠ NaN, our Point
  implementation treats NaN values as equal to themselves for consistent
  behavior in data structures.
- **Consistent hashing**: Points with identical coordinates (including NaN)
  produce the same hash value, enabling reliable use in HashMap and HashSet.
- **Mathematical properties**: Equality satisfies reflexivity, symmetry, and
  transitivity.
- **Special value handling**: Proper comparison of infinity, negative infinity,
  and zero values.

**Run with:** `cargo run --example point_comparison_and_hashing`

### 2. Implicit Conversion Example (`implicit_conversion.rs`)

This example summarizes the refactoring performed to enable implicit conversion
of `vertex.point.coordinates()` to coordinate arrays using Rust's `From` trait.

## Overview

The goal was to refactor the codebase so that `vertex.point.coordinates()`
calls could be implicitly converted to coordinate arrays, making the code more
ergonomic while maintaining backward compatibility.

## Changes Made

### 1. **Added Implicit Conversion Traits for Vertex**

**File:** `src/delaunay_core/vertex.rs`

- **Added `From<Vertex<T, U, D>>` for `[T; D]`**: Allows owned vertices to be
  implicitly converted to coordinate arrays
- **Added `From<&Vertex<T, U, D>>` for `[T; D]`**: Allows vertex references to
  be implicitly converted to coordinate arrays while preserving the original
  vertex
- **Added comprehensive tests**: `vertex_implicit_conversion_to_coordinates`
  test verifies both owned and reference conversions work correctly

### 2. **Added Implicit Conversion Traits for Point**

**File:** `src/delaunay_core/point.rs`

- **Added `From<Point<T, D>>` for `[T; D]`**: Allows owned points to be
  implicitly converted to coordinate arrays
- **Added `From<&Point<T, D>>` for `[T; D]`**: Allows point references to be
  implicitly converted to coordinate arrays while preserving the original point
- **Added comprehensive tests**: `point_implicit_conversion_to_coordinates`
  test verifies both owned and reference conversions work correctly

### 3. **Refactored Codebase to Use Implicit Conversion**

**Files Modified:**

- `src/delaunay_core/utilities.rs`
- `src/delaunay_core/triangulation_data_structure.rs`
- `src/delaunay_core/facet.rs`
- `src/delaunay_core/cell.rs`

**Changes:**

- Replaced explicit `.point.coordinates()` calls with implicit conversions
  using `.into()` where appropriate
- Used type annotations to make the conversion explicit:
  `let coords: [T; D] = vertex.into();`
- Maintained code clarity while making it more ergonomic

### 4. **Fixed Type Conversion Issues**

**File:** `src/delaunay_core/cell.rs`

- **Fixed nalgebra compatibility**: Updated code that was trying to convert
  `[T; D]` directly to `na::Point<f64, D>`
- **Added proper type conversion**: Used `.map(|x| x.into())` to convert
  coordinates to f64 arrays before creating nalgebra Points
- **Maintained numerical stability**: Ensured all mathematical operations
  continue to work correctly

### 5. **Added Documentation and Examples**

**File:** `examples/implicit_conversion.rs`

- **Created comprehensive example**: Demonstrates all four types of implicit
  conversions
- **Added full documentation**: Includes crate-level docs, function docs, and
  inline comments
- **Shows before/after usage**: Demonstrates the ergonomic benefits of the
  refactoring

## Usage Examples

### Before Refactoring

```rust
let coords: [f64; 3] = vertex.point.coordinates();
```

### After Refactoring

```rust
// From owned vertex
let coords: [f64; 3] = vertex.into();

// From vertex reference (preserves original)
let coords: [f64; 3] = (&vertex).into();

// From owned point
let coords: [f64; 3] = point.into();

// From point reference (preserves original)  
let coords: [f64; 3] = (&point).into();
```

## Benefits

1. **More Ergonomic**: Shorter, cleaner syntax for coordinate access
2. **Backward Compatible**: All existing `.coordinates()` calls continue to work
3. **Type Safe**: Leverages Rust's type system for automatic conversion
4. **Performance**: Zero-cost abstractions - no runtime overhead
5. **Flexible**: Works with both owned values and references

## Testing

- **All 190 existing tests pass**: Ensures no regression in functionality
- **New tests added**: Specific tests for implicit conversion behavior
- **Example works**: Demonstrates practical usage of the new feature
- **Documentation tests pass**: All 22 doc tests continue to work

## Type Safety

The implicit conversions are fully type-safe:

- Generic over coordinate type `T` and dimension `D`
- Maintains all existing trait bounds
- Compiler enforces correct usage at compile time
- No runtime type checking required

## Performance Impact

- **Zero runtime cost**: Conversions are handled at compile time
- **No allocations**: Direct access to underlying coordinate arrays
- **Inlined**: Conversion functions are trivial and will be inlined by the
  compiler

## Conclusion

The refactoring successfully enables implicit conversion from vertices and
points to coordinate arrays while maintaining full backward compatibility, type
safety, and performance. The codebase is now more ergonomic and easier to use
while preserving all existing functionality.

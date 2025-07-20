//! `OrderedEq` trait for floating-point equality comparison.
//!
//! This module provides the `OrderedEq` trait that enables proper equality comparison
//! for floating-point types, treating NaN values as equal to themselves. This is
//! essential for implementing the `Eq` trait for types containing floating-point
//! coordinates, which allows them to be used as keys in hash-based collections.

use ordered_float::OrderedFloat;

/// Helper trait for OrderedFloat-based equality comparison that handles NaN properly
pub trait OrderedEq {
    /// Compares two values for equality using ordered comparison semantics.
    ///
    /// This method provides a way to compare floating-point numbers that treats
    /// NaN values as equal to themselves, which is different from the default
    /// floating-point equality comparison where NaN != NaN.
    ///
    /// # Arguments
    ///
    /// * `other` - The other value to compare with
    ///
    /// # Returns
    ///
    /// Returns `true` if the values are equal according to ordered comparison,
    /// `false` otherwise.
    fn ordered_eq(&self, other: &Self) -> bool;
}

// Unified macro for implementing OrderedEq
macro_rules! impl_ordered_eq {
    (float: $($t:ty),*) => {
        $(
            impl OrderedEq for $t {
                #[inline(always)]
                fn ordered_eq(&self, other: &Self) -> bool {
                    OrderedFloat(*self) == OrderedFloat(*other)
                }
            }
        )*
    };
}

impl_ordered_eq!(float: f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_eq_trait_coverage() {
        // Test OrderedEq trait implementations

        // Test floating point types with normal values
        assert!(1.0f32.ordered_eq(&1.0f32));
        assert!(1.0f64.ordered_eq(&1.0f64));
        assert!(!1.0f32.ordered_eq(&2.0f32));
        assert!(!1.0f64.ordered_eq(&2.0f64));

        // Test NaN equality (should be true with OrderedEq)
        assert!(f32::NAN.ordered_eq(&f32::NAN));
        assert!(f64::NAN.ordered_eq(&f64::NAN));

        // Test infinity values
        assert!(f32::INFINITY.ordered_eq(&f32::INFINITY));
        assert!(f64::INFINITY.ordered_eq(&f64::INFINITY));
        assert!(f32::NEG_INFINITY.ordered_eq(&f32::NEG_INFINITY));
        assert!(f64::NEG_INFINITY.ordered_eq(&f64::NEG_INFINITY));
        assert!(!f32::INFINITY.ordered_eq(&f32::NEG_INFINITY));
        assert!(!f64::INFINITY.ordered_eq(&f64::NEG_INFINITY));

        // Test zero comparisons
        assert!(0.0f32.ordered_eq(&(-0.0f32)));
        assert!(0.0f64.ordered_eq(&(-0.0f64)));
    }
}

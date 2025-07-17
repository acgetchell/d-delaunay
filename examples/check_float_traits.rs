//! This example demonstrates which traits are included in the `Float` trait from `num_traits`.
//!
//! The `Float` trait includes:
//! - `PartialEq`
//! - `PartialOrd`
//! - `Copy`
//! - `Clone`
//!
//! But it does NOT include:
//! - `Default`
//!
//! This is why in our Point implementation, we need to explicitly include `Default`
//! in our trait bounds even though we have `Float`.

use num_traits::Float;

fn check_float_traits<T>()
where
    T: Float,
{
    // Let's check if Float already includes the traits we're specifying
    fn requires_partial_eq<U: PartialEq>() {}
    fn requires_partial_ord<U: PartialOrd>() {}
    fn requires_copy<U: Copy>() {}
    fn requires_clone<U: Clone>() {}

    requires_partial_eq::<T>();
    requires_partial_ord::<T>();
    requires_copy::<T>();
    requires_clone::<T>();
    // Note: Default is NOT included in Float, so we don't test it here
}

fn main() {
    check_float_traits::<f32>();
    check_float_traits::<f64>();
    println!("Float trait includes PartialEq, PartialOrd, Copy, and Clone");
    println!(
        "But Float does NOT include Default - that's why we need it explicitly in Point's trait bounds"
    );
}

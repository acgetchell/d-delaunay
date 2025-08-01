//! Test file to figure out allocation-counter API

/// Test the allocation counter API
fn main() {
    println!("Testing allocation counter APIs...");

    #[cfg(feature = "count-allocations")]
    {
        let result = allocation_counter::measure(|| {
            let _v: Vec<i32> = (0..100).collect();
            println!("Created vector with {} elements", 100);
        });
        println!("✓ allocation_counter::measure() works: {result:?}");
    }

    #[cfg(not(feature = "count-allocations"))]
    {
        println!("count-allocations feature not enabled");
        // Still do the work without measuring
        let v: Vec<i32> = (0..100).collect();
        println!("Created vector with {} elements", v.len());
    }
}

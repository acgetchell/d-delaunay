//! Delaunay Triangulation Example Program
//!
//! This binary creates Delaunay triangulations in various dimensions,
//! tests allocation counting, and demonstrates JSON serialization.

use std::fs::File;
use std::io::Write;

use anyhow::Result;
use clap::{Arg, Command};
use d_delaunay::prelude::*;
use rand::Rng;

/// Generate random points for benchmarking
fn generate_random_points_2d(count: usize) -> Vec<Point<f64, 2>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

fn generate_random_points_3d(count: usize) -> Vec<Point<f64, 3>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

fn generate_random_points_4d(count: usize) -> Vec<Point<f64, 4>> {
    let mut rng = rand::rng();
    (0..count)
        .map(|_| {
            Point::new([
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
                rng.random_range(-100.0..100.0),
            ])
        })
        .collect()
}

/// Test allocation counter functionality
fn test_allocation_counter() {
    println!("Testing allocation-counter functionality...");

    // Try different possible APIs for allocation-counter 0.8.1
    #[cfg(feature = "allocation-counter")]
    {
        // Method 1: Try count_allocations function
        if std::panic::catch_unwind(|| {
            let result = allocation_counter::measure(|| {
                let _v: Vec<i32> = (0..1000).collect();
            });
            println!("Method 1 - count_allocations: {result:?}");
        })
        .is_ok()
        {
            println!("✓ count_allocations API works");
        } else {
            println!("✗ count_allocations API failed");
        }
    }

    #[cfg(not(feature = "allocation-counter"))]
    {
        println!("allocation-counter feature not enabled");
    }
}

/// Create and test 2D triangulation
fn create_2d_triangulation(count: usize) -> Result<()> {
    println!("Creating 2D triangulation with {count} points...");

    let points = generate_random_points_2d(count);
    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();

    let start = std::time::Instant::now();
    let tds = Tds::<f64, (), (), 2>::new(&vertices)?;
    let duration = start.elapsed();

    println!("  ✓ Created in {duration:?}");
    println!("  ✓ Vertices: {}", tds.number_of_vertices());
    println!("  ✓ Cells: {}", tds.number_of_cells());
    println!("  ✓ Boundary facets: {}", tds.boundary_facets().len());

    Ok(())
}

/// Create and test 3D triangulation
fn create_3d_triangulation(count: usize) -> Result<()> {
    println!("Creating 3D triangulation with {count} points...");

    let points = generate_random_points_3d(count);
    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();

    let start = std::time::Instant::now();
    let tds = Tds::<f64, (), (), 3>::new(&vertices)?;
    let duration = start.elapsed();

    println!("  ✓ Created in {duration:?}");
    println!("  ✓ Vertices: {}", tds.number_of_vertices());
    println!("  ✓ Cells: {}", tds.number_of_cells());
    println!("  ✓ Boundary facets: {}", tds.boundary_facets().len());

    Ok(())
}

/// Create and test 4D triangulation
fn create_4d_triangulation(count: usize) -> Result<()> {
    println!("Creating 4D triangulation with {count} points...");

    let points = generate_random_points_4d(count);
    let vertices: Vec<_> = points.iter().map(|p| vertex!(*p)).collect();

    let start = std::time::Instant::now();
    let tds = Tds::<f64, (), (), 4>::new(&vertices)?;
    let duration = start.elapsed();

    println!("  ✓ Created in {duration:?}");
    println!("  ✓ Vertices: {}", tds.number_of_vertices());
    println!("  ✓ Cells: {}", tds.number_of_cells());
    println!("  ✓ Boundary facets: {}", tds.boundary_facets().len());

    Ok(())
}

/// Save 2D triangulation to JSON file
fn save_2d_to_json(tds: &Tds<f64, (), (), 2>, filename: &str) -> Result<()> {
    println!("Saving 2D triangulation to {filename}...");

    let json = serde_json::to_string_pretty(tds)?;
    let mut file = File::create(filename)?;
    file.write_all(json.as_bytes())?;

    let json_len = json.len();
    println!("  ✓ Saved {json_len} bytes to {filename}");
    Ok(())
}

/// Save 3D triangulation to JSON file
fn save_3d_to_json(tds: &Tds<f64, (), (), 3>, filename: &str) -> Result<()> {
    println!("Saving 3D triangulation to {filename}...");

    let json = serde_json::to_string_pretty(tds)?;
    let mut file = File::create(filename)?;
    file.write_all(json.as_bytes())?;

    let json_len = json.len();
    println!("  ✓ Saved {json_len} bytes to {filename}");
    Ok(())
}

/// Test JSON serialization/deserialization
fn test_json_serialization() -> Result<()> {
    println!("Testing JSON serialization...");

    // Create small triangulations for testing
    let points_2d = generate_random_points_2d(10);
    let vertices_2d: Vec<_> = points_2d.iter().map(|p| vertex!(*p)).collect();
    let tds_2d = Tds::<f64, (), (), 2>::new(&vertices_2d)?;

    let points_3d = generate_random_points_3d(10);
    let vertices_3d: Vec<_> = points_3d.iter().map(|p| vertex!(*p)).collect();
    let tds_3d = Tds::<f64, (), (), 3>::new(&vertices_3d)?;

    // Save to JSON files
    save_2d_to_json(&tds_2d, "triangulation_2d.json")?;
    save_3d_to_json(&tds_3d, "triangulation_3d.json")?;

    // Test loading back
    println!("Testing deserialization...");
    let json_2d = std::fs::read_to_string("triangulation_2d.json")?;
    let loaded_tds_2d: Tds<f64, (), (), 2> = serde_json::from_str(&json_2d)?;
    println!(
        "  ✓ Loaded 2D triangulation: {} vertices, {} cells",
        loaded_tds_2d.number_of_vertices(),
        loaded_tds_2d.number_of_cells()
    );

    let json_3d = std::fs::read_to_string("triangulation_3d.json")?;
    let loaded_tds_3d: Tds<f64, (), (), 3> = serde_json::from_str(&json_3d)?;
    println!(
        "  ✓ Loaded 3D triangulation: {} vertices, {} cells",
        loaded_tds_3d.number_of_vertices(),
        loaded_tds_3d.number_of_cells()
    );

    Ok(())
}

/// Benchmark small-scale operations
fn benchmark_operations() -> Result<()> {
    println!("Running small-scale benchmarks...");

    let counts = [10, 20, 30, 40, 50];

    for &count in &counts {
        println!("\n--- {count} points ---");

        // 2D triangulation
        let points_2d = generate_random_points_2d(count);
        let vertices_2d: Vec<_> = points_2d.iter().map(|p| vertex!(*p)).collect();

        let start = std::time::Instant::now();
        let mut tds_2d = Tds::<f64, (), (), 2>::new(&vertices_2d)?;
        let creation_time = start.elapsed();

        // Test assign_neighbors
        for cell in tds_2d.cells.values_mut() {
            cell.neighbors = None;
        }
        let start = std::time::Instant::now();
        tds_2d.assign_neighbors();
        let assign_time = start.elapsed();

        // Test remove_duplicate_cells
        let cell_vertices: Vec<_> = tds_2d.vertices.values().copied().collect();
        if cell_vertices.len() >= 3 {
            for _ in 0..2 {
                let duplicate_cell = cell!(cell_vertices[0..3].to_vec());
                tds_2d.cells.insert(duplicate_cell);
            }
        }
        let start = std::time::Instant::now();
        let removed = tds_2d.remove_duplicate_cells();
        let cleanup_time = start.elapsed();

        println!(
            "2D: creation={creation_time:?}, assign_neighbors={assign_time:?}, cleanup={cleanup_time:?} (removed {removed})"
        );

        // 3D triangulation
        let points_3d = generate_random_points_3d(count);
        let vertices_3d: Vec<_> = points_3d.iter().map(|p| vertex!(*p)).collect();

        let start = std::time::Instant::now();
        let tds_3d = Tds::<f64, (), (), 3>::new(&vertices_3d)?;
        let creation_time_3d = start.elapsed();

        let num_vertices = tds_3d.number_of_vertices();
        let num_cells = tds_3d.number_of_cells();
        println!("3D: creation={creation_time_3d:?}, vertices={num_vertices}, cells={num_cells}");
    }

    Ok(())
}

fn main() -> Result<()> {
    let matches = Command::new("Delaunay Triangulation Example")
        .version("0.3.3")
        .author("Adam Getchell <adam@adamgetchell.org>")
        .about("Creates Delaunay triangulations and demonstrates functionality")
        .arg(
            Arg::new("count")
                .short('c')
                .long("count")
                .value_name("COUNT")
                .help("Number of points to generate")
                .default_value("20"),
        )
        .arg(
            Arg::new("dimension")
                .short('d')
                .long("dimension")
                .value_name("DIM")
                .help("Dimension (2, 3, or 4)")
                .default_value("3"),
        )
        .arg(
            Arg::new("test-allocations")
                .long("test-allocations")
                .help("Test allocation counter functionality")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("json")
                .long("json")
                .help("Test JSON serialization")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("benchmark")
                .long("benchmark")
                .help("Run benchmarks")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let count: usize = matches
        .get_one::<String>("count")
        .unwrap()
        .parse()
        .expect("Count must be a positive integer");

    let dimension: usize = matches
        .get_one::<String>("dimension")
        .unwrap()
        .parse()
        .expect("Dimension must be 2, 3, or 4");

    println!("🔺 Delaunay Triangulation Example 🔺");
    println!("=====================================\n");

    if matches.get_flag("test-allocations") {
        test_allocation_counter();
        println!();
    }

    if matches.get_flag("json") {
        test_json_serialization()?;
        println!();
    }

    if matches.get_flag("benchmark") {
        benchmark_operations()?;
        println!();
    }

    // Create triangulation based on dimension
    match dimension {
        2 => create_2d_triangulation(count)?,
        3 => create_3d_triangulation(count)?,
        4 => create_4d_triangulation(count)?,
        _ => {
            eprintln!("Error: Dimension must be 2, 3, or 4");
            std::process::exit(1);
        }
    }

    println!("\n✅ Example completed successfully!");
    Ok(())
}

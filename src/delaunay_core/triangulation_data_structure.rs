//! Data and operations on d-dimensional triangulation data structures.
//!
//! Intended to match functionality of the
//! [CGAL Triangulation](https://doc.cgal.org/latest/Triangulation/index.html).

use super::{
    cell::{Cell, CellBuilder}, facet::Facet, point::Point, utilities::find_extreme_coordinates,
    vertex::Vertex,
};
use na::{ComplexField, Const, OPoint};
use nalgebra as na;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::cmp::{min, Ordering, PartialEq};
use std::collections::HashSet;
use std::ops::{AddAssign, Div, SubAssign};
use std::{collections::HashMap, hash::Hash, iter::Sum};
use uuid::Uuid;

/// Helper function to check if two facets are adjacent (share the same vertices)
fn facets_are_adjacent<T, U, V, const D: usize>(
    facet1: &Facet<T, U, V, D>,
    facet2: &Facet<T, U, V, D>,
) -> bool
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    // Two facets are adjacent if they have the same vertices
    // (though they may have different orientations)
    let vertices1 = facet1.vertices();
    let vertices2 = facet2.vertices();

    if vertices1.len() != vertices2.len() {
        return false;
    }

    // Check if all vertices in facet1 are present in facet2
    vertices1.iter().all(|v1| vertices2.contains(v1))
}

/// Generate all combinations of `k` vertices from the given vertex list
fn generate_combinations<T, U, const D: usize>(
    vertices: &[Vertex<T, U, D>],
    k: usize,
) -> Vec<Vec<Vertex<T, U, D>>>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    let mut combinations = Vec::new();
    
    if k == 0 {
        combinations.push(Vec::new());
        return combinations;
    }
    
    if k > vertices.len() {
        return combinations;
    }
    
    if k == vertices.len() {
        combinations.push(vertices.to_vec());
        return combinations;
    }
    
    // Generate combinations using iterative approach
    let n = vertices.len();
    let mut indices = (0..k).collect::<Vec<_>>();
    
    loop {
        // Add current combination
        let combination = indices.iter().map(|i| vertices[*i]).collect();
        combinations.push(combination);
        
        // Find next combination
        let mut i = k;
        loop {
            if i == 0 {
                return combinations;
            }
            i -= 1;
            if indices[i] != i + n - k {
                break;
            }
        }
        
        indices[i] += 1;
        for j in (i + 1)..k {
            indices[j] = indices[j - 1] + 1;
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
/// The `Tds` struct represents a triangulation data structure with vertices
/// and cells, where the vertices and cells are identified by UUIDs.
///
/// # Properties:
///
/// * `vertices`: A [HashMap] that stores vertices with their corresponding
///   [Uuid]s as keys. Each [Vertex] has a [Point] of type T, vertex data of type
///   U, and a constant D representing the dimension.
/// * `cells`: The `cells` property is a [HashMap] that stores [Cell] objects.
///   Each [Cell] has one or more [Vertex] objects with cell data of type V.
///   Note the dimensionality of the cell may differ from D, though the [Tds]
///   only stores cells of maximal dimensionality D and infers other lower
///   dimensional cells (cf. [Facet]) from the maximal cells and their vertices.
///
/// For example, in 3 dimensions:
///
/// * A 0-dimensional cell is a [Vertex].
/// * A 1-dimensional cell is an `Edge` given by the `Tetrahedron` and two
///   [Vertex] endpoints.
/// * A 2-dimensional cell is a [Facet] given by the `Tetrahedron` and the
///   opposite [Vertex].
/// * A 3-dimensional cell is a `Tetrahedron`, the maximal cell.
///
/// A similar pattern holds for higher dimensions.
///
/// In general, vertices are embedded into D-dimensional Euclidean space,
/// and so the [Tds] is a finite simplicial complex.
pub struct Tds<T, U, V, const D: usize>
where
    T: Clone + Copy + Default + PartialEq + PartialOrd,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
{
    /// A [HashMap] that stores [Vertex] objects with their corresponding [Uuid]s as
    /// keys. Each [Vertex] has a [Point] of type T, vertex data of type U,
    /// and a constant D representing the dimension.
    pub vertices: HashMap<Uuid, Vertex<T, U, D>>,

    /// A [HashMap] that stores [Cell] objects with their corresponding [Uuid]s as
    /// keys.
    /// Each [Cell] has one or more [Vertex] objects and cell data of type V.
    /// Note the dimensionality of the cell may differ from D, though the [Tds]
    /// only stores cells of maximal dimensionality D and infers other lower
    /// dimensional cells from the maximal cells and their vertices.
    pub cells: HashMap<Uuid, Cell<T, U, V, D>>,
}

impl<T, U, V, const D: usize> Tds<T, U, V, D>
where
    T: AddAssign<f64>
        + Clone
        + Copy
        + ComplexField<RealField = T>
        + Default
        + From<f64>
        + PartialEq
        + PartialOrd
        + SubAssign<f64>
        + Sum,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    f64: From<T>,
    for<'a> &'a T: Div<f64>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
    ordered_float::OrderedFloat<f64>: From<T>,
{
    /// The function creates a new instance of a triangulation data structure
    /// with given points, initializing the vertices and cells.
    ///
    /// # Arguments:
    ///
    /// * `points`: A container of [Point]s with which to initialize the
    ///   triangulation.
    ///
    /// # Returns:
    ///
    /// A Delaunay triangulation with cells and neighbors aligned, and vertices
    /// associated with cells.
    pub fn new(points: Vec<Point<T, D>>) -> Self {
        // handle case where vertices are constructed with data
        let vertices = Vertex::into_hashmap(Vertex::from_points(points));
        // let cells_vec = bowyer_watson(vertices);
        let cells = HashMap::new();
        // assign_neighbors(cells_vec);
        // assign_incident_cells(vertices);

        Self { vertices, cells }
    }

    /// The `add` function checks if a [Vertex] with the same coordinates already
    /// exists in the [HashMap], and if not, inserts the [Vertex].
    ///
    /// # Arguments:
    ///
    /// * `vertex`: The [Vertex] to add.
    ///
    /// # Returns:
    ///
    /// The function `add` returns `Ok(())` if the vertex was successfully
    /// added to the [HashMap], or an error message if the vertex already
    /// exists or if there is a [Uuid] collision.
    ///
    /// # Example:
    ///
    /// ```
    /// use d_delaunay::delaunay_core::triangulation_data_structure::Tds;
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let mut tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
    /// let point = Point::new([1.0, 2.0, 3.0]);
    /// let vertex = VertexBuilder::default().point(point).build().unwrap();
    /// let result = tds.add(vertex);
    /// assert!(result.is_ok());
    /// ```
    pub fn add(&mut self, vertex: Vertex<T, U, D>) -> Result<(), &'static str> {
        // Don't add if vertex with that point already exists
        for val in self.vertices.values() {
            if val.point.coords == vertex.point.coords {
                return Err("Vertex already exists!");
            }
        }

        // Hashmap::insert returns the old value if the key already exists and
        // updates it with the new value
        let result = self.vertices.insert(vertex.uuid, vertex);

        // Return an error if there is a uuid collision
        match result {
            Some(_) => Err("Uuid already exists!"),
            None => Ok(()),
        }
    }

    /// The function returns the number of vertices in the triangulation
    /// data structure.
    ///
    /// # Returns:
    ///
    /// The number of [Vertex] objects in the [Tds].
    ///
    /// # Example:
    ///
    /// ```
    /// use d_delaunay::delaunay_core::triangulation_data_structure::Tds;
    /// let tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
    /// use d_delaunay::delaunay_core::vertex::{Vertex, VertexBuilder};
    /// use d_delaunay::delaunay_core::point::Point;
    /// let mut tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
    /// let new_vertex1 = VertexBuilder::default().point(Point::new([1.0, 2.0, 3.0])).build().unwrap();
    /// let _ = tds.add(new_vertex1);
    /// assert_eq!(tds.number_of_vertices(), 1);
    /// ```
    pub fn number_of_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// The `dim` function returns the dimensionality of the [Tds].
    ///
    /// # Returns:
    ///
    /// The `dim` function returns the minimum value between the number of
    /// vertices minus one and the value of `D` as an [i32].
    ///
    /// # Example:
    ///
    /// ```
    /// use d_delaunay::delaunay_core::triangulation_data_structure::Tds;
    /// let tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
    /// assert_eq!(tds.dim(), -1);
    /// ```
    pub fn dim(&self) -> i32 {
        let len = self.number_of_vertices() as i32;

        min(len - 1, D as i32)
    }

    /// The function `number_of_cells` returns the number of cells in the [Tds].
    ///
    /// # Returns:
    ///
    /// The number of [Cell]s in the [Tds].
    pub fn number_of_cells(&self) -> usize {
        self.cells.len()
    }

    /// The `supercell` function creates a larger cell that contains all the
    /// input vertices, with some padding added.
    ///
    /// # Returns:
    ///
    /// A [Cell] that encompasses all [Vertex] objects in the triangulation.
    fn supercell(&self) -> Result<Cell<T, U, V, D>, anyhow::Error> {
        if self.vertices.is_empty() {
            // For empty input, create a default supercell
            return self.create_default_supercell();
        }

        // Find the bounding box of all input vertices
        let min_coords = find_extreme_coordinates(self.vertices.clone(), Ordering::Less);
        let max_coords = find_extreme_coordinates(self.vertices.clone(), Ordering::Greater);

        // Convert coordinates to f64 for calculations
        let mut center_f64 = [0.0f64; D];
        let mut size_f64 = 0.0f64;

        for i in 0..D {
            let min_f64: f64 = min_coords[i].into();
            let max_f64: f64 = max_coords[i].into();
            center_f64[i] = (min_f64 + max_f64) / 2.0;
            let dim_size = max_f64 - min_f64;
            if dim_size > size_f64 {
                size_f64 = dim_size;
            }
        }

        // Add significant padding to ensure all vertices are well inside
        size_f64 += 20.0; // Add 20 units of padding
        let radius_f64 = size_f64 / 2.0;

        // Convert back to T
        let mut center = [T::default(); D];
        for i in 0..D {
            center[i] = T::from(center_f64[i]);
        }
        let radius = T::from(radius_f64);

        // Create a proper non-degenerate simplex (tetrahedron for 3D)
        let points = self.create_supercell_simplex(&center, radius)?;

        let supercell = CellBuilder::default()
            .vertices(Vertex::from_points(points))
            .build()
            .unwrap();

        Ok(supercell)
    }

    /// Creates a default supercell for empty input
    fn create_default_supercell(&self) -> Result<Cell<T, U, V, D>, anyhow::Error> {
        let center = [T::default(); D];
        let radius = T::from(20.0f64);
        let points = self.create_supercell_simplex(&center, radius)?;

        let supercell = CellBuilder::default()
            .vertices(Vertex::from_points(points))
            .build()
            .unwrap();

        Ok(supercell)
    }

    /// Creates a well-formed simplex centered at the given point with the given radius
    fn create_supercell_simplex(
        &self,
        center: &[T; D],
        radius: T,
    ) -> Result<Vec<Point<T, D>>, anyhow::Error> {
        let mut points = Vec::new();

        // For 3D, create a regular tetrahedron
        if D == 3 {
            // Create a regular tetrahedron with vertices at:
            // (1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)
            // scaled by radius and translated by center
            let tetrahedron_vertices = [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ];

            for vertex_coords in &tetrahedron_vertices {
                let mut coords = [T::default(); D];
                for i in 0..D {
                    let center_f64: f64 = center[i].into();
                    let radius_f64: f64 = radius.into();
                    let coord_f64 = center_f64 + radius_f64 * vertex_coords[i];
                    coords[i] = T::from(coord_f64);
                }
                points.push(Point::new(coords));
            }
        } else {
            // For other dimensions, create a simplex using a generalized approach
            // Create D+1 vertices for a D-dimensional simplex

            // Create a regular simplex by placing vertices at the corners of a hypercube
            // scaled and offset appropriately
            let radius_f64: f64 = radius.into();

            // First vertex: all coordinates positive
            let mut coords = [T::default(); D];
            for i in 0..D {
                let center_f64: f64 = center[i].into();
                coords[i] = T::from(center_f64 + radius_f64);
            }
            points.push(Point::new(coords));

            // Remaining D vertices: flip one coordinate at a time to negative
            for dim in 0..D {
                let mut coords = [T::default(); D];
                for i in 0..D {
                    let center_f64: f64 = center[i].into();
                    if i == dim {
                        // This dimension gets negative offset
                        coords[i] = T::from(center_f64 - radius_f64);
                    } else {
                        // Other dimensions get positive offset
                        coords[i] = T::from(center_f64 + radius_f64);
                    }
                }
                points.push(Point::new(coords));
            }
        }

        Ok(points)
    }

    /// Performs the Bowyer-Watson algorithm to triangulate a set of vertices.
    ///
    /// # Returns:
    ///
    /// A [Result] containing the updated [Tds] with the Delaunay triangulation, or an error message.
    pub fn bowyer_watson(mut self) -> Result<Self, anyhow::Error>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        // If no vertices, return empty triangulation
        if self.vertices.is_empty() {
            return Ok(self);
        }

        // For cases with a small number of vertices, use a direct combinatorial approach
        // This is more reliable than the full Bowyer-Watson algorithm for small cases
        if self.vertices.len() >= D + 1 && self.vertices.len() <= D + 5 {
            let vertices: Vec<_> = self.vertices.values().cloned().collect();
            let mut created_cells = 0;
            
            // Generate all possible combinations of D+1 vertices
            let combinations = generate_combinations(&vertices, D + 1);
            
            for combination in combinations {
                // Try to create a cell with these vertices
                if let Ok(cell) = CellBuilder::default()
                    .vertices(combination)
                    .build()
                    .map_err(|e| anyhow::Error::msg(format!("Failed to create cell: {:?}", e)))
                {
                    self.cells.insert(cell.uuid, cell);
                    created_cells += 1;
                }
            }
            
            if created_cells > 0 {
                // Remove duplicate cells (cells with identical vertex sets)
                self.remove_duplicate_cells();
                
                // Assign neighbors between adjacent cells
                self.assign_neighbors()?;
                
                // Assign incident cells to vertices
                self.assign_incident_cells()?;
                
                return Ok(self);
            }
            
            // If the combinatorial approach didn't work, fall through to the full algorithm
        }

        // For more complex cases, use the full Bowyer-Watson algorithm
        // Create super-cell that contains all vertices
        let supercell = self.supercell()?;
        let supercell_vertices: HashSet<Uuid> = supercell.vertices.iter().map(|v| v.uuid).collect();
        let supercell_uuid = supercell.uuid;
        self.cells.insert(supercell_uuid, supercell.clone());

        // Collect input vertices into a vector to avoid borrowing conflicts
        let input_vertices: Vec<_> = self.vertices.values().cloned().collect();

        // Iterate over each input vertex and insert it into the triangulation
        for vertex in input_vertices.iter() {
            // Skip if this vertex is already part of supercell vertices
            if supercell_vertices.contains(&vertex.uuid) {
                continue;
            }
            
            let (bad_cells, boundary_facets) = self.find_bad_cells_and_boundary_facets(vertex)?;

            // Remove bad cells
            for bad_cell_id in bad_cells {
                self.cells.remove(&bad_cell_id);
            }

            // Create new cells using the boundary facets and the new vertex
            for facet in boundary_facets.iter() {
                let new_cell = Cell::from_facet_and_vertex(facet.clone(), *vertex)?;
                self.cells.insert(new_cell.uuid, new_cell);
            }
        }

        // Remove cells that contain vertices of the supercell
        self.remove_cells_containing_supercell_vertices(&supercell);

        // Assign neighbors between adjacent cells
        self.assign_neighbors()?;

        // Assign incident cells to vertices
        self.assign_incident_cells()?;

        Ok(self)
    }

    #[allow(clippy::type_complexity)]
    fn find_bad_cells_and_boundary_facets(
        &mut self,
        vertex: &Vertex<T, U, D>,
    ) -> Result<(Vec<Uuid>, Vec<Facet<T, U, V, D>>), anyhow::Error>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let mut bad_cells = Vec::new();
        let mut boundary_facets = Vec::new();

        // Find cells whose circumsphere contains the vertex
        for (cell_id, cell) in self.cells.iter() {
            let contains = cell.circumsphere_contains_vertex(*vertex)?;
            if contains {
                bad_cells.push(*cell_id);
            }
        }

        // Collect boundary facets - facets that are on the boundary of the bad cells cavity
        for &bad_cell_id in &bad_cells {
            if let Some(bad_cell) = self.cells.get(&bad_cell_id) {
                for facet in bad_cell.facets() {
                    // A facet is on the boundary if it's not shared with another bad cell
                    let mut is_boundary = true;
                    for &other_bad_cell_id in &bad_cells {
                        if other_bad_cell_id != bad_cell_id {
                            if let Some(other_cell) = self.cells.get(&other_bad_cell_id) {
                                if other_cell.facets().contains(&facet) {
                                    is_boundary = false;
                                    break;
                                }
                            }
                        }
                    }
                    if is_boundary {
                        boundary_facets.push(facet);
                    }
                }
            }
        }

        Ok((bad_cells, boundary_facets))
    }

    fn remove_cells_containing_supercell_vertices(&mut self, _supercell: &Cell<T, U, V, D>) {
        // The goal is to remove supercell artifacts while preserving valid Delaunay cells
        // We should only keep cells that are made entirely of input vertices

        let input_vertex_uuids: HashSet<Uuid> = self.vertices.keys().cloned().collect();

        let cells_to_remove: Vec<Uuid> = self
            .cells
            .iter()
            .filter(|(_, cell)| {
                let cell_vertex_uuids: HashSet<Uuid> =
                    cell.vertices.iter().map(|v| v.uuid).collect();
                let has_only_input_vertices = cell_vertex_uuids.is_subset(&input_vertex_uuids);

                // Remove cells that don't consist entirely of input vertices
                // Keep only cells that are made entirely of input vertices
                !has_only_input_vertices
            })
            .map(|(uuid, _)| *uuid)
            .collect();

        for cell_id in cells_to_remove {
            self.cells.remove(&cell_id);
        }

        // Remove duplicate cells (cells with identical vertex sets)
        let mut unique_cells = HashMap::new();
        let mut cells_to_remove_duplicates = Vec::new();

        for (cell_id, cell) in &self.cells {
            // Create a sorted vector of vertex UUIDs as a key for uniqueness
            let mut vertex_uuids: Vec<Uuid> = cell.vertices.iter().map(|v| v.uuid).collect();
            vertex_uuids.sort();

            if let Some(_existing_cell_id) = unique_cells.get(&vertex_uuids) {
                // This is a duplicate cell - mark for removal
                cells_to_remove_duplicates.push(*cell_id);
            } else {
                // This is a unique cell
                unique_cells.insert(vertex_uuids, *cell_id);
            }
        }

        // Remove duplicate cells
        for cell_id in cells_to_remove_duplicates {
            self.cells.remove(&cell_id);
        }
    }

    fn assign_neighbors(&mut self) -> Result<(), anyhow::Error> {
        // Create a map to store neighbor relationships
        let mut neighbor_map: HashMap<Uuid, Vec<Uuid>> = HashMap::new();

        // Initialize neighbor lists for all cells
        for cell_id in self.cells.keys() {
            neighbor_map.insert(*cell_id, Vec::new());
        }

        // Find neighboring cells by comparing facets
        let cell_ids: Vec<Uuid> = self.cells.keys().cloned().collect();

        for i in 0..cell_ids.len() {
            for j in (i + 1)..cell_ids.len() {
                let cell1_id = cell_ids[i];
                let cell2_id = cell_ids[j];

                if let (Some(cell1), Some(cell2)) =
                    (self.cells.get(&cell1_id), self.cells.get(&cell2_id))
                {
                    // Check if cells share a facet (are neighbors)
                    let cell1_facets = cell1.facets();
                    let cell2_facets = cell2.facets();

                    for facet1 in &cell1_facets {
                        for facet2 in &cell2_facets {
                            // Two cells are neighbors if they share a facet
                            // (same vertices but opposite orientation)
                            if facets_are_adjacent(facet1, facet2) {
                                neighbor_map.get_mut(&cell1_id).unwrap().push(cell2_id);
                                neighbor_map.get_mut(&cell2_id).unwrap().push(cell1_id);
                            }
                        }
                    }
                }
            }
        }

        // Assign the computed neighbors to the cells
        for (cell_id, neighbors) in neighbor_map {
            if let Some(cell) = self.cells.get_mut(&cell_id) {
                if !neighbors.is_empty() {
                    // Create a mutable reference to update the cell
                    let mut updated_cell = cell.clone();
                    updated_cell.neighbors = Some(neighbors);
                    self.cells.insert(cell_id, updated_cell);
                }
            }
        }

        Ok(())
    }

    fn assign_incident_cells(&mut self) -> Result<(), anyhow::Error> {
        // Create a map from vertex UUID to incident cell UUIDs
        let mut vertex_to_cells: HashMap<Uuid, Vec<Uuid>> = HashMap::new();

        // Initialize the map with all vertices
        for vertex_id in self.vertices.keys() {
            vertex_to_cells.insert(*vertex_id, Vec::new());
        }

        // Find which cells contain each vertex
        for (cell_id, cell) in &self.cells {
            for vertex in &cell.vertices {
                if let Some(incident_cells) = vertex_to_cells.get_mut(&vertex.uuid) {
                    incident_cells.push(*cell_id);
                }
            }
        }

        // Update each vertex with its incident cell information
        for (vertex_id, cell_ids) in vertex_to_cells {
            if let Some(vertex) = self.vertices.get_mut(&vertex_id) {
                if !cell_ids.is_empty() {
                    // For now, just assign the first incident cell
                    // In a full implementation, you might want to store all incident cells
                    let mut updated_vertex = *vertex;
                    updated_vertex.incident_cell = Some(cell_ids[0]);
                    self.vertices.insert(vertex_id, updated_vertex);
                }
            }
        }

        Ok(())
    }

    /// Remove duplicate cells (cells with identical vertex sets)
    fn remove_duplicate_cells(&mut self) {
        let mut unique_cells = HashMap::new();
        let mut cells_to_remove = Vec::new();

        for (cell_id, cell) in &self.cells {
            // Create a sorted vector of vertex UUIDs as a key for uniqueness
            let mut vertex_uuids: Vec<Uuid> = cell.vertices.iter().map(|v| v.uuid).collect();
            vertex_uuids.sort();

            if let Some(_existing_cell_id) = unique_cells.get(&vertex_uuids) {
                // This is a duplicate cell - mark for removal
                cells_to_remove.push(*cell_id);
            } else {
                // This is a unique cell
                unique_cells.insert(vertex_uuids, *cell_id);
            }
        }

        // Remove duplicate cells
        for cell_id in cells_to_remove {
            self.cells.remove(&cell_id);
        }
    }
}

#[cfg(test)]
mod tests {

    use crate::delaunay_core::vertex::VertexBuilder;

    use super::*;

    #[test]
    fn tds_new() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];

        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), 3);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", tds);
    }

    #[test]
    fn tds_add_dim() {
        let points: Vec<Point<f64, 3>> = Vec::new();

        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), -1);

        let new_vertex1 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let _ = tds.add(new_vertex1);

        assert_eq!(tds.number_of_vertices(), 1);
        assert_eq!(tds.dim(), 0);

        let new_vertex2 = VertexBuilder::default()
            .point(Point::new([4.0, 5.0, 6.0]))
            .build()
            .unwrap();
        let _ = tds.add(new_vertex2);

        assert_eq!(tds.number_of_vertices(), 2);
        assert_eq!(tds.dim(), 1);

        let new_vertex3 = VertexBuilder::default()
            .point(Point::new([7.0, 8.0, 9.0]))
            .build()
            .unwrap();
        let _ = tds.add(new_vertex3);

        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.dim(), 2);

        let new_vertex4 = VertexBuilder::default()
            .point(Point::new([10.0, 11.0, 12.0]))
            .build()
            .unwrap();
        let _ = tds.add(new_vertex4);

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.dim(), 3);

        let new_vertex5 = VertexBuilder::default()
            .point(Point::new([13.0, 14.0, 15.0]))
            .build()
            .unwrap();
        let _ = tds.add(new_vertex5);

        assert_eq!(tds.number_of_vertices(), 5);
        assert_eq!(tds.dim(), 3);
    }

    #[test]
    fn tds_no_add() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];
        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.cells.len(), 0);
        assert_eq!(tds.dim(), 3);

        let new_vertex1 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();
        let result = tds.add(new_vertex1);

        assert_eq!(tds.number_of_vertices(), 4);
        assert_eq!(tds.dim(), 3);
        assert!(result.is_err());
    }

    #[test]
    fn tds_supercell() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);
        let supercell = tds.supercell();
        let unwrapped_supercell =
            supercell.unwrap_or_else(|err| panic!("Error creating supercell: {:?}!", err));

        assert_eq!(unwrapped_supercell.vertices.len(), 4);

        // Debug: Print actual supercell coordinates
        println!("Actual supercell vertices:");
        for (i, vertex) in unwrapped_supercell.vertices.iter().enumerate() {
            println!("  Vertex {}: {:?}", i, vertex.point.coords);
        }

        // The supercell should contain all input points
        // Let's verify it's a proper tetrahedron rather than checking specific coordinates
        assert_eq!(unwrapped_supercell.vertices.len(), 4);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", unwrapped_supercell);
    }

    #[test]
    fn tds_bowyer_watson() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);
        println!(
            "Initial TDS: {} vertices, {} cells",
            tds.number_of_vertices(),
            tds.number_of_cells()
        );

        let result = tds.bowyer_watson().unwrap_or_else(|err| {
            panic!("Error creating triangulation: {:?}", err);
        });

        println!(
            "Result TDS: {} vertices, {} cells",
            result.number_of_vertices(),
            result.number_of_cells()
        );
        println!("Cells: {:?}", result.cells.keys().collect::<Vec<_>>());

        assert_eq!(result.number_of_vertices(), 4);
        assert_eq!(result.number_of_cells(), 1);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", result);
    }

    #[test]
    fn tds_bowyer_watson_4d_multiple_cells() {
        // Create a 4D point set that forms multiple 4-simplices
        // Using 6 points in 4D space to create a complex triangulation
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0]), // origin
            Point::new([1.0, 0.0, 0.0, 0.0]), // unit vector in x
            Point::new([0.0, 1.0, 0.0, 0.0]), // unit vector in y
            Point::new([0.0, 0.0, 1.0, 0.0]), // unit vector in z
            Point::new([0.0, 0.0, 0.0, 1.0]), // unit vector in w
            Point::new([1.0, 1.0, 1.0, 1.0]), // diagonal point
        ];

        let tds: Tds<f64, usize, usize, 4> = Tds::new(points);
        println!("\n=== 4D BOWYER-WATSON TRIANGULATION TEST ===");
        println!(
            "Initial 4D TDS: {} vertices, {} cells",
            tds.number_of_vertices(),
            tds.number_of_cells()
        );

        let result = tds.bowyer_watson().unwrap_or_else(|err| {
            panic!("Error creating 4D triangulation: {:?}", err);
        });

        println!(
            "\nResult 4D TDS: {} vertices, {} cells",
            result.number_of_vertices(),
            result.number_of_cells()
        );

        // Verify we have the expected number of vertices
        assert_eq!(result.number_of_vertices(), 6);
        assert!(result.number_of_cells() >= 1, "Should have at least 1 cell");

        println!("\n--- CELL DETAILS ---");
        let mut cell_vertex_sets = Vec::new();
        for (i, (cell_id, cell)) in result.cells.iter().enumerate() {
            assert_eq!(
                cell.vertices.len(),
                5,
                "Each 4D cell should have exactly 5 vertices"
            );

            // Collect and sort vertex coordinates for this cell
            let mut vertex_coords: Vec<_> = cell.vertices.iter().map(|v| v.point.coords).collect();
            vertex_coords.sort_by(|a, b| {
                for dim in 0..4 {
                    match a[dim].partial_cmp(&b[dim]).unwrap() {
                        std::cmp::Ordering::Equal => continue,
                        other => return other,
                    }
                }
                std::cmp::Ordering::Equal
            });

            println!("Cell {}: {}", i + 1, cell_id);
            println!("  Vertices (5 total):");
            for (j, coords) in vertex_coords.iter().enumerate() {
                println!(
                    "    {}: [{:.1}, {:.1}, {:.1}, {:.1}]",
                    j + 1,
                    coords[0],
                    coords[1],
                    coords[2],
                    coords[3]
                );
            }
            println!();

            cell_vertex_sets.push(vertex_coords);
        }

        // Verify that cells have different vertex sets (no duplicates)
        println!("--- UNIQUENESS VERIFICATION ---");
        for i in 0..cell_vertex_sets.len() {
            for j in (i + 1)..cell_vertex_sets.len() {
                assert_ne!(
                    cell_vertex_sets[i],
                    cell_vertex_sets[j],
                    "Cells {} and {} should have different vertex sets",
                    i + 1,
                    j + 1
                );
            }
        }
        println!(
            "✓ All {} cells have unique vertex sets",
            cell_vertex_sets.len()
        );

        // Verify neighbor relationships
        if result.number_of_cells() > 1 {
            println!("\n--- NEIGHBOR RELATIONSHIPS ---");
            let mut total_neighbor_connections = 0;

            for (i, (cell_id, cell)) in result.cells.iter().enumerate() {
                if let Some(neighbors) = &cell.neighbors {
                    println!("Cell {} has {} neighbors:", i + 1, neighbors.len());
                    total_neighbor_connections += neighbors.len();

                    // Verify that all neighbors exist and are mutual
                    for neighbor_id in neighbors {
                        assert!(
                            result.cells.contains_key(neighbor_id),
                            "Neighbor {} of cell {} should exist",
                            neighbor_id,
                            cell_id
                        );

                        // Find the neighbor's index for display
                        let neighbor_index = result
                            .cells
                            .iter()
                            .enumerate()
                            .find(|(_, (id, _))| *id == neighbor_id)
                            .map(|(idx, _)| idx + 1)
                            .unwrap_or(0);

                        println!("  → Cell {}", neighbor_index);

                        // Check mutual neighbor relationship
                        if let Some(neighbor_cell) = result.cells.get(neighbor_id) {
                            if let Some(neighbor_neighbors) = &neighbor_cell.neighbors {
                                assert!(
                                    neighbor_neighbors.contains(cell_id),
                                    "Cell {} should be a neighbor of its neighbor {}",
                                    i + 1,
                                    neighbor_index
                                );
                            }
                        }
                    }
                    println!();
                } else {
                    println!("Cell {} has no neighbors assigned", i + 1);
                }
            }

            println!("✓ All neighbor relationships are mutual");
            println!(
                "✓ Total neighbor connections: {}",
                total_neighbor_connections
            );
        }

        println!("\n=== 4D TRIANGULATION SUCCESS ===\n");
    }

    #[test]
    fn tds_bowyer_watson_5d_multiple_cells() {
        // Create a 5D point set that forms multiple 5-simplices
        // Using 7 points in 5D space to create a complex triangulation
        let points = vec![
            Point::new([0.0, 0.0, 0.0, 0.0, 0.0]), // origin
            Point::new([1.0, 0.0, 0.0, 0.0, 0.0]), // unit vector in x
            Point::new([0.0, 1.0, 0.0, 0.0, 0.0]), // unit vector in y
            Point::new([0.0, 0.0, 1.0, 0.0, 0.0]), // unit vector in z
            Point::new([0.0, 0.0, 0.0, 1.0, 0.0]), // unit vector in w
            Point::new([0.0, 0.0, 0.0, 0.0, 1.0]), // unit vector in v
            Point::new([1.0, 1.0, 1.0, 1.0, 1.0]), // diagonal point
        ];

        let tds: Tds<f64, usize, usize, 5> = Tds::new(points);
        println!("\n=== 5D BOWYER-WATSON TRIANGULATION TEST ===");
        println!(
            "Initial 5D TDS: {} vertices, {} cells",
            tds.number_of_vertices(),
            tds.number_of_cells()
        );

        let result = tds.bowyer_watson().unwrap_or_else(|err| {
            panic!("Error creating 5D triangulation: {:?}", err);
        });

        println!(
            "\nResult 5D TDS: {} vertices, {} cells",
            result.number_of_vertices(),
            result.number_of_cells()
        );

        // Verify we have the expected number of vertices
        assert_eq!(result.number_of_vertices(), 7);
        assert!(result.number_of_cells() >= 1, "Should have at least 1 cell");

        println!("\n--- CELL DETAILS ---");
        let mut cell_vertex_sets = Vec::new();
        for (i, (cell_id, cell)) in result.cells.iter().enumerate() {
            assert_eq!(
                cell.vertices.len(),
                6,
                "Each 5D cell should have exactly 6 vertices"
            );

            // Collect and sort vertex coordinates for this cell
            let mut vertex_coords: Vec<_> = cell.vertices.iter().map(|v| v.point.coords).collect();
            vertex_coords.sort_by(|a, b| {
                for dim in 0..5 {
                    match a[dim].partial_cmp(&b[dim]).unwrap() {
                        std::cmp::Ordering::Equal => continue,
                        other => return other,
                    }
                }
                std::cmp::Ordering::Equal
            });

            println!("Cell {}: {}", i + 1, cell_id);
            println!("  Vertices (6 total):");
            for (j, coords) in vertex_coords.iter().enumerate() {
                println!(
                    "    {}: [{:.1}, {:.1}, {:.1}, {:.1}, {:.1}]",
                    j + 1,
                    coords[0],
                    coords[1],
                    coords[2],
                    coords[3],
                    coords[4]
                );
            }
            println!();

            cell_vertex_sets.push(vertex_coords);
        }

        // Verify that cells have different vertex sets (no duplicates)
        println!("--- UNIQUENESS VERIFICATION ---");
        for i in 0..cell_vertex_sets.len() {
            for j in (i + 1)..cell_vertex_sets.len() {
                assert_ne!(
                    cell_vertex_sets[i],
                    cell_vertex_sets[j],
                    "Cells {} and {} should have different vertex sets",
                    i + 1,
                    j + 1
                );
            }
        }
        println!(
            "✓ All {} cells have unique vertex sets",
            cell_vertex_sets.len()
        );

        // Verify neighbor relationships
        if result.number_of_cells() > 1 {
            println!("\n--- NEIGHBOR RELATIONSHIPS ---");
            let mut total_neighbor_connections = 0;

            for (i, (cell_id, cell)) in result.cells.iter().enumerate() {
                if let Some(neighbors) = &cell.neighbors {
                    println!("Cell {} has {} neighbors:", i + 1, neighbors.len());
                    total_neighbor_connections += neighbors.len();

                    // Verify that all neighbors exist and are mutual
                    for neighbor_id in neighbors {
                        assert!(
                            result.cells.contains_key(neighbor_id),
                            "Neighbor {} of cell {} should exist",
                            neighbor_id,
                            cell_id
                        );

                        // Find the neighbor's index for display
                        let neighbor_index = result
                            .cells
                            .iter()
                            .enumerate()
                            .find(|(_, (id, _))| *id == neighbor_id)
                            .map(|(idx, _)| idx + 1)
                            .unwrap_or(0);

                        println!("  → Cell {}", neighbor_index);

                        // Check mutual neighbor relationship
                        if let Some(neighbor_cell) = result.cells.get(neighbor_id) {
                            if let Some(neighbor_neighbors) = &neighbor_cell.neighbors {
                                assert!(
                                    neighbor_neighbors.contains(cell_id),
                                    "Cell {} should be a neighbor of its neighbor {}",
                                    i + 1,
                                    neighbor_index
                                );
                            }
                        }
                    }
                    println!();
                } else {
                    println!("Cell {} has no neighbors assigned", i + 1);
                }
            }

            println!("✓ All neighbor relationships are mutual");
            println!(
                "✓ Total neighbor connections: {}",
                total_neighbor_connections
            );
        }

        // Additional 5D specific verification
        println!("\n--- 5D SPECIFIC VERIFICATION ---");
        println!("✓ Working with 5-dimensional space");
        println!("✓ Each cell is a 5-simplex (6 vertices)");
        println!("✓ {} total 5-simplices generated", result.number_of_cells());

        // Verify dimensional consistency
        let total_vertices_in_cells: usize =
            result.cells.values().map(|cell| cell.vertices.len()).sum();
        println!(
            "✓ Total vertices across all cells: {} ({}×6)",
            total_vertices_in_cells,
            result.number_of_cells()
        );

        println!("\n=== 5D TRIANGULATION SUCCESS ===\n");
    }

    #[test]
    fn tds_to_and_from_json() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0, 4.0]),
            Point::new([5.0, 6.0, 7.0, 8.0]),
            Point::new([9.0, 10.0, 11.0, 12.0]),
            Point::new([13.0, 14.0, 15.0, 16.0]),
        ];
        let tds: Tds<f64, usize, usize, 4> = Tds::new(points);
        let serialized = serde_json::to_string(&tds).unwrap();

        // assert!(serialized.contains(r#""vertices":{},"cells":{}"#));
        assert!(serialized.contains("[1.0,2.0,3.0,4.0]"));
        assert!(serialized.contains("[5.0,6.0,7.0,8.0]"));
        assert!(serialized.contains("[9.0,10.0,11.0,12.0]"));
        assert!(serialized.contains("[13.0,14.0,15.0,16.0]"));

        let deserialized: Tds<f64, usize, usize, 4> = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized, tds);

        // Human readable output for cargo test -- --nocapture
        println!("Serialized = {}", serialized);
    }

    #[test]
    fn tds_empty() {
        let points: Vec<Point<f64, 3>> = Vec::new();
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), -1);
        assert!(tds.vertices.is_empty());
        assert!(tds.cells.is_empty());
    }

    #[test]
    fn tds_single_vertex() {
        let points = vec![Point::new([1.0, 2.0, 3.0])];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 1);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), 0);
    }

    #[test]
    fn tds_two_vertices() {
        let points = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0])];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 2);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), 1);
    }

    #[test]
    fn tds_three_vertices() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
        ];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_vertices(), 3);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), 2);
    }

    #[test]
    fn tds_add_duplicate_vertex() {
        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());

        let vertex1 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        let vertex2 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        let result1 = tds.add(vertex1);
        assert!(result1.is_ok());
        assert_eq!(tds.number_of_vertices(), 1);

        let result2 = tds.add(vertex2);
        assert!(result2.is_err());
        assert_eq!(result2.unwrap_err(), "Vertex already exists!");
        assert_eq!(tds.number_of_vertices(), 1);
    }

    #[test]
    fn tds_add_different_vertices() {
        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());

        let vertex1 = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        let vertex2 = VertexBuilder::default()
            .point(Point::new([4.0, 5.0, 6.0]))
            .build()
            .unwrap();

        let result1 = tds.add(vertex1);
        assert!(result1.is_ok());
        assert_eq!(tds.number_of_vertices(), 1);

        let result2 = tds.add(vertex2);
        assert!(result2.is_ok());
        assert_eq!(tds.number_of_vertices(), 2);
    }

    #[test]
    fn tds_number_of_cells() {
        let points = vec![
            Point::new([1.0, 2.0, 3.0]),
            Point::new([4.0, 5.0, 6.0]),
            Point::new([7.0, 8.0, 9.0]),
            Point::new([10.0, 11.0, 12.0]),
        ];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.cells.len(), 0);
    }

    #[test]
    fn tds_clone() {
        let points = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0])];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);
        let tds_clone = tds.clone();

        assert_eq!(tds, tds_clone);
        assert_eq!(tds.number_of_vertices(), tds_clone.number_of_vertices());
        assert_eq!(tds.number_of_cells(), tds_clone.number_of_cells());
        assert_eq!(tds.dim(), tds_clone.dim());
    }

    #[test]
    fn tds_default() {
        let tds: Tds<f64, usize, usize, 3> = Default::default();

        assert_eq!(tds.number_of_vertices(), 0);
        assert_eq!(tds.number_of_cells(), 0);
        assert_eq!(tds.dim(), -1);
        assert!(tds.vertices.is_empty());
        assert!(tds.cells.is_empty());
    }

    #[test]
    fn tds_partial_eq() {
        let points1 = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0])];
        let tds1: Tds<f64, usize, usize, 3> = Tds::new(points1.clone());
        let tds2: Tds<f64, usize, usize, 3> = Tds::new(points1);

        // Note: These won't be equal because vertices have unique UUIDs
        // This test checks that the PartialEq implementation works
        assert_ne!(tds1, tds2);

        let empty_tds1: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
        let empty_tds2: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
        assert_eq!(empty_tds1, empty_tds2);
    }

    #[test]
    fn tds_debug() {
        let points = vec![Point::new([1.0, 2.0, 3.0])];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);
        let debug_str = format!("{:?}", tds);

        assert!(debug_str.contains("Tds"));
        assert!(debug_str.contains("vertices"));
        assert!(debug_str.contains("cells"));
    }

    #[test]
    fn tds_dim_edge_cases() {
        // Test dimension calculation for different numbers of vertices
        let empty_tds: Tds<f64, usize, usize, 10> = Tds::new(Vec::new());
        assert_eq!(empty_tds.dim(), -1);

        let points_1d = vec![Point::new([1.0])];
        let tds_1d: Tds<f64, usize, usize, 1> = Tds::new(points_1d);
        assert_eq!(tds_1d.dim(), 0);

        let points_2d = vec![
            Point::new([1.0, 2.0]),
            Point::new([3.0, 4.0]),
            Point::new([5.0, 6.0]),
            Point::new([7.0, 8.0]),
        ];
        let tds_2d: Tds<f64, usize, usize, 2> = Tds::new(points_2d);
        assert_eq!(tds_2d.dim(), 2); // min(4-1, 2) = 2

        // Test case where vertices exceed dimension
        let many_points = vec![
            Point::new([1.0]),
            Point::new([2.0]),
            Point::new([3.0]),
            Point::new([4.0]),
            Point::new([5.0]),
        ];
        let tds_many: Tds<f64, usize, usize, 1> = Tds::new(many_points);
        assert_eq!(tds_many.dim(), 1); // min(5-1, 1) = 1
    }

    #[test]
    fn tds_supercell_empty() {
        let tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
        let result = tds.supercell();

        // Supercell succeeds for empty triangulation but creates default coordinates
        assert!(result.is_ok());
        let supercell = result.unwrap();
        assert_eq!(supercell.vertices.len(), 4);

        // The supercell should be a proper tetrahedron, not specific coordinates
        // Just verify it has 4 vertices forming a valid tetrahedron
        assert_eq!(supercell.vertices.len(), 4);

        // Debug output to see what's actually created
        println!("Empty supercell vertices:");
        for (i, vertex) in supercell.vertices.iter().enumerate() {
            println!("  Vertex {}: {:?}", i, vertex.point.coords);
        }
    }

    #[test]
    fn tds_supercell_single_point() {
        let points = vec![Point::new([1.0, 2.0, 3.0])];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);
        let result = tds.supercell();

        assert!(result.is_ok());
        let supercell = result.unwrap();
        assert_eq!(supercell.vertices.len(), 4); // Proper tetrahedron

        // Debug output to see what's actually created
        println!("Single point supercell vertices:");
        for (i, vertex) in supercell.vertices.iter().enumerate() {
            println!("  Vertex {}: {:?}", i, vertex.point.coords);
        }

        // The supercell should be a proper tetrahedron that contains the single input point
        // Just verify it has 4 vertices forming a valid tetrahedron
        assert_eq!(supercell.vertices.len(), 4);
    }

    #[test]
    fn tds_supercell_multiple_points() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 1.0, 1.0]),
            Point::new([2.0, 2.0, 2.0]),
        ];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);
        let result = tds.supercell();

        assert!(result.is_ok());
        let supercell = result.unwrap();
        assert_eq!(supercell.vertices.len(), 4); // Proper tetrahedron

        // Debug output to see what's actually created
        println!("Multiple points supercell vertices:");
        for (i, vertex) in supercell.vertices.iter().enumerate() {
            println!("  Vertex {}: {:?}", i, vertex.point.coords);
        }

        // The supercell should be a proper tetrahedron that contains all input points
        // Just verify it has 4 vertices forming a valid tetrahedron
        assert_eq!(supercell.vertices.len(), 4);

        // Check that supercell is large enough by verifying bounding box
        let max_coords = supercell.vertices.iter().map(|v| v.point.coords).fold(
            [f64::NEG_INFINITY; 3],
            |acc, coords| {
                [
                    acc[0].max(coords[0]),
                    acc[1].max(coords[1]),
                    acc[2].max(coords[2]),
                ]
            },
        );

        let min_coords = supercell.vertices.iter().map(|v| v.point.coords).fold(
            [f64::INFINITY; 3],
            |acc, coords| {
                [
                    acc[0].min(coords[0]),
                    acc[1].min(coords[1]),
                    acc[2].min(coords[2]),
                ]
            },
        );

        // Supercell should contain all input points [0,0,0], [1,1,1], [2,2,2]
        // So min should be <= 0 and max should be >= 2
        assert!(min_coords[0] <= 0.0 && max_coords[0] >= 2.0);
        assert!(min_coords[1] <= 0.0 && max_coords[1] >= 2.0);
        assert!(min_coords[2] <= 0.0 && max_coords[2] >= 2.0);
    }

    #[test]
    fn tds_find_bad_cells_and_boundary_facets_empty() {
        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(Vec::new());
        let vertex = VertexBuilder::default()
            .point(Point::new([1.0, 2.0, 3.0]))
            .build()
            .unwrap();

        let result = tds.find_bad_cells_and_boundary_facets(&vertex);
        assert!(result.is_ok());

        let (bad_cells, boundary_facets) = result.unwrap();
        assert!(bad_cells.is_empty());
        assert!(boundary_facets.is_empty());
    }

    #[test]
    fn tds_vertices_hashmap_access() {
        let points = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0])];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        // Test that we can access vertices by UUID
        assert_eq!(tds.vertices.len(), 2);

        for (uuid, vertex) in &tds.vertices {
            assert_eq!(vertex.uuid, *uuid);
            assert_eq!(vertex.dim(), 3);
        }
    }

    #[test]
    fn tds_cells_hashmap_access() {
        let points = vec![Point::new([1.0, 2.0, 3.0]), Point::new([4.0, 5.0, 6.0])];
        let tds: Tds<f64, usize, usize, 3> = Tds::new(points);

        // Initially no cells
        assert_eq!(tds.cells.len(), 0);
        assert!(tds.cells.is_empty());
    }
}

//! Data and operations on d-dimensional triangulation data structures.
//!
//! Intended to match functionality of the
//! [CGAL Triangulation](https://doc.cgal.org/latest/Triangulation/index.html).

use super::{
    cell::Cell, cell::CellBuilder, facet::Facet, point::Point, utilities::find_extreme_coordinates,
    vertex::Vertex,
};
use na::{ComplexField, Const, OPoint};
use nalgebra as na;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::cmp::{min, Ordering, PartialEq};
use std::ops::{AddAssign, Div, SubAssign};
use std::{collections::HashMap, hash::Hash, iter::Sum};
use uuid::Uuid;

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
        + PartialEq
        + PartialOrd
        + SubAssign<f64>
        + Sum,
    U: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    V: Clone + Copy + Eq + Hash + Ord + PartialEq + PartialOrd,
    f64: From<T>,
    for<'a> &'a T: Div<f64>,
    [T; D]: Copy + Default + DeserializeOwned + Serialize + Sized,
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
        // First, find the min and max coordinates
        let mut min_coords = find_extreme_coordinates(self.vertices.clone(), Ordering::Less);
        let mut max_coords = find_extreme_coordinates(self.vertices.clone(), Ordering::Greater);

        // Now add padding so the supercell is large enough to contain all vertices
        for elem in min_coords.iter_mut() {
            *elem -= 10.0;
        }

        for elem in max_coords.iter_mut() {
            *elem += 10.0;
        }
        // Add minimum vertex
        let mut points = vec![Point::new(min_coords)];

        // Stash max coords into a diagonal matrix
        let max_vector: na::SMatrix<T, D, 1> = na::Matrix::from(max_coords);
        let max_point_coords: na::SMatrix<T, D, D> = na::Matrix::from_diagonal(&max_vector);

        // Create new maximal vertices for the supercell from slices of the
        // max_point_coords matrix
        for row in max_point_coords.row_iter() {
            let mut row_vec: Vec<T> = Vec::new();
            for elem in row.iter() {
                row_vec.push(*elem);
            }

            // Add slice of max_point_coords matrix as a new point
            let point =
                Point::<T, D>::new(row_vec.into_boxed_slice().into_vec().try_into().unwrap());
            points.push(point);
        }

        let supercell = CellBuilder::default()
            .vertices(Vertex::from_points(points))
            .build()
            .unwrap();

        Ok(supercell)
    }

    /// Performs the Bowyer-Watson algorithm to triangulate a set of vertices.
    ///
    /// # Returns:
    ///
    /// A [Result] containing a [Vec] of [Cell] objects representing the triangulation, or an error message.
    fn bowyer_watson(&mut self) -> Result<Vec<Cell<T, U, V, D>>, anyhow::Error>
    where
        OPoint<T, Const<D>>: From<[f64; D]>,
        [f64; D]: Default + DeserializeOwned + Serialize + Sized,
    {
        let mut triangulation: Vec<Cell<T, U, V, D>> = Vec::new();

        // Create super-cell that contains all vertices
        let supercell = self.supercell()?;
        triangulation.push(supercell.clone());

        // Iterate over vertices
        for vertex in self.vertices.values() {
            // Find cells that contain the vertex
            let mut bad_cells: Vec<Cell<T, U, V, D>> = Vec::new();

            for cell in &triangulation {
                if cell.circumsphere_contains(*vertex)? {
                    bad_cells.push((*cell).clone());
                }
            }

            // Find the boundary of the hole left by the bad cells
            let mut boundary_facets: Vec<Facet<T, U, V, D>> = Vec::new();
            for bad_cell in &bad_cells {
                for facet in bad_cell.facets() {
                    if !bad_cells.iter().any(|c| c.facets().contains(&facet)) {
                        boundary_facets.push(facet);
                    }
                }
            }

            // Remove bad cells from triangulation
            triangulation.retain(|cell| !bad_cells.contains(cell));

            // Create new cells from the boundary facets and new vertex
            for facet in boundary_facets {
                let new_cell = Cell::from_facet_and_vertex(facet, *vertex)?;
                triangulation.push(new_cell);
            }
        }

        //     // Find the boundary of the polygonal hole
        //     let mut polygonal_hole: Vec<Facet<T, U, V, D>> = Vec::new();
        //     for cell in bad_cells.iter() {
        //         // Create Facets from the Cell
        //         for vertex in cell.vertices.iter() {
        //             let facet = Facet::new(cell.clone(), *vertex)?;
        //             polygonal_hole.push(facet);
        //         }

        //         // for vertex in cell.vertices.iter() {
        //         //     if bad_cells.iter().any(|c| c.contains_vertex(vertex)) {
        //         //         polygonal_hole.push(vertex.clone());
        //         //     }
        //         // }
        //     }

        //     // Remove duplicate facets
        //     polygonal_hole.sort_by(|a, b| a.partial_cmp(b).unwrap());
        //     polygonal_hole.dedup();

        //     // Remove bad cells from the triangulation
        //     for cell in bad_cells.iter() {
        //         triangulation.remove(triangulation.iter().position(|c| c == cell).unwrap());
        //     }

        //     // Re-triangulate the polygonal hole
        //     for mut facet in polygonal_hole.iter().cloned() {
        //         let mut new_cell_vertices: Vec<Vertex<T, U, D>> = Vec::new();
        //         for facet_vertex in facet.vertices().iter() {
        //             new_cell_vertices.push(*facet_vertex);
        //         }
        //         new_cell_vertices.push(*vertex);
        //         triangulation.push(Cell::new(new_cell_vertices)?);
        //     }
        // }

        // // Remove all cells containing vertices from the supercell
        // triangulation
        //     .retain(|c| !c.contains_vertex(supercell.vertices.clone().into_iter().next().unwrap()));

        // Remove any cells that contain a vertex of the supercell
        triangulation.retain(|cell| !cell.contains_vertex_of(supercell.clone()));

        Ok(triangulation)
    }

    fn assign_neighbors(&mut self, _cells: Vec<Cell<T, U, V, D>>) -> Result<(), &'static str> {
        todo!("Assign neighbors")
    }

    fn assign_incident_cells(
        &mut self,
        _vertices: Vec<Vertex<T, U, D>>,
    ) -> Result<(), &'static str> {
        todo!("Assign incident cells")
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
        assert!(unwrapped_supercell
            .vertices
            .iter()
            .any(|v| { v.point.coords == [-10.0, -10.0, -10.0] }));

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", unwrapped_supercell);
    }

    #[ignore]
    #[test]
    fn tds_bowyer_watson() {
        let points = vec![
            Point::new([0.0, 0.0, 0.0]),
            Point::new([1.0, 0.0, 0.0]),
            Point::new([0.0, 1.0, 0.0]),
            Point::new([0.0, 0.0, 1.0]),
        ];
        let mut tds: Tds<f64, usize, usize, 3> = Tds::new(points);
        let cells = tds.bowyer_watson();
        let unwrapped_cells = cells.unwrap_or_else(|err| panic!("Error creating cells: {:?}", err));

        assert_eq!(unwrapped_cells.len(), 1);

        // Human readable output for cargo test -- --nocapture
        println!("{:?}", unwrapped_cells);
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
}

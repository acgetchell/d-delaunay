# d-delaunay

[![CI](https://github.com/acgetchell/d-delaunay/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/d-delaunay/actions/workflows/ci.yml)
[![rust-clippy analyze](https://github.com/acgetchell/d-delaunay/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/d-delaunay/actions/workflows/rust-clippy.yml)
[![codecov](https://codecov.io/gh/acgetchell/d-delaunay/graph/badge.svg?token=WT7qZGT9bO)](https://codecov.io/gh/acgetchell/d-delaunay)
[![Audit dependencies](https://github.com/acgetchell/d-delaunay/actions/workflows/audit.yml/badge.svg)](https://github.com/acgetchell/d-delaunay/actions/workflows/audit.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/3cad94f994f5434d877ae77f0daee692)](https://app.codacy.com/gh/acgetchell/d-delaunay/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

D-dimensional Delaunay triangulations in [Rust], inspired by [CGAL].

## Introduction

This library implements d-dimensional Delaunay triangulations in [Rust]. It is
inspired by [CGAL], which is a [C++] library for computational geometry;
and [Spade], a [Rust] library implementing 2D [Delaunay triangulations],
[Constrained Delaunay triangulations], and [Voronoi diagrams]. The eventual
goal of this library is to provide a lightweight alternative to [CGAL] for
the [Rust] ecosystem.

## Features

- [x]  d-dimensional [Delaunay triangulations]
- [x]  Tested for 3-, 4-, and 5-dimensional triangulations
- [x]  Arbitrary data types associated with vertices and cells
- [x]  Serialization/Deserialization of all data structures to/from [JSON]

At some point I may merge into another library, such as [Spade] or [delaunay],
but for now I am developing this to use in my [research] without trying to
figure out how to mesh with other libraries and coding conventions, and with
the minimum number of [traits] to do generic computational geometry.

## Benchmarks

The library includes comprehensive performance benchmarks for circumsphere containment algorithms. Key findings:

- **insphere_lifted**: Fastest method (~50% better than standard)
- **insphere**: Best balance of performance and numerical stability
- **insphere_distance**: Slowest but most transparent for educational use

Run benchmarks with:

```bash
cargo bench --bench circumsphere_containment
```

See [benches/README.md](benches/README.md) for detailed performance results and analysis.

## References

The library's geometric predicates and algorithms are based on established computational geometry literature:

### Circumcenter and Circumradius Calculations

- Lévy, Bruno, and Yang Liu. "Lp Centroidal Voronoi Tessellation and Its Applications." *ACM Transactions on Graphics* 29, no. 4 (July 26, 2010):
  119:1-119:11. DOI: [10.1145/1778765.1778856](https://doi.org/10.1145/1778765.1778856)

### Robust Geometric Predicates

- Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates." *Discrete & Computational Geometry* 18,
  no. 3 (1997): 305-363. DOI: [10.1007/PL00009321](https://doi.org/10.1007/PL00009321)
- Shewchuk, J. R. "Robust Adaptive Floating-Point Geometric Predicates." *Proceedings of the Twelfth Annual Symposium on Computational Geometry* (1996): 141-150.

### Lifted Paraboloid Method

- Preparata, Franco P., and Michael Ian Shamos. "Computational Geometry: An Introduction." Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
- Edelsbrunner, Herbert. "Algorithms in Combinatorial Geometry." EATCS Monographs on Theoretical Computer Science. Berlin: Springer-Verlag, 1987.

[Rust]: https://rust-lang.org
[CGAL]: https://www.cgal.org/
[C++]: https://isocpp.org
[Spade]: https://github.com/Stoeoef/spade
[delaunay]: https://crates.io/crates/delaunay
[JSON]: https://www.json.org/json-en.html
[Delaunay triangulations]: https://en.wikipedia.org/wiki/Delaunay_triangulation
[Constrained Delaunay triangulations]: https://en.wikipedia.org/wiki/Constrained_Delaunay_triangulation
[Voronoi diagrams]: https://en.wikipedia.org/wiki/Voronoi_diagram
[research]: https://github.com/acgetchell/cdt-rs
[traits]: https://doc.rust-lang.org/book/ch10-02-traits.html

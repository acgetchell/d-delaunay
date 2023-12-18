# d-delaunay

[![CI](https://github.com/acgetchell/d-delaunay/actions/workflows/ci.yml/badge.svg)](https://github.com/acgetchell/d-delaunay/actions/workflows/ci.yml)
[![rust-clippy analyze](https://github.com/acgetchell/d-delaunay/actions/workflows/rust-clippy.yml/badge.svg)](https://github.com/acgetchell/d-delaunay/actions/workflows/rust-clippy.yml)
[![codecov](https://codecov.io/gh/acgetchell/d-delaunay/graph/badge.svg?token=WT7qZGT9bO)](https://codecov.io/gh/acgetchell/d-delaunay)
[![Security audit](https://github.com/acgetchell/d-delaunay/actions/workflows/audit-check.yml/badge.svg)](https://github.com/acgetchell/d-delaunay/actions/workflows/audit-check.yml)

D-dimensional Delaunay triangulations in [Rust], inspired by [CGAL].

## Introduction

This library implements d-dimensional Delaunay triangulations in [Rust]. It is inspired by the [CGAL] library, which is a C++ library for computational geometry; and [Spade], a Rust library
implementing 2D Delaunay triangulations, Constrained Delaunay triangulations, and Voronoi diagrams. The eventual goal of this library is to provide a lightweight alternative to [CGAL] for the [Rust]
ecosystem.

At some point I may merge it into another library, such as [Spade], or [delaunay], but for now I am developing this library without trying to figure out how to fit it into other coding styles and standards.

[Rust]: https://rust-lang.org
[CGAL]: https://www.cgal.org/
[Spade]: https://github.com/Stoeoef/spade
[delaunay]: https://crates.io/crates/delaunay

# PID-Star

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3871800.svg)](https://doi.org/10.5281/zenodo.3871800)

This program computes for a given graph *G* a tree *T* of minimum depth such that *G* is contained in the closure of *T,* that is, an optimal [treedepth decomposition](https://en.wikipedia.org/wiki/Tree-depth) of *G.*

The tool was developed by *Max Bannach, Sebastian Berndt, Martin Schuster* and *Marcel Wienöbst* as submission for the exact track of [PACE 2020](https://pacechallenge.org/2020/). As such, the input format and I/O behavior is as specified by the PACE.

# Algorithm
The core algorithm is the *positive-instance driven* (PID) dynamic programing framework developed for a general form of graph searching [1]. However, the algorithm presented in [1] works in two phases: it first computes the set of positive instances and, then, solves distance queries on this set. By replacing the reverse breadth-first search by a reverse version of Dijkstra's algorithm we are able to compute these distances on-the-fly. This means the algorithm maintains at any point in time a locally optimal treedepth decomposition for every positive instance (and not just one that can be extended to a global optimal one). If this reverse search has multiple candidates of equal quality, we use a heuristic to determine which candidate is most likely to be good for an optimal global solution and pick this one first, leading to some sort of reverse A* – therefore the name.

1. Max Bannach and Sebastian Berndt: *[Positive-Instance Driven Dynamic Programming for Graph Searching](https://arxiv.org/abs/1905.01134).* (WADS 2019)

# Dependencies
The following open source [crates](https://crates.io) are used. They are automatically downloaded and compiled when the tool is build using *Cargo.*
- [bit-set](https://crates.io/crates/bit-set)
- [bimap](https://crates.io/crates/bimap)
- [rand](https://crates.io/crates/rand)

# Build
PID-Star is implemented in [Rust](https://www.rust-lang.org) and can simply be build using [Cargo](https://doc.rust-lang.org/cargo/getting-started/installation.html):

```
cargo build --release
```

# Run
After the build is completed, the tool can either be executed directly via

```
./target/release/pid-star < <mygraph.gr>
```

or by using Cargo

```
cargo run --release < <mygraph.gr>
```

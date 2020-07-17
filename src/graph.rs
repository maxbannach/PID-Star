use std::io::BufRead;
use std::error::Error;
use bit_set::BitSet;
use bimap::BiMap;

pub struct Graph {
    pub n:              usize,                       // universe size
    pub neighbors:      Vec<Vec<usize>>,             // neighbors of vertex v
    pub matrix:         Vec<BitSet>,                 // neighbors of vertex v as bitset
    pub parent:         Vec<Option<usize>>,          // representation of treedepth decomposition
    pub descendants:    Vec<Vec<usize>>,             // descendants of v
    pub not_ancestors:  Vec<Vec<usize>>,             // vertices of which v is not an ancestor
}                                                  

impl Graph {

    /// Creates a new graph with vertex set *V={0,1,...,n-1}* and without any edges.
    ///
    fn new(n: usize) -> Graph {
        Graph {
            n:              n,
            neighbors:      vec![Vec::new(); n],
            matrix:         vec![BitSet::new(); n],
            parent:         vec![None; n],
            descendants:    vec![Vec::new(); n],
            not_ancestors:  vec![Vec::new(); n],
        }
    }

    /// Read a graph structure from stdin by parsing a PACE 2020 formated graph.
    ///
    /// *Errors:*
    /// - May fail if the p-line is wrongly formated, an edge appears before the p-line, or there
    ///   is an uncommend line that does not encode an edge.
    ///
    pub fn new_from_stdin() -> Result<Graph, Box<dyn Error>> {
        let mut g: Option<Graph> = None;
        for line in std::io::stdin().lock().lines() {
            let line = line?;
            let ll: Vec<&str> = line.split(" ").collect();
            match ll[0] {
                "c" => {} // skip comments
                "p" => {  // parse header
                    let n = ll[2].parse::<usize>()?;
                    g = Some(Graph::new(n));
                },
                _ => { // parse edges
                    match g {
                        None => return Err(From::from("c Found edge before p-line. Abort!")),
                        Some(ref mut g) => {
                            g.add_edge(ll[0].parse::<usize>()? - 1, ll[1].parse::<usize>()? - 1);
                        }
                    }
                }
            }
        }
        match g {
            Some(g) => Ok(g),
            None    => Err(From::from("c Failed to parse a graph! Maybe the input was empty?"))
        }
    }

    /// Creates a copy of the given graph.
    ///
    pub fn clone(g: &Graph) -> Graph {
        Graph {
            n:              g.n,
            neighbors:      g.neighbors.clone(),
            matrix:         g.matrix.clone(),
            parent:         g.parent.clone(),
            descendants:    g.descendants.clone(),
            not_ancestors:  g.not_ancestors.clone(),
        }
    }
    
    /// Adds the undirected edge *{u,v}* to the graph.
    ///
    /// *Assumes:*
    /// - that the edge is not already in the graph
    ///
    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.neighbors[u].push(v);
        self.neighbors[v].push(u);
        self.matrix[u].insert(v);
        self.matrix[v].insert(u);
    }

    /// Computes the neighbors of the given vertex within the given subgraph.
    ///
    pub fn neighbors_in(&self, subgraph: &BitSet, v: usize) -> BitSet {
        let mut neighborhood = BitSet::new();
        for w in self.neighbors[v].iter() {
            if subgraph.contains(*w) { neighborhood.insert(*w); }
        }
        return neighborhood;
    }
    
    /// Compute the closed border (i.e., closed neighborhood) of the given subgraph.
    ///
    pub fn closed_border_of(&self, subgraph: &BitSet) -> BitSet {
        let mut result = subgraph.clone();
        for v in subgraph.iter() {
            result.union_with(&self.matrix[v]);
        }
        return result;
    }
    
    /// Compute the border (i.e., neighborhood) of the given subgraph.
    ///
    pub fn border_of(&self, subgraph: &BitSet) -> BitSet {
        let mut result = self.closed_border_of(subgraph);
        result.difference_with(&subgraph);
        return result;
    }

    /// Returns the connected components within the given subgraph when separated with the given
    /// separator.
    ///
    pub fn components_in(&self, subgraph: &BitSet, separator: &BitSet) -> Vec<BitSet> {
        let mut components = Vec::new();
        let mut stack      = Vec::new();
        let mut visited: BitSet = (0..self.n)
            .filter(|v| !subgraph.contains(*v) || separator.contains(*v) )
            .collect();        
        for s in subgraph.iter() {
            if visited.contains(s) { continue; }
            let mut component = BitSet::with_capacity(self.n);
            component.insert(s);
            visited.insert(s);
            stack.push(s);
            while let Some(v) = stack.pop() {
                for &w in self.neighbors[v].iter() {
                    if visited.contains(w) { continue; }
                    component.insert(w);
                    visited.insert(w);
                    stack.push(w);
                }
            }
            components.push(component);
        }        
        return components;
    }

    /// Computes for each vertex *v* a list of vertices *w* such that
    /// *v* can be an ancestor of all *w* in an optimal tree-depth decomposition.
    ///
    /// *Note:*
    /// - this function will clear any previous data stored in the descendants array
    ///
    pub fn compute_descendants(&mut self) {
        for v in 0..self.n {
            self.descendants[v].clear();
            self.not_ancestors[v].clear();
            for w in 0..self.n {
                let mask: BitSet = self.matrix[w].difference(&self.matrix[v]).filter(|w| *w != v).collect();
                if mask.is_empty() && (self.neighbors[v].len() > self.neighbors[w].len() || v < w) {
                    if self.matrix[v].contains(w) {
                        self.descendants[v].push(w);
                    } else {
                        self.not_ancestors[v].push(w);
                    }
                }
            }           
        }        
    }

    /// Checks whether the given subgraph contains a decendant of *v*.
    ///
    pub fn contains_descendant(&self, v: usize, subgraph: &BitSet) -> bool {
        self.descendants[v].iter().any(|w| subgraph.contains(*w))
    }

    /// Checks whether the given subgraph contains all decendants of *v*.
    ///
    pub fn contains_all_descendants(&self, v: usize, subgraph: &BitSet) -> bool {
        self.descendants[v].iter().all(|w| subgraph.contains(*w))
    }

    /// Computes a set of vertices that are not allowed to be an ancestors of
    /// the given subgraph.
    ///
    pub fn forbidden_ancestors(&self, subgraph: &BitSet) -> BitSet {
        let mut forbidden = BitSet::with_capacity(self.n);
        for v in subgraph.iter() {
            for &w in self.not_ancestors[v].iter() {
                forbidden.insert(w);
            }
        }
        return forbidden;
    }
    
    /// Shrinks the graph to the given subgraph. This will not change this strcuture, but rather
    /// generate a new graph that is isomorphic to this structure on the given subgraph.
    ///
    /// The new vertex set will be {0,..,|subgraph|-1} and the function returns a map that
    /// indicates which original vertex is represented by a vertex of the new graph.
    ///
    /// The bimap maps vertices in the new graph to vertices in the old graph, i.e.,
    /// left = new graph and right = old graph.
    ///
    pub fn shrink_to(&self, subgraph: &BitSet) -> (Graph, BiMap<usize,usize>) {
        let mut h   = Graph::new(subgraph.len());
        let mut iso = BiMap::new();

        let mut idx = 0;
        for v in subgraph.iter() {
            iso.insert(idx, v);
            idx = idx+1;
        }
        for v in subgraph.iter() {
            for &w in self.neighbors[v].iter() {
                if w >= v || !subgraph.contains(w) { continue; } // edge not induced
                h.add_edge(*iso.get_by_right(&v).unwrap(), *iso.get_by_right(&w).unwrap());
            }
        }
        
        return (h, iso);
    }    
}

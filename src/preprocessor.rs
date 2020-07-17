use crate::graph::Graph;
use bit_set::BitSet;
use bimap::BiMap;

pub struct Preprocessor{}
impl Preprocessor {

    /// Removes all leaves *v* of *g* as long as the neighbor of *v* is adjacent to another leaf.
    /// Returns the shrunken graph *h* (a copy, *g* itself is not directly modified) and an isomorphism between
    /// *h* and *g*.
    ///
    /// The parent vector of *g* is updated in order to store the parents of the removed leaves.
    ///
    pub fn remove_leaves(g: &mut Graph) -> (Graph, BiMap<usize,usize>) {
        let mut subgraph = BitSet::with_capacity(g.n);
        let mut covered  = BitSet::with_capacity(g.n);
        for v in 0..g.n {
            subgraph.insert(v);
            if g.neighbors[v].len() != 1 { continue; }
            let &w = g.neighbors[v].iter().next().unwrap();
            if covered.contains(w) {
                g.parent[v] = Some(w);
                subgraph.remove(v);
            } else {
                covered.insert(w);
            }
        }
        return g.shrink_to(&subgraph);
    }

    /// Translates a tree-depth decomposition of *h* to one of *g* using the given isomorphism.
    ///
    /// *Assumes:*
    /// - *h* was solved, i.e., the parent vector is filled properly
    ///
    pub fn reinsert_leaves(g: &mut Graph, h: &Graph, iso: &BiMap<usize,usize>) {
        // merge reduced graph with original graph
        for v in 0..h.n {
            g.parent[*iso.get_by_left(&v).unwrap()] = match h.parent[v] {
                None    => None,
                Some(p) => Some(*iso.get_by_left(&p).unwrap()),
            }            
        }

        // If the covering leaf of a vertex was eliminated before that vertex it covers, we have
        // to swap the two – otherwise adding back removed leaves may increase the treedepth.
        'search: for v in 0..g.n {
            if iso.contains_right(&v) { continue; }
            if let Some(p) = g.parent[v] {
                for w in 0..h.n {
                    let w = *iso.get_by_left(&w).unwrap();
                    if g.parent[w] == Some(p) {
                        continue 'search;
                    }
                }

                let w = g.neighbors[p].iter().filter(|&x| g.neighbors[*x].len() == 1 && iso.contains_right(&x)).nth(0);
                if let Some(&w) = w {
                    for x in 0..g.n {
                        if g.parent[x] == Some(w) { g.parent[x] = Some(p); }
                    }
                    g.parent[p] = g.parent[w];
                    g.parent[w] = Some(p);
                }
            }
        }
    }

    /// Creates a copy of *g* and computes the k-edge-improved graph of it.
    /// Also returns a flag that indicates whether some improvement was achieved at all.
    ///
    pub fn improve_graph(g: &Graph, k: usize) -> (Graph, bool) {
        let mut h = Graph::clone(g);
        let mut stack = Vec::with_capacity(g.n);
        for v in 0..h.n { stack.push(v); }

        let mut modified = false;
        while let Some(v) = stack.pop() {
            for w in 0..h.n {
                if v >= w || h.matrix[v].contains(w) { continue; }
                if h.matrix[v].intersection(&h.matrix[w]).into_iter().count() >= k {                    
                    h.add_edge(v, w);
                    stack.push(v);
                    stack.push(w);
                    modified = true;
                }
            }
        }
        
        return (h, modified);
    }

    /// Creates a copy of *g* without simplicial vertices that have only neighbors of degree > k.
    /// Also returns a flag that indicates whether at least one vertex was removed.
    ///   
    pub fn remove_simplicials(g: &Graph, k: usize) -> (Graph, BiMap<usize,usize>, bool) {
        let mut subgraph = BitSet::with_capacity(g.n);
        let mut stack    = Vec::with_capacity(g.n);
        for v in 0..g.n {
            subgraph.insert(v);
            stack.push(v);
        }

        let mut modified = false;
        'search: while let Some(v) = stack.pop() {
            if !subgraph.contains(v) { continue; }
            for &x in g.neighbors[v].iter() {
                if !subgraph.contains(x) { continue; }
                if g.neighbors[x].iter().filter(|&a| subgraph.contains(*a)).count() <= k { continue 'search; }
                for &y in g.neighbors[v].iter() {
                    if x >= y || !subgraph.contains(y) { continue; }
                    if !g.matrix[x].contains(y) { continue 'search; }                    
                }
            }
            modified = true;
            subgraph.remove(v);
            for &x in g.neighbors[v].iter() { stack.push(x); }
        }

        let (h, iso) = g.shrink_to(&subgraph);
        return (h, iso, modified);
    }

    /// Translates a tree-depth decomposition of *h* to one of *g* using the given isomorphism.
    ///
    /// *Assumes:*
    /// - *h* was solved, i.e., the parent vector is filled properly
    ///
    pub fn reinsert_simplicials(g: &mut Graph, h: &Graph, iso: &BiMap<usize,usize>) {
        // merge reduced graph with original graph
        for v in 0..h.n {
            g.parent[*iso.get_by_left(&v).unwrap()] = match h.parent[v] {
                None    => None,
                Some(p) => Some(*iso.get_by_left(&p).unwrap()),
            }            
        }

        // attach the simplicial vertices at the deepest neighbor
        for v in 0..g.n {
            if iso.contains_right(&v) { continue; }
            let (mut ancestor, mut depth) = (0,0);
            for &w in g.neighbors[v].iter() {
                let mut crawler = w;
                let mut counter = 1;
                while let Some(p) = g.parent[crawler] {
                    crawler  = p;
                    counter += 1;
                }
                if counter > depth {
                    depth    = counter;
                    ancestor = w;
                }
            }
            g.parent[v] = Some(ancestor);
        }        
    }   
}

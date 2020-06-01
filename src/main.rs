use std::io::BufRead;
use std::error::Error;
use bit_set::BitSet;
use std::collections::{ HashMap, BinaryHeap };
use std::rc::Rc;
use std::cmp::Ordering;
use bimap::BiMap;

/// Simple macro to compute integer log2 – or, more precisely, max( floor(log2(x))+1, 0 ).
/// Only sensible defined for usize.
///
#[macro_export]
macro_rules! log2 {
    ($x:expr) => ( (0usize.leading_zeros() - ($x as usize).leading_zeros()) as usize )
}

/// Number of iterations performed by the heuristic.
const HEURISTIC_ITERATIONS: usize = 1000;

/**********************************
 * Graph implementation
 **********************************/

struct Graph {
    n:            usize,                       // universe size
    neighbors:    Vec<Vec<usize>>,             // neighbors of vertex v
    matrix:       Vec<BitSet>,                 // neighbors of vertex v as bitset
    parent:       Vec<Option<usize>>,          // representation of treedepth decomposition
    descendants:  Vec<Vec<usize>>,             // stores for each vertex v a list of vertices
}                                              // for which v can be an ancestor

impl Graph {

    /// Creates a new graph with vertex set *V={0,1,...,n-1}* and without any edges.
    ///
    fn new(n: usize) -> Graph {
        Graph {
            n:            n,
            neighbors:    vec![Vec::new(); n],
            matrix:       vec![BitSet::new(); n],
            parent:       vec![None; n],
            descendants:  vec![Vec::new(); n],
        }
    }

    /// Creates a copy of the given graph.
    ///
    fn clone(g: &Graph) -> Graph {
        Graph {
            n:            g.n,
            neighbors:    g.neighbors.clone(),
            matrix:       g.matrix.clone(),
            parent:       g.parent.clone(),
            descendants:  g.descendants.clone(),
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
    fn closed_border_of(&self, subgraph: &BitSet) -> BitSet {
        let mut result = subgraph.clone();
        for v in subgraph.iter() {
            result.union_with(&self.matrix[v]);
        }
        return result;
    }
    
    /// Compute the border (i.e., neighborhood) of the given subgraph.
    ///
    fn border_of(&self, subgraph: &BitSet) -> BitSet {
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
        let mut visited    = BitSet::with_capacity(self.n);
        for v in 0..self.n {
            if !subgraph.contains(v) || separator.contains(v) { visited.insert(v); }            
        }
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

    /// Checks whether the given subgraph is an induced path.
    ///
    pub fn is_path(&self, subgraph: &BitSet) -> bool {
        for v in subgraph.iter() {
            if self.neighbors[v].len() > 2 {
                return false;
            }
        }
        return true;
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
            for &w in self.neighbors[v].iter() {
                let mut mask = self.matrix[w].clone();
                mask.difference_with(&self.matrix[v]);
                mask.remove(v);
                if mask.is_empty() && (self.neighbors[v].len() > self.neighbors[w].len() || v < w) {
                    self.descendants[v].push(w);
                }
            }
        }        
    }

    /// Checks whether the given subgraph contains a decendant of *v*.
    ///
    pub fn contains_decendant(&self, v: usize, subgraph: &BitSet) -> bool {
        for &w in self.descendants[v].iter() {
            if subgraph.contains(w) { return true; }
        }
        return false;
    }

    /// Checks whether the given subgraph contains all decendants of *v*.
    ///
    pub fn contains_all_decendants(&self, v: usize, subgraph: &BitSet) -> bool {
        for &w in self.descendants[v].iter() {
            if !subgraph.contains(w) { return false; }
        }
        return true;
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

/**********************************
 * Preprocessor
 **********************************/

struct Preprocessor{}
impl Preprocessor {

    /// Removes all leaves *v* of *g* as long as the neighbor of *v* is adjacent to another leaf.
    /// Returns the shrunken graph *h* (a copy, *g* itself is not directly modified) and an isomorphism between
    /// *h* and *g*.
    ///
    /// The parent vector of *g* is updated in order to store the parents of the removed leaves.
    ///
    fn remove_leaves(g: &mut Graph) -> (Graph, BiMap<usize,usize>) {
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
    fn reinsert_leaves(g: &mut Graph, h: &Graph, iso: &BiMap<usize,usize>) {
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
    fn improve_graph(g: &Graph, k: usize) -> (Graph, bool) {
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
    fn remove_simplicials(g: &Graph, k: usize) -> (Graph, BiMap<usize,usize>, bool) {
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
    fn reinsert_simplicials(g: &mut Graph, h: &Graph, iso: &BiMap<usize,usize>) {
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

/**********************************
 * Block implementation
 **********************************/

#[derive(Clone, Eq, PartialEq, Debug)]
struct Block {
    subgraph: BitSet,                           // vertices in the block, this is a connected graph
    depth:    usize,                            // the treedepth of the block (not necessarily optimal)
    center:   usize,                            // center vertex of the subgraph     
}

/// We order Blocks first by depth, then by size, and finally lexicographically. 
///
impl Ord for Block {
    fn cmp(&self, other: &Block) -> Ordering {
        other.depth.cmp(&self.depth)
            .then_with(|| self.subgraph.len().cmp(&other.subgraph.len()))
            .then_with(|| self.subgraph.cmp(&other.subgraph))
    }
}

/// As the normal ordering, but without the lexicographical tie break.
///
impl PartialOrd for Block {
    fn partial_cmp(&self, other: &Block) -> Option<Ordering> {
        Some(other.depth.cmp(&self.depth)
            .then_with(|| self.subgraph.len().cmp(&other.subgraph.len())))
    }
}

/**********************************
 * Core Solver
 *********************************/

struct Solver {
    positive: Vec<Vec<Rc<BitSet>>>,               // positive subgraphs that are adjacent to a given vertex
    memory:   HashMap<BitSet, (usize, usize)>,    // memoization for best known treedepth and center vertex
    queue:    BinaryHeap<Block>,                  // priority queue of the algorithm
    halt:     bool,                               // found solution
    k:        usize,                              // target treedepth
}

impl Solver {

    fn solve(&mut self, g: &mut Graph) -> bool {
        
        // compute win configurations
        for v in 0..g.n {
            if g.neighbors[v].len() >= self.k { continue; }
            if g.descendants[v].len() > 0 { continue; }
            let mut subgraph = BitSet::new();
            subgraph.insert(v);            
            self.queue.push(Block{ subgraph: subgraph, depth: 1, center: v});
        }

        // compute winning region
        while let Some(mut block) = self.queue.pop() {
            if self.halt { break; }
            
            // memoization
            if self.memory.contains_key(&block.subgraph) { continue; }
            self.memory.insert(block.subgraph.clone(), (block.depth, block.center));

            // get the neighborhood of the block
            let border = g.border_of(&block.subgraph);

            // Domination-Rule
            if self.is_dominated(g, &block.subgraph, &border) { continue; }

            // Fast-Contamination Rule
            if let Some(v) = self.get_contaminatable_vertex(g, &block.subgraph, &border) {
                block.subgraph.insert(v);
                self.offer(g, block.subgraph, v, block.depth+1);
                continue;
            }
            
            // Attachment-Rule
            if self.has_attachment(g, &mut block.subgraph, &border, block.depth) { continue; }

            // compute new configurations
            for v in border.iter() {
                
                // Ancestor-Rule
                if g.contains_decendant(v, &border) { continue; }
                
                // Fast-Glue Rule
                let mut start = block.subgraph.clone();
                for other in self.positive[v].iter() {
                    if !start.is_disjoint(&other) { continue; }
                    if g.border_of(&other).is_subset(&border) {
                        start.union_with(&other);
                    }
                }

                // sort positive instances such that we do not compare with instances that can't work
                let     mask = g.closed_border_of(&start);
                let mut len  = self.positive[v].len();
                let mut i    = 0;
                while i < len {
                    if !mask.is_disjoint(&self.positive[v][i]) {
                        self.positive[v].swap(i, len - 1);
                        len -= 1;
                    } else {
                        i += 1;
                    }
                }
                
                // reverse reveal-move
                let mut stack = Vec::new();
                stack.push( (start, block.depth, -1) );                
                while let Some( (mut subgraph, depth, lex) ) = stack.pop() {
                    let mask = g.closed_border_of(&subgraph);
                    for (i,other) in self.positive[v].iter().enumerate() {
                        if i >= len { break; }

                        // we only glue with lexicographical larger components
                        let index = other.iter().next().unwrap() as isize;
                        if index < lex { continue; }

                        // we can only glue disjoint configurations
                        if !mask.is_disjoint(&other) { continue; }   

                        // actually glue them together
                        let mut next = subgraph.clone();             
                        next.union_with(&other);

                        // pruning while gluing
                        if g.border_of(&next).len() + depth > self.k {
                            continue;
                        }

                        // add to stack for further gluing
                        stack.push((next, depth, std::cmp::max(lex,index)));
                    }
                    
                    // done gluing this block -> offer it
                    subgraph.insert(v);

                    // Ancestor-Rule
                    if !g.contains_all_decendants(v, &subgraph) { continue; }

                    // Pruning
                    if lex < 0 && self.can_skip_block(g, &subgraph, v, depth+1) { continue; }
                    
                    self.offer(g, subgraph, v, depth+1);
                    if self.halt { return true; }
                }
            }
            
            // the block is optimal, so we can store it as positive instance
            let subgraph = Rc::new(block.subgraph);
            for v in border.iter() {
                self.purge_list(g, v, &subgraph, &border);
                self.positive[v].push(Rc::clone(&subgraph));                
            }
        }
        
        // done
        return self.halt;
    }

    /// Offer a new block to the queue. This checks whether we found a solution,
    /// we can prune the block, or just inserts it.
    ///
    fn offer(&mut self, g: &mut Graph, mut subgraph: BitSet, center: usize, depth: usize) {
        
        // final configurations
        let border = g.border_of(&subgraph);
        if depth + border.len() > self.k { return; }               // negative block
        if g.n - subgraph.len() <= self.k - depth {                // solution found
            self.halt = true;

            let mut mask = BitSet::new();                          // the complete graph
            for v in 0..g.n { mask.insert(v); }
            
            let mut diff = mask.clone();                           // we choose arbitrary centers
            diff.difference_with(&subgraph);                       // till we reach the subgraph
            for v in diff.iter() {
                self.memory.insert(mask.clone(), (0,v));
                mask.remove(v);
            }            
            self.memory.insert(mask.clone(), (0,center));
            
            // done
            return;
        }

        // check if we can discard the block using the chain rule
        if !g.is_path(&subgraph) && self.chain_rule(&g, &mut subgraph, &border, center, depth) { return; }
        
        // done -> just insert new block into queue
        self.queue.push(Block{ subgraph: subgraph, center: center, depth: depth });
    }

    /// Try to find a vertex *v* in *N(C)* such that *N(v)* is contained in *N[C]*.
    /// Such a vertex is called *covered* and can be contaminated right away.
    ///
    fn get_contaminatable_vertex(&self, g: &Graph, subgraph: &BitSet, border: &BitSet) -> Option<usize> {
        'search: for v in border.iter() {
            for &w in g.neighbors[v].iter() {
                if !(subgraph.contains(w) || border.contains(w)) { continue 'search; }                
            }
            return Some(v);
        }
        return None;
    }

    /// Check if the given block is dominated by a block that we have already discovered.
    /// If this is the case, the block can be skipped and does not have to be added to the
    /// lists.
    ///
    fn is_dominated(&self, g: &Graph, subgraph: &BitSet, border: &BitSet) -> bool {
        for v in border.iter() {
            for other in self.positive[v].iter() {
                if other.is_superset(&subgraph)
                    && g.border_of(&other).is_subset(&border) {
                        return true;
                }
            }
        }
        return false;
    }

    /// Purges the list of positive instances attached to *v* by removing
    /// blocks that are dominated by the given block.
    ///
    fn purge_list(&mut self, g: &Graph, v: usize, subgraph: &BitSet, border: &BitSet) {       
        let mut len    = self.positive[v].len();
        let mut i      = 0;
        while i < len {
            let other = &self.positive[v][i];

            // check if subgraph is better than the other
            if subgraph.is_superset(&other) && border.is_subset(&g.border_of(other)) {
                self.positive[v].swap(i, len-1);
                len -= 1;                
            }
            
            i += 1;
        }
        self.positive[v].truncate(len);
    }

    /// Check whether it is safe to skip the given block.
    ///
    fn can_skip_block(&self, g: &Graph, subgraph: &BitSet, center: usize, depth: usize) -> bool {        
        let neighbors = g.neighbors_in(&subgraph, center);

        // Local Leaf-Rule
        if neighbors.len() == 1 {
            let x = neighbors.iter().next().unwrap();
            if g.neighbors_in(&subgraph, x).len() > 1 { return true; }
        }

        // Local Simplical-Rule
        if  neighbors.len() == 2 {
            let mut iter = neighbors.iter();
            let x = iter.next().unwrap();
            let y = iter.next().unwrap();
            if  g.matrix[x].contains(y)
                && g.neighbors_in(&subgraph, x).len() >= depth
                && g.neighbors_in(&subgraph, y).len() >= depth {
                    return true;
                }
        }
        
        // Local Ansestor-Rule
        for w in subgraph.iter() {
            if w < center
                && !g.descendants[center].contains(&w) 
                &&  g.neighbors_in(&subgraph, w).is_superset(&neighbors) {
                return true;
            }
        }
        return false;
    }

    /// Checks whether the given block can be pruned by the Chain-Rule. A chain is an induced path, 
    /// i.e., a subgraph of degree-2 vertices. Assume such a chain is attached to a block with *in*
    /// being the set of vertices of the chain within the block and *m* the first vertex of the chain
    /// outsize the block. 
    ///
    /// Clearly, ceil(log(in+1)) depth will be generated by the part of the chain within the block.
    /// We check whether this value increases if we add *m* to the block, i.e., if we increase the
    /// path fragment by one vertex. If this is not the case, the current block is suboptimal as we
    /// could take *m* as well and, hence, we can prune it.
    ///
    fn chain_rule(&self, g: &Graph, subgraph: &mut BitSet, border: &BitSet, center: usize, depth: usize) -> bool {
        for s in border.iter() {
            if g.neighbors[s].len() != 2 { continue; }

            // use memorization, maybe we have solved it for a larger path already
            subgraph.insert(s);
            if let Some((d,_)) = self.memory.get(&subgraph) {
                if d <= &depth { return true; }
            }
            subgraph.remove(s);
            
            // compute the chain
            let (mut visited, mut stack, mut interior, mut interior_end) = (BitSet::new(), Vec::new(), Vec::new(), 0);
            visited.insert(s);
            stack.push(s);
            while let Some(v) = stack.pop() {
                for &w in g.neighbors[v].iter() {
                    if subgraph.contains(w) {
                        interior.push(w);
                        if g.neighbors[w].len() > 2 { interior_end += 1; }
                    }
                    if visited.contains(w) || g.neighbors[w].len() != 2 { continue; }                    
                    visited.insert(w);
                    stack.push(w);
                }
            }
            if interior_end > 1 { continue; }           
            
            // check if we can safly add the border vertex to the path
            if let Some(r) = self.get_first_eliminated_vertex(g, subgraph, Some(center), &visited) {
                let q = interior.len() - 1;
                let i = q - interior.iter().position(|&x| x == r).unwrap();
                if log2!(q-i+1) == log2!(q-i+2) { return true; }
            }
        }
        return false;
    }

    /// Checks whether the given block as an attachment, that is, a vertex that can
    /// be added to the block without increasing its treedepth nor increasings its border.
    ///
    fn has_attachment(&self, g: &Graph, subgraph: &mut BitSet, border: &BitSet, depth: usize) -> bool {
        for v in border.iter() {
            subgraph.insert(v);
            let w = g.neighbors[v].iter().filter(|&x| !subgraph.contains(*x) && !border.contains(*x)).nth(0);
            if w.is_some() && border.len() >= g.border_of(&subgraph).len() {
                let &w = w.unwrap();
                if w > v { subgraph.remove(v); continue; }
                if let Some((d,_)) = self.memory.get(&subgraph) {
                    if *d <= depth {
                        subgraph.remove(v);
                        return true;
                    }
                }
            }
            subgraph.remove(v);
        }                      
        return false;
    }
    
    /// Rest the solver and prepare it for the next k.
    ///
    fn next(&mut self) {
        self.k = self.k + 1;
        for vec in self.positive.iter_mut() { vec.clear(); }
        self.memory.clear();
        self.queue.clear();
        self.halt = false;
    }

    /// Returns the first vertex that gets eliminated from the given set.
    ///
    /// *Assums*
    /// - that the set is connected
    ///
    fn get_first_eliminated_vertex(&self, g: &Graph, subgraph: &mut BitSet, center: Option<usize>, set: &BitSet) -> Option<usize> {
        let center = match center {
            Some(c) => c,
            None    => match self.memory.get(&subgraph) {
                Some((_,c)) => *c,
                None        => return None
            }
        };
        if set.contains(center) { return Some(center); }
        let mut result = None;
        subgraph.remove(center);
        for mut component in g.components_in(&subgraph, &BitSet::new()) {
            result = self.get_first_eliminated_vertex(g, &mut component, None, set);
            if result.is_some() { break; }
        }
        subgraph.insert(center);
        return result;
    }

    /// Computes the parent field in the given subgraph in the graph.
    /// This is done by extracting the corresponding center color.
    ///
    fn extract_decomposition(&self, g: &mut Graph, subgraph: Option<BitSet>, parent: Option<usize>) {
        let mut subgraph = match subgraph {
            Some(c) => c,
            None => {
                let mut all = BitSet::with_capacity(g.n);
                for v in 0..g.n { all.insert(v); }
                all
            }
        };        
        if let Some(&(_depth, center)) = self.memory.get(&subgraph) {
            g.parent[center] = parent;
            subgraph.remove(center);
            for component in g.components_in(&subgraph, &BitSet::new()) {
                self.extract_decomposition(g, Some(component), Some(center));
            }
        } else {
            eprintln!("c While printing the decomposition, I found a subgraph that has no center. Abort!");
        }
    }

}

/**********************************
 * Utility functions
 *********************************/

fn main() {

    // read input graph
    let mut g = match read_graph() {
        Ok(g)  => g,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    // do some simple preprocessing
    let (mut h, iso) = Preprocessor::remove_leaves(&mut g);
    h.compute_descendants();

    // compute an upper bound, may allow us to skip the last iteration
    let (ub, ub_parents) = compute_upper_bound(&h);
    eprintln!("c td <= {}", ub);

    // find optimal treedepth decomposition
    let mut solver = Solver{ positive: vec![Vec::new(); h.n], memory: HashMap::new(), queue: BinaryHeap::new(), halt: false, k: 0 };
    let mut do_preprocessing = true;
    loop {
        solver.next();
        if solver.k == ub { // we are lucky and done!
            h.parent = ub_parents;
            break;
        }
        if do_preprocessing {
            do_preprocessing = false;
            let (p, improved)         = Preprocessor::improve_graph(&mut h, solver.k);
            let (mut p, iso, reduced) = Preprocessor::remove_simplicials(&p, solver.k);
            do_preprocessing |= improved | reduced;
            p.compute_descendants();
            if solver.solve(&mut p) {
                solver.extract_decomposition(&mut p, None, None);
                Preprocessor::reinsert_simplicials(&mut h, &p, &iso);
                break;
            }
        } else if solver.solve(&mut h) {
            solver.extract_decomposition(&mut h, None, None);
            break;
        }
        eprintln!("c td >  {}", solver.k);
    }
    eprintln!("c td =  {}", solver.k);

    // undo removal of leaves
    Preprocessor::reinsert_leaves(&mut g, &h, &iso);    

    // output decomposition
    println!("{}", solver.k);
    for v in 0..g.n {
        match g.parent[v] {
            None    => println!("0"),
            Some(p) => println!("{}", p+1),
        }
    }
    
}

/// Read a graph structure from stdin by parsing a PACE 2020 formated graph.
///
/// *Errors:*
/// - May fail if the p-line is wrongly formated, an edge appears before the p-line, or there
///   is an uncommend line that does not encode an edge.
///
fn read_graph() -> Result<Graph, Box<dyn Error>> {
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

/// Compute an upper bound by using a heuristic.
/// Returns a tuple, consisting of the depth and the parent vector of the computed
/// treedepth decomposition.
///
fn compute_upper_bound(g: &Graph) -> (usize, Vec<Option<usize>>) {

    // translate g to fluid graph
    let mut h = fluid::graph::Graph::new(g.n);
    for v in 0..g.n {
        for &w in g.neighbors[v].iter() {
            if w > v { h.add_edge(v, w); }
        }
    }

    // run multiple iterations of the heuristic and take the best
    let mut decomposition = None;
    let mut depth         = g.n+1;
    for _ in 0..HEURISTIC_ITERATIONS {
        if let Ok(mut tmp) = fluid::separation::solve(&h, 3, 0) {
            fluid::postprocessor::optimize_dec(&mut tmp);
            let d = tmp.get_depth();
            if d < depth {
                depth = d;
                decomposition = Some(tmp);
            }
        }
    }
    
    // done
    if let Some(dec) = decomposition {
        return (depth, dec.parent);
    }        
    (g.n, vec![None; g.n])
}

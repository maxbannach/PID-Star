use crate::graph::Graph;
use crate::block_sieve::BlockSieve;
use bit_set::BitSet;
use std::rc::Rc;
use std::collections::{ HashMap, BinaryHeap };
use std::cmp::Ordering;

// size of list that we glue without further queries to the block sieve
const DIRECT_GLUE: usize = 10; 

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

pub struct Solver {
    positive: BlockSieve,                         // positive subgraphs that are adjacent to a given vertex
    memory:   HashMap<BitSet, (usize, usize)>,    // memoization for best known treedepth and center vertex
    queue:    BinaryHeap<Block>,                  // priority queue of the algorithm
    halt:     bool,                               // found solution
    pub k:    usize,                              // target treedepth
}

impl Solver {

    // simple constructur, just needs size of universe
    pub fn new(n: usize) -> Solver {
        Solver{ positive: BlockSieve::new(n), memory: HashMap::new(), queue: BinaryHeap::new(), halt: false, k: 0 }
    }
    
    pub fn solve(&mut self, g: &mut Graph) -> bool {

        // graph size may change due to preprocessing -> then we need new block sieve
        if g.n != self.positive.n { self.positive = BlockSieve::new(g.n); }
        
        // compute win configurations
        let k = self.k;
        (0..g.n).filter(|&v| g.neighbors[v].len() < k && g.descendants[v].is_empty() )
            .for_each(|v| self.queue.push(Block{ subgraph: (v..=v).collect(), depth: 1, center: v}) );

        // compute winning region
        while let Some(mut block) = self.queue.pop() {
            if self.halt { break; }
            
            // memoization
            if self.memory.contains_key(&block.subgraph) { continue; }
            self.memory.insert(block.subgraph.clone(), (block.depth, block.center));

            // get the neighborhood of the block
            let border = g.border_of(&block.subgraph);

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
                if g.contains_descendant(v, &border) { continue; }
                
                // reverse reveal-move
                let mut stack = Vec::new();
                stack.push( (block.subgraph.clone(), block.depth, -1) );                
                while let Some( (mut subgraph, depth, lex) ) = stack.pop() {

                    // Fast-Glue
                    let border   = g.border_of(&subgraph);
                    let mut list = self.positive.query(&g, self.k, &subgraph, &border, v, depth, lex);
                    self.fast_glue(g, &mut subgraph, &border, &mut list);
                    
                    // if the list is small, we can directly glue it without further queries to the block sieve
                    if list.len() <= DIRECT_GLUE {
                        if self.glue_list(g,subgraph, depth, v, &list) { return true; } else { continue; }
                    }
                    
                    // normal glue
                    list.iter()
                        .filter(  |other| other.is_disjoint(&subgraph) )
                        .for_each(|other| stack.push((subgraph.union(&other).collect(), depth, other.iter().next().unwrap() as isize)) );

                    // Ancestor-Rule
                    if !g.contains_all_descendants(v, &subgraph)     { continue; }
                    if  g.forbidden_ancestors(&subgraph).contains(v) { continue; }
                    
                    // done gluing this block -> offer it
                    subgraph.insert(v);                    
                    self.offer(g, subgraph, v, depth+1);
                    if self.halt { return true; }
                }                
            }
            
            // the block is optimal, so we can store it as positive instance
            self.positive.insert(&g, block.subgraph);
        }
        
        // done
        return self.halt;
    }
    
    /// Offer a new block to the queue. This checks whether we found a solution,
    /// we can prune the block, or just inserts it.
    ///
    fn offer(&mut self, g: &mut Graph, subgraph: BitSet, center: usize, depth: usize) {
        
        // final configurations
        let border = g.border_of(&subgraph);
        if depth + border.len() > self.k { return; }               // negative block
        if g.n - subgraph.len() <= self.k - depth {                // solution found
            self.halt = true;
            
            let mut mask: BitSet = (0..g.n).collect();             // the complete graph
            mask.difference(&subgraph).collect::<BitSet>().iter().for_each(|v| {
                self.memory.insert(mask.clone(), (0,v));
                mask.remove(v);
            });            
            self.memory.insert(mask.clone(), (0,center));
            
            // done
            return;
        }
        
        // done -> just insert new block into queue
        self.queue.push(Block{ subgraph: subgraph, center: center, depth: depth });
    }

    /// Immediately glue all blocks of the list to the subgraph that are safe to glue.
    ///
    fn fast_glue(&self, g: &Graph, subgraph: &mut BitSet, border: &BitSet, list: &mut Vec<Rc<BitSet>>) {
        let mut len  = list.len();
        let mut i    = 0;                    
        while i < len {
            if g.border_of(&list[i]).is_subset(&border) {
                subgraph.union_with(&list[i]);
                list.swap(i, len-1);
                len -= 1;
            }
            i += 1;
        }
        list.truncate(len);
    }

    /// Glue the elements in the list without further queries to the block sieve.
    /// Returns true if an optimal solution was found.
    ///
    fn glue_list(&mut self, g: &mut Graph, start: BitSet, depth: usize, v: usize, list: &Vec<Rc<BitSet>>) -> bool {
        let mut stack = Vec::new();
        stack.push( (start, depth, -1) );                
        while let Some( (mut subgraph, depth, lex) ) = stack.pop() {
            let mask = g.closed_border_of(&subgraph);
            list.iter().enumerate()
                .filter  (|(index, other)| *index as isize > lex && mask.is_disjoint(&other ) )
                .map     (|(index, other)| (index,subgraph.union(&other).collect::<BitSet>()) )
                .filter  (|(_,other)|      g.border_of(&other).len() + depth <= self.k        )
                .for_each(|(index, other)| stack.push((other, depth, index as isize))         );            
            if !g.contains_all_descendants(v, &subgraph)     { continue; }
            if  g.forbidden_ancestors(&subgraph).contains(v) { continue; }            
            subgraph.insert(v);            
            self.offer(g, subgraph, v, depth+1);
            if self.halt { return true; }
        }
        return false;
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
    pub fn next(&mut self) {
        self.k = self.k + 1;
        self.positive.clear();
        self.memory.clear();
        self.queue.clear();
        self.halt = false;
    }

    /// Computes the parent field in the given subgraph in the graph.
    /// This is done by extracting the corresponding center color.
    ///
    pub fn extract_decomposition(&self, g: &mut Graph, subgraph: Option<BitSet>, parent: Option<usize>) {
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

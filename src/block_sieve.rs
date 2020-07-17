use crate::graph::Graph;
use std::rc::Rc;
use std::boxed::Box;
use bit_set::BitSet;
use rand::{Rng, SeedableRng, seq::IteratorRandom, rngs::StdRng};

/// Simple macro to compute integer log2 – or, more precisely, max( floor(log2(x))+1, 0 ).
/// Only sensible defined for usize.
///
#[macro_export]
macro_rules! log2 {
    ($x:expr) => ( (0usize.leading_zeros() - ($x as usize).leading_zeros()) as usize );
}

/// Absolute difference of unsigned integers.
///
macro_rules! abs_diff {
    ($x:expr, $y:expr) => ( if $x > $y { $x - $y } else { $y - $x } );
}


const SEED:       u64   = 1337;     // seed for the random number generator
const TRIE_SPLIT: usize = 500;      // constant that indicates when to split lists stored in tries

// Strategy used to split the tries.
const TRIE_STRATEGY: SplittingStrategy = SplittingStrategy::Greedy;


/// Implementation of a randomized multi-level block sieve.
/// It can be used to store blocks and to then efficently query for a vertex v and a block C
/// all blocks C' with: 
/// - N[C] is disjoint to C'
/// - N(C cup C') < k - rho - 1 is possible
/// - C < C'
/// - v in N(C)
///
pub struct BlockSieve {
    sieves: Vec<LevelTwoSieve>,    // level-2 sieve for each vertex
    rng:    StdRng,                // random number generator
    pub n:  usize                  // size of the universe    
}

impl BlockSieve {
    
    pub fn new(n: usize) -> BlockSieve {
        let mut rng = SeedableRng::seed_from_u64(SEED);
        let sieves = vec![LevelTwoSieve::new(n, &mut rng); n];
        BlockSieve{ sieves, rng, n }
    }

    pub fn insert(&mut self, g: &Graph, subgraph: BitSet) {
        let border   = g.border_of(&subgraph);
        let subgraph = Rc::new(subgraph);
        border.iter().for_each(|v| self.sieves[v].insert(g, Rc::clone(&subgraph), &mut self.rng) );
    }

    pub fn query(&mut self, g: &Graph, k: usize, subgraph: &BitSet, border: &BitSet, v: usize, rho: usize, lex: isize) -> Vec<Rc<BitSet>> {
        self.sieves[v].query(g, k, subgraph, border, rho, lex)
    }

    pub fn clear(&mut self) {
        self.sieves.iter_mut().for_each(|sieve| sieve.clear() );
    }
}

/// The randomized multi-level block sieve consists of
/// multiple *sieves.* Each of these sieves implements this trait.
///
trait Sieve {
    
    /// Generate a new sieve for an n-vertex graph.
    ///
    fn new(n: usize, rng: &mut StdRng) -> Self;

    /// Insert a reference to a block into this sieve.
    ///
    fn insert(&mut self, g: &Graph, subgraph: Rc<BitSet>, rng: &mut StdRng);

    /// Returns a vector containing references to blocks that are stored within this sieve and that are
    /// compatible to the given tuple (subgraph, rho, lex).
    ///
    fn query(&self, g: &Graph,  k: usize, subgraph: &BitSet, border: &BitSet, rho: usize, lex: isize) -> Vec<Rc<BitSet>>;

    /// Removes all entries from the sieve.
    ///
    fn clear(&mut self);
}


/// Level-2 sieves partition the blocks by the minimal vertex they contain.
///
#[derive(Clone)]
struct LevelTwoSieve {
    sieves: Vec<LevelThreeSieve>,    
}

impl Sieve for LevelTwoSieve {
    
    fn new(n: usize, rng: &mut StdRng) -> LevelTwoSieve {
        LevelTwoSieve{ sieves: vec![LevelThreeSieve::new(n, rng); log2!(n)+1] }
    }

    fn insert(&mut self, g: &Graph, subgraph: Rc<BitSet>, rng: &mut StdRng) {
        let i = log2!(subgraph.iter().next().unwrap());
        self.sieves[i].insert(g, Rc::clone(&subgraph), rng);
    }

    fn query(&self, g: &Graph, k: usize, subgraph: &BitSet, border: &BitSet, rho: usize, lex: isize) -> Vec<Rc<BitSet>> {
        let start = match lex {
           x if x > 0 => log2!(x),
            _         => 0,
        };
        (start..self.sieves.len()).map(|i| self.sieves[i].query(g, k, subgraph, border, rho, lex) ).flatten().collect()
    }
    
    fn clear(&mut self) {
        self.sieves.iter_mut().for_each(|sieve| sieve.clear() );
    }    
}

/// Level-3 sieves partition the blocks with a random partition of the graph.
/// For that end, the vertices of the graph get randomly colored with two colors, say, blue and orange,
/// and two blocks are in the same equivalence class if they have the same amount of orange neighbors.
///
#[derive(Clone)]
struct LevelThreeSieve {
    sieves:   Vec<LevelFourSieve>,
    coloring: BitSet,
}

impl Sieve for LevelThreeSieve {
    
    fn new(n: usize, rng: &mut StdRng) -> LevelThreeSieve {
        let coloring = (0..n).filter(|_| rng.gen_bool(0.5)).collect();
        LevelThreeSieve{ sieves: vec![LevelFourSieve::new(n, rng); n], coloring: coloring }
    }

    fn insert(&mut self, g: &Graph, subgraph: Rc<BitSet>, rng: &mut StdRng) {
        let j = self.orange_neighbors(g, &subgraph);        
        self.sieves[j].insert(g, subgraph, rng);
    }

    fn query(&self, g: &Graph, k: usize, subgraph: &BitSet, border: &BitSet, rho: usize, lex: isize) -> Vec<Rc<BitSet>> {
        let l = self.blue_neighbors(border);
        (0..k+1-rho-l).map(|j| self.sieves[j].query(g, k, subgraph, border, rho, lex)).flatten().collect()
    }
    
    fn clear(&mut self) {
        self.sieves.iter_mut().for_each(|sieve| sieve.clear() );
    }    
}

impl LevelThreeSieve {

    /// Compute the number of blue neighbors of the given subgraph.
    ///
    fn blue_neighbors(&self, border: &BitSet) -> usize {
        border.iter().filter(|v| self.coloring.contains(*v)).count()
    }

    /// Compute the number of orange neighbors of the given subgraph.
    ///
    fn orange_neighbors(&self, g: &Graph, subgraph: &BitSet) -> usize {
        g.border_of(&subgraph).iter().filter(|v| !self.coloring.contains(*v)).count()            
    }    
}

/// The level-4 sieve stores blocks with a lazily build set trie.
///
#[derive(Clone)]
struct LevelFourSieve {
    n:           usize,                       // size of universe   
    undecided:   BitSet,                      // vertices that are not on a path to the root 
    required:    BitSet,                      // all blocks in this trie contain these vertices
    forbidden:   BitSet,                      // all blocks in this trie are disjoint to these vertices
    pins:        BitSet,                      // blocked vertices that are adjacent to an contained vertex 
    splitter:    Option<usize>,               // if this is not a leaf, split vertex for children
    left:        Option<Box<LevelFourSieve>>, // left child in trie
    right:       Option<Box<LevelFourSieve>>, // right child in trie
    list:        Vec<Rc<BitSet>>,             // content of node, if it is a leaf
}

/// Strategy used by the LevelFourSieve to split lists into tries.
///
#[allow(dead_code)]
enum SplittingStrategy {

    /// Pick a random vertex that was not used on the
    /// path to the root.
    ///
    Random,

    /// Pick two random vertices and take the one that splits better.
    ///
    PowerOfChoice,

    /// Take the vertex that splits best.
    ///
    Greedy,
}

impl Sieve for LevelFourSieve {

    fn new(n: usize, _rng: &mut StdRng) -> LevelFourSieve {
        LevelFourSieve::new_from(n, None, (0..n).collect(), BitSet::with_capacity(n), BitSet::with_capacity(n))
    }
    
    fn insert(&mut self, g: &Graph, subgraph: Rc<BitSet>, rng: &mut StdRng) {
        if self.splitter.is_none() {
            self.purge_list(g, &subgraph);
            self.list.push(subgraph);
            if self.list.len() > TRIE_SPLIT && !self.undecided.is_empty() {
                self.split(g, rng);
            }
        } else if let Some(x) = self.splitter {
            match subgraph.contains(x) {
                true  => self.left.as_mut().unwrap().insert(g, subgraph, rng),
                false => self.right.as_mut().unwrap().insert(g, subgraph, rng),
            }
        }
    }

    fn query(&self, g: &Graph, k: usize, subgraph: &BitSet, border: &BitSet, rho: usize, lex: isize) -> Vec<Rc<BitSet>> {
        // prune search in trie
        if let Some(index) = self.required.iter().next() {
            if (index as isize) <= lex { return Vec::new(); }
        }
        if !subgraph.is_disjoint(&self.pins) { return Vec::new(); }
        if border.union(&self.pins).count() + rho > k { return Vec::new(); }
        
        if self.splitter.is_none() {
            return self.collect_list(g, k, subgraph, border, rho, lex);
        } else if let Some(x) = self.splitter {
            let mut result = self.right.as_ref().unwrap().query(g, k, subgraph, border, rho, lex);
            if !subgraph.contains(x) {
                result.append(&mut self.left.as_ref().unwrap().query(g, k, subgraph, border, rho, lex));
            }
            return result;
        }
        Vec::new()
    }
    
    fn clear(&mut self) {
        if self.splitter.is_none() {
            self.list.clear();
        } else {
            self.left.as_mut().unwrap().clear();
            self.right.as_mut().unwrap().clear();
        }
    }    
}

impl LevelFourSieve {

    fn new_from(n: usize, g: Option<&Graph>, undecided: BitSet, required: BitSet, forbidden: BitSet) -> LevelFourSieve {        
        let mut pins = BitSet::with_capacity(n);
        if let Some(g) = g {
            forbidden.iter()
                .filter(|v| !g.matrix[*v].is_disjoint(&required))
                .for_each(|v| {pins.insert(v);});
        }
        LevelFourSieve{
            n:         n,
            undecided: undecided,
            required:  required,
            forbidden: forbidden,
            pins:      pins,
            splitter:  None,
            left:      None,
            right:     None,
            list:      Vec::new()
        }
    }
    
    /// Collect the blocks stored at a leaf.
    ///
    fn collect_list(&self, g: &Graph, k: usize, subgraph: &BitSet, border: &BitSet, rho: usize, lex: isize) -> Vec<Rc<BitSet>> {
        let mut result = Vec::new();
        let mask = g.closed_border_of(&subgraph);        
        for other in self.list.iter() {
            if !mask.is_disjoint(&other)                           { continue; }
            if g.border_of(&other).union(border).count() + rho > k { continue; }                        
            if (other.iter().next().unwrap() as isize) <= lex      { continue; }
            result.push(Rc::clone(other));
        }
        result
    }

    /// If this trie is just a list, this method splits it. Aftewards, this sieve becomes
    /// the root of a set trie.
    ///    
    fn split(&mut self, g: &Graph, rng: &mut StdRng) {       
        if let Some(x) = self.get_splitter(rng) {
            self.splitter = Some(x);
            let mut undecided = self.undecided.clone();
            undecided.remove(x);           
            let mut required = self.required.clone();
            required.insert(x);
            self.left = Some(Box::new(
                LevelFourSieve::new_from(self.n, Some(g), undecided.clone(), required, self.forbidden.clone())
            ));
           
            let mut forbidden = self.forbidden.clone();
            forbidden.insert(x);
            self.right = Some(Box::new(
                LevelFourSieve::new_from(self.n, Some(g), undecided, self.required.clone(), forbidden)
            ));
            
            for block in self.list.drain(..) {
                if block.contains(x) {
                    self.left.as_mut().unwrap().insert(g, block, rng);
                } else {
                    self.right.as_mut().unwrap().insert(g, block, rng);
                }
            }            
        }
    }

    /// Get a vertex to split the list stored in this trie to two new lists.
    /// The way this is done depends on the *SplittingStrategy.*
    ///
    fn get_splitter(&self, rng: &mut StdRng) -> Option<usize> {
        let len = self.list.len();
        if len == 1 { return self.undecided.iter().next() }
        match TRIE_STRATEGY {
            // selects used a random vertex
            SplittingStrategy::Random => self.undecided.iter().choose(rng),
            // picks to vertices, the one that is in about n/2 of the blocks is taken
            // ties are broken at random
            SplittingStrategy::PowerOfChoice => {
                self.undecided.iter().choose_multiple(rng, 2).iter().map(|&v| v)
                    .min_by_key(|v| abs_diff!(self.list.len()/2, self.list.iter().filter(|block| block.contains(*v)).count()) )
            },
            // as power of choice, but test all vertices
            SplittingStrategy::Greedy => self.undecided.iter()
                .min_by_key(|v| abs_diff!(self.list.len()/2, self.list.iter().filter(|block| block.contains(*v)).count()) )
        }       
    }

    /// Purges a list of blocks, i.e., a leaf of a trie.
    /// This function removes blocks that are dominated by the given block.
    ///
    fn purge_list(&mut self, g: &Graph, subgraph: &BitSet) {
        let     border = g.border_of(&subgraph);
        let mut len    = self.list.len();
        let mut i      = 0;
        while i < len {
            let other = &self.list[i];

            // check if subgraph is better than the other
            if subgraph.is_superset(&other) && border.is_subset(&g.border_of(other)) {
                self.list.swap(i, len-1);
                len -= 1;
            }
            
            i += 1;
        }
        self.list.truncate(len);
    }    
}

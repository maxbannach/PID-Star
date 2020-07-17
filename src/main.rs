use pid_star::graph::Graph;
use pid_star::preprocessor::Preprocessor;
use pid_star::solver::Solver;

fn main() {
    // read input graph
    let mut g = match Graph::new_from_stdin() {
        Ok(g)  => g,
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    };

    // do some simple preprocessing
    let (mut h, iso) = Preprocessor::remove_leaves(&mut g);
    h.compute_descendants();
    
    // setup the solver and solve instance
    let mut solver = Solver::new(g.n);
    let mut do_preprocessing = true;
    loop {
        solver.next();
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

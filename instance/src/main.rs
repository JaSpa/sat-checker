use std::{
    io::{self, Write},
    path::PathBuf,
    process::exit,
};

use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    /// Path to underlying instance in DIMACS format.
    instance: PathBuf,
}

fn main() -> io::Result<()> {
    println!("c  If you are seeing this output on a terminal you called the `minisat-instance`");
    println!("c  program directly which is not intended. Use the test runner instead.");

    let args = Args::parse();

    // Create the solver.
    let mut solver = ffi::Solver::new();
    // Read the instance.
    solver.read_instance(args.instance)?;
    // Respond with the number of variables occuring inside the instance.
    let inst_size = solver.var_count();
    println!("n  {inst_size}");

    // Wait for a set of assumptions to arrive.
    let mut line = String::new();
    loop {
        line.clear();
        let n = io::stdin().read_line(&mut line)?;

        // EOF reached.
        if n == 0 {
            break;
        }

        let parts = line.split_ascii_whitespace().collect::<Vec<_>>();

        // Skip empty lines.
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "q" => {
                // Quit.
                break;
            }
            "s" => {
                // Solve with the following assumptions.
                let Ok(assumptions) = parts[1..].iter().map(|s| s.parse::<ffi::Lit>()).collect::<Result<Vec<_>, _>>() else {
                    eprintln!("failed to parse literals in solve command: {line}");
                    exit(1);
                };

                if !assumptions
                    .iter()
                    .all(|n| (1..=inst_size).contains(&(n.abs() as usize)))
                {
                    eprintln!("invalid variable in solve command: {line}");
                    exit(1);
                }

                match solver.solve(&assumptions) {
                    Ok(()) => println!("S"),
                    Err(confl) => {
                        let mut out = io::stdout().lock();
                        write!(out, "U ")?;
                        for lit in confl {
                            write!(out, " {lit}")?;
                        }
                        writeln!(out)?;
                        out.flush()?;
                    }
                }
            }
            _ => {
                eprintln!("invalid command: {}", parts[0]);
                exit(5);
            }
        }
    }

    Ok(())
}

mod ffi {
    use std::{ffi, io, os::unix::prelude::OsStrExt, path::Path, ptr::NonNull};

    use cpp::cpp;

    pub struct Solver(NonNull<ffi::c_void>);

    pub type Lit = ffi::c_int;

    cpp! {{
        #include <minisat/core/Dimacs.h>
        #include <minisat/core/Solver.h>
        #include <zlib.h>

        using Solver = Minisat::Solver;
        using Lit = Minisat::Lit;
    }}

    impl Solver {
        pub fn new() -> Self {
            unsafe {
                let ptr = cpp!([] -> *mut ffi::c_void as "Solver*" {
                    return new Solver;
                });
                Solver(NonNull::new(ptr).expect("allocation failure"))
            }
        }

        fn with_ptr<R>(&self, body: impl FnOnce(*const ffi::c_void) -> R) -> R {
            body(self.0.as_ptr())
        }

        fn with_mut_ptr<R>(&mut self, body: impl FnOnce(*mut ffi::c_void) -> R) -> R {
            body(self.0.as_ptr())
        }

        pub fn var_count(&self) -> usize {
            self.with_ptr(|ptr| {
                cpp!(unsafe [ptr as "const Solver*"] -> ffi::c_int as "int" {
                    return ptr->nVars();
                })
            }) as usize
        }

        pub fn read_instance(&mut self, path: impl AsRef<Path>) -> io::Result<()> {
            let encoded_path = {
                let os_path = path.as_ref().as_os_str();
                let mut os_bytes = os_path.as_bytes().to_vec();
                os_bytes.push(0); // zero-termination
                os_bytes
            };

            let path_ptr = encoded_path.as_ptr();

            let good = self.with_mut_ptr(|ptr| {
                cpp!(unsafe [ptr as "Solver*", path_ptr as "const char*"] -> bool as "bool" {
                    gzFile infile = gzopen(path_ptr, "rb");
                    if (!infile)
                        return false;

                    Minisat::parse_DIMACS(infile, *ptr);
                    gzclose(infile);
                    return true;
                })
            });

            if good {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        }

        fn read_conflict(&self) -> Vec<Lit> {
            self.with_ptr(|ptr| {
                // Determine the size of the conflict vector.
                let confl_size = cpp!(unsafe [ptr as "const Solver*"] -> ffi::c_int as "int" {
                    return ptr->conflict.size();
                });

                // Prepare a vector to which we will write the data.
                let mut confl_lits = vec![0; confl_size as usize];
                let cptr = confl_lits.as_mut_ptr();
                cpp!(unsafe [ptr as "const Solver*", cptr as "int*"] {
                    using namespace Minisat;
                    auto &confl = ptr->conflict;
                    for (size_t i = 0, n = confl.size(); i < n; ++i) {
                        // Translate *2+1 back to +/- encoding.
                        Lit c = confl[i];
                        cptr[i] = sign(c) ? -var(c) : var(c);
                    }
                });

                confl_lits
            })
        }

        pub fn solve(&mut self, assumptions: &[Lit]) -> Result<(), Vec<Lit>> {
            let success =self.with_mut_ptr(|ptr| {
                let a_ptr = assumptions.as_ptr();
                let a_cnt = assumptions.len();
                cpp!(unsafe [ptr as "Solver*", a_ptr as "const int*", a_cnt as "size_t"] -> bool as "bool" {
                    using namespace Minisat;
                    vec<Lit> assumptions(a_cnt);
                    for (size_t i = 0; i < a_cnt; ++i) {
                        // Translate pos/neg to minisat's internal encoding.
                        Lit a = mkLit(abs(a_ptr[i]) - 1, /*neg=*/a_ptr[i] < 0);
                        assumptions.push(a);
                    }

                    return ptr->solve(assumptions);
                })
            });

            if success {
                Ok(())
            } else {
                Err(self.read_conflict())
            }
        }
    }

    impl Default for Solver {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for Solver {
        fn drop(&mut self) {
            self.with_mut_ptr(|ptr| unsafe {
                cpp!([ptr as "Solver*"] {
                    delete ptr;
                });
            })
        }
    }
}

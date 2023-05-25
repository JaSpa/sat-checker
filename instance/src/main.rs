use cpp::cpp;

cpp! {{
    #include <iostream>
    #include <minisat/core/Solver.h>
}}

fn main() {
    println!("c  If you are seeing this output on a terminal you called the `minisat-instance`");
    println!("c  program directly, which is not intended. Use `minisat-tests` instead.");

    let i = 42i32;

    cpp!(unsafe [i as "int32_t"] {
        std::cout << "Hi from rusty c++ " << i << "\n";
        Minisat::Solver s;
        std::cout << "I created a solver!\n";
    })
}

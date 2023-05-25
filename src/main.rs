use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
struct Args {
    /// Seed to reproduce results.
    #[arg(short, long)]
    seed: Option<usize>,

    /// CNF instance to test with.
    instance: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    println!("{args:?}");
}

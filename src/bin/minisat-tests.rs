use clap::Parser;
use color_eyre::eyre::Result;
use rand::SeedableRng;
use std::path::PathBuf;

#[derive(Debug, Parser)]
struct Args {
    /// Seed to reproduce results.
    #[arg(short, long)]
    seed: Option<u64>,

    /// CNF instance to test with.
    instance: Option<PathBuf>,
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = Args::parse();
    let seed = match args.seed {
        Some(seed) => seed,
        None => gen_seed()?,
    };

    println!("Rerun with  --seed={seed}");

    let _rng = rand::rngs::SmallRng::seed_from_u64(seed);

    println!("{args:?}");
    Ok(())
}

fn gen_seed() -> Result<u64> {
    let mut bytes = [0; 8];
    getrandom::getrandom(&mut bytes)?;
    Ok(u64::from_ne_bytes(bytes))
}

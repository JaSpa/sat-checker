#![feature(iter_collect_into)]

use std::{
    ffi::{OsStr, OsString},
    process::Stdio,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

use clap::Parser;
use color_eyre::eyre::Result;
use tokio::{process, sync::mpsc::unbounded_channel};

#[derive(Debug, Parser)]
struct Args {
    /// Seed to reproduce results.
    #[arg(short, long)]
    seed: Option<u64>,

    /// Print additional output.
    #[arg(short, long)]
    verbose: bool,

    /// CNF instance to test with.
    instance: OsString,

    #[clap(env = "MINISAT_TEST_RUNNER_INSTANCE_A", long)]
    solver_a: OsString,

    #[clap(env = "MINISAT_TEST_RUNNER_INSTANCE_B", long)]
    solver_b: OsString,
}

#[tokio::main]
async fn main() -> Result<()> {
    color_eyre::install()?;

    let args = Args::parse();
    let seed = match args.seed {
        Some(seed) => seed,
        None => gen_seed()?,
    };

    println!("Rerun with  --seed={seed}");

    let term_signal = Arc::new(AtomicBool::new(false));
    let term_signal_copy = term_signal.clone();

    tokio::spawn(async move {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install signal handler");
        term_signal_copy.store(true, Ordering::Relaxed);
    });

    if args.verbose {
        println!("launching test instances");
    }

    let mut a = spawn_instance(&args.solver_a, &args.instance)?;
    let mut b = spawn_instance(&args.solver_b, &args.instance)?;

    let verbose = args.verbose;
    let a_in = a.stdin.take().expect("no stdin pipe to A");
    let a_out = a.stdout.take().expect("no stdout pipe to A");
    let a_err = a.stderr.take().expect("no stderr pipe to A");
    let b_in = b.stdin.take().expect("no stdin pipe to B");
    let b_out = b.stdout.take().expect("no stdout pipe to B");
    let b_err = b.stderr.take().expect("no stderr pipe to B");

    let (events_out, events_in) = unbounded_channel();

    let stream_relay = tokio::spawn(async move {
        relay::StreamRelay::new(verbose, a_out, a_err, b_out, b_err)
            .run(events_out)
            .await
    });

    let coordinator = tokio::spawn(async move {
        coordinator::Coordinator::new(a_in, b_in, events_in, seed, verbose, term_signal)
            .run()
            .await
    });

    coordinator.await??;
    stream_relay.await??;
    a.wait().await?;
    b.wait().await?;

    Ok(())
}

fn gen_seed() -> Result<u64> {
    let mut bytes = [0; 8];
    getrandom::getrandom(&mut bytes)?;
    Ok(u64::from_ne_bytes(bytes))
}

fn spawn_instance(exe: &OsStr, cnf_path: &OsStr) -> Result<process::Child> {
    Ok(process::Command::new(exe)
        .arg(cnf_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?)
}

mod comm {
    use std::fmt;

    use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender};

    use crate::util::WriteJoined;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Tag {
        A,
        B,
    }

    impl fmt::Display for Tag {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Tag::A => write!(f, "A"),
                Tag::B => write!(f, "B"),
            }
        }
    }

    #[derive(Debug)]
    pub enum Event {
        SAT,
        UNSAT(Vec<i32>),
        LOADED(usize),
    }

    impl fmt::Display for Event {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Event::SAT => write!(f, "SAT"),
                Event::UNSAT(confl) => write!(f, "UNSAT {}", WriteJoined::by_space(confl)),
                Event::LOADED(n) => write!(f, "LOADED {n}"),
            }
        }
    }

    pub type EventsSend = UnboundedSender<(Tag, Event)>;
    pub type EventsRecv = UnboundedReceiver<(Tag, Event)>;
}

mod coordinator {
    use std::{
        io::{self, Write},
        sync::{atomic::AtomicBool, Arc},
    };

    use color_eyre::{
        eyre::{eyre, Context},
        Result,
    };
    use ndhistogram::{
        axis::{BinInterval, UniformNoFlow},
        ndhistogram, Histogram,
    };
    use patricia_tree::PatriciaSet;
    use rand::{prelude::Distribution, rngs::SmallRng, seq::IteratorRandom, Rng, SeedableRng};
    use tokio::{
        io::{AsyncWriteExt, BufWriter},
        process,
    };

    use crate::{
        comm::{Event, EventsRecv, Tag},
        util::{Sorted, WriteJoined},
    };

    struct Data {
        a: BufWriter<process::ChildStdin>,
        b: BufWriter<process::ChildStdin>,
        events: EventsRecv,
        stop_flag: Arc<AtomicBool>,
        verbose: bool,
    }

    #[must_use]
    pub struct Coordinator(Data, u64);

    impl Coordinator {
        pub fn new(
            a: process::ChildStdin,
            b: process::ChildStdin,
            events: EventsRecv,
            rng_seed: u64,
            verbose: bool,
            stop_flag: Arc<AtomicBool>,
        ) -> Self {
            Coordinator(
                Data {
                    a: BufWriter::new(a),
                    b: BufWriter::new(b),
                    events,
                    verbose,
                    stop_flag,
                },
                rng_seed,
            )
        }

        pub async fn run(mut self) -> Result<()> {
            while let Some((tag, ev)) = self.0.events.recv().await {
                if let Event::LOADED(n) = ev {
                    return self.make_sized(n).run().await;
                }

                println!("* ignoring early event from {tag}: {ev}");
            }

            Err(eyre!("instance did not load"))
        }

        fn make_sized(self, size: usize) -> SizedCoordinator {
            let max_assump_size = if size < 5 { size } else { size * 2 / 3 };
            let hist_steps = if size / 16 <= 1 { size } else { 16 };
            SizedCoordinator {
                data: self.0,
                rng: SmallRng::seed_from_u64(self.1),
                var_count: size as i32,
                count_distrib: rand::distributions::Uniform::new(1, max_assump_size),
                chosen_sizes: ndhistogram!(
                    UniformNoFlow::with_step_size(hist_steps, 1, (size / 16) + 1);
                    usize
                ),
                sat_challenges: PatriciaSet::new(),
                encoded_challenge: Vec::new(),
            }
        }
    }

    #[must_use]
    struct SizedCoordinator {
        data: Data,
        rng: SmallRng,
        var_count: i32,
        count_distrib: rand::distributions::Uniform<usize>,
        chosen_sizes: ndhistogram::Hist1D<UniformNoFlow<usize>, usize>,
        /// Stores the challenges which resulted in SAT. We don't have to spend time on rechecking
        /// subsets of these.
        sat_challenges: PatriciaSet,
        encoded_challenge: Vec<u8>,
    }

    impl SizedCoordinator {
        async fn run(mut self) -> Result<()> {
            let mut i = 1 as usize;

            while !self
                .data
                .stop_flag
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                println!("{:=>80}", format!(" {i}"));
                i += 1;

                let challenge = self.select_challenge()?;
                self.perform_challenge(challenge).await?;
            }

            Ok(())
        }

        fn select_challenge(&mut self) -> Result<Sorted<i32>> {
            for tryn in 0..50 {
                if self.data.verbose && tryn > 0 && tryn % 5 == 0 {
                    println!("* retrying challenge generation #{tryn}");
                }

                // Choose size.
                let size = self.count_distrib.sample(&mut self.rng);

                // Choose actual inhabitants.
                let mut vars = (1..=self.var_count).choose_multiple(&mut self.rng, size);

                // For each variable decide if it should appear negated.
                for v in vars.iter_mut() {
                    if self.rng.gen() {
                        *v = *v * -1;
                    }
                }

                let vars = Sorted::new(vars);

                // Check that it isn't a challenge which would result in SAT anyways.
                self.encode_challenge(&vars);
                if self
                    .sat_challenges
                    .iter_prefix(&self.encoded_challenge)
                    .next()
                    .is_none()
                {
                    // Return challenge.
                    return Ok(vars);
                }
            }

            Err(eyre!("failed to generate a new challenge after 50 tries"))
        }

        fn encode_challenge(&mut self, challenge: &[i32]) {
            self.encoded_challenge.clear();
            if self.var_count <= i8::MAX as i32 {
                challenge
                    .iter()
                    .map(|&v| v as u8)
                    .collect_into(&mut self.encoded_challenge);
            } else if self.var_count <= i16::MAX as i32 {
                challenge
                    .iter()
                    .flat_map(|&v| (v as i16).to_le_bytes())
                    .collect_into(&mut self.encoded_challenge);
            } else {
                challenge
                    .iter()
                    .flat_map(|&v| v.to_le_bytes())
                    .collect_into(&mut self.encoded_challenge);
            }
        }

        async fn send_challenge(&mut self, tag: Tag, challenge: &[i32]) -> Result<()> {
            let handle = match tag {
                Tag::A => &mut self.data.a,
                Tag::B => &mut self.data.b,
            };

            handle.write_u8(b's').await?;
            for v in challenge {
                handle.write_u8(b' ').await?;
                handle.write_all(v.to_string().as_bytes()).await?;
            }
            handle.write_u8(b'\n').await?;
            handle.flush().await?;
            Ok(())
        }

        async fn read_challenge_answer(&mut self) -> Result<(Option<Vec<i32>>, Option<Vec<i32>>)> {
            let mut a_res: Option<Result<(), Vec<i32>>> = None;
            let mut b_res: Option<Result<(), Vec<i32>>> = None;

            while let Some(ev) = self.data.events.recv().await {
                match ev {
                    (Tag::A, Event::SAT) => a_res = Some(Ok(())),
                    (Tag::B, Event::SAT) => b_res = Some(Ok(())),
                    (Tag::A, Event::UNSAT(confl)) => a_res = Some(Err(confl)),
                    (Tag::B, Event::UNSAT(confl)) => b_res = Some(Err(confl)),
                    (_, Event::LOADED(_)) => {}
                }

                if a_res.is_some() && b_res.is_some() {
                    break;
                }
            }

            match (a_res, b_res) {
                (None, None) | (Some(_), None) | (None, Some(_)) => {
                    Err(eyre!("missing challenge results from A/B"))
                }
                (Some(res_a), Some(res_b)) => Ok((res_a.err(), res_b.err())),
            }
        }

        async fn perform_challenge(&mut self, assumptions: Sorted<i32>) -> Result<()> {
            println!(
                "Challenging with {} {}",
                assumptions.len(),
                if assumptions.len() == 1 {
                    "assumption"
                } else {
                    "assumptions"
                }
            );
            if self.data.verbose {
                println!("  {}", WriteJoined::by_space(&assumptions))
            }

            self.chosen_sizes.fill(&assumptions.len());
            self.show_histogram()?;

            // Send challange.
            self.send_challenge(Tag::A, &assumptions)
                .await
                .wrap_err("failed to send solve request to A")?;
            self.send_challenge(Tag::B, &assumptions)
                .await
                .wrap_err("failed to send solve request to B")?;

            // Wait for results.
            let result = self
                .read_challenge_answer()
                .await
                .wrap_err("an error occured while waiting for the challenge result")?;
            let (a_confl, b_confl) = match result {
                (None, None) => {
                    println!("=> SAT");
                    self.sat_challenges.insert(&self.encoded_challenge);
                    return Ok(());
                }

                (Some(mut a_confl), Some(mut b_confl)) => {
                    for x in a_confl.iter_mut().chain(b_confl.iter_mut()) {
                        *x = -*x;
                    }

                    let xs = Sorted::new(a_confl);
                    let ys = Sorted::new(b_confl);
                    let m = xs.len();
                    let n = ys.len();

                    // Check for a trivial equivalence in which case we can save some computation
                    // time and move on to the next test case.
                    if m == n {
                        if xs == ys {
                            println!("=> UNSAT, same conflict clauses");
                            return Ok(());
                        }
                    }

                    println!("UNSAT, {m} vs {n}");
                    (xs, ys)
                }

                (None, Some(_)) | (Some(_), None) => {
                    return Err(eyre!("SAT/UNSAT conflict!"));
                }
            };

            // Check that the conflicts
            //  (1) are a subset of the challenge
            //  (2) result in UNSAT in the other instance
            if !a_confl.is_subset_of(&assumptions) {
                return Err(eyre!(
                    concat!(
                        "conflict from A not a subset of the assumptions\n",
                        "  assumptions: {}\n",
                        "     conflict: {}"
                    ),
                    WriteJoined::by_space(&assumptions),
                    WriteJoined::by_space(&a_confl),
                ));
            }
            if !b_confl.is_subset_of(&assumptions) {
                return Err(eyre!(
                    concat!(
                        "conflict from B not a subset of the assumptions\n",
                        "  assumptions: {}\n",
                        "     conflict: {}"
                    ),
                    WriteJoined::by_space(&assumptions),
                    WriteJoined::by_space(&b_confl),
                ));
            }

            // Dispatch renewed challenges for (2).
            self.send_challenge(Tag::A, &b_confl)
                .await
                .wrap_err("failed to send check-request to A")?;
            self.send_challenge(Tag::B, &a_confl)
                .await
                .wrap_err("failed to send check-request to B")?;

            // Wait for check results.
            let (a_res, b_res) = self
                .read_challenge_answer()
                .await
                .wrap_err("an error occured while waiting for the re-check result")?;

            if a_res.is_none() {
                return Err(eyre!(
                    concat!(
                        "conflict returned by B is not a conflict in A\n",
                        "    assumptions: {}\n",
                        "  conflict by B: {}",
                    ),
                    WriteJoined::by_space(&assumptions),
                    WriteJoined::by_space(&b_confl),
                ));
            }
            if b_res.is_none() {
                return Err(eyre!(
                    concat!(
                        "conflict returned by A is not a conflict in B\n",
                        "    assumptions: {}\n",
                        "  conflict by A: {}",
                    ),
                    WriteJoined::by_space(&assumptions),
                    WriteJoined::by_space(&a_confl),
                ));
            }

            Ok(())
        }

        fn show_histogram(&self) -> io::Result<()> {
            // Pre-format labels for aligned output
            let labels = self
                .chosen_sizes
                .iter()
                .filter_map(|bucket| {
                    let BinInterval::Bin { start, end } = bucket.bin else {
                        return None
                    };
                    Some(if start + 1 == end {
                        start.to_string()
                    } else {
                        format!("{start}-{end}", end = end - 1)
                    })
                })
                .collect::<Vec<_>>();
            let label_length = labels.iter().map(|lbl| lbl.len()).max().unwrap_or(0);

            // Normalize widths to a max of 80 columns.
            let max_widths = self.chosen_sizes.values().copied().max().unwrap_or(0);
            let norm_factor = if max_widths <= 80 {
                1.0
            } else {
                80.0 / (max_widths as f64)
            };

            // Output histogram.
            let mut out = io::stdout().lock();
            for (label, bucket) in std::iter::zip(labels, self.chosen_sizes.into_iter()) {
                let n = (*bucket.value as f64 * norm_factor) as usize;
                writeln!(out, "{label:>label_length$} {:*<n$}", "")?;
            }

            Ok(())
        }
    }
}

mod relay {
    use std::io;

    use color_eyre::Result;
    use tokio::{
        io::{AsyncBufRead, AsyncBufReadExt, BufReader},
        process,
    };

    use crate::comm::{Event, EventsSend, Tag};

    #[must_use]
    pub struct StreamRelay {
        verbose: bool,
        is_eof: [bool; 4],
        buffers: [Vec<u8>; 4],
        readers: [Box<dyn AsyncBufRead + Send + std::marker::Unpin>; 4],
    }

    impl StreamRelay {
        pub fn new(
            verbose: bool,
            a_out: process::ChildStdout,
            a_err: process::ChildStderr,
            b_out: process::ChildStdout,
            b_err: process::ChildStderr,
        ) -> Self {
            StreamRelay {
                verbose,
                is_eof: [false; 4],
                buffers: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
                readers: [
                    Box::new(BufReader::new(a_out)),
                    Box::new(BufReader::new(a_err)),
                    Box::new(BufReader::new(b_out)),
                    Box::new(BufReader::new(b_err)),
                ],
            }
        }

        const TAGS: [Tag; 4] = [Tag::A, Tag::A, Tag::B, Tag::B];

        fn parse_line(
            verbose: bool,
            line: &str,
        ) -> Result<(bool, &'static str, Option<Event>), &'static str> {
            let mut parts = line.split_ascii_whitespace();
            let (ev, consumed) = match parts.next() {
                Some("c") | None => return Ok((false, "", None)),
                Some("n") => {
                    let Some(nstr) = parts.next() else {
                        return Err("bad `n` response");
                    };
                    let Ok(n) = nstr.parse() else {
                        return Err("bad `n` response");
                    };
                    (Event::LOADED(n), parts.next().is_none())
                }
                Some("S") => (Event::SAT, parts.next().is_none()),
                Some("U") => {
                    let Some(confl) = parts.map(|s| s.parse().ok()).collect::<Option<Vec<_>>>() else {
                        return Err("bad `U` response");
                    };
                    (Event::UNSAT(confl), true)
                }
                Some("r") => return Ok((verbose, "", None)),
                Some(_) => return Ok((true, "", None)),
            };

            if consumed {
                Ok((verbose, "", Some(ev)))
            } else {
                Ok((true, "interpreting overfull response as command", Some(ev)))
            }
        }

        fn handle_line(&mut self, events: &EventsSend, index: usize) -> Result<()> {
            let buf = &mut self.buffers[index];
            let full_line = String::from_utf8_lossy(buf);
            let ln = full_line.trim_end();
            let opt_ev = match Self::parse_line(self.verbose, &ln) {
                Ok((print, warn, opt_ev)) => {
                    if print && !warn.is_empty() {
                        println!("{}> {ln}\n[warning: {warn}]", Self::TAGS[index]);
                    } else if print {
                        println!("{}> {ln}", Self::TAGS[index]);
                    }
                    opt_ev
                }
                Err(msg) => {
                    println!("{}> {ln}\n[error: {msg}, skipping]", Self::TAGS[index]);
                    None
                }
            };

            buf.clear();

            if let Some(ev) = opt_ev {
                // Ignore any errors/discard untrasmittable events.
                _ = events.send((Self::TAGS[index], ev));
            }

            Ok(())
        }

        fn process_result(
            &mut self,
            events: &EventsSend,
            result: io::Result<usize>,
            index: usize,
        ) -> Result<()> {
            match result {
                Ok(0) => self.is_eof[index] = true,
                Ok(_) => self.handle_line(events, index)?,
                Err(e) => {
                    self.is_eof[index] = true;
                    for ln in e.to_string().lines() {
                        println!("{}> {ln}", Self::TAGS[index])
                    }
                }
            }

            Ok(())
        }

        async fn try_read(
            is_eof: bool,
            reader: &mut (impl AsyncBufRead + std::marker::Unpin),
            buffer: &mut Vec<u8>,
        ) -> io::Result<usize> {
            if is_eof {
                tokio::task::yield_now().await;
                Ok(0)
            } else {
                reader.read_until(b'\n', buffer).await
            }
        }

        pub async fn run(&mut self, events: EventsSend) -> Result<()> {
            while self.is_eof.contains(&false) {
                let (lo, hi) = self.readers.split_at_mut(2);
                let (ra, rb) = lo.split_at_mut(1);
                let (rc, rd) = hi.split_at_mut(1);
                let (lo, hi) = self.buffers.split_at_mut(2);
                let (ba, bb) = lo.split_at_mut(1);
                let (bc, bd) = hi.split_at_mut(1);
                tokio::select! {
                    r = Self::try_read(self.is_eof[0], &mut ra[0], &mut ba[0]) => self.process_result(&events, r, 0)?,
                    r = Self::try_read(self.is_eof[1], &mut rb[0], &mut bb[0]) => self.process_result(&events, r, 1)?,
                    r = Self::try_read(self.is_eof[2], &mut rc[0], &mut bc[0]) => self.process_result(&events, r, 2)?,
                    r = Self::try_read(self.is_eof[3], &mut rd[0], &mut bd[0]) => self.process_result(&events, r, 3)?,
                }
            }

            Ok(())
        }
    }
}

mod util {
    use core::fmt;
    use std::ops;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Sorted<T>(Vec<T>);

    impl<T> Sorted<T> {
        pub fn new(mut values: Vec<T>) -> Self
        where
            T: Ord,
        {
            values.sort();
            Sorted(values)
        }

        pub fn is_subset_of(&self, full_set: &Sorted<T>) -> bool
        where
            T: Eq,
        {
            if self.len() < full_set.len() {
                // continue below
            } else if self.len() > full_set.len() {
                return false;
            } else {
                return self == full_set;
            }

            let mut sub_it = self.iter();
            let mut full_it = full_set.iter();

            'next_subval: while let Some(val) = sub_it.next() {
                while let Some(full_val) = full_it.next() {
                    if val == full_val {
                        continue 'next_subval;
                    }
                }

                return false;
            }

            return true;
        }
    }

    impl<T> ops::Deref for Sorted<T> {
        type Target = [T];

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    pub struct WriteJoined<'a, T, Sep>(&'a [T], Sep);

    impl<'a, T> WriteJoined<'a, T, char> {
        pub fn by_space(values: &'a [T]) -> Self {
            Self::by_char(values, ' ')
        }

        pub fn by_char(values: &'a [T], char: char) -> Self {
            WriteJoined(values, char)
        }
    }

    impl<'a, T: fmt::Display, Sep: fmt::Display> fmt::Display for WriteJoined<'a, T, Sep> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let mut it = self.0.iter();
            let Some(first) = it.next() else {
                return Ok(());
            };

            write!(f, "{first}")?;
            for val in it {
                write!(f, "{}{val}", self.1)?;
            }

            Ok(())
        }
    }
}

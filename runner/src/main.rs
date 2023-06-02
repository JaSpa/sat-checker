use std::{fmt, process::ExitStatus, str::FromStr};

use color_eyre::{eyre::WrapErr, Result};
use futures_core::future::LocalBoxFuture;
use solver::Solver;
use util::MergeErrors;

use crate::coordinator::RandomSearch;

macro_rules! concat_lines {
    ($($ln:literal),+ $(,)?) => {
        concat!($($ln, "\n"),+)
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Tag {
    A,
    B,
}

impl Tag {
    #[must_use]
    fn select<T>(self, a: T, b: T) -> T {
        self.select_with(|| a, || b)
    }

    fn select_with<T>(self, a: impl FnOnce() -> T, b: impl FnOnce() -> T) -> T {
        match self {
            Tag::A => a(),
            Tag::B => b(),
        }
    }

    fn to_str(self) -> &'static str {
        self.select("A", "B")
    }

    /// Returns the other tag.
    fn other(self) -> Self {
        self.select(Tag::B, Tag::A)
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.pad(self.to_str())
    }
}

impl FromStr for Tag {
    type Err = ();
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "A" | "a" => Ok(Tag::A),
            "B" | "b" => Ok(Tag::B),
            _ => Err(()),
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    color_eyre::install()?;

    if cmd::args().debug() {
        eprintln!("{:#?}", cmd::args());
    }

    with_solvers(|a, b| Box::pin(run(a, b))).await
}

/// Launches the solvers and passes them to the given computation. However `inner` returns (`Err`
/// or `Ok`) the solvers will be shut down before passing the result on.
async fn with_solvers<R>(
    inner: impl for<'a> FnOnce(&'a mut Solver, &'a mut Solver) -> LocalBoxFuture<'a, Result<R>>,
) -> Result<R> {
    let mut solver_a = solver::launch(Tag::A)?;
    let mut solver_b = solver::launch(Tag::B)?;

    let r = inner(&mut solver_a, &mut solver_b).await;

    tokio::join!(
        async { solver_a.terminate().await.map(print_solver_exit(Tag::A)) },
        async { solver_b.terminate().await.map(print_solver_exit(Tag::B)) },
    )
    .merge_errors_msg("Multiple errors occured while terminating solver processes")?;

    r
}

async fn run(a: &mut Solver, b: &mut Solver) -> Result<()> {
    let args = cmd::args();
    let mut coordinator = coordinator::Coordinator::new(a, b);

    // Run every set of specified assumptions.
    for assumptions in args.assume.iter() {
        if let Some(fail) = coordinator.run_challenge(assumptions).await? {
            println!("{fail}");
        }
    }

    // We did what we came here for.
    if !cmd::args().assume.is_empty() {
        return Ok(());
    }

    // Otherwise start the random search.
    let size = coordinator.size().await?;
    let (mut searcher, seed) =
        RandomSearch::new(args.seed, size).wrap_err("failed to initialize random search")?;
    let mut fail = coordinator.run_search(&mut searcher).await?;
    fail.push_rerun(format!(
        "use  --seed={seed}  to rerun the complete sequence"
    ));
    println!("{fail}");
    Ok(())
}

fn print_solver_exit(tag: Tag) -> impl Fn(ExitStatus) {
    move |exit| {
        let args = cmd::args();
        if exit.success() && !args.verbose() {
            return;
        }

        eprintln!("Solver {} terminated with {}", args.solver_name(tag), exit);
    }
}

mod cmd {
    use std::{
        ffi::{OsStr, OsString},
        mem::MaybeUninit,
        path::PathBuf,
        sync::Once,
    };

    use crate::{solver, Tag};

    #[derive(clap::Parser)]
    struct ArgsParser {
        #[command(flatten)]
        args: Args,
    }

    #[derive(Debug, clap::Args)]
    pub struct Args {
        /// Path to the solver executable A.
        #[arg(short = 'A', long, env = "SAT_SOLVER_A")]
        pub solver_a: PathBuf,

        /// Path to the solver executable B.
        #[arg(short = 'B', long, env = "SAT_SOLVER_B")]
        pub solver_b: PathBuf,

        /// Name prefix to use when outputting messages by solver A.
        #[arg(long, default_value = "A")]
        pub name_a: String,

        /// Name prefix to use when outputting messages by solver B.
        #[arg(long, default_value = "B")]
        pub name_b: String,

        /// Tell sat-runner that the specified solver's result is to be trusted. I.e. don't pass
        /// its results to the other solver for verification. If this option is not given, both
        /// solvers' results are passed to the other one for verification.
        #[arg(long = "trust", value_enum)]
        pub trusted_solver: Option<Tag>,

        /// Start the random search from a specific seed to reproduce the sequence of tests.
        ///
        /// Note that the size of the test instance is an implicit parameter in generating the
        /// sets of assumptions. Thus varying the size while keeping the seed will result in
        /// different choices.
        #[arg(short, long, group = "entrypoint")]
        pub seed: Option<u64>,

        /// Instead of performing a a random search, check a given set of comma or white-space
        /// separated assumptions. Can be given multiple times to check different sets of
        /// assumptions.
        #[arg(
            short,
            long = "assume",
            group = "entrypoint",
            allow_hyphen_values = true,
            value_parser = parse_clause,
        )]
        pub assume: Vec<solver::Clause>,

        /// Enable verbose mode.
        #[arg(short, long)]
        verbose: bool,

        /// Enable even more verbose debug output.
        #[arg(long)]
        debug: bool,

        /// Arguments passed to the solver executables.
        pub solver_args: Vec<OsString>,
    }

    /// Returns a string slice with any balanced bracketing (and potentially separating ASCII
    /// whitespace) removed.
    fn strip_bracketing(mut s: &str) -> &str {
        loop {
            s = s.trim_matches(|c: char| c.is_ascii_whitespace());
            if s.len() < 2 {
                return s;
            }

            let mut cs = s.chars();
            let Some(first) = cs.next() else {
                return "";
            };
            let Some(last) = cs.next_back() else {
                return "";
            };

            match (first, last) {
                ('(', ')') | ('{', '}') | ('[', ']') => s = cs.as_str(),
                _ => return s,
            }
        }
    }

    fn parse_clause(s: &str) -> Result<solver::Clause, std::num::ParseIntError> {
        strip_bracketing(s)
            .split_terminator(|c: char| c.is_ascii_whitespace() || c == ',')
            .filter(|part| !part.is_empty())
            .map(|part| part.parse())
            .collect::<Result<_, _>>()
    }

    static ARGS_PARSED: Once = Once::new();
    static mut ARGS: MaybeUninit<Args> = MaybeUninit::uninit();

    pub fn args() -> &'static Args {
        ARGS_PARSED.call_once(|| {
            use clap::Parser;
            let args = ArgsParser::parse().args;
            unsafe {
                ARGS.write(args);
            }
        });

        // We know that initialization has completed. We only hand out un-mutable references, which
        // is safe.
        unsafe { ARGS.assume_init_ref() }
    }

    impl Args {
        pub fn solver_exe(&self, tag: Tag) -> &OsStr {
            tag.select(&self.solver_a, &self.solver_b).as_os_str()
        }

        /// Creates a new `std::process::Command` using the solver executable and solver arguments.
        pub fn solver_cmd(&self, tag: Tag) -> std::process::Command {
            let mut cmd = std::process::Command::new(self.solver_exe(tag));
            cmd.args(&self.solver_args);
            cmd
        }

        pub fn solver_name(&self, tag: Tag) -> &str {
            tag.select(&self.name_a, &self.name_b)
        }

        pub fn verbose(&self) -> bool {
            self.verbose || self.debug
        }

        pub fn debug(&self) -> bool {
            self.debug
        }
    }
}
mod solver {
    use std::{
        borrow::Borrow,
        fmt, io,
        mem::{self, MaybeUninit},
        process::{ExitStatus, Stdio},
        str::FromStr,
        time::Duration,
    };

    use color_eyre::{eyre::eyre, eyre::WrapErr, Help, Result};
    use either::Either;
    use tokio::{
        io::{AsyncBufReadExt, AsyncRead, AsyncWriteExt, BufReader, BufWriter},
        process,
        sync::{
            mpsc::{channel, Receiver, Sender},
            oneshot,
        },
        task::JoinHandle,
    };

    use crate::{cmd, Tag};

    pub type Lit = i64;
    pub type Clause = Vec<Lit>;
    pub type ClauseRef<'a> = &'a [Lit];

    pub enum Res {
        Sat,
        Unsat(Clause),
    }

    impl Res {
        pub fn is_sat(&self) -> bool {
            matches!(self, Res::Sat)
        }
    }

    impl fmt::Display for Res {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.pad(if self.is_sat() { "SAT" } else { "UNSAT" })
        }
    }

    enum ResultRelay {
        Open {
            relay_handle: JoinHandle<()>,
            result_output: Receiver<Res>,
            instance_size: Either<oneshot::Receiver<u64>, u64>,
        },
        Closed {
            result_output: Receiver<Res>,
            instance_size: Either<oneshot::Receiver<u64>, u64>,
        },
    }

    impl ResultRelay {
        async fn instance_size(&mut self) -> Option<u64> {
            let size_ref = match self {
                ResultRelay::Open { instance_size, .. } => instance_size,
                ResultRelay::Closed { instance_size, .. } => instance_size,
            };

            let recvd_size = match size_ref {
                Either::Left(recv) => recv.await,
                Either::Right(sz) => return Some(*sz),
            };

            if let Ok(sz) = recvd_size {
                *size_ref = Either::Right(sz);
                Some(sz)
            } else {
                self.close().await;
                None
            }
        }

        async fn recv(&mut self) -> Option<Res> {
            let recv_ref = match self {
                ResultRelay::Open { result_output, .. } => result_output,
                ResultRelay::Closed { result_output, .. } => result_output,
            };

            if let Some(res) = recv_ref.recv().await {
                Some(res)
            } else {
                self.close().await;
                None
            }
        }

        async fn close(&mut self) {
            // Turn our `self` reference into a reference to `MaybeUninit<Self>`.
            fn transmute_uninit<T>(x: &mut T) -> &mut MaybeUninit<T> {
                unsafe { mem::transmute(x) }
            }
            let self_uninit = transmute_uninit(self);

            // Take the current value out of self.
            let self_value = mem::replace(self_uninit, MaybeUninit::uninit());

            // Perform the actual close operation. This statment is not allowed to panic or return
            // because the `self` reference does not contain a valid value at this point.
            let (closed_value, relay_panic) = match unsafe { self_value.assume_init() } {
                ResultRelay::Open {
                    relay_handle,
                    result_output,
                    instance_size,
                } => {
                    let relay_res = relay_handle.await;
                    (
                        ResultRelay::Closed {
                            result_output,
                            instance_size,
                        },
                        relay_res
                            .err()
                            .filter(|e| e.is_panic())
                            .map(|e| e.into_panic()),
                    )
                }
                closed @ ResultRelay::Closed { .. } => (closed, None),
            };

            // Put a valid value back into `self`.
            _ = mem::replace(self_uninit, MaybeUninit::new(closed_value));

            // If the relay thread panicked, pass the panic on.
            if let Some(panic_payload) = relay_panic {
                std::panic::resume_unwind(panic_payload)
            }
        }
    }

    /// Represents the connection to a running solver process.
    pub struct Solver {
        tag: Tag,
        child_handle: process::Child,
        child_input: BufWriter<process::ChildStdin>,
        result_relay: ResultRelay,
    }

    impl Solver {
        pub fn name(&self) -> &'static str {
            cmd::args().solver_name(self.tag)
        }

        fn relay_closed_error(&self) -> color_eyre::Report {
            eyre!("Solver {} closed communcation unexpectedly", self.name())
        }

        pub async fn instance_size(&mut self) -> Result<u64> {
            self.result_relay
                .instance_size()
                .await
                .ok_or_else(|| self.relay_closed_error())
                .wrap_err("Instance size not communicated")
        }

        async fn write_challenge<T: Borrow<Lit> + ToString>(
            &mut self,
            assumptions: impl IntoIterator<Item = T>,
        ) -> io::Result<()> {
            self.child_input.write_u8(b's').await?;
            for lit in assumptions {
                self.child_input.write_u8(b' ').await?;
                self.child_input
                    .write_all(lit.to_string().as_bytes())
                    .await?;
            }
            self.child_input.write_u8(b'\n').await?;
            self.child_input.flush().await?;
            Ok(())
        }

        async fn read_event(&mut self) -> Result<Res> {
            self.result_relay
                .recv()
                .await
                .ok_or_else(|| self.relay_closed_error())
        }

        pub async fn run_challenge<T: Borrow<Lit> + ToString>(
            &mut self,
            assumptions: impl IntoIterator<Item = T>,
        ) -> Result<Res> {
            // Send the challenge.
            self.write_challenge(assumptions)
                .await
                .wrap_err("failed to send challenge to solver process")?;

            // Wait for the answer.
            self.read_event()
                .await
                .wrap_err("no answer to solve request")
        }

        pub async fn terminate(mut self) -> Result<ExitStatus> {
            // We destroy `self` by moving parts out. Extract the name for a later error message as
            // long as it's this simple.
            let name = self.name();

            // Terminate the child process and close the result relay.
            let (exit, _) = tokio::join!(
                Self::terminate_impl(self.name(), self.child_handle, self.child_input),
                self.result_relay.close()
            );

            // Return the (potentially failed) process exit information.
            exit.wrap_err_with(|| format!("Failed to terminate the solver process {name}"))
        }

        async fn terminate_impl<ChildInput>(
            name: &str,
            mut child: process::Child,
            input: ChildInput,
        ) -> Result<ExitStatus> {
            // There are multiple ways we could implement termination.
            //
            //     (1)  send SIGKILL using `Child::kill`
            //     (2)  send `q` over stdin and `CHILD::wait`
            //     (3)  close STDIN and expect the child to terminate automatically.
            //
            // Below we use a combination of (1) and (3): We close STDIN and give the child process
            // a grace period of `TERMINATION_TIMEOUT` after which we send SIGKILL.

            // To avoid unnecessary verbose messages in case the solver already terminated we check
            // with the non-blocking `child_handle.try_wait()` first.
            if let Some(exit) = child.try_wait()? {
                return Ok(exit);
            }

            const TERMINATION_TIMEOUT: Duration = Duration::from_secs(1);

            let verbose = cmd::args().verbose();
            if verbose {
                eprintln!("terminating solver {name}")
            }

            // Send EOF by dropping the input handle.
            drop(input);

            // Give the child a timeout of 1s to terminate gracefully.
            tokio::select! {
                r = child.wait() => return Ok(r?),
                _ = tokio::time::sleep(TERMINATION_TIMEOUT) => {}
            };

            if verbose {
                eprintln!(
                    "solver took longer than {}s to terminate ... killing",
                    TERMINATION_TIMEOUT.as_secs_f64()
                );
            }

            // We "want" the exit code even if we kill the process but `Child::kill` does not
            // return any information. Therefore, we use a combination of `Child::start_kill` and
            // `Child::wait`.
            child.start_kill()?;
            Ok(child.wait().await?)
        }
    }

    fn annotate_with_solver_launch(tag: Tag) -> impl Fn(color_eyre::Report) -> color_eyre::Report {
        move |err| {
            err.note("solver process spawned as")
                .note("")
                .note(format!("  {:?}", cmd::args().solver_cmd(tag)))
                .note("")
        }
    }

    pub fn launch(tag: Tag) -> Result<Solver> {
        let args = cmd::args();
        let cmd = args.solver_cmd(tag);
        if args.verbose() {
            eprintln!("launching solver {}:  {:?}", args.solver_name(tag), cmd);
        }

        // Convert to a tokio command before executing.
        let mut child_handle = tokio::process::Command::from(cmd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .wrap_err_with(|| format!("Failed to launch solver {}", cmd::args().solver_name(tag)))
            .map_err(annotate_with_solver_launch(tag))?;

        // Create channels for interaction.
        let (result_send, result_output) = channel(1);
        let (instance_size_send, instance_size_recv) = oneshot::channel();

        // Spawn the relaying thread.
        let relay_handle =
            start_output_relay(tag, &mut child_handle, result_send, instance_size_send);

        let child_input = child_handle.stdin.take().expect("input not piped");
        let child_input = BufWriter::new(child_input);

        // Return the solver instance.
        Ok(Solver {
            tag,
            child_handle,
            child_input,
            result_relay: ResultRelay::Open {
                relay_handle,
                result_output,
                instance_size: Either::Left(instance_size_recv),
            },
        })
    }

    fn start_output_relay(
        tag: Tag,
        process: &mut process::Child,
        result_output: Sender<Res>,
        size_output: oneshot::Sender<u64>,
    ) -> JoinHandle<()> {
        let out = process.stdout.take().expect("output not piped");
        let err = process.stderr.take().expect("output not piped");
        tokio::spawn(async move {
            let name = cmd::args().solver_name(tag);
            _ = tokio::join!(
                // stdout is parsed.
                ProcessingRelay::new(result_output, size_output).relay_output(&name, out),
                // stderr is not parsed but simply relayed.
                UnfilteredRelay.relay_output(&name, err),
            )
        })
    }

    #[async_trait::async_trait]
    trait OutputRelay {
        /// Process the line and decide wether it should be shown to the user.
        async fn process_line(&mut self, line: &[u8]) -> Result<bool, &'static str>;

        async fn relay_output(
            mut self,
            name: &str,
            reader: impl AsyncRead + Send + std::marker::Unpin,
        ) where
            Self: Sized,
        {
            use std::io::Write;

            let mut buf = Vec::new();
            let mut buf_reader = BufReader::new(reader);

            loop {
                buf.clear();
                let n = buf_reader
                    .read_until(b'\n', &mut buf)
                    .await
                    .wrap_err_with(|| format!("communication with solver {name} failed"))
                    .unwrap();

                // If we've reached EOF simply return.
                if n == 0 {
                    return;
                }

                // Process the read string.
                //
                // Strip trailing whitespace.
                let Some(end_idx) = buf.iter().rposition(|&b| !b.is_ascii_whitespace()) else {
                    continue;
                };
                let line = &buf[..=end_idx];

                // Parse the line and decide wether to output.
                let res = self.process_line(line).await;
                if res == Ok(true) || res.is_err() {
                    let mut out = io::stderr().lock();
                    write!(out, "{}> ", name)
                        .and_then(|_| out.write_all(line))
                        .and_then(|_| writeln!(out))
                        .and_then(|_| {
                            if let Err(warning) = res {
                                writeln!(out, "* warning: {warning}")
                            } else {
                                Ok(())
                            }
                        })
                        .wrap_err_with(|| format!("failed to relay message from solver {name}"))
                        .unwrap();
                }
            }
        }
    }

    struct UnfilteredRelay;

    #[async_trait::async_trait]
    impl OutputRelay for UnfilteredRelay {
        async fn process_line(&mut self, _line: &[u8]) -> Result<bool, &'static str> {
            Ok(true)
        }
    }

    struct ProcessingRelay {
        result_output: Sender<Res>,
        size_output: Option<oneshot::Sender<u64>>,
    }

    impl ProcessingRelay {
        fn new(result_output: Sender<Res>, size_output: oneshot::Sender<u64>) -> Self {
            ProcessingRelay {
                result_output,
                size_output: Some(size_output),
            }
        }
    }

    #[async_trait::async_trait]
    impl OutputRelay for ProcessingRelay {
        async fn process_line(&mut self, line: &[u8]) -> Result<bool, &'static str> {
            fn parse_slice<T: FromStr>(slice: &[u8]) -> Option<T> {
                std::str::from_utf8(slice).ok().and_then(|s| s.parse().ok())
            }

            let mut components = line
                .split(u8::is_ascii_whitespace)
                .filter(|slice| !slice.is_empty());

            match components.next() {
                // Comment/empty line. Skip
                Some(b"c") | None => return Ok(false),

                // SIZE answer.
                //
                // TODO: we might want to support multiple modes in our command interface to
                // silence this and similar warnings or to turn them into hard errors.
                Some(b"n") => {
                    // We expect to parse exactly one number.
                    let Some(n) = components.next() else {
                        return Err("not interpreted as a SIZE answer (missing number)");
                    };
                    let None = components.next() else {
                        return Err("not interpreted as a SIZE answer (additional content)");
                    };
                    let Some(n) = parse_slice::<u64>(n) else {
                        return Err("not interpreted as a SIZE answer (can't parse number)");
                    };
                    let Some(size_chan) = self.size_output.take() else {
                        return Err("not interpreted as a SIZE answer (size already received)");
                    };
                    _ = size_chan.send(n);
                }

                // SAT answer.
                Some(b"S") => {
                    let None = components.next() else {
                        return Err("not interpreted as SAT answer (additional content)");
                    };
                    _ = self.result_output.send(Res::Sat).await;
                }

                // UNSAT answer.
                Some(b"U") => {
                    let Some(conflict) = components
                        .map(parse_slice)
                        .collect::<Option<Clause>>()
                    else {
                        return Err("not interpreted as UNSAT answer (failed to parse conflict set)");
                    };

                    _ = self.result_output.send(Res::Unsat(conflict)).await;
                }

                // not a recognized command, print.
                Some(_) => return Ok(true),
            }

            // Print commands if we are in verbose mode.
            Ok(cmd::args().verbose())
        }
    }
}

mod coordinator {
    use std::fmt;

    use color_eyre::{eyre::eyre, eyre::WrapErr, Result};
    use rand::{
        distributions::{self, Distribution},
        rngs::SmallRng,
        Rng, SeedableRng,
    };
    use sorted_vec::SortedSet;
    use these::These;

    use crate::{
        cmd,
        solver::{Clause, ClauseRef, Lit, Res, Solver},
        util::{DigitCount, Joined, MergeErrors, MergeThese},
    };

    /// Describes a detected failure.
    #[derive(Debug)]
    pub struct Failure {
        /// Set of assumptions which lead to the error.
        pub assumptions: Clause,
        /// Describes the failure that occured.
        pub mode: FailureMode,
        /// Each entry is printed as a way to rerun the solver.
        pub rerun_options: Vec<String>,
    }

    /// Describes the kind of detected failure.
    #[derive(Debug)]
    pub enum FailureMode {
        /// Solver A returned a SAT result while solver B returned an UNSAT result and the
        /// contained conflict clause.
        SatUnsat { b_conflict: Clause },

        /// Solver B returned a SAT result while solver A returned an UNSAT result and the
        /// contained conflict clause.
        UnsatSat { a_conflict: Clause },

        /// The conflict clause returned by `origin` is not a conflict clause at all. The
        /// `These::This` variant indicates a faulty conflict clause from solver A, `These::That`
        /// indicates a faulty conflict clause by solver B, and `These::Both` indicates that no
        /// solver produced a conflict clause which is valid in the other one.
        NoConflict { clauses: These<Clause, Clause> },
    }

    impl Failure {
        pub fn new(assumptions: impl Into<Clause>, mode: FailureMode) -> Self {
            Failure {
                assumptions: assumptions.into(),
                mode,
                rerun_options: Vec::new(),
            }
        }

        fn unsat_sat(assumptions: impl Into<Clause>, a_conflict: Clause) -> Self {
            Self::new(assumptions, FailureMode::UnsatSat { a_conflict })
        }

        fn sat_unsat(assumptions: impl Into<Clause>, b_conflict: Clause) -> Self {
            Self::new(assumptions, FailureMode::SatUnsat { b_conflict })
        }

        fn no_conflict(assumptions: impl Into<Clause>, conflicts: These<Clause, Clause>) -> Self {
            Self::new(assumptions, FailureMode::NoConflict { clauses: conflicts })
        }

        pub fn push_rerun(&mut self, rerun: impl Into<String>) {
            self.rerun_options.push(rerun.into())
        }
    }

    impl fmt::Display for Failure {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            fn write_border(f: &mut fmt::Formatter, title: &str) -> fmt::Result {
                writeln!(f, "{title:-^79}")
            }

            write_border(f, " FAILED ")?;
            writeln!(f, "assumptions: {}", self.assumptions.joined())?;

            let args = cmd::args();
            match &self.mode {
                FailureMode::SatUnsat { b_conflict } => {
                    writeln!(f)?;
                    writeln!(f, "Solver {} reports   SAT", args.name_a)?;
                    writeln!(f, "Solver {} reports UNSAT", args.name_b)?;
                    writeln!(f, "  conflict clause: {}", b_conflict.joined())?;
                }
                FailureMode::UnsatSat { a_conflict } => {
                    writeln!(f)?;
                    writeln!(f, "Solver {} reports UNSAT", args.name_a)?;
                    writeln!(f, "  conflict clause: {}", a_conflict.joined())?;
                    writeln!(f, "Solver {} reports   SAT", args.name_b)?;
                }
                FailureMode::NoConflict { clauses } => {
                    if let Some(a_confl) = clauses.as_ref().here() {
                        writeln!(f)?;
                        writeln!(f, "Bad conflict reported by solver {}", args.name_a)?;
                        writeln!(f, "  conflict clause: {}", a_confl.joined())?;
                    }
                    if let Some(b_confl) = clauses.as_ref().there() {
                        writeln!(f)?;
                        writeln!(f, "Bad conflict reported by solver {}", args.name_b)?;
                        writeln!(f, "  conflict clause: {}", b_confl.joined())?;
                    }
                }
            }

            writeln!(f)?;
            write!(f, "* use  --assume=")?;
            if self.assumptions.is_empty() {
                write!(f, "\"\"")?;
            } else {
                write!(f, "{}", self.assumptions.joined_by(','))?;
            }
            writeln!(f, "  to rerun this test case")?;
            for rerun in &self.rerun_options {
                writeln!(f, "* {rerun}")?;
            }

            write_border(f, "")?;
            Ok(())
        }
    }

    pub type OrderedClause = SortedSet<Lit>;

    pub struct Coordinator<'s> {
        challenge_index: u32,
        solver_a: &'s mut Solver,
        solver_b: &'s mut Solver,
    }

    impl<'s> Coordinator<'s> {
        pub fn new(solver_a: &'s mut Solver, solver_b: &'s mut Solver) -> Self {
            Coordinator {
                challenge_index: 0,
                solver_a,
                solver_b,
            }
        }

        pub async fn size(&mut self) -> Result<u64> {
            let (sz_a, sz_b) =
                tokio::join!(self.solver_a.instance_size(), self.solver_b.instance_size())
                    .merge_errors()?;

            if sz_a == sz_b {
                Ok(sz_a)
            } else {
                Err(eyre!(
                    concat!(
                        "Solvers reported conflicting sizes:\n",
                        "  {}: {:>width$}\n",
                        "  {}: {:>width$}"
                    ),
                    self.solver_a.name(),
                    sz_a,
                    self.solver_b.name(),
                    sz_b,
                    width = sz_a.max(sz_b).count_digits(),
                ))
            }
        }

        pub async fn run_challenge(&mut self, challenge: ClauseRef<'_>) -> Result<Option<Failure>> {
            Ok(self.run_challenge_impl(challenge).await?.err())
        }

        pub async fn run_challenge_impl(
            &mut self,
            challenge: ClauseRef<'_>,
        ) -> Result<Result<bool, Failure>> {
            self.challenge_index += 1;

            print!(
                concat_lines!(
                    //
                    "{}{:=>79}",
                    "assumptions: {}"
                ),
                if self.challenge_index > 1 { "\n" } else { "" },
                format!(" {}", self.challenge_index),
                challenge.joined()
            );

            let res = tokio::join!(
                self.solver_a.run_challenge(challenge),
                self.solver_b.run_challenge(challenge)
            )
            .merge_errors()?;

            match res {
                (Res::Sat, Res::Sat) => Ok(Ok(true)),
                (Res::Unsat(a), Res::Unsat(b)) => {
                    if let Some(fail) = self.verify_conflicts(challenge, a, b).await? {
                        Ok(Err(fail))
                    } else {
                        Ok(Ok(false))
                    }
                }
                (Res::Sat, Res::Unsat(confl_b)) => Ok(Err(Failure::sat_unsat(challenge, confl_b))),
                (Res::Unsat(confl_a), Res::Sat) => Ok(Err(Failure::unsat_sat(challenge, confl_a))),
            }
        }

        async fn verify_conflicts(
            &mut self,
            challenge: ClauseRef<'_>,
            confl_a: impl Into<OrderedClause>,
            confl_b: impl Into<OrderedClause>,
        ) -> Result<Option<Failure>> {
            let args = cmd::args();
            let confl_a = confl_a.into();
            let confl_b = confl_b.into();

            if confl_a == confl_b {
                print!(
                    concat_lines!(
                        /**/ "",
                        "reported conflict clauses are the same",
                        "  {}"
                    ),
                    confl_a.joined()
                );
                return Ok(None);
            }

            print!(
                concat_lines!(
                    /**/ "",
                    "reported conflict clauses",
                    "  {}: {}",
                    "  {}: {}"
                ),
                args.name_a,
                confl_a.joined(),
                args.name_b,
                confl_b.joined()
            );

            fn bad_conflict(res: Res, clause: Clause) -> Option<Clause> {
                if res.is_sat() {
                    Some(clause)
                } else {
                    None
                }
            }

            let failure = if let Some(trusted) = args.trusted_solver {
                print!(
                    concat_lines!(
                        //
                        "",
                        "verifying {}'s conflict using {}"
                    ),
                    args.solver_name(trusted.other()),
                    args.solver_name(trusted)
                );

                let trusted_solver = trusted.select(&mut self.solver_a, &mut self.solver_b);
                let verified_conflict = trusted.select(confl_b, confl_a);
                let res = trusted_solver
                    .run_challenge(verified_conflict.iter())
                    .await?;
                bad_conflict(res, verified_conflict.into_vec()).map(|c| {
                    Failure::no_conflict(
                        challenge,
                        trusted.select::<fn(Clause) -> These<Clause, Clause>>(
                            These::That,
                            These::This,
                        )(c),
                    )
                })
            } else {
                print!(concat_lines!(
                    /**/ "",
                    "verifying clauses against each other"
                ));

                let (res_a, res_b) = tokio::join!(
                    self.solver_a.run_challenge(confl_b.iter()),
                    self.solver_b.run_challenge(confl_a.iter())
                )
                .merge_errors()?;

                let failed_a = bad_conflict(res_b, confl_a.into_vec());
                let failed_b = bad_conflict(res_a, confl_b.into_vec());
                These::merge_these(failed_a, failed_b)
                    .map(|bad_conflicts| Failure::no_conflict(challenge, bad_conflicts))
            };

            if failure.is_none() {
                println!(
                    "{} successfully verified!",
                    if args.trusted_solver.is_some() {
                        "Conflict"
                    } else {
                        "Conflicts"
                    }
                );
            }

            Ok(failure)
        }

        async fn search_once(
            &mut self,
            searcher: &mut impl AssumptionSearch,
        ) -> Result<Option<Failure>> {
            let challenge = searcher.next_challenge();
            match self.run_challenge_impl(challenge.as_ref()).await? {
                Ok(true) => {
                    searcher.register_challenge_sat(challenge);
                    Ok(None)
                }
                Ok(false) => {
                    searcher.register_challenge_unsat(challenge);
                    Ok(None)
                }
                Err(fail) => Ok(Some(fail)),
            }
        }

        pub async fn run_search(
            &mut self,
            searcher: &mut impl AssumptionSearch,
        ) -> Result<Failure> {
            loop {
                if let Some(fail) = self.search_once(searcher).await? {
                    return Ok(fail);
                }
            }
        }
    }

    pub trait AssumptionSearch {
        type Challenge: AsRef<[Lit]>;

        fn next_challenge(&mut self) -> Self::Challenge;
        fn register_challenge_sat(&mut self, _challenge: Self::Challenge) {}
        fn register_challenge_unsat(&mut self, _challenge: Self::Challenge) {}
    }

    pub struct RandomSearch {
        rng: SmallRng,
        size: u64,
        size_dist: distributions::Uniform<u64>,
    }

    impl RandomSearch {
        pub fn new(seed: Option<u64>, size: u64) -> Result<(Self, u64)> {
            let seed = if let Some(seed) = seed {
                seed
            } else {
                let mut bytes = [0; 8];
                getrandom::getrandom(&mut bytes).wrap_err("failed to generate initial seed")?;
                u64::from_ne_bytes(bytes)
            };

            // Overall we want to have clauses containing at most 2/3 of all possible variables. If
            // the max clause size resolves to three or less we simply take the complete size.
            let max_clause_size = size / 3 * 2;
            let max_clause_size = if max_clause_size <= 3 {
                size
            } else {
                max_clause_size
            };
            let random_search = RandomSearch {
                size,
                size_dist: distributions::Uniform::new_inclusive(0, max_clause_size),
                rng: SmallRng::seed_from_u64(seed),
            };

            Ok((random_search, seed))
        }
    }

    impl AssumptionSearch for RandomSearch {
        type Challenge = Clause;

        fn next_challenge(&mut self) -> Self::Challenge {
            // Decide how big the clause should be.
            let n = self.size_dist.sample(&mut self.rng);
            // Draw `n` variables for the clause.
            let vars = rand::seq::index::sample(&mut self.rng, self.size as usize, n as usize);
            // Turn the variables into literals.
            vars.into_iter()
                .map(|v| {
                    if self.rng.gen() {
                        v as i64 + 1
                    } else {
                        -(v as i64 + 1)
                    }
                })
                .collect()
        }
    }
}

mod util {
    use std::fmt::{self, Debug, Display};

    use color_eyre::{eyre::eyre, owo_colors::OwoColorize, Result, Section, SectionExt};
    use num_traits::AsPrimitive;
    use these::These;

    use crate::cmd;

    pub use joined::Joined;

    pub trait DigitCount {
        /// Counts the numbers of digits. The sign is not included for negative numbers.
        ///
        /// The return type is `usize` becasue this is the type expected by `format!` and friends
        /// for field widhts.
        fn count_digits(&self) -> usize;
    }

    impl<T: AsPrimitive<f64>> DigitCount for T {
        fn count_digits(&self) -> usize {
            self.as_().abs().log10() as usize + 1
        }
    }

    pub trait MergeErrors: Sized {
        type Merged;

        fn merge_errors(self) -> Result<Self::Merged> {
            self.merge_errors_msg("Multiple errors occured.")
        }

        fn merge_errors_msg<M: Display + Debug + Send + Sync + 'static>(
            self,
            msg: M,
        ) -> Result<Self::Merged> {
            self.merge_errors_with(|| msg)
        }

        fn merge_errors_with<M: Display + Debug + Send + Sync + 'static>(
            self,
            msg: impl FnOnce() -> M,
        ) -> Result<Self::Merged>;
    }

    impl<T, U> MergeErrors for (Result<T>, Result<U>) {
        type Merged = (T, U);

        fn merge_errors_with<M: Display + Debug + Send + Sync + 'static>(
            self,
            msg: impl FnOnce() -> M,
        ) -> Result<Self::Merged> {
            match self {
                (Ok(t), Ok(u)) => Ok((t, u)),
                (Err(e1), Ok(_)) => Err(e1),
                (Ok(_), Err(e2)) => Err(e2),
                (Err(e1), Err(e2)) => {
                    let args = cmd::args();
                    Err(eyre!(msg())
                        .section(
                            format!("{:?}", e1)
                                .header(ErrorSectionHeader(format!("Solver {}", args.name_a))),
                        )
                        .section(
                            format!("{:?}", e2)
                                .header(ErrorSectionHeader(format!("Solver {}", args.name_b))),
                        ))
                }
            }
        }
    }

    pub trait MergeThese<This, That>: Sized {
        fn merge_these(this: Option<This>, that: Option<That>) -> Option<Self>;
    }

    impl<This, That> MergeThese<This, That> for These<This, That> {
        fn merge_these(this: Option<This>, that: Option<That>) -> Option<Self> {
            Some(match (this, that) {
                (Some(this), Some(that)) => These::Both(this, that),
                (Some(this), None) => These::This(this),
                (None, Some(that)) => These::That(that),
                (None, None) => return None,
            })
        }
    }

    struct ErrorSectionHeader<H>(H);

    impl<H: Display> Display for ErrorSectionHeader<H> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.bright_red().fmt(f)
        }
    }

    pub mod joined {
        use std::{fmt, ops::Deref};

        pub trait Joined<Item>: Deref<Target = [Item]> {
            fn joined_by<'a, Sep>(&'a self, sep: Sep) -> WriteJoined<'a, Item, Sep> {
                WriteJoined::new(self, sep)
            }

            fn joined<'a>(&'a self) -> WriteJoined<'a, Item, char> {
                self.joined_by(' ')
            }
        }

        impl<T: Deref<Target = [Item]>, Item> Joined<Item> for T {}

        pub struct WriteJoined<'a, T, Sep>(&'a [T], Sep);

        impl<'a, T, Sep> WriteJoined<'a, T, Sep> {
            pub fn new(values: &'a [T], sep: Sep) -> Self {
                WriteJoined(values, sep)
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
}

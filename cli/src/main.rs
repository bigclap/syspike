//! CLI entry point exposing training, evaluation, inference, and profiling utilities.

mod commands;
mod config;
mod telemetry;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use commands::{run_eval, run_infer, run_profile, run_train};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:?}");
        std::process::exit(1);
    }
}

#[derive(Parser)]
#[command(
    name = "syspike",
    version,
    about = "syspike reasoning workspace CLI",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the offline decoder trainer and persist a checkpoint summary.
    Train {
        /// Optional TOML configuration file.
        #[arg(short, long, value_name = "PATH")]
        config: Option<PathBuf>,
    },
    /// Produce an evaluation report against a dataset using the offline trainer.
    Eval {
        /// Optional TOML configuration file.
        #[arg(short, long, value_name = "PATH")]
        config: Option<PathBuf>,
    },
    /// Execute the XOR inference demo.
    Infer {
        /// Optional TOML configuration file.
        #[arg(short, long, value_name = "PATH")]
        config: Option<PathBuf>,
    },
    /// Collect a CPU flamegraph while running the XOR demo.
    Profile {
        /// Optional TOML configuration file.
        #[arg(short, long, value_name = "PATH")]
        config: Option<PathBuf>,
    },
}

fn run() -> Result<()> {
    let cli = Cli::parse();
    telemetry::init_telemetry();
    match cli.command {
        Command::Train { config } => run_train(config),
        Command::Eval { config } => run_eval(config),
        Command::Infer { config } => run_infer(config),
        Command::Profile { config } => run_profile(config),
    }
}

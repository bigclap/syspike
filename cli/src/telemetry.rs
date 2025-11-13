use std::{fs::File, path::Path, sync::Mutex};

use once_cell::sync::Lazy;
use pprof::{ProfilerGuard, ProfilerGuardBuilder};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

pub static PROFILER: Lazy<Mutex<Option<ProfilerGuard<'static>>>> = Lazy::new(Default::default);

/// Initialise tracing output and start a CPU profiler.
pub fn init_telemetry() {
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = fmt::layer().with_target(false);
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .init();

    let guard = ProfilerGuardBuilder::default()
        .frequency(1000)
        .blocklist(&["libc", "libpthread", "libgcc", "libm"])
        .build()
        .ok();
    *PROFILER.lock().unwrap() = guard;
}

/// Persist the collected CPU profile if profiling was active.
pub fn write_profile(guard: ProfilerGuard<'_>, output_path: impl AsRef<Path>) {
    if let Ok(report) = guard.report().build() {
        if let Ok(mut file) = File::create(output_path) {
            let _ = report.flamegraph(&mut file);
        }
    }
}

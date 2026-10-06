//! Device profiling and automatic placement.
//!
//! The planner knows nothing about specific hardware: it only uses what plugins report
//! (`device_info`) plus a short matmul micro-benchmark run through the plugin itself.

use crate::{Device, Tensor};
use pytorches_plugin_abi as abi;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Fraction of reported free memory the planner is willing to plan against.
const HEADROOM: f64 = 0.85;

static CALIBRATION: Mutex<Option<HashMap<String, f64>>> = Mutex::new(None);

/// Measured matmul throughput of `device` in GFLOP/s (cached per process).
pub fn calibrate(device: &Device) -> f64 {
    let key = device.to_string();
    if let Some(v) = CALIBRATION.lock().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return *v;
    }
    // Small problem on CPU so calibration stays quick; GPUs get a size that keeps them busy.
    let n = if device.info().kind == abi::KIND_CPU { 192 } else { 2048 };
    let a = Tensor::randn_on(&[n, n], 1, device);
    let b = Tensor::randn_on(&[n, n], 2, device);
    let flop = 2.0 * (n as f64).powi(3);

    // Runs matmuls back to back for at least `min` and returns the achieved GFLOP/s.
    let run_for = |min: Duration| -> f64 {
        let start = Instant::now();
        let mut iters = 0u32;
        loop {
            let _ = a.matmul(&b);
            iters += 1;
            if iters >= 2 && start.elapsed() >= min {
                device.synchronize(); // include the queued work in the measurement
                if start.elapsed() >= min {
                    break;
                }
            }
        }
        flop * iters as f64 / start.elapsed().as_secs_f64().max(1e-9) / 1e9
    };

    // Warm up first: includes kernel JIT and lets power-managed GPUs leave their idle clocks,
    // which otherwise makes a short benchmark read several times too slow.
    run_for(Duration::from_millis(250));
    let gflops = (0..3).map(|_| run_for(Duration::from_millis(60))).fold(0.0, f64::max);
    CALIBRATION.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, gflops);
    gflops
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub device: Device,
    pub name: String,
    pub kind: u32,
    pub total_memory: Option<u64>,
    pub free_memory: Option<u64>,
    pub gflops: f64,
    /// Does the requirement fit in this device's (headroom-adjusted) free memory?
    /// Devices that don't report memory are assumed to fit.
    pub fits: bool,
}

#[derive(Clone, Debug)]
pub struct Plan {
    pub required_bytes: u64,
    pub chosen: Device,
    /// Fastest first.
    pub candidates: Vec<Candidate>,
    pub reason: String,
    /// True if no device had enough reported free memory.
    pub may_oom: bool,
}

fn fmt_bytes(b: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    if b as f64 >= GIB / 4.0 { format!("{:.1} GiB", b as f64 / GIB) } else { format!("{:.0} MiB", b as f64 / (1u64 << 20) as f64) }
}

/// Picks the fastest device whose free memory fits `required_bytes`; if none fits, the one with
/// the most free memory (and says so).
pub fn plan(required_bytes: u64) -> Plan {
    let mut candidates: Vec<Candidate> = Device::all()
        .into_iter()
        .map(|device| {
            let info = device.info();
            let gflops = calibrate(&device);
            let fits = match info.free_memory {
                Some(free) => (free as f64) * HEADROOM >= required_bytes as f64,
                None => true,
            };
            Candidate {
                name: info.name,
                kind: info.kind,
                total_memory: info.total_memory,
                free_memory: info.free_memory,
                gflops,
                fits,
                device,
            }
        })
        .collect();
    candidates.sort_by(|a, b| b.gflops.partial_cmp(&a.gflops).unwrap());
    assert!(!candidates.is_empty(), "no devices available");

    let fastest = candidates[0].clone();
    let need = fmt_bytes(required_bytes);
    let (chosen, reason, may_oom) = match candidates.iter().find(|c| c.fits) {
        Some(c) if c.device == fastest.device => (
            c.clone(),
            format!("needs {need}; fits on the fastest device ({}), so use it", c.device),
            false,
        ),
        Some(c) => (
            c.clone(),
            format!(
                "needs {need}; the fastest device ({}) has only {} free, so fall back to {} ({} free)",
                fastest.device,
                fastest.free_memory.map(fmt_bytes).unwrap_or_else(|| "unknown".into()),
                c.device,
                c.free_memory.map(fmt_bytes).unwrap_or_else(|| "unknown".into()),
            ),
            false,
        ),
        None => {
            let c = candidates
                .iter()
                .max_by_key(|c| c.free_memory.unwrap_or(0))
                .unwrap()
                .clone();
            (
                c.clone(),
                format!("needs {need}, which fits on no device; using {} (most free memory) and it may run out of memory", c.device),
                true,
            )
        }
    };
    Plan { required_bytes, chosen: chosen.device, candidates, reason, may_oom }
}

pub fn format_bytes(b: u64) -> String {
    fmt_bytes(b)
}

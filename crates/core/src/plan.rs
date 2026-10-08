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
pub(crate) const HEADROOM: f64 = 0.85;

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
            // Bound the queued work. The loop is paced by host time, and with a fast (cached)
            // allocator a host can enqueue minutes of GPU work in a fraction of a second, so wait
            // every few calls. Cheap: a matmul big enough to keep a GPU busy takes milliseconds.
            if iters % 4 == 0 {
                device.synchronize();
            }
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

static BANDWIDTH: Mutex<Option<HashMap<String, f64>>> = Mutex::new(None);

/// Measured device memory bandwidth of `device` in GB/s (cached per process): an elementwise add
/// on tensors too big to cache, which reads two and writes one.
pub fn calibrate_bandwidth(device: &Device) -> f64 {
    let key = device.to_string();
    if let Some(v) = BANDWIDTH.lock().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return *v;
    }
    let n = 16 << 20; // 64 MiB per tensor
    let a = Tensor::randn_on(&[n], 1, device);
    let b = Tensor::randn_on(&[n], 2, device);
    let once = || {
        let t = Instant::now();
        let _ = a.add(&b);
        device.synchronize();
        t.elapsed().as_secs_f64()
    };
    once();
    once();
    let best = (0..5).map(|_| once()).fold(f64::INFINITY, f64::min);
    let gbps = 3.0 * n as f64 * 4.0 / best.max(1e-9) / 1e9;
    BANDWIDTH.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, gbps);
    gbps
}

/// Measured cost of moving a tensor from one device to another through `Tensor::to`, which is the
/// path a placed model's activations really take (staged through host memory today).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Link {
    pub bytes_per_sec: f64,
    /// Fixed cost of one transfer, dominated by synchronization and launch overhead.
    pub latency_secs: f64,
}

impl Link {
    pub fn secs(&self, bytes: u64) -> f64 {
        self.latency_secs + bytes as f64 / self.bytes_per_sec
    }
}

static LINKS: Mutex<Option<HashMap<String, Link>>> = Mutex::new(None);

/// Measures `from -> to` (cached per process). Bandwidth comes from a large transfer, latency
/// from a tiny one; each is the best of a few runs after a warm-up.
pub fn calibrate_link(from: &Device, to: &Device) -> Link {
    let key = format!("{from}->{to}");
    if let Some(l) = LINKS.lock().unwrap().as_ref().and_then(|m| m.get(&key)) {
        return *l;
    }
    let time_one = |n: usize| -> f64 {
        let src = Tensor::randn_on(&[n], 3, from);
        from.synchronize();
        let once = || {
            let t = Instant::now();
            let moved = src.to(to);
            to.synchronize();
            drop(moved);
            t.elapsed().as_secs_f64()
        };
        once(); // warm-up: allocator, driver, page faults
        (0..4).map(|_| once()).fold(f64::INFINITY, f64::min)
    };
    let big = 8 << 20; // 32 MiB of f32
    let small = 256;
    let (t_big, t_small) = (time_one(big), time_one(small));
    let latency_secs = t_small;
    let bytes_per_sec = (big as f64 * 4.0) / (t_big - latency_secs).max(1e-9);
    let link = Link { bytes_per_sec, latency_secs };
    LINKS.lock().unwrap().get_or_insert_with(HashMap::new).insert(key, link);
    link
}

#[derive(Clone, Debug)]
pub struct Candidate {
    pub device: Device,
    pub name: String,
    pub kind: u32,
    pub total_memory: Option<u64>,
    /// Free memory the planner plans against. For a device that shares system RAM this is capped
    /// at the host's free memory; see `effective_free`.
    pub free_memory: Option<u64>,
    /// The device allocates from system RAM.
    pub shared_host_memory: bool,
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
    /// Non-fatal problems with the chosen placement, e.g. it may push the host into paging.
    pub warnings: Vec<String>,
}

/// A plan on host-resident memory that needs more than this fraction of the host's free memory
/// is likely to contend with the OS and other programs, and may page.
const PAGING_WARN_FRACTION: f64 = 0.5;

/// Free memory to plan against. A device that allocates from system RAM can never have more free
/// than the host does, whatever its plugin can see (OpenCL has no free-memory query, so the Arc
/// plugin reports `total - tracked`, which ignores every other process).
pub(crate) fn effective_free(info: &crate::DeviceInfo, host_free: Option<u64>) -> Option<u64> {
    match (info.free_memory, host_free) {
        (Some(dev), Some(host)) if info.shared_host_memory => Some(dev.min(host)),
        (None, Some(host)) if info.shared_host_memory => Some(host),
        (free, _) => free,
    }
}

/// Warning text if putting `required` bytes on a host-resident device could page.
fn paging_warning(device: &Device, required: u64, host_free: Option<u64>) -> Option<String> {
    let host_free = host_free?;
    (required as f64 > host_free as f64 * PAGING_WARN_FRACTION).then(|| {
        format!(
            "{device} uses system RAM and this needs {} of the {} the host has free; \
             other programs may push it into paging, which is far slower than either GPU",
            fmt_bytes(required),
            fmt_bytes(host_free)
        )
    })
}

fn fmt_bytes(b: u64) -> String {
    const GIB: f64 = (1u64 << 30) as f64;
    if b as f64 >= GIB / 4.0 { format!("{:.1} GiB", b as f64 / GIB) } else { format!("{:.0} MiB", b as f64 / (1u64 << 20) as f64) }
}

/// Picks the fastest device whose free memory fits `required_bytes`; if none fits, the one with
/// the most free memory (and says so).
pub fn plan(required_bytes: u64) -> Plan {
    let devices = Device::all();
    // The host's own free memory is what a CPU device reports.
    let host_free = devices
        .iter()
        .map(|d| d.info())
        .find(|i| i.kind == abi::KIND_CPU)
        .and_then(|i| i.free_memory);
    let mut candidates: Vec<Candidate> = devices
        .into_iter()
        .map(|device| {
            let info = device.info();
            let gflops = calibrate(&device);
            let free_memory = effective_free(&info, host_free);
            let fits = match free_memory {
                Some(free) => (free as f64) * HEADROOM >= required_bytes as f64,
                None => true,
            };
            Candidate {
                name: info.name,
                kind: info.kind,
                total_memory: info.total_memory,
                free_memory,
                shared_host_memory: info.shared_host_memory,
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
    let mut warnings = Vec::new();
    if chosen.shared_host_memory || chosen.kind == abi::KIND_CPU {
        warnings.extend(paging_warning(&chosen.device, required_bytes, host_free));
    }
    Plan { required_bytes, chosen: chosen.device, candidates, reason, may_oom, warnings }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DeviceInfo;

    const GIB: u64 = 1 << 30;

    fn info(free: Option<u64>, shared: bool) -> DeviceInfo {
        DeviceInfo { name: "t".into(), kind: abi::KIND_XPU, total_memory: Some(32 * GIB), free_memory: free, shared_host_memory: shared }
    }

    #[test]
    fn shared_device_is_capped_by_host_free() {
        // The plugin thinks 30 GiB are free, but the host only has 6.
        assert_eq!(effective_free(&info(Some(30 * GIB), true), Some(6 * GIB)), Some(6 * GIB));
    }

    #[test]
    fn shared_device_keeps_its_own_lower_figure() {
        assert_eq!(effective_free(&info(Some(2 * GIB), true), Some(6 * GIB)), Some(2 * GIB));
    }

    #[test]
    fn shared_device_with_unknown_free_uses_host() {
        assert_eq!(effective_free(&info(None, true), Some(6 * GIB)), Some(6 * GIB));
    }

    #[test]
    fn dedicated_device_ignores_host_memory() {
        assert_eq!(effective_free(&info(Some(8 * GIB), false), Some(1 * GIB)), Some(8 * GIB));
        assert_eq!(effective_free(&info(None, false), Some(1 * GIB)), None);
    }

    #[test]
    fn unknown_host_changes_nothing() {
        assert_eq!(effective_free(&info(Some(30 * GIB), true), None), Some(30 * GIB));
    }
}

pub fn format_bytes(b: u64) -> String {
    fmt_bytes(b)
}

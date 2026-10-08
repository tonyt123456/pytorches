//! Placement strategies: how to spread a model over the devices of a machine.
//!
//! What to do when a model does not fit one device depends on the situation (how big it is, how
//! fast each device is, how expensive moving data between them is), so the decision is a trait
//! rather than a fixed algorithm. Each `PlacementStrategy` looks at a `GraphInfo` (the model) and a
//! `MachineProfile` (the hardware) and either proposes a placement with an estimated step time or
//! says why it cannot. `choose` asks every strategy and takes the fastest proposal, or runs one
//! strategy the caller named.
//!
//! The core (memory accounting, measurement, moving tensors) is fixed; strategies only decide.
//! A `MachineProfile` is plain data, so strategies are tested against made-up machines with no
//! hardware attached.

use crate::Device;
use crate::graph::{GraphInfo, PlacementReport, price_chain};
use crate::plan::{HEADROOM, Link, calibrate, calibrate_bandwidth, calibrate_link, effective_free, format_bytes};
use pytorches_plugin_abi as abi;

/// Backward costs about twice the forward arithmetic.
const STEP_FLOPS_PER_FORWARD_FLOP: f64 = 3.0;

#[derive(Clone, Debug)]
pub struct DeviceProfile {
    /// Device string, e.g. `"cuda:0"`.
    pub id: String,
    pub name: String,
    pub kind: u32,
    /// Measured matmul throughput.
    pub gflops: f64,
    /// Measured memory bandwidth in GB/s.
    pub gbps: f64,
    /// Free memory to plan against (shared-memory devices already capped at the host's); `None`
    /// if the device does not say.
    pub free_bytes: Option<u64>,
    pub shared_host: bool,
}

impl DeviceProfile {
    /// Draws from system RAM: the host itself, or a device that shares it.
    pub fn in_host_pool(&self) -> bool {
        self.kind == abi::KIND_CPU || self.shared_host
    }
}

#[derive(Clone, Debug)]
pub struct MachineProfile {
    pub devices: Vec<DeviceProfile>,
    /// `links[from][to]`; the diagonal is unused.
    pub links: Vec<Vec<Link>>,
    pub host_free: Option<u64>,
}

impl MachineProfile {
    /// Measures this machine: every device's speed and free memory, and every device pair's link.
    /// Speeds and links are cached after the first call; free memory is read fresh.
    pub fn measure() -> Self {
        let devs = Device::all();
        let infos: Vec<_> = devs.iter().map(|d| d.info()).collect();
        let host_free = infos.iter().find(|i| i.kind == abi::KIND_CPU).and_then(|i| i.free_memory);
        let devices = devs
            .iter()
            .zip(&infos)
            .map(|(d, i)| DeviceProfile {
                id: d.to_string(),
                name: i.name.clone(),
                kind: i.kind,
                gflops: calibrate(d),
                gbps: calibrate_bandwidth(d),
                free_bytes: effective_free(i, host_free),
                shared_host: i.shared_host_memory,
            })
            .collect();
        let none = Link { bytes_per_sec: f64::INFINITY, latency_secs: 0.0 };
        let links = devs
            .iter()
            .map(|a| devs.iter().map(|b| if a == b { none } else { calibrate_link(a, b) }).collect())
            .collect();
        MachineProfile { devices, links, host_free }
    }

    pub fn link(&self, from: usize, to: usize) -> Link {
        self.links[from][to]
    }
}

/// A strategy's answer: where every layer goes and what that is expected to cost.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub strategy: String,
    /// Index into `MachineProfile::devices` for each layer.
    pub assignment: Vec<usize>,
    pub pricing: PlacementReport<usize>,
    pub compute_secs: f64,
    pub transfer_secs: f64,
    /// Estimated seconds per training step. Stages run one after another (no overlap), so this is
    /// compute plus transfers.
    pub est_step_secs: f64,
    pub reason: String,
}

impl Proposal {
    pub fn n_devices(&self) -> usize {
        self.pricing.per_device.len()
    }
}

pub trait PlacementStrategy: Send + Sync {
    fn name(&self) -> &str;
    /// A placement for `graph` on `machine`, or the reason this strategy cannot provide one.
    fn propose(&self, graph: &GraphInfo, machine: &MachineProfile) -> Result<Proposal, String>;
}

/// Prices `assignment` against `machine`: `Err` if it does not fit, otherwise the pricing and the
/// estimated compute and transfer seconds.
pub fn evaluate(
    graph: &GraphInfo,
    machine: &MachineProfile,
    assignment: &[usize],
) -> Result<(PlacementReport<usize>, f64, f64), String> {
    let pricing = price_chain(graph, assignment, |&i| machine.devices[i].in_host_pool())?;
    for &(i, bytes) in &pricing.per_device {
        let d = &machine.devices[i];
        if let Some(free) = d.free_bytes {
            if bytes as f64 > free as f64 * HEADROOM {
                return Err(format!("needs {} on {}, which has {} free", format_bytes(bytes), d.id, format_bytes(free)));
            }
        }
    }
    if let Some(host) = machine.host_free {
        if pricing.host_pool_bytes as f64 > host as f64 * HEADROOM {
            return Err(format!(
                "needs {} of system RAM across the CPU and shared-memory devices, which have {} free between them",
                format_bytes(pricing.host_pool_bytes),
                format_bytes(host)
            ));
        }
    }
    let compute_secs = graph
        .layers
        .iter()
        .zip(assignment)
        .map(|(l, &d)| {
            let dev = &machine.devices[d];
            let arithmetic = STEP_FLOPS_PER_FORWARD_FLOP * l.flops as f64 / (dev.gflops.max(1e-9) * 1e9);
            let memory = l.traffic_bytes() as f64 / (dev.gbps.max(1e-9) * 1e9);
            arithmetic.max(memory) // whichever limits this layer
        })
        .fold(0.0, |a, b| a + b);
    let transfer_secs = pricing
        .transfers
        .iter()
        .map(|t| {
            let one_way = t.bytes / 2; // forward activation; the gradient comes back the other way
            machine.link(t.from, t.to).secs(one_way) + machine.link(t.to, t.from).secs(one_way)
        })
        .fold(0.0, |a, b| a + b); // not `sum()`: an empty f64 sum is -0.0
    Ok((pricing, compute_secs, transfer_secs))
}

fn proposal(
    strategy: &str,
    graph: &GraphInfo,
    machine: &MachineProfile,
    assignment: Vec<usize>,
    reason: String,
) -> Result<Proposal, String> {
    let (pricing, compute_secs, transfer_secs) = evaluate(graph, machine, &assignment)?;
    Ok(Proposal {
        strategy: strategy.into(),
        assignment,
        pricing,
        compute_secs,
        transfer_secs,
        est_step_secs: compute_secs + transfer_secs,
        reason,
    })
}

/// Faster first; on a tie, fewer devices.
fn better(a: &Proposal, b: &Proposal) -> bool {
    match a.est_step_secs.partial_cmp(&b.est_step_secs) {
        Some(std::cmp::Ordering::Less) => true,
        Some(std::cmp::Ordering::Equal) => a.n_devices() < b.n_devices(),
        _ => false,
    }
}

/// The whole model on one device: the fastest one it fits on.
pub struct SingleDevice;

impl PlacementStrategy for SingleDevice {
    fn name(&self) -> &str {
        "single_device"
    }

    fn propose(&self, graph: &GraphInfo, machine: &MachineProfile) -> Result<Proposal, String> {
        let n = graph.layers.len();
        let mut best: Option<Proposal> = None;
        for (i, d) in machine.devices.iter().enumerate() {
            let reason = format!("the whole model fits on {}", d.id);
            if let Ok(p) = proposal(self.name(), graph, machine, vec![i; n], reason) {
                if best.as_ref().is_none_or(|b| better(&p, b)) {
                    best = Some(p);
                }
            }
        }
        best.ok_or_else(|| {
            let most = machine.devices.iter().filter_map(|d| d.free_bytes).max().unwrap_or(0);
            format!(
                "the model needs {} and no single device has that free (most free: {})",
                format_bytes(graph.total_training_bytes()),
                format_bytes(most)
            )
        })
    }
}

/// Cuts the chain into up to `max_devices` consecutive pieces and gives each piece its own device,
/// choosing the cuts and devices with the lowest estimated step time that fit in memory.
/// Exhaustive, which is cheap at this size (a few devices, tens to hundreds of layers).
pub struct LayerSplit {
    pub max_devices: usize,
}

impl Default for LayerSplit {
    fn default() -> Self {
        LayerSplit { max_devices: 3 }
    }
}

/// Ordered selections of `k` distinct items from `0..n`.
fn permutations(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn go(n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for i in 0..n {
            if !cur.contains(&i) {
                cur.push(i);
                go(n, k, cur, out);
                cur.pop();
            }
        }
    }
    let mut out = Vec::new();
    go(n, k, &mut Vec::new(), &mut out);
    out
}

/// Increasing selections of `k` cut points from `1..n`.
fn cuts(n: usize, k: usize) -> Vec<Vec<usize>> {
    fn go(start: usize, n: usize, k: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == k {
            out.push(cur.clone());
            return;
        }
        for c in start..n {
            cur.push(c);
            go(c + 1, n, k, cur, out);
            cur.pop();
        }
    }
    let mut out = Vec::new();
    go(1, n, k, &mut Vec::new(), &mut out);
    out
}

impl PlacementStrategy for LayerSplit {
    fn name(&self) -> &str {
        "layer_split"
    }

    fn propose(&self, graph: &GraphInfo, machine: &MachineProfile) -> Result<Proposal, String> {
        let layers = graph.layers.len();
        let top = self.max_devices.min(machine.devices.len()).min(layers);
        if top < 2 {
            return Err("needs at least two devices and two layers".into());
        }
        let mut best: Option<Proposal> = None;
        for k in 2..=top {
            for devs in permutations(machine.devices.len(), k) {
                for cut in cuts(layers, k - 1) {
                    let mut assignment = Vec::with_capacity(layers);
                    let mut lo = 0;
                    for (j, &dev) in devs.iter().enumerate() {
                        let hi = cut.get(j).copied().unwrap_or(layers);
                        assignment.extend(std::iter::repeat_n(dev, hi - lo));
                        lo = hi;
                    }
                    let Ok((pricing, compute_secs, transfer_secs)) = evaluate(graph, machine, &assignment) else {
                        continue;
                    };
                    let est = compute_secs + transfer_secs;
                    let beats = best.as_ref().is_none_or(|b| {
                        est < b.est_step_secs || (est == b.est_step_secs && pricing.per_device.len() < b.n_devices())
                    });
                    if beats {
                        let mut parts = Vec::new();
                        let mut lo = 0;
                        for (j, &dev) in devs.iter().enumerate() {
                            let hi = cut.get(j).copied().unwrap_or(layers);
                            parts.push(format!("layers {lo}..{hi} on {}", machine.devices[dev].id));
                            lo = hi;
                        }
                        best = Some(Proposal {
                            strategy: self.name().into(),
                            assignment,
                            pricing,
                            compute_secs,
                            transfer_secs,
                            est_step_secs: est,
                            reason: parts.join(", "),
                        });
                    }
                }
            }
        }
        best.ok_or_else(|| {
            format!(
                "no split over up to {top} devices fits (needs {} in total)",
                format_bytes(graph.total_training_bytes())
            )
        })
    }
}

/// Every proposal that was considered, the winner, and why the rest were not offered.
#[derive(Clone, Debug)]
pub struct ModelPlan {
    pub chosen: Proposal,
    /// All proposals, fastest first (the first is `chosen`).
    pub considered: Vec<Proposal>,
    /// `(strategy, reason)` for strategies that could not propose.
    pub declined: Vec<(String, String)>,
}

pub fn default_strategies() -> Vec<Box<dyn PlacementStrategy>> {
    vec![Box::new(SingleDevice), Box::new(LayerSplit::default())]
}

/// Why `choose` returned no plan.
#[derive(Clone, Debug, PartialEq)]
pub enum ChooseError {
    /// The request itself is wrong: an empty model, or a strategy name nobody registered.
    Invalid(String),
    /// The request is fine but no strategy can place the model on this machine.
    NoPlacement(String),
}

impl std::fmt::Display for ChooseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (ChooseError::Invalid(m) | ChooseError::NoPlacement(m)) = self;
        f.write_str(m)
    }
}

/// Asks the strategies (only the one called `only`, if given) and returns the fastest proposal.
pub fn choose(
    strategies: &[Box<dyn PlacementStrategy>],
    graph: &GraphInfo,
    machine: &MachineProfile,
    only: Option<&str>,
) -> Result<ModelPlan, ChooseError> {
    if graph.layers.is_empty() {
        return Err(ChooseError::Invalid("the model has no layers".into()));
    }
    let asked: Vec<&Box<dyn PlacementStrategy>> = match only {
        Some(name) => {
            let found: Vec<_> = strategies.iter().filter(|s| s.name() == name).collect();
            if found.is_empty() {
                let known: Vec<&str> = strategies.iter().map(|s| s.name()).collect();
                return Err(ChooseError::Invalid(format!("unknown strategy '{name}' (available: {known:?})")));
            }
            found
        }
        None => strategies.iter().collect(),
    };
    let mut considered = Vec::new();
    let mut declined = Vec::new();
    for s in asked {
        match s.propose(graph, machine) {
            Ok(p) => considered.push(p),
            Err(why) => declined.push((s.name().to_string(), why)),
        }
    }
    considered.sort_by(|a, b| {
        a.est_step_secs.partial_cmp(&b.est_step_secs).unwrap_or(std::cmp::Ordering::Equal).then(a.n_devices().cmp(&b.n_devices()))
    });
    match considered.first().cloned() {
        Some(chosen) => Ok(ModelPlan { chosen, considered, declined }),
        None => Err(ChooseError::NoPlacement(format!(
            "no strategy can place this model: {}",
            declined.iter().map(|(s, w)| format!("{s}: {w}")).collect::<Vec<_>>().join("; ")
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::LayerInfo;

    const GIB: u64 = 1 << 30;
    const FAST_LINK: Link = Link { bytes_per_sec: 1e12, latency_secs: 0.0 };

    /// `(gflops, free GiB, in the host pool)` per device.
    fn machine(devs: &[(f64, u64, bool)], link: Link, host_free_gib: Option<u64>) -> MachineProfile {
        let n = devs.len();
        MachineProfile {
            devices: devs
                .iter()
                .enumerate()
                .map(|(i, &(gflops, free, pool))| DeviceProfile {
                    id: format!("d{i}"),
                    name: format!("device {i}"),
                    kind: if pool { abi::KIND_CPU } else { abi::KIND_CUDA },
                    gflops,
                    gbps: 1e9, // effectively unlimited, so these tests are about arithmetic
                    free_bytes: Some(free * GIB),
                    shared_host: false,
                })
                .collect(),
            links: vec![vec![link; n]; n],
            host_free: host_free_gib.map(|g| g * GIB),
        }
    }

    /// `n` layers; each trains in `gib` GiB (params only: training bytes are 3x the parameters).
    fn chain(n: usize, gib_each: u64, gflop_each: f64) -> GraphInfo {
        GraphInfo {
            input_bytes: 1 << 20,
            layers: (0..n)
                .map(|i| LayerInfo {
                    name: format!("l{i}"),
                    param_bytes: gib_each * GIB / 3,
                    out_bytes: 1 << 20,
                    flops: (gflop_each * 1e9 / STEP_FLOPS_PER_FORWARD_FLOP) as u64,
                })
                .collect(),
        }
    }

    fn pick(m: &MachineProfile, g: &GraphInfo, only: Option<&str>) -> Result<ModelPlan, String> {
        choose(&default_strategies(), g, m, only).map_err(|e| e.to_string())
    }

    #[test]
    fn step_time_is_compute_over_throughput() {
        // 4 layers x 100 GFLOP per step on a 1000 GFLOP/s device = 0.4 s
        let m = machine(&[(1000.0, 64, false)], FAST_LINK, None);
        let p = pick(&m, &chain(4, 1, 100.0), None).unwrap();
        assert!((p.chosen.est_step_secs - 0.4).abs() < 1e-6, "{}", p.chosen.est_step_secs);
        assert_eq!(p.chosen.assignment, vec![0; 4]);
    }

    #[test]
    fn memory_bound_layers_are_priced_by_bandwidth_not_arithmetic() {
        // No arithmetic to speak of, 1 GiB of parameters per layer: step time is the time to
        // stream them. Device 1 has half the arithmetic but four times the bandwidth, so it wins.
        let mut m = machine(&[(4000.0, 64, false), (2000.0, 64, false)], FAST_LINK, None);
        m.devices[0].gbps = 100.0;
        m.devices[1].gbps = 400.0;
        let g = GraphInfo {
            input_bytes: 0,
            layers: vec![LayerInfo { name: "w".into(), param_bytes: GIB, out_bytes: 0, flops: 1 }],
        };
        let p = pick(&m, &g, None).unwrap();
        assert_eq!(p.chosen.assignment, vec![1]);
        let expect = (6 * GIB) as f64 / 400e9;
        assert!((p.chosen.est_step_secs - expect).abs() < 1e-9, "{} vs {expect}", p.chosen.est_step_secs);
    }

    #[test]
    fn a_layer_costs_the_slower_of_arithmetic_and_memory() {
        let mut m = machine(&[(1000.0, 64, false)], FAST_LINK, None);
        m.devices[0].gbps = 100.0;
        // 100 GFLOP/step on 1000 GFLOP/s = 0.1 s; 6 GiB of traffic on 100 GB/s = ~0.064 s
        let g = chain(1, 2, 100.0);
        let p = pick(&m, &g, None).unwrap();
        assert!((p.chosen.est_step_secs - 0.1).abs() < 1e-9);
        // make memory the limit
        m.devices[0].gbps = 10.0;
        let p = pick(&m, &g, None).unwrap();
        let mem = g.layers[0].traffic_bytes() as f64 / 10e9;
        assert!((p.chosen.est_step_secs - mem).abs() < 1e-9 && mem > 0.1);
    }

    #[test]
    fn a_model_that_fits_goes_on_the_fastest_device() {
        let m = machine(&[(500.0, 64, false), (4000.0, 16, false)], FAST_LINK, None);
        let p = pick(&m, &chain(4, 2, 10.0), None).unwrap();
        assert_eq!(p.chosen.strategy, "single_device");
        assert_eq!(p.chosen.assignment, vec![1; 4]);
    }

    #[test]
    fn a_model_too_big_for_the_fast_device_goes_to_the_big_one() {
        let m = machine(&[(500.0, 64, false), (4000.0, 8, false)], FAST_LINK, None);
        let p = pick(&m, &chain(4, 4, 10.0), Some("single_device")).unwrap();
        assert_eq!(p.chosen.assignment, vec![0; 4]);
    }

    #[test]
    fn nothing_fits_alone_so_it_splits() {
        // 4 layers x 3 GiB = 12 GiB; each device has 8 GiB (6.8 usable): two layers per device.
        let m = machine(&[(2000.0, 8, false), (2000.0, 8, false)], FAST_LINK, None);
        let p = pick(&m, &chain(4, 3, 10.0), None).unwrap();
        assert_eq!(p.chosen.strategy, "layer_split");
        assert_eq!(p.chosen.n_devices(), 2);
        let split = &p.chosen.assignment;
        assert_eq!(split[0], split[1]);
        assert_eq!(split[2], split[3]);
        assert_ne!(split[0], split[3]);
        assert!(p.declined.iter().any(|(s, _)| s == "single_device"));
    }

    #[test]
    fn a_split_puts_more_layers_on_the_faster_device() {
        // 6 layers x 1 GiB; both devices could hold at most 5 of them. The fast one is 4x faster,
        // so it should take as many layers as it can hold.
        let m = machine(&[(1000.0, 6, false), (4000.0, 6, false)], FAST_LINK, None);
        let p = pick(&m, &chain(6, 1, 100.0), Some("layer_split")).unwrap();
        let on_fast = p.chosen.assignment.iter().filter(|&&d| d == 1).count();
        assert_eq!(on_fast, 5, "{:?}", p.chosen.assignment); // 5 GiB fits in 6 * 0.85, 6 does not
    }

    #[test]
    fn a_split_is_not_chosen_when_the_model_fits() {
        let m = machine(&[(2000.0, 32, false), (2000.0, 32, false)], FAST_LINK, None);
        let p = pick(&m, &chain(4, 1, 10.0), None).unwrap();
        assert_eq!(p.chosen.strategy, "single_device");
        // the split was still considered, and is never faster
        let split = p.considered.iter().find(|c| c.strategy == "layer_split");
        assert!(split.is_none_or(|s| s.est_step_secs >= p.chosen.est_step_secs));
    }

    #[test]
    fn an_expensive_link_makes_a_split_lose_to_one_slow_big_device() {
        // Fast device holds half; the link is so slow that moving activations costs more than
        // running everything on the slow device with enough memory.
        let slow_link = Link { bytes_per_sec: 1e3, latency_secs: 1.0 };
        let m = machine(&[(100.0, 64, false), (10_000.0, 4, false)], slow_link, None);
        let g = chain(4, 2, 100.0);
        let p = pick(&m, &g, None).unwrap();
        assert_eq!(p.chosen.strategy, "single_device");
        assert_eq!(p.chosen.assignment, vec![0; 4]);
    }

    #[test]
    fn host_pool_is_shared_between_the_cpu_and_other_pool_devices() {
        // Two pool devices (think CPU and an iGPU), each with 20 GiB free on its own, but the host
        // has 20 GiB between them. 16 GiB fits on one; 8+8 split across both would not fit twice
        // over, so a 24 GiB model fits nowhere.
        let m = machine(&[(100.0, 20, true), (100.0, 20, true)], FAST_LINK, Some(20));
        assert!(pick(&m, &chain(4, 6, 1.0), None).is_err()); // 24 GiB > 20 * 0.85 shared
        assert!(pick(&m, &chain(2, 6, 1.0), None).is_ok()); // 12 GiB fits
    }

    #[test]
    fn unknown_strategy_is_an_error_that_lists_the_known_ones() {
        let m = machine(&[(1000.0, 64, false)], FAST_LINK, None);
        let e = pick(&m, &chain(2, 1, 1.0), Some("nope")).unwrap_err();
        assert!(e.contains("single_device") && e.contains("layer_split"), "{e}");
    }

    #[test]
    fn layer_split_needs_two_devices() {
        let m = machine(&[(1000.0, 64, false)], FAST_LINK, None);
        let e = pick(&m, &chain(2, 1, 1.0), Some("layer_split")).unwrap_err();
        assert!(e.contains("at least two"), "{e}");
    }

    #[test]
    fn a_model_that_fits_nowhere_is_an_error_with_reasons() {
        let m = machine(&[(1000.0, 4, false), (1000.0, 4, false)], FAST_LINK, None);
        let e = pick(&m, &chain(4, 6, 1.0), None).unwrap_err();
        assert!(e.contains("single_device") && e.contains("layer_split"), "{e}");
    }

    #[test]
    fn empty_model_is_an_error() {
        let m = machine(&[(1000.0, 4, false)], FAST_LINK, None);
        assert!(pick(&m, &GraphInfo::default(), None).is_err());
    }

    #[test]
    fn a_custom_strategy_plugs_in() {
        struct AlwaysLast;
        impl PlacementStrategy for AlwaysLast {
            fn name(&self) -> &str {
                "always_last"
            }
            fn propose(&self, g: &GraphInfo, m: &MachineProfile) -> Result<Proposal, String> {
                proposal("always_last", g, m, vec![m.devices.len() - 1; g.layers.len()], "test".into())
            }
        }
        let m = machine(&[(5000.0, 64, false), (10.0, 64, false)], FAST_LINK, None);
        let strategies: Vec<Box<dyn PlacementStrategy>> = vec![Box::new(AlwaysLast)];
        let p = choose(&strategies, &chain(2, 1, 1.0), &m, None).unwrap();
        assert_eq!(p.chosen.strategy, "always_last");
        assert!(matches!(choose(&strategies, &chain(2, 1, 1.0), &m, Some("x")), Err(ChooseError::Invalid(_))));
        assert_eq!(p.chosen.assignment, vec![1, 1]);
    }

    #[test]
    fn enumeration_helpers() {
        assert_eq!(permutations(3, 2).len(), 6);
        assert_eq!(permutations(3, 3).len(), 6);
        assert_eq!(cuts(5, 2).len(), 6); // C(4, 2)
        assert_eq!(cuts(5, 1), vec![vec![1], vec![2], vec![3], vec![4]]);
    }
}

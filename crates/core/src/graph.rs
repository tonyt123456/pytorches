//! What a placement strategy needs to know about a model, and what a placement costs.
//!
//! `GraphInfo` is a description of a model as a chain of layers: how many bytes each layer's
//! parameters and output take, and how much arithmetic it does. It carries no tensors and knows
//! nothing about devices, so a strategy can reason about a model that does not exist yet (that is
//! the point: deciding where to put it before allocating it). A `Placement` assigns one device to
//! each layer, and `Placement::report` prices that assignment.

use crate::Device;
use pytorches_plugin_abi as abi;

/// One layer of a chain. All sizes are for one training step at the batch size the graph was
/// built for.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerInfo {
    pub name: String,
    /// Bytes of this layer's parameters (a gradient of the same size is kept while training).
    pub param_bytes: u64,
    /// Bytes of this layer's output activation. It is kept for backward, its gradient is
    /// materialized, and it is the tensor that crosses a device boundary after this layer.
    pub out_bytes: u64,
    /// Forward floating-point operations (backward is about twice that).
    pub flops: u64,
}

impl LayerInfo {
    /// Estimated peak device memory to train this layer, in bytes: parameters and their gradients,
    /// the saved output plus its gradient plus one transient, and a transient parameter-sized
    /// buffer for backward. An estimate, intentionally a bit generous (same convention as
    /// `nn.estimate_mlp_training_bytes`).
    pub fn training_bytes(&self) -> u64 {
        2 * self.param_bytes + 3 * self.out_bytes + self.param_bytes
    }

    /// Estimated bytes moved through device memory in one training step, which bounds the step
    /// time of layers that do little arithmetic per byte (small batches, elementwise layers).
    /// Parameters are streamed about six times: read in forward, read again for the input
    /// gradient, written as the weight gradient, then read twice and written by the update.
    /// Activations are read and written a few times in each direction.
    pub fn traffic_bytes(&self) -> u64 {
        6 * self.param_bytes + 6 * self.out_bytes
    }
}

/// A model as a chain: `input_bytes` flows into `layers[0]`, each layer feeds the next.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GraphInfo {
    pub input_bytes: u64,
    pub layers: Vec<LayerInfo>,
}

impl GraphInfo {
    pub fn total_param_bytes(&self) -> u64 {
        self.layers.iter().map(|l| l.param_bytes).sum()
    }

    pub fn total_training_bytes(&self) -> u64 {
        self.layers.iter().map(LayerInfo::training_bytes).sum()
    }

    pub fn total_flops(&self) -> u64 {
        self.layers.iter().map(|l| l.flops).sum()
    }

    /// Training bytes of the layers in `lo..hi`.
    pub fn range_training_bytes(&self, lo: usize, hi: usize) -> u64 {
        self.layers[lo..hi].iter().map(LayerInfo::training_bytes).sum()
    }
}

/// One device per layer.
#[derive(Clone, Debug)]
pub struct Placement {
    pub devices: Vec<Device>,
}

/// A change of device between two consecutive layers.
#[derive(Clone, Debug, PartialEq)]
pub struct Transfer<D = Device> {
    /// The activation produced by this layer moves on to the next one.
    pub after_layer: usize,
    pub from: D,
    pub to: D,
    /// Bytes moved in one training step: the activation forward and its gradient back.
    pub bytes: u64,
}

/// What a placement costs. `D` identifies a device: a real `Device`, or an index into a
/// `MachineProfile` when a strategy prices a placement that does not exist yet.
#[derive(Clone, Debug)]
pub struct PlacementReport<D = Device> {
    /// Estimated training bytes on each device, in first-use order.
    pub per_device: Vec<(D, u64)>,
    pub transfers: Vec<Transfer<D>>,
    /// Bytes drawn from system RAM: the sum over every device that is the host or shares its
    /// memory. Those devices compete for one pool, so each one's own figure understates the load.
    pub host_pool_bytes: u64,
    pub transfer_bytes: u64,
}

/// Prices `devices` (one per layer of `graph`). `in_host_pool` says whether a device draws from
/// system RAM.
pub fn price_chain<D: PartialEq + Clone>(
    graph: &GraphInfo,
    devices: &[D],
    in_host_pool: impl Fn(&D) -> bool,
) -> Result<PlacementReport<D>, String> {
    if devices.len() != graph.layers.len() {
        return Err(format!("placement has {} device(s) for a graph of {} layer(s)", devices.len(), graph.layers.len()));
    }
    let mut per_device: Vec<(D, u64)> = Vec::new();
    for (layer, dev) in graph.layers.iter().zip(devices) {
        match per_device.iter_mut().find(|(d, _)| d == dev) {
            Some((_, b)) => *b += layer.training_bytes(),
            None => per_device.push((dev.clone(), layer.training_bytes())),
        }
    }
    let transfers: Vec<Transfer<D>> = devices
        .windows(2)
        .enumerate()
        .filter(|(_, w)| w[0] != w[1])
        .map(|(i, w)| Transfer { after_layer: i, from: w[0].clone(), to: w[1].clone(), bytes: 2 * graph.layers[i].out_bytes })
        .collect();
    let host_pool_bytes = per_device.iter().filter(|(d, _)| in_host_pool(d)).map(|(_, b)| *b).sum();
    let transfer_bytes = transfers.iter().map(|t| t.bytes).sum();
    Ok(PlacementReport { per_device, transfers, host_pool_bytes, transfer_bytes })
}

impl Placement {
    pub fn new(devices: Vec<Device>) -> Self {
        Placement { devices }
    }

    /// Every layer on `device`.
    pub fn uniform(graph: &GraphInfo, device: &Device) -> Self {
        Placement { devices: vec![device.clone(); graph.layers.len()] }
    }

    /// Maximal runs of consecutive layers on one device, as `(device, lo, hi)` with `lo..hi`.
    pub fn segments(&self) -> Vec<(Device, usize, usize)> {
        let mut out: Vec<(Device, usize, usize)> = Vec::new();
        for (i, d) in self.devices.iter().enumerate() {
            match out.last_mut() {
                Some((last, _, hi)) if last == d => *hi = i + 1,
                _ => out.push((d.clone(), i, i + 1)),
            }
        }
        out
    }

    pub fn report(&self, graph: &GraphInfo) -> Result<PlacementReport, String> {
        price_chain(graph, &self.devices, |d| {
            let info = d.info();
            info.kind == abi::KIND_CPU || info.shared_host_memory
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(name: &str, param: u64, out: u64) -> LayerInfo {
        LayerInfo { name: name.into(), param_bytes: param, out_bytes: out, flops: 0 }
    }

    fn graph() -> GraphInfo {
        GraphInfo {
            input_bytes: 100,
            layers: vec![layer("a", 1000, 10), layer("b", 0, 10), layer("c", 2000, 20), layer("d", 0, 20)],
        }
    }

    #[test]
    fn training_bytes_model() {
        // 2 * params (weights + grads) + 3 * out + params (backward transient)
        assert_eq!(layer("x", 1000, 10).training_bytes(), 3030);
        assert_eq!(layer("relu", 0, 10).training_bytes(), 30);
    }

    #[test]
    fn totals_and_ranges() {
        let g = graph();
        assert_eq!(g.total_param_bytes(), 3000);
        assert_eq!(g.total_training_bytes(), 3030 + 30 + 6060 + 60);
        assert_eq!(g.range_training_bytes(0, 2), 3060);
        assert_eq!(g.range_training_bytes(2, 4), 6120);
    }

    #[test]
    fn segments_group_consecutive_layers() {
        let Some(cpu) = Device::all().into_iter().next() else { return };
        let p = Placement::uniform(&graph(), &cpu);
        assert_eq!(p.segments(), vec![(cpu, 0, 4)]);
    }

    #[test]
    fn report_prices_uniform_placement_with_no_transfers() {
        let Some(cpu) = Device::all().into_iter().next() else { return };
        let g = graph();
        let r = Placement::uniform(&g, &cpu).report(&g).unwrap();
        assert_eq!(r.per_device, vec![(cpu, g.total_training_bytes())]);
        assert!(r.transfers.is_empty());
        assert_eq!(r.transfer_bytes, 0);
    }

    #[test]
    fn report_rejects_a_placement_of_the_wrong_length() {
        let Some(cpu) = Device::all().into_iter().next() else { return };
        let p = Placement::new(vec![cpu.clone(), cpu]);
        assert!(p.report(&graph()).is_err());
    }
}

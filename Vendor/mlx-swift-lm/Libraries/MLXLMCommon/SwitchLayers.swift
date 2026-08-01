import Foundation
import MLX
import MLXFast
import MLXNN

// Port of https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/models/switch_layers.py

/// Compiled SiLU-gated product (`silu(gate) * up`) for the common MoE GLU path.
/// Fusing activation + product into one compiled, shapeless kernel cuts kernel
/// dispatches and intermediates on the hot decode path. Upstream ef85ed0.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on) like the sibling `compiledSwiGLU` / `safeGeluApproximate` fusions.
/// The default SiLU `SwitchGLU` path wires this in as `activationProduct` (the
/// highest-precedence branch in `callAsFunction`) and `LFM2MoE` calls it directly,
/// so without the gate both would keep hitting compiled kernels on the very M1/M2 +
/// macOS Tahoe machines the opt-out (MLX #3329) is meant to protect. Falls back to
/// the plain uncompiled closure when off; the default (env unset) stays compiled.
public let compiledSiluProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled weighted expert-output combine (`(outputs * weights[..., None]).sum(-2)`).
/// Shared by MoE routers (e.g. Gemma 4) to fuse the scale + reduce. Upstream ef85ed0.
public let weightedExpertSum: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(
    shapeless: true
) { outputs, weights in
    (outputs * MLX.expandedDimensions(weights, axis: -1)).sum(axis: -2)
}

// MARK: - Compiled activation fusions (vMLX / osaurus-main port)

/// Approximate (tanh) GELU written with `x * x * x` instead of the Power
/// primitive (`x ** 3`). The Power primitive returns zero results under the
/// macOS Tahoe Metal JIT (MLX #3329), so the explicit multiplies keep it safe
/// under `compile(shapeless: true)`. Numerically identical to
/// `MLXNN.geluApproximate`.
///
/// Gated by `MLXHardwareInfo.isCompiledDecodeSupported` (env `MLX_COMPILED_DECODE`,
/// default on); falls back to the plain closure when compiled fusions are off.
public let safeGeluApproximate: @Sendable (MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray) -> MLXArray = { (x: MLXArray) -> MLXArray in
        0.5 * x * (1 + tanh(sqrt(2 / Float.pi) * (x + 0.044715 * x * x * x)))
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Drop-in replacement for `MLXNN.GELU(approximation: .tanh)` that avoids the
/// Power primitive crash. Use anywhere a tanh-approx GELU unary layer is needed.
public class SafeGELU: Module, UnaryLayer {
    public override init() { super.init() }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        safeGeluApproximate(x)
    }
}

/// Compiled SiLU-gated GLU product (`silu(gate) * up`). Same math as
/// `compiledSiluProduct` above, but gated by `MLXHardwareInfo` so M1/M2 + macOS
/// Tahoe can opt out. Used by `SwitchGLU` when a SiLU activation is supplied via
/// the custom-activation initializer (where `activationProduct` is nil).
private let compiledSwiGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        MLXNN.silu(gate) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Compiled GELU-gated GLU product (`geluApprox(gate) * up`), fusing the tanh
/// GELU and the element-wise multiply into one shapeless kernel. Uses the
/// Power-free `x * x * x` GELU so it is safe under `compile(shapeless: true)`.
private let compiledGeGLU: @Sendable (MLXArray, MLXArray) -> MLXArray = {
    let body: @Sendable (MLXArray, MLXArray) -> MLXArray = {
        (gate: MLXArray, up: MLXArray) -> MLXArray in
        (0.5 * gate * (1 + tanh(sqrt(2 / Float.pi) * (gate + 0.044715 * gate * gate * gate)))) * up
    }
    if MLXHardwareInfo.isCompiledDecodeSupported {
        return compile(shapeless: true, body)
    }
    return body
}()

/// Linear inverse-permutation scatter for the sorted MoE route table.
/// `argSort` returns `order` as a uint32 permutation, so
/// `inverse[order[i]] = i` has exactly one writer per output and produces the
/// same integer bits as `argSort(order)` without another comparison sort.
/// DEFAULT ON; set `DARKBLOOM_INVERSE_SCATTER=0` to restore the original
/// second `argSort` path inside the same binary.
private let inversePermutationScatterEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_INVERSE_SCATTER"] != "0"

private let inversePermutationScatterKernel = MLXFast.metalKernel(
    name: "mlx_lm_inverse_permutation_scatter_u32_v1",
    inputNames: ["order"],
    outputNames: ["inverse"],
    source: """
        uint i = thread_position_in_grid.x;
        inverse[order[i]] = i;
        """,
    ensureRowContiguous: false
)

/// `DARKBLOOM_COUNTING_ROUTE_SORT` (default OFF; set "1" to enable): replaces
/// the MoE route table's `argSort` with a stable counting sort.
///
/// The sorted keys are expert ids: `indices.size` values drawn from
/// `[0, expertCount)` with `expertCount == 256` for Laguna. A comparison
/// merge sort is the wrong algorithm for that key space, and MLX's is also
/// badly shaped for the machine. `gpu_merge_sort` on 4096 uint32 keys
/// (`backend/metal/sort.cpp:286-312`: `tn=4`, `potential_bn=1024 > 256` so
/// `bn=512`, `n_per_block=2048`, `n_blocks=2 > 1` so `multi_block_sort`)
/// issues FOUR dispatches whose threadgroup counts are 2, 1, 2 and 1 — and
/// the partition kernel runs `n_thr_per_group = min(n_blocks + 1, 1024) == 3`
/// THREADS in ONE threadgroup on a ~64-core M5 Max. A fifth dispatch
/// (`inversePermutationScatterKernel`, 16 threadgroups) then builds the
/// inverse permutation. All five sit on the serial dependency chain — the
/// gather that follows cannot start until `order` lands — and this runs once
/// per sparse layer, 39 times per prefill forward.
///
/// Both arms emit `order` AND `inverseOrder` from the scatter pass (rank(i)
/// IS `inverse[i]`, so the scatter that writes `order[rank] = i` can write
/// `inverse[i] = rank` for free), removing the fifth dispatch outright.
///
/// **`=1` — three dispatches at 32 / 1 / 32 threadgroups.** The middle scan
/// is a single threadgroup: a global exclusive scan over 256 bin totals is an
/// irreducible small serial dependency. It runs 256 threads against the 3
/// `partition_mbsort` ran, and the net is 5 dispatches down to 3 with two
/// lifted from 2 to 32 threadgroups — but it does leave one 1-threadgroup
/// kernel on the serial chain.
///
/// **`=2` — two dispatches at 32 / 32 threadgroups, no 1-threadgroup kernel
/// at all.** Each of the 32 scatter threadgroups re-derives the 256-bin scan
/// from `tile_hist` itself, so the middle dispatch disappears. That is the
/// `lagunaTailNormQKVGate` shape (re-derive per threadgroup rather than
/// dispatch a tiny kernel to share the result) which is the only fusion
/// pattern that has actually measured positive on this box.
///
/// The two arms are numerically identical by construction — arm 2 computes
/// the same `tile_base[t][b]` value arm 1 reads from device memory, from the
/// same `tile_hist` bytes in the same order — so they are an occupancy A/B,
/// not a numerics A/B. Both are bit-exact against stock `argSort` (see
/// `routeScatterKernel`). Keeping both in one binary means a single build
/// answers "is the 1-threadgroup scan worth removing?" without a rebuild,
/// per this repo's one-binary ablation rule.
///
/// Cost of arm 2: each threadgroup reads the whole `tiles x bins` histogram
/// (32 KiB at the ranked prefill shape) instead of one `uint`, so ~1 MiB of
/// mostly L2-resident traffic per sparse layer against the ~8.4 MiB of expert
/// weights that layer already streams, plus 16 extra threadgroup barriers for
/// the in-register scan. Whether that beats one 1-threadgroup dispatch is
/// exactly the measurement.
/// **DEFAULT `2`** — the ranked runner sets no `DARKBLOOM_*` variable, so the
/// shipped default has to BE the mechanism. An unset variable selects arm 2;
/// `0` is the explicit off switch that restores the stock `argSort` chain
/// inside the same binary, and `1` selects the three-dispatch arm.
private let countingRouteSortMode: Int = {
    guard let raw = ProcessInfo.processInfo.environment["DARKBLOOM_COUNTING_ROUTE_SORT"]
    else {
        return 2
    }
    // An unrecognized value falls back to the shipped default rather than to
    // off, matching `lagunaRouterRowsPerGroup`'s convention: a typo in an
    // ablation variable must not silently ship the control.
    guard let value = Int(raw), (0 ... 2).contains(value) else { return 2 }
    return value
}()

private let countingRouteSortEnabled = countingRouteSortMode > 0

private let routeSortTraceEnabled =
    ProcessInfo.processInfo.environment["DARKBLOOM_TRACE_FUSION"] == "1"

/// Elements per tile AND threads per threadgroup for the histogram/scatter
/// passes: the scatter's intra-tile stable rank indexes its threadgroup
/// scratch by lane, so the two must stay equal.
private let routeSortTileSize = 128

/// Threadgroup scratch width of the scan pass. `expertCount` may be smaller
/// (lanes above it contribute zero and write nothing) but never larger.
private let routeSortMaxBins = 256

/// One-shot ground-truth trace, fired at the DISPATCH decision rather than at
/// the flag read, mirroring the `stage2_gather` line in
/// `backend/metal/quantized.cpp`. "active" requires BOTH the flag and an
/// admitting guard; a declining guard prints "inactive" with the fields that
/// declined, so a flagged run can never silently measure its own control.
private final class RouteSortTraceOnce: @unchecked Sendable {
    private var fired = false
    private let lock = NSLock()

    func fire(_ body: () -> String) {
        lock.lock()
        defer { lock.unlock() }
        guard !fired else { return }
        fired = true
        FileHandle.standardError.write(Data(body().utf8))
    }
}

private let routeSortTrace = RouteSortTraceOnce()

/// Per-tile 256-bin histogram. One threadgroup per tile, one element per
/// lane. No atomics: lane `b` owns bins `b`, `b + 128`, ... and counts its
/// own bins by walking the tile's threadgroup-resident values, so the only
/// synchronization is the single barrier after the tile load.
private let routeTileHistKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_tile_hist_u32_v1",
    inputNames: ["indices", "count", "bin_count"],
    outputNames: ["tile_hist"],
    source: """
        constexpr uint tile_size = 128;
        constexpr uint tg_size = 128;

        uint lid = thread_position_in_threadgroup.x;
        uint tile = threadgroup_position_in_grid.x;
        uint n = uint(count);
        uint bins = uint(bin_count);
        uint base = tile * tile_size;

        threadgroup uint tv[128];  // literal: must equal tile_size
        uint i = base + lid;
        bool live = i < n;
        uint v = 0;
        if (live) {
            v = uint(indices[i]);
            // Memory-safety clamp only. The contract is
            // `0 <= indices[i] < bin_count`, which the router guarantees by
            // construction; an out-of-range id would already produce an
            // out-of-bounds gather on the stock argSort path.
            if (v >= bins) { v = bins - 1; }
        }
        tv[lid] = live ? v : 0xFFFFFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint valid = (n > base) ? min(tile_size, n - base) : 0u;
        for (uint b = lid; b < bins; b += tg_size) {
            uint c = 0;
            for (uint j = 0; j < valid; ++j) {
                c += (tv[j] == b) ? 1u : 0u;
            }
            tile_hist[tile * bins + b] = c;
        }
        """,
    ensureRowContiguous: true
)

/// Exclusive scan of the tile histogram into per-(tile, bin) base offsets.
/// Lane `b` owns bin `b`: it sums its own column to get the bin total, the
/// threadgroup runs one Hillis-Steele inclusive scan across bins, and each
/// lane re-walks its column writing the running base. `tile_base[t][b]` is
/// therefore `(number of keys with value < b) + (number of keys equal to b
/// in tiles before t)` — the stable rank base for that (tile, bin).
private let routeBinScanKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_bin_scan_u32_v1",
    inputNames: ["tile_hist", "bin_count", "tile_count"],
    outputNames: ["tile_base"],
    source: """
        constexpr uint max_bins = 256;

        uint b = thread_position_in_grid.x;
        uint bins = uint(bin_count);
        uint tiles = uint(tile_count);

        uint total = 0;
        if (b < bins) {
            for (uint t = 0; t < tiles; ++t) {
                total += tile_hist[t * bins + b];
            }
        }

        // Lanes at or above `bins` carry zero, so they never perturb the
        // prefix of any live lane (a prefix only reads to its left, and every
        // live bin is to the left of every dead one).
        threadgroup uint scan[256];  // literal: must equal max_bins
        scan[b] = total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1; off < max_bins; off <<= 1) {
            uint add = (b >= off) ? scan[b - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scan[b] += add;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        uint bin_offset = scan[b] - total;

        if (b < bins) {
            uint running = bin_offset;
            for (uint t = 0; t < tiles; ++t) {
                uint c = tile_hist[t * bins + b];
                tile_base[t * bins + b] = running;
                running += c;
            }
        }
        """,
    ensureRowContiguous: true
)

/// Stable scatter. Lane `i`'s rank is its (tile, bin) base plus the number of
/// EARLIER lanes in the same tile holding the same value, which is exactly
/// `#{j < i : v_j == v_i}` because the base already accounts for every
/// earlier tile. That is the textbook stable rank, so `order` is
/// bit-identical to `argSort`'s (MLX's merge takes from the left run on ties
/// — `backend/metal/kernels/sort.h:139`, `pred = (b_idx < B_sz) && (a_idx >=
/// A_sz || op(b, a))` with `LessThan` false on equality — so its argsort is
/// stable, and a stable sort's permutation is unique).
///
/// `inverse[i] = rank` is written here rather than in a follow-up dispatch:
/// `inverse[order[k]] = k` and `order[rank(i)] = i` together give
/// `inverse[i] = rank(i)`. Every rank in `[0, count)` has exactly one writer
/// (the map is a permutation) and so does every `i`, so neither output needs
/// initialization.
private let routeScatterKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_scatter_u32_v1",
    inputNames: ["indices", "tile_base", "count", "bin_count"],
    outputNames: ["order", "inverse"],
    source: """
        constexpr uint tile_size = 128;

        uint lid = thread_position_in_threadgroup.x;
        uint tile = threadgroup_position_in_grid.x;
        uint n = uint(count);
        uint bins = uint(bin_count);
        uint base = tile * tile_size;

        threadgroup uint tv[128];  // literal: must equal tile_size
        uint i = base + lid;
        bool live = i < n;
        uint v = 0;
        if (live) {
            v = uint(indices[i]);
            if (v >= bins) { v = bins - 1; }
        }
        tv[lid] = live ? v : 0xFFFFFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (live) {
            uint prior = 0;
            for (uint j = 0; j < lid; ++j) {
                prior += (tv[j] == v) ? 1u : 0u;
            }
            uint rank = tile_base[tile * bins + v] + prior;
            order[rank] = i;
            inverse[i] = rank;
        }
        """,
    ensureRowContiguous: true
)

/// Arm-2 scatter: identical output to `routeScatterKernel`, but each
/// threadgroup re-derives its own `tile_base` row from `tile_hist` instead of
/// reading one produced by a separate single-threadgroup dispatch. The
/// arithmetic is the same sum over the same bytes in the same order — device
/// integer addition is associative and exact, so "re-derive" here is literally
/// recomputing the same integer, not a reassociation.
///
/// Requires `max_bins == 2 * tg_size`: the scan carries two bins per lane.
private let routeScatterFusedKernel = MLXFast.metalKernel(
    name: "mlx_lm_route_scatter_fused_u32_v1",
    inputNames: ["indices", "tile_hist", "count", "bin_count", "tile_count"],
    outputNames: ["order", "inverse"],
    source: """
        constexpr uint tile_size = 128;
        constexpr uint tg_size = 128;
        constexpr uint max_bins = 256;

        uint lid = thread_position_in_threadgroup.x;
        uint tile = threadgroup_position_in_grid.x;
        uint n = uint(count);
        uint bins = uint(bin_count);
        uint tiles = uint(tile_count);
        uint base = tile * tile_size;

        // Per-bin global total, and how many keys of that bin live in tiles
        // strictly before this one. Lane `lid` owns bins `lid` and
        // `lid + tg_size`, so one walk of the histogram column yields both.
        threadgroup uint bin_total[256];  // literal: must equal max_bins
        threadgroup uint bin_before[256];  // literal: must equal max_bins
        for (uint b = lid; b < max_bins; b += tg_size) {
            uint total = 0;
            uint before = 0;
            if (b < bins) {
                for (uint t = 0; t < tiles; ++t) {
                    uint c = tile_hist[t * bins + b];
                    if (t < tile) { before += c; }
                    total += c;
                }
            }
            bin_total[b] = total;
            bin_before[b] = before;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inclusive Hillis-Steele scan of bin_total, two bins per lane.
        threadgroup uint scan[256];  // literal: must equal max_bins
        uint b0 = lid;
        uint b1 = lid + tg_size;
        scan[b0] = bin_total[b0];
        scan[b1] = bin_total[b1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint off = 1; off < max_bins; off <<= 1) {
            uint add0 = (b0 >= off) ? scan[b0 - off] : 0u;
            uint add1 = (b1 >= off) ? scan[b1 - off] : 0u;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            scan[b0] += add0;
            scan[b1] += add1;
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        threadgroup uint tv[128];  // literal: must equal tile_size
        uint i = base + lid;
        bool live = i < n;
        uint v = 0;
        if (live) {
            v = uint(indices[i]);
            if (v >= bins) { v = bins - 1; }
        }
        tv[lid] = live ? v : 0xFFFFFFFFu;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (live) {
            uint prior = 0;
            for (uint j = 0; j < lid; ++j) {
                prior += (tv[j] == v) ? 1u : 0u;
            }
            // (inclusive scan - own total) is the exclusive bin offset; adding
            // this tile's within-bin predecessors gives the same tile_base the
            // separate scan dispatch writes in arm 1.
            uint tile_base_v = (scan[v] - bin_total[v]) + bin_before[v];
            uint rank = tile_base_v + prior;
            order[rank] = i;
            inverse[i] = rank;
        }
        """,
    ensureRowContiguous: true
)

/// Stable counting sort of `indices` (values in `[0, bins)`), returning the
/// same `(order, inverseOrder)` pair the `argSort` +
/// `inversePermutationScatterKernel` chain produces.
private func countingRouteSort(
    indices: MLXArray, count: Int, bins: Int
) -> (order: MLXArray, inverseOrder: MLXArray) {
    let tileSize = routeSortTileSize
    let tiles = (count + tileSize - 1) / tileSize

    let histInputs: [any ScalarOrArray] = [indices, Int32(count), Int32(bins)]
    let tileHist = routeTileHistKernel(
        histInputs,
        grid: (tiles * tileSize, 1, 1),
        threadGroup: (tileSize, 1, 1),
        outputShapes: [[tiles * bins]],
        outputDTypes: [.uint32]
    )[0]

    if countingRouteSortMode == 2 {
        // Two dispatches, both `tiles` threadgroups: no 1-threadgroup kernel
        // anywhere on the chain.
        let fusedInputs: [any ScalarOrArray] = [
            indices, tileHist, Int32(count), Int32(bins), Int32(tiles),
        ]
        let fused = routeScatterFusedKernel(
            fusedInputs,
            grid: (tiles * tileSize, 1, 1),
            threadGroup: (tileSize, 1, 1),
            outputShapes: [[count], [count]],
            outputDTypes: [.uint32, .uint32]
        )
        return (fused[0], fused[1])
    }

    let scanInputs: [any ScalarOrArray] = [tileHist, Int32(bins), Int32(tiles)]
    let tileBase = routeBinScanKernel(
        scanInputs,
        grid: (routeSortMaxBins, 1, 1),
        threadGroup: (routeSortMaxBins, 1, 1),
        outputShapes: [[tiles * bins]],
        outputDTypes: [.uint32]
    )[0]

    let scatterInputs: [any ScalarOrArray] = [
        indices, tileBase, Int32(count), Int32(bins),
    ]
    let scattered = routeScatterKernel(
        scatterInputs,
        grid: (tiles * tileSize, 1, 1),
        threadGroup: (tileSize, 1, 1),
        outputShapes: [[count], [count]],
        outputDTypes: [.uint32, .uint32]
    )
    return (scattered[0], scattered[1])
}

/// `expertCount` enables the counting-sort path when supplied (the caller
/// knows the routing table's key space; `nil` keeps the stock `argSort`).
public func gatherSort(
    x: MLXArray, indices: MLXArray, expertCount: Int? = nil
) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let indices = indices.flattened()

    let count = indices.size
    let bins = expertCount ?? 0
    let countingAdmissible =
        indices.dtype == .uint32
        && count > 0
        && bins > 0
        && bins <= routeSortMaxBins
    if countingRouteSortEnabled || routeSortTraceEnabled {
        routeSortTrace.fire {
            let state = (countingRouteSortEnabled && countingAdmissible) ? "active" : "inactive"
            let tiles = (count + routeSortTileSize - 1) / routeSortTileSize
            // mode 0 = flag off (stock argSort), 1 = three dispatches,
            // 2 = two dispatches. "active" already requires mode > 0.
            return """
                mlxfast: fusion \(state): counting_route_sort \
                (dispatch counting=\(countingAdmissible ? 1 : 0) \
                mode=\(countingRouteSortMode) n=\(count) bins=\(bins) \
                tiles=\(tiles) dtype=\(indices.dtype))

                """
        }
    }
    if countingRouteSortEnabled, countingAdmissible {
        let sorted = countingRouteSort(indices: indices, count: count, bins: bins)
        return (
            x.flattened(start: 0, end: -3)[sorted.order.floorDivide(m)],
            indices[sorted.order],
            sorted.inverseOrder
        )
    }

    let order = argSort(indices)
    let inverseOrder: MLXArray
    if inversePermutationScatterEnabled && order.size > 0 {
        inverseOrder = inversePermutationScatterKernel(
            [order],
            grid: (order.size, 1, 1),
            threadGroup: (min(order.size, 256), 1, 1),
            outputShapes: [[order.size]],
            outputDTypes: [.uint32]
        )[0]
    } else {
        inverseOrder = argSort(order)
    }

    return (
        x.flattened(start: 0, end: -3)[order.floorDivide(m)],
        indices[order],
        inverseOrder
    )
}

public func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]? = nil) -> MLXArray {
    var x = x[invOrder]
    if let shape {
        x = unflatten(x, axis: 0, shape: shape)
    }
    return x
}

// MARK: - SwitchGLU

public class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear?
    @ModuleInfo(key: "up_proj") var upProj: SwitchLinear?
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear?
    @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

    let inputDims: Int
    let hiddenDims: Int
    let numExperts: Int
    let activation: (MLXArray) -> MLXArray
    /// Optional fused (activation * up) kernel. Set for the default SiLU path so
    /// the GLU product runs as one compiled op; nil when a custom activation is
    /// supplied (we then fall back to `activation(gate) * up`). Upstream ef85ed0.
    let activationProduct: (@Sendable (MLXArray, MLXArray) -> MLXArray)?

    /// Activation-type flags detected once at init from a tiny test input (vMLX
    /// approach — no per-token check). Only consulted when `activationProduct` is
    /// nil (the custom-activation path): they let SiLU/GELU custom activations use
    /// the compiled `compiledSwiGLU` / `compiledGeGLU` fusions instead of the
    /// uncompiled `activation(gate) * up`. On any mismatch we fall back to that
    /// exact uncompiled path, so detection only ever enables a numerically
    /// equivalent fast path — it can never change results.
    let isSiluActivation: Bool
    let isGeluActivation: Bool

    /// Default SiLU GLU path -- uses the compiled fused (silu * up) kernel.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = MLXNN.silu
        self.activationProduct = compiledSiluProduct
        // Default path is SiLU and `activationProduct` is non-nil, so these are
        // not consulted on the hot path; set them accurately for completeness
        // (and to avoid a needless probe eval at load for every MoE layer).
        self.isSiluActivation = true
        self.isGeluActivation = false

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    /// Custom-activation GLU path -- runs `activation(gate) * up` uncompiled.
    public init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray,
        bias: Bool = false,
        fuseGateUp: Bool = false
    ) {
        self.inputDims = inputDims
        self.hiddenDims = hiddenDims
        self.numExperts = numExperts
        self.activation = activation
        self.activationProduct = nil
        // Detect SiLU/GELU once via a tiny test input (vMLX approach) so the hot
        // path can select the compiled fusion without a per-token check. Exact
        // equality is intentional: a match means the supplied closure computes
        // that exact function; any non-match falls back to `activation(gate) * up`
        // in callAsFunction, so this can only ever enable an equivalent fast path.
        let probe = MLXArray([Float(1.0)])
        let probeOut = activation(probe)
        let detectedSilu = (probeOut .== MLXNN.silu(probe)).all().item(Bool.self)
        self.isSiluActivation = detectedSilu
        self.isGeluActivation =
            !detectedSilu && (probeOut .== safeGeluApproximate(probe)).all().item(Bool.self)

        if fuseGateUp {
            self._gateUpProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims * 2, numExperts: numExperts, bias: bias)
        } else {
            self._gateProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
            self._upProj.wrappedValue = SwitchLinear(
                inputDims: inputDims, outputDims: hiddenDims, numExperts: numExperts, bias: bias)
        }
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims, outputDims: inputDims, numExperts: numExperts, bias: bias)

        super.init()
    }

    public func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            // `numExperts` is this module's own routing key space, so it is
            // exactly the counting sort's bin count. Passing it only ADMITS
            // that path; `DARKBLOOM_COUNTING_ROUTE_SORT` still gates it and
            // defaults OFF.
            (x, idx, inverseOrder) = gatherSort(
                x: x, indices: indices, expertCount: numExperts)
        }

        let xGate: MLXArray
        let xUp: MLXArray
        if let gateUpProj {
            // Pre-fused gate_up_proj weight from checkpoint — one gathered
            // matmul via the polymorphic SwitchLinear call, then split.
            let xGateUp = gateUpProj(x, idx, sortedIndices: doSort)
            xGate = xGateUp[.ellipsis, ..<hiddenDims]
            xUp = xGateUp[.ellipsis, hiddenDims...]
        } else {
            // Separate gate_proj / up_proj checkpoints — two gathered matmuls.
            guard let gateProj, let upProj else {
                fatalError("SwitchGLU requires either gate_up_proj or gate_proj/up_proj")
            }
            xUp = upProj(x, idx, sortedIndices: doSort)
            xGate = gateProj(x, idx, sortedIndices: doSort)
        }

        let activated: MLXArray
        if let activationProduct {
            activated = activationProduct(xGate, xUp)
        } else if isSiluActivation {
            activated = compiledSwiGLU(xGate, xUp)
        } else if isGeluActivation {
            activated = compiledGeGLU(xGate, xUp)
        } else {
            activated = activation(xGate) * xUp
        }

        x = downProj(activated, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }
}

public class SwitchLinear: Module, Quantizable {
    @ModuleInfo(key: "weight") var weight: MLXArray
    @ModuleInfo(key: "bias") var bias: MLXArray?

    let inputDims: Int
    let outputDims: Int
    let numExperts: Int

    public init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = true) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        let scale = sqrt(1.0 / Float(inputDims))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )

        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        }

        super.init()
    }

    /// Initializer meant for subclasses to provide weight and bias arrays directly.
    ///
    /// This is used e.g. by ``QuantizedSwitchLinear`` to provide quantized weights and biases
    /// rather than have ``SwitchLinear`` compute them.
    public init(
        inputDims: Int, outputDims: Int, numExperts: Int,
        weight: MLXArray, bias: MLXArray? = nil
    ) {
        self.inputDims = inputDims
        self.outputDims = outputDims
        self.numExperts = numExperts

        self._weight.wrappedValue = weight
        self._bias.wrappedValue = bias
    }

    public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        let weightT = self.weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

public class QuantizedSwitchLinear: SwitchLinear, Quantized {
    @ModuleInfo(key: "scales") var scales: MLXArray
    @ModuleInfo(key: "biases") var biases: MLXArray?

    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    public init(
        _ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode

        let (quantizedWeight, scales, biases) = MLX.quantized(
            other.weight, groupSize: groupSize, bits: bits, mode: mode)

        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases

        super.init(
            inputDims: other.inputDims, outputDims: other.outputDims, numExperts: other.numExperts,
            weight: quantizedWeight, bias: other.bias)

        self.freeze()
    }

    override public func callAsFunction(
        _ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool = false
    ) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            self.weight,
            scales: self.scales,
            biases: self.biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: self.groupSize,
            bits: self.bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias = self.bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}

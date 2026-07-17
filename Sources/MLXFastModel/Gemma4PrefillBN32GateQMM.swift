import Foundation
import MLX
import MLXFast

// Promoted second-stage integration of the exact BN32/BK128 affine QMM.
// The M512 route is default-on; DARKBLOOM_PREFILL_BN32_GATE=0 restores the
// compiled RMSNorm + stock gate head without constructing the candidate
// normalization-only graph or projection object.
//
// Scope is deliberately narrow: only the Gemma 4 M512 gate projection with
// affine 4-bit/group-64 weights, K=5376, and N=21504 can engage. Decode,
// last-layer M64 tail pruning, other projections, edge tiles, and nonstandard
// layouts retain the promoted implementation.
//
// PROMOTION EVIDENCE
// ------------------
// The latest-tip fair NAX-only component used identical RMSNorm-only graphs,
// forced BN64 mode-0 versus this BN32 gate, and the shared promoted M512
// BN32-up/LUT path. All normalized values, 11,010,048 gate BF16 words, and
// activated values matched exactly before timing. Across 22 balanced pairs:
//
//   BN64 median:           21.0970005 ms
//   BN32 median:           20.9053960 ms
//   paired median speedup: 1.0114916777244858x
//   paired IQR:            [1.005482480055281, 1.0172232299342543]
//   wins:                  19/22 (10/11 and 9/11 by order)
//
// This clears the predeclared >=1.01x median and >1.0 lower-quartile bars on
// physical M4 without any architecture override. It is directional local
// performance evidence; ranked M5 timing remains authoritative.
//
// Separately, a fresh-process correctness-only probe forced the stock path's
// gen17 dispatch with MLX_METAL_GPU_ARCH=applegpu_g17s on physical M4. The
// production topology passed raw BF16 equality for normalized, all gate
// words, shared-up activation, and final MLP output in 1.076 seconds. That is
// strong evidence that the candidate matches the M5-selected NAX path, but it
// does not emulate M5 hardware and must never be used for performance claims
// or a full-model benchmark.

let gemma4PrefillBN32GateEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_PREFILL_BN32_GATE"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Debug-only three-checkpoint verifier for the second-stage integration.
/// When enabled while the BN32 route is active, every eligible call also
/// executes the rollback segmented path and requires raw BF16
/// equality for (1) normalized input, (2) gate output, and (3) final MLP
/// result after the common promoted up+LUT kernel and down/post-norm tail.
let gemma4VerifyPrefillBN32GateBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_PREFILL_BN32_GATE_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

private let gemma4PrefillBN32GateSource: String = #"""
    constexpr int BM = 64;
    constexpr int BK = 128;
    constexpr int BN = 32;
    constexpr int WM = 2;
    constexpr int WN = 1;

    // CustomKernel receives the ordinary physical (N tile, M tile, batch)
    // grid. Reproduce the promoted M-major scheduling transform for exactly
    // M512: eight consecutive physical groups consume one weight tile before
    // the logical N coordinate advances. This only permutes tile ownership.
    const uint n_tiles = uint(N) / BN;
    const uint physical_linear =
        threadgroup_position_in_grid.y * n_tiles
        + threadgroup_position_in_grid.x;
    const uint3 logical_tid = uint3(
        physical_linear >> 3,
        physical_linear & 7,
        threadgroup_position_in_grid.z);

    // BN32 * (BK128 + eight BF16 pad values) = 8,704 bytes. The primitive
    // uses two SIMDgroups / 64 threads and the same M32xN32 NAX accumulator
    // tile per SIMDgroup as the promoted BN32 up projection.
    threadgroup bfloat16_t Ws[BN * 136];
    qmm_t_nax_tgp_impl<
        bfloat16_t, 64, 4, true, BM, BK, BN, WM, WN>(
        w,
        scales,
        biases,
        x,
        output,
        Ws,
        K,
        N,
        M,
        logical_tid,
        simdgroup_index_in_threadgroup * 32
            + thread_index_in_simdgroup,
        simdgroup_index_in_threadgroup,
        thread_index_in_simdgroup);
    """#

private let gemma4PrefillBN32GateKernel = MLXFast.metalKernel(
    name: "gemma4_prefill_gate_qmm_nax_bn32_bk128_v1",
    inputNames: ["w", "scales", "biases", "x", "K", "N", "M"],
    outputNames: ["output"],
    source: gemma4PrefillBN32GateSource,
    // Reuse the specialization already promoted for the common BN32-up path.
    // Swift multiline literals omit boundary newlines; retain an explicit
    // separator so the NAX stack's trailing `////` cannot comment out the
    // loader's opening `template <short dst_ld>` declaration.
    header: gemma4GeluEpilogueNAXStack
        + "\n"
        + gemma4PrefillUpBN32QuantizedLoader,
    ensureRowContiguous: true
)

/// Exact M512-only BN32 gate projection. Shape scalars are materialized once
/// per layer rather than rebuilt for every forward call; the global kernel
/// object is shared by all layers.
struct Gemma4PrefillBN32GateProjection: @unchecked Sendable {
    private static let rows = 512
    private static let inputWidth = 5_376
    private static let outputWidth = 21_504

    let weight: MLXArray
    let scales: MLXArray
    let biases: MLXArray
    private let kScalar: MLXArray
    private let nScalar: MLXArray
    private let mScalar: MLXArray

    init?(gate: FastQuantizedProjection) {
        guard gate.groupSize == 64, gate.bits == 4,
              let biases = gate.biases,
              gate.weight.dtype == .uint32,
              gate.weight.shape == [Self.outputWidth, Self.inputWidth / 8],
              gate.scales.dtype == .bfloat16,
              gate.scales.shape == [Self.outputWidth, Self.inputWidth / 64],
              biases.dtype == .bfloat16,
              biases.shape == [Self.outputWidth, Self.inputWidth / 64]
        else { return nil }

        self.weight = gate.weight
        self.scales = gate.scales
        self.biases = biases
        self.kScalar = MLXArray(Int32(Self.inputWidth))
        self.nScalar = MLXArray(Int32(Self.outputWidth))
        self.mScalar = MLXArray(Int32(Self.rows))
    }

    func supports(_ x: MLXArray) -> Bool {
        x.dtype == .bfloat16 && x.shape == [Self.rows, Self.inputWidth]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        precondition(supports(x))
        return gemma4PrefillBN32GateKernel(
            [weight, scales, biases, x, kScalar, nScalar, mScalar],
            grid: ((Self.outputWidth / 32) * 32, (Self.rows / 64) * 2, 1),
            threadGroup: (32, 2, 1),
            outputShapes: [[Self.rows, Self.outputWidth]],
            outputDTypes: [.bfloat16]
        )[0]
    }
}

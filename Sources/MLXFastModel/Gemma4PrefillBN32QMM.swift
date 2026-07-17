import Foundation
import MLX
import MLXFast

// BN32 narrow-residency plain-qmm clone for the Gemma 4 prefill projections.
//
// This extends the promoted up-projection mechanism of
// Gemma4GeluEpilogueQMM.swift (BM64xBN32xBK128, WM2xWN1, 64 threads /
// 2 SIMDgroups, 8,704 B scratch vs the stock 17,408 B) to the remaining
// prefill qmm families: the combined QKV projection, o_proj, and the
// gate/down projections inside the fused MLP tail. Each family keeps the
// stock eager/compiled surroundings; only the qmm dispatch changes.
//
// BIT-EXACTNESS CONTRACT
// ----------------------
// The kernel is a verbatim clone of the aligned fast path of the stock JIT
// twin's BK128 production instantiation
// (Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp,
// affine_qmm_t_nax's gemma4_bk128 dispatch ->
// qmm_t_nax_tgp_impl<bfloat16_t, 64, 4, aligned_N=true, 64, 128, 64, 2, 2>),
// generalized across the threadgroup-owned N tile exactly like the trusted
// gelu-epilogue clone: BN=32/WN=1 retains the same M32xN32 per-SIMDgroup
// accumulator, the same BK=128 QuantizedBlockLoader specialization spelled
// per-thread (one (row, group-64 half) pair per thread via the trusted
// gemma4PrefillUpBN32QuantizedLoader), the same ascending BK128/SK64/K16
// per-output-element K-reduction order, the same gemma4_m_major logical
// decode, and the twin's plain aligned Dtile.store / ragged-M store_safe.
// Only staging depth, barrier frequency, and which threadgroup owns which
// output tile change, so every output element is bit-identical to the stock
// qmm. The ragged-M path reproduces the twin's
// dispatch_bool(!is_unaligned_sm) -> Atile.load_safe / Dtile.store_safe
// spelling node-for-node, so non-multiple-of-64 row counts are bit-exact
// too (the production prefill engages only at L % 64 == 0 through the gelu
// tail, but the combined QKV and o_proj dispatches engage at any L > 1).
//
// Verified raw-bit equal against the stock
// affine_qmm_t_nax<bfloat16_t, 64, 4, alN=true, batch=false, 64, 64, 64, 2, 2>
// production instantiation over every prefill projection shape
// (K/N in {5376, 8192, 16384, 21504} x {5376, 16384, 18432, 21504}) at
// M = 512, at M = 64 (the tail-pruned last-layer chain and the Q-prune's
// retained-Q projection), and over ragged M above the qmv batch limit
// (direct-Metal harness, fast math off, the same JIT source assembly as the
// stock kernel, plus the GPU-gated tests in
// Tests/MLXFastTests/Gemma4PrefillBN32QMMTests.swift).
//
// Engagement requires B == 1, L >= 32, bf16, affine 4-bit group-64 layout,
// N % 64 == 0, and K in {5376, 8192, 16384, 21504} -- the fully-aligned case
// where the stock kernel takes its BK128 kAlignedN path. The L floor sits
// above every get_qmv_batch_limit branch in the frozen host dispatch (max
// 32), so the replaced stock path is always the affine_qmm_t_nax family,
// never qmv. Anything else falls back to the stock quantizedMM.

/// Default-on engagement switch. `DARKBLOOM_PREFILL_BN32_QMM=0` rolls every
/// plain-qmm BN32 dispatch back to the stock quantizedMM.
let gemma4PrefillBN32QMMEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_PREFILL_BN32_QMM"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Debug-only raw-bit verify hook: when enabled, each engaged BN32 dispatch
/// additionally computes the stock quantizedMM and preconditions the two
/// results are bit-identical.
let gemma4VerifyPrefillBN32QMMBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_PREFILL_BN32_QMM_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// The cloned aligned BK128 NAX plain-qmm implementation. Requires the full
/// gemma4GeluEpilogueNAXStack header environment plus the trusted
/// gemma4PrefillUpBN32QuantizedLoader. Textually the store epilogue of the
/// trusted gemma4_qmm_t_nax_gelu_impl reduced to the twin's plain
/// Dtile.store, with the twin's ragged-M dispatch_bool safe paths retained.
let gemma4PrefillBN32QMMPlainImpl: String = #"""
    // Clone of the aligned fast path of the stock JIT twin's BK128
    // production instantiation, generalized only across the threadgroup-owned
    // N tile (promoted BN64/WN2 or qualified BN32/WN1). Only the fully
    // N-aligned case is implemented: callers guarantee N % 64 == 0 and
    // K % 128 == 0. The gemma4_m_major logical decode from the stock entry
    // point is reproduced verbatim above the tile math.
    template <const int BN, const int WN>
    METAL_FUNC void gemma4_qmm_t_nax_plain_impl(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        device bfloat16_t* y,
        threadgroup bfloat16_t* Ws,
        const constant int& K,
        const constant int& N,
        const constant int& M,
        uint3 tid,
        uint simd_gid,
        uint simd_lid) {
      using T = bfloat16_t;
      constexpr int group_size = 64;
      constexpr int bits = 4;
      constexpr int BM = 64;
      constexpr int BK = 128;
      constexpr int WM = 2;

      static_assert(
          (BN == 64 && WN == 2) || (BN == 32 && WN == 1),
          "supported Gemma 4 plain-qmm geometries are BN64/WN2 and BN32/WN1");

      constexpr int pack_factor = get_pack_factor<bits, 8>();
      constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

      constexpr int BK_padded = (BK + 16 / sizeof(T));

      using loader_w_t = QuantizedBlockLoader<
          T,
          BN,
          BK,
          BK_padded,
          1,
          WM * WN * SIMD_SIZE,
          group_size,
          bits>;

      // The gemma4_m_major logical decode, verbatim from the stock
      // affine_qmm_t_nax entry: the physical grid stays (N tile, M tile,
      // batch) with N in the contiguous x dimension; linearize it and
      // re-decode with M tiles minor so consecutive threadgroups consume the
      // same weight tile across all M tiles before advancing N. Scheduling
      // only -- no element's accumulation order changes.
      const uint m_tiles = (uint(M) + BM - 1) / BM;
      const uint n_tiles = uint(N) / BN;
      const uint physical_linear = tid.y * n_tiles + tid.x;
      uint logical_m;
      uint logical_n;
      if (M == 512) {
        // The scored prefill has exactly eight M tiles. Avoid an integer
        // divide in every threadgroup on that hot shape.
        logical_n = physical_linear >> 3;
        logical_m = physical_linear & 7;
      } else {
        logical_n = physical_linear / m_tiles;
        logical_m = physical_linear - logical_n * m_tiles;
      }

      // Set the block
      const int K_w = K * bytes_per_pack / pack_factor;
      const int K_g = K / group_size;
      const int y_row = int(logical_m) * BM;
      const int y_col = int(logical_n) * BN;

      auto wl = (const device uint8_t*)w;

      x += y_row * static_cast<int64_t>(K);
      wl += y_col * K_w;
      scales += y_col * K_g;
      biases += y_col * K_g;
      y += y_row * static_cast<int64_t>(N) + y_col;

      // Make the weight loader (the BK=128 Gemma 4 specialization: each
      // thread owns one full group-64 quantization group).
      loader_w_t loader_w(wl, scales, biases, K, Ws, simd_gid, simd_lid);

      constexpr short SM = BM / WM;
      constexpr short SN = BN / WN;
      constexpr short SK = (BK % 64 == 0) ? 64 : 32;

      constexpr short TM = SM / 16;
      constexpr short TN = SN / 16;
      constexpr short TK = SK / 16;

      const short tm = SM * (simd_gid / WN);
      const short tn = SN * (simd_gid % WN);

      constexpr bool transpose_a = false;
      constexpr bool transpose_b = true;

      const short sgp_sm = min(int(SM), M - (y_row + tm));
      const bool is_unaligned_sm = (sgp_sm != SM);

      // aligned_N is fixed true for this clone family (engagement requires
      // N % 64 == 0), so the twin's sgp_sn/tgp_bn reduce to SN/BN.
      const short sgp_sn = SN;

      using AccumType = float;

      NAXTile<AccumType, TM, TN> Dtile;
      Dtile.clear();

      x += tm * K;

      dispatch_bool(!is_unaligned_sm, [&](auto kAlignedM) {
        for (int k = 0; k < K; k += BK) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          loader_w.load_unsafe();
          threadgroup_barrier(mem_flags::mem_threadgroup);

          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            NAXTile<T, TM, TK> Atile;
            NAXTile<T, TN, TK> Btile;

            volatile int compiler_barrier;

            if constexpr (kAlignedM.value) {
              Atile.load(x + kk1, K);
            } else {
              Atile.load_safe(x + kk1, K, short2(SK, sgp_sm));
            }

            Btile.template load<T, BK_padded, 1>(Ws + tn * BK_padded + kk1);

            tile_matmad_nax(
                Dtile,
                Atile,
                metal::bool_constant<transpose_a>{},
                Btile,
                metal::bool_constant<transpose_b>{});

            (void)compiler_barrier;
          }

          x += BK;
          loader_w.next();
        }

        // Store results to device memory
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if constexpr (kAlignedM.value) {
          Dtile.store(y + tm * N + tn, N);
        } else {
          // kAlignedN is always true in this clone family, so the twin's
          // store selection reduces to store_safe with (sgp_sn == SN,
          // sgp_sm).
          Dtile.store_safe(y + tm * N + tn, N, short2(sgp_sn, sgp_sm));
        }
      });
    }
    """#

/// Candidate body: the same M32xN32 NAX accumulator per SIMDgroup, but one N
/// SIMDgroup column and two M SIMDgroup rows. The 32x136 BF16 weight stage is
/// 8,704 bytes (stock BN64: 17,408).
private let gemma4PrefillBN32QMMSource: String = #"""
        threadgroup bfloat16_t Ws[32 * 136];
        gemma4_qmm_t_nax_plain_impl<32, 1>(
            w,
            scales,
            biases,
            x,
            output,
            Ws,
            K,
            N,
            M,
            threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
        """#

private let gemma4PrefillBN32QMMKernel = MLXFast.metalKernel(
    name: "gemma4_prefill_bn32_qmm_t_nax_bk128_v1",
    inputNames: ["w", "scales", "biases", "x", "K", "N", "M"],
    outputNames: ["output"],
    source: gemma4PrefillBN32QMMSource,
    // The NAX stack ends with a bare `//` comment line and no trailing
    // newline; the separator keeps it from swallowing the loader's
    // `template <short dst_ld>` line (the gelu header is safe only because
    // its epilogue-function comment sits at that join).
    header: gemma4GeluEpilogueNAXStack
        + "\n"
        + gemma4PrefillUpBN32QuantizedLoader
        + gemma4PrefillBN32QMMPlainImpl,
    ensureRowContiguous: true
)

/// Runs the BN32 plain-qmm clone.
///
/// - Parameters:
///   - x: `[M, K]` bf16 row-major activations; any `M >= 1` (ragged M takes
///     the twin's safe paths; production engages only at prefill `L > 1`).
///   - weight: packed affine weights `[N, K/8]` uint32.
///   - scales/biases: `[N, K/64]` bf16 affine metadata.
/// - Returns: `[M, N]` bf16 `x @ dequant(weight)^T`, bit-identical to the
///   stock `affine_qmm_t_nax` BN64 production kernel.
func gemma4PrefillBN32QMM(
    x: MLXArray,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray
) -> MLXArray {
    let M = x.shape[0]
    let K = x.shape[1]
    let N = weight.shape[0]
    precondition(M >= 1 && N % 64 == 0 && K % 128 == 0)
    precondition(x.dtype == .bfloat16 && x.ndim == 2)
    precondition(weight.dtype == .uint32 && weight.shape == [N, K / 8])
    precondition(scales.dtype == .bfloat16 && scales.shape == [N, K / 64])
    precondition(biases.dtype == .bfloat16 && biases.shape == [N, K / 64])
    let inputs = [
        weight, scales, biases, x,
        MLXArray(Int32(K)), MLXArray(Int32(N)), MLXArray(Int32(M)),
    ]
    return gemma4PrefillBN32QMMKernel(
        inputs,
        grid: ((N / 32) * 32, (M + 63) / 64, 2),
        threadGroup: (32, 1, 2),
        outputShapes: [[M, N]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// True when a projection has the standard affine 4-bit group-64 layout with
/// BN32-eligible aligned dims (N % 64 == 0, K in the gemma4_bk128 set). The K
/// restriction pins engagement to the shapes where the stock kernel takes the
/// exact BK128 instantiation this clone was verified against.
func gemma4PrefillBN32QMMSupports(
    _ projection: FastQuantizedProjection
) -> Bool {
    guard projection.groupSize == 64, projection.bits == 4,
          let biases = projection.biases,
          projection.weight.dtype == .uint32,
          projection.weight.ndim == 2,
          projection.weight.dim(0) % 64 == 0,
          [5_376, 8_192, 16_384, 21_504].contains(projection.weight.dim(1) * 8),
          projection.scales.dtype == .bfloat16,
          projection.scales.ndim == 2,
          biases.dtype == .bfloat16,
          biases.shape == projection.scales.shape,
          projection.scales.shape
              == [projection.weight.dim(0), projection.weight.dim(1) / 8]
    else {
        return false
    }
    return true
}

/// Runs `projection(x)` through the BN32 clone when engaged and eligible,
/// otherwise through the stock quantizedMM. `x` is `[B=1, L>=32, K]` bf16.
///
/// The L >= 32 floor keeps engagement above every get_qmv_batch_limit branch
/// in the frozen host dispatch (max 32), so the stock path the clone replaces
/// is always qmm_splitk -> qmm -> affine_qmm_t_nax -- never the qmv family,
/// whose per-element accumulation order differs.
///
/// With `DARKBLOOM_VERIFY_PREFILL_BN32_QMM_BITS=1` the stock result is
/// computed as well and the two are preconditioned bit-identical.
func gemma4PrefillBN32QMMDispatchIfSupported(
    _ x: MLXArray,
    projection: FastQuantizedProjection
) -> MLXArray {
    guard gemma4PrefillBN32QMMEnabled,
          x.ndim == 3, x.dim(0) == 1, x.dim(1) >= 32,
          x.dtype == .bfloat16,
          gemma4PrefillBN32QMMSupports(projection)
    else {
        return projection(x)
    }
    let L = x.dim(1)
    let K = x.dim(2)
    let N = projection.weight.dim(0)
    let out = gemma4PrefillBN32QMM(
        x: x.reshaped([L, K]),
        weight: projection.weight,
        scales: projection.scales,
        biases: projection.biases!
    ).reshaped([1, L, N])
    if gemma4VerifyPrefillBN32QMMBits {
        let reference = projection(x)
        let matches = arrayEqual(
            out.view(dtype: .uint16), reference.view(dtype: .uint16))
        eval(matches)
        precondition(
            matches.item(Bool.self),
            "prefill BN32 QMM differs from stock quantizedMM")
    }
    return out
}

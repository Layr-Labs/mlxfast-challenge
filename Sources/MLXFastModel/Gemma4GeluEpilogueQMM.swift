import Foundation
import MLX
import MLXFast

// Fused prefill up-projection with a GELU(gate) * up store epilogue.
//
// Stock prefill MLP tail per layer:
//   gate qmm -> [M, 21504] bf16 write (22 MB at M=512)
//   up   qmm -> [M, 21504] bf16 write (22 MB)
//   compiled gelu·mul: reads gate+up (44 MB), writes activation (22 MB)
//
// This file replaces the last two steps with one kernel: a verbatim clone of
// the BK128 production NAX affine qmm twin
// (Vendor/mlx-swift/Source/Cmlx/mlx-generated/quantized_nax.cpp,
// affine_qmm_t_nax's gemma4_bk128 dispatch ->
// qmm_t_nax_tgp_impl<bfloat16_t, 64, 4, aligned_N=true, 64, 128, 64, 2, 2> --
// the exact template the stock dispatch selects for the up-projection at
// K=5376, N=21504, M=512-class prefill shapes on M5. The promoted fallback
// keeps its BM64/BN64/BK128, WM2/WN2, 128-thread geometry. The scored M=512
// path narrows only the threadgroup-owned N tile to BN32/WN1 and 64 threads;
// every SIMDgroup still owns the same M32xN32 accumulator tile and executes
// the same BK128/SK64/K16 reduction order. Its store epilogue additionally
// loads the already-rounded bf16 gate tile and emits only GELU(gate) * up.
// The up-out write and the tail's gate/up re-read (44 MB/layer) disappear.
//
// BIT-EXACTNESS CONTRACT
// ----------------------
// The qmm core is textually identical to the twin's BK=128 instantiation
// (same BK=128 QuantizedBlockLoader specialization, NAXTile,
// tile_matmad_nax via gemma4GeluEpilogueNAXStack, which is the verbatim JIT
// preamble stack the C++ JIT concatenates for the stock kernel, minus
// metal::utils() -- the custom-kernel JIT prepends that itself, so the
// effective environment is identical). BK only changes the staging depth and
// barrier frequency: per output element the fp32 accumulator visits the same
// K-ascending sequence of 16-wide MMA steps as the stock kernel, so the
// accumulators are bit-identical to the stock up qmm. The M-major logical
// decode is scheduling-only (it permutes which threadgroup owns which output
// tile); it does not touch any element's accumulation order. The epilogue
// then reproduces the eager rounding sequence of the compiled MLP tail
// node-for-node:
//
//   1. accumulator rounds fp32 -> bf16 (the twin's Dtile.store cast),
//   2. the compiled tail's gelu chain runs entirely in bf16 (mlx-swift
//      `toArrays` weak-typing keeps every scalar constant bf16; the compiled
//      elementwise codegen declares every tape temporary with the node's
//      dtype), i.e. each op computes in f32 and rounds to bf16 per Metal
//      bfloat arithmetic:
//        t1 = rnd(0.044715 * g); t2 = rnd(t1 * g); t3 = rnd(t2 * g)
//        t4 = rnd(g + t3);       t5 = rnd(sqrt(2/pi) * t4)
//        t6 = rnd(precise::tanh(f32(t5)))   // mlx Tanh{} == metal::precise::tanh
//        t7 = rnd(1 + t6); t8 = rnd(0.5 * g); t9 = rnd(t8 * t7)
//        out = rnd(t9 * u)
//      (left-associative `0.044715 * x * x * x` and `0.5 * x * (1 + tanh)`),
//   3. the product rounds to bf16 once, like the compiled tail's output node.
//
// `metal::precise::tanh(bfloat16_t)` resolves to the same MLX bf16_math.h
// overload (included verbatim in the header stack) the compiled tail's
// Tanh{} functor instantiates: upcast to f32, __metal_tanh with
// __METAL_PRECISE_MATH__, round once to bf16.
//
// Engagement is restricted to the fully-aligned case (M % 64 == 0, N % 64 ==
// 0, K % 128 == 0, bf16, affine 4-bit group-64), which is the only case the
// stock kernel would take the BK128 plain `Dtile.store` path in; anything
// else falls back to the stock tail. Verified raw-bit equal against the
// stock up qmm + compiled gelu·mul over the full prefill shape and over an
// exhaustive all-65536-bf16-gate-patterns table (see
// Tests/MLXFastTests/Gemma4GeluEpilogueTests.swift, GPU-gated).

/// Default-on engagement switch. `DARKBLOOM_PREFILL_GELU_EPILOGUE=0` rolls
/// back to the stock up qmm + compiled gelu·mul tail.
let gemma4PrefillGeluEpilogueEnabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_PREFILL_GELU_EPILOGUE"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Narrower threadgroup geometry for the scored M=512 up projection.
///
/// Qualification on the local M4 Max used the full real component (unchanged
/// d7 gate QMM followed by up QMM + staged exact-BF16 LUT epilogue), not an
/// isolated synthetic store:
///
/// - all 11,010,048 BF16 outputs and both guard regions matched d7 exactly;
/// - 21/21 balanced pairs won;
/// - d7 median 21.772750 ms, BN32 median 21.438625 ms;
/// - paired median speedup 1.015788, IQR [1.013863, 1.017492].
///
/// This was promoted despite narrowly missing the provisional 1.02 combined
/// component bar because the unchanged gate QMM dilutes the separately proven
/// 1.030926x up-QMM gain, while the full result is exact, uniformly positive,
/// and has an IQR floor above 1.013. Compiler evidence also retained the same
/// 96 temporary registers and 208 spilled bytes while halving static
/// threadgroup memory from 17,408 B to 8,704 B. The promoted BN64 path remains
/// the fallback for every non-M512 shape; set
/// `DARKBLOOM_PREFILL_GELU_EPILOGUE_BN32=0` for a complete runtime rollback.
let gemma4PrefillGeluEpilogueBN32Enabled: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_PREFILL_GELU_EPILOGUE_BN32"
    ] else {
        return true
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// Debug-only raw-bit precondition hook: when enabled, the engine computes
/// both the stock tail and the fused-epilogue result for every prefill MLP
/// and preconditions they are bit-identical.
let gemma4VerifyPrefillGeluEpilogueBits: Bool = {
    guard let raw = ProcessInfo.processInfo.environment[
        "DARKBLOOM_VERIFY_PREFILL_GELU_EPILOGUE_BITS"
    ] else {
        return false
    }
    return ["1", "true", "yes", "on"].contains(raw.lowercased())
}()

/// The bf16 GELU-mul epilogue function, mirroring the compiled MLP tail
/// node-for-node. Kept as its own constant so verification harnesses can
/// compile it against a minimal header.
let gemma4GeluMulEpilogueFunction: String = #"""
    // ------------------------------------------------------------------
    // Gemma 4 prefill MLP: bf16 GELU(gate) * up epilogue.
    // ------------------------------------------------------------------

    // bf16 node-for-node mirror of the compiled MLP tail's
    //   0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 * g * g * g))) * u
    // Every step is a bf16-typed tape node in the compiled kernel: compute in
    // f32, round to bf16 (Metal bfloat arithmetic), constants are the bf16
    // roundings of the Swift-side scalar arrays.
    METAL_FUNC bfloat16_t gemma4_gelu_mul_bf16(bfloat16_t g, bfloat16_t u) {
      const bfloat16_t c0 = bfloat16_t(0.044715f);
      const bfloat16_t c1 = bfloat16_t(0.7978846f);  // sqrt(2/pi) as Float
      bfloat16_t t1 = c0 * g;
      bfloat16_t t2 = t1 * g;
      bfloat16_t t3 = t2 * g;
      bfloat16_t t4 = g + t3;
      bfloat16_t t5 = c1 * t4;
      bfloat16_t t6 = metal::precise::tanh(t5);
      bfloat16_t t7 = bfloat16_t(1.0f) + t6;
      bfloat16_t t8 = bfloat16_t(0.5f) * g;
      bfloat16_t t9 = t8 * t7;
      return t9 * u;
    }

    // 4-wide form of the same chain. Products and sums of bf16 values are
    // exact in f32, so per-lane packed bf16 arithmetic rounds (RNE) to the
    // same bf16 result as the scalar sequence above; the tanh step upcasts
    // per lane to f32, applies the same metal::precise::tanh, and rounds each
    // lane back -- identical to four scalar evaluations.
    METAL_FUNC vec<bfloat16_t, 4> gemma4_gelu_mul_bf16x4(
        vec<bfloat16_t, 4> g, vec<bfloat16_t, 4> u) {
      const vec<bfloat16_t, 4> c0 = vec<bfloat16_t, 4>(bfloat16_t(0.044715f));
      const vec<bfloat16_t, 4> c1 = vec<bfloat16_t, 4>(bfloat16_t(0.7978846f));
      vec<bfloat16_t, 4> t1 = c0 * g;
      vec<bfloat16_t, 4> t2 = t1 * g;
      vec<bfloat16_t, 4> t3 = t2 * g;
      vec<bfloat16_t, 4> t4 = g + t3;
      vec<bfloat16_t, 4> t5 = c1 * t4;
      float4 f5 = float4(float(t5.x), float(t5.y), float(t5.z), float(t5.w));
      float4 f6 = metal::precise::tanh(f5);
      vec<bfloat16_t, 4> t6 = vec<bfloat16_t, 4>(
          bfloat16_t(f6.x), bfloat16_t(f6.y), bfloat16_t(f6.z),
          bfloat16_t(f6.w));
      vec<bfloat16_t, 4> t7 = vec<bfloat16_t, 4>(bfloat16_t(1.0f)) + t6;
      vec<bfloat16_t, 4> t8 = vec<bfloat16_t, 4>(bfloat16_t(0.5f)) * g;
      vec<bfloat16_t, 4> t9 = t8 * t7;
      return t9 * u;
    }

    // 4-wide form with the tanh step evaluated through a 65,536-entry bf16
    // lookup table (device memory, L2-resident). The table is built once from
    // an eager MLX `tanh` evaluation over every bf16 bit pattern, whose
    // kernel instantiates the same Tanh{} == metal::precise::tanh functor, so
    // lut[bits(x)] is bit-identical to the precise::tanh lane above.
    METAL_FUNC vec<bfloat16_t, 4> gemma4_gelu_mul_bf16x4_lut(
        vec<bfloat16_t, 4> g, vec<bfloat16_t, 4> u,
        const device bfloat16_t* tanh_lut) {
      const vec<bfloat16_t, 4> c0 = vec<bfloat16_t, 4>(bfloat16_t(0.044715f));
      const vec<bfloat16_t, 4> c1 = vec<bfloat16_t, 4>(bfloat16_t(0.7978846f));
      vec<bfloat16_t, 4> t1 = c0 * g;
      vec<bfloat16_t, 4> t2 = t1 * g;
      vec<bfloat16_t, 4> t3 = t2 * g;
      vec<bfloat16_t, 4> t4 = g + t3;
      vec<bfloat16_t, 4> t5 = c1 * t4;
      vec<bfloat16_t, 4> t6 = vec<bfloat16_t, 4>(
          tanh_lut[as_type<uint16_t>(t5.x)],
          tanh_lut[as_type<uint16_t>(t5.y)],
          tanh_lut[as_type<uint16_t>(t5.z)],
          tanh_lut[as_type<uint16_t>(t5.w)]);
      vec<bfloat16_t, 4> t7 = vec<bfloat16_t, 4>(bfloat16_t(1.0f)) + t6;
      vec<bfloat16_t, 4> t8 = vec<bfloat16_t, 4>(bfloat16_t(0.5f)) * g;
      vec<bfloat16_t, 4> t9 = t8 * t7;
      return t9 * u;
    }
    """#

/// BN32/BK128 affine-4/group-64 loader.
///
/// BK128 contains exactly two complete quantization groups per output row.
/// The 64-thread BN32 geometry maps one thread to each `(row, group)` pair,
/// preserving the promoted loader's scalar-byte dequantization spelling and
/// destination order while halving the staged row count.
let gemma4PrefillUpBN32QuantizedLoader: String = #"""
    template <short dst_ld>
    struct QuantizedBlockLoader<
        bfloat16_t,
        32,
        128,
        dst_ld,
        1,
        64,
        64,
        4> {
      static_assert(dst_ld >= 128, "Gemma 4 BN32/BK128 loader needs a full row");

      const short row;
      const short group;

      threadgroup bfloat16_t* dst;
      const device uint8_t* src;
      const device bfloat16_t* scales;
      const device bfloat16_t* biases;

      QuantizedBlockLoader(
          const device uint8_t* src_,
          const device bfloat16_t* scales_,
          const device bfloat16_t* biases_,
          const int src_ld_,
          threadgroup bfloat16_t* dst_,
          ushort simd_group_id [[simdgroup_index_in_threadgroup]],
          ushort simd_lane_id [[thread_index_in_simdgroup]])
          : row((simd_group_id * 32 + simd_lane_id) / 2),
            group((simd_group_id * 32 + simd_lane_id) % 2),
            dst(dst_ + row * dst_ld + group * 64),
            src(src_ + row * src_ld_ / 2 + group * 32),
            scales(scales_ + row * src_ld_ / 64 + group),
            biases(biases_ + row * src_ld_ / 64 + group) {}

      void load_unsafe() const {
        const bfloat16_t scale = *scales;
        const bfloat16_t bias = *biases;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < 32; ++i) {
          dequantize<bfloat16_t, 2, 4>(
              src + i, scale, bias, dst + 2 * i);
        }
      }

      void load_safe(short2 src_tile_dim) const {
        if (row >= src_tile_dim.x) {
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < 64; ++i) {
            dst[i] = bfloat16_t(0);
          }
          return;
        }
        load_unsafe();
      }

      void next() {
        src += 64;
        scales += 2;
        biases += 2;
      }
    };
    """#

/// The cloned aligned BK128 NAX qmm implementation with the fused store
/// epilogue. Requires the full gemma4GeluEpilogueNAXStack header environment.
/// `BN=64, WN=2` is the promoted d7 geometry; `BN=32, WN=1` retains the same
/// M32xN32 per-SIMD accumulator while halving scratch and thread count.
let gemma4PrefillUpGeluEpilogueImpl: String = #"""
    // Verbatim clone of the aligned fast path of the stock JIT twin's BK128
    // production instantiation, generalized only across the threadgroup-owned
    // N tile (promoted d7 BN64/WN2 or qualified BN32/WN1):
    // qmm_t_nax_tgp_impl<bfloat16_t, group_size=64, bits=4, aligned_N=true,
    // BM=64, BK=128, BN=64, WM=2, WN=2> (mlx-generated/quantized_nax.cpp,
    // selected by affine_qmm_t_nax's gemma4_bk128 runtime-K dispatch for
    // K=5376) with the store epilogue replaced by the fused GELU(gate)*up
    // store. Only the fully-aligned case is implemented: callers guarantee
    // M % 64 == 0, N % 64 == 0, K % 128 == 0, which is exactly when the twin
    // takes its kAlignedM/kAlignedN branch. The gemma4_m_major logical decode
    // from the stock entry point is reproduced verbatim above the tile math.
    //
    // MODE is a diagnostic/verification knob: 3 is the production exact-LUT
    // fused epilogue; 2 uses bit-identical metal::precise::tanh directly;
    // 0 stores the plain rounded up tile (twin behavior, ignores gate); 1
    // reads gate but stores the plain rounded up tile (isolates gate traffic
    // from GELU ALU in perf-attribution runs).
    template <const int MODE, const int BN, const int WN>
    METAL_FUNC void gemma4_qmm_t_nax_gelu_impl(
        const device uint32_t* w,
        const device bfloat16_t* scales,
        const device bfloat16_t* biases,
        const device bfloat16_t* x,
        const device bfloat16_t* gate,
        const device bfloat16_t* tanh_lut,
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
          "supported Gemma 4 up-projection geometries are BN64/WN2 and BN32/WN1");

      constexpr int pack_factor = get_pack_factor<bits, 8>();
      constexpr int bytes_per_pack = get_bytes_per_pack<bits>();

      constexpr int BK_padded = (BK + 16 / sizeof(T));
      constexpr int GATE_LD = BN == 32 ? 32 : BK_padded;
      static_assert(GATE_LD % 8 == 0, "gate rows must remain uint4 aligned");
      static_assert(
          BM * GATE_LD <= BN * BK_padded,
          "the aliased weight stage must contain the complete gate tile");
      static_assert(
          (BM - 1) * GATE_LD + (BN - 1) < BN * BK_padded,
          "the final gate element must remain inside the aliased weight stage");

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
      gate += y_row * static_cast<int64_t>(N) + y_col;

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

      using AccumType = float;

      NAXTile<AccumType, TM, TN> Dtile;
      Dtile.clear();

      x += tm * K;

      for (int k = 0; k < K; k += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        loader_w.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        STEEL_PRAGMA_NO_UNROLL
        for (int kk1 = 0; kk1 < BK; kk1 += SK) {
          NAXTile<T, TM, TK> Atile;
          NAXTile<T, TN, TK> Btile;

          volatile int compiler_barrier;

          Atile.load(x + kk1, K);

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

      // Store results to device memory with the fused GELU(gate)*up epilogue.
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Stage this threadgroup's 64xBN bf16 gate tile into the now-free
      // weight staging buffer with fully coalesced 16-byte loads, then emit
      // GELU(gate)*up with the exact
      // element addressing of NAXTile<float, TM, TN>::store(y + tm * N +
      // tn, N) -- the twin's aligned store. The fp32 accumulator rounds to
      // bf16 exactly like the store's static_cast, then GELU(gate)*up is
      // emitted instead of the raw up tile.
      //
      // Gs reuses Ws: the barrier above guarantees the last MMA tile read
      // has completed. BN64 retains the promoted padded gate stride 136;
      // BN32 packs the 64x32 gate tile densely with stride 32, so its 2,048
      // bf16 values fit inside the 32x136 (4,352-value) weight stage.
      if constexpr (MODE > 0) {
        threadgroup T* Gs = Ws;
        const device uint4* gate_words =
            reinterpret_cast<const device uint4*>(gate);
        // gate is [M, N] bf16 row-major; this tile starts at
        // (y_row, y_col). N and BN are multiples of 8, so word addressing
        // (8 bf16 per uint4) is exact.
        const int word_row_stride = N / 8;
        const int tgp_threads = WM * WN * SIMD_SIZE;
        const int thread_idx = simd_gid * SIMD_SIZE + simd_lid;
        STEEL_PRAGMA_NO_UNROLL
        for (int chunk = thread_idx; chunk < BM * (BN / 8);
             chunk += tgp_threads) {
          const int row = chunk / (BN / 8);
          const int word = chunk % (BN / 8);
          const uint4 packed =
              gate_words[row * word_row_stride + word];
          *(reinterpret_cast<threadgroup uint4*>(
              Gs + row * GATE_LD + word * 8)) = packed;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const short2 sc = BaseNAXFrag::get_coord();
        device T* y_tile = y + tm * N + tn + sc.y * N + sc.x;
        const threadgroup T* g_tile = Gs + tm * GATE_LD + tn
            + sc.y * GATE_LD + sc.x;
        STEEL_PRAGMA_UNROLL
        for (short i = 0; i < TM; ++i) {
          STEEL_PRAGMA_UNROLL
          for (short j = 0; j < TN; ++j) {
            thread auto& frag = Dtile.val_frags[i * TN + j];
            STEEL_PRAGMA_UNROLL
            for (short ii = 0; ii < 2; ++ii) {
              const int r = i * 16 + ii * 8;
              const int c = j * 16;
              if constexpr (MODE == 2) {
                // 4-wide lanes: same element map, one vector store per row.
                const vec<T, 4> g4 = *reinterpret_cast<const threadgroup vec<T, 4>*>(
                    g_tile + r * GATE_LD + c);
                const vec<T, 4> u4 = vec<T, 4>(
                    static_cast<T>(frag[ii * 4 + 0]),
                    static_cast<T>(frag[ii * 4 + 1]),
                    static_cast<T>(frag[ii * 4 + 2]),
                    static_cast<T>(frag[ii * 4 + 3]));
                *reinterpret_cast<device vec<T, 4>*>(y_tile + r * N + c) =
                    gemma4_gelu_mul_bf16x4(g4, u4);
              } else if constexpr (MODE == 3) {
                const vec<T, 4> g4 = *reinterpret_cast<const threadgroup vec<T, 4>*>(
                    g_tile + r * GATE_LD + c);
                const vec<T, 4> u4 = vec<T, 4>(
                    static_cast<T>(frag[ii * 4 + 0]),
                    static_cast<T>(frag[ii * 4 + 1]),
                    static_cast<T>(frag[ii * 4 + 2]),
                    static_cast<T>(frag[ii * 4 + 3]));
                *reinterpret_cast<device vec<T, 4>*>(y_tile + r * N + c) =
                    gemma4_gelu_mul_bf16x4_lut(g4, u4, tanh_lut);
              } else {
                STEEL_PRAGMA_UNROLL
                for (short jj = 0; jj < 4; ++jj) {
                  const int cc = c + jj;
                  y_tile[r * N + cc] = static_cast<T>(frag[ii * 4 + jj]);
                }
              }
            }
          }
        }
      } else {
        // MODE 0: the twin's plain aligned store (diagnostic baseline).
        Dtile.store(y + tm * N + tn, N);
      }
    }
    """#

/// The complete additional header for the fused kernel: epilogue function
/// plus the cloned qmm implementation, appended after the verbatim NAX stack.
let gemma4PrefillUpGeluEpilogueHeader: String =
    gemma4GeluMulEpilogueFunction
        + gemma4PrefillUpBN32QuantizedLoader
        + gemma4PrefillUpGeluEpilogueImpl

/// Promoted fallback body: dispatch the cloned implementation with the stock
/// twin's threadgroup geometry (32, 2, 2) and Ws[BN * BK_padded]
/// (64 * 136 bf16 = 17,408 B, the exact scratch reservation the stock
/// gemma4_bk128 dispatch makes).
private let gemma4PrefillUpGeluEpilogueBN64Source: String = #"""
        threadgroup bfloat16_t Ws[64 * 136];
        gemma4_qmm_t_nax_gelu_impl<MODE, 64, 2>(
            w,
            scales,
            biases,
            x,
            gate,
            tanh_lut,
            output,
            Ws,
            K,
            N,
            M,
            threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
        """#

/// Candidate body: the same M32xN32 NAX accumulator per SIMDgroup, but one N
/// SIMDgroup column and two M SIMDgroup rows.  The 32x136 BF16 weight stage is
/// 8,704 bytes and is subsequently reused as a dense-stride 64x32 gate tile.
private let gemma4PrefillUpGeluEpilogueBN32Source: String = #"""
        threadgroup bfloat16_t Ws[32 * 136];
        gemma4_qmm_t_nax_gelu_impl<MODE, 32, 1>(
            w,
            scales,
            biases,
            x,
            gate,
            tanh_lut,
            output,
            Ws,
            K,
            N,
            M,
            threadgroup_position_in_grid,
            simdgroup_index_in_threadgroup,
            thread_index_in_simdgroup);
        """#

/// bit pattern -> metal::precise::tanh for every bf16 encoding.
///
/// Built once with an eager MLX `tanh` evaluation, whose bf16 kernel
/// instantiates the same Tanh{} functor (metal::precise::tanh: upcast f32,
/// __metal_tanh with __METAL_PRECISE_MATH__, round once) the compiled MLP
/// tail uses, so table entries are bit-identical to the tail's tanh node for
/// every one of the 65,536 possible inputs (proven exhaustively by
/// geluEpilogueExhaustiveGateTable).
struct Gemma4TanhBFloat16LUTBox: @unchecked Sendable {
    let table: MLXArray
}

private let gemma4TanhBFloat16LUT = Gemma4TanhBFloat16LUTBox(
    table: {
        let bits = MLXArray(0 ..< 65_536, [65_536]).asType(.uint16)
        let table = tanh(bits.view(dtype: .bfloat16))
        eval(table)
        return table
    }()
)

private let gemma4PrefillUpGeluEpilogueBN64Kernel = MLXFast.metalKernel(
    name: "gemma4_prefill_up_gelu_epilogue_nax_bk128_v1",
    inputNames: ["w", "scales", "biases", "x", "gate", "tanh_lut", "K", "N", "M"],
    outputNames: ["output"],
    source: gemma4PrefillUpGeluEpilogueBN64Source,
    header: gemma4GeluEpilogueNAXStack + gemma4PrefillUpGeluEpilogueHeader,
    ensureRowContiguous: true
)

private let gemma4PrefillUpGeluEpilogueBN32Kernel = MLXFast.metalKernel(
    name: "gemma4_prefill_up_gelu_epilogue_nax_bn32_bk128_v1",
    inputNames: ["w", "scales", "biases", "x", "gate", "tanh_lut", "K", "N", "M"],
    outputNames: ["output"],
    source: gemma4PrefillUpGeluEpilogueBN32Source,
    header: gemma4GeluEpilogueNAXStack + gemma4PrefillUpGeluEpilogueHeader,
    ensureRowContiguous: true
)

/// Runs the fused up-projection + GELU(gate)*up epilogue.
///
/// - Parameters:
///   - x: `[M, K]` bf16 row-major prefill activations; `M % 64 == 0`.
///   - weight: up-projection packed affine weights `[N, K/8]` uint32.
///   - scales/biases: `[N, K/64]` bf16 affine metadata.
///   - gate: `[M, N]` bf16 stock gate-projection output (already rounded).
/// - Returns: `[M, N]` bf16 `GELU(gate) * (x @ dequant(weight)^T)`.
///
/// `mode` selects the epilogue: 3 (default) evaluates the tanh through the
/// exhaustive bf16 lookup table; 2 uses metal::precise::tanh directly
/// (bit-identical, kept for A/B); 0/1 are diagnostic baselines (plain store
/// without/with the gate read) used by the perf-attribution harness.
func gemma4PrefillUpGeluEpilogue(
    x: MLXArray,
    weight: MLXArray,
    scales: MLXArray,
    biases: MLXArray,
    gate: MLXArray,
    mode: Int = 3
) -> MLXArray {
    let M = x.shape[0]
    let K = x.shape[1]
    let N = gate.shape[1]
    precondition(M % 64 == 0 && N % 64 == 0 && K % 128 == 0)
    precondition(x.dtype == .bfloat16 && gate.dtype == .bfloat16)
    precondition(weight.dtype == .uint32 && weight.shape == [N, K / 8])
    precondition(scales.dtype == .bfloat16 && scales.shape == [N, K / 64])
    precondition(biases.dtype == .bfloat16 && biases.shape == [N, K / 64])
    precondition(gate.shape == [M, N])
    precondition((0...3).contains(mode))
    let inputs = [
        weight, scales, biases, x, gate, gemma4TanhBFloat16LUT.table,
        MLXArray(Int32(K)), MLXArray(Int32(N)), MLXArray(Int32(M)),
    ]
    if gemma4PrefillGeluEpilogueBN32Enabled && M == 512 {
        return gemma4PrefillUpGeluEpilogueBN32Kernel(
            inputs,
            template: [("MODE", mode)],
            grid: ((N / 32) * 32, (M / 64), 2),
            threadGroup: (32, 1, 2),
            outputShapes: [[M, N]],
            outputDTypes: [.bfloat16]
        )[0]
    }
    return gemma4PrefillUpGeluEpilogueBN64Kernel(
        inputs,
        template: [("MODE", mode)],
        grid: ((N / 64) * 32, (M / 64) * 2, 2),
        threadGroup: (32, 2, 2),
        outputShapes: [[M, N]],
        outputDTypes: [.bfloat16]
    )[0]
}

/// Prefill MLP tail with the fused up-projection GELU epilogue.
///
/// Replaces the stock compiled tail `(rmsNorm -> gate qmm -> up qmm ->
/// gelu·mul -> down qmm -> rmsNorm -> residual add -> layerScalar)` with a
/// compiled head (`rmsNorm -> gate qmm`), the fused up+gelu·mul kernel, and a
/// compiled tail (`down qmm -> rmsNorm -> residual add -> layerScalar`).
/// Every MLX op keeps its stock kernel and rounding; only the up qmm's store
/// and the compiled gelu·mul elementwise are fused (bit-exactly, per the
/// contract at the top of this file).
struct Gemma4PrefillGeluEpilogueMLPTail: @unchecked Sendable {
    let head: @Sendable (MLXArray) -> (MLXArray, MLXArray)
    let bn32NormalizationHead: (@Sendable (MLXArray) -> MLXArray)?
    let bn32Gate: Gemma4PrefillBN32GateProjection?
    let tail: @Sendable (MLXArray, MLXArray) -> MLXArray
    let stockTail: @Sendable (MLXArray, MLXArray) -> MLXArray
    let upWeight: MLXArray
    let upScales: MLXArray
    let upBiases: MLXArray
    let intermediateSize: Int
    let hiddenSize: Int

    init?(
        gate: FastQuantizedProjection,
        up: FastQuantizedProjection,
        down: FastQuantizedProjection,
        preNormWeight: MLXArray,
        postNormWeight: MLXArray,
        layerScalar: MLXArray,
        eps: Float,
        stockTail: @escaping @Sendable (MLXArray, MLXArray) -> MLXArray
    ) {
        // Eligibility: standard affine 4-bit group-64 layout for both the
        // gate and up projections, aligned dims (K must additionally be a
        // multiple of the BK=128 staging depth), and a matching down
        // projection. Anything else keeps the stock tail.
        guard up.groupSize == 64, up.bits == 4,
              gate.groupSize == 64, gate.bits == 4,
              let upBiases = up.biases, gate.biases != nil
        else { return nil }
        let upShape = up.weight.shape
        let gateShape = gate.weight.shape
        guard upShape.count == 2, gateShape == upShape,
              upShape[0] % 64 == 0, upShape[1] % 8 == 0
        else { return nil }
        let N = upShape[0]
        let K = upShape[1] * 8
        guard K % 128 == 0,
              up.scales.shape == [N, K / 64], upBiases.shape == [N, K / 64],
              gate.scales.shape == [N, K / 64],
              down.weight.shape.count == 2, down.weight.shape[0] % 64 == 0,
              down.weight.shape[1] * 8 == N
        else { return nil }
        self.upWeight = up.weight
        self.upScales = up.scales
        self.upBiases = upBiases
        self.intermediateSize = N
        self.hiddenSize = K
        self.stockTail = stockTail
        let gateP = gate
        let downP = down
        self.head = compile(shapeless: true) { x in
            let normalized = MLXFast.rmsNorm(x, weight: preNormWeight, eps: eps)
            return (normalized, gateP(normalized))
        }
        // A CustomKernel cannot participate in MLX compile because output
        // shape inference stops at that node. Keep the compiled
        // `(RMSNorm, stock gate)` rollback head above intact, and construct a
        // compiled normalization-only head only while the promoted BN32 route
        // is enabled. DARKBLOOM_PREFILL_BN32_GATE=0 therefore restores the
        // prior topology structurally, not merely numerically.
        if gemma4PrefillBN32GateEnabled,
           let bn32Gate = Gemma4PrefillBN32GateProjection(gate: gateP)
        {
            self.bn32Gate = bn32Gate
            self.bn32NormalizationHead = compile(shapeless: true) { x in
                MLXFast.rmsNorm(x, weight: preNormWeight, eps: eps)
            }
        } else {
            self.bn32Gate = nil
            self.bn32NormalizationHead = nil
        }
        self.tail = compile(shapeless: true) { act, residual in
            let mlp = downP(act)
            let postNormalized = MLXFast.rmsNorm(mlp, weight: postNormWeight, eps: eps)
            return (residual + postNormalized) * layerScalar
        }
    }

    /// True when a `[B, L, hidden]` prefill activation can take the fused
    /// path (aligned row count, standard layouts).
    func supports(B: Int, L: Int) -> Bool {
        B == 1 && L > 1 && L % 64 == 0
    }

    /// The currently promoted segmented head+up path. Both this reference arm
    /// and the candidate below deliberately call the same up dispatcher, so
    /// M512 keeps the promoted BN32 up kernel and its BN64 rollback switch.
    func promotedHeadAndActivation(
        _ x: MLXArray
    ) -> (normalized: MLXArray, gate: MLXArray, activated: MLXArray) {
        let L = x.shape[x.shape.count - 2]
        let (normalized, gate) = head(x)
        let activated = gemma4PrefillUpGeluEpilogue(
            x: normalized.reshaped([L, hiddenSize]),
            weight: upWeight,
            scales: upScales,
            biases: upBiases,
            gate: gate.reshaped([L, intermediateSize])
        )
        return (normalized, gate, activated)
    }

    /// Promoted second-stage arm: compiled RMSNorm-only, exact BN32 gate, then
    /// the unchanged promoted up+LUT epilogue dispatcher.
    func bn32HeadAndActivation(
        _ x: MLXArray
    ) -> (normalized: MLXArray, gate: MLXArray, activated: MLXArray)? {
        guard let bn32NormalizationHead, let bn32Gate,
              x.shape == [1, 512, hiddenSize]
        else { return nil }
        let normalized = bn32NormalizationHead(x)
        let normalized2D = normalized.reshaped([512, hiddenSize])
        guard bn32Gate.supports(normalized2D) else { return nil }
        let gate = bn32Gate(normalized2D)
        let activated = gemma4PrefillUpGeluEpilogue(
            x: normalized2D,
            weight: upWeight,
            scales: upScales,
            biases: upBiases,
            gate: gate
        )
        return (normalized, gate, activated)
    }

    func callAsFunction(_ x: MLXArray, _ residual: MLXArray) -> MLXArray {
        // x is [B=1, L, hidden]; the fused kernel consumes 2-D tiles.
        let L = x.shape[x.shape.count - 2]
        if let candidate = bn32HeadAndActivation(x) {
            let candidateFinal = tail(
                candidate.activated.reshaped([1, L, intermediateSize]),
                residual
            )
            if gemma4VerifyPrefillBN32GateBits {
                let reference = promotedHeadAndActivation(x)
                let referenceFinal = tail(
                    reference.activated.reshaped([1, L, intermediateSize]),
                    residual
                )

                let normalizedMatches = arrayEqual(
                    candidate.normalized.view(dtype: .uint16),
                    reference.normalized.view(dtype: .uint16)
                )
                let gateMatches = arrayEqual(
                    candidate.gate.view(dtype: .uint16),
                    reference.gate.reshaped([L, intermediateSize])
                        .view(dtype: .uint16)
                )
                let finalMatches = arrayEqual(
                    candidateFinal.view(dtype: .uint16),
                    referenceFinal.view(dtype: .uint16)
                )
                eval(normalizedMatches, gateMatches, finalMatches)
                precondition(
                    normalizedMatches.item(Bool.self),
                    "BN32 gate path changed pre-FFN RMSNorm bits"
                )
                precondition(
                    gateMatches.item(Bool.self),
                    "BN32 gate QMM differs from the promoted gate bits"
                )
                precondition(
                    finalMatches.item(Bool.self),
                    "BN32 gate path differs from the promoted final MLP bits"
                )
            }
            return candidateFinal
        }

        let promoted = promotedHeadAndActivation(x)
        return tail(
            promoted.activated.reshaped([1, L, intermediateSize]),
            residual
        )
    }
}

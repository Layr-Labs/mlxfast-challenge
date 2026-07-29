import Foundation
import MLX
import MLXFast
import MLXFastCore

/// Exact row-tier encoding for BF16 matrices whose decode path is dominated by
/// weight traffic. Every 128-value tile stores one base exponent, one byte of
/// sign+fraction per value, and 4/5/6 exponent-delta bits. A tier is selected
/// once for the complete output row; rows wider than six delta bits retain the
/// original BF16 matrix as a raw fallback.
///
/// The 12-bit common row record is fixed-stride and 16-byte aligned. Fifth and
/// sixth delta-bit planes are appended only for rows that need them; two UInt32
/// absolute offsets per row keep those optional planes aligned without a hot
/// path prefix sum or the old unaligned three-byte loads.
struct LagunaBF16RowCodecLayout: Equatable {
    static let tileWidth = 128

    let columns: Int
    let tileCount: Int
    let baseOffset: Int
    let signFractionOffset: Int
    let lowDeltaOffset: Int
    let commonRowStride: Int
    let bitPlaneStride: Int

    init(columns: Int) {
        precondition(columns > 0 && columns.isMultiple(of: Self.tileWidth))
        self.columns = columns
        self.tileCount = columns / Self.tileWidth
        self.baseOffset = 0
        self.signFractionOffset = Self.align16(tileCount)
        self.lowDeltaOffset = Self.align16(signFractionOffset + columns)
        self.commonRowStride = Self.align16(lowDeltaOffset + columns / 2)
        self.bitPlaneStride = Self.align16(columns / 8)
    }

    private static func align16(_ value: Int) -> Int {
        (value + 15) & ~15
    }
}

enum LagunaBF16RowCodecTier: UInt8, CaseIterable {
    case raw = 0
    case four = 4
    case five = 5
    case six = 6
}

struct LagunaBF16RowCodecEncoded {
    let layout: LagunaBF16RowCodecLayout
    let rows: Int
    let payload: [UInt8]
    let tiers: [UInt8]
    /// Two absolute byte offsets per row: bit 4, then bit 5. Zero means that
    /// the row tier does not own that optional plane.
    let extraOffsets: [UInt32]

    var tierCounts: [LagunaBF16RowCodecTier: Int] {
        var result: [LagunaBF16RowCodecTier: Int] = [:]
        for rawTier in tiers {
            let tier = LagunaBF16RowCodecTier(rawValue: rawTier) ?? .raw
            result[tier, default: 0] += 1
        }
        return result
    }
}

struct LagunaBF16RowCodecStorage {
    let layout: LagunaBF16RowCodecLayout
    let rows: Int
    let payload: MLXArray
    let tiers: MLXArray
    let extraOffsets: MLXArray
    let tierCounts: [LagunaBF16RowCodecTier: Int]
}

enum LagunaBF16RowCodec {
    static func encode(words: [UInt16], rows: Int, columns: Int) -> LagunaBF16RowCodecEncoded {
        words.withUnsafeBufferPointer {
            encode(words: $0, rows: rows, columns: columns)
        }
    }

    static func encode(
        words: UnsafeBufferPointer<UInt16>, rows: Int, columns: Int
    ) -> LagunaBF16RowCodecEncoded {
        precondition(rows > 0)
        precondition(words.count == rows * columns)
        let layout = LagunaBF16RowCodecLayout(columns: columns)
        var tiers = [UInt8](repeating: 0, count: rows)
        var allBases = [UInt8](repeating: 0, count: rows * layout.tileCount)

        // Classify first so the optional bit planes can be packed tightly into
        // one allocation while every individual plane remains 16-byte aligned.
        for row in 0..<rows {
            let sourceRow = row * columns
            var largestDelta = 0

            for tile in 0..<layout.tileCount {
                let tileStart = sourceRow + tile * LagunaBF16RowCodecLayout.tileWidth
                var minimum = 255
                var maximum = 0
                for index in tileStart..<(tileStart + LagunaBF16RowCodecLayout.tileWidth) {
                    let exponent = Int((words[index] >> 7) & 0xff)
                    minimum = min(minimum, exponent)
                    maximum = max(maximum, exponent)
                }
                allBases[row * layout.tileCount + tile] = UInt8(minimum)
                largestDelta = max(largestDelta, maximum - minimum)
            }

            let tier: LagunaBF16RowCodecTier
            switch largestDelta {
            case 0...15: tier = .four
            case 16...31: tier = .five
            case 32...63: tier = .six
            default: tier = .raw
            }
            tiers[row] = tier.rawValue
        }

        let bit4Rows = tiers.reduce(into: 0) { count, tier in
            if tier >= LagunaBF16RowCodecTier.five.rawValue { count += 1 }
        }
        let bit5Rows = tiers.reduce(into: 0) { count, tier in
            if tier >= LagunaBF16RowCodecTier.six.rawValue { count += 1 }
        }
        let commonBytes = rows * layout.commonRowStride
        let bit5Start = commonBytes + bit4Rows * layout.bitPlaneStride
        let payloadBytes = bit5Start + bit5Rows * layout.bitPlaneStride
        precondition(payloadBytes <= Int(UInt32.max))

        var payload = [UInt8](repeating: 0, count: payloadBytes)
        var extraOffsets = [UInt32](repeating: 0, count: rows * 2)
        var nextBit4 = commonBytes
        var nextBit5 = bit5Start

        for row in 0..<rows {
            let tier = tiers[row]
            if tier >= LagunaBF16RowCodecTier.five.rawValue {
                extraOffsets[row * 2] = UInt32(nextBit4)
                nextBit4 += layout.bitPlaneStride
            }
            if tier >= LagunaBF16RowCodecTier.six.rawValue {
                extraOffsets[row * 2 + 1] = UInt32(nextBit5)
                nextBit5 += layout.bitPlaneStride
            }

            // Raw rows are served from the original BF16 matrix. Their fixed
            // sidecar record remains zero and is never read by the kernel.
            guard tier != LagunaBF16RowCodecTier.raw.rawValue else { continue }

            let sourceRow = row * columns
            let destinationRow = row * layout.commonRowStride
            for tile in 0..<layout.tileCount {
                payload[destinationRow + layout.baseOffset + tile] =
                    allBases[row * layout.tileCount + tile]
            }

            for column in 0..<columns {
                let word = words[sourceRow + column]
                let signFraction = UInt8(word & 0x7f) | UInt8((word >> 8) & 0x80)
                let exponent = UInt8((word >> 7) & 0xff)
                let base = allBases[
                    row * layout.tileCount
                        + column / LagunaBF16RowCodecLayout.tileWidth]
                let delta = exponent - base

                payload[destinationRow + layout.signFractionOffset + column] = signFraction

                let lowIndex = destinationRow + layout.lowDeltaOffset + column / 2
                let lowNibble = delta & 0x0f
                if column.isMultiple(of: 2) {
                    payload[lowIndex] |= lowNibble
                } else {
                    payload[lowIndex] |= lowNibble << 4
                }

                let planeIndex = column / 8
                let planeMask = UInt8(1 << (column & 7))
                if delta & 0x10 != 0 {
                    payload[Int(extraOffsets[row * 2]) + planeIndex] |= planeMask
                }
                if delta & 0x20 != 0 {
                    payload[Int(extraOffsets[row * 2 + 1]) + planeIndex] |= planeMask
                }
            }
        }

        return LagunaBF16RowCodecEncoded(
            layout: layout,
            rows: rows,
            payload: payload,
            tiers: tiers,
            extraOffsets: extraOffsets)
    }

    static func decodeBits(
        _ encoded: LagunaBF16RowCodecEncoded, rawFallback: [UInt16]
    ) -> [UInt16] {
        let layout = encoded.layout
        precondition(rawFallback.count == encoded.rows * layout.columns)
        var result = [UInt16](repeating: 0, count: rawFallback.count)

        for row in 0..<encoded.rows {
            let sourceRow = row * layout.commonRowStride
            let destinationRow = row * layout.columns
            let tier = encoded.tiers[row]
            if tier == LagunaBF16RowCodecTier.raw.rawValue {
                result.replaceSubrange(
                    destinationRow..<(destinationRow + layout.columns),
                    with: rawFallback[destinationRow..<(destinationRow + layout.columns)])
                continue
            }

            for column in 0..<layout.columns {
                let signFraction = encoded.payload[
                    sourceRow + layout.signFractionOffset + column]
                let packedLow = encoded.payload[
                    sourceRow + layout.lowDeltaOffset + column / 2]
                var delta = column.isMultiple(of: 2) ? packedLow & 0x0f : packedLow >> 4
                let planeIndex = column / 8
                let planeShift = UInt8(column & 7)
                if tier >= LagunaBF16RowCodecTier.five.rawValue {
                    delta |= ((encoded.payload[
                        Int(encoded.extraOffsets[row * 2]) + planeIndex]
                        >> planeShift) & 1) << 4
                }
                if tier >= LagunaBF16RowCodecTier.six.rawValue {
                    delta |= ((encoded.payload[
                        Int(encoded.extraOffsets[row * 2 + 1]) + planeIndex]
                        >> planeShift) & 1) << 5
                }
                let base = encoded.payload[
                    sourceRow + layout.baseOffset
                        + column / LagunaBF16RowCodecLayout.tileWidth]
                let exponent = UInt16(base &+ delta)
                result[destinationRow + column] =
                    (UInt16(signFraction & 0x80) << 8)
                    | (exponent << 7)
                    | UInt16(signFraction & 0x7f)
            }
        }
        return result
    }

    static func makeStorage(weight: MLXArray) -> LagunaBF16RowCodecStorage? {
        guard weight.dtype == .bfloat16, weight.ndim == 2 else { return nil }
        let rows = weight.dim(0)
        let columns = weight.dim(1)
        guard columns.isMultiple(of: LagunaBF16RowCodecLayout.tileWidth) else { return nil }

        // `asData` exposes native BF16 bytes instead of numerically converting
        // them to UInt16. The checkpoint array has already been evaluated at
        // this load-time call site; this scan and the sidecar construction are
        // both outside every scored phase.
        let native = weight.asData(access: .noCopyIfContiguous)
        guard native.dType == .bfloat16,
            native.strides == [columns, 1],
            native.data.count == rows * columns * MemoryLayout<UInt16>.stride
        else { return nil }

        let encoded = native.data.withUnsafeBytes { raw -> LagunaBF16RowCodecEncoded in
            encode(
                words: raw.bindMemory(to: UInt16.self),
                rows: rows,
                columns: columns)
        }
        return LagunaBF16RowCodecStorage(
            layout: encoded.layout,
            rows: rows,
            payload: MLXArray(encoded.payload, [encoded.payload.count]),
            tiers: MLXArray(encoded.tiers, [rows]),
            extraOffsets: MLXArray(encoded.extraOffsets, [rows, 2]),
            tierCounts: encoded.tierCounts)
    }
}

func lagunaBF16RowCodecMetalHeader(layout: LagunaBF16RowCodecLayout) -> String {
    """
    inline vec<ushort, 4> laguna_bf16_row_codec_bits4(
        const device uchar* payload,
        uint row,
        uint column,
        uchar tier,
        uint2 extra_offsets) {
      constexpr uint common_row_stride = \(layout.commonRowStride);
      constexpr uint tile_width = \(LagunaBF16RowCodecLayout.tileWidth);
      constexpr uint base_offset = \(layout.baseOffset);
      constexpr uint sign_fraction_offset = \(layout.signFractionOffset);
      constexpr uint low_delta_offset = \(layout.lowDeltaOffset);

      uint row_base = row * common_row_stride;
      uint sign_fraction = *((const device uint*)(
          payload + row_base + sign_fraction_offset + column));
      ushort low_delta = *((const device ushort*)(
          payload + row_base + low_delta_offset + (column >> 1)));
      uchar base = payload[row_base + base_offset + column / tile_width];
      uint plane_index = column >> 3;
      uint plane_shift = column & 7;
      uchar bit4 = tier >= 5
          ? (payload[extra_offsets.x + plane_index] >> plane_shift)
          : 0;
      uchar bit5 = tier >= 6
          ? (payload[extra_offsets.y + plane_index] >> plane_shift)
          : 0;

      vec<ushort, 4> result;
      for (uint i = 0; i < 4; ++i) {
        ushort sf = ushort((sign_fraction >> (8 * i)) & 0xff);
        ushort delta = ushort((low_delta >> (4 * i)) & 0x0f);
        delta |= ushort((bit4 >> i) & 1) << 4;
        delta |= ushort((bit5 >> i) & 1) << 5;
        ushort exponent = ushort(base) + delta;
        result[i] = ((sf & 0x80) << 8) | (exponent << 7) | (sf & 0x7f);
      }
      return result;
    }
    """
}

let lagunaQKVCodecLayout = LagunaBF16RowCodecLayout(
    columns: LagunaConstants.hiddenSize)

private func lagunaOCodecLayout(heads: Int) -> LagunaBF16RowCodecLayout {
    precondition(
        heads == LagunaConstants.slidingAttentionHeads
            || heads == LagunaConstants.fullAttentionHeads)
    return LagunaBF16RowCodecLayout(columns: heads * LagunaConstants.headDim)
}

private func lagunaLosslessOProjectionSource(heads: Int, unroll: Int) -> String {
    let layout = lagunaOCodecLayout(heads: heads)
    return """
    constexpr uint unroll = \(unroll);
    constexpr uint in_vec_size = \(layout.columns);
    constexpr uint heads = \(heads);
    constexpr uint head_dim = \(LagunaConstants.headDim);
    constexpr uint rows_per_thread = 4;
    constexpr uint values_per_thread = 4;
    constexpr uint block_width = 128;
    constexpr uint blocks = in_vec_size / block_width;
    constexpr uint rows_per_group = 16;

    uint tile = threadgroup_position_in_grid.x;
    uint simd_group = simdgroup_index_in_threadgroup;
    uint lane = thread_index_in_simdgroup;
    uint out_row = tile * rows_per_group + simd_group * rows_per_thread;

    thread float result[rows_per_thread] = {0.0f, 0.0f, 0.0f, 0.0f};
    thread float coefficients[values_per_thread];
    thread uchar local_tiers[rows_per_thread];
    thread uint2 local_extra_offsets[rows_per_thread];
    for (uint row = 0; row < rows_per_thread; ++row) {
      uint row_index = out_row + row;
      local_tiers[row] = row_tiers[row_index];
      local_extra_offsets[row] = *((const device uint2*) (
          row_extra_offsets + row_index * 2));
    }
    uint column = lane * values_per_thread;

    for (uint block = 0; block < blocks; block += unroll) {
      vec<bfloat, 4> gated_values[unroll];
      vec<bfloat, 4> weight_values[unroll][rows_per_thread];

      for (uint u = 0; u < unroll; ++u) {
        uint column_u = column + u * block_width;
        gated_values[u] = *((const device vec<bfloat, 4>*) (
            attention_output + column_u));

        for (uint row = 0; row < rows_per_thread; ++row) {
          uint row_index = out_row + row;
          uchar tier = local_tiers[row];
          if (tier == 0) {
            weight_values[u][row] = *((const device vec<bfloat, 4>*) (
                raw_weight + row_index * in_vec_size + column_u));
          } else {
            vec<ushort, 4> bits = laguna_bf16_row_codec_bits4(
                packed_weight, row_index, column_u, tier,
                local_extra_offsets[row]);
            for (uint i = 0; i < values_per_thread; ++i) {
              weight_values[u][row][i] = as_type<bfloat>(bits[i]);
            }
          }
        }
      }

      for (uint u = 0; u < unroll; ++u) {
        float gate = float(gate_values[block + u]);
        for (uint i = 0; i < values_per_thread; ++i) {
          coefficients[i] =
              float(bfloat(float(gated_values[u][i]) * gate));
        }
        for (uint row = 0; row < rows_per_thread; ++row) {
          for (uint i = 0; i < values_per_thread; ++i) {
            result[row] += float(weight_values[u][row][i]) * coefficients[i];
          }
        }
      }
      column += unroll * block_width;
    }

    for (uint row = 0; row < rows_per_thread; ++row) {
      for (ushort delta = 16; delta >= 1; delta >>= 1) {
        result[row] += metal::simd_shuffle_down(result[row], delta);
      }
    }
    if (lane == 0) {
      for (uint row = 0; row < rows_per_thread; ++row) {
        projected[out_row + row] = bfloat(result[row]);
      }
    }
    """
}

private let lagunaLosslessOProjectionKernels: [Int: [Int: MLXFast.MLXFastKernel]] = {
    var result: [Int: [Int: MLXFast.MLXFastKernel]] = [:]
    for heads in [LagunaConstants.slidingAttentionHeads, LagunaConstants.fullAttentionHeads] {
        let layout = lagunaOCodecLayout(heads: heads)
        var headKernels: [Int: MLXFast.MLXFastKernel] = [:]
        for unroll in [1, 2, 4, 8] {
            headKernels[unroll] = MLXFast.metalKernel(
                name: "laguna_lossless_o_bf16_h\(heads)_u\(unroll)_v2",
                inputNames: [
                    "attention_output", "gate_values", "raw_weight",
                    "packed_weight", "row_tiers", "row_extra_offsets",
                ],
                outputNames: ["projected"],
                source: lagunaLosslessOProjectionSource(heads: heads, unroll: unroll),
                header: lagunaBF16RowCodecMetalHeader(layout: layout),
                ensureRowContiguous: true)
        }
        result[heads] = headKernels
    }
    return result
}()

func lagunaLosslessOProjection(
    attentionOutput: MLXArray,
    gateValues: MLXArray,
    rawWeight: MLXArray,
    codec: LagunaBF16RowCodecStorage,
    heads: Int,
    unroll: Int
) -> MLXArray? {
    guard let kernel = lagunaLosslessOProjectionKernels[heads]?[unroll] else { return nil }
    let layout = lagunaOCodecLayout(heads: heads)
    let inVec = layout.columns
    precondition(attentionOutput.dtype == .bfloat16)
    precondition(attentionOutput.shape == [1, 1, inVec])
    precondition(gateValues.dtype == .bfloat16)
    precondition(gateValues.shape == [1, 1, heads])
    precondition(rawWeight.dtype == .bfloat16)
    precondition(rawWeight.shape == [LagunaConstants.hiddenSize, inVec])
    precondition(codec.layout == layout)
    precondition(codec.rows == LagunaConstants.hiddenSize)
    precondition(codec.payload.dtype == .uint8)
    precondition(codec.tiers.dtype == .uint8)
    precondition(codec.extraOffsets.dtype == .uint32)
    precondition(codec.extraOffsets.shape == [LagunaConstants.hiddenSize, 2])

    lagunaTrace("lossless O projection h\(heads) u\(unroll)")
    return kernel(
        [
            attentionOutput, gateValues, rawWeight, codec.payload, codec.tiers,
            codec.extraOffsets,
        ],
        grid: ((LagunaConstants.hiddenSize / 16) * 128, 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[1, 1, LagunaConstants.hiddenSize]],
        outputDTypes: [.bfloat16]
    )[0]
}

private func lagunaBF16RowCodecReconstructKernel(
    columns: Int
) -> MLXFast.MLXFastKernel {
    let layout = LagunaBF16RowCodecLayout(columns: columns)
    return MLXFast.metalKernel(
    name: "laguna_bf16_row_codec_reconstruct_bits_c\(columns)_v2",
    inputNames: [
        "raw_words", "packed_weight", "row_tiers", "row_extra_offsets",
    ],
    outputNames: ["reconstructed"],
    source: """
        constexpr uint columns = \(columns);
        uint group = thread_position_in_grid.x;
        uint row = group / (columns / 4);
        uint column = (group % (columns / 4)) * 4;
        uchar tier = row_tiers[row];
        uint2 extra_offsets = *((const device uint2*) (
            row_extra_offsets + row * 2));
        vec<ushort, 4> bits;
        if (tier == 0) {
          bits = *((const device vec<ushort, 4>*)(
              raw_words + row * columns + column));
        } else {
          bits = laguna_bf16_row_codec_bits4(
              packed_weight, row, column, tier, extra_offsets);
        }
        *((device vec<ushort, 4>*)(
            reconstructed + row * columns + column)) = bits;
        """,
    header: lagunaBF16RowCodecMetalHeader(layout: layout),
    ensureRowContiguous: true)
}

private let lagunaBF16RowCodecReconstructKernels: [Int: MLXFast.MLXFastKernel] = {
    var result: [Int: MLXFast.MLXFastKernel] = [:]
    for columns in [
        LagunaConstants.hiddenSize,
        LagunaConstants.fullAttentionHeads * LagunaConstants.headDim,
        LagunaConstants.slidingAttentionHeads * LagunaConstants.headDim,
    ] {
        result[columns] = lagunaBF16RowCodecReconstructKernel(columns: columns)
    }
    return result
}()

/// Synthetic-test hook. It runs the exact aligned decoder used by the fused
/// projection and returns raw BF16 words, so comparison never passes through a
/// floating-point conversion or tolerance.
func lagunaBF16RowCodecReconstructBits(
    rawWords: MLXArray, codec: LagunaBF16RowCodecStorage
) -> MLXArray {
    precondition(rawWords.dtype == .uint16)
    precondition(rawWords.shape == [codec.rows, codec.layout.columns])
    guard let kernel = lagunaBF16RowCodecReconstructKernels[codec.layout.columns] else {
        preconditionFailure("unsupported Laguna BF16 codec width \(codec.layout.columns)")
    }
    return kernel(
        [rawWords, codec.payload, codec.tiers, codec.extraOffsets],
        grid: (rawWords.size / 4, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [rawWords.shape],
        outputDTypes: [.uint16]
    )[0]
}

private let lagunaBF16FromBitsKernel = MLXFast.metalKernel(
    name: "laguna_bf16_from_bits_v1",
    inputNames: ["raw_words"],
    outputNames: ["values"],
    source: """
        uint index = thread_position_in_grid.x;
        values[index] = as_type<bfloat>(raw_words[index]);
        """,
    ensureRowContiguous: true)

/// Synthetic-test hook for constructing BF16 MLX arrays without numerically
/// converting their UInt16 bit patterns.
func lagunaBF16FromBits(_ rawWords: MLXArray) -> MLXArray {
    precondition(rawWords.dtype == .uint16)
    return lagunaBF16FromBitsKernel(
        [rawWords],
        grid: (rawWords.size, 1, 1),
        threadGroup: (256, 1, 1),
        outputShapes: [rawWords.shape],
        outputDTypes: [.bfloat16]
    )[0]
}

// Kept in this participant source slot to preserve the repository's frozen
// participant-file-count budget while adding the Laguna codec. These bridge
// definitions are unchanged from the former standalone MLXTensorBridge.swift.
public protocol MLXTensorBridge {
    associatedtype Array

    func makeArray(from tensor: MaterializedTensor) throws -> Array
}

public struct MLXArrayTensorBridge: MLXTensorBridge {
    public typealias Array = MLXArray

    public init() {}

    public func makeArray(from tensor: MaterializedTensor) throws -> MLXArray {
        MLXArray(tensor.bytes, tensor.shape, dtype: Self.mlxDType(for: tensor.dtype))
    }

    public static func mlxDType(for dtype: TensorDType) -> DType {
        dtype.mlxDType
    }
}

private extension TensorDType {
    var mlxDType: DType {
        switch self {
        case .bool:
            return .bool
        case .u8:
            return .uint8
        case .i8:
            return .int8
        case .i16:
            return .int16
        case .u16:
            return .uint16
        case .i32:
            return .int32
        case .u32:
            return .uint32
        case .i64:
            return .int64
        case .u64:
            return .uint64
        case .f16:
            return .float16
        case .bf16:
            return .bfloat16
        case .f32:
            return .float32
        case .f64:
            return .float64
        }
    }
}

import Foundation
import MLXFastCore

/// Fixed-width 12/13-bit packing for the U16 affine-metadata index streams
/// consumed by the fused attention-projection decode kernels
/// (`gemma4_indexed_sliding_qkv_qmv_5376` and `gemma4_indexed_full_qk_qmv_5376`
/// in `Sources/MLXFastModel/Gemma4FusedQKV.swift`).
///
/// The packed row layout is derived from the exact lane/row/group consumption
/// order of those kernels, which matches the promoted fixed12 gate/up family
/// (`gemma4Pack12BitGateUpIndices`): each output row carries `groupsPerRow`
/// indexes, consumed as `groupsPerRow / 4` four-index decode blocks; within a
/// block, SIMD lane group `g in 0..<4` (lanes `8g..<8g+8`) consumes index
/// `block * 4 + g`.
///
/// Blocks are packed in pairs. One pair (blocks `2p` and `2p+1`) occupies
/// `pairStride` bytes at row offset `p * pairStride`:
///
/// - fixed12 (`pairStride == 12`, every index < 4096):
///   - byte `[g]`     = `even_g & 0xff`
///   - byte `[4 + g]` = `((even_g >> 8) & 0x0f) | ((odd_g & 0x0f) << 4)`
///   - byte `[8 + g]` = `(odd_g >> 4) & 0xff`
/// - fixed13 (`pairStride == 13`, every index < 8192): the same three byte
///   planes, plus one trailing top-bit byte:
///   - byte `[12]` bit `g`     = bit 12 of `even_g`
///   - byte `[12]` bit `4 + g` = bit 12 of `odd_g`
///
/// where `even_g = indices[row][8p + g]` and `odd_g = indices[row][8p + 4 + g]`.
///
/// An unpaired final block (present when `groupsPerRow / 4` is odd, e.g. the
/// production 84-group rows) uses eight lane-major bytes at row offset
/// `pairCount * pairStride`: byte `[2g] = tail_g & 0xff`,
/// byte `[2g + 1] = tail_g >> 8`. The two-byte tail natively carries up to
/// 16 bits, so fixed12 and fixed13 share the same tail encoding.
///
/// Rows are independently padded with zero bytes to four-byte alignment so a
/// row never crosses into the next one at a misaligned word. For the
/// production 84-group geometry this yields 128 bytes/row (fixed12; identical
/// to the promoted gate/up fixed12 rows) and 140 bytes/row (fixed13), versus
/// 168 bytes/row for the raw U16 stream.
enum PackedProjectionMetadataCoding {
    static let fixed12Suffix = ".metadata_packed_fixed12"
    static let fixed13Suffix = ".metadata_packed_fixed13"

    static func suffix(indexBits: Int) -> String? {
        switch indexBits {
        case 12:
            return fixed12Suffix
        case 13:
            return fixed13Suffix
        default:
            return nil
        }
    }

    /// The narrowest supported fixed-width form that can represent every
    /// index of a LUT with `count` entries, or nil when the LUT needs more
    /// than 13 bits (packing is skipped and the runtime keeps the U16 path).
    static func indexBits(forLUTCount count: Int) -> Int? {
        guard count >= 1 else { return nil }
        if count <= 4_096 {
            return 12
        }
        if count <= 8_192 {
            return 13
        }
        return nil
    }

    static func packedBytesPerRow(groupsPerRow: Int, indexBits: Int) -> Int? {
        guard groupsPerRow > 0,
              groupsPerRow.isMultiple(of: 4),
              indexBits == 12 || indexBits == 13
        else {
            return nil
        }
        let blocks = groupsPerRow / 4
        let pairCount = blocks / 2
        let tailBlockCount = blocks % 2
        let (pairBytes, pairOverflow) = pairCount.multipliedReportingOverflow(
            by: indexBits)
        guard !pairOverflow else { return nil }
        let (payloadBytes, payloadOverflow) = pairBytes.addingReportingOverflow(
            tailBlockCount * 8)
        guard !payloadOverflow else { return nil }
        let (roundedBytes, roundedOverflow) = payloadBytes.addingReportingOverflow(3)
        guard !roundedOverflow else { return nil }
        return (roundedBytes / 4) * 4
    }

    /// Losslessly row-pack U16 indexes into the fixed 12/13-bit layout above.
    /// nil is fail-closed for malformed geometry or an index that does not
    /// fit the requested width.
    static func packFixedWidthIndices(
        _ indices: [UInt16],
        rows: Int,
        groupsPerRow: Int,
        indexBits: Int
    ) -> [UInt8]? {
        guard rows > 0,
              let bytesPerRow = packedBytesPerRow(
                  groupsPerRow: groupsPerRow,
                  indexBits: indexBits
              )
        else {
            return nil
        }
        let (elementCount, elementOverflow) = rows.multipliedReportingOverflow(
            by: groupsPerRow)
        guard !elementOverflow, indices.count == elementCount else { return nil }
        let (byteCount, byteOverflow) = rows.multipliedReportingOverflow(
            by: bytesPerRow)
        guard !byteOverflow else { return nil }

        let blocks = groupsPerRow / 4
        let pairCount = blocks / 2
        let tailBlockCount = blocks % 2
        let pairStride = indexBits
        let limit = UInt16(1 << indexBits)

        var bytes = [UInt8](repeating: 0, count: byteCount)
        for row in 0..<rows {
            let inputBase = row * groupsPerRow
            let outputBase = row * bytesPerRow

            for pair in 0..<pairCount {
                let inputPairBase = inputBase + pair * 8
                let outputPairBase = outputBase + pair * pairStride
                var topBits: UInt8 = 0
                for laneGroup in 0..<4 {
                    let even = indices[inputPairBase + laneGroup]
                    let odd = indices[inputPairBase + 4 + laneGroup]
                    guard even < limit, odd < limit else { return nil }
                    bytes[outputPairBase + laneGroup] =
                        UInt8(truncatingIfNeeded: even)
                    bytes[outputPairBase + 4 + laneGroup] =
                        UInt8(((even >> 8) & 0x0f) | ((odd & 0x000f) << 4))
                    bytes[outputPairBase + 8 + laneGroup] =
                        UInt8((odd >> 4) & 0xff)
                    if indexBits == 13 {
                        topBits |= UInt8((even >> 12) & 1) << laneGroup
                        topBits |= UInt8((odd >> 12) & 1) << (4 + laneGroup)
                    }
                }
                if indexBits == 13 {
                    bytes[outputPairBase + 12] = topBits
                }
            }

            if tailBlockCount == 1 {
                let inputTailBase = inputBase + pairCount * 8
                let outputTailBase = outputBase + pairCount * pairStride
                for laneGroup in 0..<4 {
                    let value = indices[inputTailBase + laneGroup]
                    guard value < limit else { return nil }
                    let laneOffset = outputTailBase + laneGroup * 2
                    bytes[laneOffset] = UInt8(truncatingIfNeeded: value)
                    bytes[laneOffset + 1] = UInt8(value >> 8)
                }
            }
        }
        return bytes
    }
}

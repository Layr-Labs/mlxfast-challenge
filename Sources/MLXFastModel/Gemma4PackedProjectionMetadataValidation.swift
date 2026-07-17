import Foundation
import MLXFastCore

/// Layout constants for the transform-authored fixed 12/13-bit projection
/// index metadata (`<stem>.metadata_packed_fixed12` / `..._fixed13`).
///
/// The byte layout mirrors `PackedProjectionMetadataCoding` in
/// `Sources/MLXFastTransform`: each output row packs `groupsPerRow / 4`
/// four-index decode blocks as block pairs of `indexBits` bytes (three
/// four-byte lane planes, plus one top-bit byte for fixed13) followed by an
/// optional eight-byte lane-major tail block, padded to a four-byte row
/// stride. `gemma4UnpackPackedProjectionIndex` reproduces the exact byte
/// reads the packed decode kernels in `Gemma4FusedQKV.swift` perform, so the
/// exhaustive load-time compare below proves kernel-visible index equality.
enum Gemma4PackedProjectionMetadataLayout {
    static let fixed12Suffix = ".metadata_packed_fixed12"
    static let fixed13Suffix = ".metadata_packed_fixed13"
    static let indicesSuffix = ".metadata_indices"
    static let lutSuffix = ".metadata_lut"

    static func maximumLUTCount(indexBits: Int) -> Int? {
        switch indexBits {
        case 12:
            return 4_096
        case 13:
            return 8_192
        default:
            return nil
        }
    }

    static func bytesPerRow(groupsPerRow: Int, indexBits: Int) -> Int? {
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
}

/// Decode one index from a packed row exactly as the packed QKV/QK kernels
/// do: full-byte loads from the three lane planes (and the fixed13 top-bit
/// byte), with no masking beyond the kernels' own nibble extractions. Any
/// stray bit an author left in a consumed byte therefore fails the
/// exhaustive equality check instead of being silently ignored.
func gemma4UnpackPackedProjectionIndex(
    rowBytes: UnsafeRawBufferPointer,
    rowBase: Int,
    group: Int,
    groupsPerRow: Int,
    indexBits: Int
) -> UInt16? {
    guard group >= 0,
          group < groupsPerRow,
          groupsPerRow.isMultiple(of: 4),
          indexBits == 12 || indexBits == 13
    else {
        return nil
    }
    let blocks = groupsPerRow / 4
    let pairCount = blocks / 2
    let block = group / 4
    let laneGroup = group % 4
    let pairStride = indexBits

    if block == pairCount * 2 {
        // Unpaired tail block: two lane-major bytes per index.
        let base = rowBase + pairCount * pairStride + laneGroup * 2
        let low = UInt16(rowBytes[base])
        let high = UInt16(rowBytes[base + 1])
        return low | (high << 8)
    }

    let pair = block / 2
    let base = rowBase + pair * pairStride
    let middle = UInt16(rowBytes[base + 4 + laneGroup])
    var value: UInt16
    if block.isMultiple(of: 2) {
        let low = UInt16(rowBytes[base + laneGroup])
        value = low | ((middle & 0x0f) << 8)
        if indexBits == 13 {
            let top = UInt16(rowBytes[base + 12])
            value |= ((top >> laneGroup) & 1) << 12
        }
    } else {
        let high = UInt16(rowBytes[base + 8 + laneGroup])
        value = (middle >> 4) | (high << 4)
        if indexBits == 13 {
            let top = UInt16(rowBytes[base + 12])
            value |= ((top >> (4 + laneGroup)) & 1) << 12
        }
    }
    return value
}

/// Array-based convenience wrapper for tests.
func gemma4UnpackPackedProjectionIndices(
    _ packed: [UInt8],
    rows: Int,
    groupsPerRow: Int,
    indexBits: Int
) -> [UInt16]? {
    guard rows > 0,
          let bytesPerRow = Gemma4PackedProjectionMetadataLayout.bytesPerRow(
              groupsPerRow: groupsPerRow,
              indexBits: indexBits
          ),
          packed.count == rows * bytesPerRow
    else {
        return nil
    }
    var values = [UInt16]()
    values.reserveCapacity(rows * groupsPerRow)
    let complete: Bool = packed.withUnsafeBytes { bytes in
        for row in 0..<rows {
            for group in 0..<groupsPerRow {
                guard let value = gemma4UnpackPackedProjectionIndex(
                    rowBytes: bytes,
                    rowBase: row * bytesPerRow,
                    group: group,
                    groupsPerRow: groupsPerRow,
                    indexBits: indexBits
                ) else {
                    return false
                }
                values.append(value)
            }
        }
        return true
    }
    return complete ? values : nil
}

/// Proof that one transform-authored packed projection index stream
/// exhaustively reconstructed the stem's U16 `.metadata_indices` tensor from
/// raw safetensor bytes before any MLX array was loaded. Partitioning
/// requires these descriptors and binds them to the loaded arrays.
struct ValidatedGemma4PackedProjectionMetadata: Sendable {
    let indexBits: Int
    let rows: Int
    let groupsPerRow: Int
    let bytesPerRow: Int
    let lutCount: Int
}

/// Exhaustively verifies every transform-authored packed projection index
/// tensor against the stem's raw U16 index stream. This file intentionally
/// does not import MLX: the compare must not start a GPU command buffer or
/// extend the loaded telemetry window around model initialization.
///
/// The result is keyed by the runtime stem (the `language_model.` prefix the
/// loader strips). An empty dictionary means the checkpoint carries no packed
/// projection metadata (old transformed weights) and the runtime keeps its
/// U16 kernels everywhere.
func validateGemma4PackedProjectionMetadataBytes(
    denseStore: DenseTensorStore
) throws -> [String: ValidatedGemma4PackedProjectionMetadata] {
    let runtimePrefix = "language_model."
    var validated: [String: ValidatedGemma4PackedProjectionMetadata] = [:]
    var seenSourceStems: Set<String> = []

    for name in denseStore.tensorNames {
        let indexBits: Int
        let suffix: String
        if name.hasSuffix(Gemma4PackedProjectionMetadataLayout.fixed12Suffix) {
            indexBits = 12
            suffix = Gemma4PackedProjectionMetadataLayout.fixed12Suffix
        } else if name.hasSuffix(Gemma4PackedProjectionMetadataLayout.fixed13Suffix) {
            indexBits = 13
            suffix = Gemma4PackedProjectionMetadataLayout.fixed13Suffix
        } else {
            continue
        }
        let stem = String(name.dropLast(suffix.count))
        guard seenSourceStems.insert(stem).inserted else {
            throw MLXFastError.invalidInput(
                "duplicate packed projection metadata formats for \(stem)"
            )
        }

        guard let packedRecord = denseStore.record(named: name),
              let indicesRecord = denseStore.record(
                  named: stem + Gemma4PackedProjectionMetadataLayout.indicesSuffix
              ),
              let lutRecord = denseStore.record(
                  named: stem + Gemma4PackedProjectionMetadataLayout.lutSuffix
              )
        else {
            throw MLXFastError.invalidInput(
                "packed projection metadata is missing indexed companions for \(stem)"
            )
        }
        guard let maximumLUTCount =
                Gemma4PackedProjectionMetadataLayout.maximumLUTCount(
                    indexBits: indexBits
                ),
              indicesRecord.dtype == TensorDType.u16.rawValue,
              indicesRecord.shape.count == 2,
              indicesRecord.shape[0] > 0,
              indicesRecord.shape[1] > 0,
              lutRecord.dtype == TensorDType.u32.rawValue,
              lutRecord.shape.count == 1,
              (1...maximumLUTCount).contains(lutRecord.shape[0]),
              packedRecord.dtype == TensorDType.u8.rawValue,
              let bytesPerRow = Gemma4PackedProjectionMetadataLayout.bytesPerRow(
                  groupsPerRow: indicesRecord.shape[1],
                  indexBits: indexBits
              ),
              packedRecord.shape == [indicesRecord.shape[0], bytesPerRow]
        else {
            throw MLXFastError.invalidInput(
                "packed projection metadata has invalid dtype or shape for \(stem)"
            )
        }

        let rows = indicesRecord.shape[0]
        let groupsPerRow = indicesRecord.shape[1]
        let lutCount = lutRecord.shape[0]
        try validateGemma4PackedProjectionIndexBytes(
            packed: denseStore.tensorBytes(named: name),
            indices: denseStore.tensorBytes(
                named: stem + Gemma4PackedProjectionMetadataLayout.indicesSuffix
            ),
            rows: rows,
            groupsPerRow: groupsPerRow,
            bytesPerRow: bytesPerRow,
            indexBits: indexBits,
            lutCount: lutCount,
            stem: stem
        )

        let runtimeStem = stem.hasPrefix(runtimePrefix)
            ? String(stem.dropFirst(runtimePrefix.count))
            : stem
        guard validated.updateValue(
            ValidatedGemma4PackedProjectionMetadata(
                indexBits: indexBits,
                rows: rows,
                groupsPerRow: groupsPerRow,
                bytesPerRow: bytesPerRow,
                lutCount: lutCount
            ),
            forKey: runtimeStem
        ) == nil else {
            throw MLXFastError.invalidInput(
                "packed projection metadata stems collide after runtime rename: "
                    + runtimeStem
            )
        }
    }
    return validated
}

func validateGemma4PackedProjectionIndexBytes(
    packed: Data,
    indices: Data,
    rows: Int,
    groupsPerRow: Int,
    bytesPerRow: Int,
    indexBits: Int,
    lutCount: Int,
    stem: String
) throws {
    let (elementCount, elementOverflow) = rows.multipliedReportingOverflow(
        by: groupsPerRow)
    let (indexByteCount, indexOverflow) = elementCount.multipliedReportingOverflow(
        by: 2)
    let (packedByteCount, packedOverflow) = rows.multipliedReportingOverflow(
        by: bytesPerRow)
    guard !elementOverflow, !indexOverflow, !packedOverflow,
          indices.count == indexByteCount,
          packed.count == packedByteCount,
          lutCount > 0
    else {
        throw MLXFastError.invalidInput(
            "packed projection metadata byte lengths are invalid for \(stem)"
        )
    }

    try packed.withUnsafeBytes { (packedBytes: UnsafeRawBufferPointer) in
        try indices.withUnsafeBytes { (indexBytes: UnsafeRawBufferPointer) in
            for row in 0..<rows {
                let rowBase = row * bytesPerRow
                let indexRowBase = row * groupsPerRow * 2
                for group in 0..<groupsPerRow {
                    guard let value = gemma4UnpackPackedProjectionIndex(
                        rowBytes: packedBytes,
                        rowBase: rowBase,
                        group: group,
                        groupsPerRow: groupsPerRow,
                        indexBits: indexBits
                    ) else {
                        throw MLXFastError.invalidInput(
                            "packed projection metadata layout is invalid for \(stem)"
                        )
                    }
                    let expected = indexBytes.loadUnaligned(
                        fromByteOffset: indexRowBase + group * 2,
                        as: UInt16.self
                    ).littleEndian
                    guard value == expected, Int(value) < lutCount else {
                        throw MLXFastError.invalidInput(
                            "packed projection metadata does not reconstruct the "
                                + "U16 index stream for \(stem) at row \(row), "
                                + "group \(group)"
                        )
                    }
                }
            }
        }
    }
}

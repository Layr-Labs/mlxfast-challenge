import Foundation
import MLXFastCore

/// Layout constants for the transform-authored co-tiled tied vocabulary head
/// decode payload (`language_model.model.embed_tokens.tied_head_cotiled_payload`).
///
/// The byte layout mirrors `CoTiledTiedHeadPayloadCoding` in
/// `Sources/MLXFastTransform` and is documented there. Summary — the co-tiled
/// tied-head kernel dispatches 16_384 threadgroups of four 32-lane simdgroup
/// slots, each slot owning four consecutive output rows (16 rows per
/// threadgroup). Per threadgroup:
///
/// - Ten pair tiles of 1_076 U32 words each: even-block weights (slot-major,
///   row-major, 128 bytes per row), odd-block weights, then the packed13
///   metadata pair runs (slot-major, 13 bytes per row — block pair `p`
///   consumes columns `8p..<8p+8`, bits `[104p, 104p+104)`, which are exactly
///   bytes `[13p, 13p+13)` of the 140-byte packed13 row).
/// - One tail tile of 544 words: block-20 weights, then 8 tail metadata bytes
///   per row (the packed13 row's bytes `[130, 138)`, carrying columns
///   `80..<84`; the row's trailing padding bytes are never copied).
///
/// `validateGemma4TiedHeadCoTiledPayloadBytes` reproduces the exact byte runs
/// the co-tiled kernel in `Gemma4TiedVocabularyHead.swift` consumes, so the
/// exhaustive load-time compare below proves kernel-visible equality with the
/// separate weight and packed13 index tensors. This file intentionally does
/// not import MLX: the compare must not start a GPU command buffer around
/// model initialization.
enum Gemma4TiedHeadCoTiledPayloadLayout {
    static let payloadName =
        "language_model.model.embed_tokens.tied_head_cotiled_payload"
    static let weightName = "language_model.model.embed_tokens.weight"
    static let packedIndicesName =
        "language_model.model.embed_tokens.tied_head_packed13_indices"

    static let rowCount = 262_144
    static let weightWordsPerRow = 672
    static let weightBytesPerRow = 2_688
    static let blockBytes = 128
    static let pairCount = 10
    static let rowsPerSlot = 4
    static let slotsPerThreadgroup = 4
    static let rowsPerThreadgroup = 16
    static let packedWordsPerRow = 35
    static let packedBytesPerRow = 140
    static let pairMetadataBytesPerRow = 13
    static let tailMetadataBytesPerRow = 8
    static let tailMetadataRowByteOffset = 130

    static let pairTileBytes = 2 * rowsPerThreadgroup * blockBytes
        + rowsPerThreadgroup * pairMetadataBytesPerRow
    static let tailTileBytes = rowsPerThreadgroup * blockBytes
        + rowsPerThreadgroup * tailMetadataBytesPerRow
    /// Pinned against the transform-side implementation by a CPU test.
    static let wordsPerThreadgroup =
        (pairCount * pairTileBytes + tailTileBytes) / 4
    static let threadgroups = rowCount / rowsPerThreadgroup
}

/// Proof that the transform-authored co-tiled tied-head payload exhaustively
/// reconstructed, byte for byte, every kernel-consumed weight and packed13
/// metadata run from the separate tensors before any MLX array was loaded.
/// Partitioning requires this descriptor and binds it to the loaded array.
struct ValidatedGemma4TiedHeadCoTiledPayload: Sendable {
    let threadgroups: Int
    let wordsPerThreadgroup: Int
}

/// Exhaustively verifies one co-tiled tied-head payload against the raw
/// weight and packed13 index bytes by walking the exact per-threadgroup tile
/// stream the co-tiled kernel traverses: for every (threadgroup, slot, row)
/// the ten 128-byte even/odd weight runs and 13-byte metadata runs, the
/// 128-byte tail weight run, and the 8-byte tail metadata run must equal the
/// corresponding source-tensor runs. Any mismatch, and any geometry the
/// kernel cannot address, throws.
func validateGemma4TiedHeadCoTiledPayloadBytes(
    payload: Data,
    weight: Data,
    packedIndices: Data,
    rows: Int,
    name: String
) throws {
    typealias Layout = Gemma4TiedHeadCoTiledPayloadLayout
    guard rows > 0, rows.isMultiple(of: Layout.rowsPerThreadgroup) else {
        throw MLXFastError.invalidInput(
            "co-tiled tied-head payload has invalid row geometry for \(name)"
        )
    }
    let (weightBytes, weightOverflow) =
        rows.multipliedReportingOverflow(by: Layout.weightBytesPerRow)
    let (packedBytes, packedOverflow) =
        rows.multipliedReportingOverflow(by: Layout.packedBytesPerRow)
    guard !weightOverflow, !packedOverflow,
          weight.count == weightBytes,
          packedIndices.count == packedBytes
    else {
        throw MLXFastError.invalidInput(
            "co-tiled tied-head payload source byte lengths are invalid for \(name)"
        )
    }
    let threadgroups = rows / Layout.rowsPerThreadgroup
    let threadgroupBytes = Layout.pairCount * Layout.pairTileBytes
        + Layout.tailTileBytes
    let (expectedPayloadBytes, payloadOverflow) =
        threadgroups.multipliedReportingOverflow(by: threadgroupBytes)
    guard !payloadOverflow, payload.count == expectedPayloadBytes else {
        throw MLXFastError.invalidInput(
            "co-tiled tied-head payload byte length is invalid for \(name)"
        )
    }

    func compareRun(
        _ payloadBytes: UnsafeRawBufferPointer,
        _ sourceBytes: UnsafeRawBufferPointer,
        payloadOffset: Int,
        sourceOffset: Int,
        count: Int
    ) -> Bool {
        guard let payloadBase = payloadBytes.baseAddress,
              let sourceBase = sourceBytes.baseAddress
        else {
            return false
        }
        return memcmp(
            payloadBase + payloadOffset,
            sourceBase + sourceOffset,
            count
        ) == 0
    }

    try payload.withUnsafeBytes { (payloadBytes: UnsafeRawBufferPointer) in
        try weight.withUnsafeBytes { (weightBuffer: UnsafeRawBufferPointer) in
            try packedIndices.withUnsafeBytes { (packedBuffer: UnsafeRawBufferPointer) in
                for threadgroup in 0..<threadgroups {
                    let tileBase = threadgroup * threadgroupBytes
                    for slot in 0..<Layout.slotsPerThreadgroup {
                        for row in 0..<Layout.rowsPerSlot {
                            let sourceRow = threadgroup * Layout.rowsPerThreadgroup
                                + slot * Layout.rowsPerSlot + row
                            let weightRowBase = sourceRow * Layout.weightBytesPerRow
                            let packedRowBase = sourceRow * Layout.packedBytesPerRow
                            for pair in 0..<Layout.pairCount {
                                let pairBase = tileBase + pair * Layout.pairTileBytes
                                let evenMatches = compareRun(
                                    payloadBytes,
                                    weightBuffer,
                                    payloadOffset: pairBase + 512 * slot + 128 * row,
                                    sourceOffset: weightRowBase
                                        + (2 * pair) * Layout.blockBytes,
                                    count: Layout.blockBytes
                                )
                                let oddMatches = compareRun(
                                    payloadBytes,
                                    weightBuffer,
                                    payloadOffset: pairBase
                                        + 512 * Layout.slotsPerThreadgroup
                                        + 512 * slot + 128 * row,
                                    sourceOffset: weightRowBase
                                        + (2 * pair + 1) * Layout.blockBytes,
                                    count: Layout.blockBytes
                                )
                                let metadataMatches = compareRun(
                                    payloadBytes,
                                    packedBuffer,
                                    payloadOffset: pairBase
                                        + 1_024 * Layout.slotsPerThreadgroup
                                        + Layout.rowsPerSlot
                                            * Layout.pairMetadataBytesPerRow * slot
                                        + row * Layout.pairMetadataBytesPerRow,
                                    sourceOffset: packedRowBase
                                        + pair * Layout.pairMetadataBytesPerRow,
                                    count: Layout.pairMetadataBytesPerRow
                                )
                                guard evenMatches, oddMatches, metadataMatches else {
                                    throw MLXFastError.invalidInput(
                                        "co-tiled tied-head payload does not "
                                            + "reconstruct sources at threadgroup "
                                            + "\(threadgroup), slot \(slot), pair "
                                            + "\(pair), row \(row) for \(name)"
                                    )
                                }
                            }
                            let tailBase = tileBase
                                + Layout.pairCount * Layout.pairTileBytes
                            let tailWeightMatches = compareRun(
                                payloadBytes,
                                weightBuffer,
                                payloadOffset: tailBase + 512 * slot + 128 * row,
                                sourceOffset: weightRowBase
                                    + 2 * Layout.pairCount * Layout.blockBytes,
                                count: Layout.blockBytes
                            )
                            let tailMetadataMatches = compareRun(
                                payloadBytes,
                                packedBuffer,
                                payloadOffset: tailBase
                                    + 512 * Layout.slotsPerThreadgroup
                                    + 32 * slot + 8 * row,
                                sourceOffset: packedRowBase
                                    + Layout.tailMetadataRowByteOffset,
                                count: Layout.tailMetadataBytesPerRow
                            )
                            guard tailWeightMatches, tailMetadataMatches else {
                                throw MLXFastError.invalidInput(
                                    "co-tiled tied-head payload does not "
                                        + "reconstruct sources at threadgroup "
                                        + "\(threadgroup), slot \(slot), tail row "
                                        + "\(row) for \(name)"
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Exhaustively verifies the transform-authored co-tiled tied-head payload in
/// the dense store against the separate weight and packed13 index tensors.
/// nil means the checkpoint carries no co-tiled tied-head payload (older
/// transformed weights) and the runtime keeps its current packed13 kernel. A
/// payload that is present but incomplete or unfaithful fails closed: the
/// artifact is corrupt and must not load. The packed13 metadata itself must
/// already have been validated against the stock scale/bias tensors
/// (`validateGemma4TiedHeadPacked13MetadataBytes`), so payload-vs-packed13
/// equality proves payload-vs-stock equality transitively.
func validateGemma4TiedHeadCoTiledPayloadBytes(
    denseStore: DenseTensorStore,
    validatedTiedHeadPacked13Metadata: ValidatedGemma4TiedHeadPacked13Metadata?
) throws -> ValidatedGemma4TiedHeadCoTiledPayload? {
    typealias Layout = Gemma4TiedHeadCoTiledPayloadLayout
    guard let payloadRecord = denseStore.record(named: Layout.payloadName) else {
        return nil
    }
    guard validatedTiedHeadPacked13Metadata != nil else {
        throw MLXFastError.invalidInput(
            "co-tiled tied-head payload is present without validated packed13 "
                + "metadata"
        )
    }
    guard let weightRecord = denseStore.record(named: Layout.weightName),
          weightRecord.dtype == TensorDType.u32.rawValue,
          weightRecord.shape == [Layout.rowCount, Layout.weightWordsPerRow],
          let packedRecord = denseStore.record(named: Layout.packedIndicesName),
          packedRecord.dtype == TensorDType.u32.rawValue,
          packedRecord.shape == [Layout.rowCount, Layout.packedWordsPerRow],
          payloadRecord.dtype == TensorDType.u32.rawValue,
          payloadRecord.shape == [Layout.threadgroups, Layout.wordsPerThreadgroup]
    else {
        throw MLXFastError.invalidInput(
            "co-tiled tied-head payload has invalid dtype or shape"
        )
    }

    try validateGemma4TiedHeadCoTiledPayloadBytes(
        payload: denseStore.tensorBytes(named: Layout.payloadName),
        weight: denseStore.tensorBytes(named: Layout.weightName),
        packedIndices: denseStore.tensorBytes(named: Layout.packedIndicesName),
        rows: Layout.rowCount,
        name: Layout.payloadName
    )
    return ValidatedGemma4TiedHeadCoTiledPayload(
        threadgroups: Layout.threadgroups,
        wordsPerThreadgroup: Layout.wordsPerThreadgroup
    )
}

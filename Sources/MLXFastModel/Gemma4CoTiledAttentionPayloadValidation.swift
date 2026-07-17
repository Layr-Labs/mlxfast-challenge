import Foundation
import MLXFastCore

/// Layout constants for the transform-authored co-tiled attention decode
/// payloads (`<layer>.self_attn.qkv_cotiled_payload` for sliding layers,
/// `<layer>.self_attn.qk_cotiled_payload` for full-attention layers).
///
/// The byte layout mirrors `CoTiledAttentionPayloadCoding` in
/// `Sources/MLXFastTransform` and is documented there. Summary, with `S`
/// simdgroup slots per threadgroup (sliding `[q0, q1, k, v]` = 4, full
/// `[q0..q7, k]` = 9), each slot owning four output rows:
///
/// - Ten pair tiles of `256 S + M / 4` U32 words each, where
///   `M = sum over slots of 4 * indexBits` bytes: even-block weights
///   (slot-major, row-major, 128 bytes per row), odd-block weights, then the
///   packed fixed12/13 metadata pair tiles (slot-major, `indexBits` bytes
///   per row).
/// - One tail tile of `136 S` words: block-20 weights, then 8 lane-major
///   tail metadata bytes per row (32 bytes per slot).
///
/// `validateGemma4CoTiledAttentionPayloadBytes` reproduces the exact byte
/// runs the co-tiled kernels in `Gemma4FusedQKV.swift` consume, so the
/// exhaustive load-time compare below proves kernel-visible equality with
/// the stem's separate weight and packed-metadata tensors. This file
/// intentionally does not import MLX: the compare must not start a GPU
/// command buffer around model initialization.
enum Gemma4CoTiledAttentionPayloadLayout {
    static let slidingSuffix = ".self_attn.qkv_cotiled_payload"
    static let fullSuffix = ".self_attn.qk_cotiled_payload"

    static let groupsPerRow = 84
    static let weightBytesPerRow = 2_688
    static let blockBytes = 128
    static let pairCount = 10
    static let rowsPerSlot = 4

    static let slidingSlots = [2, 1, 1]
    static let fullSlots = [8, 1]

    static func slidingSlotIndexBits(qBits: Int, kBits: Int, vBits: Int) -> [Int] {
        [qBits, qBits, kBits, vBits]
    }

    static func fullSlotIndexBits(qBits: Int, kBits: Int) -> [Int] {
        [Int](repeating: qBits, count: 8) + [kBits]
    }

    /// U32 words per threadgroup for one slot-major index-bit list, or nil
    /// for an unsupported width. Pinned against the transform-side
    /// implementation by a CPU test.
    static func wordsPerThreadgroup(slotIndexBits: [Int]) -> Int? {
        guard !slotIndexBits.isEmpty,
              slotIndexBits.allSatisfy({ $0 == 12 || $0 == 13 })
        else {
            return nil
        }
        let slots = slotIndexBits.count
        let pairMetadataBytes = slotIndexBits.reduce(0) { $0 + 4 * $1 }
        return pairCount * (256 * slots + pairMetadataBytes / 4) + 136 * slots
    }
}

/// One projection of a co-tiled payload during raw-byte validation.
struct Gemma4CoTiledPayloadProjectionSource {
    let weight: Data
    let packed: Data
    let rows: Int
    let indexBits: Int
    let slots: Int
    let stem: String
}

/// Which fused decode kernel family a validated payload feeds.
enum Gemma4CoTiledAttentionPayloadKind: Sendable, Equatable {
    case slidingQKV
    case fullQK
}

/// Proof that one transform-authored co-tiled attention payload exhaustively
/// reconstructed, byte for byte, every kernel-consumed weight and packed
/// metadata run from the stem's separate tensors before any MLX array was
/// loaded. Partitioning requires these descriptors and binds them to the
/// loaded arrays.
struct ValidatedGemma4CoTiledAttentionPayload: Sendable {
    let kind: Gemma4CoTiledAttentionPayloadKind
    let threadgroups: Int
    let wordsPerThreadgroup: Int
    let qBits: Int
    let kBits: Int
    let vBits: Int?
}

/// Exhaustively verifies one co-tiled payload against its ordered projection
/// sources by walking the exact per-threadgroup tile stream the co-tiled
/// kernels traverse: for every (threadgroup, pair tile, slot, row) the
/// 128-byte even/odd/tail weight runs and the `indexBits`-byte (pair) or
/// 8-byte (tail) metadata runs must equal the corresponding source-tensor
/// runs. Any mismatch, and any geometry the kernels cannot address, throws.
func validateGemma4CoTiledAttentionPayloadBytes(
    payload: Data,
    projections: [Gemma4CoTiledPayloadProjectionSource],
    name: String
) throws {
    typealias Layout = Gemma4CoTiledAttentionPayloadLayout
    guard !projections.isEmpty else {
        throw MLXFastError.invalidInput(
            "co-tiled attention payload has no projections for \(name)"
        )
    }

    var threadgroups: Int?
    var packedBytesPerRow: [Int] = []
    var slotProjection: [Int] = []
    var slotWithinProjection: [Int] = []
    for (index, projection) in projections.enumerated() {
        guard projection.slots > 0,
              projection.rows > 0,
              projection.rows.isMultiple(of: Layout.rowsPerSlot * projection.slots),
              projection.indexBits == 12 || projection.indexBits == 13,
              let bytesPerRow = Gemma4PackedProjectionMetadataLayout.bytesPerRow(
                  groupsPerRow: Layout.groupsPerRow,
                  indexBits: projection.indexBits
              )
        else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload has invalid projection geometry for "
                    + "\(projection.stem) in \(name)"
            )
        }
        let (weightBytes, weightOverflow) = projection.rows
            .multipliedReportingOverflow(by: Layout.weightBytesPerRow)
        let (packedBytes, packedOverflow) = projection.rows
            .multipliedReportingOverflow(by: bytesPerRow)
        guard !weightOverflow, !packedOverflow,
              projection.weight.count == weightBytes,
              projection.packed.count == packedBytes
        else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload source byte lengths are invalid for "
                    + "\(projection.stem) in \(name)"
            )
        }
        let count = projection.rows / (Layout.rowsPerSlot * projection.slots)
        if let threadgroups, threadgroups != count {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload projections disagree on threadgroup "
                    + "count for \(name)"
            )
        }
        threadgroups = count
        packedBytesPerRow.append(bytesPerRow)
        for slot in 0..<projection.slots {
            slotProjection.append(index)
            slotWithinProjection.append(slot)
        }
    }
    guard let threadgroups else {
        throw MLXFastError.invalidInput(
            "co-tiled attention payload has no threadgroups for \(name)"
        )
    }
    let slots = slotProjection.count
    var metadataOffsets: [Int] = []
    metadataOffsets.reserveCapacity(slots)
    var metadataCursor = 0
    for slot in 0..<slots {
        metadataOffsets.append(metadataCursor)
        metadataCursor += Layout.rowsPerSlot
            * projections[slotProjection[slot]].indexBits
    }
    let pairTileBytes = 1_024 * slots + metadataCursor
    let tailTileBytes = 544 * slots
    let threadgroupBytes = Layout.pairCount * pairTileBytes + tailTileBytes
    let (expectedPayloadBytes, payloadOverflow) = threadgroups
        .multipliedReportingOverflow(by: threadgroupBytes)
    guard !payloadOverflow, payload.count == expectedPayloadBytes else {
        throw MLXFastError.invalidInput(
            "co-tiled attention payload byte length is invalid for \(name)"
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
        try withGemma4CoTiledProjectionBytes(projections) { weightBuffers, packedBuffers in
            for threadgroup in 0..<threadgroups {
                let tileBase = threadgroup * threadgroupBytes
                for slot in 0..<slots {
                    let projectionIndex = slotProjection[slot]
                    let projection = projections[projectionIndex]
                    let bytesPerRow = packedBytesPerRow[projectionIndex]
                    let rowBase = threadgroup
                        * Layout.rowsPerSlot * projection.slots
                        + Layout.rowsPerSlot * slotWithinProjection[slot]
                    for row in 0..<Layout.rowsPerSlot {
                        let sourceRow = rowBase + row
                        let weightRowBase = sourceRow * Layout.weightBytesPerRow
                        let packedRowBase = sourceRow * bytesPerRow
                        for pair in 0..<Layout.pairCount {
                            let pairBase = tileBase + pair * pairTileBytes
                            let evenMatches = compareRun(
                                payloadBytes,
                                weightBuffers[projectionIndex],
                                payloadOffset: pairBase + 512 * slot + 128 * row,
                                sourceOffset: weightRowBase
                                    + (2 * pair) * Layout.blockBytes,
                                count: Layout.blockBytes
                            )
                            let oddMatches = compareRun(
                                payloadBytes,
                                weightBuffers[projectionIndex],
                                payloadOffset: pairBase + 512 * slots
                                    + 512 * slot + 128 * row,
                                sourceOffset: weightRowBase
                                    + (2 * pair + 1) * Layout.blockBytes,
                                count: Layout.blockBytes
                            )
                            let metadataMatches = compareRun(
                                payloadBytes,
                                packedBuffers[projectionIndex],
                                payloadOffset: pairBase + 1_024 * slots
                                    + metadataOffsets[slot]
                                    + row * projection.indexBits,
                                sourceOffset: packedRowBase
                                    + pair * projection.indexBits,
                                count: projection.indexBits
                            )
                            guard evenMatches, oddMatches, metadataMatches else {
                                throw MLXFastError.invalidInput(
                                    "co-tiled attention payload does not reconstruct "
                                        + "\(projection.stem) at threadgroup "
                                        + "\(threadgroup), pair \(pair), row \(row) "
                                        + "for \(name)"
                                )
                            }
                        }
                        let tailBase = tileBase + Layout.pairCount * pairTileBytes
                        let tailWeightMatches = compareRun(
                            payloadBytes,
                            weightBuffers[projectionIndex],
                            payloadOffset: tailBase + 512 * slot + 128 * row,
                            sourceOffset: weightRowBase
                                + 2 * Layout.pairCount * Layout.blockBytes,
                            count: Layout.blockBytes
                        )
                        let tailMetadataMatches = compareRun(
                            payloadBytes,
                            packedBuffers[projectionIndex],
                            payloadOffset: tailBase + 512 * slots
                                + 32 * slot + 8 * row,
                            sourceOffset: packedRowBase
                                + Layout.pairCount * projection.indexBits,
                            count: 8
                        )
                        guard tailWeightMatches, tailMetadataMatches else {
                            throw MLXFastError.invalidInput(
                                "co-tiled attention payload does not reconstruct "
                                    + "\(projection.stem) at threadgroup "
                                    + "\(threadgroup), tail row \(row) for \(name)"
                            )
                        }
                    }
                }
            }
        }
    }
}

private func withGemma4CoTiledProjectionBytes(
    _ projections: [Gemma4CoTiledPayloadProjectionSource],
    _ body: ([UnsafeRawBufferPointer], [UnsafeRawBufferPointer]) throws -> Void
) throws {
    func recurse(
        _ index: Int,
        _ weights: [UnsafeRawBufferPointer],
        _ packed: [UnsafeRawBufferPointer]
    ) throws {
        guard index < projections.count else {
            try body(weights, packed)
            return
        }
        try projections[index].weight.withUnsafeBytes { weightBytes in
            try projections[index].packed.withUnsafeBytes { packedBytes in
                try recurse(
                    index + 1,
                    weights + [weightBytes],
                    packed + [packedBytes]
                )
            }
        }
    }
    try recurse(0, [], [])
}

/// Exhaustively verifies every transform-authored co-tiled attention payload
/// in the dense store against the constituent stems' separate weight and
/// packed-metadata tensors. The result is keyed by the runtime payload
/// tensor name (the `language_model.` prefix the loader strips). An empty
/// dictionary means the checkpoint carries no co-tiled payloads (older
/// transformed weights) and the runtime keeps its current kernels
/// everywhere. A payload that is present but incomplete or unfaithful
/// fails closed: the artifact is corrupt and must not load.
func validateGemma4CoTiledAttentionPayloadBytes(
    denseStore: DenseTensorStore,
    validatedPackedProjectionMetadata: [String: ValidatedGemma4PackedProjectionMetadata]
) throws -> [String: ValidatedGemma4CoTiledAttentionPayload] {
    typealias Layout = Gemma4CoTiledAttentionPayloadLayout
    let runtimePrefix = "language_model."
    var validated: [String: ValidatedGemma4CoTiledAttentionPayload] = [:]

    for name in denseStore.tensorNames {
        let kind: Gemma4CoTiledAttentionPayloadKind
        let suffix: String
        if name.hasSuffix(Layout.slidingSuffix) {
            kind = .slidingQKV
            suffix = Layout.slidingSuffix
        } else if name.hasSuffix(Layout.fullSuffix) {
            kind = .fullQK
            suffix = Layout.fullSuffix
        } else {
            continue
        }
        let layerStem = String(name.dropLast(suffix.count))
        let stems: [String]
        let slots: [Int]
        switch kind {
        case .slidingQKV:
            stems = [
                "\(layerStem).self_attn.q_proj",
                "\(layerStem).self_attn.k_proj",
                "\(layerStem).self_attn.v_proj",
            ]
            slots = Layout.slidingSlots
        case .fullQK:
            stems = [
                "\(layerStem).self_attn.q_proj",
                "\(layerStem).self_attn.k_proj",
            ]
            slots = Layout.fullSlots
        }

        guard let payloadRecord = denseStore.record(named: name),
              payloadRecord.dtype == TensorDType.u32.rawValue,
              payloadRecord.shape.count == 2,
              payloadRecord.shape[0] > 0,
              payloadRecord.shape[1] > 0
        else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload has invalid dtype or shape for \(name)"
            )
        }
        let threadgroups = payloadRecord.shape[0]

        var projections: [Gemma4CoTiledPayloadProjectionSource] = []
        var indexBits: [Int] = []
        projections.reserveCapacity(stems.count)
        for (position, stem) in stems.enumerated() {
            let runtimeStem = stem.hasPrefix(runtimePrefix)
                ? String(stem.dropFirst(runtimePrefix.count))
                : stem
            let expectedRows = Layout.rowsPerSlot * slots[position] * threadgroups
            guard let weightRecord = denseStore.record(named: "\(stem).weight"),
                  weightRecord.dtype == TensorDType.u32.rawValue,
                  weightRecord.shape == [expectedRows, 672],
                  let validatedPacked = validatedPackedProjectionMetadata[runtimeStem],
                  validatedPacked.rows == expectedRows,
                  validatedPacked.groupsPerRow == Layout.groupsPerRow,
                  let packedSuffix = validatedPacked.indexBits == 12
                      ? Gemma4PackedProjectionMetadataLayout.fixed12Suffix
                      : (validatedPacked.indexBits == 13
                          ? Gemma4PackedProjectionMetadataLayout.fixed13Suffix
                          : nil)
            else {
                throw MLXFastError.invalidInput(
                    "co-tiled attention payload is missing validated companions "
                        + "for \(stem)"
                )
            }
            projections.append(
                Gemma4CoTiledPayloadProjectionSource(
                    weight: try denseStore.tensorBytes(named: "\(stem).weight"),
                    packed: try denseStore.tensorBytes(named: stem + packedSuffix),
                    rows: expectedRows,
                    indexBits: validatedPacked.indexBits,
                    slots: slots[position],
                    stem: stem
                )
            )
            indexBits.append(validatedPacked.indexBits)
        }

        let slotIndexBits: [Int]
        switch kind {
        case .slidingQKV:
            slotIndexBits = Layout.slidingSlotIndexBits(
                qBits: indexBits[0],
                kBits: indexBits[1],
                vBits: indexBits[2]
            )
        case .fullQK:
            slotIndexBits = Layout.fullSlotIndexBits(
                qBits: indexBits[0],
                kBits: indexBits[1]
            )
        }
        guard let wordsPerThreadgroup = Layout.wordsPerThreadgroup(
            slotIndexBits: slotIndexBits
        ), payloadRecord.shape == [threadgroups, wordsPerThreadgroup] else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload shape does not match its metadata "
                    + "formats for \(name)"
            )
        }

        try validateGemma4CoTiledAttentionPayloadBytes(
            payload: denseStore.tensorBytes(named: name),
            projections: projections,
            name: name
        )

        let runtimeName = name.hasPrefix(runtimePrefix)
            ? String(name.dropFirst(runtimePrefix.count))
            : name
        guard validated.updateValue(
            ValidatedGemma4CoTiledAttentionPayload(
                kind: kind,
                threadgroups: threadgroups,
                wordsPerThreadgroup: wordsPerThreadgroup,
                qBits: indexBits[0],
                kBits: indexBits[1],
                vBits: kind == .slidingQKV ? indexBits[2] : nil
            ),
            forKey: runtimeName
        ) == nil else {
            throw MLXFastError.invalidInput(
                "co-tiled attention payload names collide after runtime rename: "
                    + runtimeName
            )
        }
    }
    return validated
}

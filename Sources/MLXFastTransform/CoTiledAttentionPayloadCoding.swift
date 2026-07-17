import Foundation
import MLXFastCore

/// Offline authoring of the co-tiled attention decode payloads consumed by
/// `gemma4_cotiled_indexed_sliding_qkv_qmv_5376_*` and
/// `gemma4_cotiled_indexed_full_qk_qmv_5376_*` in
/// `Sources/MLXFastModel/Gemma4FusedQKV.swift`.
///
/// Motivation: packing the QKV metadata indexes into separate fixed12/13
/// streams removed bytes but produced no measured decode gain — separate
/// metadata reads overlap the weight stream. The promoted gate/up, down, and
/// output co-tiles show that interleaving the metadata bytes WITH the weight
/// words in exact consumption order is what pays. This coding brings the same
/// single-stream locality to the fused attention decode QMVs: one contiguous
/// per-threadgroup tile stream carrying every weight word and every packed
/// metadata byte in the exact order the kernel consumes them.
///
/// ## Payload layout
///
/// The decode kernels dispatch one 32-lane simdgroup per projection slot:
///
/// - sliding QKV: 4 slots per threadgroup — `[q0, q1, k, v]`; threadgroup `t`
///   owns q rows `8t..<8t+8` (q0 the first four, q1 the second four) and
///   k/v rows `4t..<4t+4`.
/// - full QK: 9 slots per threadgroup — `[q0..q7, k]`; threadgroup `t` owns
///   q rows `32t..<32t+32` and k rows `4t..<4t+4`.
///
/// Each slot walks the 84 metadata groups of its four rows as 21 four-group
/// decode blocks over K = 5376 (10 block pairs plus one unpaired tail block).
/// Per (slot, row, block) the 32 lanes consume one contiguous 128-byte run of
/// the source weight row (`row * 2688 + block * 128`), and per (slot, row,
/// block pair) the lanes consume one `indexBits`-byte fixed12/13 pair tile of
/// the transform-authored packed index stream
/// (`row * packedBytesPerRow + pair * indexBits`, layout in
/// `PackedProjectionMetadataCoding`). The payload rearranges exactly those
/// bytes, per threadgroup, with `S` = total slot count:
///
/// One pair tile `p in 0..<10` (`wordsPerPair` U32 words):
///
/// - `[0, 512 S)` bytes: even-block (block `2p`) weights, slot-major; slot
///   `g` at byte `512 g`; within a slot row-major, row `r` at byte `128 r`;
///   the 128-byte run is the source weight row's block `2p` run, verbatim.
/// - `[512 S, 1024 S)` bytes: odd-block (block `2p + 1`) weights, same order.
/// - `[1024 S, 1024 S + M)` bytes: packed metadata, slot-major; slot `g` at
///   `metadataOffset(g)` = sum of `4 * indexBits` over prior slots; within a
///   slot row-major, row `r` at byte `r * indexBits`; the `indexBits`-byte
///   run is the source packed row's pair tile `p`, verbatim (three coalesced
///   byte planes, plus the fixed13 top-bit byte).
///
/// `M = sum over slots of 4 * indexBits` bytes (always a word multiple:
/// 4 rows x 12 or 13 bytes = 48 or 52). `wordsPerPair = 256 S + M / 4`.
///
/// The tail tile after the ten pairs (`136 S` words):
///
/// - `[0, 512 S)` bytes: block-20 weights, identical slot/row order.
/// - `[512 S, 512 S + 32 S)` bytes: tail metadata, slot-major, 32 bytes per
///   slot; row `r` at byte `8 r`: the eight lane-major tail bytes of the
///   packed row (`row * packedBytesPerRow + 10 * indexBits`, two bytes per
///   lane group). Row padding bytes of the packed stream are NOT copied; the
///   payload carries only kernel-consumed bytes.
///
/// `wordsPerThreadgroup = 10 * wordsPerPair + 136 S`. The payload tensor is
/// dtype U32, shape `[threadgroups, wordsPerThreadgroup]`, written
/// little-endian byte-for-byte from the source runs so no dtype-view
/// ambiguity exists between authoring and the Metal kernel's `device uint*`.
///
/// Production geometry: sliding — 1024 threadgroups, S = 4,
/// `wordsPerPair = 1024 + (8 qBits + 4 kBits + 4 vBits) / 4`, all-fixed12
/// `wordsPerThreadgroup = 11,264` (44 MiB per layer); full — 512
/// threadgroups, S = 9, all-fixed12 `wordsPerThreadgroup = 25,344`
/// (~49.5 MiB per layer).
enum CoTiledAttentionPayloadCoding {
    static let shardName = "mlxfast-attention-cotiled-payload.safetensors"
    static let slidingSuffix = ".self_attn.qkv_cotiled_payload"
    static let fullSuffix = ".self_attn.qk_cotiled_payload"

    static let groupsPerRow = 84
    static let weightWordsPerRow = 672
    static let weightBytesPerRow = 2_688
    static let blockBytes = 128
    static let pairCount = 10
    static let rowsPerSlot = 4

    /// One projection of a co-tiled attention payload: raw little-endian
    /// weight words, the projection's transform-authored fixed12/13 packed
    /// index stream, and how many simdgroup slots the kernel gives it.
    struct ProjectionSource {
        let weight: Data
        let packed: Data
        let rows: Int
        let indexBits: Int
        let slots: Int
    }

    /// U32 words per threadgroup for one slot-major index-bit list, or nil
    /// for an unsupported width. Mirrored by
    /// `Gemma4CoTiledAttentionPayloadLayout.wordsPerThreadgroup` in
    /// `Sources/MLXFastModel`; a CPU test pins the two implementations
    /// together.
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

    /// Slot-major index-bit list for an ordered projection set.
    static func slotIndexBits(projections: [ProjectionSource]) -> [Int] {
        projections.flatMap { projection in
            [Int](repeating: projection.indexBits, count: projection.slots)
        }
    }

    /// The shared threadgroup count of an ordered projection set, or nil for
    /// inconsistent geometry.
    static func threadgroupCount(projections: [ProjectionSource]) -> Int? {
        var threadgroups: Int?
        for projection in projections {
            guard projection.slots > 0,
                  projection.rows > 0,
                  projection.rows.isMultiple(of: rowsPerSlot * projection.slots)
            else {
                return nil
            }
            let count = projection.rows / (rowsPerSlot * projection.slots)
            if let threadgroups, threadgroups != count {
                return nil
            }
            threadgroups = count
        }
        return threadgroups
    }

    /// Build one co-tiled payload from an ordered projection set. nil is
    /// fail-closed for malformed geometry or byte counts.
    static func makePayload(projections: [ProjectionSource]) -> Data? {
        guard !projections.isEmpty,
              let threadgroups = threadgroupCount(projections: projections),
              let wordsPerThreadgroup = wordsPerThreadgroup(
                  slotIndexBits: slotIndexBits(projections: projections)
              )
        else {
            return nil
        }
        var packedBytesPerRow: [Int] = []
        packedBytesPerRow.reserveCapacity(projections.count)
        for projection in projections {
            guard let bytesPerRow = PackedProjectionMetadataCoding.packedBytesPerRow(
                groupsPerRow: groupsPerRow,
                indexBits: projection.indexBits
            ) else {
                return nil
            }
            let (weightBytes, weightOverflow) =
                projection.rows.multipliedReportingOverflow(by: weightBytesPerRow)
            let (packedBytes, packedOverflow) =
                projection.rows.multipliedReportingOverflow(by: bytesPerRow)
            guard !weightOverflow, !packedOverflow,
                  projection.weight.count == weightBytes,
                  projection.packed.count == packedBytes
            else {
                return nil
            }
            packedBytesPerRow.append(bytesPerRow)
        }

        // Global slot list in kernel simdgroup order.
        var slotProjection: [Int] = []
        var slotWithinProjection: [Int] = []
        for (index, projection) in projections.enumerated() {
            for slot in 0..<projection.slots {
                slotProjection.append(index)
                slotWithinProjection.append(slot)
            }
        }
        let slots = slotProjection.count
        var metadataOffsets: [Int] = []
        metadataOffsets.reserveCapacity(slots)
        var metadataCursor = 0
        for slot in 0..<slots {
            metadataOffsets.append(metadataCursor)
            metadataCursor += rowsPerSlot * projections[slotProjection[slot]].indexBits
        }
        let pairMetadataBytes = metadataCursor
        let pairTileBytes = 1_024 * slots + pairMetadataBytes
        let tailTileBytes = 512 * slots + 32 * slots
        let threadgroupBytes = pairCount * pairTileBytes + tailTileBytes
        guard threadgroupBytes == wordsPerThreadgroup * 4 else {
            return nil
        }
        let (payloadBytes, payloadOverflow) =
            threadgroups.multipliedReportingOverflow(by: threadgroupBytes)
        guard !payloadOverflow else {
            return nil
        }

        var payload = [UInt8](repeating: 0, count: payloadBytes)
        payload.withUnsafeMutableBytes { output in
            guard let outputBase = output.baseAddress else { return }
            withProjectionBytes(projections) { weightPointers, packedPointers in
                for threadgroup in 0..<threadgroups {
                    let tileBase = threadgroup * threadgroupBytes
                    for pair in 0..<pairCount {
                        let pairBase = tileBase + pair * pairTileBytes
                        let oddBase = pairBase + 512 * slots
                        let metadataBase = pairBase + 1_024 * slots
                        for slot in 0..<slots {
                            let projectionIndex = slotProjection[slot]
                            let projection = projections[projectionIndex]
                            guard let weightBase =
                                weightPointers[projectionIndex].baseAddress,
                                let packedBase =
                                packedPointers[projectionIndex].baseAddress
                            else {
                                continue
                            }
                            let rowBase = threadgroup
                                * rowsPerSlot * projection.slots
                                + rowsPerSlot * slotWithinProjection[slot]
                            for row in 0..<rowsPerSlot {
                                let sourceRow = rowBase + row
                                let weightRowBase = sourceRow * weightBytesPerRow
                                (outputBase + pairBase + 512 * slot + 128 * row)
                                    .copyMemory(
                                        from: weightBase + weightRowBase
                                            + (2 * pair) * blockBytes,
                                        byteCount: blockBytes
                                    )
                                (outputBase + oddBase + 512 * slot + 128 * row)
                                    .copyMemory(
                                        from: weightBase + weightRowBase
                                            + (2 * pair + 1) * blockBytes,
                                        byteCount: blockBytes
                                    )
                                (outputBase + metadataBase + metadataOffsets[slot]
                                    + row * projection.indexBits)
                                    .copyMemory(
                                        from: packedBase
                                            + sourceRow
                                            * packedBytesPerRow[projectionIndex]
                                            + pair * projection.indexBits,
                                        byteCount: projection.indexBits
                                    )
                            }
                        }
                    }
                    let tailBase = tileBase + pairCount * pairTileBytes
                    let tailMetadataBase = tailBase + 512 * slots
                    for slot in 0..<slots {
                        let projectionIndex = slotProjection[slot]
                        let projection = projections[projectionIndex]
                        guard let weightBase =
                            weightPointers[projectionIndex].baseAddress,
                            let packedBase =
                            packedPointers[projectionIndex].baseAddress
                        else {
                            continue
                        }
                        let rowBase = threadgroup
                            * rowsPerSlot * projection.slots
                            + rowsPerSlot * slotWithinProjection[slot]
                        for row in 0..<rowsPerSlot {
                            let sourceRow = rowBase + row
                            (outputBase + tailBase + 512 * slot + 128 * row)
                                .copyMemory(
                                    from: weightBase
                                        + sourceRow * weightBytesPerRow
                                        + 2 * pairCount * blockBytes,
                                    byteCount: blockBytes
                                )
                            (outputBase + tailMetadataBase + 32 * slot + 8 * row)
                                .copyMemory(
                                    from: packedBase
                                        + sourceRow
                                        * packedBytesPerRow[projectionIndex]
                                        + pairCount * projection.indexBits,
                                    byteCount: 8
                                )
                        }
                    }
                }
            }
        }
        return Data(payload)
    }

    /// Recursively pins every projection's weight and packed buffers and
    /// exposes them as raw pointers for the copy loops.
    private static func withProjectionBytes(
        _ projections: [ProjectionSource],
        _ body: ([UnsafeRawBufferPointer], [UnsafeRawBufferPointer]) -> Void
    ) {
        func recurse(
            _ index: Int,
            _ weights: [UnsafeRawBufferPointer],
            _ packed: [UnsafeRawBufferPointer]
        ) {
            guard index < projections.count else {
                body(weights, packed)
                return
            }
            projections[index].weight.withUnsafeBytes { weightBytes in
                projections[index].packed.withUnsafeBytes { packedBytes in
                    recurse(
                        index + 1,
                        weights + [weightBytes],
                        packed + [packedBytes]
                    )
                }
            }
        }
        recurse(0, [], [])
    }

    private struct EligibleLayer {
        let layerStem: String
        let tensorName: String
        let stems: [String]
        let packedNames: [String]
        let indexBits: [Int]
        let slots: [Int]
        let rows: [Int]
        let threadgroups: Int
        let wordsPerThreadgroup: Int
    }

    /// Authors one co-tiled payload tensor per eligible attention layer into
    /// a dedicated generated shard. Eligibility is shape-driven and
    /// fail-open: a layer whose projections lack transform-authored packed
    /// index streams, or whose weight geometry does not match the fused
    /// decode kernels' slot layout, is skipped and the runtime keeps its
    /// current kernels. The payload derives purely from tensors already
    /// staged in this transform run (weights plus the packed metadata
    /// sidecar), so it is input-independent by construction.
    static func writeSidecar(
        stagingDirectory: URL,
        index: CheckpointIndex,
        stagedHeaders: [String: SafetensorsHeader],
        selectedKeys: Set<String>,
        destinationDirectory: URL
    ) throws -> GeneratedAffineMetadataReport {
        let metadataSidecarURL = stagingDirectory.appendingPathComponent(
            AffineMetadataCoding.shardName
        )
        guard FileManager.default.fileExists(atPath: metadataSidecarURL.path) else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        let metadataHeader = try Safetensors.readHeader(metadataSidecarURL)

        let layerPrefix = "language_model.model.layers."
        let qWeightSuffix = ".self_attn.q_proj.weight"
        var layerStems: Set<String> = []
        for key in selectedKeys where key.hasPrefix(layerPrefix)
            && key.hasSuffix(qWeightSuffix)
        {
            layerStems.insert(String(key.dropLast(qWeightSuffix.count)))
        }
        guard !layerStems.isEmpty else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }

        var eligible: [EligibleLayer] = []
        for layerStem in layerStems.sorted() {
            guard let layer = try eligibleLayer(
                layerStem: layerStem,
                index: index,
                stagedHeaders: stagedHeaders,
                selectedKeys: selectedKeys,
                metadataHeader: metadataHeader
            ) else {
                continue
            }
            eligible.append(layer)
        }
        guard !eligible.isEmpty else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        guard !Set(index.weightMap.values).contains(shardName) else {
            throw MLXFastError.invalidInput(
                "checkpoint already contains reserved generated shard \(shardName)"
            )
        }
        eligible.sort { $0.tensorName < $1.tensorName }

        var headerObject: [String: Any] = [
            "__metadata__": ["format": "mlxfast-attention-cotiled-v1"]
        ]
        var cursor = 0
        for layer in eligible {
            let (tensorBytes, tensorOverflow) = layer.threadgroups
                .multipliedReportingOverflow(by: layer.wordsPerThreadgroup * 4)
            let (end, endOverflow) = cursor.addingReportingOverflow(tensorBytes)
            guard !tensorOverflow, !endOverflow else {
                throw MLXFastError.invalidInput(
                    "co-tiled attention payload size overflows Int for \(layer.layerStem)"
                )
            }
            headerObject[layer.tensorName] = [
                "dtype": TensorDType.u32.rawValue,
                "shape": [layer.threadgroups, layer.wordsPerThreadgroup],
                "data_offsets": [cursor, end],
            ]
            cursor = end
        }
        var header = try JSONSerialization.data(
            withJSONObject: headerObject,
            options: [.sortedKeys]
        )
        while !header.count.isMultiple(of: 8) {
            header.append(0x20)
        }

        let destination = destinationDirectory.appendingPathComponent(shardName)
        try Data().write(to: destination, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        var headerLength = UInt64(header.count).littleEndian
        try output.write(contentsOf: Data(bytes: &headerLength, count: 8))
        try output.write(contentsOf: header)

        var weightMap: [String: String] = [:]
        var totalBytes = 0
        for layer in eligible {
            var projections: [ProjectionSource] = []
            projections.reserveCapacity(layer.stems.count)
            for (position, stem) in layer.stems.enumerated() {
                let weight = try stagedTensorBytes(
                    named: "\(stem).weight",
                    stagingDirectory: stagingDirectory,
                    index: index,
                    stagedHeaders: stagedHeaders
                )
                let packed = try sidecarTensorBytes(
                    named: layer.packedNames[position],
                    sidecarURL: metadataSidecarURL,
                    header: metadataHeader
                )
                projections.append(
                    ProjectionSource(
                        weight: weight,
                        packed: packed,
                        rows: layer.rows[position],
                        indexBits: layer.indexBits[position],
                        slots: layer.slots[position]
                    )
                )
            }
            // Fail-closed: the shard header already declared this payload.
            guard let payload = makePayload(projections: projections),
                  payload.count == layer.threadgroups * layer.wordsPerThreadgroup * 4
            else {
                throw MLXFastError.invalidInput(
                    "co-tiled attention payload construction failed for \(layer.layerStem)"
                )
            }
            try output.write(contentsOf: payload)
            weightMap[layer.tensorName] = shardName
            totalBytes += payload.count
        }
        try output.synchronize()

        return GeneratedAffineMetadataReport(
            weightMap: weightMap,
            tensorByteCount: totalBytes
        )
    }

    /// Shape-driven eligibility for one attention layer. Sliding layers
    /// (`q : k : v` rows `2 : 1 : 1`, q rows a multiple of 8) map onto the
    /// four-slot QKV kernel; full-attention layers (no `v_proj`, `q : k`
    /// rows `8 : 1`, k rows a multiple of 4) map onto the nine-slot QK
    /// kernel. Every projection must be affine 4-bit with K = 5376 (672-word
    /// rows) and carry a transform-authored fixed12/13 packed index stream.
    private static func eligibleLayer(
        layerStem: String,
        index: CheckpointIndex,
        stagedHeaders: [String: SafetensorsHeader],
        selectedKeys: Set<String>,
        metadataHeader: SafetensorsHeader
    ) throws -> EligibleLayer? {
        func weightInfo(_ stem: String) throws -> SafetensorInfo? {
            let name = "\(stem).weight"
            guard selectedKeys.contains(name),
                  let shardName = index.weightMap[name],
                  let info = stagedHeaders[shardName]?.tensors[name]
            else {
                return nil
            }
            return info
        }
        func packedRecord(
            _ stem: String,
            rows: Int
        ) -> (name: String, indexBits: Int)? {
            for indexBits in [12, 13] {
                guard let suffix = PackedProjectionMetadataCoding.suffix(
                    indexBits: indexBits
                ), let bytesPerRow = PackedProjectionMetadataCoding.packedBytesPerRow(
                    groupsPerRow: groupsPerRow,
                    indexBits: indexBits
                ) else {
                    continue
                }
                let name = "\(stem)\(suffix)"
                guard let info = metadataHeader.tensors[name] else {
                    continue
                }
                guard info.dtype == TensorDType.u8.rawValue,
                      info.shape == [rows, bytesPerRow]
                else {
                    return nil
                }
                return (name, indexBits)
            }
            return nil
        }

        let qStem = "\(layerStem).self_attn.q_proj"
        let kStem = "\(layerStem).self_attn.k_proj"
        let vStem = "\(layerStem).self_attn.v_proj"
        guard let qInfo = try weightInfo(qStem),
              let kInfo = try weightInfo(kStem)
        else {
            return nil
        }
        let vInfo = try weightInfo(vStem)
        func productionWeightRows(_ info: SafetensorInfo) -> Int? {
            guard info.dtype == TensorDType.u32.rawValue,
                  info.shape.count == 2,
                  info.shape[1] == weightWordsPerRow,
                  info.shape[0] > 0
            else {
                return nil
            }
            return info.shape[0]
        }
        guard let qRows = productionWeightRows(qInfo),
              let kRows = productionWeightRows(kInfo)
        else {
            return nil
        }

        let stems: [String]
        let slots: [Int]
        let rows: [Int]
        let tensorName: String
        if let vInfo {
            // Sliding QKV: four slots [q0, q1, k, v].
            guard let vRows = productionWeightRows(vInfo),
                  qRows == 2 * kRows,
                  vRows == kRows,
                  qRows.isMultiple(of: 8)
            else {
                return nil
            }
            stems = [qStem, kStem, vStem]
            slots = [2, 1, 1]
            rows = [qRows, kRows, vRows]
            tensorName = "\(layerStem)\(slidingSuffix)"
        } else {
            // Full-attention QK: nine slots [q0..q7, k].
            guard qRows == 8 * kRows, kRows.isMultiple(of: 4) else {
                return nil
            }
            stems = [qStem, kStem]
            slots = [8, 1]
            rows = [qRows, kRows]
            tensorName = "\(layerStem)\(fullSuffix)"
        }

        var packedNames: [String] = []
        var indexBits: [Int] = []
        for (position, stem) in stems.enumerated() {
            guard let record = packedRecord(stem, rows: rows[position]) else {
                return nil
            }
            packedNames.append(record.name)
            indexBits.append(record.indexBits)
        }
        var slotIndexBits: [Int] = []
        for position in 0..<stems.count {
            slotIndexBits.append(
                contentsOf: [Int](repeating: indexBits[position], count: slots[position])
            )
        }
        guard let wordsPerThreadgroup = wordsPerThreadgroup(
            slotIndexBits: slotIndexBits
        ) else {
            return nil
        }
        let threadgroups = kRows / 4
        return EligibleLayer(
            layerStem: layerStem,
            tensorName: tensorName,
            stems: stems,
            packedNames: packedNames,
            indexBits: indexBits,
            slots: slots,
            rows: rows,
            threadgroups: threadgroups,
            wordsPerThreadgroup: wordsPerThreadgroup
        )
    }

    private static func stagedTensorBytes(
        named name: String,
        stagingDirectory: URL,
        index: CheckpointIndex,
        stagedHeaders: [String: SafetensorsHeader]
    ) throws -> Data {
        guard let shardName = index.weightMap[name],
              let header = stagedHeaders[shardName],
              let info = header.tensors[name]
        else {
            throw MLXFastError.invalidInput(
                "missing staged tensor metadata for \(name)"
            )
        }
        return try readTensorBytes(
            from: stagingDirectory.appendingPathComponent(shardName),
            dataBaseOffset: header.dataBaseOffset,
            info: info,
            name: name
        )
    }

    private static func sidecarTensorBytes(
        named name: String,
        sidecarURL: URL,
        header: SafetensorsHeader
    ) throws -> Data {
        guard let info = header.tensors[name] else {
            throw MLXFastError.invalidInput(
                "missing generated metadata tensor \(name)"
            )
        }
        return try readTensorBytes(
            from: sidecarURL,
            dataBaseOffset: header.dataBaseOffset,
            info: info,
            name: name
        )
    }

    private static func readTensorBytes(
        from url: URL,
        dataBaseOffset: UInt64,
        info: SafetensorInfo,
        name: String
    ) throws -> Data {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let (offset, overflow) = dataBaseOffset.addingReportingOverflow(
            UInt64(info.dataStart)
        )
        guard !overflow else {
            throw MLXFastError.invalidInput("tensor byte offset overflows for \(name)")
        }
        try handle.seek(toOffset: offset)
        let bytes = handle.readData(ofLength: info.byteCount)
        guard bytes.count == info.byteCount else {
            throw MLXFastError.invalidInput(
                "short read while authoring co-tiled payload for \(name)"
            )
        }
        return bytes
    }
}

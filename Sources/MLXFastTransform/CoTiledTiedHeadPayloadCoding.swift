import Foundation
import MLXFastCore

/// Offline authoring of the co-tiled tied vocabulary head decode payload
/// consumed by `gemma4_cotiled_tied_vocabulary_head_packed13_softcap_qmv_*`
/// in `Sources/MLXFastModel/Gemma4TiedVocabularyHead.swift`.
///
/// Motivation: the packed13 tied-head QMV reads the 4-bit weight words
/// (~704.6 MB/token) and the packed13 metadata words (~36.7 MB/token) as two
/// separate device streams. The promoted gate/up co-tile and the D3 co-tiled
/// QKV/QK payloads showed that merely packing metadata into separate streams
/// buys nothing, while interleaving the metadata bytes WITH the weight words
/// in exact kernel consumption order measures a real decode win. This coding
/// brings the same single-stream locality to the tied head: one contiguous
/// per-threadgroup tile stream carrying every weight word and every
/// kernel-consumed packed13 metadata byte in the exact order the kernel
/// consumes them.
///
/// ## Payload layout
///
/// The decode kernel dispatches grid (32, 65_536, 1) with threadgroup
/// (32, 4, 1): 16_384 threadgroups, each carrying four 32-lane simdgroup
/// slots, each slot owning four consecutive output rows. Threadgroup `t`
/// owns rows `16t..<16t+16`; slot `g` owns rows `16t + 4g ..< 16t + 4g + 4`.
///
/// Each slot walks the 84 metadata groups of its four rows as 21 four-group
/// decode blocks over K = 5376 (10 block pairs plus one unpaired tail
/// block). Per (slot, row, block) the 32 lanes consume one contiguous
/// 128-byte run of the source weight row (`row * 2688 + block * 128`). The
/// packed13 stream stores 84 x 13-bit indexes per row in 35 U32 words
/// (140 bytes); one block pair consumes columns `8p..<8p+8`, whose bits
/// `[104p, 104p + 104)` are EXACTLY bytes `[13p, 13p + 13)` of the row —
/// byte-aligned by construction (104 bits = 13 bytes). The unpaired tail
/// block consumes columns `80..<84`, bits `[1040, 1092)`, carried as bytes
/// `[130, 138)` (the trailing padding bytes 138..<140 are never copied).
///
/// The payload rearranges exactly those bytes, per threadgroup:
///
/// One pair tile `p in 0..<10` (1_076 U32 words = 4_304 bytes):
///
/// - `[0, 2048)` bytes: even-block (block `2p`) weights, slot-major; slot
///   `g` at byte `512 g`; within a slot row-major, row `r` at byte `128 r`;
///   the 128-byte run is the source weight row's block `2p` run, verbatim.
/// - `[2048, 4096)` bytes: odd-block (block `2p + 1`) weights, same order.
/// - `[4096, 4304)` bytes: packed13 metadata, slot-major; slot `g` at byte
///   `52 g`; row `r` at byte `13 r`; the 13-byte run is the source packed
///   row's bytes `[13p, 13p + 13)`, verbatim.
///
/// The tail tile after the ten pairs (544 words = 2_176 bytes):
///
/// - `[0, 2048)` bytes: block-20 weights, identical slot/row order.
/// - `[2048, 2176)` bytes: tail metadata, slot-major, 32 bytes per slot;
///   row `r` at byte `8 r`: the source packed row's bytes `[130, 138)`.
///
/// `wordsPerThreadgroup = 10 * 1076 + 544 = 11_304`. The payload tensor is
/// dtype U32, shape `[rows / 16, 11_304]`, written little-endian
/// byte-for-byte from the source runs so no dtype-view ambiguity exists
/// between authoring and the Metal kernel's `device uint*`. Production
/// geometry: 262_144 rows -> `[16_384, 11_304]`, 740_818_944 bytes
/// (~706.5 MiB) at rest.
enum CoTiledTiedHeadPayloadCoding {
    static let shardName = "mlxfast-tied-head-cotiled-payload.safetensors"
    static let payloadName =
        "language_model.model.embed_tokens.tied_head_cotiled_payload"

    static let groupsPerRow = 84
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

    /// 2048 even weights + 2048 odd weights + 208 metadata bytes.
    static let pairTileBytes = 2 * rowsPerThreadgroup * blockBytes
        + rowsPerThreadgroup * pairMetadataBytesPerRow
    /// 2048 block-20 weights + 128 tail metadata bytes.
    static let tailTileBytes = rowsPerThreadgroup * blockBytes
        + rowsPerThreadgroup * tailMetadataBytesPerRow
    /// Mirrored by `Gemma4TiedHeadCoTiledPayloadLayout.wordsPerThreadgroup`
    /// in `Sources/MLXFastModel`; a CPU test pins the two implementations
    /// together.
    static let wordsPerThreadgroup =
        (pairCount * pairTileBytes + tailTileBytes) / 4

    /// Build one co-tiled tied-head payload from the raw little-endian
    /// weight and packed13 index bytes. nil is fail-closed for malformed
    /// geometry or byte counts.
    static func makePayload(weight: Data, packed: Data, rows: Int) -> Data? {
        guard rows > 0, rows.isMultiple(of: rowsPerThreadgroup) else {
            return nil
        }
        let (weightBytes, weightOverflow) =
            rows.multipliedReportingOverflow(by: weightBytesPerRow)
        let (packedBytes, packedOverflow) =
            rows.multipliedReportingOverflow(by: packedBytesPerRow)
        guard !weightOverflow, !packedOverflow,
              weight.count == weightBytes,
              packed.count == packedBytes
        else {
            return nil
        }
        let threadgroups = rows / rowsPerThreadgroup
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
            weight.withUnsafeBytes { (weightBuffer: UnsafeRawBufferPointer) in
                packed.withUnsafeBytes { (packedBuffer: UnsafeRawBufferPointer) in
                    guard let weightBase = weightBuffer.baseAddress,
                          let packedBase = packedBuffer.baseAddress
                    else {
                        return
                    }
                    for threadgroup in 0..<threadgroups {
                        let tileBase = threadgroup * threadgroupBytes
                        for slot in 0..<slotsPerThreadgroup {
                            for row in 0..<rowsPerSlot {
                                let sourceRow = threadgroup * rowsPerThreadgroup
                                    + slot * rowsPerSlot + row
                                let weightRowBase = sourceRow * weightBytesPerRow
                                let packedRowBase = sourceRow * packedBytesPerRow
                                for pair in 0..<pairCount {
                                    let pairBase = tileBase + pair * pairTileBytes
                                    (outputBase + pairBase + 512 * slot + 128 * row)
                                        .copyMemory(
                                            from: weightBase + weightRowBase
                                                + (2 * pair) * blockBytes,
                                            byteCount: blockBytes
                                        )
                                    (outputBase + pairBase
                                        + 512 * slotsPerThreadgroup
                                        + 512 * slot + 128 * row)
                                        .copyMemory(
                                            from: weightBase + weightRowBase
                                                + (2 * pair + 1) * blockBytes,
                                            byteCount: blockBytes
                                        )
                                    (outputBase + pairBase
                                        + 1_024 * slotsPerThreadgroup
                                        + rowsPerSlot * pairMetadataBytesPerRow
                                            * slot
                                        + row * pairMetadataBytesPerRow)
                                        .copyMemory(
                                            from: packedBase + packedRowBase
                                                + pair * pairMetadataBytesPerRow,
                                            byteCount: pairMetadataBytesPerRow
                                        )
                                }
                                let tailBase = tileBase + pairCount * pairTileBytes
                                (outputBase + tailBase + 512 * slot + 128 * row)
                                    .copyMemory(
                                        from: weightBase + weightRowBase
                                            + 2 * pairCount * blockBytes,
                                        byteCount: blockBytes
                                    )
                                (outputBase + tailBase
                                    + 512 * slotsPerThreadgroup
                                    + 32 * slot + 8 * row)
                                    .copyMemory(
                                        from: packedBase + packedRowBase
                                            + tailMetadataRowByteOffset,
                                        byteCount: tailMetadataBytesPerRow
                                    )
                            }
                        }
                    }
                }
            }
        }
        return Data(payload)
    }

    /// Authors the co-tiled tied-head payload tensor into a dedicated
    /// generated shard. Eligibility is shape-driven and fail-open: when the
    /// tied-head packed13 sidecar was not authored (tiny synthetic fixtures,
    /// unquantized checkpoints, LUT overflow) or the geometry does not match
    /// the co-tiled kernel's slot layout, no payload is authored and the
    /// runtime keeps its current packed13 kernel. The payload derives purely
    /// from tensors already staged in this transform run (the tied embedding
    /// weight plus the packed13 metadata sidecar), so it is input-independent
    /// by construction.
    static func writeSidecar(
        stagingDirectory: URL,
        index: CheckpointIndex,
        stagedHeaders: [String: SafetensorsHeader],
        selectedKeys: Set<String>,
        destinationDirectory: URL
    ) throws -> GeneratedAffineMetadataReport {
        let weightName = TiedHeadMetadataCoding.weightName
        let metadataSidecarURL = stagingDirectory.appendingPathComponent(
            TiedHeadMetadataCoding.shardName
        )
        guard selectedKeys.contains(weightName),
              FileManager.default.fileExists(atPath: metadataSidecarURL.path)
        else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        let metadataHeader = try Safetensors.readHeader(metadataSidecarURL)
        guard let packedInfo = metadataHeader.tensors[
            TiedHeadMetadataCoding.packedIndicesName
        ], packedInfo.dtype == TensorDType.u32.rawValue,
            packedInfo.shape.count == 2,
            packedInfo.shape[1] == packedWordsPerRow,
            packedInfo.shape[0] > 0,
            packedInfo.shape[0].isMultiple(of: rowsPerThreadgroup)
        else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        let rows = packedInfo.shape[0]
        guard let weightShard = index.weightMap[weightName],
              let weightInfo = stagedHeaders[weightShard]?.tensors[weightName],
              weightInfo.dtype == TensorDType.u32.rawValue,
              weightInfo.shape == [rows, weightWordsPerRow]
        else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        guard !Set(index.weightMap.values).contains(shardName) else {
            throw MLXFastError.invalidInput(
                "checkpoint already contains reserved generated shard \(shardName)"
            )
        }

        let weight = try readTensorBytes(
            from: stagingDirectory.appendingPathComponent(weightShard),
            dataBaseOffset: stagedHeaders[weightShard]?.dataBaseOffset ?? 0,
            info: weightInfo,
            name: weightName
        )
        let packed = try readTensorBytes(
            from: metadataSidecarURL,
            dataBaseOffset: metadataHeader.dataBaseOffset,
            info: packedInfo,
            name: TiedHeadMetadataCoding.packedIndicesName
        )
        let threadgroups = rows / rowsPerThreadgroup
        // Fail-closed: the geometry was already declared eligible above.
        guard let payload = makePayload(weight: weight, packed: packed, rows: rows),
              payload.count == threadgroups * wordsPerThreadgroup * 4
        else {
            throw MLXFastError.invalidInput(
                "co-tiled tied-head payload construction failed"
            )
        }

        let headerObject: [String: Any] = [
            "__metadata__": ["format": "mlxfast-tied-head-cotiled-v1"],
            payloadName: [
                "dtype": TensorDType.u32.rawValue,
                "shape": [threadgroups, wordsPerThreadgroup],
                "data_offsets": [0, payload.count],
            ],
        ]
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
        try output.write(contentsOf: payload)
        try output.synchronize()

        return GeneratedAffineMetadataReport(
            weightMap: [payloadName: shardName],
            tensorByteCount: payload.count
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
                "short read while authoring co-tiled tied-head payload for \(name)"
            )
        }
        return bytes
    }
}

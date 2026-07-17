import Foundation
import MLXFastCore

struct GeneratedAffineMetadataReport {
    let weightMap: [String: String]
    let tensorByteCount: Int

    var tensorCount: Int {
        weightMap.count
    }

    var shardCount: Int {
        weightMap.isEmpty ? 0 : 1
    }
}

enum AffineMetadataCoding {
    static let shardName = "mlxfast-projection-metadata.safetensors"

    private enum TensorKind {
        case indices
        case lut
        case packedIndices(indexBits: Int)
    }

    private struct Projection {
        let stem: String
        let scalesName: String
        let biasesName: String
        let shape: [Int]
        let lut: [UInt32]
        let packedIndexBits: Int?
    }

    private struct GeneratedTensor {
        let name: String
        let dtype: TensorDType
        let shape: [Int]
        let byteCount: Int
        let projectionIndex: Int
        let kind: TensorKind
    }

    static func writeProjectionSidecar(
        sourceDirectory: URL,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader],
        selectedKeys: Set<String>,
        destinationDirectory: URL
    ) throws -> GeneratedAffineMetadataReport {
        let stems = selectedKeys.compactMap(indexedProjectionStem).sorted()
        guard !stems.isEmpty else {
            return GeneratedAffineMetadataReport(weightMap: [:], tensorByteCount: 0)
        }
        guard !Set(index.weightMap.values).contains(shardName) else {
            throw MLXFastError.invalidInput(
                "checkpoint already contains reserved generated shard \(shardName)"
            )
        }

        var projections: [Projection] = []
        projections.reserveCapacity(stems.count)
        for stem in stems {
            let weightName = "\(stem).weight"
            let scalesName = "\(stem).scales"
            let biasesName = "\(stem).biases"
            guard selectedKeys.contains(scalesName), selectedKeys.contains(biasesName) else {
                // Some valid non-production/synthetic checkpoints expose an
                // unquantized attention projection with no affine companions.
                // It is not eligible for indexed routing. MLP projections are
                // fixed quantized invariants and remain fail-closed.
                if stem.contains(".self_attn.") {
                    continue
                }
                throw MLXFastError.invalidInput(
                    "quantized projection \(weightName) is missing BF16 scales or biases"
                )
            }

            let weightInfo = try tensorInfo(
                named: weightName,
                index: index,
                sourceHeaders: sourceHeaders
            )
            let scalesInfo = try tensorInfo(
                named: scalesName,
                index: index,
                sourceHeaders: sourceHeaders
            )
            let biasesInfo = try tensorInfo(
                named: biasesName,
                index: index,
                sourceHeaders: sourceHeaders
            )
            guard weightInfo.dtype == TensorDType.u32.rawValue,
                  weightInfo.shape.count == 2,
                  scalesInfo.dtype == TensorDType.bf16.rawValue,
                  biasesInfo.dtype == TensorDType.bf16.rawValue,
                  scalesInfo.shape == biasesInfo.shape,
                  scalesInfo.shape.count == 2,
                  scalesInfo.shape[0] == weightInfo.shape[0],
                  scalesInfo.shape.allSatisfy({ $0 > 0 })
            else {
                throw MLXFastError.invalidInput(
                    "indexed projection \(stem) has incompatible weight, scale, or bias metadata"
                )
            }

            let scales = try tensorBytes(
                named: scalesName,
                sourceDirectory: sourceDirectory,
                index: index,
                sourceHeaders: sourceHeaders
            )
            let biases = try tensorBytes(
                named: biasesName,
                sourceDirectory: sourceDirectory,
                index: index,
                sourceHeaders: sourceHeaders
            )
            let lut = try makeOrderedLUT(scales: scales, biases: biases, name: stem)
            projections.append(
                Projection(
                    stem: stem,
                    scalesName: scalesName,
                    biasesName: biasesName,
                    shape: scalesInfo.shape,
                    lut: lut,
                    packedIndexBits: packedIndexBits(
                        stem: stem,
                        shape: scalesInfo.shape,
                        lutCount: lut.count
                    )
                )
            )
        }

        var tensors: [GeneratedTensor] = []
        tensors.reserveCapacity(projections.count * 2)
        for (projectionIndex, projection) in projections.enumerated() {
            let elementCount = try checkedElementCount(
                shape: projection.shape,
                name: projection.stem
            )
            tensors.append(
                GeneratedTensor(
                    name: "\(projection.stem).metadata_indices",
                    dtype: .u16,
                    shape: projection.shape,
                    byteCount: elementCount * TensorDType.u16.byteWidth,
                    projectionIndex: projectionIndex,
                    kind: .indices
                )
            )
            tensors.append(
                GeneratedTensor(
                    name: "\(projection.stem).metadata_lut",
                    dtype: .u32,
                    shape: [projection.lut.count],
                    byteCount: projection.lut.count * TensorDType.u32.byteWidth,
                    projectionIndex: projectionIndex,
                    kind: .lut
                )
            )
            if let indexBits = projection.packedIndexBits {
                guard let suffix = PackedProjectionMetadataCoding.suffix(
                    indexBits: indexBits
                ), let bytesPerRow = PackedProjectionMetadataCoding.packedBytesPerRow(
                    groupsPerRow: projection.shape[1],
                    indexBits: indexBits
                ) else {
                    throw MLXFastError.invalidInput(
                        "packed metadata layout is invalid for \(projection.stem)"
                    )
                }
                let (packedByteCount, packedOverflow) =
                    projection.shape[0].multipliedReportingOverflow(by: bytesPerRow)
                guard !packedOverflow else {
                    throw MLXFastError.invalidInput(
                        "packed metadata shape overflows for \(projection.stem)"
                    )
                }
                tensors.append(
                    GeneratedTensor(
                        name: "\(projection.stem)\(suffix)",
                        dtype: .u8,
                        shape: [projection.shape[0], bytesPerRow],
                        byteCount: packedByteCount,
                        projectionIndex: projectionIndex,
                        kind: .packedIndices(indexBits: indexBits)
                    )
                )
            }
        }
        tensors.sort { $0.name < $1.name }

        let destination = destinationDirectory.appendingPathComponent(shardName)
        try writeSidecar(
            destination,
            tensors: tensors,
            projections: projections,
            sourceDirectory: sourceDirectory,
            index: index,
            sourceHeaders: sourceHeaders
        )

        let totalBytes = tensors.reduce(0) { $0 + $1.byteCount }
        return GeneratedAffineMetadataReport(
            weightMap: Dictionary(uniqueKeysWithValues: tensors.map { ($0.name, shardName) }),
            tensorByteCount: totalBytes
        )
    }

    private static func indexedProjectionStem(_ name: String) -> String? {        let prefix = "language_model.model.layers."
        guard name.hasPrefix(prefix), name.hasSuffix(".weight") else {
            return nil
        }
        let remainder = name.dropFirst(prefix.count)
        guard let separator = remainder.firstIndex(of: "."),
              Int(remainder[..<separator]) != nil
        else {
            return nil
        }
        let suffix = remainder[separator...]
        guard suffix == ".mlp.gate_proj.weight"
                || suffix == ".mlp.up_proj.weight"
                || suffix == ".mlp.down_proj.weight"
                || suffix == ".self_attn.q_proj.weight"
                || suffix == ".self_attn.k_proj.weight"
                || suffix == ".self_attn.v_proj.weight"
        else {
            return nil
        }
        return String(name.dropLast(".weight".count))
    }

    /// Fixed-width packed authoring targets the fused attention-projection
    /// decode kernels, whose per-block metadata consumption matches the
    /// pair/tail layout in `PackedProjectionMetadataCoding`. MLP streams keep
    /// their runtime-packed promoted forms. nil skips packed authoring
    /// (fail-open to the runtime U16 path) for incompatible geometry or a LUT
    /// wider than 13 bits.
    private static func packedIndexBits(
        stem: String,
        shape: [Int],
        lutCount: Int
    ) -> Int? {
        guard stem.contains(".self_attn."),
              shape.count == 2,
              shape[0] > 0,
              let indexBits = PackedProjectionMetadataCoding.indexBits(
                  forLUTCount: lutCount
              ),
              PackedProjectionMetadataCoding.packedBytesPerRow(
                  groupsPerRow: shape[1],
                  indexBits: indexBits
              ) != nil
        else {
            return nil
        }
        return indexBits
    }

    private static func tensorInfo(
        named name: String,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader]
    ) throws -> SafetensorInfo {
        guard let shardName = index.weightMap[name],
              let info = sourceHeaders[shardName]?.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        return info
    }

    private static func tensorBytes(
        named name: String,
        sourceDirectory: URL,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader]
    ) throws -> Data {
        guard let sourceShard = index.weightMap[name],
              let header = sourceHeaders[sourceShard],
              let info = header.tensors[name]
        else {
            throw MLXFastError.invalidInput("missing validated tensor metadata for \(name)")
        }
        let source = sourceDirectory.appendingPathComponent(sourceShard)
        let handle = try FileHandle(forReadingFrom: source)
        defer { try? handle.close() }
        let (offset, overflow) = header.dataBaseOffset.addingReportingOverflow(
            UInt64(info.dataStart)
        )
        guard !overflow else {
            throw MLXFastError.invalidInput("tensor byte offset overflows for \(name)")
        }
        try handle.seek(toOffset: offset)
        let bytes = handle.readData(ofLength: info.byteCount)
        guard bytes.count == info.byteCount else {
            throw MLXFastError.invalidInput(
                "short read while generating indexed metadata for \(name)"
            )
        }
        return bytes
    }

    private static func makeOrderedLUT(
        scales: Data,
        biases: Data,
        name: String
    ) throws -> [UInt32] {
        guard scales.count == biases.count, scales.count.isMultiple(of: 2) else {
            throw MLXFastError.invalidInput(
                "scale and bias payloads differ for indexed metadata projection \(name)"
            )
        }
        var pairToIndex: [UInt32: UInt16] = [:]
        pairToIndex.reserveCapacity(8_192)
        var lut: [UInt32] = []
        lut.reserveCapacity(8_192)
        try scales.withUnsafeBytes { scaleBytes in
            try biases.withUnsafeBytes { biasBytes in
                for offset in stride(from: 0, to: scales.count, by: 2) {
                    let scale = scaleBytes.loadUnaligned(
                        fromByteOffset: offset,
                        as: UInt16.self
                    ).littleEndian
                    let bias = biasBytes.loadUnaligned(
                        fromByteOffset: offset,
                        as: UInt16.self
                    ).littleEndian
                    let pair = UInt32(scale) | (UInt32(bias) << 16)
                    if pairToIndex[pair] == nil {
                        guard lut.count < 65_536 else {
                            throw MLXFastError.invalidInput(
                                "affine metadata LUT exceeds UInt16 capacity for \(name)"
                            )
                        }
                        pairToIndex[pair] = UInt16(lut.count)
                        lut.append(pair)
                    }
                }
            }
        }
        guard !lut.isEmpty else {
            throw MLXFastError.invalidInput("affine metadata LUT is empty for \(name)")
        }
        return lut
    }

    private static func checkedElementCount(shape: [Int], name: String) throws -> Int {
        var count = 1
        for dimension in shape {
            let result = count.multipliedReportingOverflow(by: dimension)
            guard !result.overflow else {
                throw MLXFastError.invalidInput(
                    "generated metadata shape overflows for \(name)"
                )
            }
            count = result.partialValue
        }
        return count
    }

    private static func writeSidecar(
        _ destination: URL,
        tensors: [GeneratedTensor],
        projections: [Projection],
        sourceDirectory: URL,
        index: CheckpointIndex,
        sourceHeaders: [String: SafetensorsHeader]
    ) throws {
        var headerObject: [String: Any] = [
            "__metadata__": ["format": "mlxfast-projection-indexed-v3"]
        ]
        var cursor = 0
        for tensor in tensors {
            let (end, overflow) = cursor.addingReportingOverflow(tensor.byteCount)
            guard !overflow else {
                throw MLXFastError.invalidInput("generated metadata shard size overflows Int")
            }
            headerObject[tensor.name] = [
                "dtype": tensor.dtype.rawValue,
                "shape": tensor.shape,
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

        try Data().write(to: destination, options: [.withoutOverwriting])
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        var headerLength = UInt64(header.count).littleEndian
        try output.write(contentsOf: Data(bytes: &headerLength, count: 8))
        try output.write(contentsOf: header)

        for tensor in tensors {
            let projection = projections[tensor.projectionIndex]
            switch tensor.kind {
            case .indices:
                let scales = try tensorBytes(
                    named: projection.scalesName,
                    sourceDirectory: sourceDirectory,
                    index: index,
                    sourceHeaders: sourceHeaders
                )
                let biases = try tensorBytes(
                    named: projection.biasesName,
                    sourceDirectory: sourceDirectory,
                    index: index,
                    sourceHeaders: sourceHeaders
                )
                try output.write(contentsOf: try makeIndices(
                    scales: scales,
                    biases: biases,
                    lut: projection.lut,
                    name: projection.stem
                ))
            case .lut:
                try output.write(contentsOf: littleEndianData(projection.lut))
            case .packedIndices(let indexBits):
                let scales = try tensorBytes(
                    named: projection.scalesName,
                    sourceDirectory: sourceDirectory,
                    index: index,
                    sourceHeaders: sourceHeaders
                )
                let biases = try tensorBytes(
                    named: projection.biasesName,
                    sourceDirectory: sourceDirectory,
                    index: index,
                    sourceHeaders: sourceHeaders
                )
                let indices = try makeIndices(
                    scales: scales,
                    biases: biases,
                    lut: projection.lut,
                    name: projection.stem
                )
                let packed = try makePackedIndices(
                    indices: indices,
                    shape: projection.shape,
                    indexBits: indexBits,
                    name: projection.stem
                )
                guard packed.count == tensor.byteCount else {
                    throw MLXFastError.invalidInput(
                        "packed metadata byte count mismatch for \(projection.stem)"
                    )
                }
                try output.write(contentsOf: packed)
            }
        }
        try output.synchronize()
    }

    private static func makeIndices(
        scales: Data,
        biases: Data,
        lut: [UInt32],
        name: String
    ) throws -> Data {
        guard scales.count == biases.count, scales.count.isMultiple(of: 2) else {
            throw MLXFastError.invalidInput(
                "scale and bias payloads differ for indexed metadata projection \(name)"
            )
        }
        let pairToIndex = Dictionary(
            uniqueKeysWithValues: lut.enumerated().map { ($0.element, UInt16($0.offset)) }
        )
        let count = scales.count / 2
        var output = Data(count: count * 2)
        try output.withUnsafeMutableBytes { outputBytes in
            try scales.withUnsafeBytes { scaleBytes in
                try biases.withUnsafeBytes { biasBytes in
                    for element in 0..<count {
                        let offset = element * 2
                        let scale = scaleBytes.loadUnaligned(
                            fromByteOffset: offset,
                            as: UInt16.self
                        ).littleEndian
                        let bias = biasBytes.loadUnaligned(
                            fromByteOffset: offset,
                            as: UInt16.self
                        ).littleEndian
                        let pair = UInt32(scale) | (UInt32(bias) << 16)
                        guard let index = pairToIndex[pair] else {
                            throw MLXFastError.invalidInput(
                                "affine metadata pair missing from generated LUT for \(name)"
                            )
                        }
                        outputBytes.storeBytes(
                            of: index.littleEndian,
                            toByteOffset: offset,
                            as: UInt16.self
                        )
                    }
                }
            }
        }
        return output
    }

    /// Re-encode a little-endian U16 index payload into the fixed 12/13-bit
    /// layout. Fail-closed: authoring was declared in the shard header, so a
    /// pack failure aborts the transform instead of truncating the sidecar.
    private static func makePackedIndices(
        indices: Data,
        shape: [Int],
        indexBits: Int,
        name: String
    ) throws -> Data {
        guard shape.count == 2,
              indices.count.isMultiple(of: 2),
              indices.count / 2 == shape[0] * shape[1]
        else {
            throw MLXFastError.invalidInput(
                "packed metadata source shape is invalid for \(name)"
            )
        }
        var values = [UInt16](repeating: 0, count: indices.count / 2)
        indices.withUnsafeBytes { bytes in
            for element in values.indices {
                values[element] = bytes.loadUnaligned(
                    fromByteOffset: element * 2,
                    as: UInt16.self
                ).littleEndian
            }
        }
        guard let packed = PackedProjectionMetadataCoding.packFixedWidthIndices(
            values,
            rows: shape[0],
            groupsPerRow: shape[1],
            indexBits: indexBits
        ) else {
            throw MLXFastError.invalidInput(
                "packed metadata does not fit fixed\(indexBits) form for \(name)"
            )
        }
        return Data(packed)
    }

    private static func littleEndianData(_ values: [UInt32]) -> Data {
        var output = Data(count: values.count * 4)
        output.withUnsafeMutableBytes { bytes in
            for (index, value) in values.enumerated() {
                bytes.storeBytes(
                    of: value.littleEndian,
                    toByteOffset: index * 4,
                    as: UInt32.self
                )
            }
        }
        return output
    }
}

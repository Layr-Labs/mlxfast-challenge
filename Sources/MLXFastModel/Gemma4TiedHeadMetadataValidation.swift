import Foundation
import MLXFastCore

private enum Gemma4TiedHeadPacked13ByteLayout {
    static let weightName = "language_model.model.embed_tokens.weight"
    static let scalesName = "language_model.model.embed_tokens.scales"
    static let biasesName = "language_model.model.embed_tokens.biases"
    static let packedIndicesName =
        "language_model.model.embed_tokens.tied_head_packed13_indices"
    static let lutName =
        "language_model.model.embed_tokens.tied_head_packed13_lut"

    static let rowCount = 262_144
    static let groupsPerRow = 84
    static let bitsPerIndex = 13
    static let wordsPerRow = 35
    static let maximumLUTCount = 1 << bitsPerIndex
}

/// Proof that the raw packed sidecar exhaustively reconstructed the stock
/// scale/bias tensors before any MLX arrays were loaded. Partitioning requires
/// this descriptor and binds it to the loaded LUT length.
struct ValidatedGemma4TiedHeadPacked13Metadata: Sendable {
    let lutCount: Int
}

/// Exhaustively verifies the packed tied-head affine metadata using raw
/// safetensor bytes. This file intentionally does not import MLX: validating
/// 22,020,096 scale/bias pairs must not start a GPU command buffer or extend
/// the loaded telemetry window around model initialization.
func validateGemma4TiedHeadPacked13MetadataBytes(
    denseStore: DenseTensorStore
) throws -> ValidatedGemma4TiedHeadPacked13Metadata? {
    let packedRecord = denseStore.record(
        named: Gemma4TiedHeadPacked13ByteLayout.packedIndicesName
    )
    let lutRecord = denseStore.record(
        named: Gemma4TiedHeadPacked13ByteLayout.lutName
    )

    switch (packedRecord, lutRecord) {
    case (nil, nil):
        return nil
    case (nil, _), (_, nil):
        throw MLXFastError.invalidInput(
            "tied-head packed13 metadata payload is incomplete"
        )
    case let (packedRecord?, lutRecord?):
        guard let weightRecord = denseStore.record(
            named: Gemma4TiedHeadPacked13ByteLayout.weightName
        ), let scalesRecord = denseStore.record(
            named: Gemma4TiedHeadPacked13ByteLayout.scalesName
        ), let biasesRecord = denseStore.record(
            named: Gemma4TiedHeadPacked13ByteLayout.biasesName
        ) else {
            throw MLXFastError.invalidInput(
                "tied-head packed13 metadata is missing stock affine tensors"
            )
        }
        guard weightRecord.dtype == TensorDType.u32.rawValue,
              weightRecord.shape == [
                  Gemma4TiedHeadPacked13ByteLayout.rowCount,
                  672,
              ],
              scalesRecord.dtype == TensorDType.bf16.rawValue,
              scalesRecord.shape == [
                  Gemma4TiedHeadPacked13ByteLayout.rowCount,
                  Gemma4TiedHeadPacked13ByteLayout.groupsPerRow,
              ],
              biasesRecord.dtype == TensorDType.bf16.rawValue,
              biasesRecord.shape == scalesRecord.shape,
              packedRecord.dtype == TensorDType.u32.rawValue,
              packedRecord.shape == [
                  Gemma4TiedHeadPacked13ByteLayout.rowCount,
                  Gemma4TiedHeadPacked13ByteLayout.wordsPerRow,
              ],
              lutRecord.dtype == TensorDType.u32.rawValue,
              lutRecord.shape.count == 1,
              (1...Gemma4TiedHeadPacked13ByteLayout.maximumLUTCount).contains(
                  lutRecord.shape[0]
              )
        else {
            throw MLXFastError.invalidInput(
                "tied-head packed13 metadata has invalid dtype or shape"
            )
        }

        try validateGemma4Packed13AffineMetadataBytes(
            packedIndices: denseStore.tensorBytes(
                named: Gemma4TiedHeadPacked13ByteLayout.packedIndicesName
            ),
            lut: denseStore.tensorBytes(
                named: Gemma4TiedHeadPacked13ByteLayout.lutName
            ),
            scales: denseStore.tensorBytes(
                named: Gemma4TiedHeadPacked13ByteLayout.scalesName
            ),
            biases: denseStore.tensorBytes(
                named: Gemma4TiedHeadPacked13ByteLayout.biasesName
            ),
            rowCount: Gemma4TiedHeadPacked13ByteLayout.rowCount,
            groupsPerRow: Gemma4TiedHeadPacked13ByteLayout.groupsPerRow
        )
        return ValidatedGemma4TiedHeadPacked13Metadata(
            lutCount: lutRecord.shape[0]
        )
    }
}

func validateGemma4Packed13AffineMetadataBytes(
    packedIndices: Data,
    lut: Data,
    scales: Data,
    biases: Data,
    rowCount: Int,
    groupsPerRow: Int
) throws {
    let bitsPerIndex = Gemma4TiedHeadPacked13ByteLayout.bitsPerIndex
    guard rowCount > 0, groupsPerRow > 0 else {
        throw MLXFastError.invalidInput(
            "tied-head packed13 metadata has invalid logical shape"
        )
    }
    let (rowBitCount, rowBitOverflow) = groupsPerRow.multipliedReportingOverflow(
        by: bitsPerIndex
    )
    guard !rowBitOverflow else {
        throw MLXFastError.invalidInput(
            "tied-head packed13 metadata row bit count overflows"
        )
    }
    let (paddedRowBitCount, paddingOverflow) = rowBitCount.addingReportingOverflow(31)
    guard !paddingOverflow else {
        throw MLXFastError.invalidInput(
            "tied-head packed13 metadata padded row bit count overflows"
        )
    }
    let wordsPerRow = paddedRowBitCount / 32
    let (elementCount, elementOverflow) = rowCount.multipliedReportingOverflow(
        by: groupsPerRow
    )
    let (affineByteCount, affineOverflow) = elementCount.multipliedReportingOverflow(
        by: 2
    )
    let (wordCount, wordOverflow) = rowCount.multipliedReportingOverflow(
        by: wordsPerRow
    )
    let (packedByteCount, packedOverflow) = wordCount.multipliedReportingOverflow(
        by: 4
    )
    guard !elementOverflow, !affineOverflow, !wordOverflow, !packedOverflow,
          scales.count == affineByteCount,
          biases.count == affineByteCount,
          packedIndices.count == packedByteCount,
          !lut.isEmpty,
          lut.count.isMultiple(of: 4),
          lut.count / 4 <= Gemma4TiedHeadPacked13ByteLayout.maximumLUTCount
    else {
        throw MLXFastError.invalidInput(
            "tied-head packed13 metadata byte lengths are invalid"
        )
    }

    let lutCount = lut.count / 4
    try packedIndices.withUnsafeBytes { (packedBytes: UnsafeRawBufferPointer) in
        try lut.withUnsafeBytes { (lutBytes: UnsafeRawBufferPointer) in
            try scales.withUnsafeBytes { (scaleBytes: UnsafeRawBufferPointer) in
                try biases.withUnsafeBytes { (biasBytes: UnsafeRawBufferPointer) in
                    for row in 0..<rowCount {
                        let packedRowByteOffset = row * wordsPerRow * 4
                        let affineRowByteOffset = row * groupsPerRow * 2
                        for column in 0..<groupsPerRow {
                            let bitOffset = column * bitsPerIndex
                            let wordByteOffset =
                                packedRowByteOffset + (bitOffset / 32) * 4
                            let shift = bitOffset % 32
                            let low = packedBytes.loadUnaligned(
                                fromByteOffset: wordByteOffset,
                                as: UInt32.self
                            ).littleEndian
                            var index = low >> UInt32(shift)
                            if shift > 32 - bitsPerIndex {
                                let high = packedBytes.loadUnaligned(
                                    fromByteOffset: wordByteOffset + 4,
                                    as: UInt32.self
                                ).littleEndian
                                index |= high << UInt32(32 - shift)
                            }
                            index &= UInt32((1 << bitsPerIndex) - 1)
                            guard Int(index) < lutCount else {
                                throw MLXFastError.invalidInput(
                                    "tied-head packed13 metadata index exceeds LUT bounds "
                                        + "at row \(row), group \(column)"
                                )
                            }

                            let pair = lutBytes.loadUnaligned(
                                fromByteOffset: Int(index) * 4,
                                as: UInt32.self
                            ).littleEndian
                            let affineByteOffset =
                                affineRowByteOffset + column * 2
                            let scale = scaleBytes.loadUnaligned(
                                fromByteOffset: affineByteOffset,
                                as: UInt16.self
                            ).littleEndian
                            let bias = biasBytes.loadUnaligned(
                                fromByteOffset: affineByteOffset,
                                as: UInt16.self
                            ).littleEndian
                            guard UInt16(pair & 0xffff) == scale,
                                  UInt16(pair >> 16) == bias
                            else {
                                throw MLXFastError.invalidInput(
                                    "tied-head packed13 metadata does not reconstruct "
                                        + "stock affine payloads at row \(row), group "
                                        + "\(column)"
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

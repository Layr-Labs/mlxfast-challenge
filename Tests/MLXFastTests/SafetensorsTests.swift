import Foundation
import Testing
@testable import MLXFastCore

@Test
func safetensorsCopySubsetPreservesBytesForUnsortedTensorNames() throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source.safetensors")
    let destination = root.appendingPathComponent("destination.safetensors")
    try writeSafetensors(
        source,
        tensors: [
            TensorFixture(name: "a.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
            TensorFixture(name: "b.weight", dtype: "U8", shape: [3], data: Data([3, 4, 5])),
            TensorFixture(name: "c.weight", dtype: "U8", shape: [2], data: Data([6, 7])),
        ]
    )

    let copied = try Safetensors.copySubset(
        from: source,
        to: destination,
        tensorNames: ["c.weight", "a.weight"]
    )

    #expect(copied == 2)
    let header = try Safetensors.readHeader(destination)
    #expect(header.tensors.keys.sorted() == ["a.weight", "c.weight"])
    #expect(try tensorBytes(destination, header: header, name: "a.weight") == Data([1, 2]))
    #expect(try tensorBytes(destination, header: header, name: "c.weight") == Data([6, 7]))
}

@Test
func safetensorsCopySubsetRejectsMissingRequestedTensor() throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source.safetensors")
    let destination = root.appendingPathComponent("destination.safetensors")
    try writeSafetensors(
        source,
        tensors: [
            TensorFixture(name: "a.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
        ]
    )

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.copySubset(
            from: source,
            to: destination,
            tensorNames: ["missing.weight"]
        )
    }
}

@Test
func safetensorsCopySubsetUsesThrowingDestinationCreate() throws {
    let source = try String(
        contentsOfFile: "Sources/MLXFastCore/Safetensors.swift",
        encoding: .utf8
    )

    #expect(source.contains("try Data().write(to: destination"))
    #expect(!source.contains("createFile(atPath: destination.path"))
}

@Test
func safetensorsRejectsUInt64MaxHeaderLengthWithoutTrapping() throws {
    let root = try temporaryDirectory()
    let path = root.appendingPathComponent("oversized-header.safetensors")
    var headerLength = UInt64.max.littleEndian
    try Data(bytes: &headerLength, count: 8).write(to: path)

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.readHeader(path)
    }
}

@Test
func safetensorsRejectsTensorRangePastEndOfFile() throws {
    let root = try temporaryDirectory()
    let path = root.appendingPathComponent("out-of-range.safetensors")
    try writeRawSafetensors(
        path,
        headerObject: [
            "a.weight": [
                "dtype": "U8",
                "shape": [2],
                "data_offsets": [0, 2],
            ],
        ],
        data: Data([1])
    )

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.readHeader(path)
    }
}

@Test
func safetensorsCopySubsetRejectsOutputOffsetOverflow() throws {
    let first = SafetensorInfo(
        name: "a.weight",
        dtype: "U8",
        shape: [Int.max],
        dataStart: 0,
        dataEnd: Int.max
    )
    let second = SafetensorInfo(
        name: "b.weight",
        dtype: "U8",
        shape: [1],
        dataStart: 0,
        dataEnd: 1
    )

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.makeHeaderData(
            tensors: [first, second],
            metadata: [:]
        )
    }
}

@Test
func safetensorsCopySubsetRejectsSourceAsDestinationWithoutMutatingIt() throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source.safetensors")
    try writeSafetensors(
        source,
        tensors: [
            TensorFixture(name: "a.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
        ]
    )
    let original = try Data(contentsOf: source)

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.copySubset(
            from: source,
            to: source,
            tensorNames: ["a.weight"]
        )
    }
    #expect(try Data(contentsOf: source) == original)
}

@Test
func safetensorsCopySubsetRejectsHardLinkToSourceWithoutMutatingIt() throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source.safetensors")
    let destination = root.appendingPathComponent("destination.safetensors")
    try writeSafetensors(
        source,
        tensors: [
            TensorFixture(name: "a.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
        ]
    )
    try FileManager.default.linkItem(at: source, to: destination)
    let original = try Data(contentsOf: source)

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.copySubset(
            from: source,
            to: destination,
            tensorNames: ["a.weight"]
        )
    }
    #expect(try Data(contentsOf: source) == original)
    #expect(try Data(contentsOf: destination) == original)
}

@Test
func safetensorsCopySubsetRejectsDirectoryDestinationWithoutRemovingIt() throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source.safetensors")
    let destination = root.appendingPathComponent("destination", isDirectory: true)
    let sentinel = destination.appendingPathComponent("sentinel.txt")
    try writeSafetensors(
        source,
        tensors: [
            TensorFixture(name: "a.weight", dtype: "U8", shape: [2], data: Data([1, 2])),
        ]
    )
    try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
    try "preserve".write(to: sentinel, atomically: true, encoding: .utf8)

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.copySubset(
            from: source,
            to: destination,
            tensorNames: ["a.weight"]
        )
    }
    #expect(try String(contentsOf: sentinel, encoding: .utf8) == "preserve")
}

@Test
func safetensorsCopySubsetRejectsForgedValidatedHeaderWithoutMutatingDestination() throws {
    let root = try temporaryDirectory()
    let source = root.appendingPathComponent("source.safetensors")
    let destination = root.appendingPathComponent("destination.safetensors")
    try Data(repeating: 0, count: 32).write(to: source)
    try Data([9, 8, 7]).write(to: destination)
    let tensor = SafetensorInfo(
        name: "a.weight",
        dtype: "U8",
        shape: [1],
        dataStart: Int.min,
        dataEnd: Int.max
    )
    let header = SafetensorsHeader(
        headerLength: 8,
        metadata: [:],
        tensors: [tensor.name: tensor]
    )

    #expect(throws: MLXFastError.self) {
        _ = try Safetensors.copySubset(
            from: source,
            to: destination,
            tensorNames: [tensor.name],
            validatedHeader: header
        )
    }
    #expect(try Data(contentsOf: destination) == Data([9, 8, 7]))
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
    let data: Data
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + tensor.data.count],
        ]
        cursor += tensor.data.count
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        output.append(tensor.data)
    }
    try output.write(to: path)
}

private func writeRawSafetensors(
    _ path: URL,
    headerObject: [String: Any],
    data: Data
) throws {
    var header = try JSONSerialization.data(withJSONObject: headerObject, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }
    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    output.append(data)
    try output.write(to: path)
}

private func tensorBytes(_ path: URL, header: SafetensorsHeader, name: String) throws -> Data {
    let info = try #require(header.tensors[name])
    let data = try Data(contentsOf: path)
    let start = Int(header.dataBaseOffset) + info.dataStart
    return data.subdata(in: start..<(start + info.byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

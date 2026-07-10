import Darwin
import Foundation

public struct SafetensorInfo: Equatable {
    public let name: String
    public let dtype: String
    public let shape: [Int]
    public let dataStart: Int
    public let dataEnd: Int

    public var byteCount: Int {
        dataEnd - dataStart
    }

    public init(name: String, dtype: String, shape: [Int], dataStart: Int, dataEnd: Int) {
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.dataStart = dataStart
        self.dataEnd = dataEnd
    }
}

public struct SafetensorsHeader: Equatable {
    public let headerLength: Int
    public let metadata: [String: String]
    public let tensors: [String: SafetensorInfo]

    public var dataBaseOffset: UInt64 {
        UInt64(max(0, headerLength)) + 8
    }

    public init(headerLength: Int, metadata: [String: String], tensors: [String: SafetensorInfo]) {
        self.headerLength = headerLength
        self.metadata = metadata
        self.tensors = tensors
    }
}

public enum Safetensors {
    // Matches the safetensors reference implementation's header limit. Check
    // this before allocating the declared header buffer.
    public static let maximumHeaderByteCount = 100_000_000

    public static func readHeader(_ path: URL) throws -> SafetensorsHeader {
        let attributes = try FileManager.default.attributesOfItem(atPath: path.path)
        let fileByteCount = try fileSizeByteCount(from: attributes, path: path.path)
        guard fileByteCount >= 8 else {
            throw MLXFastError.invalidInput("safetensors file is too small: \(path.path)")
        }

        let handle = try FileHandle(forReadingFrom: path)
        defer {
            try? handle.close()
        }

        let prefix = handle.readData(ofLength: 8)
        guard prefix.count == 8 else {
            throw MLXFastError.invalidInput("safetensors file is too small: \(path.path)")
        }
        let rawHeaderLength = prefix.withUnsafeBytes { raw -> UInt64 in
            raw.loadUnaligned(as: UInt64.self).littleEndian
        }
        guard rawHeaderLength > 0 else {
            throw MLXFastError.invalidInput("safetensors header is empty: \(path.path)")
        }
        guard rawHeaderLength <= UInt64(maximumHeaderByteCount) else {
            throw MLXFastError.invalidInput(
                "safetensors header exceeds \(maximumHeaderByteCount) bytes: \(path.path)"
            )
        }
        guard let headerLength = Int(exactly: rawHeaderLength) else {
            throw MLXFastError.invalidInput("safetensors header exceeds Int range: \(path.path)")
        }
        guard headerLength <= fileByteCount - 8 else {
            throw MLXFastError.invalidInput("truncated safetensors header: \(path.path)")
        }

        let headerData = handle.readData(ofLength: headerLength)
        guard headerData.count == headerLength else {
            throw MLXFastError.invalidInput("truncated safetensors header: \(path.path)")
        }

        let object: Any
        do {
            object = try JSONSerialization.jsonObject(with: headerData)
        } catch {
            throw MLXFastError.invalidInput("invalid safetensors header JSON in \(path.path)")
        }
        guard let dictionary = object as? [String: Any] else {
            throw MLXFastError.invalidInput("safetensors header must be a JSON object: \(path.path)")
        }

        let dataByteCount = fileByteCount - 8 - headerLength
        var metadata: [String: String] = [:]
        var tensors: [String: SafetensorInfo] = [:]
        for (name, value) in dictionary {
            if name == "__metadata__" {
                if let raw = value as? [String: String] {
                    metadata = raw
                }
                continue
            }
            guard let tensor = value as? [String: Any] else {
                continue
            }
            guard
                let dtype = tensor["dtype"] as? String,
                let shape = tensor["shape"] as? [Int],
                let offsets = tensor["data_offsets"] as? [Int],
                offsets.count == 2
            else {
                throw MLXFastError.invalidInput("invalid tensor header for \(name) in \(path.path)")
            }
            guard offsets[0] >= 0, offsets[1] >= offsets[0] else {
                throw MLXFastError.invalidInput("invalid data_offsets for \(name) in \(path.path)")
            }
            guard offsets[1] <= dataByteCount else {
                throw MLXFastError.invalidInput(
                    "data_offsets for \(name) exceed safetensors file size in \(path.path)"
                )
            }
            tensors[name] = SafetensorInfo(
                name: name,
                dtype: dtype,
                shape: shape,
                dataStart: offsets[0],
                dataEnd: offsets[1]
            )
        }

        return SafetensorsHeader(
            headerLength: headerLength,
            metadata: metadata,
            tensors: tensors
        )
    }

    public static func copySubset(
        from source: URL,
        to destination: URL,
        tensorNames: [String]
    ) throws -> Int {
        let header = try readHeader(source)
        return try copySubset(
            from: source,
            to: destination,
            tensorNames: tensorNames,
            validatedHeader: header
        )
    }

    public static func copySubset(
        from source: URL,
        to destination: URL,
        tensorNames: [String],
        validatedHeader header: SafetensorsHeader
    ) throws -> Int {
        guard header.headerLength > 0,
              header.headerLength <= maximumHeaderByteCount
        else {
            throw MLXFastError.invalidInput(
                "invalid validated safetensors header length for \(source.lastPathComponent)"
            )
        }
        let (baseOffsetInt, baseOffsetOverflow) = header.headerLength.addingReportingOverflow(8)
        guard !baseOffsetOverflow,
              let baseOffset = UInt64(exactly: baseOffsetInt)
        else {
            throw MLXFastError.invalidInput(
                "validated safetensors header offset overflows for \(source.lastPathComponent)"
            )
        }
        let attributes = try FileManager.default.attributesOfItem(atPath: source.path)
        let sourceByteCount = try fileSizeByteCount(from: attributes, path: source.path)
        guard baseOffsetInt <= sourceByteCount else {
            throw MLXFastError.invalidInput(
                "validated safetensors header exceeds file size for \(source.lastPathComponent)"
            )
        }
        let sourceDataByteCount = sourceByteCount - baseOffsetInt

        var selected: [SafetensorInfo] = []
        selected.reserveCapacity(tensorNames.count)
        for name in tensorNames {
            guard let tensor = header.tensors[name] else {
                throw MLXFastError.invalidInput(
                    "tensor \(name) requested from \(source.lastPathComponent) but missing from safetensors header"
                )
            }
            let (byteCount, byteCountOverflow) = tensor.dataEnd.subtractingReportingOverflow(
                tensor.dataStart
            )
            guard tensor.name == name,
                  !byteCountOverflow,
                  tensor.dataStart >= 0,
                  tensor.dataEnd >= tensor.dataStart,
                  tensor.dataEnd <= sourceDataByteCount,
                  byteCount >= 0
            else {
                throw MLXFastError.invalidInput(
                    "invalid validated safetensors tensor range for \(name)"
                )
            }
            selected.append(tensor)
        }
        selected.sort { $0.name < $1.name }
        guard !selected.isEmpty else {
            return 0
        }
        try validateCopyDestination(source: source, destination: destination)

        try FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        if FileManager.default.fileExists(atPath: destination.path) {
            try FileManager.default.removeItem(at: destination)
        }
        // FileManager.createFile can return false under macOS Seatbelt even when
        // direct writes are allowed; Data.write gives us a real throwing create.
        try Data().write(to: destination, options: [])

        let input = try FileHandle(forReadingFrom: source)
        let output = try FileHandle(forWritingTo: destination)
        defer {
            try? input.close()
            try? output.close()
        }

        let outputHeader = try makeHeaderData(
            tensors: selected,
            metadata: header.metadata
        )
        var headerLength = UInt64(outputHeader.count).littleEndian
        let prefix = Data(bytes: &headerLength, count: 8)
        output.write(prefix)
        output.write(outputHeader)

        for tensor in selected {
            guard let relativeOffset = UInt64(exactly: tensor.dataStart) else {
                throw MLXFastError.invalidInput(
                    "negative safetensors tensor offset for \(tensor.name)"
                )
            }
            let (absoluteOffset, overflow) = baseOffset.addingReportingOverflow(
                relativeOffset
            )
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "safetensors tensor offset overflows UInt64 for \(tensor.name)"
                )
            }
            try copyBytes(
                from: input,
                to: output,
                offset: absoluteOffset,
                count: tensor.byteCount
            )
        }

        return selected.count
    }

    static func makeHeaderData(
        tensors: [SafetensorInfo],
        metadata: [String: String]
    ) throws -> Data {
        var object: [String: Any] = [:]
        if !metadata.isEmpty {
            object["__metadata__"] = metadata
        }

        var cursor = 0
        for tensor in tensors.sorted(by: { $0.name < $1.name }) {
            let (end, overflow) = cursor.addingReportingOverflow(tensor.byteCount)
            guard !overflow else {
                throw MLXFastError.invalidInput(
                    "safetensors output byte offsets overflow Int for \(tensor.name)"
                )
            }
            object[tensor.name] = [
                "dtype": tensor.dtype,
                "shape": tensor.shape,
                "data_offsets": [cursor, end],
            ]
            cursor = end
        }

        var data = try JSONSerialization.data(
            withJSONObject: object,
            options: [.sortedKeys]
        )
        while data.count % 8 != 0 {
            data.append(0x20)
        }
        return data
    }

    private static func validateCopyDestination(source: URL, destination: URL) throws {
        let canonicalSource = source.standardizedFileURL.resolvingSymlinksInPath()
        let canonicalDestination = destination.standardizedFileURL.resolvingSymlinksInPath()
        guard canonicalSource.path != canonicalDestination.path else {
            throw MLXFastError.invalidInput(
                "safetensors source and destination must be different files: \(source.path)"
            )
        }

        var destinationStatus = stat()
        let destinationResult = destination.path.withCString {
            Darwin.lstat($0, &destinationStatus)
        }
        if destinationResult != 0 {
            guard errno == ENOENT else {
                throw MLXFastError.invalidInput(
                    "could not inspect safetensors destination: \(destination.path)"
                )
            }
            return
        }

        guard (destinationStatus.st_mode & S_IFMT) == S_IFREG else {
            throw MLXFastError.invalidInput(
                "safetensors destination exists and is not a regular file: \(destination.path)"
            )
        }

        var sourceStatus = stat()
        let sourceResult = source.path.withCString {
            stat($0, &sourceStatus)
        }
        if sourceResult == 0,
           sourceStatus.st_dev == destinationStatus.st_dev,
           sourceStatus.st_ino == destinationStatus.st_ino
        {
            throw MLXFastError.invalidInput(
                "safetensors source and destination refer to the same file: \(source.path)"
            )
        }
    }

    private static func copyBytes(
        from input: FileHandle,
        to output: FileHandle,
        offset: UInt64,
        count: Int
    ) throws {
        try input.seek(toOffset: offset)
        var remaining = count
        let chunkSize = 8 * 1024 * 1024
        while remaining > 0 {
            let data = input.readData(ofLength: min(chunkSize, remaining))
            if data.isEmpty {
                throw MLXFastError.invalidInput("unexpected EOF while copying safetensors tensor data")
            }
            output.write(data)
            remaining -= data.count
        }
    }
}

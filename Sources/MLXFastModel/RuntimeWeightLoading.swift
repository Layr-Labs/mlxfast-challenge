import Foundation
import MLX
import MLXFastCore

// Model-agnostic transformed-weight loading helpers shared by the runtime
// weight caches. Everything here is keyed only on the
// safetensors index/shard structure and the `language_model.` text-tower
// prefix convention -- nothing is specific to one architecture, so the
// model-specific loaders can come and go without touching this file.

func loadRuntimeWeightArrays(
    denseStore: DenseTensorStore,
    excludingOriginalNames: Set<String> = []
) throws -> [String: MLXArray] {
    let bridge = MLXArrayTensorBridge()
    return try loadRuntimeWeightValues(
        denseStore: denseStore,
        excludingOriginalNames: excludingOriginalNames
    ) { tensor in
        try bridge.makeArray(from: tensor)
    }
}

/// Materializes runtime weights one tensor at a time. `DenseTensorStore`
/// drains the source buffer's autorelease pool before visiting the next
/// tensor; only the detached value returned by `makeValue` remains resident.
/// The production value is an `MLXArray`, whose Data initializer copies the
/// bytes into MLX-owned storage.
func loadRuntimeWeightValues<Value>(
    denseStore: DenseTensorStore,
    excludingOriginalNames: Set<String> = [],
    makeValue: (MaterializedTensor) throws -> Value
) throws -> [String: Value] {
    let indexedNames = Set(denseStore.tensorNames)
    let unknownExclusions = excludingOriginalNames.subtracting(indexedNames).sorted()
    guard unknownExclusions.isEmpty else {
        throw MLXFastError.invalidInput(
            "runtime weight exclusions contain unindexed tensors: "
                + unknownExclusions.joined(separator: ", ")
        )
    }

    let directory = URL(fileURLWithPath: denseStore.weightsPath)
    let entries = try FileManager.default.contentsOfDirectory(
        at: directory, includingPropertiesForKeys: nil
    )
    let discoveredShards = entries
        .filter { $0.pathExtension == "safetensors" }
        .map(\.lastPathComponent)
    let shardNames = try validateRuntimeShardInventory(
        referencedShards: denseStore.shardNames,
        discoveredShards: discoveredShards
    )

    var loadedWeights: [String: Value] = [:]
    loadedWeights.reserveCapacity(indexedNames.count - excludingOriginalNames.count)
    var expectedPhysicalNames: Set<String> = []
    var expectedMaterializedNames: Set<String> = []
    var materializedNames: Set<String> = []
    var nameTracker = RuntimeWeightNameTracker()
    for shardName in shardNames {
        let expectedNames = denseStore.tensorNames(inShard: shardName)
        expectedPhysicalNames.formUnion(expectedNames)
        expectedMaterializedNames.formUnion(
            expectedNames.subtracting(excludingOriginalNames)
        )
        let shard = directory.appendingPathComponent(shardName)
        let discoveredNames = Set(try Safetensors.readHeader(shard).tensors.keys)
        try validateRuntimeTensorInventory(
            shardName: shardName,
            expectedNames: expectedNames,
            discoveredNames: discoveredNames
        )

        // Register every physical name before applying the materialization
        // filter. Excluded tensors therefore still participate in duplicate
        // and post-prefix collision checks, and validateComplete continues to
        // cover the full on-disk inventory rather than only returned values.
        var runtimeNamesByOriginal: [String: String] = [:]
        runtimeNamesByOriginal.reserveCapacity(expectedNames.count)
        for originalName in expectedNames.sorted() {
            runtimeNamesByOriginal[originalName] = try nameTracker.register(
                originalName: originalName,
                shardName: shardName,
                expectedNames: expectedNames
            )
        }

        try denseStore.forEachMaterializedTensor(
            inShard: shardName,
            excludingNames: excludingOriginalNames
        ) { record, tensor in
            guard materializedNames.insert(record.name).inserted else {
                throw MLXFastError.invalidInput(
                    "duplicate materialized safetensors tensor \(record.name)"
                )
            }
            guard let renamed = runtimeNamesByOriginal[record.name] else {
                throw MLXFastError.invalidInput(
                    "safetensors tensor \(record.name) was not registered in \(shardName)"
                )
            }
            loadedWeights[renamed] = try makeValue(tensor)
        }
    }
    try nameTracker.validateComplete(expectedNames: expectedPhysicalNames)

    let missingMaterialized = expectedMaterializedNames
        .subtracting(materializedNames).sorted()
    guard missingMaterialized.isEmpty else {
        throw MLXFastError.invalidInput(
            "indexed safetensors tensors were not materialized: "
                + missingMaterialized.joined(separator: ", ")
        )
    }
    let unexpectedMaterialized = materializedNames
        .subtracting(expectedMaterializedNames).sorted()
    guard unexpectedMaterialized.isEmpty else {
        throw MLXFastError.invalidInput(
            "excluded or unindexed safetensors tensors were materialized: "
                + unexpectedMaterialized.joined(separator: ", ")
        )
    }
    return loadedWeights
}

func validateRuntimeTensorInventory(
    shardName: String,
    expectedNames: Set<String>,
    discoveredNames: Set<String>
) throws {
    let missing = expectedNames.subtracting(discoveredNames).sorted()
    guard missing.isEmpty else {
        throw MLXFastError.invalidInput(
            "safetensors shard \(shardName) is missing indexed tensors: "
                + missing.joined(separator: ", ")
        )
    }
    let unindexed = discoveredNames.subtracting(expectedNames).sorted()
    guard unindexed.isEmpty else {
        throw MLXFastError.invalidInput(
            "safetensors shard \(shardName) contains unindexed or misplaced tensors: "
                + unindexed.joined(separator: ", ")
        )
    }
}

func validateRuntimeShardInventory(
    referencedShards: [String],
    discoveredShards: [String]
) throws -> [String] {
    let referenced = Set(referencedShards)
    let discovered = Set(discoveredShards)
    guard !referenced.isEmpty else {
        throw MLXFastError.missingFile("dense safetensors index references no shards")
    }
    let missing = referenced.subtracting(discovered).sorted()
    guard missing.isEmpty else {
        throw MLXFastError.missingFile(
            "dense safetensors index references missing shards: \(missing.joined(separator: ", "))"
        )
    }
    let unindexed = discovered.subtracting(referenced).sorted()
    guard unindexed.isEmpty else {
        throw MLXFastError.invalidInput(
            "weights directory contains unindexed safetensors shards: \(unindexed.joined(separator: ", "))"
        )
    }
    return referenced.sorted()
}

struct RuntimeWeightNameTracker {
    private static let languageModelPrefix = "language_model."
    private var originalNames: Set<String> = []
    private var runtimeNames: Set<String> = []

    mutating func register(
        originalName: String,
        shardName: String,
        expectedNames: Set<String>
    ) throws -> String {
        guard expectedNames.contains(originalName) else {
            throw MLXFastError.invalidInput(
                "safetensors shard \(shardName) contains unindexed or misplaced tensor \(originalName)"
            )
        }
        guard originalNames.insert(originalName).inserted else {
            throw MLXFastError.invalidInput("duplicate safetensors tensor \(originalName)")
        }
        let runtimeName = originalName.hasPrefix(Self.languageModelPrefix)
            ? String(originalName.dropFirst(Self.languageModelPrefix.count))
            : originalName
        guard runtimeNames.insert(runtimeName).inserted else {
            throw MLXFastError.invalidInput(
                "safetensors tensor names collide after runtime rename: \(runtimeName)"
            )
        }
        return runtimeName
    }

    func validateComplete(expectedNames: Set<String>) throws {
        let missing = expectedNames.subtracting(originalNames).sorted()
        guard missing.isEmpty else {
            throw MLXFastError.invalidInput(
                "indexed safetensors tensors were not loaded: \(missing.joined(separator: ", "))"
            )
        }
    }
}

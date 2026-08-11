// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

public protocol KVCache: Evaluatable, Updatable {
    var offset: Int { get }

    var maxSize: Int? { get }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray)

    var state: [MLXArray] { get set }

    var metaState: [String] { get set }

    var isTrimmable: Bool { get }

    @discardableResult
    func trim(_ n: Int) -> Int

    func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode

    func copy() -> any KVCache
}

public protocol QuantizedKVCacheProtocol: KVCache {
    var groupSize: Int { get }

    var bits: Int { get }

    var mode: QuantizationMode { get }

    func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )

    func getQuantizedState() -> ((MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?))?
}

open class BaseKVCache: KVCache {
    public var offset: Int = 0
    public var maxSize: Int? { nil }

    public func innerState() -> [MLXArray] { [] }

    open func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("update(keys:values:) must be implemented by subclass")
    }

    open var state: [MLXArray] {
        get { [] }
        set {
            if !newValue.isEmpty {
                fatalError("This cache has no state but a state was set.")
            }
        }
    }

    open var metaState: [String] {
        get { [""] }
        set {
            guard newValue.count == 1 && newValue[0].isEmpty else {
                fatalError("This cache has no meta_state but a meta_state was set.")
            }
        }
    }

    open var isTrimmable: Bool { false }

    @discardableResult
    open func trim(_ n: Int) -> Int { 0 }

    open func copy() -> any KVCache {
        fatalError("copy() must be implemented by subclass")
    }

    open func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1 {
            return .none
        }

        if returnArray || (windowSize != nil && n > windowSize!) {
            return .array(createCausalMask(n: n, offset: offset, windowSize: windowSize))
        }

        return .causal
    }
}

public func createCausalMask(
    n: Int,
    offset: Int,
    windowSize: Int? = nil,
    lengths: MLXArray? = nil,
    leftPadding: MLXArray? = nil
) -> MLXArray {
    var rinds = MLXArray(Int32(0) ..< Int32(offset + n))
    var linds = offset != 0 ? MLXArray(Int32(offset) ..< Int32(offset + n)) : rinds
    linds = linds[0..., .newAxis]
    rinds = rinds[.newAxis]
    var mask = linds .>= rinds

    if let windowSize {
        mask = mask & (linds .< rinds + windowSize)
    }

    if var lengths {
        lengths = lengths[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (rinds .< lengths)
    }

    if var leftPadding {
        leftPadding = leftPadding[0..., .newAxis, .newAxis, .newAxis]
        mask = mask & (leftPadding .<= rinds)
    }

    return mask
}

public func makeAttentionMask(
    n: Int,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    if let cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    if n == 1 {
        return .none
    }

    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }

    return .causal
}

@_disfavoredOverload
public func createAttentionMask(h: MLXArray, cache: [KVCache]?) -> MLXArray? {
    let t = h.dim(1)
    if t > 1 {
        var offset = 0
        if let c = cache?.first {
            offset = c.offset
        }
        return createCausalMask(n: t, offset: offset)
    }
    return nil
}

@available(
    *, deprecated,
    message: "Use createAttentionMask(h:cache:windowSize:returnArray:) with a single cache instead"
)
public func createAttentionMask(h: MLXArray, cache: [KVCache]?, returnArray: Bool = false)
    -> MLXFast.ScaledDotProductAttentionMaskMode
{
    let t = h.dim(1)
    if t > 1 {
        var returnArray = returnArray
        var offset = 0
        var windowSize: Int? = nil
        if let c = cache?.first {
            offset = c.offset
            if let maxSize = c.maxSize {
                windowSize = maxSize
                offset = min(maxSize - 1, offset)
                if !returnArray {
                    returnArray = offset + t > maxSize
                }
            }
        }

        if returnArray {
            return .array(createCausalMask(n: t, offset: offset, windowSize: windowSize))
        } else {
            return .causal
        }
    }
    return .none
}

public func createAttentionMask(
    h: MLXArray,
    cache: KVCache?,
    windowSize: Int? = nil,
    returnArray: Bool = false
) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let n = h.dim(1)

    if let cache = cache {
        return cache.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    if n == 1 {
        return .none
    }
    if returnArray || (windowSize != nil && n > windowSize!) {
        return .array(createCausalMask(n: n, offset: 0, windowSize: windowSize))
    }
    return .causal
}

public func createSSMMask(h: MLXArray, cache: MambaCache?) -> MLXArray? {
    if let cache {
        return cache.makeMask(N: h.dim(1))
    }
    return nil
}

public class KVCacheSimple: BaseKVCache, CustomDebugStringConvertible {
    internal var keys: MLXArray?
    internal var values: MLXArray?
    public var step = 256

    public override init() {
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let previous = self.offset
        let tokenCount = keys.dim(2)

        if self.keys == nil, previous == 0, tokenCount > 0,
            tokenCount.isMultiple(of: step)
        {
            self.keys = keys
            self.values = values
            self.offset = tokenCount
            return (keys, values)
        }

        let reset =
            if let currentKeys = self.keys, (previous + tokenCount) > currentKeys.dim(2) {
                true
            } else {
                self.keys == nil
            }
        if reset {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + tokenCount - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if previous % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<previous, 0...]
                    currentValues = currentValues[.ellipsis, ..<previous, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        self.offset += tokenCount

        self.keys?[.ellipsis, previous ..< self.offset, 0...] = keys
        self.values?[.ellipsis, previous ..< self.offset, 0...] = values

        let returnedKeys = self.keys![.ellipsis, ..<self.offset, 0...]
        let returnedValues = self.values![.ellipsis, ..<self.offset, 0...]

        return (returnedKeys, returnedValues)
    }


    private var fusedAppendContiguized = false

    public func fusedAppendPrepare() -> (keys: MLXArray, values: MLXArray, writeIdx: Int)? {
        guard let currentKeys = keys, let currentValues = values,
            offset + 1 <= currentKeys.dim(2),
            currentValues.dim(2) == currentKeys.dim(2)
        else { return nil }
        if !fusedAppendContiguized {
            keys = contiguous(currentKeys)
            values = contiguous(currentValues)
            fusedAppendContiguized = true
        }
        return (keys!, values!, offset)
    }

    public func fusedAppendAdvance() {
        offset += 1
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset == keys.dim(2) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("KVCacheSimple state must have exactly 2 arrays (keys, values)")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
            self.offset = self.keys!.dim(2)
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        if let keys = self.keys, let values = self.values {
            let currentKeys = keys[.ellipsis, ..<offset, 0...]
            let currentValues = values[.ellipsis, ..<offset, 0...]
            guard
                let effectiveGroupSize = resolvedKVQuantizationGroupSize(
                    requested: groupSize,
                    keyHeadDim: currentKeys.dim(3),
                    valueHeadDim: currentValues.dim(3)
                )
            else {
                fatalError(
                    "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(currentKeys.dim(3)). Value head dim: \(currentValues.dim(3))."
                )
            }
            let quantizedCache = QuantizedKVCache(groupSize: effectiveGroupSize, bits: bits)
            quantizedCache.offset = self.offset

            let quantizedKeys = quantized(currentKeys, groupSize: effectiveGroupSize, bits: bits)
            let quantizedValues = quantized(currentValues, groupSize: effectiveGroupSize, bits: bits)

            quantizedCache.state = [
                quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases,
                quantizedValues.wq, quantizedValues.scales, quantizedValues.biases,
            ].compactMap { $0 }

            return quantizedCache
        }

        let quantizedCache = QuantizedKVCache(groupSize: groupSize, bits: bits)
        quantizedCache.offset = self.offset
        return quantizedCache
    }

    public override func copy() -> any KVCache {
        let new = KVCacheSimple()
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        return new
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) \(Unmanaged.passUnretained(self).toOpaque()), offset: \(offset), step: \(step), keys: \(keys?.shape.description ?? "-"), values: \(values?.shape.description ?? "-")"
    }
}

public class RotatingKVCache: BaseKVCache, CustomDebugStringConvertible {
    var keep: Int
    var keys: MLXArray?
    var values: MLXArray?
    var maxCacheSize: Int
    var step: Int
    var idx: Int = 0

    public override var maxSize: Int? { maxCacheSize }

    public init(maxSize: Int, keep: Int = 0, step: Int = 256) {
        self.maxCacheSize = maxSize
        self.keep = keep
        self.step = step
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        [self.keys, self.values].compactMap { $0 }
    }

    private func trim(trimSize: Int, _ array: MLXArray, append: MLXArray? = nil) -> MLXArray {
        var toCat: [MLXArray] = []
        if trimSize > 0 {
            toCat = [
                array[.ellipsis, ..<keep, 0...],
                array[.ellipsis, (trimSize + keep)..., 0...],
            ]
        } else {
            toCat = [array]
        }
        if let append {
            toCat.append(append)
        }
        return concatenated(toCat, axis: 2)
    }

    private func temporalOrder(_ array: MLXArray) -> MLXArray {
        if idx == array.dim(2) {
            return array
        } else if idx < offset {
            return concatenated(
                [
                    array[.ellipsis, ..<keep, 0...],
                    array[.ellipsis, idx..., 0...],
                    array[.ellipsis, keep ..< idx, 0...],
                ], axis: 2)
        } else {
            return array[.ellipsis, ..<idx, 0...]
        }
    }

    private func updateConcat(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        if self.keys == nil {
            self.keys = keys
            self.values = values
        } else {
            self.keys = temporalOrder(self.keys!)
            self.values = temporalOrder(self.values!)
            idx = self.keys!.dim(2)

            let trimSize = idx - maxCacheSize + 1
            self.keys = trim(trimSize: trimSize, self.keys!, append: keys)
            self.values = trim(trimSize: trimSize, self.values!, append: values)
        }

        offset += keys.dim(2)
        idx = self.keys!.dim(2)

        return (self.keys!, self.values!)
    }

    private func updateInPlace(
        keys: MLXArray, values: MLXArray, tokenCount: Int
    ) -> (MLXArray, MLXArray) {
        let prev = offset
        var cacheLength = self.keys?.dim(2)

        if cacheLength == nil
            || (prev >= cacheLength! && cacheLength! < maxCacheSize)
        {
            let B = keys.dim(0)
            let nKVHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)
            let newSize = min(step, maxCacheSize - prev)

            let kShape = [B, nKVHeads, newSize, kHeadDim]
            let vShape = [B, nKVHeads, newSize, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if let currentKeys = self.keys, let currentValues = self.values {
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
            cacheLength = self.keys!.dim(2)
            idx = prev
        }

        let trimSize = cacheLength! - maxCacheSize
        if trimSize > 0 {
            self.keys = trim(trimSize: trimSize, self.keys!)
            self.values = trim(trimSize: trimSize, self.values!)
            idx = maxCacheSize
        }

        if idx == maxCacheSize {
            idx = keep
        }

        self.keys![.ellipsis, idx ..< (idx + tokenCount), 0...] = keys
        self.values![.ellipsis, idx ..< (idx + tokenCount), 0...] = values
        offset += tokenCount
        idx += tokenCount

        if offset < maxCacheSize {
            return (
                self.keys![.ellipsis, ..<offset, 0...],
                self.values![.ellipsis, ..<offset, 0...]
            )
        }
        return (self.keys!, self.values!)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let tokenCount = keys.dim(2)
        let result =
            if tokenCount == 1 {
                updateInPlace(keys: keys, values: values, tokenCount: tokenCount)
            } else {
                updateConcat(keys: keys, values: values)
            }
        return result
    }


    private var fusedRingContiguized = false

    public func fusedRingPrepare() -> (keys: MLXArray, values: MLXArray, writeIdx: Int)? {
        guard keep == 0, let currentKeys = keys, let currentValues = values,
            currentKeys.dim(2) == maxCacheSize,
            currentValues.dim(2) == maxCacheSize,
            offset >= maxCacheSize
        else { return nil }
        if !fusedRingContiguized {
            keys = contiguous(currentKeys)
            values = contiguous(currentValues)
            fusedRingContiguized = true
        }
        return (keys!, values!, idx == maxCacheSize ? keep : idx)
    }

    public func fusedRingAdvance() {
        if idx == maxCacheSize { idx = keep }
        offset += 1
        idx += 1
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = self.keys, let values = self.values else { return [] }
            if offset < keys.dim(2) {
                return [
                    keys[.ellipsis, ..<offset, 0...],
                    values[.ellipsis, ..<offset, 0...],
                ]
            } else {
                return [keys, values]
            }
        }
        set {
            guard newValue.count == 2 else {
                fatalError("RotatingKVCache state must have exactly 2 arrays")
            }
            self.keys = newValue[0]
            self.values = newValue[1]
        }
    }

    public override var metaState: [String] {
        get {
            return [String(keep), String(maxCacheSize), String(step), String(offset), String(idx)]
        }
        set {
            guard newValue.count == 5 else {
                fatalError("RotatingKVCache metaState must have exactly 5 values")
            }
            guard let keepVal = Int(newValue[0]),
                let stepVal = Int(newValue[2]),
                let offsetVal = Int(newValue[3]),
                let idxVal = Int(newValue[4])
            else {
                fatalError("Failed to convert metaState values to integers")
            }
            if newValue[1] == "None" {
                fatalError(
                    "RotatingKVCache requires a non-nil maxSize. Cannot load cache with maxSize=None."
                )
            }
            guard let maxSizeVal = Int(newValue[1]) else {
                fatalError("Failed to convert maxCacheSize '\(newValue[1])' to integer")
            }
            self.keep = keepVal
            self.maxCacheSize = maxSizeVal
            self.step = stepVal
            self.offset = offsetVal
            self.idx = idxVal
        }
    }

    public override var isTrimmable: Bool {
        return offset < maxCacheSize
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        idx -= trimmed
        return trimmed
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n > 1 {
            let actualWindowSize = windowSize ?? maxCacheSize
            let cappedOffset = min(maxCacheSize - 1, offset)

            if cappedOffset + n > actualWindowSize || returnArray {
                return .array(
                    createCausalMask(n: n, offset: cappedOffset, windowSize: actualWindowSize))
            }
            return .causal
        } else {
            guard let windowSize = windowSize else {
                return .none
            }

            if offset >= windowSize, maxCacheSize > windowSize {
                var currentIdx = idx
                if currentIdx >= maxCacheSize {
                    currentIdx = 0
                }

                let maskSize = offset < maxCacheSize ? offset + 1 : maxCacheSize
                let mask = MLXArray(0 ..< Int32(maskSize)) .>= Int32(maskSize - windowSize)

                let rolledMask = roll(mask, shift: currentIdx + 1)

                return .array(rolledMask)
            }
            return .none
        }
    }

    public var debugDescription: String {
        "\(String(describing: Self.self)) offset: \(offset), maxSize: \(maxCacheSize.description), keep: \(keep), idx: \(idx)"
    }

    public override func copy() -> any KVCache {
        let new = RotatingKVCache(maxSize: maxCacheSize, keep: keep, step: step)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    public func toQuantized(groupSize: Int = 64, bits: Int = 4) -> QuantizedKVCache {
        fatalError(
            "RotatingKVCache quantization not yet implemented - temporal ordering makes this complex"
        )

    }
}

private func resolvedKVQuantizationGroupSize(
    requested: Int,
    keyHeadDim: Int,
    valueHeadDim: Int
) -> Int? {
    let requested = max(1, requested)
    let compatible = [32, 64, 128].filter {
        keyHeadDim.isMultiple(of: $0) && valueHeadDim.isMultiple(of: $0)
    }
    guard !compatible.isEmpty else { return nil }
    return compatible.min { lhs, rhs in
        let lhsDistance = abs(lhs - requested)
        let rhsDistance = abs(rhs - requested)
        if lhsDistance == rhsDistance {
            return lhs < rhs
        }
        return lhsDistance < rhsDistance
    }
}

public class QuantizedKVCache: BaseKVCache, QuantizedKVCacheProtocol {
    private var keys: (MLXArray, MLXArray, MLXArray?)?
    private var values: (MLXArray, MLXArray, MLXArray?)?
    private let step: Int
    public private(set) var groupSize: Int
    public private(set) var bits: Int
    public let mode: QuantizationMode

    public init(groupSize: Int = 64, bits: Int = 8, mode: QuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.step = 256
        self.mode = mode
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        var arrays: [MLXArray] = []
        if let keys = keys {
            arrays.append(contentsOf: [keys.0, keys.1, keys.2].compactMap { $0 })
        }
        if let values = values {
            arrays.append(contentsOf: [values.0, values.1, values.2].compactMap { $0 })
        }
        return arrays
    }

    private func treeMap<T>(_ transform: (MLXArray) -> T, _ tuple: (MLXArray, MLXArray, MLXArray?))
        -> (T, T, T?)
    {
        if let biases = tuple.2 {
            return (transform(tuple.0), transform(tuple.1), transform(biases))

        } else {
            return (transform(tuple.0), transform(tuple.1), nil)
        }
    }

    private func treeMapPair<T>(
        _ transform: (MLXArray) -> T, _ tuple1: (MLXArray, MLXArray, MLXArray?),
        _ tuple2: (MLXArray, MLXArray, MLXArray?)
    ) -> ((T, T, T?), (T, T, T?)) {
        return (treeMap(transform, tuple1), treeMap(transform, tuple2))
    }

    private func initQuant(dim: Int, shape: [Int], dtype: DType) -> (MLXArray, MLXArray, MLXArray?)
    {
        let tempArray = MLXArray.zeros(shape + [dim], dtype: dtype)
        let quantized = quantized(tempArray, groupSize: groupSize, bits: bits)

        return (quantized.wq, quantized.scales, quantized.biases)
    }

    private func expandQuant(_ quantTuple: (MLXArray, MLXArray, MLXArray?), newShape: [Int]) -> (
        MLXArray, MLXArray, MLXArray?
    ) {
        return treeMap(
            { array in
                let newArray = MLXArray.zeros(newShape + [array.dim(-1)], dtype: array.dtype)
                return concatenated([array, newArray], axis: -2)
            }, quantTuple)
    }

    public func getQuantizedState() -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    )? {
        guard let keys = keys, let values = values else { return nil }

        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

        return (trimmedKeys, trimmedValues)
    }

    public func updateQuantized(keys: MLXArray, values: MLXArray) -> (
        (MLXArray, MLXArray, MLXArray?), (MLXArray, MLXArray, MLXArray?)
    ) {
        let B = keys.dim(0)
        let nKVHeads = keys.dim(1)
        let numSteps = keys.dim(2)
        let kHeadDim = keys.dim(3)
        let vHeadDim = values.dim(3)
        let prev = offset
        let effectiveGroupSize = resolvedKVQuantizationGroupSize(
            requested: groupSize,
            keyHeadDim: kHeadDim,
            valueHeadDim: vHeadDim
        )
        if let effectiveGroupSize,
            effectiveGroupSize != groupSize,
            self.keys == nil,
            self.values == nil,
            offset == 0
        {
            self.groupSize = effectiveGroupSize
        }
        guard effectiveGroupSize != nil else {
            fatalError(
                "KV cache quantization requires head dimensions divisible by one of the supported group sizes (32, 64, 128). Requested group size: \(groupSize). Key head dim: \(kHeadDim). Value head dim: \(vHeadDim)."
            )
        }

        if self.keys == nil || (prev + numSteps) > self.keys!.0.dim(-2) {
            let newSteps = ((step + numSteps - 1) / step) * step
            let shape = [B, nKVHeads, newSteps]

            if let existingKeys = self.keys, let existingValues = self.values {
                if prev % step != 0 {
                    let (trimmedKeys, trimmedValues) = treeMapPair(
                        { array in
                            array[.ellipsis, ..<prev, 0...]
                        }, existingKeys, existingValues)

                    self.keys = trimmedKeys
                    self.values = trimmedValues
                }

                self.keys = expandQuant(self.keys!, newShape: shape)
                self.values = expandQuant(self.values!, newShape: shape)
            } else {
                self.keys = initQuant(dim: kHeadDim, shape: shape, dtype: keys.dtype)
                self.values = initQuant(dim: vHeadDim, shape: shape, dtype: keys.dtype)
            }
        }

        offset += numSteps

        let quantizedKeys = quantized(keys, groupSize: groupSize, bits: bits)
        let quantizedValues = quantized(values, groupSize: groupSize, bits: bits)

        let qKeys = (quantizedKeys.wq, quantizedKeys.scales, quantizedKeys.biases)
        let qValues = (quantizedValues.wq, quantizedValues.scales, quantizedValues.biases)

        guard let currentKeys = self.keys, let currentValues = self.values else {
            fatalError("Quantized cache not properly initialized")
        }

        currentKeys.0[.ellipsis, prev ..< offset, 0...] = qKeys.0
        currentKeys.1[.ellipsis, prev ..< offset, 0...] = qKeys.1
        if let qKeysBiases = qKeys.2 {
            currentKeys.2![.ellipsis, prev ..< offset, 0...] = qKeysBiases
        }

        currentValues.0[.ellipsis, prev ..< offset, 0...] = qValues.0
        currentValues.1[.ellipsis, prev ..< offset, 0...] = qValues.1
        if let qValuesBiases = qValues.2 {
            currentValues.2![.ellipsis, prev ..< offset, 0...] = qValuesBiases
        }

        self.keys = currentKeys
        self.values = currentValues

        let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentKeys)
        let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, currentValues)

        return (trimmedKeys, trimmedValues)
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError(
            "`update` was called on `QuantizedKVCache`. Use `updateQuantized` instead."
        )
    }

    public override var state: [MLXArray] {
        get {
            guard let keys = keys, let values = values else { return [] }

            if offset < keys.0.dim(2) {
                let trimmedKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
                let trimmedValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)
                return [
                    trimmedKeys.0, trimmedKeys.1, trimmedKeys.2, trimmedValues.0, trimmedValues.1,
                    trimmedValues.2,
                ].compactMap { $0 }
            } else {
                return [keys.0, keys.1, keys.2, values.0, values.1, values.2].compactMap { $0 }
            }
        }
        set {
            switch newValue.count {
            case 4:
                keys = (newValue[0], newValue[1], nil)
                values = (newValue[2], newValue[3], nil)
            case 6:
                keys = (newValue[0], newValue[1], newValue[2])
                values = (newValue[3], newValue[4], newValue[5])
            default:
                fatalError(
                    "QuantizedKVCache state must have exactly 6 or 4 arrays (3/2 for keys, 3/2 for values)"
                )
            }
        }
    }

    public override var metaState: [String] {
        get { [String(step), String(offset), String(groupSize), String(bits)] }
        set {
            guard newValue.count == 4 else {
                fatalError("QuantizedKVCache metaState must have exactly 4 values")
            }
            guard
                let offset = Int(newValue[1]),
                let groupSize = Int(newValue[2]),
                let bits = Int(newValue[3])
            else {
                fatalError("Failed to convert QuantizedKVCache metaState values to integers")
            }

            self.offset = offset
            self.groupSize = groupSize
            self.bits = bits
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = QuantizedKVCache(groupSize: groupSize, bits: bits, mode: mode)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    public func toUnquantized() -> KVCacheSimple {
        let simpleCache = KVCacheSimple()
        simpleCache.offset = self.offset

        if let keys = keys, let values = values {
            let currentKeys = treeMap({ $0[.ellipsis, ..<offset, 0...] }, keys)
            let currentValues = treeMap({ $0[.ellipsis, ..<offset, 0...] }, values)

            let dequantizedKeys = dequantized(
                currentKeys.0, scales: currentKeys.1, biases: currentKeys.2,
                groupSize: groupSize, bits: bits, mode: mode)
            let dequantizedValues = dequantized(
                currentValues.0, scales: currentValues.1, biases: currentValues.2,
                groupSize: groupSize, bits: bits, mode: mode)

            simpleCache.state = [dequantizedKeys, dequantizedValues]
        }

        return simpleCache
    }
}

public class ChunkedKVCache: KVCacheSimple {
    private var chunkSize: Int?
    private var startPosition: Int = 0

    public init(chunkSize: Int? = nil) {
        self.chunkSize = chunkSize
        super.init()
    }

    public func maybeTrimFront() {
        guard let keys = self.keys,
            let chunkSize = chunkSize,
            keys.dim(2) >= chunkSize
        else { return }

        startPosition += keys.dim(2) - chunkSize
        self.keys = keys[.ellipsis, (-chunkSize)..., 0...]
        self.values = values?[.ellipsis, (-chunkSize)..., 0...]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let prev = offset - startPosition

        if self.keys == nil || (prev + keys.dim(2)) > self.keys!.dim(2) {
            let B = keys.dim(0)
            let kvHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)

            let nSteps = (step + keys.dim(2) - 1) / step
            let kShape = [B, kvHeads, nSteps * step, kHeadDim]
            let vShape = [B, kvHeads, nSteps * step, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentKeys = self.keys, var currentValues = self.values {
                if prev % step != 0 {
                    currentKeys = currentKeys[.ellipsis, ..<prev, 0...]
                    currentValues = currentValues[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentKeys, newK], axis: 2)
                self.values = concatenated([currentValues, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        offset += keys.dim(2)
        let end = offset - startPosition
        self.keys![.ellipsis, prev ..< end, 0...] = keys
        self.values![.ellipsis, prev ..< end, 0...] = values

        return (self.keys![.ellipsis, ..<end, 0...], self.values![.ellipsis, ..<end, 0...])
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(offset - startPosition, n)
        offset -= trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let new = ChunkedKVCache(chunkSize: chunkSize)
        new.step = self.step
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.metaState = self.metaState
        return new
    }

    public override var metaState: [String] {
        get {
            let chunkSizeStr = chunkSize?.description ?? "None"
            return [chunkSizeStr, String(startPosition)]
        }
        set {
            guard newValue.count == 2 else {
                fatalError("ChunkedKVCache metaState must have exactly 2 values")
            }
            if newValue[0] == "None" {
                self.chunkSize = nil
            } else {
                self.chunkSize = Int(newValue[0])
            }
            self.startPosition = Int(newValue[1]) ?? 0
        }
    }
}

public class ArraysCache: BaseKVCache {
    private var cache: [MLXArray?]
    internal var leftPadding: MLXArray?
    internal var lengths: MLXArray?

    public var rollbackState: (MLXArray, MLXArray)? = nil

    public init(size: Int, leftPadding: [Int]? = nil) {
        self.cache = Array(repeating: nil, count: size)
        self.leftPadding = leftPadding.map { MLXArray($0) }
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        cache.compactMap { $0 }
    }

    public subscript(index: Int) -> MLXArray? {
        get { cache[index] }
        set { cache[index] = newValue }
    }

    public override var state: [MLXArray] {
        get {
            return cache.compactMap { $0 }
        }
        set {
            cache = newValue.map { $0 as MLXArray? }
        }
    }

    public override func copy() -> any KVCache {
        let new = ArraysCache(size: cache.count)
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        return new
    }

    public func filter(batchIndices: MLXArray) {
        cache = cache.map { c in
            c?[batchIndices]
        }
        leftPadding = leftPadding.map { take($0, batchIndices, axis: 0) }
        lengths = lengths.map { take($0, batchIndices, axis: 0) }
    }

    public func extend(other: ArraysCache) {
        let lhsBatch = batchSize
        let rhsBatch = other.batchSize

        func concatenateOptional(_ lhs: MLXArray?, _ rhs: MLXArray?) -> MLXArray? {
            var shape: [Int]?
            var dtype: DType?
            if let lhs {
                shape = lhs.shape
                dtype = lhs.dtype
            }
            if let rhs {
                shape = rhs.shape
                dtype = rhs.dtype
            }
            guard let shape, let dtype else { return nil }

            let itemShape = Array(shape.dropFirst())
            let lhsValue = lhs ?? MLXArray.zeros([lhsBatch] + itemShape, dtype: dtype)
            let rhsValue = rhs ?? MLXArray.zeros([rhsBatch] + itemShape, dtype: dtype)
            return MLX.concatenated([lhsValue, rhsValue])
        }

        cache = zip(cache, other.cache).map { (c, o) in
            concatenateOptional(c, o)
        }
        leftPadding = concatenateOptional(leftPadding, other.leftPadding)
        lengths = concatenateOptional(lengths, other.lengths)
    }

    open func extract(_ idx: Int) -> ArraysCache {
        let extracted = ArraysCache(size: cache.count)
        extracted.cache = cache.map { $0?[idx ..< (idx + 1)] }
        return extracted
    }

    public func prepare(lengths: [Int]? = nil) {
        self.lengths = lengths.map { MLXArray($0.map { Int32($0) }) }
    }

    public func finalize() {
        lengths = nil
        leftPadding = nil
    }

    public func advance(_ n: Int) {
        lengths = lengths.map { $0 - Int32(n) }
        leftPadding = leftPadding.map { $0 - Int32(n) }
    }

    public func makeMask(N: Int) -> MLXArray? {
        if let leftPadding {
            return MLXArray(0 ..< N) .>= leftPadding[0..., .newAxis]
        } else if let lengths {
            return MLXArray(0 ..< N) .< lengths[0..., .newAxis]
        } else {
            return nil
        }
    }


    public override var metaState: [String] {
        get {
            var result = [
                "\(cache.count)",
                presentSlotIndices.map(String.init).joined(separator: ","),
            ]
            if let lp = leftPaddingValues {
                result.append(lp.map(String.init).joined(separator: ","))
            }
            return result
        }
        set {
            assertionFailure(
                "ArraysCache.metaState should not be set directly. Use restoreFromMetaState() instead"
            )
        }
    }

    internal func restoreFromMetaState(state: [MLXArray], savedMetaState: [String]) {
        if savedMetaState.count >= 2, let slotCount = Int(savedMetaState[0]) {
            let presentSlots =
                savedMetaState[1].isEmpty
                ? [] : savedMetaState[1].split(separator: ",").compactMap { Int($0) }
            let lp: [Int]? =
                savedMetaState.count >= 3
                ? savedMetaState[2].split(separator: ",").compactMap({ Int($0) }) : nil

            self.cache = Array(repeating: nil, count: slotCount)
            for (arrayIdx, slotIdx) in presentSlots.enumerated()
            where slotIdx < slotCount && arrayIdx < state.count {
                self.cache[slotIdx] = state[arrayIdx]
            }
            self.leftPadding = lp.map { MLXArray($0) }
        } else {
            self.cache = state.map { $0 as MLXArray? }
        }
    }

    internal var slotCount: Int { cache.count }

    internal var presentSlotIndices: [Int] {
        cache.enumerated().compactMap { (i, v) in v != nil ? i : nil }
    }

    internal var leftPaddingValues: [Int]? {
        guard let lp = leftPadding else { return nil }
        return lp.asArray(Int.self)
    }

    private var batchSize: Int {
        for item in cache {
            if let item {
                return item.dim(0)
            }
        }
        if let leftPadding {
            return leftPadding.dim(0)
        }
        if let lengths {
            return lengths.dim(0)
        }
        return 0
    }
}

public class MambaCache: ArraysCache {
    public init(leftPadding: [Int]? = nil) {
        super.init(size: 2, leftPadding: leftPadding)
    }

    public override func copy() -> any KVCache {
        let new = MambaCache()
        let s = self.state
        if !s.isEmpty {
            new.state = s.map { $0[.ellipsis] }
        }
        new.offset = self.offset
        new.leftPadding = self.leftPadding
        return new
    }

    public override func extract(_ idx: Int) -> ArraysCache {
        let extracted = MambaCache()
        extracted.state = state.map { $0[idx ..< (idx + 1)] }
        return extracted
    }
}

public class CacheList: BaseKVCache {
    private var caches: [KVCache]

    public init(_ caches: KVCache...) {
        self.caches = caches
        super.init()
    }

    internal init(caches: [KVCache]) {
        self.caches = caches
        super.init()
    }

    public override func innerState() -> [MLXArray] {
        caches.flatMap { $0.innerState() }
    }

    public subscript(index: Int) -> KVCache {
        return caches[index]
    }

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("CacheList should not use update(keys:values:) - use subscript access instead")
    }

    public override var state: [MLXArray] {
        get { caches.flatMap { $0.state } }
        set {
            let stateLengths = caches.map { $0.state.count }
            var start = 0
            for i in 0 ..< caches.count {
                let length = stateLengths[i]
                caches[i].state = Array(newValue[start ..< (start + length)])
                start += length
            }
        }
    }

    public override func copy() -> any KVCache {
        let copiedCaches = caches.map { $0.copy() }
        let new = CacheList(caches: copiedCaches)
        return new
    }

    public override var isTrimmable: Bool {
        caches.allSatisfy { $0.isTrimmable }
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        var result = 0
        for cache in caches {
            result = cache.trim(n)
        }
        return result
    }

    internal var children: [KVCache] { caches }


    public override var metaState: [String] {
        get {
            var result = ["\(caches.count)"]
            for cache in caches {
                let className = cacheClassName(cache)
                let meta = cache.metaState
                result.append(className)
                result.append("\(cache.state.count)")
                result.append("\(meta.count)")
                result.append(contentsOf: meta)
            }
            return result
        }
        set {
            assertionFailure(
                "CacheList.metaState should not be set directly. Use CacheList.fromState() instead")
        }
    }

    internal static func fromState(state: [MLXArray], metaState: [String]) throws -> CacheList {
        guard let childCount = metaState.first.flatMap({ Int($0) }) else {
            throw KVCacheError(message: "CacheList metaState missing child count")
        }

        var children: [KVCache] = []
        var metaIdx = 1
        var stateIdx = 0

        for _ in 0 ..< childCount {
            guard metaIdx + 2 < metaState.count else {
                throw KVCacheError(message: "CacheList metaState truncated")
            }
            let className = metaState[metaIdx]
            guard let stateCount = Int(metaState[metaIdx + 1]) else {
                throw KVCacheError(message: "CacheList: invalid stateCount for child")
            }
            guard let metaCount = Int(metaState[metaIdx + 2]) else {
                throw KVCacheError(message: "CacheList: invalid metaStateCount for child")
            }
            metaIdx += 3

            let childMeta = Array(metaState[metaIdx ..< min(metaIdx + metaCount, metaState.count)])
            metaIdx += metaCount

            let childState = Array(state[stateIdx ..< min(stateIdx + stateCount, state.count)])
            stateIdx += stateCount

            let child = try restoreCacheFromMetaState(
                className: className, state: childState, metaState: childMeta)
            children.append(child)
        }

        return CacheList(caches: children)
    }
}


struct KVCacheError: Error {
    let message: String
}


private func cacheClassName(_ cache: KVCache) -> String {
    switch cache {
    case is ChunkedKVCache: return "ChunkedKVCache"
    case is MambaCache: return "MambaCache"
    case is ArraysCache: return "ArraysCache"
    case is RotatingKVCache: return "RotatingKVCache"
    case is QuantizedKVCache: return "QuantizedKVCache"
    case is KVCacheSimple: return "KVCache"
    case is CacheList: return "CacheList"
    default: return "KVCache"
    }
}

public func savePromptCache(
    url: URL,
    cache: [KVCache],
    metadata: [String: String] = [:]
) throws {
    let cacheData = cache.map { $0.state }
    let cacheInfo = cache.map { $0.metaState }
    let cacheClasses = cache.map { cacheClassName($0) }

    var flattenedData: [String: MLXArray] = [:]
    for (i, arrays) in cacheData.enumerated() {
        for (j, array) in arrays.enumerated() {
            flattenedData["\(i).\(j)"] = array
        }
    }

    var flattenedMetadata: [String: String] = [:]

    for (i, info) in cacheInfo.enumerated() {
        for (j, metaValue) in info.enumerated() {
            flattenedMetadata["0.\(i).\(j)"] = metaValue
        }
    }

    for (key, value) in metadata {
        flattenedMetadata["1.\(key)"] = value
    }

    for (i, className) in cacheClasses.enumerated() {
        flattenedMetadata["2.\(i)"] = className
    }

    try save(arrays: flattenedData, metadata: flattenedMetadata, url: url)
}

public func loadPromptCache(
    url: URL
) throws -> ([KVCache], [String: String]) {
    let (arrays, metadata) = try loadArraysAndMetadata(url: url)

    let cacheData = unflattenArrays(arrays)

    let unflattenedMetadata = unflattenMetadata(metadata)

    guard unflattenedMetadata.count >= 3 else {
        throw KVCacheError(message: "Invalid cache metadata format")
    }

    let cacheInfo = unflattenedMetadata[0] as? [[String]] ?? []
    let userMetadata = unflattenedMetadata[1] as? [String: String] ?? [:]
    let cacheClasses = unflattenedMetadata[2] as? [String] ?? []

    guard cacheData.count == cacheInfo.count && cacheData.count == cacheClasses.count else {
        throw KVCacheError(message: "Mismatch in cache counts")
    }

    var caches: [KVCache] = []
    for i in 0 ..< cacheData.count {
        let className = cacheClasses[i]
        let info = i < cacheInfo.count ? cacheInfo[i] : []

        let cache = try restoreCacheFromMetaState(
            className: className, state: cacheData[i], metaState: info)
        caches.append(cache)
    }

    return (caches, userMetadata)
}

private func restoreCacheFromMetaState(
    className: String,
    state: [MLXArray],
    metaState: [String]
) throws -> KVCache {
    switch className {
    case "KVCache", "KVCacheSimple":
        let cache = KVCacheSimple()
        cache.state = state
        cache.metaState = metaState
        return cache

    case "RotatingKVCache":
        guard metaState.count >= 5 else {
            throw KVCacheError(
                message: "Invalid RotatingKVCache metaState - expected 5 values")
        }
        if metaState[1] == "None" {
            throw KVCacheError(
                message:
                    "RotatingKVCache with maxSize=None is not supported.")
        }
        guard let maxSize = Int(metaState[1]) else {
            throw KVCacheError(
                message: "Failed to parse RotatingKVCache maxSize from: \(metaState[1])")
        }
        let cache = RotatingKVCache(maxSize: maxSize)
        cache.state = state
        cache.metaState = metaState
        return cache

    case "QuantizedKVCache":
        let cache = QuantizedKVCache()
        cache.state = state
        cache.metaState = metaState
        return cache

    case "ChunkedKVCache":
        let cache = ChunkedKVCache()
        cache.state = state
        cache.metaState = metaState
        return cache

    case "MambaCache":
        let cache = MambaCache()
        cache.restoreFromMetaState(state: state, savedMetaState: metaState)
        return cache

    case "ArraysCache":
        let cache = ArraysCache(size: 0)
        cache.restoreFromMetaState(state: state, savedMetaState: metaState)
        return cache

    case "CacheList":
        return try CacheList.fromState(state: state, metaState: metaState)

    default:
        throw KVCacheError(message: "Unknown cache class: \(className)")
    }
}

private func unflattenArrays(_ flatArrays: [String: MLXArray]) -> [[MLXArray]] {
    var arrayMap: [Int: [Int: MLXArray]] = [:]

    for (key, array) in flatArrays {
        let components = key.split(separator: ".")
        if components.count >= 2,
            let i = Int(components[0]),
            let j = Int(components[1])
        {
            if arrayMap[i] == nil {
                arrayMap[i] = [:]
            }
            arrayMap[i]![j] = array
        }
    }

    var result: [[MLXArray]] = []
    let maxI = arrayMap.keys.max() ?? -1

    for i in 0 ... maxI {
        if let innerMap = arrayMap[i] {
            let maxJ = innerMap.keys.max() ?? -1
            var innerArray: [MLXArray] = []
            for j in 0 ... maxJ {
                if let array = innerMap[j] {
                    innerArray.append(array)
                }
            }
            result.append(innerArray)
        } else {
            result.append([])
        }
    }

    return result
}

private func unflattenMetadata(_ flatMetadata: [String: String]) -> [Any] {
    var cacheInfo: [[String]] = []
    var userMetadata: [String: String] = [:]
    var cacheClasses: [String] = []

    for (key, value) in flatMetadata {
        let components = key.split(separator: ".")

        if components.count >= 3 && components[0] == "0" {
            if let i = Int(components[1]), let j = Int(components[2]) {
                while cacheInfo.count <= i {
                    cacheInfo.append([])
                }
                while cacheInfo[i].count <= j {
                    cacheInfo[i].append("")
                }
                cacheInfo[i][j] = value
            }
        } else if components.count >= 2 && components[0] == "1" {
            let metaKey = components.dropFirst().joined(separator: ".")
            userMetadata[metaKey] = value
        } else if components.count >= 2 && components[0] == "2" {
            if let i = Int(components[1]) {
                while cacheClasses.count <= i {
                    cacheClasses.append("")
                }
                cacheClasses[i] = value
            }
        }
    }

    return [cacheInfo, userMetadata, cacheClasses]
}

public func makePromptCache(
    model: any LanguageModel,
    parameters: GenerateParameters? = nil
) -> [KVCache] {
    return model.newCache(parameters: parameters)
}

public func makePromptCache(
    model: any LanguageModel,
    maxKVSize: Int? = nil
) -> [KVCache] {
    let parameters = maxKVSize.map { GenerateParameters(maxKVSize: $0) }
    return makePromptCache(model: model, parameters: parameters)
}

public func makePromptCacheWithLayerCount(
    numLayers: Int,
    maxKVSize: Int? = nil
) -> [KVCache] {
    if let maxKVSize = maxKVSize {
        return (0 ..< numLayers).map { _ in
            RotatingKVCache(maxSize: maxKVSize, keep: 4)
        }
    } else {
        return (0 ..< numLayers).map { _ in KVCacheSimple() }
    }
}

public func canTrimPromptCache(_ cache: [KVCache]) -> Bool {
    return cache.allSatisfy { $0.isTrimmable }
}

@discardableResult
public func trimPromptCache(_ cache: [KVCache], numTokens: Int) -> Int {
    guard canTrimPromptCache(cache), !cache.isEmpty else { return 0 }
    cache.dropFirst().forEach { $0.trim(numTokens) }
    return cache.first?.trim(numTokens) ?? 0
}


public typealias StandardKVCache = KVCacheSimple



private let compiledQuantizedAttentionSoftmax: @Sendable (MLXArray) -> MLXArray = compile(
    shapeless: true
) { scores in
    let rowMax = scores.max(axis: -1, keepDims: true)
    let validRow = rowMax .> MLXArray(-Float.greatestFiniteMagnitude / 2)
    let expScores = MLX.where(validRow, exp(scores - rowMax), MLXArray(Float(0)))
    let denom = expScores.sum(axis: -1, keepDims: true)
    return MLX.where(denom .> MLXArray(Float(0)), expScores / denom, MLXArray(Float(0)))
}

private let compiledQuantizedAttentionSoftmaxWithSink: @Sendable (MLXArray, MLXArray) -> MLXArray =
    compile(shapeless: true) { scores, sinkLogits in
        let rowMax = scores.max(axis: -1, keepDims: true)
        let m = maximum(rowMax, sinkLogits)
        let expScores = exp(scores - m)
        let sinkExp = exp(sinkLogits - m)
        let denom = expScores.sum(axis: -1, keepDims: true) + sinkExp
        return expScores / denom
    }

public func quantizedScaledDotProductAttention(
    queries: MLXArray,
    quantizedKeys: (MLXArray, MLXArray, MLXArray?),
    quantizedValues: (MLXArray, MLXArray, MLXArray?),
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    groupSize: Int = 64,
    bits: Int = 8,
    mode: QuantizationMode = .affine,
    sinks: MLXArray? = nil
) -> MLXArray {

    let (B, nQHeads, L, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
    let nKVHeads = quantizedKeys.0.dim(-3)
    let nRepeats = nQHeads / nKVHeads

    var scaledQueries = queries * scale

    var qKeys = quantizedKeys
    var qValues = quantizedValues
    if nRepeats > 1 {
        scaledQueries = scaledQueries.reshaped([B, nKVHeads, nRepeats, L, D])
        qKeys = (
            expandedDimensions(qKeys.0, axis: -3),
            expandedDimensions(qKeys.1, axis: -3),
            qKeys.2 == nil ? nil : expandedDimensions(qKeys.2!, axis: -3)
        )
        qValues = (
            expandedDimensions(qValues.0, axis: -3),
            expandedDimensions(qValues.1, axis: -3),
            qValues.2 == nil ? nil : expandedDimensions(qValues.2!, axis: -3)
        )
    }

    let rawScores = quantizedMM(
        scaledQueries, qKeys.0, scales: qKeys.1, biases: qKeys.2,
        transpose: true, groupSize: groupSize, bits: bits,
        mode: mode
    )
    let kL = rawScores.dim(-1)

    var scores = nRepeats > 1 ? rawScores.reshaped([B, nQHeads, L, kL]) : rawScores

    switch mask {
    case .causal:
        let (qL, kLen) = (scores.dim(-2), scores.dim(-1))
        let qIndices = MLXArray(0 ..< qL) + MLXArray(kLen - qL)
        let kIndices = MLXArray(0 ..< kLen)
        let causalMask = greaterEqual(
            expandedDimensions(qIndices, axis: -1), expandedDimensions(kIndices, axis: -2))
        scores = MLX.where(causalMask, scores, MLXArray(-Float.greatestFiniteMagnitude))

    case .array(let maskArray):
        if maskArray.dtype == .bool {
            scores = MLX.where(maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
        } else {
            scores = scores + maskArray
        }

    case .arrays(let maskArrays):
        if let maskArray = maskArrays.first {
            if maskArray.dtype == .bool {
                scores = MLX.where(maskArray, scores, MLXArray(-Float.greatestFiniteMagnitude))
            } else {
                scores = scores + maskArray
            }
        }

    case .none:
        break
    }

    let attentionWeights4D: MLXArray
    if let sinks {
        let sinkLogits = sinks.reshaped([1, nQHeads, 1, 1])
        attentionWeights4D = compiledQuantizedAttentionSoftmaxWithSink(scores, sinkLogits)
    } else {
        attentionWeights4D = compiledQuantizedAttentionSoftmax(scores)
    }

    let attentionWeights =
        nRepeats > 1
        ? attentionWeights4D.reshaped([B, nKVHeads, nRepeats, L, kL])
        : attentionWeights4D

    var output = quantizedMM(
        attentionWeights, qValues.0, scales: qValues.1, biases: qValues.2,
        transpose: false, groupSize: groupSize, bits: bits,
        mode: mode
    )

    if nRepeats > 1 {
        output = output.reshaped([B, nQHeads, L, D])
    }

    return output
}


public func maybeQuantizeKVCache(
    cache: inout [KVCache],
    kvBits: Int?,
    kvGroupSize: Int = 64,
    quantizedKVStart: Int = 0
) {
    guard let kvBits = kvBits, !cache.isEmpty else { return }

    guard let firstQuantizable = cache.first(where: { $0 is KVCacheSimple }),
        !(firstQuantizable is QuantizedKVCache),
        firstQuantizable.offset > quantizedKVStart
    else {
        return
    }

    for i in 0 ..< cache.count {
        if let simpleCache = cache[i] as? KVCacheSimple {
            let state = simpleCache.state
            if state.count == 2 {
                let keyHeadDim = state[0].dim(3)
                let valueHeadDim = state[1].dim(3)
                guard
                    resolvedKVQuantizationGroupSize(
                        requested: kvGroupSize,
                        keyHeadDim: keyHeadDim,
                        valueHeadDim: valueHeadDim
                    ) != nil
                else {
                    continue
                }
            }
            cache[i] = simpleCache.toQuantized(groupSize: kvGroupSize, bits: kvBits)
        }
    }
}

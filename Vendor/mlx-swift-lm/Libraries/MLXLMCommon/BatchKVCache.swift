
import Foundation
import MLX

public protocol BatchedCache: KVCache {
    func filterBatched(batchIndices: MLXArray)

    func extendBatched(_ other: any BatchedCache)

    func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?)

    func finalizeBatched()

    func extractBatched(_ idx: Int) -> any KVCache

    func advanceBatched(_ n: Int)
}

public class BatchKVCache: BaseKVCache, BatchPositionedKVCache, BatchedCache {

    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices)
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchKVCache else {
            preconditionFailure("BatchKVCache.extendBatched requires another BatchKVCache")
        }
        extend(other)
    }

    public func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?) {
        prepare(leftPadding: leftPadding, lengths: lengths, rightPadding: rightPadding)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx)
    }

    public func advanceBatched(_: Int) {}

    public static let allocationStep = 256

    public internal(set) var keys: MLXArray?

    public internal(set) var values: MLXArray?

    public internal(set) var batchOffset: MLXArray

    public internal(set) var leftPadding: MLXArray

    var _idx: Int = 0

    var _rightPadding: MLXArray?

    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }

    public override var maxSize: Int? { nil }
    public override var isTrimmable: Bool { true }


    public init(leftPadding: [Int]) {
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffset = MLXArray(leftPadding.map { Int32(-$0) })
        super.init()
    }

    private init(
        keys: MLXArray?,
        values: MLXArray?,
        offset: MLXArray,
        leftPadding: MLXArray,
        idx: Int
    ) {
        self.keys = keys
        self.values = values
        self.batchOffset = offset
        self.leftPadding = leftPadding
        super.init()
        self._idx = idx
    }

    public override func update(
        keys: MLXArray, values: MLXArray
    ) -> (MLXArray, MLXArray) {
        let prev = _idx
        let stepCount = keys.dim(2)
        let needGrow: Bool = {
            guard let storedKeys = self.keys else { return true }
            return (prev + stepCount) > storedKeys.dim(2)
        }()

        if needGrow {
            let B = keys.dim(0)
            let nKVHeads = keys.dim(1)
            let kHeadDim = keys.dim(3)
            let vHeadDim = values.dim(3)
            let nSteps = (Self.allocationStep + stepCount - 1) / Self.allocationStep
            let kShape = [B, nKVHeads, nSteps * Self.allocationStep, kHeadDim]
            let vShape = [B, nKVHeads, nSteps * Self.allocationStep, vHeadDim]
            let newK = MLXArray.zeros(kShape, dtype: keys.dtype)
            let newV = MLXArray.zeros(vShape, dtype: values.dtype)

            if var currentK = self.keys, var currentV = self.values {
                if prev % Self.allocationStep != 0 {
                    currentK = currentK[.ellipsis, ..<prev, 0...]
                    currentV = currentV[.ellipsis, ..<prev, 0...]
                }
                self.keys = concatenated([currentK, newK], axis: 2)
                self.values = concatenated([currentV, newV], axis: 2)
            } else {
                self.keys = newK
                self.values = newV
            }
        }

        batchOffset = batchOffset + Int32(stepCount)
        _idx += stepCount
        self.keys?[.ellipsis, prev ..< _idx, 0...] = keys
        self.values?[.ellipsis, prev ..< _idx, 0...] = values

        if stepCount == 1 {
            asyncEval(batchOffset)
        }

        return (
            self.keys![.ellipsis, ..<_idx, 0...],
            self.values![.ellipsis, ..<_idx, 0...]
        )
    }


    public func prepare(
        leftPadding additionalLeftPadding: [Int]? = nil,
        lengths _: [Int]? = nil,
        rightPadding: [Int]? = nil
    ) {
        if let additionalLeftPadding {
            precondition(
                keys == nil,
                "prepare() with leftPadding can only be called on an empty BatchKVCache"
            )
            let additional = MLXArray(additionalLeftPadding.map { Int32($0) })
            leftPadding = leftPadding + additional
            batchOffset = batchOffset - additional
        }

        if let rightPadding, rightPadding.contains(where: { $0 > 0 }) {
            _rightPadding = MLXArray(rightPadding.map { Int32($0) })
        }
    }

    public func finalize() {
        guard let pending = _rightPadding else { return }
        guard let storedK = keys, let storedV = values else {
            _rightPadding = nil
            return
        }

        let shifts = pending[0..., .newAxis]
        keys = dynamicRoll(storedK, shifts: shifts, axis: 2)
        values = dynamicRoll(storedV, shifts: shifts, axis: 2)
        batchOffset = batchOffset - pending
        leftPadding = leftPadding + pending
        _rightPadding = nil
    }

    public func filter(batchIndices: MLXArray) {
        if keys != nil {
            keys = take(keys!, batchIndices, axis: 0)
            values = take(values!, batchIndices, axis: 0)
        }
        batchOffset = take(batchOffset, batchIndices, axis: 0)
        leftPadding = take(leftPadding, batchIndices, axis: 0)

        let minLeftPad = leftPadding.min().item(Int32.self)
        if minLeftPad > 0 {
            let minPadInt = Int(minLeftPad)
            if let storedK = keys, let storedV = values {
                keys = storedK[.ellipsis, minPadInt..., 0...]
                values = storedV[.ellipsis, minPadInt..., 0...]
            }
            _idx -= minPadInt
            leftPadding = leftPadding - Int32(minPadInt)
        }
    }


    public func extend(_ other: BatchKVCache) {
        if keys == nil && other.keys == nil {
            leftPadding = concatenated([leftPadding, other.leftPadding], axis: 0)
            batchOffset = concatenated([batchOffset, other.batchOffset], axis: 0)
            return
        }

        let maxIdx = max(_idx, other._idx)
        var L1 = 0
        var L2 = 0
        var H = 0
        var Dk = 0
        var Dv = 0

        if let storedK = keys {
            L1 = storedK.dim(2)
            H = storedK.dim(1)
            Dk = storedK.dim(3)
            Dv = values!.dim(3)
        }
        if let storedK = other.keys {
            L2 = storedK.dim(2)
            H = storedK.dim(1)
            Dk = storedK.dim(3)
            Dv = other.values!.dim(3)
        }
        let maxSize = max(L1, L2)

        func pad(
            _ cacheKeys: MLXArray?,
            _ cacheValues: MLXArray?,
            cacheIdx: Int,
            cacheOffset: MLXArray,
            cacheLeftPadding: MLXArray
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            var k: MLXArray
            var v: MLXArray
            if let cacheKeys, let cacheValues {
                k = cacheKeys
                v = cacheValues
            } else {
                let Bc = cacheOffset.dim(0)
                k = MLXArray.zeros([Bc, H, 0, Dk], dtype: keys?.dtype ?? .float32)
                v = MLXArray.zeros([Bc, H, 0, Dv], dtype: values?.dtype ?? .float32)
            }

            let leftPad = maxIdx - cacheIdx
            var rightPad = maxSize - k.dim(2) - leftPad
            if rightPad < 0 {
                k = k[.ellipsis, ..<(k.dim(2) + rightPad), 0...]
                v = v[.ellipsis, ..<(v.dim(2) + rightPad), 0...]
                rightPad = 0
            }
            if leftPad != 0 || rightPad != 0 {
                let widths: [IntOrPair] = [
                    .init(0), .init(0), .init((leftPad, rightPad)), .init(0),
                ]
                k = padded(k, widths: widths)
                v = padded(v, widths: widths)
            }
            return (k, v, cacheOffset, cacheLeftPadding + Int32(leftPad))
        }

        let (k1, v1, o1, lp1) = pad(
            keys, values, cacheIdx: _idx,
            cacheOffset: batchOffset, cacheLeftPadding: leftPadding
        )
        let (k2, v2, o2, lp2) = pad(
            other.keys, other.values, cacheIdx: other._idx,
            cacheOffset: other.batchOffset, cacheLeftPadding: other.leftPadding
        )

        keys = concatenated([k1, k2], axis: 0)
        values = concatenated([v1, v2], axis: 0)
        batchOffset = concatenated([o1, o2], axis: 0)
        leftPadding = concatenated([lp1, lp2], axis: 0)
        _idx = maxIdx
    }

    public func extract(_ idx: Int) -> KVCacheSimple {
        let cache = KVCacheSimple()
        guard let storedK = keys, let storedV = values else {
            return cache
        }
        let leftPad = Int(leftPadding[idx].item(Int32.self))
        let kSlice = storedK[idx ..< (idx + 1), 0..., leftPad ..< _idx, 0...]
        let vSlice = storedV[idx ..< (idx + 1), 0..., leftPad ..< _idx, 0...]
        eval(kSlice, vSlice)
        cache.state = [kSlice, vSlice]
        return cache
    }

    public static func merge(_ caches: [KVCacheSimple]) -> BatchKVCache {
        let lengths = caches.map { $0.offset }
        let maxLength = lengths.max() ?? 0

        if maxLength == 0 {
            return BatchKVCache(leftPadding: Array(repeating: 0, count: caches.count))
        }

        let leftPaddings = lengths.map { maxLength - $0 }
        let B = caches.count

        guard let template = caches.first(where: { $0.state.count >= 2 }) else {
            return BatchKVCache(leftPadding: leftPaddings)
        }
        let templateK = template.state[0]
        let templateV = template.state[1]
        let H = templateK.dim(1)
        let Dk = templateK.dim(3)
        let Dv = templateV.dim(3)
        let dtype = templateK.dtype

        let keys = MLXArray.zeros([B, H, maxLength, Dk], dtype: dtype)
        let values = MLXArray.zeros([B, H, maxLength, Dv], dtype: dtype)

        for (i, (pad, cache)) in zip(leftPaddings, caches).enumerated() {
            guard cache.state.count >= 2 else { continue }
            let k = cache.state[0]
            let v = cache.state[1]
            let len = cache.offset
            keys[i ..< (i + 1), 0..., pad ..< (pad + len), 0...] = k[.ellipsis, ..<len, 0...]
            values[i ..< (i + 1), 0..., pad ..< (pad + len), 0...] = v[.ellipsis, ..<len, 0...]
        }

        let result = BatchKVCache(leftPadding: leftPaddings)
        result.keys = keys
        result.values = values
        result.batchOffset = result.batchOffset + Int32(maxLength)
        result._idx = maxLength
        return result
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        if n == 1, leftPadding.max().item(Int32.self) <= 0 {
            return .none
        }
        let mask = createCausalMask(
            n: n,
            offset: _idx,
            windowSize: windowSize,
            leftPadding: leftPadding
        )
        return .array(mask)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(_idx, n)
        _idx -= trimmed
        batchOffset = batchOffset - Int32(trimmed)
        return trimmed
    }

    public func size() -> Int { _idx }
    public func isEmpty() -> Bool { keys == nil }

    public override var state: [MLXArray] {
        get {
            guard let storedK = keys, let storedV = values else {
                return [batchOffset, leftPadding]
            }
            let kClipped: MLXArray
            let vClipped: MLXArray
            if _idx < storedK.dim(2) {
                kClipped = storedK[.ellipsis, ..<_idx, 0...]
                vClipped = storedV[.ellipsis, ..<_idx, 0...]
            } else {
                kClipped = storedK
                vClipped = storedV
            }
            return [kClipped, vClipped, batchOffset, leftPadding]
        }
        set {
            if newValue.count >= 4 {
                keys = newValue[0]
                values = newValue[1]
                batchOffset = newValue[2]
                leftPadding = newValue[3]
                _idx = newValue[0].dim(2)
            } else if newValue.count == 2 {
                batchOffset = newValue[0]
                leftPadding = newValue[1]
                keys = nil
                values = nil
                _idx = 0
            } else {
                fatalError("BatchKVCache.state setter expects 2 or 4 arrays")
            }
        }
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func copy() -> any KVCache {
        BatchKVCache(
            keys: keys.map { $0[.ellipsis] },
            values: values.map { $0[.ellipsis] },
            offset: batchOffset[0...],
            leftPadding: leftPadding[0...],
            idx: _idx
        )
    }
}

public class BatchRotatingKVCache: BaseKVCache, BatchPositionedKVCache, BatchedCache {
    public static let allocationStep = 256

    public internal(set) var keys: MLXArray?
    public internal(set) var values: MLXArray?
    public internal(set) var batchOffset: MLXArray
    public internal(set) var leftPadding: MLXArray

    let maxCacheSize: Int
    var _idx: Int = 0
    var _rightPadding: MLXArray?

    var _base: Int = 0

    public static let fastDecodeEnabledDefault: Bool = {
        if let raw = ProcessInfo.processInfo.environment["DARKBLOOM_FAST_BATCH_ROTATING_KV"] {
            return !["0", "false", "no", "off"].contains(raw.lowercased())
        }
        return true
    }()

    public var useFastDecodePath: Bool = BatchRotatingKVCache.fastDecodeEnabledDefault

    var windowKeys: MLXArray? {
        guard let k = keys else { return nil }
        if _base == 0, k.dim(2) == _idx { return k }
        return k[.ellipsis, _base ..< (_base + _idx), 0...]
    }
    var windowValues: MLXArray? {
        guard let v = values else { return nil }
        if _base == 0, v.dim(2) == _idx { return v }
        return v[.ellipsis, _base ..< (_base + _idx), 0...]
    }

    func normalizeToWindow() {
        guard let k = keys, let v = values else {
            _base = 0
            return
        }
        if _base != 0 || k.dim(2) != _idx {
            keys = k[.ellipsis, _base ..< (_base + _idx), 0...]
            values = v[.ellipsis, _base ..< (_base + _idx), 0...]
            _base = 0
        }
    }

    public init(maxSize: Int, leftPadding: [Int]) {
        self.maxCacheSize = maxSize
        self.leftPadding = MLXArray(leftPadding.map { Int32($0) })
        self.batchOffset = MLXArray(leftPadding.map { Int32(-$0) })
        super.init()
    }

    private init(
        maxSize: Int,
        keys: MLXArray?,
        values: MLXArray?,
        offset: MLXArray,
        leftPadding: MLXArray,
        idx: Int
    ) {
        self.maxCacheSize = maxSize
        self.keys = keys
        self.values = values
        self.batchOffset = offset
        self.leftPadding = leftPadding
        self._idx = idx
        super.init()
    }

    public override var maxSize: Int? { maxCacheSize }
    public override var isTrimmable: Bool { _idx < maxCacheSize }

    public override var offset: Int {
        get { _idx }
        set { _idx = newValue }
    }

    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices)
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? BatchRotatingKVCache else {
            preconditionFailure(
                "BatchRotatingKVCache.extendBatched requires another BatchRotatingKVCache")
        }
        extend(other)
    }

    public func prepareBatched(leftPadding: [Int]?, lengths: [Int]?, rightPadding: [Int]?) {
        prepare(leftPadding: leftPadding, lengths: lengths, rightPadding: rightPadding)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx)
    }

    public func advanceBatched(_: Int) {}

    public override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let stepCount = keys.dim(2)

        if useFastDecodePath, stepCount == 1, self.keys != nil, _rightPadding == nil,
            _idx <= maxCacheSize
        {
            return fastDecodeUpdate(keys: keys, values: values)
        }

        normalizeToWindow()

        if self.keys == nil {
            self.keys = keys
            self.values = values
            _idx = stepCount
        } else {
            if let storedK = self.keys, storedK.dim(2) > _idx {
                self.keys = storedK[.ellipsis, ..<_idx, 0...]
                self.values = self.values![.ellipsis, ..<_idx, 0...]
            }

            if stepCount > 1 {
                let trimSize = _idx - maxCacheSize + 1
                if trimSize > 0 {
                    self.keys = self.keys![.ellipsis, trimSize..., 0...]
                    self.values = self.values![.ellipsis, trimSize..., 0...]
                    leftPadding = leftPadding - Int32(trimSize)
                    _idx -= trimSize
                }
            }

            self.keys = concatenated([self.keys!, keys], axis: 2)
            self.values = concatenated([self.values!, values], axis: 2)
            _idx += stepCount
        }

        batchOffset = batchOffset + Int32(stepCount)

        if stepCount == 1, _idx > maxCacheSize {
            let trimSize = _idx - maxCacheSize
            self.keys = self.keys![.ellipsis, trimSize..., 0...]
            self.values = self.values![.ellipsis, trimSize..., 0...]
            leftPadding = leftPadding - Int32(trimSize)
            _idx = maxCacheSize
        }

        if stepCount == 1 {
            asyncEval(batchOffset, leftPadding)
        }

        return (self.keys!, self.values!)
    }

    private func fastDecodeUpdate(keys newKeys: MLXArray, values newValues: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let cap = maxCacheSize + Self.allocationStep
        var writePos = _base + _idx

        if writePos >= (keys?.dim(2) ?? 0) {
            let B = keys!.dim(0)
            let H = keys!.dim(1)
            let Dk = keys!.dim(3)
            let Dv = values!.dim(3)
            let win = windowKeys!
            let winV = windowValues!
            let newK = MLXArray.zeros([B, H, cap, Dk], dtype: keys!.dtype)
            let newV = MLXArray.zeros([B, H, cap, Dv], dtype: values!.dtype)
            if _idx > 0 {
                newK[.ellipsis, 0 ..< _idx, 0...] = win
                newV[.ellipsis, 0 ..< _idx, 0...] = winV
            }
            keys = newK
            values = newV
            _base = 0
            writePos = _idx
        }

        keys![.ellipsis, writePos ..< (writePos + 1), 0...] = newKeys
        values![.ellipsis, writePos ..< (writePos + 1), 0...] = newValues
        _idx += 1
        batchOffset = batchOffset + Int32(1)

        if _idx > maxCacheSize {
            let drop = _idx - maxCacheSize
            _base += drop
            leftPadding = leftPadding - Int32(drop)
            _idx = maxCacheSize
        }

        asyncEval(batchOffset, leftPadding)

        return (windowKeys!, windowValues!)
    }

    public func prepare(
        leftPadding additionalLeftPadding: [Int]? = nil,
        lengths _: [Int]? = nil,
        rightPadding: [Int]? = nil
    ) {
        if let additionalLeftPadding {
            precondition(
                keys == nil,
                "prepare() with leftPadding can only be called on an empty BatchRotatingKVCache"
            )
            let additional = MLXArray(additionalLeftPadding.map { Int32($0) })
            leftPadding = leftPadding + additional
            batchOffset = batchOffset - additional
        }

        if let rightPadding, rightPadding.contains(where: { $0 > 0 }) {
            _rightPadding = MLXArray(rightPadding.map { Int32($0) })
        }
    }

    public func finalize() {
        guard let pending = _rightPadding else { return }
        normalizeToWindow()
        guard let storedK = keys, let storedV = values else {
            _rightPadding = nil
            return
        }

        let shifts = pending[0..., .newAxis]
        keys = dynamicRoll(storedK, shifts: shifts, axis: 2)
        values = dynamicRoll(storedV, shifts: shifts, axis: 2)
        batchOffset = batchOffset - pending
        leftPadding = leftPadding + pending
        _rightPadding = nil
    }

    public func filter(batchIndices: MLXArray) {
        normalizeToWindow()
        if keys != nil {
            keys = take(keys!, batchIndices, axis: 0)
            values = take(values!, batchIndices, axis: 0)
        }
        batchOffset = take(batchOffset, batchIndices, axis: 0)
        leftPadding = take(leftPadding, batchIndices, axis: 0)
    }

    public func extend(_ other: BatchRotatingKVCache) {
        precondition(
            maxCacheSize == other.maxCacheSize,
            "BatchRotatingKVCache can only extend caches with the same maximum size"
        )
        normalizeToWindow()
        other.normalizeToWindow()

        if keys == nil && other.keys == nil {
            leftPadding = concatenated([leftPadding, other.leftPadding], axis: 0)
            batchOffset = concatenated([batchOffset, other.batchOffset], axis: 0)
            return
        }

        let maxIdx = max(_idx, other._idx)
        var L1 = 0
        var L2 = 0
        var H = 0
        var Dk = 0
        var Dv = 0

        if let storedK = keys {
            L1 = storedK.dim(2)
            H = storedK.dim(1)
            Dk = storedK.dim(3)
            Dv = values!.dim(3)
        }
        if let storedK = other.keys {
            L2 = storedK.dim(2)
            H = storedK.dim(1)
            Dk = storedK.dim(3)
            Dv = other.values!.dim(3)
        }
        let maxSize = max(L1, L2)

        func pad(
            _ cacheKeys: MLXArray?,
            _ cacheValues: MLXArray?,
            cacheIdx: Int,
            cacheOffset: MLXArray,
            cacheLeftPadding: MLXArray
        ) -> (MLXArray, MLXArray, MLXArray, MLXArray) {
            var k: MLXArray
            var v: MLXArray
            if let cacheKeys, let cacheValues {
                k = cacheKeys
                v = cacheValues
            } else {
                let Bc = cacheOffset.dim(0)
                k = MLXArray.zeros([Bc, H, 0, Dk], dtype: keys?.dtype ?? .float32)
                v = MLXArray.zeros([Bc, H, 0, Dv], dtype: values?.dtype ?? .float32)
            }

            let leftPad = maxIdx - cacheIdx
            var rightPad = maxSize - k.dim(2) - leftPad
            if rightPad < 0 {
                k = k[.ellipsis, ..<(k.dim(2) + rightPad), 0...]
                v = v[.ellipsis, ..<(v.dim(2) + rightPad), 0...]
                rightPad = 0
            }
            if leftPad != 0 || rightPad != 0 {
                let widths: [IntOrPair] = [
                    .init(0), .init(0), .init((leftPad, rightPad)), .init(0),
                ]
                k = padded(k, widths: widths)
                v = padded(v, widths: widths)
            }
            return (k, v, cacheOffset, cacheLeftPadding + Int32(leftPad))
        }

        let (k1, v1, o1, lp1) = pad(
            keys, values, cacheIdx: _idx,
            cacheOffset: batchOffset, cacheLeftPadding: leftPadding
        )
        let (k2, v2, o2, lp2) = pad(
            other.keys, other.values, cacheIdx: other._idx,
            cacheOffset: other.batchOffset, cacheLeftPadding: other.leftPadding
        )

        keys = concatenated([k1, k2], axis: 0)
        values = concatenated([v1, v2], axis: 0)
        batchOffset = concatenated([o1, o2], axis: 0)
        leftPadding = concatenated([lp1, lp2], axis: 0)
        _idx = maxIdx
    }

    public func extract(_ idx: Int) -> RotatingKVCache {
        let cache = RotatingKVCache(maxSize: maxCacheSize, keep: 0)
        guard let storedK = windowKeys, let storedV = windowValues else {
            return cache
        }

        let leftPad = max(0, Int(leftPadding[idx].item(Int32.self)))
        let kSlice = storedK[idx ..< (idx + 1), 0..., leftPad ..< _idx, 0...]
        let vSlice = storedV[idx ..< (idx + 1), 0..., leftPad ..< _idx, 0...]
        eval(kSlice, vSlice)
        cache.state = [kSlice, vSlice]

        let absoluteOffset = Int(batchOffset[idx].item(Int32.self))
        cache.metaState = [
            "0", "\(maxCacheSize)", "\(Self.allocationStep)", "\(absoluteOffset)",
            "\(kSlice.dim(2))",
        ]
        return cache
    }

    public static func fromSingleRow(_ src: RotatingKVCache) -> BatchRotatingKVCache {
        let meta = src.metaState
        let maxSize = (meta.count > 1 ? Int(meta[1]) : nil) ?? src.maxSize ?? 0
        let absoluteOffset = (meta.count > 3 ? Int(meta[3]) : nil) ?? src.offset
        let s = src.state
        guard maxSize > 0, s.count >= 2 else {
            return BatchRotatingKVCache(maxSize: max(1, maxSize), leftPadding: [0])
        }
        let k = s[0]
        let v = s[1]
        let idx = k.dim(2)
        let result = BatchRotatingKVCache(maxSize: maxSize, leftPadding: [0])
        result.state = [k, v, MLXArray([Int32(absoluteOffset)]), MLXArray([Int32(0)])]
        _ = idx
        return result
    }

    public override func makeMask(
        n: Int, windowSize: Int?, returnArray _: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let actualWindowSize = windowSize ?? maxCacheSize
        if n == 1, min(_idx + n, maxCacheSize) <= actualWindowSize,
            leftPadding.max().item(Int32.self) <= 0
        {
            return .none
        }
        let maskOffset = min(maxCacheSize - 1, _idx)
        let mask = createCausalMask(
            n: n,
            offset: maskOffset,
            windowSize: actualWindowSize,
            leftPadding: leftPadding
        )
        return .array(mask)
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = min(_idx, n)
        _idx -= trimmed
        batchOffset = batchOffset - Int32(trimmed)
        return trimmed
    }

    public func size() -> Int { _idx }
    public func isEmpty() -> Bool { keys == nil }

    public override var state: [MLXArray] {
        get {
            guard let storedK = windowKeys, let storedV = windowValues else {
                return [batchOffset, leftPadding]
            }
            return [storedK, storedV, batchOffset, leftPadding]
        }
        set {
            if newValue.count >= 4 {
                keys = newValue[0]
                values = newValue[1]
                batchOffset = newValue[2]
                leftPadding = newValue[3]
                _idx = newValue[0].dim(2)
                _base = 0
            } else if newValue.count == 2 {
                batchOffset = newValue[0]
                leftPadding = newValue[1]
                keys = nil
                values = nil
                _idx = 0
                _base = 0
            } else {
                fatalError("BatchRotatingKVCache.state setter expects 2 or 4 arrays")
            }
        }
    }

    public override func innerState() -> [MLXArray] {
        state
    }

    public override func copy() -> any KVCache {
        BatchRotatingKVCache(
            maxSize: maxCacheSize,
            keys: windowKeys.map { $0[.ellipsis] },
            values: windowValues.map { $0[.ellipsis] },
            offset: batchOffset[0...],
            leftPadding: leftPadding[0...],
            idx: _idx
        )
    }
}


extension ArraysCache: BatchedCache {
    public func filterBatched(batchIndices: MLXArray) {
        filter(batchIndices: batchIndices)
    }

    public func extendBatched(_ other: any BatchedCache) {
        guard let other = other as? ArraysCache else {
            preconditionFailure("ArraysCache.extendBatched requires another ArraysCache")
        }
        extend(other: other)
    }

    public func prepareBatched(leftPadding _: [Int]?, lengths: [Int]?, rightPadding _: [Int]?) {
        prepare(lengths: lengths)
    }

    public func finalizeBatched() {
        finalize()
    }

    public func extractBatched(_ idx: Int) -> any KVCache {
        extract(idx)
    }

    public func advanceBatched(_ n: Int) {
        advance(n)
    }
}

@inline(__always)
public func dynamicRoll(
    _ x: MLXArray, shifts: MLXArray, axis: Int
) -> MLXArray {
    let n = x.dim(axis)
    var arangeShape = Array(repeating: 1, count: x.ndim)
    arangeShape[axis] = n
    var shiftShape = Array(repeating: 1, count: x.ndim)
    for i in 0 ..< min(shifts.ndim, axis) {
        shiftShape[i] = shifts.dim(i)
    }

    let arange = MLXArray(Int32(0) ..< Int32(n)).reshaped(arangeShape)
    let reshapedShifts = shifts.reshaped(shiftShape)
    let idx = (arange - reshapedShifts) % Int32(n)
    return takeAlong(x, idx, axis: axis)
}


extension BatchKVCache {

    public func zeroTailPerRow(keepLengths: MLXArray) {
        guard let storedK = keys, let storedV = values else { return }
        let T = storedK.dim(2)
        let positions = MLXArray(Int32(0) ..< Int32(T))
            .reshaped([1, 1, T, 1])
        let keep = keepLengths.asType(.int32)
            .reshaped([-1, 1, 1, 1])
        let mask = (positions .< keep).asType(storedK.dtype)
        keys = storedK * mask
        values = storedV * mask
    }
}

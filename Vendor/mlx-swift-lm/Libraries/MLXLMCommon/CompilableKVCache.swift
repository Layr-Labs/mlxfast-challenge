
import Foundation
import MLX
import MLXNN

public class CompilableKVCache: BaseKVCache {

    public var keys: MLXArray?
    public var values: MLXArray?

    public var offsetArray: MLXArray

    public let maxLength: Int

    public let attentionLength: Int

    public var step: Int

    private lazy var maskRinds: MLXArray =
        MLXArray(Int32(0) ..< Int32(attentionLength))

    public init(
        maxLength: Int = 4096,
        step: Int = 256,
        attentionLength: Int? = nil
    ) {
        let attentionLength = attentionLength ?? maxLength
        precondition(
            attentionLength > 0 && attentionLength <= maxLength,
            "CompilableKVCache attention length must fit its backing buffer")
        self.maxLength = maxLength
        self.attentionLength = attentionLength
        self.step = step
        self.offsetArray = MLXArray([Int32(0)])
        super.init()
    }

    public static func promote(from cache: KVCacheSimple, maxLength: Int) -> CompilableKVCache {
        return CompilableKVCache(from: cache, maxLength: maxLength)
    }

    public convenience init(from cache: KVCache, maxLength: Int = 4096) {
        self.init(maxLength: maxLength)

        let existingState = cache.state
        if existingState.count >= 2 {
            let existingKeys = existingState[0]
            let existingValues = existingState[1]

            let seqLen = existingKeys.dim(2)
            let B = existingKeys.dim(0)
            let H = existingKeys.dim(1)
            let kD = existingKeys.dim(3)
            let vD = existingValues.dim(3)

            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: existingKeys.dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: existingValues.dtype)

            self.keys![.ellipsis, ..<seqLen, 0...] = existingKeys
            self.values![.ellipsis, ..<seqLen, 0...] = existingValues

            self.offsetArray = MLXArray([Int32(seqLen)])
        }
    }


    public override var offset: Int {
        get {
            offsetArray[0].item(Int.self)
        }
        set {
            offsetArray = MLXArray([Int32(newValue)])
        }
    }

    public override func innerState() -> [MLXArray] {
        if let keys, let values {
            return [keys, values, offsetArray]
        }
        return [offsetArray]
    }

    public override func update(keys newKeys: MLXArray, values newValues: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let nTokens = newKeys.dim(2)

        if self.keys == nil {
            let B = newKeys.dim(0)
            let H = newKeys.dim(1)
            let kD = newKeys.dim(3)
            let vD = newValues.dim(3)
            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: newKeys.dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: newValues.dtype)
        }

        let prev = offsetArray
        let newOffset = prev + MLXArray([Int32(nTokens)])

        self.keys!._updateInternal(
            dynamicSliceUpdate(self.keys!, update: newKeys, start: prev, axes: [2]))
        self.values!._updateInternal(
            dynamicSliceUpdate(self.values!, update: newValues, start: prev, axes: [2]))

        self.offsetArray._updateInternal(newOffset)

        if attentionLength == maxLength {
            return (self.keys!, self.values!)
        }
        return (
            self.keys![.ellipsis, ..<attentionLength, 0...],
            self.values![.ellipsis, ..<attentionLength, 0...]
        )
    }

    public func sharingStorage(attentionLength: Int) -> CompilableKVCache {
        precondition(
            keys != nil && values != nil,
            "CompilableKVCache storage must be allocated before making a view")
        let view = CompilableKVCache(
            maxLength: maxLength,
            step: step,
            attentionLength: attentionLength)
        view.keys = keys
        view.values = values
        view.offsetArray = offsetArray
        return view
    }


    public override func makeMask(
        n: Int, windowSize: Int?, returnArray: Bool
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        let currentOffsetArr = offsetArray

        let linds: MLXArray
        if n == 1 {
            linds = currentOffsetArr.reshaped(1, 1)
        } else {
            linds = (MLXArray(Int32(0) ..< Int32(n)) + currentOffsetArr).reshaped(n, 1)
        }

        let rinds = maskRinds.reshaped(1, attentionLength)

        var mask = linds .>= rinds

        if let windowSize {
            let windowStart = linds - Int32(windowSize - 1)
            mask = mask & (rinds .>= windowStart)
        }

        return .array(mask)
    }


    public override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            let off: Int = offsetArray[0].item(Int.self)
            if off == keys.dim(2) {
                return [keys, values]
            } else {
                return [
                    keys[.ellipsis, ..<off, 0...],
                    values[.ellipsis, ..<off, 0...],
                ]
            }
        }
        set {
            guard newValue.count == 2 else { return }
            let seqLen = newValue[0].dim(2)
            let B = newValue[0].dim(0)
            let H = newValue[0].dim(1)
            let kD = newValue[0].dim(3)
            let vD = newValue[1].dim(3)

            self.keys = MLXArray.zeros([B, H, maxLength, kD], dtype: newValue[0].dtype)
            self.values = MLXArray.zeros([B, H, maxLength, vD], dtype: newValue[1].dtype)
            self.keys![.ellipsis, ..<seqLen, 0...] = newValue[0]
            self.values![.ellipsis, ..<seqLen, 0...] = newValue[1]
            self.offsetArray = MLXArray([Int32(seqLen)])
        }
    }

    public override var isTrimmable: Bool { true }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let current: Int = offsetArray[0].item(Int.self)
        let trimmed = min(current, n)
        offsetArray = MLXArray([Int32(current - trimmed)])
        super.offset = current - trimmed
        return trimmed
    }

    public override func copy() -> any KVCache {
        let c = CompilableKVCache(
            maxLength: maxLength, step: step, attentionLength: attentionLength)
        c.keys = keys
        c.values = values
        c.offsetArray = offsetArray
        return c
    }


    public var debugDescription: String {
        "CompilableKVCache(offset=\(offset), maxLength=\(maxLength), "
            + "attentionLength=\(attentionLength), "
            + "shape=\(keys?.shape.description ?? "nil"))"
    }
}

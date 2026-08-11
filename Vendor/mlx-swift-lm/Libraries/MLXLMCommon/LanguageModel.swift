// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

public protocol BaseLanguageModel: Module {
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]
}

extension BaseLanguageModel {
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }
}

public struct THW: Sendable {

    public let t: Int
    public let h: Int
    public let w: Int

    public init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var values: (Int, Int, Int) {
        (t, h, w)
    }

    public var product: Int { t * h * w }
}

public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?

    public struct Text {

        public let tokens: MLXArray

        public let mask: MLXArray?

        public init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        public subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        public subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }
    }

    public struct ProcessedImage {

        public let pixels: MLXArray
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    public struct ProcessedVideo {

        public let pixels: MLXArray
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    public init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    public init(
        text: LMInput.Text, image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
    }
}

public struct LMOutput {

    public let logits: MLXArray

    public let state: State?

    public struct State {
        public let crossAttentionStates: MLXArray?

        public init(crossAttentionStates: MLXArray? = nil) {
            self.crossAttentionStates = crossAttentionStates
        }
    }

    public init(logits: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

public enum PrepareResult {
    case tokens(LMInput.Text)

    case logits(LMOutput)
}

public protocol LanguageModel: BaseLanguageModel {

    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    func newCache(parameters: GenerateParameters?) -> [KVCache]
}

extension LanguageModel {
    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }
}

public protocol MTPCapable: LanguageModel {
    var hasMTPHead: Bool { get }

    func mtpForward(hidden: MLXArray, nextTokenIds: MLXArray, cache: [any KVCache]) -> MLXArray

    func makeMTPCache() -> [any KVCache]

    func callWithHidden(input: LMInput.Text, cache: [any KVCache], nConfirmed: Int) -> (MLXArray, MLXArray)
}

public protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        let numLayers = kvHeads.count

        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< numLayers).map { _ in
                RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
        } else {
            return (0 ..< numLayers).map { _ in KVCacheSimple() }
        }
    }
}

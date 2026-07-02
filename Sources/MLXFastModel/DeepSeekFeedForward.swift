import Foundation
import MLX
import MLXFastCore

public struct DeepSeekMLPWeights {
    public let gate: DeepSeekLinearWeight
    public let up: DeepSeekLinearWeight
    public let down: DeepSeekLinearWeight

    public init(gate: DeepSeekLinearWeight, up: DeepSeekLinearWeight, down: DeepSeekLinearWeight) {
        self.gate = gate
        self.up = up
        self.down = down
    }

    public init(gate: MLXArray, up: MLXArray, down: MLXArray) {
        self.init(
            gate: DeepSeekLinearWeight(gate),
            up: DeepSeekLinearWeight(up),
            down: DeepSeekLinearWeight(down)
        )
    }
}

public struct DeepSeekMoEWeights {
    public let gate: MLXArray
    public let correctionBias: MLXArray?
    public let tokenToExpert: MLXArray?
    public let sharedExperts: DeepSeekMLPWeights

    public init(
        gate: MLXArray,
        correctionBias: MLXArray?,
        tokenToExpert: MLXArray?,
        sharedExperts: DeepSeekMLPWeights
    ) {
        self.gate = gate
        self.correctionBias = correctionBias
        self.tokenToExpert = tokenToExpert
        self.sharedExperts = sharedExperts
    }
}

public struct DeepSeekMoESpec: Equatable {
    public let routedExperts: DeepSeekRoutedExpertSpec
    public let expertsPerToken: Int
    public let routedScalingFactor: Double
    public let normTopKProb: Bool
    public let scoring: DeepSeekGateScoring

    public init(
        routedExperts: DeepSeekRoutedExpertSpec,
        expertsPerToken: Int,
        routedScalingFactor: Double,
        normTopKProb: Bool,
        scoring: DeepSeekGateScoring
    ) {
        self.routedExperts = routedExperts
        self.expertsPerToken = expertsPerToken
        self.routedScalingFactor = routedScalingFactor
        self.normTopKProb = normTopKProb
        self.scoring = scoring
    }

    public init(layerIndex: Int, config: DeepSeekConfig) throws {
        guard let scoring = DeepSeekGateScoring(rawValue: config.scoringFunc) else {
            throw MLXFastError.invalidInput(
                "unsupported DeepSeek MoE gate scoring function \(config.scoringFunc)"
            )
        }
        self.init(
            routedExperts: DeepSeekRoutedExpertSpec(layerIndex: layerIndex, config: config),
            expertsPerToken: config.expertsPerToken,
            routedScalingFactor: config.routedScalingFactor,
            normTopKProb: config.normTopkProb,
            scoring: scoring
        )
    }
}

public enum DeepSeekMLP {
    public static func forward(
        _ x: MLXArray,
        weights: DeepSeekMLPWeights,
        swigluLimit: Double
    ) -> MLXArray {
        let gate = DeepSeekOps.linear(input: x, weight: weights.gate)
        let up = DeepSeekOps.linear(input: x, weight: weights.up)
        let hidden = DeepSeekOps.limitedSwiGLU(gate: gate, up: up, limit: swigluLimit)
        return DeepSeekOps.linear(input: hidden, weight: weights.down)
    }
}

public enum DeepSeekMoE {
    public static func forward(
        _ x: MLXArray,
        inputIDs: MLXArray?,
        weights: DeepSeekMoEWeights,
        loader: DeepSeekWeightLoader,
        spec: DeepSeekMoESpec
    ) throws -> MLXArray {
        let routing = try DeepSeekMoEGate.route(
            hidden: x,
            inputIDs: inputIDs,
            weight: weights.gate,
            correctionBias: weights.correctionBias,
            tokenToExpert: weights.tokenToExpert,
            topK: spec.expertsPerToken,
            routedScalingFactor: spec.routedScalingFactor,
            normTopKProb: spec.normTopKProb,
            scoring: spec.scoring
        )
        let routed = try DeepSeekRoutedExperts.forward(
            x,
            expertIndices: routing.indices,
            loader: loader,
            spec: spec.routedExperts
        )
        let shared = DeepSeekMLP.forward(
            x,
            weights: weights.sharedExperts,
            swigluLimit: spec.routedExperts.swigluLimit
        )
        return combine(
            routedExpertOutput: routed,
            routeWeights: routing.weights,
            sharedExpertOutput: shared
        )
    }

    public static func combine(
        routedExpertOutput: MLXArray,
        routeWeights: MLXArray,
        sharedExpertOutput: MLXArray
    ) -> MLXArray {
        let weightedRouted = (
            routedExpertOutput * routeWeights.expandedDimensions(axis: -1).asType(routedExpertOutput.dtype)
        ).sum(axis: -2)
        return weightedRouted + sharedExpertOutput
    }

    public static func forward(
        _ x: MLXArray,
        routeWeights: MLXArray,
        sharedWeights: DeepSeekMLPWeights,
        swigluLimit: Double,
        routedExpertOutput: (_ x: MLXArray) throws -> MLXArray
    ) throws -> MLXArray {
        let routed = try routedExpertOutput(x)
        let shared = DeepSeekMLP.forward(x, weights: sharedWeights, swigluLimit: swigluLimit)
        return combine(
            routedExpertOutput: routed,
            routeWeights: routeWeights,
            sharedExpertOutput: shared
        )
    }
}

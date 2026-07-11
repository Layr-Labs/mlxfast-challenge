import CryptoKit
import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import Tokenizers

public struct CorrectnessOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String
    // Check only the [stepStart, stepStart + stepCount) slice of the teacher-forced
    // window instead of the full [0, correctnessSteps) range. Lets a fleet of machines
    // each verify a disjoint slice of the same case in parallel; see
    // compareTeacherForcedWithWorker for why this is sound under teacher forcing.
    // stepCount defaults to MLXFastConstants.correctnessSteps when nil, matching the
    // pre-existing full-range behavior. Only honored on the runtime-worker code path.
    public let stepStart: Int
    public let stepCount: Int?
    // When true, skip anchors/free-run/behavior/GPQA gates entirely and check only
    // golden.cases (the base case). Without this, a machine checking a step-range
    // slice of the base case still runs every gate in the golden file, and its
    // checked_steps total becomes base-slice-length + gate-step-counts -- not
    // comparable across machines and not what a range-coverage check expects. Use
    // this on every machine that's assigned a base-case slice; leave it off on
    // whichever machine (if any) is responsible for the gates.
    public let baseCaseOnly: Bool

    public init(
        weightsPath: String,
        goldenPath: String,
        stepStart: Int = 0,
        stepCount: Int? = nil,
        baseCaseOnly: Bool = false
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.stepStart = stepStart
        self.stepCount = stepCount
        self.baseCaseOnly = baseCaseOnly
    }
}

public struct CorrectnessTraceOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let caseName: String?
    public let step: Int
    public let topK: Int

    public init(
        weightsPath: String,
        goldenPath: String,
        caseName: String? = nil,
        step: Int,
        topK: Int = 8
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.caseName = caseName
        self.step = step
        self.topK = topK
    }
}

public struct GreedyGenerationOptions: Equatable {
    public let weightsPath: String
    public let promptTokens: [Int]
    public let steps: Int

    public init(weightsPath: String, promptTokens: [Int], steps: Int) {
        self.weightsPath = weightsPath
        self.promptTokens = promptTokens
        self.steps = steps
    }
}

public struct CorrectnessTraceLogit: Codable, Equatable {
    public let token: Int
    public let logit: Double
}

public struct CorrectnessTraceReport: Codable, Equatable {
    public let caseName: String
    public let step: Int
    public let promptTokenCount: Int
    public let expectedToken: Int
    public let actualToken: Int
    public let matchedPrefixSteps: Int
    public let generatedPrefix: [Int]
    public let actualTokenLogit: Double
    public let expectedTokenLogit: Double
    public let actualExpectedLogitDelta: Double
    public let expectedTokenRank: Int
    public let topLogitMargin: Double?
    public let topLogits: [CorrectnessTraceLogit]
    public let goldenHash: String

    enum CodingKeys: String, CodingKey {
        case caseName = "case_name"
        case step
        case promptTokenCount = "prompt_token_count"
        case expectedToken = "expected_token"
        case actualToken = "actual_token"
        case matchedPrefixSteps = "matched_prefix_steps"
        case generatedPrefix = "generated_prefix"
        case actualTokenLogit = "actual_token_logit"
        case expectedTokenLogit = "expected_token_logit"
        case actualExpectedLogitDelta = "actual_expected_logit_delta"
        case expectedTokenRank = "expected_token_rank"
        case topLogitMargin = "top_logit_margin"
        case topLogits = "top_logits"
        case goldenHash = "golden_hash"
    }
}

public struct CorrectnessReport: Codable, Equatable {
    public let passed: Bool
    public let checkedSteps: Int
    public let caseCount: Int
    public let expertCacheHits: UInt64
    public let expertCacheMisses: UInt64
    public let expertCacheEvictions: UInt64
    public let expertBytesRead: UInt64
    public let expertReadSeconds: Double
    public let expertPeakCachedTensors: UInt64
    public let expertHitRate: Double
    public let firstFailingCase: String?
    public let firstFailingStep: Int?
    public let expectedToken: Int?
    public let actualToken: Int?
    public let goldenHash: String
    public let error: String

    enum CodingKeys: String, CodingKey {
        case passed
        case checkedSteps = "checked_steps"
        case caseCount = "case_count"
        case expertCacheHits = "expert_cache_hits"
        case expertCacheMisses = "expert_cache_misses"
        case expertCacheEvictions = "expert_cache_evictions"
        case expertBytesRead = "expert_bytes_read"
        case expertReadSeconds = "expert_read_seconds"
        case expertPeakCachedTensors = "expert_peak_cached_tensors"
        case expertHitRate = "expert_hit_rate"
        case firstFailingCase = "first_failing_case"
        case firstFailingStep = "first_failing_step"
        case expectedToken = "expected_token"
        case actualToken = "actual_token"
        case goldenHash = "golden_hash"
        case error
    }

    public init(
        passed: Bool,
        checkedSteps: Int,
        caseCount: Int,
        expertCacheHits: UInt64 = 0,
        expertCacheMisses: UInt64 = 0,
        expertCacheEvictions: UInt64 = 0,
        expertBytesRead: UInt64 = 0,
        expertReadSeconds: Double = 0,
        expertPeakCachedTensors: UInt64 = 0,
        expertHitRate: Double = 0,
        firstFailingCase: String?,
        firstFailingStep: Int?,
        expectedToken: Int?,
        actualToken: Int?,
        goldenHash: String,
        error: String
    ) {
        self.passed = passed
        self.checkedSteps = checkedSteps
        self.caseCount = caseCount
        self.expertCacheHits = expertCacheHits
        self.expertCacheMisses = expertCacheMisses
        self.expertCacheEvictions = expertCacheEvictions
        self.expertBytesRead = expertBytesRead
        self.expertReadSeconds = expertReadSeconds
        self.expertPeakCachedTensors = expertPeakCachedTensors
        self.expertHitRate = expertHitRate
        self.firstFailingCase = firstFailingCase
        self.firstFailingStep = firstFailingStep
        self.expectedToken = expectedToken
        self.actualToken = actualToken
        self.goldenHash = goldenHash
        self.error = error
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(passed, forKey: .passed)
        try container.encode(checkedSteps, forKey: .checkedSteps)
        try container.encode(caseCount, forKey: .caseCount)
        try container.encode(expertCacheHits, forKey: .expertCacheHits)
        try container.encode(expertCacheMisses, forKey: .expertCacheMisses)
        try container.encode(expertCacheEvictions, forKey: .expertCacheEvictions)
        try container.encode(expertBytesRead, forKey: .expertBytesRead)
        try container.encode(expertReadSeconds, forKey: .expertReadSeconds)
        try container.encode(expertPeakCachedTensors, forKey: .expertPeakCachedTensors)
        try container.encode(expertHitRate, forKey: .expertHitRate)
        if let firstFailingCase {
            try container.encode(firstFailingCase, forKey: .firstFailingCase)
        } else {
            try container.encodeNil(forKey: .firstFailingCase)
        }
        if let firstFailingStep {
            try container.encode(firstFailingStep, forKey: .firstFailingStep)
        } else {
            try container.encodeNil(forKey: .firstFailingStep)
        }
        if let expectedToken {
            try container.encode(expectedToken, forKey: .expectedToken)
        } else {
            try container.encodeNil(forKey: .expectedToken)
        }
        if let actualToken {
            try container.encode(actualToken, forKey: .actualToken)
        } else {
            try container.encodeNil(forKey: .actualToken)
        }
        try container.encode(goldenHash, forKey: .goldenHash)
        try container.encode(error, forKey: .error)
    }

    public var expertStreamingStats: ExpertStreamingStats {
        ExpertStreamingStats(
            cacheHits: expertCacheHits,
            cacheMisses: expertCacheMisses,
            cacheEvictions: expertCacheEvictions,
            bytesRead: expertBytesRead,
            readSeconds: expertReadSeconds,
            peakCachedTensors: expertPeakCachedTensors
        )
    }
}

public struct BenchmarkOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let correctnessSteps: Int
    public let benchmarkDecodeSteps: Int
    public let semanticGPQAOutputPath: String?
    public let semanticGPQATokenizerPath: String?
    public let semanticGPQACaseCount: Int
    public let semanticGPQAMaxNewTokens: Int
    // Both default true/false to reproduce the original, single-machine
    // behavior exactly. Together with correctnessSteps == 0 (skip only the
    // base case, split elsewhere) these let the base correctness case, the
    // anchor/free-run/behavior/GPQA gates, and the timed prefill/decode
    // measurement each run on their own independent machine: checkGates false
    // skips anchors/free-run/behavior/GPQA entirely (a "timing-only" machine);
    // skipTimedBenchmark true skips the prefill/decode measurement entirely (a
    // "gates-only" machine). Never both false and both skip at once -- that
    // combination is meaningless (nothing left to check or time) and is
    // rejected by validateBenchmarkOptions.
    public let checkGates: Bool
    public let skipTimedBenchmark: Bool

    public init(
        weightsPath: String,
        goldenPath: String,
        correctnessSteps: Int = MLXFastConstants.correctnessSteps,
        benchmarkDecodeSteps: Int = MLXFastConstants.benchmarkDecodeSteps,
        semanticGPQAOutputPath: String? = nil,
        semanticGPQATokenizerPath: String? = nil,
        semanticGPQACaseCount: Int = MLXFastConstants.semanticGPQACaseCount,
        semanticGPQAMaxNewTokens: Int = MLXFastConstants.semanticGPQAMaxNewTokens,
        checkGates: Bool = true,
        skipTimedBenchmark: Bool = false
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.correctnessSteps = correctnessSteps
        self.benchmarkDecodeSteps = benchmarkDecodeSteps
        self.semanticGPQAOutputPath = semanticGPQAOutputPath
        self.semanticGPQATokenizerPath = semanticGPQATokenizerPath
        self.semanticGPQACaseCount = semanticGPQACaseCount
        self.semanticGPQAMaxNewTokens = semanticGPQAMaxNewTokens
        self.checkGates = checkGates
        self.skipTimedBenchmark = skipTimedBenchmark
    }
}

public struct LocalIterateOptions: Equatable {
    public let weightsPath: String
    public let goldenPath: String
    public let benchmarkDecodeSteps: Int
    public let timingRepeats: Int
    public let modeName: String
    public let runtime: String

    public init(
        weightsPath: String,
        goldenPath: String = MLXFastConstants.defaultPublicCorrectnessGoldenPath,
        benchmarkDecodeSteps: Int = MLXFastConstants.localIterateBenchmarkDecodeSteps,
        timingRepeats: Int = 1,
        modeName: String = "local-iterate",
        runtime: String = "swift-local-iterate"
    ) {
        self.weightsPath = weightsPath
        self.goldenPath = goldenPath
        self.benchmarkDecodeSteps = benchmarkDecodeSteps
        self.timingRepeats = timingRepeats
        self.modeName = modeName
        self.runtime = runtime
    }
}

public struct RuntimeWorkerOptions: Equatable {
    public static let defaultHelloTimeoutSeconds = 15 * 60.0
    public static let defaultRequestTimeoutSeconds = 15 * 60.0
    public static let defaultShutdownTimeoutSeconds = 2.0
    public static let defaultTerminationGraceSeconds = 1.0

    public let executablePath: String
    public let sandboxProfilePath: String?
    // Local modes only: stream the worker's stderr lines to the parent's
    // stderr while retaining the capped diagnostic tail. The pipe is always
    // drained so verbose model code cannot stall the worker; this flag controls
    // forwarding only and stays off for official/hidden runs.
    public let forwardsWorkerStderr: Bool
    public let helloTimeoutSeconds: Double
    public let requestTimeoutSeconds: Double
    public let shutdownTimeoutSeconds: Double
    public let terminationGraceSeconds: Double

    public init(
        executablePath: String,
        sandboxProfilePath: String? = nil,
        forwardsWorkerStderr: Bool = false,
        helloTimeoutSeconds: Double = RuntimeWorkerOptions.defaultHelloTimeoutSeconds,
        requestTimeoutSeconds: Double = RuntimeWorkerOptions.defaultRequestTimeoutSeconds,
        shutdownTimeoutSeconds: Double = RuntimeWorkerOptions.defaultShutdownTimeoutSeconds,
        terminationGraceSeconds: Double = RuntimeWorkerOptions.defaultTerminationGraceSeconds
    ) {
        self.executablePath = executablePath
        self.sandboxProfilePath = sandboxProfilePath
        self.forwardsWorkerStderr = forwardsWorkerStderr
        self.helloTimeoutSeconds = helloTimeoutSeconds
        self.requestTimeoutSeconds = requestTimeoutSeconds
        self.shutdownTimeoutSeconds = shutdownTimeoutSeconds
        self.terminationGraceSeconds = terminationGraceSeconds
    }
}

// Implementation lives in the GemmaRuntime*.swift split files.
public enum GemmaRuntime {}

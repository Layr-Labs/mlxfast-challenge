#if !MLXFAST_TRUSTED_HARNESS
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import MLXLLM

struct ExperimentalTrainedMTPBlockRequest: Equatable {
    let previousToken: Int
    let maxBlockSize: Int
}

struct ExperimentalTrainedMTPWorkerState {
    var began = false
    var poisoned = false
    var seedTokenCount = 0
    var decodedTokenCount = 0
}

func validateExperimentalTrainedMTPBlockRequest(
    _ request: RuntimeWorkerRequest,
    decodedTokenCount: Int
) throws -> ExperimentalTrainedMTPBlockRequest {
    guard request.id > 0, request.kind == "mtp_decode_block" else {
        throw MLXFastError.invalidInput(
            "trained MTP block request has an invalid id or kind"
        )
    }
    guard request.promptTokens == nil,
          request.seedTokens == nil,
          request.steps == nil,
          request.topK == nil,
          request.expectedToken == nil,
          let previousToken = request.token,
          previousToken >= 0,
          previousToken < MLXFastConstants.vocabSize,
          let maxBlockSize = request.maxBlockSize,
          maxBlockSize > 0,
          maxBlockSize <= MLXFastConstants.experimentalMTPMaxBlockSize
    else {
        throw MLXFastError.invalidInput(
            "trained MTP block request has invalid or cross-kind fields"
        )
    }
    guard decodedTokenCount >= 0 else {
        throw MLXFastError.invalidInput(
            "trained MTP worker has a negative committed token count"
        )
    }
    let (requestedTotal, overflow) =
        decodedTokenCount.addingReportingOverflow(maxBlockSize)
    guard !overflow,
          requestedTotal
              <= MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens
    else {
        throw MLXFastError.invalidInput(
            "trained MTP block request exceeds the fixed decode window"
        )
    }
    return ExperimentalTrainedMTPBlockRequest(
        previousToken: previousToken,
        maxBlockSize: maxBlockSize
    )
}

extension LagunaRuntime {
    public static func runExperimentalMTPWorker(
        targetWeightsPath: String,
        assistantPath: String,
        contractPath: String,
        verificationMode: MTPVerificationMode,
        requireTrainedAssistant: Bool
    ) throws {
        guard requireTrainedAssistant else {
            throw MLXFastError.invalidInput(
                "trained MTP worker requires --require-trained-assistant; "
                    + "serial fallback is available only through mtp-probe"
            )
        }
        startRuntimeWorkerOrphanReaper()
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()

        // Validate the exact organizer sidecar and target artifact before any
        // submitted model code is asked to load them.
        let beforeLoad = try validateExperimentalMTPArtifacts(
            targetWeightsPath: targetWeightsPath,
            assistantPath: assistantPath,
            contractPath: contractPath
        )
        let config = try Gemma4Config.load(from: targetWeightsPath)
        let loader = try Gemma4WeightLoader(weightsPath: targetWeightsPath)
        try loader.denseStore.validateReadableByteRanges()
        try loader.validateRequiredMetadata(config: config)
        let weightCache = Gemma4RuntimeWeightCache(
            loader: loader,
            config: config
        )
        let target = try weightCache.requireLibraryModel()
        let drafter = try waitForExperimentalMTPAsync {
            try await Gemma4AssistantDraftModel.load(
                from: URL(fileURLWithPath: assistantPath)
            )
        }
        // Match the target runtime's existing constructor-time BOS warmup for
        // the MTP-only graph shapes: one fixed, input-independent prefill and
        // one K=4 draft/verify round on throwaway cache state. No scored prompt
        // exists yet. The real begin request performs the trusted allocator
        // cache clear before its timer-charged seed prefill.
        do {
            let warmupSession = try Gemma4TrainedMTPBlockSession(
                target: target,
                drafter: drafter,
                verificationMode: verificationMode
            )
            let warmupSeed = try warmupSession.begin(
                seedTokens: Array(
                    repeating: 2,
                    count: MLXFastConstants.correctnessPromptTokens
                )
            )
            _ = try warmupSession.generateBlock(
                previousCommittedToken: warmupSeed,
                maxBlockSize: MLXFastConstants.experimentalMTPMaxBlockSize
            )
        }
        // The drafter round above is acceptance-dependent, so it cannot
        // guarantee that every timed target shape compiled. Warm the serial
        // rows, the exact-pair segments when selected, and the deep
        // sliding-attention switch unconditionally on throwaway BOS state.
        try target.warmMTPTargetKernels(
            includeExactPair: verificationMode == .exactPair
        )
        let session = try Gemma4TrainedMTPBlockSession(
            target: target,
            drafter: drafter,
            verificationMode: verificationMode
        )

        // The path-based upstream loader cannot consume already-open file
        // descriptors. Rehash after load and compare the complete inventory so
        // a replace-during-load race fails closed. Once this passes, target and
        // assistant tensors are resident and later path replacement is inert.
        let afterLoad = try validateExperimentalMTPArtifacts(
            targetWeightsPath: targetWeightsPath,
            assistantPath: assistantPath,
            contractPath: contractPath
        )
        guard beforeLoad == afterLoad else {
            throw MLXFastError.invalidInput(
                "experimental MTP artifacts changed while models were loading"
            )
        }

        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        let sessionNonce = generateRuntimeWorkerNonce()
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true
        )))

        var state = ExperimentalTrainedMTPWorkerState()
        var expectedRequestID = 1
        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(
                    RuntimeWorkerRequest.self,
                    from: Data(line.utf8)
                )
                guard request.id == expectedRequestID else {
                    throw MLXFastError.invalidInput(
                        "trained MTP request id must be monotonic; "
                            + "expected \(expectedRequestID), got \(request.id)"
                    )
                }
                expectedRequestID += 1
                do {
                    response = try handleExperimentalMTPWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        session: session,
                        state: &state
                    )
                } catch {
                    response = RuntimeWorkerResponse(
                        id: request.id,
                        nonce: sessionNonce,
                        ok: false,
                        error: "\(error)"
                    )
                }
            } catch {
                response = RuntimeWorkerResponse(
                    id: -1,
                    nonce: sessionNonce,
                    ok: false,
                    error: "\(error)"
                )
            }
            try protocolIO.writeLine(try encoder.encode(response))
        }
    }

    static func handleExperimentalMTPWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        session: Gemma4TrainedMTPBlockSession,
        state: inout ExperimentalTrainedMTPWorkerState
    ) throws -> RuntimeWorkerResponse {
        guard !state.poisoned else {
            throw MLXFastError.invalidInput(
                "trained MTP decode session is poisoned after an earlier failure"
            )
        }

        switch request.kind {
        case "mtp_decode_begin":
            guard !state.began,
                  request.id > 0,
                  let seedTokens = request.seedTokens,
                  !seedTokens.isEmpty,
                  request.promptTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil
            else {
                throw MLXFastError.invalidInput(
                    "trained MTP begin request is repeated or malformed"
                )
            }
            try resetRuntimeWorkerAllocatorForPhaseStart()
            do {
                // Charged, input-independent (no seed applied yet): re-warm the
                // allocator working set the trusted clear just freed so the
                // first timed block does not pay a one-time first-touch spike.
                try session.warmWorkingSetAfterAllocatorReset()
                let seedToken = try session.begin(seedTokens: seedTokens)
                state.began = true
                state.seedTokenCount = seedTokens.count
                state.decodedTokenCount = 0
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    seedToken: seedToken
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "mtp_decode_block":
            guard state.began else {
                throw MLXFastError.invalidInput(
                    "trained MTP block requested before begin"
                )
            }
            let block = try validateExperimentalTrainedMTPBlockRequest(
                request,
                decodedTokenCount: state.decodedTokenCount
            )
            do {
                let result = try session.generateBlock(
                    previousCommittedToken: block.previousToken,
                    maxBlockSize: block.maxBlockSize
                )
                guard !result.tokens.isEmpty,
                      result.tokens.count <= block.maxBlockSize,
                      result.tokens.allSatisfy({
                          $0 >= 0 && $0 < MLXFastConstants.vocabSize
                      })
                else {
                    throw MLXFastError.invalidInput(
                        "trained MTP model returned an invalid target-confirmed block"
                    )
                }
                let (nextCount, overflow) =
                    state.decodedTokenCount.addingReportingOverflow(
                        result.tokens.count
                    )
                let (expectedCacheOffset, offsetOverflow) =
                    state.seedTokenCount.addingReportingOverflow(nextCount)
                guard !overflow,
                      !offsetOverflow,
                      nextCount <=
                          MLXFastConstants.experimentalMTPMaxConfiguredTotalTokens,
                      result.targetCacheOffset == expectedCacheOffset
                else {
                    throw MLXFastError.invalidInput(
                        "trained MTP logical token count and target cache offset diverged"
                    )
                }
                state.decodedTokenCount = nextCount
                return RuntimeWorkerResponse(
                    id: request.id,
                    nonce: sessionNonce,
                    ok: true,
                    tokens: result.tokens
                )
            } catch {
                state.poisoned = true
                throw error
            }

        case "mtp_phase_diagnostics":
            guard state.began,
                  request.promptTokens == nil,
                  request.seedTokens == nil,
                  request.token == nil,
                  request.steps == nil,
                  request.maxBlockSize == nil,
                  request.topK == nil,
                  request.expectedToken == nil
            else {
                throw MLXFastError.invalidInput(
                    "trained MTP diagnostics request is malformed or before begin"
                )
            }
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                peakRamGB: peakResidentMemoryGB(),
                mlxActiveMemoryBytes: Memory.activeMemory,
                mlxCacheMemoryBytes: Memory.cacheMemory,
                mlxPeakMemoryBytes: Memory.peakMemory,
                targetVerificationMode: session.verificationMode.rawValue,
                exactPairSegmentCount: session.exactPairSegmentCount,
                exactPairRollbackRowCount: session.exactPairRollbackRowCount,
                serialVerificationRowCount: session.serialVerificationRowCount
            )

        default:
            throw MLXFastError.invalidInput(
                "trained MTP worker rejects non-MTP request kind \(request.kind)"
            )
        }
    }
}

private final class ExperimentalMTPAsyncResultBox<T>: @unchecked Sendable {
    var result: Result<T, Error>?
}

private func waitForExperimentalMTPAsync<T: Sendable>(
    _ operation: @escaping @Sendable () async throws -> T
) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    let box = ExperimentalMTPAsyncResultBox<T>()
    Task {
        do {
            box.result = .success(try await operation())
        } catch {
            box.result = .failure(error)
        }
        semaphore.signal()
    }
    semaphore.wait()
    guard let result = box.result else {
        throw MLXFastError.invalidInput(
            "trained MTP async model load completed without a result"
        )
    }
    return try result.get()
}
#endif

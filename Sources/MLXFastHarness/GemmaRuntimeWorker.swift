import CryptoKit
import Darwin
import Foundation
import MLX
import MLXFastCore
import MLXFastModel
import Tokenizers

// GemmaRuntime is split across GemmaRuntime*.swift for auditability.
// Generated split; behavior identical to the original single file.

extension GemmaRuntime {
    public static func runWorker(weightsPath: String) throws {
        let config = try Gemma4Config.load(from: weightsPath)
        let loader = try Gemma4WeightLoader(weightsPath: weightsPath)
        // Validate transformed-weight structure HERE, inside the sandboxed worker,
        // rather than in the trusted parent. These checks execute editable
        // MLXFastModel code (DenseTensorStore / Gemma4WeightLoader); the parent
        // used to run the equivalent via BenchmarkPreflight.check, which meant
        // submitted code ran in the unsandboxed process that authors score.json.
        // Failing here throws before the protocol hello below, so the parent's
        // worker client sees the worker fail to start and records a failed
        // benchmark -- same coverage, no submitted code in the score-writing
        // parent.
        try loader.denseStore.validateReadableByteRanges()
        try loader.validateRequiredMetadata(config: config)
        let weightCache = Gemma4RuntimeWeightCache(loader: loader, config: config)
        let decoder = JSONDecoder()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        // Keep protocol I/O off fd 0/1 so submitted model code cannot read
        // future request nonces or spoof JSON responses with normal stdio.
        let protocolIO = try RuntimeWorkerProtocolIO.isolatingStandardIO()
        let sessionNonce = generateRuntimeWorkerNonce()
        // expertStats is always the zero struct for this RAM-resident dense
        // runtime (no expert-streaming machinery); kept in the protocol hello
        // so the schema/field shape stays unchanged from earlier submissions.
        try protocolIO.writeLine(try encoder.encode(RuntimeWorkerResponse(
            id: 0,
            nonce: sessionNonce,
            ok: true,
            expertStats: expertStats(from: weightCache)
        )))
        var state = RuntimeWorkerState()

        while let line = try protocolIO.readLine() {
            guard !line.isEmpty else {
                continue
            }
            let response: RuntimeWorkerResponse
            do {
                let request = try decoder.decode(RuntimeWorkerRequest.self, from: Data(line.utf8))
                do {
                    response = try handleWorkerRequest(
                        request,
                        sessionNonce: sessionNonce,
                        weightCache: weightCache,
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
                response = RuntimeWorkerResponse(id: -1, nonce: sessionNonce, ok: false, error: "\(error)")
            }
            let data = try encoder.encode(response)
            try protocolIO.writeLine(data)
        }
    }

    static func handleWorkerRequest(
        _ request: RuntimeWorkerRequest,
        sessionNonce: String,
        weightCache: Gemma4RuntimeWeightCache,
        state: inout RuntimeWorkerState
    ) throws -> RuntimeWorkerResponse {
        switch request.kind {
        case "correctness":
            guard let promptTokens = request.promptTokens, let steps = request.steps else {
                throw MLXFastError.invalidInput("runtime worker correctness request missing prompt_tokens or steps")
            }
            let tokens = try generateGreedyCached(
                promptTokens: promptTokens,
                steps: steps,
                weightCache: weightCache
            )
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                tokens: tokens,
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "correctness_begin":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing prompt_tokens")
            }
            let cache = Gemma4ModelCache(config: weightCache.config)
            let logits = try Gemma4Model.logits(
                inputIDs: inputIDsArray(promptTokens),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            let token = try GemmaCorrectness.greedyToken(from: logits)
            state.correctnessCache = cache
            state.correctnessPromptTokenCount = promptTokens.count
            state.correctnessStep = 0
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits),
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "correctness_step":
            guard let previousToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness request missing token")
            }
            guard let cache = state.correctnessCache else {
                throw MLXFastError.invalidInput("runtime worker teacher-forced correctness step before begin")
            }
            let logits = try Gemma4Model.logits(
                inputIDs: inputIDsArray([previousToken]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: state.correctnessPromptTokenCount + state.correctnessStep
            )
            let token = try GemmaCorrectness.greedyToken(from: logits)
            state.correctnessStep += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token,
                topLogits: try topLogits(from: logits, topK: MLXFastConstants.correctnessTopLogits),
                expertStats: expertStats(from: weightCache),
                peakRamGB: peakResidentMemoryGB()
            )

        case "prefill":
            guard let promptTokens = request.promptTokens else {
                throw MLXFastError.invalidInput("runtime worker prefill request missing prompt_tokens")
            }
            let cache = Gemma4ModelCache(config: weightCache.config)
            let logits = try Gemma4Model.logits(
                inputIDs: inputIDsArray(promptTokens),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            eval(logits)
            let token = try GemmaCorrectness.greedyToken(from: logits)
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token
            )

        case "decode_begin":
            guard let seedTokens = request.seedTokens else {
                throw MLXFastError.invalidInput("runtime worker decode_begin request missing seed_tokens")
            }
            // Exactly one whole-prompt (seed) forward runs here, with NO preceding
            // warmup pass. The decode measurement deliberately charges this seed
            // prefill to the decode phase (see measureWorkerDecode). A second,
            // identical whole-prompt forward -- the warmup this used to run before
            // the seed -- let submitted model code memoize one pass and serve the
            // other from that memo (both had the same tokens at offset 0), so two
            // charged forwards collapsed into one and inflated decode_speedup with
            // no real speedup. The trusted harness cannot force editable code to
            // recompute a forward it issues, so the only robust defense is to never
            // issue two identical forwards in the timed window: with a single seed
            // forward there is no identical predecessor to reuse, and the 128
            // single-token decode steps are input-dependent and cannot be
            // precomputed. Prefill/decode/correctness each run in their own worker
            // process, so no memo persists across phases either.
            let cache = Gemma4ModelCache(config: weightCache.config)
            let logits = try Gemma4Model.logits(
                inputIDs: inputIDsArray(seedTokens),
                weightCache: weightCache,
                cache: cache,
                positionOffset: 0
            )
            let token = try GemmaCorrectness.greedyToken(from: logits)
            let seedToken = token
            cache.materializeCachedState()
            state.decodeCache = cache
            state.decodeSeedTokenCount = seedTokens.count
            state.decodeStep = 0
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                seedToken: seedToken
            )

        case "decode_step":
            guard let inputToken = request.token else {
                throw MLXFastError.invalidInput("runtime worker decode_step request missing token")
            }
            guard let cache = state.decodeCache else {
                throw MLXFastError.invalidInput("runtime worker decode_step before decode_begin")
            }
            let validationDelayMS = try submissionValidationDelayMilliseconds()
            guard validationDelayMS >= 0 else {
                throw MLXFastError.invalidInput("runtime worker validation delay must be non-negative")
            }
            let logits = try Gemma4Model.logits(
                inputIDs: inputIDsArray([inputToken]),
                weightCache: weightCache,
                cache: cache,
                positionOffset: state.decodeSeedTokenCount + state.decodeStep
            )
            let token = try GemmaCorrectness.greedyToken(from: logits)
            if validationDelayMS > 0 {
                Thread.sleep(forTimeInterval: Double(validationDelayMS) / 1_000.0)
            }
            state.decodeStep += 1
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                token: token
            )

        case "phase_diagnostics":
            let peakRamGB = peakResidentMemoryGB()
            let stats = expertStats(from: weightCache)
            Memory.clearCache()
            return RuntimeWorkerResponse(
                id: request.id,
                nonce: sessionNonce,
                ok: true,
                expertStats: stats,
                peakRamGB: peakRamGB
            )

        default:
            throw MLXFastError.invalidInput("runtime worker received unknown request kind \(request.kind)")
        }
    }

}

struct RuntimeWorkerRequest: Codable {
    let id: Int
    let kind: String
    let promptTokens: [Int]?
    let token: Int?
    let seedTokens: [Int]?
    let steps: Int?

    enum CodingKeys: String, CodingKey {
        case id
        case kind
        case promptTokens = "prompt_tokens"
        case token
        case seedTokens = "seed_tokens"
        case steps
    }
}

struct RuntimeWorkerState {
    var correctnessCache: Gemma4ModelCache?
    var correctnessPromptTokenCount = 0
    var correctnessStep = 0
    var decodeCache: Gemma4ModelCache?
    var decodeSeedTokenCount = 0
    var decodeStep = 0
}

struct RuntimeWorkerResponse: Codable {
    let id: Int
    let nonce: String?
    let ok: Bool
    let error: String?
    let token: Int?
    let topLogits: [CorrectnessTraceLogit]?
    let topLogitRows: [[CorrectnessTraceLogit]]?
    let seedToken: Int?
    let tokens: [Int]?
    let expertStats: ExpertStreamingStats?
    let peakRamGB: Double?

    init(
        id: Int,
        nonce: String? = nil,
        ok: Bool,
        error: String? = nil,
        token: Int? = nil,
        topLogits: [CorrectnessTraceLogit]? = nil,
        topLogitRows: [[CorrectnessTraceLogit]]? = nil,
        seedToken: Int? = nil,
        tokens: [Int]? = nil,
        expertStats: ExpertStreamingStats? = nil,
        peakRamGB: Double? = nil
    ) {
        self.id = id
        self.nonce = nonce
        self.ok = ok
        self.error = error
        self.token = token
        self.topLogits = topLogits
        self.topLogitRows = topLogitRows
        self.seedToken = seedToken
        self.tokens = tokens
        self.expertStats = expertStats
        self.peakRamGB = peakRamGB
    }

    enum CodingKeys: String, CodingKey {
        case id
        case nonce
        case ok
        case error
        case token
        case topLogits = "top_logits"
        case topLogitRows = "top_logit_rows"
        case seedToken = "seed_token"
        case tokens
        case expertStats = "expert_stats"
        case peakRamGB = "peak_ram_gb"
    }
}

final class RuntimeWorkerProtocolIO {
    private let input: FileHandle
    private let output: FileHandle

    private init(inputDescriptor: Int32, outputDescriptor: Int32) {
        self.input = FileHandle(fileDescriptor: inputDescriptor, closeOnDealloc: true)
        self.output = FileHandle(fileDescriptor: outputDescriptor, closeOnDealloc: true)
    }

    static func isolatingStandardIO() throws -> RuntimeWorkerProtocolIO {
        let inputFD = try duplicatePrivateDescriptor(STDIN_FILENO, label: "stdin")
        let outputFD = try duplicatePrivateDescriptor(STDOUT_FILENO, label: "stdout")
        do {
            try redirectDescriptorToDevNull(STDIN_FILENO, flags: O_RDONLY, label: "stdin")
            try redirectDescriptorToDevNull(STDOUT_FILENO, flags: O_WRONLY, label: "stdout")
        } catch {
            close(inputFD)
            close(outputFD)
            throw error
        }
        return RuntimeWorkerProtocolIO(inputDescriptor: inputFD, outputDescriptor: outputFD)
    }

    func readLine() throws -> String? {
        var data = Data()
        while true {
            let byte = input.readData(ofLength: 1)
            if byte.isEmpty {
                if data.isEmpty {
                    return nil
                }
                break
            }
            if byte[byte.startIndex] == 0x0a {
                break
            }
            data.append(byte)
        }
        guard let line = String(data: data, encoding: .utf8) else {
            throw MLXFastError.invalidInput("runtime worker received non-UTF8 protocol input")
        }
        return line
    }

    func writeLine(_ data: Data) throws {
        try output.write(contentsOf: data)
        try output.write(contentsOf: Data([0x0a]))
    }
}

func duplicatePrivateDescriptor(_ descriptor: Int32, label: String) throws -> Int32 {
    let lowerBound = Int32(64 + Int(arc4random_uniform(449)))
    let duplicatedFD = fcntl(descriptor, F_DUPFD_CLOEXEC, lowerBound)
    guard duplicatedFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to duplicate \(label) for protocol I/O")
    }
    return duplicatedFD
}

func redirectDescriptorToDevNull(_ descriptor: Int32, flags: Int32, label: String) throws {
    let devNullFD = open("/dev/null", flags)
    guard devNullFD >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to open /dev/null for \(label) redirection")
    }
    defer {
        close(devNullFD)
    }
    guard dup2(devNullFD, descriptor) >= 0 else {
        throw MLXFastError.invalidInput("runtime worker failed to redirect \(label) away from protocol I/O")
    }
}

/// Continuously drains a runtime worker's stderr pipe on a background thread,
/// forwarding each completed line to `emit` (prefixed and token-redacted) and
/// keeping a capped raw tail for the exit diagnostic. Local modes attach this
/// so participants' debug prints in model code show up live during the edit
/// loop; it also means a chatty worker can no longer fill the undrained pipe
/// buffer and stall the run. Official runs never attach it (the CLI forces
/// forwardsWorkerStderr off there), so their stderr handling is unchanged.
final class WorkerStderrDrain: @unchecked Sendable {
    private let handle: FileHandle
    private let emit: (String) -> Void
    private let lock = NSLock()
    private var tail = Data()
    private var pendingLine = Data()
    private let finished = DispatchSemaphore(value: 0)
    static let tailByteLimit = 64 * 1024
    static let forwardedLinePrefix = "mlxfast-worker: "

    init(
        handle: FileHandle,
        emit: ((String) -> Void)? = nil
    ) {
        self.handle = handle
        self.emit = emit ?? { line in
            fputs(line, stderr)
            fflush(stderr)
        }
        let thread = Thread { [self] in
            drainToEOF()
            finished.signal()
        }
        thread.name = "mlxfast.worker-stderr-drain"
        thread.start()
    }

    /// Blocks until the reader thread hits EOF (the worker exited and all
    /// output was ingested) or the timeout passes, then returns the raw tail
    /// for the exit diagnostic (which applies its own sanitization).
    func drainedOutput(timeoutSeconds: Double) -> String {
        if finished.wait(timeout: .now() + timeoutSeconds) == .success {
            // Re-signal so later calls (or repeated diagnostics) do not block.
            finished.signal()
        }
        lock.lock()
        defer {
            lock.unlock()
        }
        var data = tail
        data.append(pendingLine)
        return String(decoding: data, as: UTF8.self)
    }

    private func drainToEOF() {
        while true {
            let chunk = handle.readData(ofLength: 8192)
            if chunk.isEmpty {
                break
            }
            ingest(chunk)
        }
        flushPendingLine()
    }

    private func ingest(_ chunk: Data) {
        var completedLines: [String] = []
        lock.lock()
        pendingLine.append(chunk)
        while let newlineIndex = pendingLine.firstIndex(of: 0x0a) {
            let lineLength = pendingLine.distance(from: pendingLine.startIndex, to: newlineIndex)
            let lineData = Data(pendingLine.prefix(lineLength))
            pendingLine = Data(pendingLine.dropFirst(lineLength + 1))
            appendToTailLocked(lineData + Data([0x0a]))
            completedLines.append(String(decoding: lineData, as: UTF8.self))
        }
        lock.unlock()
        for line in completedLines {
            emitLine(line)
        }
    }

    private func flushPendingLine() {
        lock.lock()
        let remainder = pendingLine
        pendingLine = Data()
        if !remainder.isEmpty {
            appendToTailLocked(remainder + Data([0x0a]))
        }
        lock.unlock()
        if !remainder.isEmpty {
            emitLine(String(decoding: remainder, as: UTF8.self))
        }
    }

    private func appendToTailLocked(_ data: Data) {
        tail.append(data)
        if tail.count > Self.tailByteLimit {
            tail = Data(tail.suffix(Self.tailByteLimit))
        }
    }

    private func emitLine(_ line: String) {
        emit("\(Self.forwardedLinePrefix)\(redactedWorkerStderrLine(line))\n")
    }
}

/// Per-line redaction for forwarded worker stderr: worker output comes from
/// submitted model code that has seen the (possibly private) golden, so lines
/// that look like token comparisons are collapsed exactly like the shared
/// error-path redaction.
func redactedWorkerStderrLine(_ line: String) -> String {
    if line.range(of: "expected", options: .caseInsensitive) != nil
        || line.range(of: "actual", options: .caseInsensitive) != nil
    {
        return "token-validation-failed"
    }
    return line
}

final class RuntimeWorkerClient {
    private let process: Process
    private let input: FileHandle
    private let output: FileHandle
    private let errorOutput: FileHandle
    private let stderrDrain: WorkerStderrDrain?
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()
    private var sessionNonce = ""
    private var nextID = 1
    private var closed = false
    // Expert-streaming counters reported in the worker's protocol hello, before
    // any forward has run. Baseline for the seed prefill's incremental reads.
    private(set) var initialExpertStats: ExpertStreamingStats?

    init(options: RuntimeWorkerOptions, weightsPath: String) throws {
        let process = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()
        if let sandboxProfilePath = options.sandboxProfilePath {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = [
                "-f",
                sandboxProfilePath,
                options.executablePath,
                "runtime-worker",
                "--weights",
                weightsPath,
            ]
        } else {
            process.executableURL = URL(fileURLWithPath: options.executablePath)
            process.arguments = [
                "runtime-worker",
                "--weights",
                weightsPath,
            ]
        }
        process.environment = sanitizedRuntimeWorkerEnvironment(ProcessInfo.processInfo.environment)
        process.standardInput = stdin
        process.standardOutput = stdout
        process.standardError = stderr
        try process.run()

        self.process = process
        self.input = stdin.fileHandleForWriting
        self.output = stdout.fileHandleForReading
        self.errorOutput = stderr.fileHandleForReading
        // Attach the live drain before waiting for the protocol hello so even
        // output printed during model/weights setup streams immediately.
        self.stderrDrain = options.forwardsWorkerStderr
            ? WorkerStderrDrain(handle: stderr.fileHandleForReading)
            : nil
        let hello = try readResponseLine(validateNonce: false)
        guard hello.id == 0, hello.ok, let nonce = hello.nonce, !nonce.isEmpty else {
            throw MLXFastError.invalidInput("runtime worker did not return a valid protocol hello")
        }
        self.sessionNonce = nonce
        self.initialExpertStats = hello.expertStats
    }

    deinit {
        close()
    }

    func close() {
        guard !closed else {
            return
        }
        closed = true
        try? input.close()
        if process.isRunning {
            process.terminate()
            process.waitUntilExit()
        }
    }

    func generateCorrectness(promptTokens: [Int], steps: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness",
            promptTokens: promptTokens,
            steps: steps
        )
    }

    func beginTeacherForcedCorrectness(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_begin",
            promptTokens: promptTokens
        )
    }

    func teacherForcedCorrectnessStep(previousToken: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "correctness_step",
            token: previousToken
        )
    }

    func prefill(promptTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "prefill",
            promptTokens: promptTokens
        )
    }

    func beginDecode(seedTokens: [Int]) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_begin",
            seedTokens: seedTokens
        )
    }

    func decodeStep(inputToken: Int) throws -> RuntimeWorkerResponse {
        try send(
            kind: "decode_step",
            token: inputToken
        )
    }

    func phaseDiagnostics() throws -> RuntimeWorkerResponse {
        try send(kind: "phase_diagnostics")
    }

    private func send(
        kind: String,
        promptTokens: [Int]? = nil,
        token: Int? = nil,
        seedTokens: [Int]? = nil,
        steps: Int? = nil
    ) throws -> RuntimeWorkerResponse {
        guard process.isRunning else {
            throw MLXFastError.invalidInput("runtime worker exited before request \(kind): \(workerExitDiagnostic())")
        }
        let id = nextID
        nextID += 1
        let request = RuntimeWorkerRequest(
            id: id,
            kind: kind,
            promptTokens: promptTokens,
            token: token,
            seedTokens: seedTokens,
            steps: steps
        )
        var data = try encoder.encode(request)
        data.append(0x0a)
        try input.write(contentsOf: data)

        let response = try readResponseLine(validateNonce: true)
        guard response.id == id else {
            throw MLXFastError.invalidInput("runtime worker returned response id \(response.id), expected \(id)")
        }
        guard response.ok else {
            throw MLXFastError.invalidInput("runtime worker \(kind) failed: \(response.error ?? "unknown error")")
        }
        return response
    }

    private func readResponseLine(validateNonce: Bool) throws -> RuntimeWorkerResponse {
        while true {
            let data = try readWorkerOutputLine()
            guard runtimeWorkerLineLooksLikeJSONResponse(data) else {
                continue
            }
            let response = try decoder.decode(RuntimeWorkerResponse.self, from: data)
            if validateNonce, response.nonce != sessionNonce {
                throw MLXFastError.invalidInput("runtime worker returned a response with an invalid nonce")
            }
            return response
        }
    }

    private func readWorkerOutputLine() throws -> Data {
        var data = Data()
        while true {
            let byte = output.readData(ofLength: 1)
            if byte.isEmpty {
                throw MLXFastError.invalidInput(
                    "runtime worker closed stdout before returning a response: \(workerExitDiagnostic())"
                )
            }
            if byte[byte.startIndex] == 0x0a {
                return data
            }
            data.append(byte)
        }
    }

    private func workerExitDiagnostic() -> String {
        if process.isRunning {
            process.terminate()
        }
        process.waitUntilExit()
        // With a live drain attached, the pipe is consumed by its reader
        // thread; take the retained tail from there instead of racing it.
        let stderr = stderrDrain?.drainedOutput(timeoutSeconds: 2)
            ?? (String(data: errorOutput.readDataToEndOfFile(), encoding: .utf8) ?? "")
        let trimmed = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
        let redacted = sanitizeWorkerDiagnostic(trimmed)
        if redacted.isEmpty {
            return "exit_status=\(process.terminationStatus)"
        }
        return "exit_status=\(process.terminationStatus) stderr=\(redacted)"
    }

    private func sanitizeWorkerDiagnostic(_ value: String) -> String {
        let singleLine = value
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        if singleLine.range(of: "expected", options: .caseInsensitive) != nil
            || singleLine.range(of: "actual", options: .caseInsensitive) != nil
        {
            return "token-validation-failed"
        }
        return singleLine
    }
}

func sanitizedRuntimeWorkerEnvironment(_ environment: [String: String]) -> [String: String] {
    var sanitized = environment
    let blockedExactKeys: Set<String> = [
        "ANTHROPIC_API_KEY",
        "CI",
        "MLXFAST_ANTHROPIC_PRESENT",
        // The gates/timing parallel split reads these three to decide which
        // half of the original single-machine run this process covers. On one
        // machine, decode/prefill was ALWAYS timed at the same time gates were
        // checked, so there was no way for submitted code to tell "my speed
        // doesn't count right now" from "my correctness doesn't count right
        // now" -- these vars make exactly that distinction newly observable,
        // so they must be blocked the same way MLXFAST_RUN_BENCHMARK already is.
        "MLXFAST_BENCHMARK_CHECK_GATES",
        "MLXFAST_BENCHMARK_CORRECTNESS_STEPS",
        "MLXFAST_BENCHMARK_SKIP_TIMED",
        // The env-var forms of --base-case-only/--step-range are the slice
        // machines' equivalents of the three split-phase vars above: they let
        // submitted code detect "I am an unscored correctness slice checking
        // exactly steps N-M". Latent today (the slice workflow passes flags,
        // not env), but an operator using the documented env form must not
        // reopen the side channel.
        "MLXFAST_CORRECTNESS_BASE_CASE_ONLY",
        "MLXFAST_CORRECTNESS_STEP_RANGE",
        "MLXFAST_CORRECTNESS_GOLDEN_AUTH_HEADER",
        "MLXFAST_CORRECTNESS_GOLDEN_PATH",
        "MLXFAST_CORRECTNESS_GOLDEN_URL",
        "MLXFAST_FORCE_TRANSFORM",
        "MLXFAST_GPQA_REFERENCE_PATH",
        "MLXFAST_IN_SANDBOX",
        "MLXFAST_NO_SANDBOX",
        "MLXFAST_OFFICIAL_BENCHMARK_RUN",
        // Same-session measured baseline supplied by the trusted workflow's
        // paired-baseline step. Submitted model code must not observe the
        // reference implementation's live timings (or even that this run is
        // paired) from inside the sandboxed worker.
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN",
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN",
        "MLXFAST_PRIVATE_DIR",
        "MLXFAST_PRIVATE_PROMPTS_R2_PRESENT",
        "MLXFAST_REFERENCE_AUTH_HEADER",
        "MLXFAST_REFERENCE_BASE_URL",
        "MLXFAST_REFERENCE_DIR",
        "MLXFAST_RUN_BENCHMARK",
        "MLXFAST_RUNTIME_WORKER_EXECUTABLE",
        "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE",
        "MLXFAST_SEMANTIC_GPQA_MODEL",
        "MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH",
        "MLXFAST_SEMANTIC_GPQA_RESULTS_PATH",
        "MLXFAST_SKIP_TRANSFORM",
        "MLXFAST_SUBMISSION_REF",
        "MLXFAST_TRUSTED_BENCHMARK_REF",
        "MLXFAST_TRUSTED_BENCHMARK_WORKFLOW",
        "MLXFAST_TRUSTED_REPOSITORY",
        "MLXFAST_VERIFY_TRANSFORM",
        "R2_ACCESS_KEY_ID",
        "R2_BUCKET_ENDPOINT",
        "R2_SECRET_ACCESS_KEY",
    ]
    let blockedPrefixes = [
        "ACTIONS_",
        "BLACKSMITH_",
        "GITHUB_",
        "RUNNER_",
    ]
    for key in sanitized.keys where blockedExactKeys.contains(key)
        || blockedPrefixes.contains(where: { key.hasPrefix($0) })
    {
        sanitized.removeValue(forKey: key)
    }
    sanitized["MLXFAST_USE_RUNTIME_WORKER"] = "0"
    return sanitized
}

func generateRuntimeWorkerNonce() -> String {
    var bytes = [UInt8](repeating: 0, count: 16)
    bytes.withUnsafeMutableBytes { buffer in
        if let baseAddress = buffer.baseAddress {
            arc4random_buf(baseAddress, buffer.count)
        }
    }
    return bytes
        .map { String(format: "%02x", $0) }
        .joined()
}

func runtimeWorkerLineLooksLikeJSONResponse(_ data: Data) -> Bool {
    for byte in data where byte != 0x20 && byte != 0x09 && byte != 0x0d {
        return byte == 0x7b
    }
    return false
}

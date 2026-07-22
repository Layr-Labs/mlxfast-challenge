import Darwin
import Foundation
@testable import MLXFastCore
@testable import MLXFastModel
@testable import MLXFastRuntimeWorkerSupport
import Testing

@Test
func runtimeWorkerClientSkipsNonJSONStdoutLines() {
    #expect(runtimeWorkerLineLooksLikeJSONResponse(Data("  {\"id\":1,\"ok\":true}".utf8)))
    #expect(!runtimeWorkerLineLooksLikeJSONResponse(Data("Metal device initialized".utf8)))
    #expect(!runtimeWorkerLineLooksLikeJSONResponse(Data("".utf8)))
}

// metrics.commit must come from the trusted dispatch context when the ranked
// pipeline supplies it: on the ranked box the harness runs as the sandboxed
// bench uid where `git rev-parse` fails (dubious ownership in the runner-owned
// workspace copy), which produced empty commits that failed every ranked
// score's commit binding. git stays the local/dev fallback only.
@Test
func commitIdentifierPrefersTrustedDispatchSHAOverGit() {
    let fullSHA = "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f"
    #expect(GemmaRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": fullSHA]) == fullSHA)
    // Short (rev-parse --short style) values and surrounding whitespace are fine.
    #expect(GemmaRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": "5f95c4bdce07"]) == "5f95c4bdce07")
    #expect(GemmaRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": " \(fullSHA)\n"]) == fullSHA)

    // Values that could not satisfy the trusted shell predicates fall back to
    // git instead of being stamped verbatim into the sealed score.
    for invalid in [
        "",
        "5f95c4",
        fullSHA.uppercased(),
        fullSHA + "0",
        "not-a-commit-sha",
        "5f95c4bdce07;rm -rf /",
    ] {
        let fallback = GemmaRuntime.commitIdentifier(environment: ["MLXFAST_COMMIT_SHA": invalid])
        #expect(fallback != invalid || invalid.isEmpty)
        #expect(!fallback.contains(";"))
    }

    #expect(GemmaRuntime.isCommitSHAHex(fullSHA))
    #expect(GemmaRuntime.isCommitSHAHex("abcdef0"))
    #expect(!GemmaRuntime.isCommitSHAHex("abcdef"))
    #expect(!GemmaRuntime.isCommitSHAHex(String(repeating: "a", count: 41)))
    #expect(!GemmaRuntime.isCommitSHAHex("ABCDEF0"))
}

@Test
func runtimeWorkerPrivateDescriptorUsesPortableLowerBoundAndCloseOnExec() throws {
    let pipe = Pipe()
    let descriptor = try duplicatePrivateDescriptor(
        pipe.fileHandleForReading.fileDescriptor,
        label: "test"
    )
    defer { Darwin.close(descriptor) }

    #expect(descriptor >= STDERR_FILENO + 1)
    let flags = fcntl(descriptor, F_GETFD)
    #expect(flags >= 0)
    #expect(flags & FD_CLOEXEC == FD_CLOEXEC)
}

@Test
func runtimeWorkerProtocolDescriptorAllocationClosesInputWhenOutputFails() {
    var attempts: [String] = []
    var closed: [Int32] = []

    #expect(throws: RuntimeWorkerDescriptorTestError.self) {
        _ = try duplicateRuntimeWorkerProtocolDescriptors(
            inputDescriptor: STDIN_FILENO,
            outputDescriptor: STDOUT_FILENO,
            duplicate: { _, label in
                attempts.append(label)
                if label == "stdin" {
                    return 91
                }
                throw RuntimeWorkerDescriptorTestError.expected
            },
            closeDescriptor: { closed.append($0) }
        )
    }
    #expect(attempts == ["stdin", "stdout"])
    #expect(closed == [91])
}

private enum RuntimeWorkerDescriptorTestError: Error {
    case expected
}

@Test
func runtimeWorkerPinnedConfigurationAcceptsLagunaMoEArchitecture() throws {
    let data = try JSONSerialization.data(
        withJSONObject: pinnedRuntimeWorkerConfigurationObject()
    )
    try validateRuntimeWorkerPinnedConfigurationData(data)
}

@Test
func runtimeWorkerPinnedConfigurationRejectsNonLagunaArchitectures() throws {
    var cases: [(String, [String: Any])] = []

    func addCase(_ name: String, _ mutate: (inout [String: Any]) -> Void) {
        var object = pinnedRuntimeWorkerConfigurationObject()
        mutate(&object)
        cases.append((name, object))
    }

    // A dense Gemma-4-shaped config (the previous pinned target) must be
    // rejected wholesale, not merely on the model_type string.
    addCase("model-type") { $0["model_type"] = "gemma4_text" }
    addCase("hidden-size") { $0["hidden_size"] = 5_376 }
    addCase("hidden-layers") { $0["num_hidden_layers"] = MLXFastConstants.numHiddenLayers - 1 }
    addCase("intermediate-size") { $0["intermediate_size"] = MLXFastConstants.intermediateSize - 1 }
    addCase("attention-heads") { $0["num_attention_heads"] = 64 }
    addCase("uniform-heads-per-layer") {
        $0["num_attention_heads_per_layer"] =
            [Int](repeating: 48, count: MLXFastConstants.numHiddenLayers)
    }
    addCase("short-heads-per-layer") {
        $0["num_attention_heads_per_layer"] =
            [Int](repeating: 48, count: MLXFastConstants.numHiddenLayers - 1)
    }
    addCase("missing-heads-per-layer") {
        $0.removeValue(forKey: "num_attention_heads_per_layer")
    }
    addCase("kv-heads") { $0["num_key_value_heads"] = 16 }
    addCase("head-dim") { $0["head_dim"] = 256 }
    addCase("rms-norm") { $0["rms_norm_eps"] = 1e-5 }
    addCase("vocab") { $0["vocab_size"] = 262_144 }
    addCase("sliding-window") { $0["sliding_window"] = 1_024 }
    addCase("max-position") { $0["max_position_embeddings"] = 131_072 }
    addCase("attention-bias") { $0["attention_bias"] = true }
    addCase("qkv-bias") { $0["qkv_bias"] = true }
    addCase("attention-dropout") { $0["attention_dropout"] = 0.1 }
    addCase("per-element-gating") { $0["gating"] = "per-element" }
    addCase("disabled-gating") { $0["gating"] = false }
    // Laguna's lm_head is untied; a tied-embedding config is a different
    // (dense-era) checkpoint layout.
    addCase("tie-embeddings") { $0["tie_word_embeddings"] = true }
    addCase("moe-disabled") { $0["num_experts"] = 0 }
    addCase("experts") { $0["num_experts"] = 8 }
    addCase("top-k-experts") { $0["num_experts_per_tok"] = 2 }
    addCase("moe-intermediate") { $0["moe_intermediate_size"] = 1_024 }
    addCase("shared-expert-intermediate") {
        $0["shared_expert_intermediate_size"] = 1_024
    }
    addCase("routed-scaling") { $0["moe_routed_scaling_factor"] = 1.0 }
    addCase("norm-topk") { $0["norm_topk_prob"] = false }
    addCase("router-softcap") { $0["moe_router_logit_softcapping"] = 30.0 }
    addCase("missing-layer-types") { $0.removeValue(forKey: "layer_types") }
    addCase("layer-pattern") {
        // The dense Gemma 4 schedule (full attention every 6th layer) is
        // not Laguna's every-4th-full schedule.
        $0["layer_types"] = (0..<MLXFastConstants.numHiddenLayers).map { index in
            index % 6 == 5 ? "full_attention" : "sliding_attention"
        }
    }
    addCase("all-sparse-mlp") {
        $0["mlp_layer_types"] =
            [String](repeating: "sparse", count: MLXFastConstants.numHiddenLayers)
    }
    addCase("mlp-only-layers") { $0["mlp_only_layers"] = [0, 1] }
    addCase("decoder-sparse-step") { $0["decoder_sparse_step"] = 2 }
    addCase("sliding-rope-theta") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var sliding = rope["sliding_attention"] as! [String: Any]
        sliding["rope_theta"] = 20_000
        rope["sliding_attention"] = sliding
        $0["rope_parameters"] = rope
    }
    addCase("sliding-rope-type") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var sliding = rope["sliding_attention"] as! [String: Any]
        sliding["rope_type"] = "yarn"
        rope["sliding_attention"] = sliding
        $0["rope_parameters"] = rope
    }
    addCase("sliding-partial-rotary") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var sliding = rope["sliding_attention"] as! [String: Any]
        sliding["partial_rotary_factor"] = 0.5
        rope["sliding_attention"] = sliding
        $0["rope_parameters"] = rope
    }
    addCase("full-rope-theta") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var full = rope["full_attention"] as! [String: Any]
        full["rope_theta"] = 10_000
        rope["full_attention"] = full
        $0["rope_parameters"] = rope
    }
    addCase("full-rope-type") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var full = rope["full_attention"] as! [String: Any]
        full["rope_type"] = "default"
        rope["full_attention"] = full
        $0["rope_parameters"] = rope
    }
    addCase("full-partial-rotary") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var full = rope["full_attention"] as! [String: Any]
        full["partial_rotary_factor"] = 1.0
        rope["full_attention"] = full
        $0["rope_parameters"] = rope
    }
    addCase("yarn-factor") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var full = rope["full_attention"] as! [String: Any]
        full["factor"] = 8.0
        rope["full_attention"] = full
        $0["rope_parameters"] = rope
    }
    addCase("yarn-original-max-position") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var full = rope["full_attention"] as! [String: Any]
        full["original_max_position_embeddings"] = 4_096
        rope["full_attention"] = full
        $0["rope_parameters"] = rope
    }
    addCase("yarn-beta-fast") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var full = rope["full_attention"] as! [String: Any]
        full["beta_fast"] = 32.0
        rope["full_attention"] = full
        $0["rope_parameters"] = rope
    }
    addCase("yarn-beta-slow") {
        var rope = $0["rope_parameters"] as! [String: Any]
        var full = rope["full_attention"] as! [String: Any]
        full["beta_slow"] = 2.0
        rope["full_attention"] = full
        $0["rope_parameters"] = rope
    }
    addCase("quantization-bits") {
        var quantization = $0["quantization"] as! [String: Any]
        quantization["bits"] = 8
        $0["quantization"] = quantization
    }
    addCase("quantization-group") {
        var quantization = $0["quantization"] as! [String: Any]
        quantization["group_size"] = 32
        $0["quantization"] = quantization
    }
    addCase("quantization-mode") {
        var quantization = $0["quantization"] as! [String: Any]
        quantization["mode"] = "symmetric"
        $0["quantization"] = quantization
    }
    addCase("missing-quantization") { $0.removeValue(forKey: "quantization") }

    for (name, object) in cases {
        let data = try JSONSerialization.data(withJSONObject: object)
        #expect(throws: MLXFastError.self, "case \(name)") {
            try validateRuntimeWorkerPinnedConfigurationData(data)
        }
    }
}

@Test
func runtimeWorkerPinnedConfigurationAcceptsSafeOptionalRepresentations() throws {
    // The alternate spelling: quantization under quantization_config, the
    // boolean gating form, and every optional field absent (the validator
    // treats absence as the pinned default).
    var object = pinnedRuntimeWorkerConfigurationObject()
    object["quantization_config"] = object.removeValue(forKey: "quantization")
    object["gating"] = true
    for optionalKey in [
        "attention_bias",
        "qkv_bias",
        "attention_dropout",
        "moe_routed_scaling_factor",
        "norm_topk_prob",
        "moe_router_logit_softcapping",
        "mlp_layer_types",
        "mlp_only_layers",
        "decoder_sparse_step",
    ] {
        object.removeValue(forKey: optionalKey)
    }
    var rope = object["rope_parameters"] as! [String: Any]
    var sliding = rope["sliding_attention"] as! [String: Any]
    var full = rope["full_attention"] as! [String: Any]
    sliding.removeValue(forKey: "partial_rotary_factor")
    for yarnKey in [
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "attention_factor",
    ] {
        full.removeValue(forKey: yarnKey)
    }
    rope["sliding_attention"] = sliding
    rope["full_attention"] = full
    object["rope_parameters"] = rope

    try validateRuntimeWorkerPinnedConfigurationData(
        JSONSerialization.data(withJSONObject: object)
    )

    // Missing gating is also the pinned per-head default.
    var withoutGating = pinnedRuntimeWorkerConfigurationObject()
    withoutGating.removeValue(forKey: "gating")
    try validateRuntimeWorkerPinnedConfigurationData(
        JSONSerialization.data(withJSONObject: withoutGating)
    )
}

@Test
func runtimeWorkerPinnedConfigurationPathRejectsUnsafeFilesystemEntries() throws {
    let root = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: root) }
    let validData = try JSONSerialization.data(
        withJSONObject: pinnedRuntimeWorkerConfigurationObject()
    )

    let validDirectory = root.appendingPathComponent("valid", isDirectory: true)
    try FileManager.default.createDirectory(at: validDirectory, withIntermediateDirectories: true)
    try validData.write(to: validDirectory.appendingPathComponent("config.json"))
    try validateRuntimeWorkerPinnedConfiguration(weightsPath: validDirectory.path)

    let symlinkTarget = root.appendingPathComponent("config-target.json")
    try validData.write(to: symlinkTarget)
    let symlinkDirectory = root.appendingPathComponent("symlink", isDirectory: true)
    try FileManager.default.createDirectory(at: symlinkDirectory, withIntermediateDirectories: true)
    try FileManager.default.createSymbolicLink(
        at: symlinkDirectory.appendingPathComponent("config.json"),
        withDestinationURL: symlinkTarget
    )

    let directoryEntry = root.appendingPathComponent("directory-entry", isDirectory: true)
    try FileManager.default.createDirectory(
        at: directoryEntry.appendingPathComponent("config.json", isDirectory: true),
        withIntermediateDirectories: true
    )

    let emptyDirectory = root.appendingPathComponent("empty", isDirectory: true)
    try FileManager.default.createDirectory(at: emptyDirectory, withIntermediateDirectories: true)
    try Data().write(to: emptyDirectory.appendingPathComponent("config.json"))

    let oversizedDirectory = root.appendingPathComponent("oversized", isDirectory: true)
    try FileManager.default.createDirectory(at: oversizedDirectory, withIntermediateDirectories: true)
    var oversizedData = validData
    oversizedData.append(Data(repeating: 0x20, count: 1 * 1024 * 1024 + 1))
    try oversizedData.write(to: oversizedDirectory.appendingPathComponent("config.json"))

    for directory in [symlinkDirectory, directoryEntry, emptyDirectory, oversizedDirectory] {
        #expect(throws: MLXFastError.self, "unsafe config at \(directory.lastPathComponent)") {
            try validateRuntimeWorkerPinnedConfiguration(weightsPath: directory.path)
        }
    }
}

// The worker environment filter is a strict ALLOWLIST: submitted model code
// executes inside the worker and can read its whole environment, so every
// harness/CI/phase/identity variable must be dropped -- not just a hand-picked
// denylist. Any keep-by-default behavior is a regression: a variable whose
// value differs between the correctness/gates pass and the timed pass (e.g.
// MLXFAST_NOTE, MLXFAST_SCORE_PATH, MLXFAST_INTEGRITY_PATH) is a phase oracle
// that lets a submission serve a correct path while checked and a cheaper
// path while timed.
@Test
func runtimeWorkerEnvironmentDropsEverythingOutsideTheAllowlist() {
    let sanitized = sanitizedRuntimeWorkerEnvironment([
        // Confirmed phase-distinct leaks under the previous denylist: these
        // differ between the gates pass and the timed pass (or exist in only
        // one of them).
        "MLXFAST_NOTE": "ci workflow_dispatch 5f95c4bdce07 mode=single-machine-gates",
        "MLXFAST_SCORE_PATH": "score.gates.json",
        "MLXFAST_INTEGRITY_PATH": "benchmark-integrity.gates.json",
        "MLXFAST_SEMANTIC_GPQA_CASE_COUNT": "8",
        "MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS": "512",
        // Official-run identity: same value in both phases, but submitted
        // code must not observe that it runs under the ranked pipeline.
        "MLXFAST_COMMIT_SHA": "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f",
        "MLXFAST_CANDIDATE_SHA": "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f",
        "GITHUB_SHA": "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f",
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITHUB_RUN_ID": "123",
        "RUNNER_TEMP": "/tmp/runner",
        "BLACKSMITH_RUNNER": "1",
        // An UNKNOWN future harness variable must be dropped by default;
        // this is the property the old denylist could not provide.
        "MLXFAST_FUTURE_PHASE_MARKER": "x",
        // Trusted-shell wiring around the bench bridge, not for the worker.
        "BENCH_GOLDEN_PATH": "/ws/correctness_golden_ranked.json",
        "BENCH_JOB_ID": "ranked-123-1-gates",
        "MJOB_WS": "/Users/Shared/bench-jobs/ranked-123-1",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "safe.directory",
        "GIT_CONFIG_VALUE_0": "*",
        // Secrets and private-material locations.
        "ANTHROPIC_API_KEY": "secret",
        "R2_ACCESS_KEY_ID": "key",
        "MLXFAST_PRIVATE_DIR": "/private/golden",
        "MLXFAST_CORRECTNESS_GOLDEN_PATH": "/private/golden/correctness_golden.json",
        // Harness configuration the worker does not read (weights arrive via
        // argv; the byte cap is enforced in the trusted parent).
        "MLXFAST_OFFICIAL_BENCHMARK_RUN": "1",
        "MLXFAST_RUN_BENCHMARK": "1",
        "MLXFAST_BENCHMARK_CHECK_GATES": "1",
        "MLXFAST_BENCHMARK_SKIP_TIMED": "1",
        "MLXFAST_BENCHMARK_CORRECTNESS_STEPS": "64",
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        "MLXFAST_REFERENCE_DIR": "/private/reference",
        "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE": "/tmp/profile.sb",
        "MLXFAST_RUNTIME_WORKER_EXECUTABLE": "/ws/.build/release/mlxfast-runtime-worker",
        "MLXFAST_WEIGHTS_PATH": "weights",
        "MLXFAST_MAX_WEIGHTS_BYTES": "42",
        // Allowlisted entries that must survive with their exact values.
        "PATH": "/usr/bin:/bin",
        "HOME": "/Users/bench",
        "TMPDIR": "/var/folders/xx/T/",
        "USER": "bench",
        "LOGNAME": "bench",
        "SHELL": "/bin/zsh",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TERM": "xterm-256color",
        // A live SSH agent socket must be dropped: the worker never needs it
        // and forwarding it hands submitted code a usable auth channel wherever
        // the parent holds one.
        "SSH_AUTH_SOCK": "/tmp/ssh-agent.sock",
        "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DYLD_LIBRARY_PATH": "/opt/lib",
        "MTL_SHADER_VALIDATION": "0",
        "METAL_DEVICE_WRAPPER_TYPE": "0",
        "MLX_DISABLE_COMPILE": "1",
        "MLX_RESOURCE_LIMIT": "17179869184",
        "DARKBLOOM_COMPILED_DECODE": "1",
    ])

    // NOTHING outside the allowlist survives -- asserted structurally, not
    // via a hand-picked list, so a future leak cannot slip between cases.
    let expected: [String: String] = [
        "PATH": "/usr/bin:/bin",
        "HOME": "/Users/bench",
        "TMPDIR": "/var/folders/xx/T/",
        "USER": "bench",
        "LOGNAME": "bench",
        "SHELL": "/bin/zsh",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "en_US.UTF-8",
        "TERM": "xterm-256color",
        // SSH_AUTH_SOCK is intentionally absent: it is supplied above but must
        // be dropped by the allowlist.
        "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DYLD_LIBRARY_PATH": "/opt/lib",
        "MTL_SHADER_VALIDATION": "0",
        "METAL_DEVICE_WRAPPER_TYPE": "0",
        "MLX_DISABLE_COMPILE": "1",
        "MLX_RESOURCE_LIMIT": "17179869184",
        "DARKBLOOM_COMPILED_DECODE": "1",
        "MLXFAST_USE_RUNTIME_WORKER": "0",
    ]
    #expect(sanitized == expected)
}

// "MLX_"-prefixed tuning knobs are allowlisted, but that prefix must never
// admit harness "MLXFAST_*" names ("MLXFAST_" does not start with "MLX_").
@Test
func runtimeWorkerEnvironmentAllowlistPrefixDoesNotAdmitHarnessNames() {
    let sanitized = sanitizedRuntimeWorkerEnvironment([
        "MLX_METAL_FAST_SYNCH": "1",
        "MLXFAST_NOTE": "ci mode=single-machine-gates",
        "MLXFAST_SCORE_PATH": "score.gates.json",
        "MLXFAST_FUTURE_PHASE_MARKER": "x",
    ])
    #expect(sanitized["MLX_METAL_FAST_SYNCH"] == "1")
    #expect(sanitized["MLXFAST_NOTE"] == nil)
    #expect(sanitized["MLXFAST_SCORE_PATH"] == nil)
    #expect(sanitized["MLXFAST_FUTURE_PHASE_MARKER"] == nil)
    #expect(sanitized == [
        "MLX_METAL_FAST_SYNCH": "1",
        "MLXFAST_USE_RUNTIME_WORKER": "0",
    ])
}

// The property that kills the phase oracle: the gates pass and the timed pass
// hand the worker BYTE-IDENTICAL environments, so submitted code cannot tell
// which pipeline phase it is running in. The two parent environments below
// differ exactly the way the ranked pipeline's do (benchmark.yml gates step
// vs. measure-job's timed invocation); shared host basics are identical
// because both phases go through the same bench-exec bridge.
@Test
func runtimeWorkerEnvironmentIsIdenticalAcrossPipelinePhases() {
    let sharedHostEnvironment: [String: String] = [
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": "/Users/bench",
        "TMPDIR": "/var/folders/bench/T/",
        "USER": "bench",
        "LOGNAME": "bench",
        "SHELL": "/bin/bash",
        "__CF_USER_TEXT_ENCODING": "0x1F5:0x0:0x0",
        "MLXFAST_OFFICIAL_BENCHMARK_RUN": "1",
        "MLXFAST_USE_RUNTIME_WORKER": "1",
        "MLXFAST_RUNTIME_WORKER_EXECUTABLE": "/ws/.build/release/mlxfast-runtime-worker",
        "MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE": "/tmp/mlxfast-runtime-worker.gates.sb",
        "MLXFAST_REFERENCE_DIR": "/opt/bench-runner/cache/reference",
        "MLXFAST_COMMIT_SHA": "5f95c4bdce07a0ef79ea350c91d9eb0d7476cf2f",
        "MLXFAST_SKIP_TRANSFORM": "1",
        "MJOB_WS": "/Users/Shared/bench-jobs/ranked-123-1",
    ]

    var gatesEnvironment = sharedHostEnvironment
    gatesEnvironment["MLXFAST_NOTE"] = "ci workflow_dispatch 5f95c4bdce07 mode=single-machine-gates"
    gatesEnvironment["MLXFAST_SCORE_PATH"] = "score.gates.json"
    gatesEnvironment["MLXFAST_INTEGRITY_PATH"] = "benchmark-integrity.gates.json"
    gatesEnvironment["MLXFAST_SEMANTIC_GPQA_CASE_COUNT"] = "5"
    gatesEnvironment["MLXFAST_SEMANTIC_GPQA_MAX_NEW_TOKENS"] = "64"
    gatesEnvironment["MLXFAST_SEMANTIC_GPQA_OUTPUT_PATH"] = "/ws/private/semantic_gpqa_answers.json"
    gatesEnvironment["MLXFAST_BENCHMARK_CHECK_GATES"] = "1"
    gatesEnvironment["MLXFAST_BENCHMARK_SKIP_TIMED"] = "1"
    gatesEnvironment["MLXFAST_BENCHMARK_CORRECTNESS_STEPS"] = "64"
    gatesEnvironment["MLXFAST_CORRECTNESS_GOLDEN_PATH"] = "correctness_golden_ranked.json"
    gatesEnvironment["MLXFAST_WEIGHTS_PATH"] = "weights"
    gatesEnvironment["MLXFAST_PRIVATE_DIR"] = "/ws/private"
    gatesEnvironment["BENCH_GOLDEN_PATH"] = "/ws/correctness_golden_ranked.json"
    gatesEnvironment["BENCH_JOB_ID"] = "ranked-123-1-gates"
    gatesEnvironment["GIT_CONFIG_COUNT"] = "1"
    gatesEnvironment["GIT_CONFIG_KEY_0"] = "safe.directory"
    gatesEnvironment["GIT_CONFIG_VALUE_0"] = "*"

    var timedEnvironment = sharedHostEnvironment
    timedEnvironment["MLXFAST_SCORE_PATH"] = "score.mjob.json"
    timedEnvironment["MLXFAST_INTEGRITY_PATH"] = "benchmark-integrity.mjob.json"
    timedEnvironment["MLXFAST_RUNTIME_WORKER_SANDBOX_PROFILE"] = "/tmp/mlxfast-runtime-worker.timed.sb"
    timedEnvironment["BENCH_GOLDEN_PATH"] = "/ws/bench_oracle.json"
    timedEnvironment["BENCH_JOB_ID"] = "ranked-123-1-timing"

    let gatesWorkerEnvironment = sanitizedRuntimeWorkerEnvironment(gatesEnvironment)
    let timedWorkerEnvironment = sanitizedRuntimeWorkerEnvironment(timedEnvironment)
    #expect(gatesWorkerEnvironment == timedWorkerEnvironment)
    #expect(gatesWorkerEnvironment["MLXFAST_USE_RUNTIME_WORKER"] == "0")
    #expect(gatesWorkerEnvironment["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin")
    #expect(gatesWorkerEnvironment["HOME"] == "/Users/bench")
}

@Test
func pairedBaselineOverrideParsesTrustedEnvironmentFailClosed() throws {
    // Absent entirely: no pairing, callers fall back to golden/constants.
    #expect(try PairedBaselineOverride.fromEnvironment([:]) == nil)
    #expect(try PairedBaselineOverride.fromEnvironment([
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "  ",
    ]) == nil)

    // Present: both values parsed precisely.
    let override = try #require(try PairedBaselineOverride.fromEnvironment([
        "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.16518489738085937",
        "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6977511595078125",
    ]))
    #expect(override.prefillSecondsPerToken == 0.16518489738085937)
    #expect(override.decodeSecondsPerToken == 3.6977511595078125)

    // Half-set pairs and non-positive/non-finite values are operator wiring
    // errors: fail closed rather than silently repricing against constants.
    for badEnvironment: [String: String] in [
        ["MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17"],
        ["MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6"],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "0.17",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "-1",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "inf",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
        [
            "MLXFAST_PAIRED_BASELINE_PREFILL_SECONDS_PER_TOKEN": "fast",
            "MLXFAST_PAIRED_BASELINE_DECODE_SECONDS_PER_TOKEN": "3.6",
        ],
    ] {
        #expect(throws: MLXFastError.self) {
            _ = try PairedBaselineOverride.fromEnvironment(badEnvironment)
        }
    }
}

@Test
func semanticBehaviorGateRequiresPromptAndReferenceAnswer() {
    let exactOnly = GoldenBehaviorCase(
        name: "exact-only",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1
    )
    let missingReference = GoldenBehaviorCase(
        name: "missing-reference",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1,
        semanticPrompt: "question"
    )
    let semantic = GoldenBehaviorCase(
        name: "semantic",
        promptTokens: [1],
        acceptedTokenSequences: [[2]],
        maxNewTokens: 1,
        semanticPrompt: "question",
        semanticReferenceAnswer: "answer"
    )

    #expect(!GemmaRuntime.behaviorUsesSemanticJudge(exactOnly))
    #expect(!GemmaRuntime.behaviorUsesSemanticJudge(missingReference))
    #expect(GemmaRuntime.behaviorUsesSemanticJudge(semantic))
}

@Test
func correctnessAcceptsOnlyExactTopLogitTies() {
    #expect(correctnessTokenAccepted(
        expectedToken: 30,
        actualToken: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 19.25),
            CorrectnessTraceLogit(token: 30, logit: 19.25),
        ]
    ))
    #expect(!correctnessTokenAccepted(
        expectedToken: 30,
        actualToken: 1,
        topLogits: [
            CorrectnessTraceLogit(token: 1, logit: 19.25),
            CorrectnessTraceLogit(token: 30, logit: 19.0),
        ]
    ))
    #expect(!correctnessTokenAccepted(expectedToken: 30, actualToken: 1, topLogits: nil))
}

@Test
func failedScoreRedactsCorrectnessTokenMismatchByDefault() {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 7,
        caseCount: 1,
        firstFailingCase: "local-iterate",
        firstFailingStep: 6,
        expectedToken: 123,
        actualToken: 456,
        goldenHash: "golden",
        error: "token mismatch"
    )

    let payload = GemmaRuntime.failedScore(
        error: "token mismatch",
        correctness: report,
        passedCorrectness: false,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(payload.passed == false)
    #expect(payload.metrics.runtime == "swift-local-iterate")
    #expect(payload.metrics.firstFailingCase == "local-iterate")
    #expect(payload.metrics.firstFailingStep == 6)
    #expect(payload.metrics.expectedToken == nil)
    #expect(payload.metrics.actualToken == nil)
    #expect(payload.metrics.bandwidthGBPerToken == 0)
}

@Test
func failedScorePreservesExplicitPublicMismatchTokensAndRuntimeLabel() {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 7,
        caseCount: 1,
        firstFailingCase: "local-iterate",
        firstFailingStep: 6,
        expectedToken: 123,
        actualToken: 456,
        goldenHash: "golden",
        error: "token mismatch"
    )

    let payload = GemmaRuntime.failedScore(
        error: "token mismatch",
        correctness: report,
        passedCorrectness: false,
        expectedToken: report.expectedToken,
        actualToken: report.actualToken,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(payload.passed == false)
    #expect(payload.metrics.runtime == "swift-local-iterate")
    #expect(payload.metrics.firstFailingCase == "local-iterate")
    #expect(payload.metrics.firstFailingStep == 6)
    #expect(payload.metrics.expectedToken == 123)
    #expect(payload.metrics.actualToken == 456)
}

@Test
func decodeTimingPlanStartsAfterSeedPrefill() throws {
    let plan = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 4)
    var offsets: [Int] = []
    for step in 0..<plan.decodeSteps {
        offsets.append(try plan.positionOffset(forDecodedStep: step))
    }

    #expect(offsets == [32, 33, 34, 35])
}

@Test
func decodeTimingPlanRejectsInvalidRanges() throws {
    #expect(throws: MLXFastError.self) {
        _ = try DecodeTimingPlan(seedTokenCount: 0, decodeSteps: 4)
    }
    #expect(throws: MLXFastError.self) {
        _ = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 0)
    }

    let plan = try DecodeTimingPlan(seedTokenCount: 32, decodeSteps: 4)
    #expect(throws: MLXFastError.self) {
        _ = try plan.positionOffset(forDecodedStep: 4)
    }
}

// The former editable decode-delay knob (Gemma4SubmissionControls
// .measuredDecodeDelayMilliseconds, read via
// GemmaRuntime.submissionValidationDelayMilliseconds) was model code invoked by
// trusted code ONLY on the scored decode path. Because submitted model code is
// editable, being invoked only while timed was itself a phase oracle. The hook
// is now removed entirely: the editable file is gone and no trusted harness
// code invokes it. See decodeMeasurementInvokesNoPhaseVaryingEditableHook for
// the structural phase-independence guard.
@Test
func editableDecodeDelayHookIsFullyRemoved() throws {
    #expect(!FileManager.default.fileExists(
        atPath: "Sources/MLXFastModel/Gemma4SubmissionControls.swift"
    ))
    let harnessDirectory = "Sources/MLXFastHarness"
    let harnessFiles = try FileManager.default.contentsOfDirectory(atPath: harnessDirectory)
        .filter { $0.hasSuffix(".swift") }
    for file in harnessFiles {
        let source = try String(contentsOfFile: "\(harnessDirectory)/\(file)", encoding: .utf8)
        #expect(!source.contains("submissionValidationDelayMilliseconds"))
        #expect(!source.contains("measuredDecodeDelayMilliseconds"))
    }
}

@Test
func benchmarkGoldenOracleFieldsAreValidatedOnLoad() throws {
    let prefill = Array(0..<MLXFastConstants.benchmarkPrefillPromptTokens)
    let seed = Array(0..<MLXFastConstants.benchmarkDecodeSeedTokens)
    let decode = Array(repeating: 9, count: MLXFastConstants.benchmarkDecodeSteps)
    let benchmark = BenchmarkGolden(
        prefillPromptTokens: prefill,
        expectedPrefillToken: 17,
        decodeSeedTokens: seed,
        expectedDecodeSeedToken: 23,
        expectedDecodeTokens: decode
    )
    try validateBenchmarkGolden(benchmark)

    #expect(benchmark.prefillPromptTokens == prefill)
    #expect(benchmark.expectedPrefillToken == 17)
    #expect(benchmark.decodeSeedTokens == seed)
    #expect(benchmark.expectedDecodeSeedToken == 23)
    #expect(benchmark.expectedDecodeTokens == decode)
}

@Test
func validateBenchmarkGoldenRejectsMalformedOracle() {
    #expect(throws: MLXFastError.self) {
        try validateBenchmarkGolden(BenchmarkGolden(
            prefillPromptTokens: [1],
            expectedPrefillToken: 7,
            decodeSeedTokens: Array(repeating: 1, count: MLXFastConstants.benchmarkDecodeSeedTokens),
            expectedDecodeSeedToken: 7,
            expectedDecodeTokens: Array(repeating: 7, count: MLXFastConstants.benchmarkDecodeSteps)
        ))
    }
}

@Test
func defaultTransformedWeightsLimitIsTwentyFiveGiB() {
    #expect(MLXFastConstants.defaultMaxTransformedWeightsBytes == 25 * 1024 * 1024 * 1024)
}

@Test
func benchmarkPreflightAcceptsRequiredArtifacts() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try checkPreflight(fixture)

    #expect(report.weightsPath == fixture.weights.path)
    #expect(report.goldenPath == fixture.golden.path)
    #expect(report.weightsByteCount > 0)
    #expect(report.maxWeightsByteCount == MLXFastConstants.defaultMaxTransformedWeightsBytes)
}

@Test
func benchmarkPreflightRejectsWeightsAboveDefaultByteLimit() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try writeSparseFile(
        fixture.weights.appendingPathComponent("oversized.bin"),
        byteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes + 1
    )

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightHonorsConfiguredWeightsByteLimit() throws {
    let fixture = try makePreflightFixture()
    defer { try? FileManager.default.removeItem(at: fixture.root) }
    try writeSparseFile(
        fixture.weights.appendingPathComponent("large-but-allowed.bin"),
        byteCount: MLXFastConstants.defaultMaxTransformedWeightsBytes + 1
    )
    let override = MLXFastConstants.defaultMaxTransformedWeightsBytes * 2

    let report = try checkPreflight(
        fixture,
        environment: [
            "MLXFAST_MAX_WEIGHTS_BYTES": "\(override)",
        ]
    )

    #expect(report.weightsByteCount > MLXFastConstants.defaultMaxTransformedWeightsBytes)
    #expect(report.maxWeightsByteCount == override)
}

@Test
func benchmarkPreflightRejectsMalformedGolden() throws {
    let fixture = try makePreflightFixture(goldenContents: "{}")
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: Error.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightRejectsShortBenchmarkPrompt() throws {
    let fixture = try makePreflightFixture(goldenContents: validGoldenJSON(benchmarkPromptTokens: [1]))
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func benchmarkPreflightRejectsMissingBenchmarkOracle() throws {
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    let fixture = try makePreflightFixture(goldenContents: """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "expected_tokens": \(expected)
        }
      ]
    }
    """)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

@Test
func correctnessPreflightAcceptsPublicGoldenWithoutBenchmarkOracle() throws {
    let fixture = try makePreflightFixture(goldenContents: correctnessOnlyGoldenJSON())
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let report = try checkCorrectnessPreflight(fixture)

    #expect(report.weightsPath == fixture.weights.path)
    #expect(report.goldenPath == fixture.golden.path)
}

@Test
func correctnessPreflightHonorsConfiguredWeightsByteLimit() throws {
    let fixture = try makePreflightFixture(goldenContents: correctnessOnlyGoldenJSON())
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkCorrectnessPreflight(
            fixture,
            environment: ["MLXFAST_MAX_WEIGHTS_BYTES": "1"]
        )
    }
}

@Test
func nonWorkerBenchmarkRejectsBehaviorGatesBecauseTTFTRequiresWorker() throws {
    let behaviorGate = """
    {
      "behavior": [
        {
          "name": "gpqa-hidden-ttft",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "accepted_token_sequences": [[7]],
          "max_new_tokens": 1,
          "semantic_prompt": "Hidden short-answer prompt",
          "semantic_reference_answer": "Reference answer"
        }
      ]
    }
    """
    let fixture = try makePreflightFixture(goldenContents: validGoldenJSON(correctnessGates: behaviorGate))
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    let score = GemmaRuntime.benchmark(
        BenchmarkOptions(
            weightsPath: fixture.weights.path,
            goldenPath: fixture.golden.path,
            correctnessSteps: 1,
            benchmarkDecodeSteps: 1
        ),
        worker: nil
    )

    #expect(score.passed == false)
    #expect(score.metrics.passedCorrectness == false)
    #expect(score.metrics.error == "benchmark behavior and GPQA TTFT gates require runtime worker timing")
    #expect(score.metrics.gpqaTTFTCaseCount == 0)
    #expect(score.metrics.preflightSeconds == 0)
    #expect(score.metrics.weightsByteCount == 0)
    #expect(score.metrics.weightsFileCount == 0)
}

@Test
func benchmarkPreflightRejectsMissingSemanticTensor() throws {
    let fixture = try makePreflightFixture(omitDenseTensorName: LagunaWeightNames.finalNorm)
    defer { try? FileManager.default.removeItem(at: fixture.root) }

    #expect(throws: MLXFastError.self) {
        _ = try checkPreflight(fixture)
    }
}

private struct PreflightFixture {
    let root: URL
    let weights: URL
    let golden: URL
}

private struct TensorFixture {
    let name: String
    let dtype: String
    let shape: [Int]
}

private func checkPreflight(
    _ fixture: PreflightFixture,
    environment: [String: String] = [:]
) throws -> BenchmarkPreflightReport {
    try BenchmarkPreflight.check(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: environment
    )
}

private func checkCorrectnessPreflight(
    _ fixture: PreflightFixture,
    environment: [String: String] = [:]
) throws -> BenchmarkPreflightReport {
    try BenchmarkPreflight.checkCorrectnessArtifacts(
        weightsPath: fixture.weights.path,
        goldenPath: fixture.golden.path,
        environment: environment
    )
}

private func makePreflightFixture(
    goldenContents: String? = nil,
    omitDenseTensorName: String? = nil
) throws -> PreflightFixture {
    let directory = try temporaryDirectory()
    let weights = directory.appendingPathComponent("weights", isDirectory: true)
    try FileManager.default.createDirectory(at: weights, withIntermediateDirectories: true)

    try lagunaPreflightConfigJSON().write(
        to: weights.appendingPathComponent("config.json"),
        atomically: true,
        encoding: .utf8
    )

    var denseTensors = requiredLagunaDenseTensorFixtures()
    if let omitDenseTensorName {
        denseTensors.removeAll { $0.name == omitDenseTensorName }
    }
    let denseShard = "model-00001.safetensors"
    try writeSafetensors(weights.appendingPathComponent(denseShard), tensors: denseTensors)
    try writeIndex(
        weights.appendingPathComponent("model.safetensors.index.json"),
        tensors: denseTensors,
        shardName: denseShard
    )

    let golden = directory.appendingPathComponent("correctness_golden.json")
    try (goldenContents ?? validGoldenJSON()).write(to: golden, atomically: true, encoding: .utf8)

    return PreflightFixture(root: directory, weights: weights, golden: golden)
}

/// The pinned Laguna runtime config as the transform writes it, with the
/// 8-bit router-gate quantization override present for every sparse layer
/// (the shared `pinnedRuntimeWorkerConfigurationObject()` carries a single
/// representative override; the preflight weight validation resolves the
/// expected router bit width per layer from these entries).
private func lagunaPreflightConfigJSON() throws -> String {
    var root = pinnedRuntimeWorkerConfigurationObject()
    var quantization = root["quantization"] as? [String: Any] ?? [:]
    for layerIndex in 1..<MLXFastConstants.numHiddenLayers {
        quantization["language_model.model.layers.\(layerIndex).mlp.gate.proj"] = [
            "group_size": 64,
            "bits": 8,
        ]
    }
    root["quantization"] = quantization
    let data = try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
    return String(decoding: data, as: UTF8.self)
}

private func validGoldenJSON(
    correctnessPromptTokens: [Int] = Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens),
    benchmarkPromptTokens: [Int] = Array(repeating: 1, count: MLXFastConstants.benchmarkPrefillPromptTokens),
    correctnessGates: String? = nil
) -> String {
    let correctnessPrompt = arrayJSON(correctnessPromptTokens)
    let benchmarkPrompt = arrayJSON(benchmarkPromptTokens)
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    let seed = arrayJSON(Array(benchmarkPromptTokens.prefix(MLXFastConstants.benchmarkDecodeSeedTokens)))
    let decode = arrayJSON(Array(repeating: 9, count: MLXFastConstants.benchmarkDecodeSteps))
    let gates = correctnessGates.map { ",\n  \"correctness_gates\": \($0)" } ?? ""
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(correctnessPrompt),
          "expected_tokens": \(expected)
        }
      ],
      "benchmark": {
        "prefill_prompt_tokens": \(benchmarkPrompt),
        "expected_prefill_token": 8,
        "decode_seed_tokens": \(seed),
        "expected_decode_seed_token": 7,
        "expected_decode_tokens": \(decode)
      }\(gates)
    }
    """
}

private func correctnessOnlyGoldenJSON() -> String {
    let expected = arrayJSON(Array(repeating: 7, count: MLXFastConstants.correctnessSteps))
    return """
    {
      "version": 1,
      "cases": [
        {
          "name": "preflight",
          "prompt_tokens": \(arrayJSON(Array(repeating: 1, count: MLXFastConstants.correctnessPromptTokens))),
          "expected_tokens": \(expected)
        }
      ]
    }
    """
}

/// Every tensor `LagunaWeightLoader.validateRequiredMetadata` requires for a
/// synthetic Laguna-shaped checkpoint built from the pinned geometry (40
/// layers, one full-attention layer per block of four, dense MLP at layer 0,
/// 256-expert MoE with a shared expert elsewhere, per-head attention
/// gating, untied lm_head). Real byte contents do not matter for these
/// preflight tests -- only declared dtype/shape -- so the shard is written
/// sparse: quantized projections use the packed `U32` affine layout the
/// real checkpoint ships (4-bit group-64 everywhere, 8-bit router gates)
/// with BF16 `.scales`/`.biases` companions.
private func requiredLagunaDenseTensorFixtures() -> [TensorFixture] {
    let hidden = MLXFastConstants.hiddenSize
    let denseIntermediate = MLXFastConstants.intermediateSize
    let vocab = MLXFastConstants.vocabSize
    let layers = MLXFastConstants.numHiddenLayers
    let kvHeads = 8
    let headDim = 128
    let experts = 256
    let expertIntermediate = 512
    let sharedExpertIntermediate = 512
    let groupSize = 64

    var tensors: [TensorFixture] = []

    func appendQuantized(
        _ name: String,
        leadingShape: [Int],
        inputFeatures: Int,
        bits: Int = 4
    ) {
        tensors.append(TensorFixture(
            name: name,
            dtype: "U32",
            shape: leadingShape + [inputFeatures * bits / 32]
        ))
        let stem = name.hasSuffix(".weight") ? String(name.dropLast(".weight".count)) : name
        let companionShape = leadingShape + [inputFeatures / groupSize]
        for suffix in ["scales", "biases"] {
            tensors.append(TensorFixture(
                name: "\(stem).\(suffix)",
                dtype: "BF16",
                shape: companionShape
            ))
        }
    }

    appendQuantized(LagunaWeightNames.embedTokens, leadingShape: [vocab], inputFeatures: hidden)
    tensors.append(TensorFixture(name: LagunaWeightNames.finalNorm, dtype: "BF16", shape: [hidden]))
    appendQuantized(LagunaWeightNames.lmHead, leadingShape: [vocab], inputFeatures: hidden)

    for layerIndex in 0..<layers {
        let layerHeads = layerIndex % 4 == 0 ? 48 : 64

        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
            tensors.append(TensorFixture(
                name: LagunaWeightNames.layer(layerIndex, suffix),
                dtype: "BF16",
                shape: [hidden]
            ))
        }

        appendQuantized(
            LagunaWeightNames.attention(layerIndex, "q_proj.weight"),
            leadingShape: [layerHeads * headDim],
            inputFeatures: hidden
        )
        for suffix in ["k_proj.weight", "v_proj.weight"] {
            appendQuantized(
                LagunaWeightNames.attention(layerIndex, suffix),
                leadingShape: [kvHeads * headDim],
                inputFeatures: hidden
            )
        }
        appendQuantized(
            LagunaWeightNames.attention(layerIndex, "o_proj.weight"),
            leadingShape: [hidden],
            inputFeatures: layerHeads * headDim
        )
        // Per-head attention gating: one gate per query head.
        appendQuantized(
            LagunaWeightNames.attention(layerIndex, "g_proj.weight"),
            leadingShape: [layerHeads],
            inputFeatures: hidden
        )
        for suffix in ["q_norm.weight", "k_norm.weight"] {
            tensors.append(TensorFixture(
                name: LagunaWeightNames.attention(layerIndex, suffix),
                dtype: "BF16",
                shape: [headDim]
            ))
        }

        if layerIndex == 0 {
            for suffix in ["gate_proj.weight", "up_proj.weight"] {
                appendQuantized(
                    LagunaWeightNames.mlp(layerIndex, suffix),
                    leadingShape: [denseIntermediate],
                    inputFeatures: hidden
                )
            }
            appendQuantized(
                LagunaWeightNames.mlp(layerIndex, "down_proj.weight"),
                leadingShape: [hidden],
                inputFeatures: denseIntermediate
            )
        } else {
            appendQuantized(
                LagunaWeightNames.mlp(layerIndex, "gate.proj.weight"),
                leadingShape: [experts],
                inputFeatures: hidden,
                bits: 8
            )
            tensors.append(TensorFixture(
                name: LagunaWeightNames.mlp(layerIndex, "gate.e_score_correction_bias"),
                dtype: "F32",
                shape: [experts]
            ))
            for suffix in ["switch_mlp.gate_proj.weight", "switch_mlp.up_proj.weight"] {
                appendQuantized(
                    LagunaWeightNames.mlp(layerIndex, suffix),
                    leadingShape: [experts, expertIntermediate],
                    inputFeatures: hidden
                )
            }
            appendQuantized(
                LagunaWeightNames.mlp(layerIndex, "switch_mlp.down_proj.weight"),
                leadingShape: [experts, hidden],
                inputFeatures: expertIntermediate
            )
            for suffix in ["shared_expert.gate_proj.weight", "shared_expert.up_proj.weight"] {
                appendQuantized(
                    LagunaWeightNames.mlp(layerIndex, suffix),
                    leadingShape: [sharedExpertIntermediate],
                    inputFeatures: hidden
                )
            }
            appendQuantized(
                LagunaWeightNames.mlp(layerIndex, "shared_expert.down_proj.weight"),
                leadingShape: [hidden],
                inputFeatures: sharedExpertIntermediate
            )
        }
    }

    return tensors
}

private func writeIndex(_ path: URL, tensors: [TensorFixture], shardName: String) throws {
    let entries = tensors.map { #""\#($0.name)": "\#(shardName)""# }.joined(separator: ",")
    try """
    {
      "weight_map": {
        \(entries)
      }
    }
    """.write(to: path, atomically: true, encoding: .utf8)
}

@Test
func localIterateDecodeProgressIntervalReportsEveryStepForShortRuns() {
    // local-iterate: 16 decode steps get a running-number line on every token.
    #expect(GemmaRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 16, timingRepeats: 1) == 1)
    #expect(GemmaRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 32, timingRepeats: 1) == 1)
    // local-submit: 1023 steps keep the historical 8-step cadence.
    #expect(GemmaRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 1023, timingRepeats: 1) == 8)
    // Multi-repeat runs keep the sparser 64-step cadence.
    #expect(GemmaRuntime.localIterateDecodeProgressInterval(totalDecodeSteps: 512, timingRepeats: 4) == 64)
}

@Test
func localIterateProjectedDecodeSecondsPerTokenConvergesToChargedMean() {
    // Mid-run: charged 20s so far (18s seed + 2s steps), 2 of 16 tokens done at
    // 1s/step mean -> project 14 more step-seconds on top of the charged 20.
    let projected = GemmaRuntime.localIterateProjectedDecodeSecondsPerToken(
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16
    )
    #expect(abs(projected - (20.0 + 14.0) / 16.0) < 1e-12)

    // Final token: exactly the charged mean the score payload will report.
    let final = GemmaRuntime.localIterateProjectedDecodeSecondsPerToken(
        chargedSecondsSoFar: 32,
        stepOnlySecondsSoFar: 14,
        decodedTokens: 16,
        totalDecodeTokens: 16
    )
    #expect(final == 2.0)

    // Guards.
    #expect(
        GemmaRuntime.localIterateProjectedDecodeSecondsPerToken(
            chargedSecondsSoFar: 1,
            stepOnlySecondsSoFar: 1,
            decodedTokens: 0,
            totalDecodeTokens: 16
        ) == 0
    )
}

@Test
func localIterateLiveDecodeStatusIncludesProjectedSpeedupAndScore() {
    let status = GemmaRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 0.9,
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: 0.05
    )
    #expect(status.contains("last_step_seconds=0.900000"))
    #expect(status.contains("mean_step_seconds=1.000000"))
    // 14 remaining tokens at 1s mean -> 14s ETA.
    #expect(status.contains("decode_eta_seconds=14.0"))
    #expect(status.contains("projected_decode_seconds_per_token="))
    #expect(status.contains("projected_decode_speedup="))
    #expect(status.contains("projected_score="))
    // The RAM-resident dense runtime reports no expert-bandwidth live fields.
    #expect(!status.contains("expert_gb_per_token="))
    #expect(!status.contains("expert_hit_rate="))

    // The final step has no ETA.
    let finalStep = GemmaRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 1,
        chargedSecondsSoFar: 32,
        stepOnlySecondsSoFar: 14,
        decodedTokens: 16,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: 0.05
    )
    #expect(!finalStep.contains("decode_eta_seconds="))

    // Before prefill has a positive measurement there is no score estimate.
    let withoutPrefill = GemmaRuntime.localIterateLiveDecodeStatus(
        lastStepSeconds: 0.9,
        chargedSecondsSoFar: 20,
        stepOnlySecondsSoFar: 2,
        decodedTokens: 2,
        totalDecodeTokens: 16,
        prefillSecondsPerToken: nil
    )
    #expect(withoutPrefill.contains("projected_decode_speedup="))
    #expect(!withoutPrefill.contains("projected_score="))

    #expect(
        GemmaRuntime.localIterateLiveDecodeStatus(
            lastStepSeconds: 0,
            chargedSecondsSoFar: 0,
            stepOnlySecondsSoFar: 0,
            decodedTokens: 0,
            totalDecodeTokens: 16,
            prefillSecondsPerToken: nil
        ).isEmpty
    )
}

@Test
func workerStderrDrainForwardsRedactedLinesAndKeepsTailForDiagnostics() throws {
    final class LineBox: @unchecked Sendable {
        private let lock = NSLock()
        private var lines: [String] = []
        func append(_ line: String) {
            lock.lock()
            defer { lock.unlock() }
            lines.append(line)
        }
        func snapshot() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return lines
        }
    }

    let pipe = Pipe()
    let box = LineBox()
    let drain = WorkerStderrDrain(
        handle: pipe.fileHandleForReading,
        emit: { box.append($0) }
    )

    // Two complete lines (one needing token redaction) and one unterminated
    // partial line that must still be flushed at EOF.
    try pipe.fileHandleForWriting.write(contentsOf: Data("model debug: layer 3 routed\n".utf8))
    try pipe.fileHandleForWriting.write(contentsOf: Data("expected 5 actual 7\npartial".utf8))
    try pipe.fileHandleForWriting.close()

    let tail = drain.drainedOutput(timeoutSeconds: 5)
    let emitted = box.snapshot()

    #expect(emitted == [
        "mlxfast-worker: model debug: layer 3 routed\n",
        "mlxfast-worker: token-validation-failed\n",
        "mlxfast-worker: partial\n",
    ])
    // The diagnostic tail keeps the RAW content (workerExitDiagnostic applies
    // its own whole-blob sanitization, unchanged from before).
    #expect(tail.contains("model debug: layer 3 routed"))
    #expect(tail.contains("expected 5 actual 7"))
    #expect(tail.contains("partial"))

    // A second read must not block or lose the tail.
    #expect(drain.drainedOutput(timeoutSeconds: 1).contains("partial"))
}

@Test
func workerStderrDrainCapsRetainedTail() throws {
    let pipe = Pipe()
    let drain = WorkerStderrDrain(
        handle: pipe.fileHandleForReading,
        emit: { _ in }
    )

    let filler = String(repeating: "x", count: 1024)
    for index in 0..<128 {
        try pipe.fileHandleForWriting.write(contentsOf: Data("line-\(index) \(filler)\n".utf8))
    }
    try pipe.fileHandleForWriting.write(contentsOf: Data("final-marker\n".utf8))
    try pipe.fileHandleForWriting.close()

    let tail = drain.drainedOutput(timeoutSeconds: 5)
    #expect(tail.utf8.count <= WorkerStderrDrain.tailByteLimit + 16)
    #expect(tail.contains("final-marker"))
    #expect(!tail.contains("line-0 "))
}

@Test
func workerStderrDrainCapsUnterminatedLine() throws {
    let pipe = Pipe()
    let drain = WorkerStderrDrain(
        handle: pipe.fileHandleForReading,
        emit: { _ in }
    )
    try pipe.fileHandleForWriting.write(
        contentsOf: Data(repeating: 0x78, count: WorkerStderrDrain.tailByteLimit * 3)
    )
    try pipe.fileHandleForWriting.close()

    let tail = drain.drainedOutput(timeoutSeconds: 5)
    #expect(tail.contains(WorkerStderrDrain.truncatedLine))
    #expect(tail.utf8.count <= WorkerStderrDrain.tailByteLimit)
}

@Test
func bufferedFileLineReaderPreservesBufferedLinesAndEOFFragment() throws {
    let pipe = Pipe()
    let reader = BufferedFileLineReader(handle: pipe.fileHandleForReading, maximumLineByteCount: 32)
    try pipe.fileHandleForWriting.write(contentsOf: Data("first\nsecond\nthird".utf8))
    try pipe.fileHandleForWriting.close()

    #expect(try reader.readLine() == Data("first".utf8))
    #expect(try reader.readLine() == Data("second".utf8))
    #expect(try reader.readLine() == Data("third".utf8))
    #expect(try reader.readLine() == nil)
}

@Test
func bufferedFileLineReaderRejectsOversizedLine() throws {
    let pipe = Pipe()
    let reader = BufferedFileLineReader(handle: pipe.fileHandleForReading, maximumLineByteCount: 8)
    try pipe.fileHandleForWriting.write(contentsOf: Data("123456789\n".utf8))
    try pipe.fileHandleForWriting.close()

    #expect(throws: MLXFastError.self) {
        _ = try reader.readLine()
    }
}

@Test
func runtimeWorkerClientTimesOutWaitingForHello() throws {
    // The fixture ignores TERM to force the client's SIGKILL escalation, but
    // it must NOT live forever: an interrupted test run cannot reap it, and
    // unbounded `while :; do :; done` fixtures have been observed lingering
    // as ppid-1 orphans burning a CPU core for days. Sleep-loop for a
    // bounded lifetime instead (worst case ~2 minutes if orphaned).
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    trap '' TERM
    \(boundedFixtureIdleLoop)
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let start = Date()
    var message = ""
    do {
        _ = try RuntimeWorkerClient(
            options: shortRuntimeWorkerOptions(executable: executable),
            weightsPath: executable.deletingLastPathComponent().path
        )
    } catch {
        message = String(describing: error)
    }
    #expect(message.contains("timed out waiting for protocol hello"))
    #expect(Date().timeIntervalSince(start) < 3)
}

@Test
func runtimeWorkerWatchdogAtomicCancellationDisarmsTimer() {
    let watchdog = RuntimeWorkerWatchdog(
        process: Process(),
        timeoutSeconds: 0.05,
        terminationGraceSeconds: 0
    )

    #expect(!watchdog.cancelAndReturnDidFire())
    Thread.sleep(forTimeInterval: 0.1)
    #expect(!watchdog.cancelAndReturnDidFire())
}

@Test
func runtimeWorkerClientCancelsSuccessfulRequestWatchdogs() throws {
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    IFS= read -r first_request || exit 0
    printf '%s\\n' '{"id":1,"nonce":"test-nonce","ok":true}'
    IFS= read -r second_request || exit 0
    printf '%s\\n' '{"id":2,"nonce":"test-nonce","ok":true}'
    IFS= read -r final_request || exit 0
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 2,
            requestTimeoutSeconds: 0.1,
            shutdownTimeoutSeconds: 0.1,
            terminationGraceSeconds: 0.05
        ),
        weightsPath: executable.deletingLastPathComponent().path
    )
    defer { client.close() }

    _ = try client.prefill(promptTokens: [1])
    // A stale watchdog from the first request would terminate the worker while
    // it is idle and make this second request fail before receiving id 2.
    Thread.sleep(forTimeInterval: 0.25)
    _ = try client.prefill(promptTokens: [2])
}

@Test
func runtimeWorkerClientTimesOutStalledRequest() throws {
    // Bounded stall (see runtimeWorkerClientTimesOutWaitingForHello): the
    // client SIGKILLs it within the test, and an orphaned copy self-expires.
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    trap '' TERM
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    IFS= read -r request || exit 0
    \(boundedFixtureIdleLoop)
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 2,
            requestTimeoutSeconds: 0.1,
            shutdownTimeoutSeconds: 0.1,
            terminationGraceSeconds: 0.05
        ),
        weightsPath: executable.deletingLastPathComponent().path
    )
    defer { client.close() }
    let start = Date()
    var message = ""
    do {
        _ = try client.prefill(promptTokens: [1])
    } catch {
        message = String(describing: error)
    }
    #expect(message.contains("timed out handling request prefill"))
    #expect(Date().timeIntervalSince(start) < 3)
}

@Test
func runtimeWorkerClientCloseEscalatesPastIgnoredTerminate() throws {
    // Bounded stall (see runtimeWorkerClientTimesOutWaitingForHello): the
    // client SIGKILLs it within the test, and an orphaned copy self-expires.
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    trap '' TERM
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    \(boundedFixtureIdleLoop)
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 2,
            requestTimeoutSeconds: 0.1,
            shutdownTimeoutSeconds: 0.1,
            terminationGraceSeconds: 0.05
        ),
        weightsPath: executable.deletingLastPathComponent().path
    )
    let start = Date()
    client.close()
    #expect(Date().timeIntervalSince(start) < 3)
}

@Test
func runtimeWorkerClientDrainsLargeStderrBeforeHelloWhenForwardingIsOff() throws {
    let executable = try makeRuntimeWorkerScript("""
    #!/bin/sh
    /usr/bin/yes x | /usr/bin/head -c 196608 | /usr/bin/tr -d '\\n' >&2
    printf '%s\\n' '{"id":0,"nonce":"test-nonce","ok":true}'
    IFS= read -r request || exit 0
    """)
    defer { try? FileManager.default.removeItem(at: executable.deletingLastPathComponent()) }
    let client = try RuntimeWorkerClient(
        options: RuntimeWorkerOptions(
            executablePath: executable.path,
            helloTimeoutSeconds: 2,
            requestTimeoutSeconds: 0.1,
            shutdownTimeoutSeconds: 0.1,
            terminationGraceSeconds: 0.05
        ),
        weightsPath: executable.deletingLastPathComponent().path
    )
    client.close()
}

@Test
func runtimeWorkerClientRejectsNonfiniteTimeoutsBeforeLaunch() throws {
    let options = RuntimeWorkerOptions(
        executablePath: "/does/not/exist",
        helloTimeoutSeconds: .infinity
    )
    #expect(throws: MLXFastError.self) {
        _ = try RuntimeWorkerClient(options: options, weightsPath: "/does/not/exist")
    }
}

@Test
func runtimeWorkerStartsTheOrphanReaperBeforeLoadingTheModel() throws {
    // Source-level guard: runWorker must arm the reaper BEFORE any of the
    // model-loading work, or the load window (the exact window stdin-EOF
    // cannot cover) is left unprotected.
    let worker = try String(
        contentsOfFile: "Sources/MLXFastHarness/GemmaRuntimeWorker.swift",
        encoding: .utf8
    )
    let reaperCall = try #require(worker.range(of: "startRuntimeWorkerOrphanReaper()"))
    let protocolIO = try #require(worker.range(of: "RuntimeWorkerProtocolIO.isolatingStandardIO()"))
    let weightCache = try #require(worker.range(of: "LagunaRuntimeWeightCache(loader:"))
    #expect(reaperCall.lowerBound < protocolIO.lowerBound)
    #expect(reaperCall.lowerBound < weightCache.lowerBound)
    #expect(worker.contains("getppid() == 1"))
}

@Test
func localIteratePrefillStatusReportsPerTokenAndSpeedup() {
    let status = GemmaRuntime.localIteratePrefillStatus(
        elapsedSeconds: 51.2,
        promptTokens: 512
    )
    #expect(status.contains("seconds=51.2"))
    #expect(status.contains("seconds_per_token=0.100000"))
    let expectedSpeedup = MLXFastConstants.officialBaselinePrefillSecondsPerToken / 0.1
    #expect(status.contains("prefill_speedup=\(String(format: "%.3f", expectedSpeedup))"))

    // Zero-duration or zero-token inputs fall back to the plain seconds field.
    #expect(
        GemmaRuntime.localIteratePrefillStatus(elapsedSeconds: 0, promptTokens: 512)
            == "seconds=0.0"
    )
}

@Test
func localIterateSummaryEmitsSpeedupsAndEstimatedScore() {
    let timing = GemmaRuntime.LocalIterateTimingResult(
        correctness: GemmaRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken / 2,
        decode: GemmaRuntime.DecodeMeasurement(
            secondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken / 2,
            bandwidthGBPerToken: 0,
            bandwidthSource: "ram_resident_model"
        ),
        expertStats: .zero,
        peakRamGB: 24.5
    )

    var lines: [String] = []
    GemmaRuntime.emitLocalIterateSummary(
        modeName: "local-iterate",
        timing: timing,
        progress: { lines.append($0) }
    )

    let joined = lines.joined(separator: "\n")
    #expect(joined.contains("local-iterate summary prefill_seconds_per_token="))
    #expect(joined.contains("prefill_speedup=2.000"))
    #expect(joined.contains("local-iterate summary decode_seconds_per_token="))
    #expect(joined.contains("decode_speedup=2.000"))
    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    #expect(joined.contains("est_score=2.000"))
    #expect(joined.contains("published as this run's local estimated score, not a ranked score"))
    #expect(joined.contains("decode_bandwidth_gb_per_token=0"))
    #expect(joined.contains("peak_ram_gb=24.500"))
    #expect(!joined.contains("expert_hit_rate="))
}

// The Yukon participant CLI (`mlxfast run`) executes benchmarkCommand and then
// validates the contract scorePath as `{ "score": <finite number>, ... }`;
// `score: null` fails its schema. Local modes therefore publish the estimated
// score (the same decode_speedup^0.75 * prefill_speedup^0.25 estimate the
// summary prints) as a numeric, CLI-usable score.
@Test
func localIterateScorePublishesCLIUsableEstimatedScore() throws {
    let payload = GemmaRuntime.localIterateScore(
        peakRamGB: 24.5,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken / 2,
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken / 2,
        wallSeconds: 10,
        validationSeconds: 1,
        correctnessSeconds: 5,
        timedSeconds: 5,
        correctness: GemmaRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    let estimated = try #require(payload.score)
    #expect(abs(estimated - 2) < 1e-9)
    #expect(payload.passed)
    // The runtime label is the local-mode marker that keeps the estimate
    // clearly distinguishable from a ranked payload (runtime == "swift").
    #expect(payload.metrics.runtime == "swift-local-iterate")

    // The sealed scorePath JSON must carry the score as a finite JSON number
    // -- the exact thing the CLI's schema checks -- and not null.
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("score.json").path
    try writeScorePayload(payload, to: path)
    let raw = try String(contentsOfFile: path, encoding: .utf8)
    #expect(!raw.contains("\"score\" : null"))
    let object = try #require(
        try JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
    )
    let rawScore = try #require(object["score"] as? Double)
    #expect(rawScore.isFinite)
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: Data(raw.utf8))
    #expect(decoded.score == estimated)
}

@Test
func localIterateScoreFailsForUnusableTimings() {
    // Zero/invalid timings make the estimate non-finite. The payload must be a
    // failed run, not a passing result that merely omits its score.
    let payload = GemmaRuntime.localIterateScore(
        peakRamGB: 0,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: 0,
        prefillSecondsPerToken: 0,
        wallSeconds: 0,
        validationSeconds: 0,
        correctnessSeconds: 0,
        timedSeconds: 0,
        correctness: GemmaRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(!payload.passed)
    #expect(payload.metrics.passedCorrectness)
    #expect(payload.metrics.error.contains("timing metrics must be finite and positive"))
}

@Test
func localIterateScoreSanitizesNonfiniteFailureMetricsForJSON() throws {
    let payload = GemmaRuntime.localIterateScore(
        peakRamGB: .nan,
        bandwidthGBPerToken: .infinity,
        decodeSecondsPerToken: .nan,
        prefillSecondsPerToken: -.infinity,
        wallSeconds: -.infinity,
        validationSeconds: .nan,
        correctnessSeconds: .infinity,
        timedSeconds: -1,
        correctness: GemmaRuntime.localIterateCorrectnessReport(
            passed: true,
            checkedSteps: 18,
            caseCount: 1,
            firstFailingStep: nil,
            expectedToken: nil,
            actualToken: nil,
            goldenHash: "hash",
            expertStats: .zero,
            error: "",
            modeName: "local-iterate"
        ),
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        runtime: "swift-local-iterate"
    )

    #expect(payload.score == nil)
    #expect(!payload.passed)
    #expect(payload.metrics.peakRamGB == 0)
    #expect(payload.metrics.bandwidthGBPerToken == 0)
    #expect(payload.metrics.decodeSecondsPerToken == 0)
    #expect(payload.metrics.prefillSecondsPerToken == 0)
    #expect(payload.metrics.benchmarkWallSeconds == 0)
    #expect(payload.metrics.preflightSeconds == 0)
    #expect(payload.metrics.correctnessSeconds == 0)
    #expect(payload.metrics.timedBenchmarkSeconds == 0)

    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("score.json").path
    try writeScorePayload(payload, to: path)
    let encoded = try Data(contentsOf: URL(fileURLWithPath: path))
    let encodedText = String(decoding: encoded, as: UTF8.self).lowercased()
    #expect(!encodedText.contains("nan"))
    #expect(!encodedText.contains("infinity"))
    let decoded = try JSONDecoder().decode(
        ScorePayload.self,
        from: encoded
    )
    #expect(decoded.score == nil)
    #expect(!decoded.passed)
    #expect(decoded.metrics.peakRamGB == 0)
    #expect(decoded.metrics.bandwidthGBPerToken == 0)
    #expect(decoded.metrics.decodeSecondsPerToken == 0)
    #expect(decoded.metrics.prefillSecondsPerToken == 0)
    #expect(decoded.metrics.benchmarkWallSeconds == 0)
    #expect(decoded.metrics.preflightSeconds == 0)
    #expect(decoded.metrics.correctnessSeconds == 0)
    #expect(decoded.metrics.timedBenchmarkSeconds == 0)
}

// The immediate token-mismatch report carries the non-M5 caveat: the public
// goldens are M5-generated greedy continuations, so near-tie argmaxes diverge
// deterministically on other Apple Silicon generations even for a correct
// build. Messaging only -- the mismatch still fails the gate, and the line
// stays redacted (no expected/actual token values in the progress stream).
@Test
func firstTokenMismatchReportIncludesNonM5GoldenCaveat() {
    var lines: [String] = []
    GemmaRuntime.reportFirstTokenMismatch(
        { lines.append($0) },
        modeName: "local-iterate",
        checkedStep: 17
    )

    #expect(lines.count == 2)
    #expect(lines[0].contains("local-iterate correctness FAIL first token mismatch at checked_step=17"))
    #expect(lines[1].contains("local-iterate note: the public goldens are M5-generated"))
    #expect(lines[1].contains("deterministic near-tie token mismatch is expected for a correct build"))
    #expect(lines[1].contains("Correctness fixtures are M5-generated"))
    #expect(lines[1].contains("the ranked M5 runner is the source of truth"))
}

// A local run whose public-gate correctness check failed but whose timings
// were measured (the deterministic non-M5 near-tie divergence is exactly this
// shape) must still publish the finite estimated score: the Yukon CLI
// validates scorePath as `{ "score": <finite number>, ... }`, and score: null
// used to make `mlxfast run` error at the contract layer on every non-M5 box.
// passed stays false and every failure field survives, so the failure remains
// unmistakable.
@Test
func localFailedRunWithMeasuredTimingsStillPublishesNumericEstimatedScore() throws {
    let report = CorrectnessReport(
        passed: false,
        checkedSteps: 17,
        caseCount: 1,
        firstFailingCase: "local-iterate",
        firstFailingStep: 16,
        expectedToken: 236761,
        actualToken: 618,
        goldenHash: "golden",
        error: "local-iterate teacher-forced token mismatch"
    )
    let failed = GemmaRuntime.failedScore(
        error: "local-iterate teacher-forced token mismatch",
        correctness: report,
        passedCorrectness: false,
        expectedToken: report.expectedToken,
        actualToken: report.actualToken,
        decodeSecondsPerToken: MLXFastConstants.officialBaselineDecodeSecondsPerToken / 2,
        prefillSecondsPerToken: MLXFastConstants.officialBaselinePrefillSecondsPerToken / 2,
        runtime: "swift-local-iterate"
    )

    var lines: [String] = []
    let payload = GemmaRuntime.localModeFailedPayloadWithEstimatedScore(
        failed,
        modeName: "local-iterate",
        progress: { lines.append($0) }
    )

    // 2x on both axes -> estimated score 2 under decode^0.75 * prefill^0.25.
    let estimated = try #require(payload.score)
    #expect(abs(estimated - 2) < 1e-9)
    #expect(!payload.passed)
    #expect(!payload.metrics.passedCorrectness)
    #expect(payload.metrics.error == "local-iterate teacher-forced token mismatch")
    #expect(payload.metrics.firstFailingStep == 16)
    #expect(payload.metrics.expectedToken == 236761)
    #expect(payload.metrics.actualToken == 618)
    #expect(payload.metrics.runtime == "swift-local-iterate")
    #expect(lines.joined(separator: "\n").contains("publishing est_score=2.000 with passed=false"))

    // The sealed scorePath JSON carries the score as a finite JSON number --
    // the exact thing the CLI's schema checks -- while staying a failed run.
    let directory = try temporaryDirectory()
    defer { try? FileManager.default.removeItem(at: directory) }
    let path = directory.appendingPathComponent("score.json").path
    try writeScorePayload(payload, to: path)
    let raw = try String(contentsOfFile: path, encoding: .utf8)
    #expect(!raw.contains("\"score\" : null"))
    let decoded = try JSONDecoder().decode(ScorePayload.self, from: Data(raw.utf8))
    #expect(decoded.score == estimated)
    #expect(!decoded.passed)
}

// Failure payloads without measured timings (validation errors, missing
// artifacts) cannot carry a meaningful estimate and keep score: null; passing
// payloads are never touched.
@Test
func localFailedPayloadWithoutTimingsKeepsNullScore() {
    let noTimings = GemmaRuntime.failedScore(
        error: "local-iterate public golden must contain at least one case",
        correctness: nil,
        passedCorrectness: false,
        runtime: "swift-local-iterate"
    )
    var lines: [String] = []
    let unchanged = GemmaRuntime.localModeFailedPayloadWithEstimatedScore(
        noTimings,
        modeName: "local-iterate",
        progress: { lines.append($0) }
    )
    #expect(unchanged.score == nil)
    #expect(!unchanged.passed)
    #expect(lines.isEmpty)

    let passing = ScorePayload(score: 1.5, passed: true, metrics: noTimings.metrics)
    let stillPassing = GemmaRuntime.localModeFailedPayloadWithEstimatedScore(
        passing,
        modeName: "local-iterate",
        progress: { lines.append($0) }
    )
    #expect(stillPassing == passing)
    #expect(lines.isEmpty)
}

// Guard the ranked score semantics against the local-mode estimate: the
// official benchmark path still publishes exactly the score it was given on
// pass, and null on failure. Only localIterateScore (reached exclusively via
// --local-iterate/--local-submit) synthesizes an estimate.
@Test
func rankedScoreSemanticsAreUnchangedByLocalEstimatedScore() {
    let correctness = CorrectnessReport(
        passed: true,
        checkedSteps: 513,
        caseCount: 1,
        firstFailingCase: nil,
        firstFailingStep: nil,
        expectedToken: nil,
        actualToken: nil,
        goldenHash: "hash",
        error: ""
    )
    let passed = GemmaRuntime.passedScore(
        score: 1.25,
        peakRamGB: 20,
        bandwidthGBPerToken: 0,
        decodeSecondsPerToken: 0.1,
        prefillSecondsPerToken: 0.01,
        benchmarkWallSeconds: 100,
        preflightSeconds: 1,
        correctnessSeconds: 50,
        timedBenchmarkSeconds: 40,
        numLayers: MLXFastConstants.numHiddenLayers,
        correctness: correctness,
        expertStats: .zero,
        bandwidthSource: "ram_resident_model",
        weightsDigest: nil,
        gpqaTTFT: .zero
    )
    #expect(passed.score == 1.25)
    #expect(passed.metrics.runtime == "swift")

    let failed = GemmaRuntime.failedScore(
        error: "boom",
        correctness: nil,
        passedCorrectness: false,
        runtime: "swift"
    )
    #expect(failed.score == nil)
    #expect(failed.passed == false)
}

@Test
func localIteratePhaseHeartbeatFiresWhileBlockedAndStopsAfterCancel() throws {
    final class MessageBox: @unchecked Sendable {
        private let lock = NSLock()
        private var messages: [String] = []
        func append(_ message: String) {
            lock.lock()
            defer { lock.unlock() }
            messages.append(message)
        }
        func snapshot() -> [String] {
            lock.lock()
            defer { lock.unlock() }
            return messages
        }
    }

    // No progress sink means no timer at all.
    #expect(GemmaRuntime.startPhaseHeartbeat(label: "x", progress: nil) == nil)

    let box = MessageBox()
    let heartbeat = try #require(
        GemmaRuntime.startPhaseHeartbeat(
            label: "local-iterate prefill measured",
            intervalSeconds: 0.05,
            progress: { box.append($0) }
        )
    )
    Thread.sleep(forTimeInterval: 0.3)
    let cancellationDrained = DispatchSemaphore(value: 0)
    heartbeat.setCancelHandler {
        cancellationDrained.signal()
    }
    heartbeat.cancel()
    #expect(cancellationDrained.wait(timeout: .now() + 1) == .success)
    let firedWhileRunning = box.snapshot()
    #expect(!firedWhileRunning.isEmpty)
    #expect(
        firedWhileRunning.allSatisfy {
            $0.hasPrefix("local-iterate prefill measured still running phase_seconds=")
        }
    )

    // After cancellation has drained, the heartbeat must stay quiet.
    Thread.sleep(forTimeInterval: 0.2)
    #expect(box.snapshot().count == firedWhileRunning.count)
}

private func writeSafetensors(_ path: URL, tensors: [TensorFixture]) throws {
    var object: [String: Any] = [:]
    var cursor = 0
    for tensor in tensors.sorted(by: { $0.name < $1.name }) {
        let byteCount = try expectedTensorByteCount(
            name: tensor.name,
            dtype: TensorDType.parse(tensor.dtype),
            shape: tensor.shape
        )
        object[tensor.name] = [
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "data_offsets": [cursor, cursor + byteCount],
        ]
        cursor += byteCount
    }

    var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    while header.count % 8 != 0 {
        header.append(0x20)
    }

    var output = Data()
    var headerLength = UInt64(header.count).littleEndian
    output.append(Data(bytes: &headerLength, count: 8))
    output.append(header)
    try output.write(to: path)

    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(output.count + cursor))
}

private func arrayJSON(_ values: [Int]) -> String {
    "[\(values.map(String.init).joined(separator: ","))]"
}

private func writeSparseFile(_ path: URL, byteCount: Int) throws {
    _ = FileManager.default.createFile(atPath: path.path, contents: nil)
    let handle = try FileHandle(forWritingTo: path)
    defer {
        try? handle.close()
    }
    try handle.truncate(atOffset: UInt64(byteCount))
}

private func temporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString,
        isDirectory: true
    )
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

/// Idle loop for fake workers that must stay unresponsive while the client
/// times out / escalates to SIGKILL, without becoming immortal debris: it
/// ignores nothing itself (TERM handling is up to the fixture), burns no CPU
/// (sleep, not a busy loop), and self-expires after ~2 minutes so a fixture
/// orphaned by an interrupted `swift test` run disappears on its own.
private let boundedFixtureIdleLoop = """
idle_iterations=0
while [ "${idle_iterations}" -lt 1200 ]; do
  sleep 0.1
  idle_iterations=$((idle_iterations + 1))
done
"""

private func makeRuntimeWorkerScript(_ contents: String) throws -> URL {
    let directory = try temporaryDirectory()
    let executable = directory.appendingPathComponent("fake-runtime-worker")
    try contents.write(to: executable, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
        [.posixPermissions: 0o755],
        ofItemAtPath: executable.path
    )
    return executable
}

/// The pinned Laguna XS 2.1 MoE architecture as the transformed runtime
/// config declares it (mirroring mlx-community/Laguna-XS-2.1-4bit's
/// config.json): a full-attention layer with 48 query heads and YaRN
/// partial RoPE at layers 0, 4, ..., 36, sliding-window layers with 64
/// query heads and default RoPE elsewhere, a dense MLP only at layer 0,
/// and 256-expert top-8 MoE blocks with a 512-wide shared expert on the
/// other 39 layers.
private func pinnedRuntimeWorkerConfigurationObject() -> [String: Any] {
    [
        "model_type": "laguna",
        "hidden_size": MLXFastConstants.hiddenSize,
        "num_hidden_layers": MLXFastConstants.numHiddenLayers,
        "intermediate_size": MLXFastConstants.intermediateSize,
        "num_attention_heads": MLXFastConstants.attentionHeads,
        "num_attention_heads_per_layer": (0..<MLXFastConstants.numHiddenLayers).map {
            $0 % 4 == 0 ? 48 : 64
        },
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rms_norm_eps": 1e-6,
        "vocab_size": MLXFastConstants.vocabSize,
        "sliding_window": 512,
        "max_position_embeddings": 262_144,
        "attention_bias": false,
        "attention_dropout": 0.0,
        "gating": "per-head",
        "tie_word_embeddings": false,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "moe_routed_scaling_factor": 2.5,
        "norm_topk_prob": true,
        "mlp_layer_types": (0..<MLXFastConstants.numHiddenLayers).map {
            $0 == 0 ? "dense" : "sparse"
        },
        "mlp_only_layers": [0],
        "decoder_sparse_step": 1,
        "layer_types": (0..<MLXFastConstants.numHiddenLayers).map {
            $0 % 4 == 0 ? "full_attention" : "sliding_attention"
        },
        "rope_parameters": [
            "sliding_attention": [
                "rope_theta": 10_000.0,
                "rope_type": "default",
                "partial_rotary_factor": 1.0,
            ],
            "full_attention": [
                "rope_theta": 500_000.0,
                "rope_type": "yarn",
                "factor": 32.0,
                "original_max_position_embeddings": 8_192,
                "beta_fast": 64.0,
                "beta_slow": 1.0,
                "attention_factor": 1.0,
                "partial_rotary_factor": 0.5,
            ],
        ],
        "quantization": [
            "group_size": 64,
            "bits": 4,
            "mode": "affine",
            // The shipped checkpoint declares per-tensor 8-bit overrides for
            // the MoE router gates; the global pinned-config gate tolerates
            // them (the runtime weight loader validates them against the
            // stored tensor geometry).
            "language_model.model.layers.1.mlp.gate.proj": [
                "group_size": 64,
                "bits": 8,
            ],
        ],
    ]
}

private func shortRuntimeWorkerOptions(executable: URL) -> RuntimeWorkerOptions {
    RuntimeWorkerOptions(
        executablePath: executable.path,
        helloTimeoutSeconds: 0.1,
        requestTimeoutSeconds: 0.1,
        shutdownTimeoutSeconds: 0.1,
        terminationGraceSeconds: 0.05
    )
}

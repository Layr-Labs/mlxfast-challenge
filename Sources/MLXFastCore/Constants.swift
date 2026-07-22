public enum MLXFastConstants {
    public static let referenceModelName = "laguna-xs-2.1-4bit"
    public static let defaultReferencePath = "reference_weights/laguna-xs-2.1-4bit"
    public static let defaultReferenceCachePath = ".cache/huggingface/hub/models--mlx-community--Laguna-XS-2.1-4bit/snapshots/main"
    public static let defaultWeightsPath = "weights"
    public static let defaultGoldenPath = "correctness_golden.json"
    public static let defaultPublicCorrectnessPromptPath = "correctness_prompts/public_longcopy_gate_english_512.txt"
    public static let defaultPublicCorrectnessGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_256.json"
    public static let defaultPublicLocalSubmitGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_1024.json"
    public static let defaultScorePath = "score.json"
    public static let defaultLocalIterateScorePath = "score.local-iterate.json"

    // Frozen text-tower geometry of the pinned Poolside Laguna XS 2.1 4-bit
    // target (mlx-community/Laguna-XS-2.1-4bit), mirrored from
    // Sources/MLXFastModel/LagunaConfig.swift's LagunaConstants (MLXFastCore
    // is trusted and cannot import the editable model target).
    // `intermediateSize` is the dense MLP width used only by layer 0 -- the
    // 39 sparse layers use the MoE widths pinned in the track contract.
    // `attentionHeads` is the checkpoint's top-level `num_attention_heads`
    // fallback (48); per-layer counts are 48 on full-attention layers
    // (0, 4, 8, ..., 36) and 64 on the 30 sliding-window layers.
    public static let vocabSize = 100_352
    public static let hiddenSize = 2_048
    public static let intermediateSize = 8_192
    public static let numHiddenLayers = 40
    public static let attentionHeads = 48
    public static let correctnessPromptTokens = 512
    // Keep the public gate long enough to catch broad decode regressions while
    // leaving budget for the hidden GPQA behavior checks in the official job.
    public static let correctnessSteps = 64
    public static let correctnessTopLogits = 8
    public static let correctnessLogitTieTolerance = 1e-6
    public static let correctnessMaxAnchorContextTokens = 1_024
    public static let correctnessMaxFreeRunSteps = 256
    public static let correctnessMaxBehaviorPromptTokens = 2_048
    public static let correctnessMaxBehaviorSteps = 64
    public static let correctnessGPQACaseCount = 5
    // Cross-machine greedy decode can drift on hidden GPQA even with pinned
    // Swift/MLX. Semantic GPQA behavior captures a short continuation for the
    // private judge; exact token enforcement stays on the long copy gate and
    // non-semantic behavior fixtures.
    // 64 (was 10, DeepSeek-era): originally a Gemma-era calibration (a
    // 2026-07-06 budget capture at 10/32/64 showed shorter budgets cut most
    // candidates off mid-sentence; 64, the correctnessMaxBehaviorSteps
    // ceiling, gave the judge the most usable text). Retained unchanged for
    // the Laguna XS 2.1 re-pin: the four ranked Laguna runs on m5-bench
    // (29891158354, 29898041981, 29931435233, 29934545157) all passed the
    // judge gate at this budget against the regenerated hidden Laguna GPQA
    // reference, so 64 remains the working budget.
    public static let correctnessGPQAMaxNewTokens = 64
    // Semantic judging uses short hidden GPQA answers as a baseline-calibrated
    // gate for optimizations that preserve the exact prefix but damage answer
    // sense. Five cases keeps the full GitHub job near the 30-minute budget.
    // The captured answer is a prefix of the behavior-gate generation, so
    // semanticGPQAMaxNewTokens is only effective up to
    // correctnessGPQAMaxNewTokens (and must stay <= correctnessMaxBehaviorSteps).
    public static let semanticGPQACaseCount = 5
    public static let semanticGPQAMaxNewTokens = 64
    // Baseline-calibrated threshold, re-validated 2026-07-22 for the Laguna
    // XS 2.1 re-pin against the regenerated hidden Laguna GPQA reference:
    // the four ranked official-runner Laguna baseline observations judged
    // 2/5, 2/5, 1/5, and 2/5 (runs 29891158354, 29898041981, 29931435233,
    // and 29934545157; Opus 4.8 judge). min(observed) is 1, so 1 stays the
    // conservative floor: an observed 1/5 sits exactly AT the threshold
    // (raising it would flake honest runs on single-case judge
    // nondeterminism), while a submission that wrecks answer quality (0/5
    // judged) still fails the gate instead of merely being recorded. Keep in
    // sync with the workflow env MLXFAST_SEMANTIC_GPQA_MIN_PASS. If the
    // hidden prompts, the judge model, or the reference model change, a
    // fresh judged official-runner baseline must recalibrate this threshold.
    // (Historical: the 2026-07-09 Gemma 4 31B-IT calibration observed a
    // stable 2/5 across five runs and set min(observed) - 1 = 1.)
    public static let semanticGPQAMinPassCount = 1
    public static let benchmarkPrefillPromptTokens = 512
    // Stable public identifier for the private timed-evaluation prompt. The
    // prompt bytes remain operator-provisioned; changing this identifier is a
    // ranking-contract change and forces build-hash-keyed timed oracles to be
    // regenerated.
    public static let benchmarkEvaluationTargetID = "lowsim-prose-v1"
    // Offline prompt-lookup susceptibility gate. The analyzer simulates a
    // longest recurrent suffix predictor over these orders. At <= 3% accepted
    // draft tokens, even an idealized zero-overhead predictor is capped near
    // 1.03x before verification and lookup overhead.
    public static let benchmarkNGramSelfSimilarityOrders = [1, 2, 3]
    public static let benchmarkMaxPromptLookupHitRate = 0.03
    // Scored decode is parent-measured wall time for decode setup plus this
    // many checked token steps. Charging setup prevents submitted model code
    // from precomputing future decode tokens in an unscored seed-prefill phase.
    public static let benchmarkDecodeSteps = 128
    // EXPERIMENTAL MTP track limits. This track is deliberately separate from
    // the frozen one-token benchmark above: it emits diagnostics only and does
    // not contribute to score.json. Keep the total explicit (rather than
    // deriving it) so changing either protocol requires a conscious rebaseline.
    public static let experimentalMTPMaxBlockSize = 4
    // Compatibility default for callers that do not select an explicit public
    // oracle length.
    public static let experimentalMTPMaxTotalTokens = 128
    // Trusted-parent configured experimental runs may exercise longer tail
    // boundaries when the selected oracle contains enough tokens. 1,536 is
    // the laguna-xs-2.1-mtp-v1 contract's `maximum_decode_tokens`: it keeps
    // the untimed correctness legs able to wrap Laguna's 512-position
    // sliding-window cache (3x the window) while ranked timed decode stays
    // fixed at 512 by the workflow env.
    public static let experimentalMTPMaxConfiguredTotalTokens = 1_536
    public static let localIterateBenchmarkDecodeSteps = 16
    // Local submit uses a longer public fixture so the Yukon pre-submit hook
    // exercises one continuous decode trajectory for about ten minutes instead
    // of repeating the short local-iterate correctness window.
    public static let localSubmitBenchmarkDecodeSteps = 1023
    public static let localSubmitBenchmarkRepeats = 1
    // Seed measured decode with the full prompt. A short instruction-prefix
    // seed can free-run differently across Apple Silicon/MLX versions even
    // when teacher-forced correctness agrees, which makes the timed oracle
    // fragile for reasons unrelated to kernel performance.
    public static let benchmarkDecodeSeedTokens = 512
    // The timed benchmark runs FIRST, before correctness/GPQA, specifically so
    // the measured model path is cold (the correctness gate must not be able to
    // warm it). The official baselines below were measured the same way, so a
    // warmup run here would both diverge from how the baseline was calibrated
    // and add minutes to the job -- keep zero warmup and one measured run.
    public static let benchmarkPrefillWarmupRuns = 0
    public static let benchmarkPrefillTimedRuns = 1
    // Acceptance bands (see AcceptanceBand + docs/thermal-variance-investigation.md).
    // Prefill and decode are noisy single measurements, gated against the same-VM
    // paired baseline B (which cancels host-speed differences). Each run's value must
    // land within [B*(1-down), B*(1+up)]; > +up = slowdown/regression (fail),
    // < -down = improvement too large for one submission / lucky reading (fail).
    //
    // Prefill: +/-5% symmetric -- prefill is not a real optimization axis, so it is a
    // health gate (regression and lucky-fast both fail past 5%).
    //
    // Decode: +2% regression / -5% gain -- tight on regressions (decode is the primary
    // scored axis), and a single submission's decode gain is capped at 5%; larger wins
    // must be CHUNKED across submissions (bounds lucky-measurement inflation and forces
    // incremental, verifiable progress). Decode is the axis the score rewards, but the
    // per-submission step is capped, not the cumulative total across submissions.
    public static let prefillBandUpTolerance = 0.05
    public static let prefillBandDownTolerance = 0.05
    public static let decodeBandUpTolerance = 0.02
    public static let decodeBandDownTolerance = 0.05
    // CACHED serial baseline: the Poolside Laguna XS 2.1 4-bit calibration.
    // Calibration provenance: the live M5 paired baseline as of 2026-07-22 --
    // the mean of the `baseline_decode_seconds_per_token` /
    // `baseline_prefill_seconds_per_token` fields published by the 4
    // consecutive successful ranked Laguna runs on the self-hosted M5 Max
    // (m5-bench): 29891158354, 29898041981, 29931435233, and 29934545157
    // (each field is measure-job's on-box timing of the pinned Laguna
    // reference tree in that run's session): decode clustered
    // 0.0093606-0.0094083 (CV 0.20%), prefill 0.0003596-0.0003664 (CV
    // 0.71%). The on-box /opt/bench-runner/state/baseline-calibration.json
    // is runner-private, so the published per-run baseline fields are the
    // audited public source. Supersedes the Gemma 4 31B 4-bit calibration of
    // 2026-07-12 (prefill 0.0016216554767252605 / decode
    // 0.04405625764973958, runs 29179374395-29197772284), which was ~4.5x /
    // ~4.7x slower than the Laguna baseline and inflated local score
    // estimates accordingly after the model re-pin.
    //
    // These constants are NOT the ranked scoring denominator. The ranked runner
    // is the single self-hosted M5 Max (m5-bench), where measure-job times the
    // candidate and the PINNED on-box reference tree back to back in the same
    // session behind the same 40C thermal gate; the paired ratio against that
    // live same-session baseline is what overlay-paired-timing.sh folds into
    // the final score. These constants keep two roles: local-mode score
    // estimates (--local-iterate / --local-submit) and the gates-only pass's
    // placeholder timing fields, which the paired-timing overlay replaces. See
    // the paired-baseline section of docs/benchmark-window-freeze.md.
    public static let officialBaselinePrefillSecondsPerToken = 0.00036280499267578125
    public static let officialBaselineDecodeSecondsPerToken = 0.00938833593359375
    public static let scorePrefillWeight = 0.25
    public static let scoreDecodeWeight = 0.75
    public static let scorePrefillSpeedupFloor = 0.95
    public static let scoreDecodeSpeedupFloor = 0.95
    // The Laguna XS 2.1 4-bit text tower is ~19 GB; 25 GiB keeps ample
    // headroom for shard alignment/padding without approving a second full
    // copy of the model. (The MTP track contract enforces its own tighter
    // 20 GiB `maximum_transformed_bytes` cap separately.)
    public static let defaultMaxTransformedWeightsBytes = 25 * 1024 * 1024 * 1024
    public static let defaultMaxSubmissionSourceBytes = 256 * 1024 * 1024
    // Diagnostic (non-ranking) real-valued score fields are published rounded to
    // this many significant figures. Submitted model code controls its own
    // latency/memory, so every full-precision analog field it can influence
    // (RAM, bandwidth, wall/preflight/correctness/TTFT seconds, hit rate) is a
    // covert channel for exfiltrating the hidden prompt/golden it sees. Coarsening
    // these -- which carry no ranking weight -- collapses each from ~30 bits to a
    // few. The ranking fields (decode/prefill seconds-per-token and speedups) are
    // left precise here on purpose; bounding that residual channel is a publishing/
    // rate-limit decision on the scoring backend, not a repo-side change.
    public static let publicDiagnosticSignificantFigures = 2
}

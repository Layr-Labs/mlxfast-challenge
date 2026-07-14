public enum MLXFastConstants {
    public static let referenceModelName = "gemma-4-31b-4bit"
    public static let defaultReferencePath = "reference_weights/gemma-4-31b-4bit"
    public static let defaultReferenceCachePath = ".cache/huggingface/hub/models--mlx-community--gemma-4-31b-4bit/snapshots/main"
    public static let defaultWeightsPath = "weights"
    public static let defaultGoldenPath = "correctness_golden.json"
    public static let defaultPublicCorrectnessPromptPath = "correctness_prompts/public_longcopy_gate_english_512.txt"
    public static let defaultPublicCorrectnessGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_256.json"
    public static let defaultPublicLocalSubmitGoldenPath = "correctness_prompts/public_longcopy_gate_english_512_1024.json"
    public static let defaultScorePath = "score.json"
    public static let defaultLocalIterateScorePath = "score.local-iterate.json"

    public static let vocabSize = 262_144
    public static let hiddenSize = 5_376
    public static let intermediateSize = 21_504
    public static let numHiddenLayers = 60
    public static let attentionHeads = 32
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
    // 64 (was 10, DeepSeek-era): baseline Gemma 4 31B 4-bit greedy
    // continuations of the raw (untemplated) hidden GPQA prompts rarely
    // express the selected option within 10 tokens -- ranked run 28813130022
    // judged 0/5 on an unmodified baseline. A 2026-07-06 baseline capture at
    // budgets 10/32/64 confirmed most candidates are cut off mid-sentence and
    // continue the option list rather than answer; 64 (the
    // correctnessMaxBehaviorSteps ceiling) gives the judge the most usable
    // candidate text at ~35s extra decode on the gates machine.
    public static let correctnessGPQAMaxNewTokens = 64
    // Semantic judging uses short hidden GPQA answers as a baseline-calibrated
    // gate for optimizations that preserve the exact prefix but damage answer
    // sense. Five cases keeps the full GitHub job near the 30-minute budget.
    // The captured answer is a prefix of the behavior-gate generation, so
    // semanticGPQAMaxNewTokens is only effective up to
    // correctnessGPQAMaxNewTokens (and must stay <= correctnessMaxBehaviorSteps).
    public static let semanticGPQACaseCount = 5
    public static let semanticGPQAMaxNewTokens = 64
    // Baseline-calibrated threshold, recalibrated 2026-07-09 after the hidden
    // GPQA reference object was regenerated on the M5 ranked runner from the
    // mlx-swift-lm-rebase reference model. Five judged official-runner
    // baseline observations of unmodified main (tip 3c94f4e) all scored 2/5
    // with identical per-case verdicts (cases 2 and 3 judged correct every
    // time): runs 29040771374, 29048752714, 29051462434, 29052276465, and
    // 29053091705. The threshold is min(observed) - 1 = 1: one judged case of
    // margin below the stable 2/5 baseline floor absorbs single-case judge
    // nondeterminism, while a submission that wrecks answer quality (0/5
    // judged) now fails the gate instead of merely being recorded. The prior
    // 0 ("aggregate-recording") value came from the pre-regeneration prompts,
    // where runs 28813130022 and 28817200585 judged the raw-completion Gemma
    // baseline 0/5. If the hidden prompts or the reference model change
    // again, a fresh judged official-runner baseline must recalibrate this
    // threshold.
    public static let semanticGPQAMinPassCount = 1
    public static let benchmarkPrefillPromptTokens = 512
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
    // boundaries when the selected oracle contains enough tokens.
    public static let experimentalMTPMaxConfiguredTotalTokens = 512
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
    // CACHED Gemma 4 31B 4-bit dense baseline. Calibration provenance: the
    // live M5 paired baseline as of 2026-07-12 -- the mean of the
    // `baseline_decode_seconds_per_token` / `baseline_prefill_seconds_per_token`
    // fields published by 12 consecutive successful ranked runs on the
    // self-hosted M5 Max (m5-bench), runs 29179374395 through 29197772284
    // (each field is measure-job's on-box timing of the pinned reference tree
    // in that run's session): decode clustered 0.04401-0.04411 (CV 0.08%),
    // prefill 0.001618-0.001626 (CV 0.17%). Supersedes the retired
    // tenki-macos-latest-xlarge VM-era values (prefill 0.010605031949609375 /
    // decode 0.1336139485703125, 2026-07-07), which were ~6.5x / ~3.0x slower
    // than the M5 and inflated local score estimates to ~3.7 for code that
    // ranks ~1.0.
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
    public static let officialBaselinePrefillSecondsPerToken = 0.0016216554767252605
    public static let officialBaselineDecodeSecondsPerToken = 0.04405625764973958
    public static let scorePrefillWeight = 0.25
    public static let scoreDecodeWeight = 0.75
    public static let scorePrefillSpeedupFloor = 0.95
    public static let scoreDecodeSpeedupFloor = 0.95
    // The Gemma 4 31B 4-bit text tower is ~17 GB; 25 GiB keeps ample headroom
    // for shard alignment/padding without approving a second full copy of the
    // model.
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

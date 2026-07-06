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
    // Baseline-calibrated threshold, recalibrated at the 64-token budget from
    // the official-runner baseline verification run 28817200585 (2026-07-06,
    // https://github.com/Layr-Labs/mlxfast-challenge-dev/actions/runs/28817200585):
    // unmodified baseline main judged 0/5. The DeepSeek-era baseline
    // established 3/5; run 28813130022 judged the Gemma baseline 0/5 at the
    // old 10-token budget, and the 64-token judged rerun confirms baseline
    // Gemma expresses no judged-correct answers on these raw-completion
    // hidden GPQA prompts, so the measured Gemma baseline -- and this gate's
    // floor -- is 0. Aggregate pass counts are still recorded; the pass/fail
    // check stays aggregate-recording until the hidden prompts change (for
    // example to the Gemma chat template), after which a fresh judged
    // official-runner baseline must recalibrate this threshold.
    public static let semanticGPQAMinPassCount = 0
    public static let benchmarkPrefillPromptTokens = 512
    // Scored decode is parent-measured wall time for decode setup plus this
    // many checked token steps. Charging setup prevents submitted model code
    // from precomputing future decode tokens in an unscored seed-prefill phase.
    public static let benchmarkDecodeSteps = 128
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
    // Official Gemma 4 31B 4-bit dense baseline, measured 2026-07-06 on the
    // ranked Blacksmith runner class (blacksmith-12vcpu-macos-26) via
    // gemma-baseline-timing-probe run 28809531890
    // (https://github.com/Layr-Labs/mlxfast-challenge-dev/actions/runs/28809531890):
    // the unmodified reference implementation at main commit
    // eff7e7f2c85a5a6cef11110442ba4624a6ab3986 (the Gemma migration merge, the
    // same commit the timing machine's paired-baseline ref pins), full
    // official 128-step timing path under the official sandbox/worker posture.
    // Supersedes the DeepSeek-era values (prefill 0.17330563175390626 /
    // decode 3.6366560638046876). With paired-baseline measurement active,
    // ranked speedups divide by the same-session measured reference; these
    // constants keep three roles: local-mode scoring, the gates-only
    // machine's placeholder timing, and the paired sanity-band anchor.
    public static let officialBaselinePrefillSecondsPerToken = 0.01010573933984375
    public static let officialBaselineDecodeSecondsPerToken = 0.131727461265625
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

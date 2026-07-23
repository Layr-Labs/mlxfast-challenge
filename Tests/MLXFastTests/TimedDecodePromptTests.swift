import Foundation
@testable import MLXFastCore
import Testing

private func timedPromptWorkflow() throws -> String {
    try String(contentsOfFile: ".github/workflows/benchmark.yml", encoding: .utf8)
}

private func timedPromptStep(
    _ workflow: String,
    from startMarker: String,
    to endMarker: String
) throws -> String {
    let start = try #require(workflow.range(of: startMarker))
    let end = try #require(
        workflow.range(of: endMarker, range: start.upperBound..<workflow.endIndex)
    )
    return String(workflow[start.lowerBound..<end.lowerBound])
}

@Test
func hiddenTimedPromptIsIndependentOfCorrectnessFixtures() throws {
    let workflow = try timedPromptWorkflow()
    let correctness = try #require(workflow.range(of: "- name: Correctness and gates"))
    let scrub = try #require(workflow.range(of: "- name: Scrub hidden material from bench workspace"))
    let prepareTimedPrompt = try #require(
        workflow.range(of: "- name: Prepare hidden timed decode prompt")
    )
    let measure = try #require(workflow.range(of: "- name: Timed paired benchmark (measure-job)"))

    #expect(correctness.lowerBound < scrub.lowerBound)
    #expect(scrub.lowerBound < prepareTimedPrompt.lowerBound)
    #expect(prepareTimedPrompt.lowerBound < measure.lowerBound)

    // Existing public and hidden correctness fixtures remain untouched.
    #expect(workflow.contains(
        "MLXFAST_PUBLIC_CORRECTNESS_PROMPT_PATH: correctness_prompts/public_longcopy_gate_english_512.txt"
    ))
    #expect(workflow.contains(
        "MLXFAST_PUBLIC_CORRECTNESS_GOLDEN_PATH: correctness_prompts/public_longcopy_gate_english_512_256.json"
    ))
    #expect(workflow.contains(
        "MLXFAST_CORRECTNESS_GOLDEN_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/hidden-golden-${{ env.MLXFAST_RAW_CORRECTNESS_GOLDEN_SHA256 }}.json"
    ))
    #expect(workflow.contains(
        "MLXFAST_GPQA_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/gpqa-reference-${{ env.MLXFAST_GPQA_REFERENCE_SHA256 }}.json"
    ))
}

@Test
func hiddenTimedPromptIsPinnedAndScopedToTrustedSteps() throws {
    let workflow = try timedPromptWorkflow()
    let steps = try #require(workflow.range(of: "\n    steps:"))
    let jobHeader = String(workflow[..<steps.lowerBound])
    let prepare = try timedPromptStep(
        workflow,
        from: "- name: Prepare hidden timed decode prompt",
        to: "- name: Wait for quiescence before timing"
    )

    #expect(!jobHeader.contains("MLXFAST_TIMED_DECODE_PROMPT_R2_PATH:"))
    #expect(
        workflow.components(
            separatedBy: "MLXFAST_TIMED_DECODE_PROMPT_R2_PATH: correctness_prompts/laguna-xs-2.1-serial-v2/timed-prompt-${{ env.MLXFAST_TIMED_DECODE_PROMPT_SHA256 }}.txt"
        ).count - 1 == 1
    )
    #expect(prepare.contains(".github/scripts/download-r2-object.sh"))
    #expect(prepare.contains("hidden timed decode prompt pin mismatch"))
    #expect(prepare.contains("shasum -a 256"))
    #expect(prepare.contains("wc -c"))
    #expect(jobHeader.contains(
        "MLXFAST_TIMED_DECODE_PROMPT_SHA256: __POOLSIDE_V2_TIMED_PROMPT_SHA256_PENDING__"
    ))
    #expect(jobHeader.contains(
        "MLXFAST_TIMED_DECODE_PROMPT_BYTES: \"0\""
    ))
    #expect(jobHeader.contains(
        "MLXFAST_TIMED_DECODE_PROMPT_PATH: /tmp/mlxfast-private-laguna-xs-2.1-serial-v2-${{ github.run_id }}-${{ github.run_attempt }}/timed_decode_prompt.txt"
    ))
}

@Test
func measureJobReceivesExplicitTimedTarget() throws {
    let workflow = try timedPromptWorkflow()
    let measure = try timedPromptStep(
        workflow,
        from: "- name: Timed paired benchmark (measure-job)",
        to: "- name: Overlay paired timing into final score"
    )

    #expect(workflow.contains(
        "MLXFAST_TIMED_DECODE_TARGET_ID: \(MLXFastConstants.benchmarkEvaluationTargetID)"
    ))
    #expect(measure.contains("--prompt \"${MLXFAST_TIMED_DECODE_PROMPT_PATH}\""))
    #expect(measure.contains("--prompt-sha256 \"${MLXFAST_TIMED_DECODE_PROMPT_SHA256}\""))
    #expect(measure.contains("--target-id \"${MLXFAST_TIMED_DECODE_TARGET_ID}\""))
    #expect(MLXFastConstants.benchmarkNGramSelfSimilarityOrders == [1, 2, 3])
    #expect(MLXFastConstants.benchmarkMaxPromptLookupHitRate == 0.03)
}

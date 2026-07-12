public enum Gemma4SubmissionControls {
    // Temporary validation hook: submitted branches may set this to a positive
    // value to prove the benchmark detects slower measured decode. Keep this in
    // the editable model surface, not in the trusted workflow inputs.
    public static var measuredDecodeDelayMilliseconds: Int {
        0
    }
}

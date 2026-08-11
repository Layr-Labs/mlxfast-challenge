import Foundation

private func lagunaOProjGeometryValue(
    _ key: String,
    allowed: [Int],
    default defaultValue: Int
) -> Int {
    guard let raw = ProcessInfo.processInfo.environment[key],
        let value = Int(raw),
        allowed.contains(value)
    else { return defaultValue }
    return value
}

let lagunaOProjRowsPerSimdgroup = lagunaOProjGeometryValue(
    "DARKBLOOM_OPROJ_ROWS_PER_SIMDGROUP",
    allowed: [1, 2, 4, 8, 16],
    default: 2)

// Two SIMDgroups are required to initialize the 64-entry gate vector before the barrier.
let lagunaOProjSimdgroups = lagunaOProjGeometryValue(
    "DARKBLOOM_OPROJ_SIMDGROUPS",
    allowed: [2, 4],
    default: 2)

let lagunaOProjThreads = 32 * lagunaOProjSimdgroups
let lagunaOProjGeometrySuffix =
    lagunaOProjRowsPerSimdgroup == 4 && lagunaOProjSimdgroups == 2
    ? ""
    : "_rps\(lagunaOProjRowsPerSimdgroup)ns\(lagunaOProjSimdgroups)"
let lagunaOProjResultInit = Array(
    repeating: "0.0f",
    count: lagunaOProjRowsPerSimdgroup
).joined(separator: ", ")

func lagunaOProjTiles(outVec: Int) -> Int? {
    let rowsPerTile = lagunaOProjRowsPerSimdgroup * lagunaOProjSimdgroups
    guard outVec % rowsPerTile == 0 else { return nil }
    return outVec / rowsPerTile
}

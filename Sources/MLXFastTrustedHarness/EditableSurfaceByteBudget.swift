import Foundation
import MLXFastCore

/// Deterministic byte budget over the editable submission surface, enforced
/// by the trusted CLI immediately before a participant worker launch.
///
/// The full kernel-bypass review policy is applied by the LLM judge in
/// `.github/scripts/run-submission-static-review.sh` (which only runs for
/// submissions/* dispatches); these mechanical caps are the launch-time
/// backstop that binds every ranked worker launch path, including dispatches
/// that never pass through the review step. Keep the default limits in sync
/// with the script's `MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_BYTES` and
/// `MLXFAST_SUBMISSION_STATIC_REVIEW_MAX_FILE_BYTES`.
public enum EditableSurfaceByteBudget {
    /// The unmodified full-stack surface is 2,139,781 bytes (~2.14 MB);
    /// 3.0 MB leaves 860,219 bytes for merged frontier work while still
    /// bounding lookup-table smuggling.
    public static let defaultMaxTotalBytes = 3_000_000
    public static let defaultMaxFileBytes = 524_288
    public static let defaultContractRelativePath = "benchmark.json"
}

public enum EditableSurfaceBudgetVerification: Equatable {
    case verified(totalBytes: Int, fileCount: Int)
    /// Nothing to check (no contract on disk). Official runs treat this as
    /// fatal; local invocations from outside a checkout proceed.
    case skipped(reason: String)
    case exceeded(reason: String)
}

/// Walk the contract's `editablePaths` under the contract's directory and
/// enforce the per-file and total byte caps over every regular file. Symlinks
/// and non-regular entries never count (the overlay and surface enforcement
/// reject them separately); a malformed contract fails closed.
public func verifyEditableSurfaceByteBudget(
    contractPath: String,
    maxTotalBytes: Int,
    maxFileBytes: Int
) -> EditableSurfaceBudgetVerification {
    let fileManager = FileManager.default
    guard let contractData = fileManager.contents(atPath: contractPath) else {
        return .skipped(reason: "no benchmark contract at \(contractPath)")
    }
    guard let contract = try? JSONSerialization.jsonObject(with: contractData)
        as? [String: Any],
        let editablePaths = contract["editablePaths"] as? [String],
        !editablePaths.isEmpty
    else {
        return .exceeded(
            reason: "benchmark contract at \(contractPath) has no usable editablePaths"
        )
    }

    let contractRoot = URL(fileURLWithPath: contractPath).deletingLastPathComponent()
    var totalBytes = 0
    var fileCount = 0

    func account(fileAt url: URL, sizeBytes: Int) -> EditableSurfaceBudgetVerification? {
        if sizeBytes > maxFileBytes {
            return .exceeded(
                reason: "editable file \(url.path) is \(sizeBytes) bytes, above the "
                    + "per-file static review limit \(maxFileBytes)"
            )
        }
        totalBytes += sizeBytes
        fileCount += 1
        if totalBytes > maxTotalBytes {
            return .exceeded(
                reason: "editable surface is at least \(totalBytes) bytes, above the "
                    + "static review limit \(maxTotalBytes)"
            )
        }
        return nil
    }

    for editablePath in editablePaths {
        let rootURL = contractRoot.appendingPathComponent(editablePath)
        var isDirectory = ObjCBool(false)
        guard fileManager.fileExists(atPath: rootURL.path, isDirectory: &isDirectory) else {
            continue
        }
        if !isDirectory.boolValue {
            guard let attributes = try? fileManager.attributesOfItem(atPath: rootURL.path),
                  attributes[.type] as? FileAttributeType == .typeRegular,
                  let sizeBytes = (attributes[.size] as? NSNumber)?.intValue
            else {
                continue
            }
            if let verdict = account(fileAt: rootURL, sizeBytes: sizeBytes) {
                return verdict
            }
            continue
        }
        guard let enumerator = fileManager.enumerator(atPath: rootURL.path) else {
            return .exceeded(
                reason: "could not enumerate editable path \(rootURL.path)"
            )
        }
        while let entry = enumerator.nextObject() as? String {
            guard let type = enumerator.fileAttributes?[.type] as? FileAttributeType,
                  type == .typeRegular,
                  let sizeBytes = (enumerator.fileAttributes?[.size] as? NSNumber)?.intValue
            else {
                continue
            }
            if let verdict = account(
                fileAt: rootURL.appendingPathComponent(entry),
                sizeBytes: sizeBytes
            ) {
                return verdict
            }
        }
    }
    return .verified(totalBytes: totalBytes, fileCount: fileCount)
}

import Darwin
import Foundation
import MLXFastCore

/// Issues macOS kernel read-ahead hints (fcntl F_RDADVISE) for the expert
/// tensor byte ranges that `DeepSeekRoutedExperts` is about to read through
/// `ExpertSlotBank`. Purely advisory: it never surfaces data and never
/// bypasses the bank, so `expert_bytes_read` metrics stay truthful. The
/// advisory pages land in the unified buffer cache, so the bank's subsequent
/// `pread` calls hit page cache instead of waiting on serial SSD latency.
///
/// Range resolution happens on the model thread (mirroring
/// `DeepSeekWeightLoader.expertLinearWeight` name/slice logic); only plain
/// integers cross onto the background queue, which exclusively owns the
/// per-shard file descriptors.
public final class ExpertPrefetcher {
    struct ByteRange {
        let shard: String
        let offset: Int
        let length: Int
    }

    private let expertBank: ExpertSlotBank
    private let referenceBaseURL: URL
    private let enabled: Bool
    private let queue = DispatchQueue(label: "mlxfast.expert.prefetch", qos: .userInitiated)
    private var fdsByShard: [String: Int32] = [:]  // accessed only on `queue`

    public init(
        expertBank: ExpertSlotBank,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) {
        self.expertBank = expertBank
        // Mirrors ExpertSlotBank's referenceBaseURL construction so advisories
        // target byte-identical files.
        self.referenceBaseURL = URL(
            fileURLWithPath: expertBank.manifest.referencePath
        ).standardizedFileURL
        self.enabled = environment["MLXFAST_EXPERT_PREFETCH"].map {
            !["0", "false", "no", "off"].contains($0.lowercased())
        } ?? true
    }

    deinit {
        let fds = fdsByShard.values
        queue.sync {
            for fd in fds {
                close(fd)
            }
        }
    }

    /// Call on the model thread right after routing indices materialize.
    public func prefetch(layerIndex: Int, expertIndices: [Int]) {
        guard enabled, !expertIndices.isEmpty else {
            return
        }
        var seen = Set<Int>()
        var ranges: [ByteRange] = []
        ranges.reserveCapacity(expertIndices.count * 6)
        for expertIndex in expertIndices where seen.insert(expertIndex).inserted {
            for projection in Self.projections {
                appendRanges(
                    layerIndex: layerIndex,
                    expertIndex: expertIndex,
                    projection: projection,
                    into: &ranges
                )
            }
        }
        guard !ranges.isEmpty else {
            return
        }
        ranges.sort { ($0.shard, $0.offset) < ($1.shard, $1.offset) }
        // Weak capture so a queued advisory can never hold the last strong
        // reference: if it did, deinit would run ON this serial queue and its
        // queue.sync fd cleanup would self-deadlock. Once deinit is reachable,
        // pending advisories see nil and no-op before the fds close.
        queue.async { [weak self] in
            self?.advise(ranges)
        }
    }

    // MARK: - Range resolution (model thread)

    nonisolated(unsafe) private static let projections:
        [(projection: DeepSeekExpertProjection, expectedRank: Int)] = [
            (.gate, 2), (.up, 2), (.down, 2),
        ]

    private func appendRanges(
        layerIndex: Int,
        expertIndex: Int,
        projection: (projection: DeepSeekExpertProjection, expectedRank: Int),
        into ranges: inout [ByteRange]
    ) {
        let candidates = DeepSeekWeightNames.routedExpert(
            layerIndex: layerIndex,
            expertIndex: expertIndex,
            projection: projection.projection
        )
        // First matching candidate wins, matching expertLinearWeight order.
        for candidate in candidates {
            guard let record = expertBank.record(named: candidate) else {
                continue
            }
            let isStacked = record.shape.count == projection.expectedRank + 1
                && record.shape.first.map { expertIndex < $0 } == true
            append(record: record, expertIndex: expertIndex, sliced: isStacked, into: &ranges)
            // Companion tensors are read only for quantized (U32) weights.
            if record.dtype == "U32" {
                for suffix in ["scales", "biases"] {
                    let companion = companionName(for: candidate, suffix: suffix)
                    if let companionRecord = expertBank.record(named: companion) {
                        append(
                            record: companionRecord,
                            expertIndex: expertIndex,
                            sliced: isStacked,
                            into: &ranges
                        )
                    }
                }
            }
            return
        }
    }

    private func append(
        record: ExpertTensorRecord,
        expertIndex: Int,
        sliced: Bool,
        into ranges: inout [ByteRange]
    ) {
        if sliced {
            // Same slice math as ExpertSlotBank.materializedTensor(named:firstAxisIndex:).
            guard let firstDimension = record.shape.first,
                  record.shape.count >= 2,
                  expertIndex >= 0, expertIndex < firstDimension,
                  record.byteLength % firstDimension == 0
            else {
                return
            }
            let sliceLength = record.byteLength / firstDimension
            ranges.append(
                ByteRange(
                    shard: record.shard,
                    offset: record.byteOffset + expertIndex * sliceLength,
                    length: sliceLength
                )
            )
        } else {
            ranges.append(
                ByteRange(
                    shard: record.shard,
                    offset: record.byteOffset,
                    length: record.byteLength
                )
            )
        }
    }

    private func companionName(for weightName: String, suffix: String) -> String {
        if weightName.hasSuffix(".weight") {
            return String(weightName.dropLast(".weight".count)) + ".\(suffix)"
        }
        return "\(weightName).\(suffix)"
    }

    // MARK: - Advisory syscalls (serial queue only)

    private func advise(_ ranges: [ByteRange]) {
        for range in ranges {
            guard range.length > 0, range.length <= Int(Int32.max),
                  let fd = descriptor(forShard: range.shard)
            else {
                continue
            }
            var advisory = radvisory(
                ra_offset: off_t(range.offset),
                ra_count: Int32(range.length)
            )
            _ = withUnsafeMutablePointer(to: &advisory) { pointer in
                fcntl(fd, F_RDADVISE, pointer)  // best effort; errors ignored
            }
        }
    }

    private func descriptor(forShard shard: String) -> Int32? {
        if let fd = fdsByShard[shard] {
            return fd
        }
        let shardURL = referenceBaseURL
            .appendingPathComponent(shard)
            .standardizedFileURL
        let fd = open(shardURL.path, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
        guard fd >= 0 else {
            return nil
        }
        var status = stat()
        guard fstat(fd, &status) == 0, (status.st_mode & S_IFMT) == S_IFREG else {
            close(fd)
            return nil
        }
        fdsByShard[shard] = fd
        return fd
    }
}

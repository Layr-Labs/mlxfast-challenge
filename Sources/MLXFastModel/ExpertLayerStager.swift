import Foundation
import MLXFastCore

/// Whole-stacked-tensor staging for prefill-shaped expert consumption.
///
/// A long prefill touches essentially every routed expert in every layer, so
/// reading a layer's experts as ~770 individual 0.5-4.5 MB slice preads wastes
/// the SSD: each pays open/fstat/pread/close and arrives in routing order
/// rather than file order. This stager instead reads each stacked expert code
/// tensor (~1 GiB) as ONE sequential pread on a background thread, one layer
/// ahead of consumption, and the consumer builds per-expert weights from
/// no-copy Data views into the staged buffer.
///
/// Byte identity: the trusted bank's firstAxisIndex slice read returns
/// bytes[byteOffset + i*(byteLength/firstDim), +byteLength/firstDim); a
/// whole-tensor read returns [byteOffset, byteOffset+byteLength), so the CPU
/// sub-range is the same file bytes by arithmetic identity. Weights built
/// from staged views therefore feed the exact bytes — and the exact same
/// kernels — as the streamed path.
///
/// Reads go through a dedicated capacity-0 ExpertSlotBank sharing the
/// loader's metrics: no LRU is mutated (capacity-0 banks never insert, which
/// also makes them safe to use from this background thread) and every staged
/// byte is recorded honestly on the same counters the benchmark reports.
public final class ExpertLayerStager {
    public struct LayerPlan {
        public let layerIndex: Int
        public let recordNames: [String]

        public init(layerIndex: Int, recordNames: [String]) {
            self.layerIndex = layerIndex
            self.recordNames = recordNames
        }
    }

    private let sideBank: ExpertSlotBank
    private let queue = DispatchQueue(label: "mlxfast.expert.stager", qos: .userInitiated)
    private let condition = NSCondition()
    // All three guarded by `condition`.
    private var stagedBytesByRecordName: [String: Data] = [:]
    private var recordNamesByLayer: [Int: [String]] = [:]
    private var pendingLayers: Set<Int> = []
    private var failedLayers: Set<Int> = []

    public init?(manifestPath: String, metrics: ExpertStreamingMetrics?) {
        guard let bank = try? ExpertSlotBank(
            manifestPath: manifestPath,
            capacity: 0,
            metrics: metrics
        ) else {
            return nil
        }
        self.sideBank = bank
    }

    /// Enqueues background staging for a layer. Non-blocking, so callers can
    /// schedule before their routing sync and let the sequential reads overlap
    /// GPU work. Duplicate schedules are ignored.
    public func schedule(_ plan: LayerPlan) {
        guard !plan.recordNames.isEmpty else {
            return
        }
        condition.lock()
        if !failedLayers.contains(plan.layerIndex) {
            scheduleLocked(plan)
        }
        condition.unlock()
    }

    /// Blocks until a scheduled layer is staged. Returns false when staging
    /// failed (or was never scheduled) — callers must fall back to the
    /// per-slice streaming path, which reproduces today's behavior exactly.
    public func waitForLayer(_ layerIndex: Int) -> Bool {
        condition.lock()
        while pendingLayers.contains(layerIndex) {
            condition.wait()
        }
        let isStaged = recordNamesByLayer[layerIndex] != nil
        failedLayers.remove(layerIndex)
        condition.unlock()
        return isStaged
    }

    /// Whole-tensor bytes for a staged record, or nil when not staged.
    public func stagedBytes(recordName: String) -> Data? {
        condition.lock()
        defer { condition.unlock() }
        return stagedBytesByRecordName[recordName]
    }

    /// Frees a consumed layer's staged buffers.
    public func releaseLayer(_ layerIndex: Int) {
        condition.lock()
        if let names = recordNamesByLayer.removeValue(forKey: layerIndex) {
            for name in names {
                stagedBytesByRecordName.removeValue(forKey: name)
            }
        }
        condition.unlock()
    }

    private func scheduleLocked(_ plan: LayerPlan) {
        guard
            recordNamesByLayer[plan.layerIndex] == nil,
            !pendingLayers.contains(plan.layerIndex)
        else {
            return
        }
        pendingLayers.insert(plan.layerIndex)
        queue.async { [self] in
            var loaded: [String: Data] = [:]
            var succeeded = true
            for name in plan.recordNames {
                guard let tensor = try? sideBank.materializedTensor(named: name) else {
                    succeeded = false
                    break
                }
                loaded[name] = tensor.bytes
            }
            condition.lock()
            pendingLayers.remove(plan.layerIndex)
            if succeeded {
                recordNamesByLayer[plan.layerIndex] = plan.recordNames
                for (name, bytes) in loaded {
                    stagedBytesByRecordName[name] = bytes
                }
            } else {
                failedLayers.insert(plan.layerIndex)
            }
            condition.broadcast()
            condition.unlock()
        }
    }
}

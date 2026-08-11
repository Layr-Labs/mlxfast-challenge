
import Cmlx
import Foundation
import MLX

public func dynamicSlice(
    _ src: MLXArray,
    start: MLXArray,
    axes: [Int32],
    sliceSize: [Int32],
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    var axesInt = axes.map { Int32($0) }
    var sizes = sliceSize.map { Int32($0) }
    let rc = mlx_slice_dynamic(
        &result,
        src.ctx,
        start.ctx,
        &axesInt,
        axesInt.count,
        &sizes,
        sizes.count,
        stream.ctx)
    if rc != 0 {
        fatalError("[dynamicSlice] mlx_slice_dynamic failed with rc=\(rc)")
    }
    return MLXArray(result)
}

public func dynamicSliceUpdate(
    _ src: MLXArray,
    update: MLXArray,
    start: MLXArray,
    axes: [Int32],
    stream: StreamOrDevice = .default
) -> MLXArray {
    var result = mlx_array_new()
    var axesInt = axes.map { Int32($0) }
    let rc = mlx_slice_update_dynamic(
        &result,
        src.ctx,
        update.ctx,
        start.ctx,
        &axesInt,
        axesInt.count,
        stream.ctx)
    if rc != 0 {
        fatalError("[dynamicSliceUpdate] mlx_slice_update_dynamic failed with rc=\(rc)")
    }
    return MLXArray(result)
}

import Metal

final class BitonicSort {
    // MARK: - Properties

    private let firstPass: FirstPass
    private let generalPass: GeneralPass
    private let finalPass: FinalPass

    init(
        library: MTLLibrary
    ) throws {
        self.firstPass = try .init(
            library: library
        )
        self.generalPass = try .init(
            library: library
        )
        self.finalPass = try .init(
            library: library
        )
    }

    func encode(
        data: MTLBuffer,
        count: Int,
        in commandBuffer: MTLCommandBuffer
    ) {
        let elementStride = data.length / count
        let gridSize = count >> 1
        let unitSize = min(
            gridSize,
            self.generalPass
                .pipelineState
                .maxTotalThreadsPerThreadgroup
        )

        var params = SIMD2<UInt32>(repeating: 1)

        self.firstPass.encode(
            data: data,
            elementStride: elementStride,
            gridSize: gridSize,
            unitSize: unitSize,
            in: commandBuffer
        )
        params.x = .init(unitSize << 1)

        while params.x < count {
            params.y = params.x
            params.x <<= 1
            repeat {
                if unitSize < params.y {
                    self.generalPass.encode(
                        data: data,
                        params: params,
                        gridSize: gridSize,
                        unitSize: unitSize,
                        in: commandBuffer
                    )
                    params.y >>= 1
                } else {
                    self.finalPass.encode(
                        data: data,
                        elementStride: elementStride,
                        params: params,
                        gridSize: gridSize,
                        unitSize: unitSize,
                        in: commandBuffer
                    )
                    params.y = .zero
                }
            } while params.y > .zero
        }
    }
    
    static func buffer(
        count: Int,
        options: MTLResourceOptions = .cpuCacheModeWriteCombined,
        bufferAllocator: MTLBufferAllocator
    ) throws -> (buffer: MTLBuffer, paddedCount: Int) {
        return try Self.buffer(
            count: count,
            paddingValue: SIMD2<UInt32>(UInt32.max, UInt32.max),
            bufferAllocator: bufferAllocator
        )
    }

    private static func buffer<T>(
        count: Int,
        paddingValue: T,
        bufferAllocator: MTLBufferAllocator
    ) throws -> (buffer: MTLBuffer, paddedCount: Int) {
        let paddedCount = 1 << UInt(ceil(log2f(.init(count))))
        var count = count
        if paddedCount > count {
            count += paddedCount - count
        }
        return try (
            buffer: bufferAllocator.buffer(for: T.self, count: count),
            paddedCount: paddedCount
        )
    }
}

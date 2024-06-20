import MetalTools

public final class SpatialHashing {
    private let computeVertexHashState: MTLComputePipelineState
    private let findCellBoundsState: MTLComputePipelineState
    private let cacheCollisionsState: MTLComputePipelineState
    private let storeHalfPositionsState: MTLComputePipelineState
    private let storeSortedHalfPositionsState: MTLComputePipelineState
    
    private let bitonicSort: BitonicSort

    private let halfPositions: MTLBuffer
    private let halfSortedPositions: MTLBuffer

    private let cellStart: MTLBuffer
    private let cellEnd: MTLBuffer
    private let hashTable: (buffer: MTLBuffer, paddedCount: Int)
    
    private let hashTableCapacity: Int
    private let cellSize: Float
    private let collisionType: SelfCollisionType
    private let spacingScale: Float
    
    /// Initializes a new `SpatialHashing` instance.
    ///
    /// - Parameters:
    ///   - device: The Metal device.
    ///   - collisionType: The type of collision detection to use.
    ///   - positions: An array of vertex positions.
    ///   - cellSize: The size of a cell in spatial grid.
    ///   - spacingScale: The scale factor for cell spacing.
    ///   - heap: The Metal heap for resource allocation.
    /// - Throws: An error if the Metal library or pipeline states cannot be created.
    public init(
        device: MTLDevice,
        collisionType: SelfCollisionType,
        positions: [SIMD4<Float>],
        spacingScale: Float = 1,
        cellSize: Float,
        heap: MTLHeap?
    ) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
        let vertexCount = positions.count

        self.spacingScale = spacingScale
        self.collisionType = collisionType
        self.cellSize = cellSize
        computeVertexHashState = try library.computePipelineState(function: "computeVertexHash")
        findCellBoundsState = try library.computePipelineState(function: "findCellBounds")
        cacheCollisionsState = try library.computePipelineState(function: "cacheCollisions")
        storeHalfPositionsState = try library.computePipelineState(function: "storeHalfPositions")
        storeSortedHalfPositionsState = try library.computePipelineState(function: "storeSortedHalfPositions")

        bitonicSort = try .init(library: library)
        
        hashTableCapacity = vertexCount
        hashTable = try BitonicSort.buffer(count: vertexCount, device: device, heap: heap)
        cellStart = try device.buffer(for: UInt32.self, count: hashTableCapacity, heap: heap)
        cellEnd = try device.buffer(for: UInt32.self, count: hashTableCapacity, heap: heap)
        halfPositions = try device.buffer(for: SIMD4<Float16>.self, count: vertexCount, heap: heap)
        halfSortedPositions = try device.buffer(for: SIMD4<Float16>.self, count: vertexCount, heap: heap)
    }
    
    /// Builds the spatial hash and collision pairs for the given positions.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to encode the commands into.
    ///   - positions: The buffer containing vertex positions.
    ///   - collisionCandidates: The buffer to store collision pairs.
    ///   - connectedVertices: The buffer containing vertex neighborhood information.
    public func build(
        commandBuffer: MTLCommandBuffer,
        positions: TypedMTLBuffer<SIMD4<Float>>,
        collisionCandidates: TypedMTLBuffer<UInt32>,
        connectedVertices: TypedMTLBuffer<UInt32>?
    ) {
        commandBuffer.compute { encoder in
            encoder.setBuffer(positions.buffer, offset: 0, index: 0)
            encoder.setBuffer(halfPositions, offset: 0, index: 1)
            encoder.dispatch1d(state: storeHalfPositionsState, exactlyOrCovering: positions.count)
            
            encoder.setBuffer(halfPositions, offset: 0, index: 0)
            encoder.setBuffer(hashTable.buffer, offset: 0, index: 1)
            encoder.setValue(UInt32(hashTableCapacity), at: 2)
            encoder.setValue(cellSize, at: 3)
            encoder.dispatch1d(state: computeVertexHashState, exactlyOrCovering: positions.count)
        }

        bitonicSort.encode(data: hashTable.buffer, count: hashTable.paddedCount, in: commandBuffer)

        commandBuffer.compute { encoder in
            encoder.setBuffer(halfPositions, offset: 0, index: 0)
            encoder.setBuffer(halfSortedPositions, offset: 0, index: 1)
            encoder.setBuffer(hashTable.buffer, offset: 0, index: 2)
            encoder.dispatch1d(state: storeSortedHalfPositionsState, exactlyOrCovering: positions.count)
            
            encoder.setBuffer(cellStart, offset: 0, index: 0)
            encoder.setBuffer(cellEnd, offset: 0, index: 1)
            encoder.setValue(UInt32(positions.count), at: 3)

            let threadgroupWidth = 256
            encoder.setThreadgroupMemoryLength((threadgroupWidth + 16) * MemoryLayout<UInt32>.size, index: 0)
            encoder.dispatch1d(state: findCellBoundsState, exactlyOrCovering: positions.count, threadgroupWidth: threadgroupWidth)

            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(hashTable.buffer, offset: 0, index: 1)
            encoder.setBuffer(cellStart, offset: 0, index: 2)
            encoder.setBuffer(cellEnd, offset: 0, index: 3)
            encoder.setBuffer(halfSortedPositions, offset: 0, index: 4)
            if let connectedVertices {
                encoder.setBuffer(connectedVertices.buffer, offset: 0, index: 5)
            } else {
                encoder.setValue([UInt32.zero], at: 5)
            }
            encoder.setValue(UInt32(hashTableCapacity), at: 6)
            encoder.setValue(spacingScale, at: 7)
            encoder.setValue(cellSize, at: 8)
            encoder.setValue(UInt32(collisionCandidates.count / positions.count), at: 9)
            encoder.setValue(UInt32((connectedVertices?.count ?? 0) / positions.count), at: 10)

            encoder.dispatch1d(state: cacheCollisionsState, exactlyOrCovering: positions.count)
        }
    }
}

public extension SpatialHashing {
    static func totalBuffersSize(positionsCount: Int) -> Int {
        let halfPositionsSize = positionsCount * MemoryLayout<SIMD4<Float16>>.stride * 2
        let cellStartSize = positionsCount * MemoryLayout<UInt32>.stride
        let cellEndSize = positionsCount * MemoryLayout<UInt32>.stride
        let hashTableSize = positionsCount * MemoryLayout<SIMD2<UInt32>>.stride * 2
        
        return halfPositionsSize + cellStartSize + cellEndSize + hashTableSize
    }
}

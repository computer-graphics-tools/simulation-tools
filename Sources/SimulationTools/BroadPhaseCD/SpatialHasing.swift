import MetalTools

public final class SpatialHashing {
    private let computeParticleHashState: MTLComputePipelineState
    private let findCellStartState: MTLComputePipelineState
    private let cacheCollisionsState: MTLComputePipelineState
    private let reuseTrianglesCacheState: MTLComputePipelineState
    private let storeHalfPositionsState: MTLComputePipelineState
    private let bitonicSort: BitonicSort

    private let halfPositions: MTLBuffer
    private let cellStart: MTLBuffer
    private let cellEnd: MTLBuffer
    public let hashTable: (buffer: MTLBuffer, paddedCount: Int)
    
    private let hashTableCapacity: Int
    private let gridCellSpacing: Float
    private let collisionType: SelfCollisionType
    private let spacingScale: Float
    
    /// Initializes a new `SpatialHashing` instance.
    ///
    /// - Parameters:
    ///   - device: The Metal device.
    ///   - collisionType: The type of collision detection to use.
    ///   - positions: An array of particle positions.
    ///   - triangles: An array of triangle indices.
    ///   - spacingScale: The scale factor for grid cell spacing.
    ///   - heap: The Metal heap for resource allocation.
    /// - Throws: An error if the Metal library or pipeline states cannot be created.
    public init(
        device: MTLDevice,
        collisionType: SelfCollisionType,
        positions: [SIMD4<Float>],
        triangles: [SIMD3<UInt32>],
        spacingScale: Float = 1,
        heap: MTLHeap?
    ) throws {
        let library = try device.makeDefaultLibrary(bundle: .main)
        let vertexCount = positions.count
        let trianglesCount = triangles.count

        self.spacingScale = spacingScale
        self.collisionType = collisionType
        computeParticleHashState = try library.computePipelineState(function: "computeParticleHash")
        findCellStartState = try library.computePipelineState(function: "findCellStart")
        cacheCollisionsState = try library.computePipelineState(
            function: collisionType == .vertexVertex ? "cacheCollisions" : "cacheTriangleCollisions"
        )
        reuseTrianglesCacheState = try library.computePipelineState(function: "reuseTrianglesCache")
        storeHalfPositionsState = try library.computePipelineState(function: "storeHalfPositions")
                
        bitonicSort = try .init(library: library)
        
        hashTableCapacity = vertexCount
        hashTable = try BitonicSort.buffer(count: vertexCount, device: device, heap: heap)
        cellStart = try device.buffer(for: UInt32.self, count: hashTableCapacity, heap: heap)
        cellEnd = try device.buffer(for: UInt32.self, count: hashTableCapacity, heap: heap)
        halfPositions = try device.buffer(for: SIMD4<Float16>.self, count: vertexCount, heap: heap)

        let gridCellSpacingSum = triangles.reduce(Float.zero) { totalSum, triangleVertices in
            let edgeLengthSum = (0..<3).reduce(0.0) { sum, index in
                sum + length(positions[triangleVertices[index]].xyz - positions[triangleVertices[(index + 1) % 3]].xyz)
            } / 3.0
            
            return totalSum + edgeLengthSum
        }
        
        gridCellSpacing = (gridCellSpacingSum / Float(trianglesCount))
    }
    
    /// Builds the spatial hash and collision pairs for the given positions and triangles.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to encode the commands into.
    ///   - positions: The buffer containing particle positions.
    ///   - collisionPairs: The buffer to store collision pairs.
    ///   - vertexNeighborhood: The buffer containing vertex neighborhood information.
    ///   - triangleNeighborhood: The buffer containing triangle neighborhood information.
    ///   - triangles: The buffer containing triangle indices.
    ///   - positionsCount: The number of positions.
    ///   - trianglesCount: The number of triangles.
    public func build(
        commandBuffer: MTLCommandBuffer,
        positions: MTLBuffer,
        collisionPairs: MTLBuffer,
        vertexNeighborhood: MTLBuffer,
        triangleNeighborhood: MTLBuffer,
        triangles: MTLBuffer,
        positionsCount: Int,
        trianglesCount: Int
    ) {
        commandBuffer.compute { encoder in
            encoder.setBuffer(positions, offset: 0, index: 0)
            encoder.setBuffer(halfPositions, offset: 0, index: 1)
            encoder.dispatch1d(state: storeHalfPositionsState, exactly: positionsCount)
            
            encoder.setBuffer(halfPositions, offset: 0, index: 0)
            encoder.setBuffer(hashTable.buffer, offset: 0, index: 1)
            encoder.setValue(UInt32(hashTableCapacity), at: 2)
            encoder.setValue(gridCellSpacing, at: 3)
            encoder.dispatch1d(state: computeParticleHashState, exactly: positionsCount)
        }

        bitonicSort.encode(data: hashTable.buffer, count: hashTable.paddedCount, in: commandBuffer)

        commandBuffer.compute { encoder in
            encoder.setBuffer(cellStart, offset: 0, index: 0)
            encoder.setBuffer(cellEnd, offset: 0, index: 1)
            encoder.setBuffer(hashTable.buffer, offset: 0, index: 2)
            encoder.setValue(UInt32(positionsCount), at: 3)
            let threadgroupWidth = 256
            encoder.setThreadgroupMemoryLength((threadgroupWidth + 16) * MemoryLayout<UInt32>.size, index: 0)
            encoder.dispatch1d(state: findCellStartState, exactly: positionsCount, threadgroupWidth: threadgroupWidth)

            encoder.setBuffer(collisionPairs, offset: 0, index: 0)
            encoder.setBuffer(hashTable.buffer, offset: 0, index: 1)
            encoder.setBuffer(cellStart, offset: 0, index: 2)
            encoder.setBuffer(cellEnd, offset: 0, index: 3)
            encoder.setBuffer(halfPositions, offset: 0, index: 4)
            encoder.setBuffer(vertexNeighborhood, offset: 0, index: 5)
            encoder.setValue(UInt32(hashTableCapacity), at: 6)
            encoder.setValue(spacingScale, at: 7)
            encoder.setValue(gridCellSpacing, at: 8)

            if collisionType == .vertexVertex {
                encoder.dispatch1d(state: cacheCollisionsState, exactly: positionsCount)
            } else {
                encoder.setBuffer(triangles, offset: 0, index: 9)
                encoder.dispatch1d(state: cacheCollisionsState, exactly: trianglesCount)
            }
        }
    }

    /// Updates the collision pairs for vertex-triangle collisions by performing spatial-temporal reuse.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to encode the commands into.
    ///   - positions: The buffer containing particle positions.
    ///   - collisionPairs: The buffer to store collision pairs.
    ///   - vertexNeighborhood: The buffer containing vertex neighborhood information.
    ///   - triangleNeighborhood: The buffer containing triangle neighborhood information.
    ///   - triangles: The buffer containing triangle indices.
    ///   - positionsCount: The number of positions.
    ///   - trianglesCount: The number of triangles.
    public func update(
        commandBuffer: MTLCommandBuffer,
        positions: MTLBuffer,
        collisionPairs: MTLBuffer,
        vertexNeighborhood: MTLBuffer,
        triangleNeighborhood: MTLBuffer,
        triangles: MTLBuffer,
        positionsCount: Int,
        trianglesCount: Int
    ) {
        if collisionType == .vertexTriangle {
            commandBuffer.compute { encoder in
                encoder.setBuffer(collisionPairs, offset: 0, index: 0)
                encoder.setBuffer(hashTable.buffer, offset: 0, index: 1)
                encoder.setBuffer(cellStart, offset: 0, index: 2)
                encoder.setBuffer(cellEnd, offset: 0, index: 3)
                encoder.setBuffer(positions, offset: 0, index: 4)
                encoder.setBuffer(vertexNeighborhood, offset: 0, index: 5)
                encoder.setBuffer(triangles, offset: 0, index: 6)
                encoder.setBuffer(triangleNeighborhood, offset: 0, index: 7)
                encoder.setValue(UInt32(hashTableCapacity), at: 8)
                encoder.setValue(spacingScale, at: 9)
                encoder.setValue(gridCellSpacing, at: 10)

                encoder.dispatch1d(state: reuseTrianglesCacheState, exactly: trianglesCount)
            }
        }
    }
}

public extension SpatialHashing {
    static func totalBuffersSize(positionsCount: Int) -> Int {
        let halfPositionsSize = positionsCount * MemoryLayout<SIMD4<Float16>>.stride
        let cellStartSize = positionsCount * MemoryLayout<UInt32>.stride
        let cellEndSize = positionsCount * MemoryLayout<UInt32>.stride
        let hashTableSize = positionsCount * MemoryLayout<SIMD2<UInt32>>.stride
        
        return halfPositionsSize + cellStartSize + cellEndSize + hashTableSize
    }
}

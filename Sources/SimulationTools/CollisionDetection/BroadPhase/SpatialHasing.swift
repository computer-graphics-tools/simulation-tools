import MetalTools

public final class SpatialHashing {
    public struct Configuration {
        let cellSize: Float
        let spacingScale: Float
        let collisionType: SelfCollisionType
        
        public init(
            cellSize: Float32,
            spacingScale: Float32 = 1,
            collisionType: SelfCollisionType = .vertexToVertex
        ) {
            self.cellSize = cellSize
            self.spacingScale = spacingScale
            self.collisionType = collisionType
        }
    }

    public let configuration: Configuration

    private let computeVertexHashAndIndexState: MTLComputePipelineState
    private let computeCellBoundariesState: MTLComputePipelineState
    private let findCollisionCandidatesState: MTLComputePipelineState
    private let convertToHalfPrecisionPositionsState: MTLComputePipelineState
    private let reorderHalfPrecisionPositionsState: MTLComputePipelineState
    
    private let bitonicSort: BitonicSort

    private let halfPositions: MTLBuffer
    private let sortedHalfPositions: MTLBuffer

    private let cellStart: MTLBuffer
    private let cellEnd: MTLBuffer
    private let hashTable: (buffer: MTLBuffer, paddedCount: Int)
    private let hashTableCapacity: Int

    /// Initializes a new `SpatialHashing` instance.
    ///
    /// - Parameters:
    ///   - heap: The Metal heap for resource allocation.
    ///   - configuration: The configuration for spatial hashing.
    ///   - positions: An array of vertex positions.
    /// - Throws: An error if the Metal library or pipeline states cannot be created.
    public convenience init(
        heap: MTLHeap,
        configuration: Configuration,
        positions: [SIMD4<Float>]
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .heap(heap)),
            configuration: configuration,
            positions: positions
        )
    }
    
    /// Initializes a new `SpatialHashing` instance.
    ///
    /// - Parameters:
    ///   - device: The Metal device for resource allocation.
    ///   - configuration: The configuration for spatial hashing.
    ///   - positions: An array of vertex positions.
    /// - Throws: An error if the Metal library or pipeline states cannot be created.
    public convenience init(
        device: MTLDevice,
        configuration: Configuration,
        positions: [SIMD4<Float>]
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .device(device)),
            configuration: configuration,
            positions: positions
        )
    }

    private init(
        bufferAllocator: MTLBufferAllocator,
        configuration: Configuration,
        positions: [SIMD4<Float>],
        heap: MTLHeap? = nil
    ) throws {
        let library = try bufferAllocator.device.makeDefaultLibrary(bundle: .module)
        let deviceSupportsNonuniformThreadgroups = library.device
            .supports(feature: .nonUniformThreadgroups)

        let constantValues = MTLFunctionConstantValues()
        constantValues.set(deviceSupportsNonuniformThreadgroups, at: 0)

        let vertexCount = positions.count

        self.configuration = configuration
        self.computeVertexHashAndIndexState = try library.computePipelineState(
            function: "computeVertexHashAndIndex",
            constants: constantValues
        )
        self.computeCellBoundariesState = try library.computePipelineState(
            function: "computeCellBoundaries",
            constants: constantValues
        )
        self.findCollisionCandidatesState = try library.computePipelineState(
            function: "findCollisionCandidates",
            constants: constantValues
        )
        self.convertToHalfPrecisionPositionsState = try library.computePipelineState(
            function: "convertToHalfPrecisionPositions",
            constants: constantValues
        )
        self.reorderHalfPrecisionPositionsState = try library.computePipelineState(
            function: "reorderHalfPrecisionPositions",
            constants: constantValues
        )

        self.bitonicSort = try .init(library: library)
        
        self.hashTableCapacity = vertexCount * 2
        self.hashTable = try BitonicSort.buffer(count: vertexCount, bufferAllocator: bufferAllocator)
        self.cellStart = try bufferAllocator.buffer(for: UInt32.self, count: self.hashTableCapacity)
        self.cellEnd = try bufferAllocator.buffer(for: UInt32.self, count: self.hashTableCapacity)
        self.halfPositions = try bufferAllocator.buffer(for: SIMD4<Float16>.self, count: vertexCount)
        self.sortedHalfPositions = try bufferAllocator.buffer(for: SIMD4<Float16>.self, count: vertexCount)
    }
    
    /// Builds the spatial hash and collision pairs for the given positions.
    /// - Parameters:
    ///   - commandBuffer: The Metal command buffer to encode the commands into.
    ///   - positions: The buffer containing vertex positions.
    ///   - collisionCandidates: The buffer to store collision pairs.
    ///   - connectedVertices: The buffer containing vertex neighborhood information.
    public func build(
        positions: MTLTypedBuffer<SIMD4<Float>>,
        collisionCandidates: MTLTypedBuffer<UInt32>,
        connectedVertices: MTLTypedBuffer<UInt32>?,
        in commandBuffer: MTLCommandBuffer
    ) {
        commandBuffer.pushDebugGroup("Convert To Half Precision & Compute Vertex Hash And Index")
        commandBuffer.compute { encoder in
            encoder.setBuffer(positions.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.halfPositions, offset: 0, index: 1)
            encoder.setValue(UInt32(positions.count), at: 2)
            encoder.dispatch1d(state: self.convertToHalfPrecisionPositionsState, exactlyOrCovering: positions.count)

            encoder.setBuffer(self.halfPositions, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 1)
            encoder.setValue(UInt32(self.hashTableCapacity), at: 2)
            encoder.setValue(self.configuration.cellSize, at: 3)
            encoder.setValue(UInt32(positions.count), at: 4)
            encoder.dispatch1d(state: self.computeVertexHashAndIndexState, exactlyOrCovering: positions.count)
        }
        commandBuffer.popDebugGroup()
        
        commandBuffer.pushDebugGroup("Sort Hash Table")
        self.bitonicSort.encode(data: self.hashTable.buffer, count: self.hashTable.paddedCount, in: commandBuffer)
        commandBuffer.popDebugGroup()
        
        commandBuffer.pushDebugGroup("Compute Cell Bounds & Find Collision Candidates")
        commandBuffer.compute { encoder in
            encoder.setBuffer(self.halfPositions, offset: 0, index: 0)
            encoder.setBuffer(self.sortedHalfPositions, offset: 0, index: 1)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 2)
            encoder.setValue(UInt32(positions.count), at: 3)
            encoder.dispatch1d(state: self.reorderHalfPrecisionPositionsState, exactlyOrCovering: positions.count)

            let threadgroupWidth = 256
            encoder.setBuffer(self.cellStart, offset: 0, index: 0)
            encoder.setBuffer(self.cellEnd, offset: 0, index: 1)
            encoder.setThreadgroupMemoryLength((threadgroupWidth + 16) * MemoryLayout<UInt32>.size, index: 0)
            encoder.dispatch1d(state: self.computeCellBoundariesState, exactlyOrCovering: positions.count, threadgroupWidth: threadgroupWidth)

            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 1)
            encoder.setBuffer(self.cellStart, offset: 0, index: 2)
            encoder.setBuffer(self.cellEnd, offset: 0, index: 3)
            encoder.setBuffer(self.sortedHalfPositions, offset: 0, index: 4)
            if let connectedVertices {
                encoder.setBuffer(connectedVertices.buffer, offset: 0, index: 5)
            } else {
                encoder.setValue([UInt32.zero], at: 5)
            }
            encoder.setValue(UInt32(self.hashTableCapacity), at: 6)
            encoder.setValue(self.configuration.spacingScale, at: 7)
            encoder.setValue(self.configuration.cellSize, at: 8)
            encoder.setValue(UInt32(collisionCandidates.count / positions.count), at: 9)
            encoder.setValue(UInt32((connectedVertices?.count ?? 0) / positions.count), at: 10)
            encoder.setValue(UInt32(positions.count), at: 11)

            encoder.dispatch1d(state: self.findCollisionCandidatesState, exactlyOrCovering: positions.count)
        }
        commandBuffer.popDebugGroup()
    }
}

public extension SpatialHashing {
    /// Calculates the total size of buffers required for spatial hashing.
    ///
    /// - Parameter positionsCount: The number of positions to hash.
    /// - Returns: The total size of buffers in bytes.
    static func totalBuffersSize(positionsCount: Int) -> Int {
        let halfPositionsSize = positionsCount * MemoryLayout<SIMD4<Float16>>.stride * 2
        let cellStartSize = positionsCount * MemoryLayout<UInt32>.stride * 2
        let cellEndSize = positionsCount * MemoryLayout<UInt32>.stride * 2
        let hashTableSize = positionsCount * MemoryLayout<SIMD2<UInt32>>.stride * 2
        
        return halfPositionsSize + cellStartSize + cellEndSize + hashTableSize
    }
}

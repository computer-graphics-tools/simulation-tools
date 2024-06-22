import MetalTools

public final class SpatialHashing {
    public struct Configuration {
        let cellSize: Float
        let spacingScale: Float
        let collisionType: SelfCollisionType
        
        public init(cellSize: Float, spacingScale: Float = 1, collisionType: SelfCollisionType) {
            self.cellSize = cellSize
            self.spacingScale = spacingScale
            self.collisionType = collisionType
        }
    }

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
    private let configuration: Configuration
    
    /// Initializes a new `SpatialHashing` instance.
    ///
    /// - Parameters:
    ///   - device: The Metal device.
    ///   - configuration: The configuration for spatial hashing.
    ///   - positions: An array of vertex positions.
    ///   - heap: The Metal heap for resource allocation.
    /// - Throws: An error if the Metal library or pipeline states cannot be created.
    public init(
        device: MTLDevice,
        configuration: Configuration,
        positions: [SIMD4<Float>],
        heap: MTLHeap?
    ) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
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
        self.hashTable = try BitonicSort.buffer(count: vertexCount, device: device, heap: heap)
        self.cellStart = try device.buffer(for: UInt32.self, count: self.hashTableCapacity, heap: heap)
        self.cellEnd = try device.buffer(for: UInt32.self, count: self.hashTableCapacity, heap: heap)
        self.halfPositions = try device.buffer(for: SIMD4<Float16>.self, count: vertexCount, heap: heap)
        self.sortedHalfPositions = try device.buffer(for: SIMD4<Float16>.self, count: vertexCount, heap: heap)
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

        self.bitonicSort.encode(data: self.hashTable.buffer, count: self.hashTable.paddedCount, in: commandBuffer)

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

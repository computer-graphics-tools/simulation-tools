import Metal
import MetalTools

public final class SpatialHashing {
    /// Configuration for the SpatialHashing algorithm.
    public struct Configuration {
        /// The size of each cell in the spatial grid.
        let cellSize: Float
        /// The radius used for collision detection.
        let radius: Float
        
        /// Initializes a new Configuration instance.
        /// - Parameters:
        ///   - cellSize: The size of each cell in the spatial grid.
        ///   - radius: The radius used for collision detection.
        public init(cellSize: Float, radius: Float) {
            self.cellSize = cellSize
            self.radius = radius
        }
    }

    public let configuration: Configuration

    private let computeHashAndIndexState: MTLComputePipelineState
    private let computeCellBoundariesState: MTLComputePipelineState
    private let convertToHalfState: MTLComputePipelineState
    private let reorderHalfPrecisionState: MTLComputePipelineState
    private let findCollisionCandidatesState: MTLComputePipelineState
    private let bitonicSort: BitonicSort

    private let cellStart: MTLBuffer
    private let cellEnd: MTLBuffer
    private let hashTable: (buffer: MTLBuffer, paddedCount: Int)
    private let halfPositions: MTLBuffer
    private let sortedHalfPositions: MTLTypedBuffer
    private let hashTableCapacity: Int
    
    /// Initializes a new instance of SpatialHashing using the specified Metal device.
    ///
    /// - Parameters:
    ///   - device: The Metal device to use for computations.
    ///   - configuration: The configuration for spatial hashing.
    ///   - maxPositionsCount: The maximum number of positions that can be handled.
    /// - Throws: An error if initialization fails.
    public convenience init(
        heap: MTLHeap,
        configuration: Configuration,
        maxPositionsCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .heap(heap)),
            configuration: configuration,
            maxPositionsCount: maxPositionsCount
        )
    }
    
    /// Initializes a new instance of SpatialHashing using the specified Metal heap.
    ///
    /// - Parameters:
    ///   - heap: The Metal heap to allocate resources from.
    ///   - configuration: The configuration for spatial hashing.
    ///   - maxPositionsCount: The maximum number of positions that can be handled.
    /// - Throws: An error if initialization fails.
    public convenience init(
        device: MTLDevice,
        configuration: Configuration,
        maxPositionsCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .device(device)),
            configuration: configuration,
            maxPositionsCount: maxPositionsCount
        )
    }

    private init(
        bufferAllocator: MTLBufferAllocator,
        configuration: Configuration,
        maxPositionsCount: Int
    ) throws {
        let library = try bufferAllocator.device.makeDefaultLibrary(bundle: .module)
        let deviceSupportsNonuniformThreadgroups = bufferAllocator.device.supports(feature: .nonUniformThreadgroups)

        let constantValues = MTLFunctionConstantValues()
        constantValues.set(deviceSupportsNonuniformThreadgroups, at: 0)

        self.configuration = configuration
        self.computeHashAndIndexState = try library.computePipelineState(function: "computeHashAndIndexState", constants: constantValues)
        self.computeCellBoundariesState = try library.computePipelineState(function: "computeCellBoundaries", constants: constantValues)
        self.convertToHalfState = try library.computePipelineState(function: "convertToHalf", constants: constantValues)
        self.reorderHalfPrecisionState = try library.computePipelineState(function: "reorderHalfPrecision", constants: constantValues)
        self.findCollisionCandidatesState = try library.computePipelineState(function: "findCollisionCandidates", constants: constantValues)
        self.bitonicSort = try .init(library: library)
        
        self.hashTableCapacity = maxPositionsCount * 2
        self.hashTable = try BitonicSort.buffer(count: maxPositionsCount, bufferAllocator: bufferAllocator)
        self.cellStart = try bufferAllocator.buffer(for: UInt32.self, count: self.hashTableCapacity)
        self.cellEnd = try bufferAllocator.buffer(for: UInt32.self, count: self.hashTableCapacity)
        self.halfPositions = try bufferAllocator.buffer(for: SIMD3<Float16>.self, count: maxPositionsCount)
        self.sortedHalfPositions = try bufferAllocator.typedBuffer(descriptor: .init(valueType: .half3, count: maxPositionsCount))
    }
    
    /// Builds the spatial hash structure for the given positions.
    ///
    /// - Parameters:
    ///   - positions: A buffer containing the positions.
    ///   - commandBuffer: The command buffer to encode the operation into.
    public func build(
        positions: MTLTypedBuffer,
        in commandBuffer: MTLCommandBuffer
    ) {
        let positionsPacked = positions.descriptor.valueType.isPacked
        
        commandBuffer.blit { encoder in
            encoder.fill(buffer: self.hashTable.buffer, range: 0..<self.hashTable.buffer.length, value: .max)
        }
    
        commandBuffer.pushDebugGroup("Compute Hash")
        commandBuffer.compute { encoder in
            encoder.setBuffer(positions.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.halfPositions, offset: 0, index: 1)
            encoder.setValue(UInt32(positions.descriptor.count), at: 2)
            encoder.setValue(positionsPacked, at: 3)
            encoder.dispatch1d(state: self.convertToHalfState, exactlyOrCovering: positions.descriptor.count)
            
            encoder.setBuffer(self.halfPositions, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 1)
            encoder.setValue(UInt32(self.hashTableCapacity), at: 2)
            encoder.setValue(self.configuration.cellSize, at: 3)
            encoder.setValue(UInt32(positions.descriptor.count), at: 4)
            encoder.dispatch1d(state: self.computeHashAndIndexState, exactlyOrCovering: positions.descriptor.count)
        }
        commandBuffer.popDebugGroup()

        commandBuffer.pushDebugGroup("Sort Hashes And Values")

        self.bitonicSort.encode(data: self.hashTable.buffer, count: self.hashTable.paddedCount, in: commandBuffer)
        commandBuffer.popDebugGroup()

        commandBuffer.pushDebugGroup("Compute Cell Bounds")
        commandBuffer.compute { encoder in
            encoder.setBuffer(self.halfPositions, offset: 0, index: 0)
            encoder.setBuffer(self.sortedHalfPositions.buffer, offset: 0, index: 1)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 2)
            encoder.setValue(UInt32(positions.descriptor.count), at: 3)
            encoder.dispatch1d(state: self.reorderHalfPrecisionState, exactlyOrCovering: positions.descriptor.count)

            let threadgroupWidth = 256
            encoder.setBuffer(self.cellStart, offset: 0, index: 0)
            encoder.setBuffer(self.cellEnd, offset: 0, index: 1)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 2)
            encoder.setValue(UInt32(positions.descriptor.count), at: 3)
            encoder.setThreadgroupMemoryLength((threadgroupWidth + 16) * MemoryLayout<UInt32>.size, index: 0)
            encoder.dispatch1d(state: self.computeCellBoundariesState, exactlyOrCovering: positions.descriptor.count, threadgroupWidth: threadgroupWidth)
        }
        commandBuffer.popDebugGroup()
    }
    
    /// Finds collision candidates for the given positions.
    ///
    /// - Parameters:
    ///   - collidablePositions: Optional buffer containing collidable positions. If nil, uses the positions from the build step.
    ///   - collisionCandidates: Buffer to store the found collision candidates.
    ///   - connectedVertices: Optional buffer containing connected vertices to exclude from collision checks.
    ///   - commandBuffer: The command buffer to encode the operation into.
    public func find(
        collidablePositions: MTLTypedBuffer?,
        collisionCandidates: MTLTypedBuffer,
        connectedVertices: MTLTypedBuffer?,
        in commandBuffer: MTLCommandBuffer
    ) {
        let collidablePositionsCount = collidablePositions?.descriptor.count ?? 0
        let collidablePositions = collidablePositions ?? self.sortedHalfPositions
        let maxCandidatesCount = collisionCandidates.descriptor.count / collidablePositions.descriptor.count
        let positionsPacked = collidablePositions.descriptor.valueType.isPacked

        commandBuffer.pushDebugGroup("Find Collision Candidates")
        commandBuffer.compute { encoder in
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 1)
            encoder.setBuffer(self.cellStart, offset: 0, index: 2)
            encoder.setBuffer(self.cellEnd, offset: 0, index: 3)
            encoder.setBuffer(self.sortedHalfPositions.buffer, offset: 0, index: 4)
            encoder.setBuffer(collidablePositions.buffer, offset: 0, index: 5)
            if let connectedVertices {
                encoder.setBuffer(connectedVertices.buffer, offset: 0, index: 6)
            } else {
                encoder.setValue([UInt32.zero], at: 6)
            }
            encoder.setValue(UInt32(self.hashTableCapacity), at: 7)
            encoder.setValue(self.configuration.radius, at: 8)
            encoder.setValue(self.configuration.cellSize, at: 9)
            encoder.setValue(UInt32(maxCandidatesCount), at: 10)
            encoder.setValue(UInt32((connectedVertices?.descriptor.count ?? 0) / collidablePositions.descriptor.count), at: 11)
            encoder.setValue(UInt32(collidablePositionsCount), at: 12)
            encoder.setValue(UInt32(collidablePositions.descriptor.count), at: 13)
            encoder.setValue(positionsPacked, at: 14)
            encoder.dispatch1d(state: self.findCollisionCandidatesState, exactlyOrCovering: collidablePositions.descriptor.count)
        }
        commandBuffer.popDebugGroup()
    }
}

public extension SpatialHashing {
    static func totalBuffersSize(maxPositionsCount: Int) -> Int {
        let cellStartSize = maxPositionsCount * MemoryLayout<UInt32>.stride * 2
        let cellEndSize = maxPositionsCount * MemoryLayout<UInt32>.stride * 2
        let hashTableSize = maxPositionsCount * MemoryLayout<SIMD2<UInt32>>.stride * 2
        let halfPositionsSize = maxPositionsCount * MemoryLayout<SIMD3<Float16>>.stride * 2
        
        return cellStartSize + cellEndSize + hashTableSize + halfPositionsSize
    }
}

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
    ///   - maxElementsCount: The maximum number of elements that can be handled.
    /// - Throws: An error if initialization fails.
    public convenience init(
        heap: MTLHeap,
        configuration: Configuration,
        maxElementsCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .heap(heap)),
            configuration: configuration,
            maxElementsCount: maxElementsCount
        )
    }
    
    /// Initializes a new instance of SpatialHashing using the specified Metal heap.
    ///
    /// - Parameters:
    ///   - heap: The Metal heap to allocate resources from.
    ///   - configuration: The configuration for spatial hashing.
    ///   - maxElementsCount: The maximum number of elements that can be handled.
    /// - Throws: An error if initialization fails.
    public convenience init(
        device: MTLDevice,
        configuration: Configuration,
        maxElementsCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .device(device)),
            configuration: configuration,
            maxElementsCount: maxElementsCount
        )
    }

    private init(
        bufferAllocator: MTLBufferAllocator,
        configuration: Configuration,
        maxElementsCount: Int
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
        
        self.hashTableCapacity = maxElementsCount * 2
        self.hashTable = try BitonicSort.buffer(count: maxElementsCount, bufferAllocator: bufferAllocator)
        self.cellStart = try bufferAllocator.buffer(for: UInt32.self, count: self.hashTableCapacity)
        self.cellEnd = try bufferAllocator.buffer(for: UInt32.self, count: self.hashTableCapacity)
        self.halfPositions = try bufferAllocator.buffer(for: SIMD3<Float16>.self, count: maxElementsCount)
        self.sortedHalfPositions = try bufferAllocator.typedBuffer(descriptor: .init(valueType: .half3, count: maxElementsCount))
    }
    
    /// Builds the spatial hash structure for the given elements.
    ///
    /// - Parameters:
    ///   - elements: A buffer containing the positions of elements.
    ///   - commandBuffer: The command buffer to encode the operation into.
    public func build(
        elements: MTLTypedBuffer,
        in commandBuffer: MTLCommandBuffer
    ) {
        let positionsPacked = elements.descriptor.valueType.isPacked
        
        commandBuffer.blit { encoder in
            encoder.fill(buffer: self.hashTable.buffer, range: 0..<self.hashTable.buffer.length, value: .max)
        }
    
        commandBuffer.pushDebugGroup("Compute Hash")
        commandBuffer.compute { encoder in
            encoder.setBuffer(elements.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.halfPositions, offset: 0, index: 1)
            encoder.setValue(UInt32(elements.descriptor.count), at: 2)
            encoder.setValue(positionsPacked, at: 3)
            encoder.dispatch1d(state: self.convertToHalfState, exactlyOrCovering: elements.descriptor.count)
            
            encoder.setBuffer(self.halfPositions, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 1)
            encoder.setValue(UInt32(self.hashTableCapacity), at: 2)
            encoder.setValue(self.configuration.cellSize, at: 3)
            encoder.setValue(UInt32(elements.descriptor.count), at: 4)
            encoder.dispatch1d(state: self.computeHashAndIndexState, exactlyOrCovering: elements.descriptor.count)
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
            encoder.setValue(UInt32(elements.descriptor.count), at: 3)
            encoder.dispatch1d(state: self.reorderHalfPrecisionState, exactlyOrCovering: elements.descriptor.count)

            let threadgroupWidth = 256
            encoder.setBuffer(self.cellStart, offset: 0, index: 0)
            encoder.setBuffer(self.cellEnd, offset: 0, index: 1)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 2)
            encoder.setValue(UInt32(elements.descriptor.count), at: 3)
            encoder.setThreadgroupMemoryLength((threadgroupWidth + 16) * MemoryLayout<UInt32>.size, index: 0)
            encoder.dispatch1d(state: self.computeCellBoundariesState, exactlyOrCovering: elements.descriptor.count, threadgroupWidth: threadgroupWidth)
        }
        commandBuffer.popDebugGroup()
    }
    
    /// Finds collision candidates for the given elements.
    ///
    /// - Parameters:
    ///   - externalElements: Optional buffer containing external elements to check for collisions. If nil, uses the elements from the build step.
    ///   - collisionCandidates: Buffer to store the found collision candidates.
    ///   - connectedVertices: Optional buffer containing connected vertices to exclude from collision checks.
    ///   - commandBuffer: The command buffer to encode the operation into.
    public func find(
        externalElements: MTLTypedBuffer?,
        collisionCandidates: MTLTypedBuffer,
        connectedVertices: MTLTypedBuffer?,
        in commandBuffer: MTLCommandBuffer
    ) {
        let elements = externalElements ?? self.sortedHalfPositions
        let maxCandidatesCount = collisionCandidates.descriptor.count / elements.descriptor.count
        let positionsPacked = elements.descriptor.valueType.isPacked

        commandBuffer.pushDebugGroup("Find Collision Candidates")
        commandBuffer.compute { encoder in
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 1)
            encoder.setBuffer(self.cellStart, offset: 0, index: 2)
            encoder.setBuffer(self.cellEnd, offset: 0, index: 3)
            encoder.setBuffer(self.sortedHalfPositions.buffer, offset: 0, index: 4)
            encoder.setBuffer(elements.buffer, offset: 0, index: 5)
            if let connectedVertices {
                encoder.setBuffer(connectedVertices.buffer, offset: 0, index: 6)
            } else {
                encoder.setValue([UInt32.zero], at: 6)
            }
            encoder.setValue(UInt32(self.hashTableCapacity), at: 7)
            encoder.setValue(self.configuration.radius, at: 8)
            encoder.setValue(self.configuration.cellSize, at: 9)
            encoder.setValue(UInt32(maxCandidatesCount), at: 10)
            encoder.setValue(UInt32((connectedVertices?.descriptor.count ?? 0) / elements.descriptor.count), at: 11)
            encoder.setValue(UInt32(externalElements?.descriptor.count ?? 0), at: 12)
            encoder.setValue(UInt32(elements.descriptor.count), at: 13)
            encoder.setValue(positionsPacked, at: 14)
            encoder.dispatch1d(state: self.findCollisionCandidatesState, exactlyOrCovering: elements.descriptor.count)
        }
        commandBuffer.popDebugGroup()
    }
}

public extension SpatialHashing {
    static func totalBuffersSize(maxElementsCount: Int) -> Int {
        let cellStartSize = maxElementsCount * MemoryLayout<UInt32>.stride * 2
        let cellEndSize = maxElementsCount * MemoryLayout<UInt32>.stride * 2
        let hashTableSize = maxElementsCount * MemoryLayout<SIMD2<UInt32>>.stride * 2
        let halfPositionsSize = maxElementsCount * MemoryLayout<SIMD3<Float16>>.stride * 2
        
        return cellStartSize + cellEndSize + hashTableSize + halfPositionsSize
    }
}

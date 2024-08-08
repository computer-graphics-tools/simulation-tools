import Metal
import SimulationToolsSharedTypes

public final class TriangleSpatialHashing {
    /// Configuration for the TriangleSpatialHashing algorithm.
    public struct Configuration {
        /// The size of each cell in the spatial grid.
        let cellSize: Float
        /// The size of each bucket in the hash table.
        let bucketSize: Int
        
        /// Initializes a new Configuration instance.
        /// - Parameters:
        ///   - cellSize: The size of each cell in the spatial grid.
        ///   - bucketSize: The size of each bucket in the hash table.
        public init(cellSize: Float, bucketSize: Int = 8) {
            self.cellSize = cellSize
            self.bucketSize = bucketSize
        }
    }

    private let hashTrianglesState: MTLComputePipelineState
    private let findTriangleCandidatesState: MTLComputePipelineState
    private let reuseTrianglesCacheState: MTLComputePipelineState

    private let configuration: Configuration
    private let hashTable: MTLBuffer
    private let hashTableCounter: MTLBuffer
    private var counter = 0
    
    /// Initializes a new instance of TriangleSpatialHashing using the specified Metal device.
    ///
    /// - Parameters:
    ///   - device: The Metal device to use for computations.
    ///   - configuration: The configuration for triangle spatial hashing.
    ///   - maxTrianglesCount: The maximum number of triangles that can be handled.
    /// - Throws: An error if initialization fails.
    public convenience init(
        heap: MTLHeap,
        configuration: Configuration,
        maxTrianglesCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .heap(heap)),
            configuration: configuration,
            maxTrianglesCount: maxTrianglesCount
        )
    }

    /// Initializes a new instance of TriangleSpatialHashing using the specified Metal heap.
    ///
    /// - Parameters:
    ///   - heap: The Metal heap to allocate resources from.
    ///   - configuration: The configuration for triangle spatial hashing.
    ///   - maxTrianglesCount: The maximum number of triangles that can be handled.
    /// - Throws: An error if initialization fails.
    public convenience init(
        device: MTLDevice,
        configuration: Configuration,
        maxTrianglesCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .device(device)),
            configuration: configuration,
            maxTrianglesCount: maxTrianglesCount
        )
    }
    
    private init(
        bufferAllocator: MTLBufferAllocator,
        configuration: Configuration,
        maxTrianglesCount: Int
    ) throws {
        let library = try bufferAllocator.device.makeDefaultLibrary(bundle: .module)

        self.hashTrianglesState = try library.computePipelineState(function: "hashTriangles")
        self.findTriangleCandidatesState = try library.computePipelineState(function: "findTriangleCandidates")
        self.reuseTrianglesCacheState = try library.computePipelineState(function: "reuseTrianglesCache")
        
        self.configuration = configuration
        self.hashTable = try bufferAllocator.buffer(for: UInt32.self, count: maxTrianglesCount * configuration.bucketSize, options: .storageModePrivate)
        self.hashTableCounter = try bufferAllocator.buffer(for: UInt32.self, count: maxTrianglesCount, options: .storageModePrivate)
    }
    
    /// Builds the spatial hash structure for the given triangle mesh.
    ///
    /// - Parameters:
    ///   - colliderPositions: A buffer containing the vertex positions of the mesh.
    ///   - indices: A buffer containing the triangle indices of the mesh.
    ///   - commandBuffer: The command buffer to encode the operation into.
    public func build(
        colliderPositions: MTLTypedBuffer,
        indices: MTLTypedBuffer,
        in commandBuffer: MTLCommandBuffer
    ) {
        let colliderPositionsPacked = colliderPositions.descriptor.valueType.isPacked
        let trianglesPacked = indices.descriptor.valueType.isPacked
        
        commandBuffer.blit { encoder in
            encoder.fill(buffer: self.hashTableCounter, range: 0..<hashTableCounter.length, value: 0)
            encoder.fill(buffer: self.hashTable, range: 0..<self.hashTable.length, value: .max)
        }
        
        commandBuffer.pushDebugGroup("Hash Triangles")
        commandBuffer.compute { encoder in
            encoder.setBuffer(colliderPositions.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable, offset: 0, index: 1)
            encoder.setBuffer(self.hashTableCounter, offset: 0, index: 2)
            encoder.setValue(self.configuration.cellSize, at: 3)
            encoder.setBuffer(indices.buffer, offset: 0, index: 4)
            encoder.setValue(UInt32(indices.descriptor.count), at: 5)
            encoder.setValue(UInt32(configuration.bucketSize), at: 6)
            encoder.setValue(UInt32(counter), at: 7)
            encoder.setValue(colliderPositionsPacked, at: 8)
            encoder.setValue(trianglesPacked, at: 9)
            
            encoder.dispatch1d(state: self.hashTrianglesState, exactlyOrCovering: indices.descriptor.count)
        }
        commandBuffer.popDebugGroup()
        
        counter += 1
    }

    /// Finds collision candidates between the given colliderPositions and the triangle mesh.
    ///
    /// - Parameters:
    ///   - collidablePositions: Optional buffer containing external colliderPositions to check for collisions. If nil, uses the mesh vertices.
    ///   - colliderPositions: A buffer containing the vertex positions of the mesh.
    ///   - indices: A buffer containing the triangle indices of the mesh.
    ///   - collisionCandidates: Buffer to store the found collision candidates.
    ///   - commandBuffer: The command buffer to encode the operation into.
    public func find(
        collidablePositions: MTLTypedBuffer?,
        colliderPositions: MTLTypedBuffer,
        indices: MTLTypedBuffer,
        collisionCandidates: MTLTypedBuffer,
        in commandBuffer: MTLCommandBuffer
    ) {
        let useExternalCollidable = collidablePositions != nil
        let positions = collidablePositions ?? colliderPositions
        let collidablePositionsPacked = positions.descriptor.valueType.isPacked
        let colliderPositionsPacked = colliderPositions.descriptor.valueType.isPacked
        let trianglesPacked = indices.descriptor.valueType.isPacked

        let params = TriangleSHParameters(
            hashTableCapacity: UInt32(indices.descriptor.count),
            cellSize: self.configuration.cellSize,
            maxCollisionCandidatesCount: UInt32(collisionCandidates.descriptor.count / positions.descriptor.count),
            connectedVerticesCount: 0,
            bucketSize: UInt32(self.configuration.bucketSize),
            gridSize: UInt32(positions.descriptor.count),
            useExternalCollidable: useExternalCollidable,
            usePackedCollidablePositions: collidablePositionsPacked,
            usePackedColliderPositions: colliderPositionsPacked,
            usePackedIndices: trianglesPacked
        )

        commandBuffer.pushDebugGroup("Find Collision Candidates")
        commandBuffer.compute { encoder in
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(positions.buffer, offset: 0, index: 1)
            encoder.setBuffer(colliderPositions.buffer, offset: 0, index: 2)
            encoder.setBuffer(indices.buffer, offset: 0, index: 3)
            encoder.setBuffer(self.hashTable, offset: 0, index: 4)
            encoder.setValue([UInt32.max], at: 5)
            encoder.setValue(params, at: 6)
            encoder.dispatch1d(state: self.findTriangleCandidatesState, exactlyOrCovering: positions.descriptor.count)
        }
        commandBuffer.popDebugGroup()
    }

    
    /// Reuses previously computed collision information to optimize subsequent collision queries.
    ///
    /// - Parameters:
    ///   - collidablePositions: Optional buffer containing external colliderPositions to check for collisions. If nil, uses the mesh vertices.
    ///   - colliderPositions: A buffer containing the vertex positions of the mesh.
    ///   - indices: A buffer containing the triangle indices of the mesh.
    ///   - collisionCandidates: Buffer to store the found collision candidates.
    ///   - vertexNeighbors: Buffer containing vertex neighbor information.
    ///   - trinagleNeighbors: Optional buffer containing triangle neighbor information.
    ///   - commandBuffer: The command buffer to encode the operation into.
    public func reuse(
        collidablePositions: MTLTypedBuffer?,
        colliderPositions: MTLTypedBuffer,
        indices: MTLTypedBuffer,
        collisionCandidates: MTLTypedBuffer,
        vertexNeighbors: MTLTypedBuffer,
        trinagleNeighbors: MTLTypedBuffer?,
        in commandBuffer: MTLCommandBuffer
    ) {
        let useExternalCollidable = collidablePositions != nil
        let colldablePositions = collidablePositions ?? colliderPositions
        let collidablePositionsPacked = colldablePositions.descriptor.valueType.isPacked
        let colliderPositionsPacked = colliderPositions.descriptor.valueType.isPacked
        let trianglesPacked = indices.descriptor.valueType.isPacked

        let params = TriangleSHParameters(
            hashTableCapacity: UInt32(indices.descriptor.count),
            cellSize: self.configuration.cellSize,
            maxCollisionCandidatesCount: UInt32(collisionCandidates.descriptor.count / colldablePositions.descriptor.count),
            connectedVerticesCount: 0,
            bucketSize: UInt32(self.configuration.bucketSize),
            gridSize: UInt32(colldablePositions.descriptor.count),
            useExternalCollidable: useExternalCollidable,
            usePackedCollidablePositions: collidablePositionsPacked,
            usePackedColliderPositions: colliderPositionsPacked,
            usePackedIndices: trianglesPacked
        )

        commandBuffer.pushDebugGroup("Reuse Triangles")
        commandBuffer.compute { encoder in
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(colldablePositions.buffer, offset: 0, index: 1)
            encoder.setBuffer(colliderPositions.buffer, offset: 0, index: 2)
            encoder.setBuffer(indices.buffer, offset: 0, index: 3)
            encoder.setBuffer(vertexNeighbors.buffer, offset: 0, index: 4)
            encoder.setValue([UInt32.max], at: 5)
            if let trinagleNeighbors {
                encoder.setBuffer(trinagleNeighbors.buffer, offset: 0, index: 6)
                encoder.setValue(true, at: 9)
            } else {
                encoder.setValue(SIMD3(repeating: UInt32.max), at: 6)
                encoder.setValue(false, at: 9)
            }
            
            encoder.setValue(UInt32(vertexNeighbors.descriptor.count / colldablePositions.descriptor.count), at: 7)
            encoder.setValue(params, at: 8)
            encoder.dispatch1d(state: reuseTrianglesCacheState, exactlyOrCovering: colldablePositions.descriptor.count)
        }
        commandBuffer.popDebugGroup()
    }
}

public extension TriangleSpatialHashing {
    static func totalBuffersSize(maxTrianglesCount: Int, configuration: Configuration) -> Int {
        let triangleHashTableSize = maxTrianglesCount * MemoryLayout<UInt32>.stride * configuration.bucketSize
        let triangleHashTableCounterSize = maxTrianglesCount * MemoryLayout<UInt32>.stride
        
        return triangleHashTableSize + triangleHashTableCounterSize
    }
}

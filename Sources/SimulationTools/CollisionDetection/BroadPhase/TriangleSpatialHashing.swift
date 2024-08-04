import Metal
import SimulationToolsSharedTypes

public final class TriangleSpatialHashing {
    public struct Configuration {
        let cellSize: Float
        let bucketSize: Int
        
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
    
    public convenience init(
        heap: MTLHeap,
        configuration: Configuration,
        trianglesCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .heap(heap)),
            configuration: configuration,
            trianglesCount: trianglesCount
        )
    }

    public convenience init(
        device: MTLDevice,
        configuration: Configuration,
        trianglesCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .device(device)),
            configuration: configuration,
            trianglesCount: trianglesCount
        )
    }
    
    private init(
        bufferAllocator: MTLBufferAllocator,
        configuration: Configuration,
        trianglesCount: Int
    ) throws {
        let library = try bufferAllocator.device.makeDefaultLibrary(bundle: .module)

        self.hashTrianglesState = try library.computePipelineState(function: "hashTriangles")
        self.findTriangleCandidatesState = try library.computePipelineState(function: "findTriangleCandidates")
        self.reuseTrianglesCacheState = try library.computePipelineState(function: "reuseTrianglesCache")
        
        self.configuration = configuration
        self.hashTable = try bufferAllocator.buffer(for: UInt32.self, count: trianglesCount * configuration.bucketSize, options: .storageModePrivate)
        self.hashTableCounter = try bufferAllocator.buffer(for: UInt32.self, count: trianglesCount, options: .storageModePrivate)
    }
    
    public func build(
        elements: MTLTypedBuffer,
        indices: MTLTypedBuffer,
        in commandBuffer: MTLCommandBuffer
    ) {
        let colliderPositionsPacked = elements.descriptor.valueType.isPacked
        let trianglesPacked = indices.descriptor.valueType.isPacked
        
        commandBuffer.blit { encoder in
            encoder.fill(buffer: self.hashTableCounter, range: 0..<hashTableCounter.length, value: 0)
            encoder.fill(buffer: self.hashTable, range: 0..<self.hashTable.length, value: .max)
        }
        
        commandBuffer.pushDebugGroup("Hash Triangles")
        commandBuffer.compute { encoder in
            encoder.setBuffer(elements.buffer, offset: 0, index: 0)
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

    public func find(
        extrnalElements: MTLTypedBuffer?,
        elements: MTLTypedBuffer,
        indices: MTLTypedBuffer,
        collisionCandidates: MTLTypedBuffer,
        in commandBuffer: MTLCommandBuffer
    ) {
        let useExternalCollidable = extrnalElements != nil
        let positions = extrnalElements ?? elements
        let collidablePositionsPacked = positions.descriptor.valueType.isPacked
        let colliderPositionsPacked = elements.descriptor.valueType.isPacked
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
            encoder.setBuffer(elements.buffer, offset: 0, index: 2)
            encoder.setBuffer(indices.buffer, offset: 0, index: 3)
            encoder.setBuffer(self.hashTable, offset: 0, index: 4)
            encoder.setValue([UInt32.max], at: 5)
            encoder.setValue(params, at: 6)
            encoder.dispatch1d(state: self.findTriangleCandidatesState, exactlyOrCovering: positions.descriptor.count)
        }
        commandBuffer.popDebugGroup()
    }

    public func reuse(
        extrnalElements: MTLTypedBuffer?,
        elements: MTLTypedBuffer,
        indices: MTLTypedBuffer,
        collisionCandidates: MTLTypedBuffer,
        vertexNeighbors: MTLTypedBuffer,
        trinagleNeighbors: MTLTypedBuffer?,
        in commandBuffer: MTLCommandBuffer
    ) {
        let useExternalCollidable = extrnalElements != nil
        let positions = extrnalElements ?? elements
        let collidablePositionsPacked = positions.descriptor.valueType.isPacked
        let colliderPositionsPacked = elements.descriptor.valueType.isPacked
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

        commandBuffer.pushDebugGroup("Reuse Triangles")
        commandBuffer.compute { encoder in
            encoder.setBuffer(vertexNeighbors.buffer, offset: 0, index: 0)
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 1)
            encoder.setBuffer(positions.buffer, offset: 0, index: 2)
            encoder.setBuffer(elements.buffer, offset: 0, index: 3)
            encoder.setBuffer(indices.buffer, offset: 0, index: 4)
            encoder.setValue([UInt32.max], at: 5)
            if let trinagleNeighbors {
                encoder.setBuffer(trinagleNeighbors.buffer, offset: 0, index: 6)
                encoder.setValue(true, at: 9)
            } else {
                encoder.setValue(SIMD3(repeating: UInt32.max), at: 6)
                encoder.setValue(false, at: 9)
            }
            
            encoder.setValue(UInt32(vertexNeighbors.descriptor.count / positions.descriptor.count), at: 7)
            encoder.setValue(params, at: 8)
            encoder.dispatch1d(state: reuseTrianglesCacheState, exactlyOrCovering: positions.descriptor.count)
        }
        commandBuffer.popDebugGroup()
    }
}

public extension TriangleSpatialHashing {
    static func totalBuffersSize(triangleCount: Int, configuration: Configuration) -> Int {
        let triangleHashTableSize = triangleCount * MemoryLayout<UInt32>.stride * configuration.bucketSize
        let triangleHashTableCounterSize = triangleCount * MemoryLayout<UInt32>.stride
        
        return triangleHashTableSize + triangleHashTableCounterSize
    }
}

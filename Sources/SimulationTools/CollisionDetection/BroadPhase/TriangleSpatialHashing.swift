import Metal

public final class TriangleSpatialHashing {
    public struct Configuration {
        let cellSize: Float
        
        public init(cellSize: Float32) {
            self.cellSize = cellSize
        }
    }

    private let hashTrianglesState: MTLComputePipelineState
    private let findTriangleCandidatesState: MTLComputePipelineState
    private let reuseTrianglesCacheState: MTLComputePipelineState

    private let configuration: Configuration
    private let triangleHashTable: MTLBuffer
    private let triangleHashTableCounter: MTLBuffer
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

    init(
        bufferAllocator: MTLBufferAllocator,
        configuration: Configuration,
        trianglesCount: Int
    ) throws {
        let library = try bufferAllocator.device.makeDefaultLibrary(bundle: .module)

        self.hashTrianglesState = try library.computePipelineState(function: "hashTriangles")
        self.findTriangleCandidatesState = try library.computePipelineState(function: "findTriangleCandidates")
        self.reuseTrianglesCacheState = try library.computePipelineState(function: "reuseTrianglesCache")
        
        self.configuration = configuration
        self.triangleHashTable = try bufferAllocator.buffer(for: UInt32.self, count: trianglesCount * 8, options: .storageModePrivate)
        self.triangleHashTableCounter = try bufferAllocator.buffer(for: UInt32.self, count: trianglesCount, options: .storageModePrivate)
    }
    
    public func build<PositionType, TriangleType>(
        positions: MTLTypedBuffer<SIMD4<Float>>,
        scenePositions: MTLTypedBuffer<PositionType>,
        sceneTriangles: MTLTypedBuffer<TriangleType>,
        collisionCandidates: MTLTypedBuffer<UInt32>,
        positionsCount: Int,
        trianglesCount: Int,
        in commandBuffer: MTLCommandBuffer
    ) {
        commandBuffer.blit { encoder in
            encoder.fill(buffer: triangleHashTableCounter, range: 0..<(MemoryLayout<UInt32>.stride * trianglesCount), value: 0)
        }

        commandBuffer.pushDebugGroup("Hash Triangles")
        commandBuffer.compute { encoder in
            encoder.setBuffer(scenePositions.buffer, offset: 0, index: 0)
            encoder.setBuffer(triangleHashTable, offset: 0, index: 1)
            encoder.setBuffer(triangleHashTableCounter, offset: 0, index: 2)
            encoder.setValue(configuration.cellSize, at: 3)
            encoder.setBuffer(sceneTriangles.buffer, offset: 0, index: 4)
            encoder.setValue(UInt32(trianglesCount), at: 5)
            encoder.setValue(UInt32(counter), at: 6)
            encoder.dispatch1d(state: hashTrianglesState, exactlyOrCovering: trianglesCount)
        }
        commandBuffer.popDebugGroup()
        
        
        commandBuffer.pushDebugGroup("Find Collision Candidates")
        commandBuffer.compute { encoder in
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(positions.buffer, offset: 0, index: 1)
            encoder.setBuffer(scenePositions.buffer, offset: 0, index: 2)
            encoder.setBuffer(triangleHashTable, offset: 0, index: 3)
            encoder.setBuffer(sceneTriangles.buffer, offset: 0, index: 4)
            encoder.setValue(UInt32(trianglesCount), at: 5)
            encoder.setValue(configuration.cellSize, at: 6)
            encoder.setValue(UInt32(collisionCandidates.count / positions.count), at: 7)
            encoder.setValue(UInt32(positionsCount), at: 8)
            encoder.dispatch1d(state: findTriangleCandidatesState, exactlyOrCovering: positionsCount)
        }
        commandBuffer.popDebugGroup()
        
        counter += 1
    }
    
    public func reuse<PositionType, TriangleType>(
        positions: MTLTypedBuffer<SIMD4<Float>>,
        scenePositions: MTLTypedBuffer<PositionType>,
        sceneTriangles: MTLTypedBuffer<TriangleType>,
        collisionCandidates: MTLTypedBuffer<UInt32>,
        vertexNeighbors: MTLTypedBuffer<UInt32>,
        positionsCount: Int,
        in commandBuffer: MTLCommandBuffer
    ) {
        commandBuffer.pushDebugGroup("Reuse Triangles")
        commandBuffer.compute { encoder in
            encoder.setBuffer(vertexNeighbors.buffer, offset: 0, index: 0)
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 1)
            encoder.setBuffer(positions.buffer, offset: 0, index: 2)
            encoder.setBuffer(scenePositions.buffer, offset: 0, index: 3)
            encoder.setBuffer(sceneTriangles.buffer, offset: 0, index: 4)
            encoder.setValue(UInt32(positionsCount), at: 5)
            encoder.setValue(UInt32(collisionCandidates.count / positions.count), at: 6)
            encoder.setValue(UInt32(vertexNeighbors.count / positions.count), at: 7)

            encoder.dispatch1d(state: reuseTrianglesCacheState, exactlyOrCovering: positionsCount)
        }
        commandBuffer.popDebugGroup()
    }
}

public extension TriangleSpatialHashing {
    static func totalBuffersSize(triangleCount: Int) -> Int {
        let triangleHashTableSize = triangleCount * MemoryLayout<UInt32>.stride * 8
        let triangleHashTableCounterSize = triangleCount * MemoryLayout<UInt32>.stride
        
        return triangleHashTableSize + triangleHashTableCounterSize
    }
}

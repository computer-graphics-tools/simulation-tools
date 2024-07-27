import Metal

public final class TriangleSpatialHashing {
    public struct Configuration {
        let cellSize: Float
        
        public init(cellSize: Float) {
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
    
    public func build(
        positions: MTLTypedBuffer,
        scenePositions: MTLTypedBuffer,
        sceneTriangles: MTLTypedBuffer,
        collisionCandidates: MTLTypedBuffer,
        rehash: Bool,
        in commandBuffer: MTLCommandBuffer
    ) {
        if rehash {
            commandBuffer.blit { encoder in
                encoder.fill(buffer: triangleHashTableCounter, range: 0..<(MemoryLayout<UInt32>.stride * sceneTriangles.descriptor.count), value: 0)
            }
            
            commandBuffer.pushDebugGroup("Hash Triangles")
            commandBuffer.compute { encoder in
                encoder.setBuffer(scenePositions.buffer, offset: 0, index: 0)
                encoder.setBuffer(triangleHashTable, offset: 0, index: 1)
                encoder.setBuffer(triangleHashTableCounter, offset: 0, index: 2)
                encoder.setValue(configuration.cellSize, at: 3)
                encoder.setBuffer(sceneTriangles.buffer, offset: 0, index: 4)
                encoder.setValue(UInt32(sceneTriangles.descriptor.count), at: 5)
                encoder.setValue(UInt32(counter), at: 6)
                encoder.dispatch1d(state: hashTrianglesState, exactlyOrCovering: sceneTriangles.descriptor.count)
            }
            commandBuffer.popDebugGroup()
        }

        commandBuffer.pushDebugGroup("Find Collision Candidates")
        commandBuffer.compute { encoder in
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(positions.buffer, offset: 0, index: 1)
            encoder.setBuffer(scenePositions.buffer, offset: 0, index: 2)
            encoder.setBuffer(triangleHashTable, offset: 0, index: 3)
            encoder.setBuffer(sceneTriangles.buffer, offset: 0, index: 4)
            encoder.setValue(UInt32(sceneTriangles.descriptor.count), at: 5)
            encoder.setValue(configuration.cellSize, at: 6)
            encoder.setValue(UInt32(collisionCandidates.descriptor.count / positions.descriptor.count), at: 7)
            encoder.setValue(UInt32(positions.descriptor.count), at: 8)
            encoder.dispatch1d(state: findTriangleCandidatesState, exactlyOrCovering: positions.descriptor.count)
        }
        commandBuffer.popDebugGroup()
        
        counter += 1
    }
    
    public func reuse(
        positions: MTLTypedBuffer,
        scenePositions: MTLTypedBuffer,
        sceneTriangles: MTLTypedBuffer,
        collisionCandidates: MTLTypedBuffer,
        vertexNeighbors: MTLTypedBuffer,
        trinagleNeighbors: MTLTypedBuffer? = nil,
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
            if let trinagleNeighbors {
                encoder.setBuffer(trinagleNeighbors.buffer, offset: 0, index: 5)
                encoder.setValue(true, at: 10)
            } else {
                encoder.setValue(SIMD3(repeating: UInt32.max), at: 5)
                encoder.setValue(false, at: 10)
            }
            encoder.setValue(UInt32(positionsCount), at: 6)
            encoder.setValue(UInt32(collisionCandidates.descriptor.count / positions.descriptor.count), at: 7)
            encoder.setValue(UInt32(vertexNeighbors.descriptor.count / positions.descriptor.count), at: 8)
            encoder.setValue(UInt32(sceneTriangles.descriptor.count), at: 9)

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

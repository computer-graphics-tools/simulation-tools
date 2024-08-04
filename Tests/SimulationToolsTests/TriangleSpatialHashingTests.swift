import XCTest
import Metal
@testable import SimulationTools

final class TriangleSpatialHashingTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
    override func setUp() {
        super.setUp()
        self.device = MTLCreateSystemDefaultDevice()
        self.commandQueue = self.device.makeCommandQueue()
    }
    
    override func tearDown() {
        self.device = nil
        self.commandQueue = nil
        super.tearDown()
    }
    
    func testTriangleSpatialHashingInitialization() throws {
        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0, bucketSize: 8)
        
        XCTAssertNoThrow(
            try TriangleSpatialHashing(
                device: self.device,
                configuration: config,
                trianglesCount: 100
            )
        )
    }
    
    func testBuildAndFindExternalCollision() throws {
        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0, bucketSize: 8)
        let spatialHashing = try TriangleSpatialHashing(device: self.device, configuration: config, trianglesCount: 2)
        
        let colliderPositions = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0),  // Triangle 1
            SIMD3<Float>(2, 0, 0), SIMD3<Float>(3, 0, 0), SIMD3<Float>(2, 1, 0)   // Triangle 2
        ]
        let colliderTriangles = [SIMD3<UInt32>(0, 1, 2), SIMD3<UInt32>(3, 4, 5)]
        let positions = [SIMD3<Float>(0.5, 0.5, 0), SIMD3<Float>(2.5, 0.5, 0)]
        
        let positionsBuffer = try createTypedBuffer(from: positions, type: .float3)
        let colliderPositionsBuffer = try createTypedBuffer(from: colliderPositions, type: .float3)
        let colliderTrianglesBuffer = try createTypedBuffer(from: colliderTriangles, type: .uint3)
        
        let collisionCandidates = try findTriangleCollisionCandidates(
            spatialHashing: spatialHashing,
            positions: positionsBuffer,
            colliderPositions: colliderPositionsBuffer,
            colliderTriangles: colliderTrianglesBuffer
        )
        
        let chunks = collisionCandidates.chunked(into: 8)
        XCTAssertEqual(chunks[0][0], 0) // First position should collide with first triangle
        XCTAssertEqual(chunks[1][0], 1) // Second position should collide with second triangle
    }
    
    func testBuildAndFindSelfCollision() throws {
        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0, bucketSize: 8)
        let spatialHashing = try TriangleSpatialHashing(device: self.device, configuration: config, trianglesCount: 3)
        
        let positions = [
            SIMD3<Float>(0, 0, 0), SIMD3<Float>(1, 0, 0), SIMD3<Float>(0, 1, 0),  // Triangle 1
            SIMD3<Float>(0.5, 0.5, 0), SIMD3<Float>(1.5, 0.5, 0), SIMD3<Float>(0.5, 1.5, 0),  // Triangle 2
            SIMD3<Float>(2, 0, 0), SIMD3<Float>(3, 0, 0), SIMD3<Float>(2, 1, 0)   // Triangle 3
        ]
        let triangles = [SIMD3<UInt32>(0, 1, 2), SIMD3<UInt32>(3, 4, 5), SIMD3<UInt32>(6, 7, 8)]
        
        let positionsBuffer = try createTypedBuffer(from: positions, type: .float3)
        let trianglesBuffer = try createTypedBuffer(from: triangles, type: .uint3)
        
        let collisionCandidates = try findTriangleCollisionCandidates(
            spatialHashing: spatialHashing,
            positions: nil,
            colliderPositions: positionsBuffer,
            colliderTriangles: trianglesBuffer
        )
        
        let chunks = collisionCandidates.chunked(into: 8)
        XCTAssertEqual(chunks[3][0], 0) // Vertex of triangle 2 should collide with triangle 1
        XCTAssertEqual(chunks[0][0], 1) // Vertex of triangle 1 should collide with triangle 2
    }
    
    func testPackedFormatExternalCollision() throws {
        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0, bucketSize: 8)
        let spatialHashing = try TriangleSpatialHashing(device: self.device, configuration: config, trianglesCount: 2)
        
        let colliderPositions = [
            packed_float3(0, 0, 0), packed_float3(1, 0, 0), packed_float3(0, 1, 0),  // Triangle 1
            packed_float3(2, 0, 0), packed_float3(3, 0, 0), packed_float3(2, 1, 0)   // Triangle 2
        ]
        let colliderTriangles = [packed_uint3(0, 1, 2), packed_uint3(3, 4, 5)]
        let positions = [SIMD3<Float>(0.5, 0.5, 0), SIMD3<Float>(2.5, 0.5, 0)]
        
        let positionsBuffer = try createTypedBuffer(from: positions, type: .float3)
        let colliderPositionsBuffer = try createTypedBuffer(from: colliderPositions, type: .packedFloat3)
        let colliderTrianglesBuffer = try createTypedBuffer(from: colliderTriangles, type: .packedUInt3)
        
        let collisionCandidates = try findTriangleCollisionCandidates(
            spatialHashing: spatialHashing,
            positions: positionsBuffer,
            colliderPositions: colliderPositionsBuffer,
            colliderTriangles: colliderTrianglesBuffer
        )
        
        let chunks = collisionCandidates.chunked(into: 8)
        XCTAssertEqual(chunks[0][0], 0) // First position should collide with first triangle
        XCTAssertEqual(chunks[1][0], 1) // Second position should collide with second triangle
    }
    
    func testMixedFormatSelfCollision() throws {
        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0, bucketSize: 8)
        let spatialHashing = try TriangleSpatialHashing(device: self.device, configuration: config, trianglesCount: 3)
        
        let positions = [
            packed_float3(0, 0, 0), packed_float3(1, 0, 0), packed_float3(0, 1, 0),  // Triangle 1
            packed_float3(0.5, 0.5, 0), packed_float3(1.5, 0.5, 0), packed_float3(0.5, 1.5, 0),  // Triangle 2
            packed_float3(2, 0, 0), packed_float3(3, 0, 0), packed_float3(2, 1, 0)   // Triangle 3
        ]
        let triangles = [SIMD3<UInt32>(0, 1, 2), SIMD3<UInt32>(3, 4, 5), SIMD3<UInt32>(6, 7, 8)]
        
        let positionsBuffer = try createTypedBuffer(from: positions, type: .packedFloat3)
        let trianglesBuffer = try createTypedBuffer(from: triangles, type: .uint3)
        
        let collisionCandidates = try findTriangleCollisionCandidates(
            spatialHashing: spatialHashing,
            positions: nil,
            colliderPositions: positionsBuffer,
            colliderTriangles: trianglesBuffer
        )
        
        let chunks = collisionCandidates.chunked(into: 8)
        XCTAssertTrue(chunks[3].contains(0)) // Vertex of triangle 2 should collide with triangle 1
        XCTAssertTrue (chunks[0].contains(1)) // Vertex of triangle 1 should collide with triangle 2
        XCTAssertTrue(!chunks[0].contains(0)) // Vertex of triangle 1 should not collide with triangle 1
        XCTAssertTrue(!chunks[3].contains(1)) // Vertex of triangle 2 should not collide with triangle 2
    }
    
    func testPerformanceFor10kTriangles() throws {
        try testPerformanceForTriangles(10_000)
    }
    
    func testPerformanceFor100kTriangles() throws {
        try testPerformanceForTriangles(100_000)
    }
    
    func testPerformanceForTriangles(_ count: Int) throws {
        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0, bucketSize: 8)
        let spatialHashing = try TriangleSpatialHashing(device: self.device, configuration: config, trianglesCount: count)
        
        let (meshPositions, triangles, meshDimensions) = generateUniformMesh(triangleCount: count)
        let randomPositions = generateRandomPositions(count: count, meshDimensions: meshDimensions)
        
        let meshPositionsBuffer = try createTypedBuffer(from: meshPositions, type: .float3)
        let trianglesBuffer = try createTypedBuffer(from: triangles, type: .uint3)
        let randomPositionsBuffer = try createTypedBuffer(from: randomPositions, type: .float3)
        let collisionCandidatesBuffer = try device.typedBuffer(
            with: Array(repeating: UInt32.max, count: count * 8),
            valueType: .uint
        )

        measure {
            guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
                XCTFail("Failed to create command buffer")
                return
            }

            spatialHashing.build(
                elements: meshPositionsBuffer,
                indices: trianglesBuffer,
                in: commandBuffer
            )
            
            spatialHashing.find(
                extrnalElements: randomPositionsBuffer,
                elements: meshPositionsBuffer,
                indices: trianglesBuffer,
                collisionCandidates: collisionCandidatesBuffer,
                in: commandBuffer
            )
            let startTime = CFAbsoluteTimeGetCurrent()
            commandBuffer.addCompletedHandler { _ in
                let endTime = CFAbsoluteTimeGetCurrent()
                let duration = (endTime - startTime) * 1000
                print("Performance test for \(count) triangles and \(count) random positions took \(duration) ms")
            }

            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
    }

    func generateUniformMesh(triangleCount: Int) -> (positions: [SIMD3<Float>], triangles: [SIMD3<UInt32>], dimensions: SIMD2<Float>) {
        let gridSize = Int(ceil(sqrt(Float(triangleCount))))
        let cellSize: Float = 1.0
        var positions: [SIMD3<Float>] = []
        var triangles: [SIMD3<UInt32>] = []
        
        for i in 0..<gridSize {
            for j in 0..<gridSize {
                let baseIndex = UInt32(positions.count)
                let x = Float(i) * cellSize
                let z = Float(j) * cellSize
                
                positions.append(SIMD3<Float>(x, 0, z))
                positions.append(SIMD3<Float>(x + cellSize, 0, z))
                positions.append(SIMD3<Float>(x, 0, z + cellSize))
                positions.append(SIMD3<Float>(x + cellSize, 0, z + cellSize))
                
                triangles.append(SIMD3<UInt32>(baseIndex, baseIndex + 1, baseIndex + 2))
                triangles.append(SIMD3<UInt32>(baseIndex + 1, baseIndex + 3, baseIndex + 2))
                
                if triangles.count >= triangleCount {
                    break
                }
            }
            if triangles.count >= triangleCount {
                break
            }
        }
        
        triangles = Array(triangles.prefix(triangleCount))
        
        let dimensions = SIMD2<Float>(Float(gridSize) * cellSize, Float(gridSize) * cellSize)
        
        return (positions, triangles, dimensions)
    }

    func generateRandomPositions(count: Int, meshDimensions: SIMD2<Float>) -> [SIMD3<Float>] {
        return (0..<count).map { _ in
            SIMD3<Float>(
                Float.random(in: 0...meshDimensions.x),
                Float.random(in: 0...2.0),  // Random height up to 2 units above the plane
                Float.random(in: 0...meshDimensions.y)
            )
        }
    }
    
    private func findTriangleCollisionCandidates(
        spatialHashing: TriangleSpatialHashing,
        positions: MTLTypedBuffer?,
        colliderPositions: MTLTypedBuffer,
        colliderTriangles: MTLTypedBuffer
    ) throws -> [UInt32] {
        let count = positions?.descriptor.count ?? colliderPositions.descriptor.count
        let collisionCandidatesBuffer = try device.typedBuffer(
            with: Array(repeating: UInt32.max, count: count * 8),
            valueType: .uint
        )
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        
        spatialHashing.build(
            elements: colliderPositions,
            indices: colliderTriangles,
            in: commandBuffer
        )
        
        spatialHashing.find(
            extrnalElements: positions,
            elements: colliderPositions,
            indices: colliderTriangles,
            collisionCandidates: collisionCandidatesBuffer,
            in: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return collisionCandidatesBuffer.values()!
    }
    
    private func createTriangles(count: Int) -> [SIMD3<UInt32>] {
        var triangles: [SIMD3<UInt32>] = []
        triangles.reserveCapacity(count)
        
        for i in stride(from: 0, to: count * 3, by: 3) {
            let triangle = SIMD3<UInt32>(UInt32(i), UInt32(i + 1), UInt32(i + 2))
            triangles.append(triangle)
        }
        
        return triangles
    }
    
    private func createTypedBuffer<T>(from array: [T], type: MTLBufferValueType) throws -> MTLTypedBuffer {
        try device.typedBuffer(with: array, valueType: type, options: [])
    }
    
    private func createTypedBuffer(count: Int, type: MTLBufferValueType) throws -> MTLTypedBuffer {
        try device.typedBuffer(descriptor: .init(valueType: type, count: count))
    }
}

private extension Array {
    func chunked(into size: Int) -> [[Element]] {
        return stride(from: 0, to: count, by: size).map {
            Array(self[$0 ..< Swift.min($0 + size, count)])
        }
    }
}

private struct packed_float3 {
    let x: Float
    let y: Float
    let z: Float
    
    init(_ x: Float, _ y: Float, _ z: Float) {
        self.x = x
        self.y = y
        self.z = z
    }
}

private struct packed_uint3 {
    let x: UInt32
    let y: UInt32
    let z: UInt32
    
    init(_ x: UInt32, _ y: UInt32, _ z: UInt32) {
        self.x = x
        self.y = y
        self.z = z
    }
}


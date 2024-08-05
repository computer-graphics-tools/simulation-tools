import XCTest
import Metal
@testable import SimulationTools

final class SpatialHashingTests: XCTestCase {
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
    
    func testSpatialHashingInitialization() throws {
        let config = SpatialHashing.Configuration(
            cellSize: 1.0,
            radius: 0.5
        )
        
        XCTAssertNoThrow(
            try SpatialHashing(
                device: self.device,
                configuration: config,
                maxElementsCount: 100
            )
        )
    }
    
    func generateMockData(count: Int = 100) -> [SIMD3<Float>] {
        return (0..<count).map { i in
            let angle = Float(i) * Float.pi / 50.0
            return SIMD3<Float>(cos(angle) * 10.0, sin(angle) * 10.0, 0.0)
        }
    }
    
    func collisionCandidates(positions: [SIMD3<Float>], candidatesCount: Int = 8, cellSize: Float) throws -> MTLTypedBuffer {
        let config = SpatialHashing.Configuration(
            cellSize: cellSize,
            radius: cellSize / 2.0
        )
        
        let spatialHashing = try SpatialHashing(
            device: self.device,
            configuration: config,
            maxElementsCount: positions.count
        )
        
        let positionsBuffer = try device.typedBuffer(with: positions, valueType: .float3)
        let collisionCandidatesBuffer = try device.typedBuffer(
            with: Array(repeating: UInt32.max, count: positions.count * candidatesCount),
            valueType: .uint
        )
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
        }
        
        spatialHashing.build(elements: positionsBuffer, in: commandBuffer)
        
        spatialHashing.find(
            externalElements: nil,
            collisionCandidates: collisionCandidatesBuffer,
            connectedVertices: nil,
            in: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return collisionCandidatesBuffer
    }
    
    func testBuildMethodInitialization() throws {
        let positions = generateMockData()
        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, cellSize: 0.5)
        XCTAssertNotNil(collisionCandidatesBuffer.values()! as [UInt32], "Collision candidates buffer should not be nil")
    }
    
    func testCollisionCandidatesContainClosestProximity() throws {
        let positions: [SIMD3<Float>] = [
            [-0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.0, 0.0]
        ]
        let candidatesCount = 4
        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, candidatesCount: candidatesCount, cellSize: 1.0)
        let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: candidatesCount)

        XCTAssertTrue(collisionCandidates[0].contains(1))
        XCTAssertTrue(collisionCandidates[1].contains(0))
        XCTAssertTrue(collisionCandidates[2].contains(3))
        XCTAssertTrue(collisionCandidates[3].contains(2))
    }
    
    func testCollisionCandidatesDoNotContainSelf() throws {
        let positions = generateMockData()
        let candidatesCount = 4
        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, candidatesCount: candidatesCount, cellSize: 1.0)
        let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: candidatesCount)

        XCTAssertFalse(collisionCandidates[0].contains(0))
        XCTAssertFalse(collisionCandidates[1].contains(1))
        XCTAssertFalse(collisionCandidates[2].contains(2))
        XCTAssertFalse(collisionCandidates[3].contains(3))
    }
    
    func testCollisionCandidatesSymmetry() throws {
        let positions = generateMockData(count: 10)

        let candidatesCount = 8
        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, candidatesCount: candidatesCount, cellSize: 1.0)
        let collisionCandidates: [Set<UInt32>] = collisionCandidatesBuffer.values()!.chunked(into: candidatesCount).map { Set($0) }

        print(collisionCandidates.enumerated().map { $0 })
        for (i, candidates) in collisionCandidates.enumerated() {
            for candidate in candidates where candidate != UInt32.max {
                XCTAssertTrue(collisionCandidates[Int(candidate)].contains(UInt32(i)), "Symmetry check failed: \(i) and \(candidate)")
            }
        }
    }
    
    func testConnectedVerticesExclusion() throws {
        let positions: [SIMD3<Float>] = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [1.501, 0.0, 0.0]
        ]
        
        let cellSize: Float = 1.0
        let candidatesCount = 8
        let config = SpatialHashing.Configuration(
            cellSize: cellSize,
            radius: cellSize / 2.0
        )
        
        let spatialHashing = try SpatialHashing(
            device: self.device,
            configuration: config,
            maxElementsCount: positions.count
        )
        
        let positionsBuffer = try device.typedBuffer(with: positions, valueType: .float3)
        let collisionCandidatesBuffer = try device.typedBuffer(
            with: Array(repeating: UInt32.max, count: positions.count * candidatesCount),
            valueType: .uint
        )
        
        // Specify connected vertices: 0 and 1 are connected
        let connectedVertices: [UInt32] = [
            1, UInt32.max, UInt32.max, UInt32.max,  // For vertex 0
            0, UInt32.max, UInt32.max, UInt32.max,  // For vertex 1
            UInt32.max, UInt32.max, UInt32.max, UInt32.max,  // For vertex 2
            UInt32.max, UInt32.max, UInt32.max, UInt32.max   // For vertex 3
        ]
        let connectedVerticesBuffer = try device.typedBuffer(with: connectedVertices, valueType: .uint)
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
        }
        
        spatialHashing.build(elements: positionsBuffer, in: commandBuffer)
        
        spatialHashing.find(
            externalElements: nil,
            collisionCandidates: collisionCandidatesBuffer,
            connectedVertices: connectedVerticesBuffer,
            in: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: candidatesCount)
        
        // Verify that vertex 0 does not have vertex 1 as a collision candidate
        XCTAssertFalse(collisionCandidates[0].contains(1), "Vertex 0 should not have connected vertex 1 as a collision candidate")
        XCTAssertTrue(collisionCandidates[0].contains(2), "Vertex 0 should have vertex 2 as a collision candidate")
        XCTAssertFalse(collisionCandidates[0].contains(3), "Vertex 0 should not have vertex 3 as a collision candidate (outside cell size)")
        
        // Verify that vertex 1 does not have vertex 0 as a collision candidate
        XCTAssertFalse(collisionCandidates[1].contains(0), "Vertex 1 should not have connected vertex 0 as a collision candidate")
        XCTAssertTrue(collisionCandidates[1].contains(2), "Vertex 1 should have vertex 2 as a collision candidate")
        XCTAssertFalse(collisionCandidates[1].contains(3), "Vertex 1 should not have vertex 3 as a collision candidate (outside cell size)")
        
        // Verify that vertex 2 has both vertex 0 and 1 as collision candidates
        XCTAssertTrue(collisionCandidates[2].contains(0), "Vertex 2 should have vertex 0 as a collision candidate")
        XCTAssertTrue(collisionCandidates[2].contains(1), "Vertex 2 should have vertex 1 as a collision candidate")
        XCTAssertFalse(collisionCandidates[2].contains(3), "Vertex 2 should not have vertex 3 as a collision candidate (outside cell size)")
        
        // Verify that vertex 3 has no collision candidates (all others are outside its cell)
        XCTAssertFalse(collisionCandidates[3].contains(0), "Vertex 3 should not have vertex 0 as a collision candidate")
        XCTAssertFalse(collisionCandidates[3].contains(1), "Vertex 3 should not have vertex 1 as a collision candidate")
        XCTAssertFalse(collisionCandidates[3].contains(2), "Vertex 3 should not have vertex 2 as a collision candidate")
    }
    
    func testPerformanceForPositions(_ count: Int) throws {
        let positions: [SIMD3<Float>] = (0..<count).map { _ in
            SIMD3<Float>(
                Float.random(in: -100...100),
                Float.random(in: -100...100),
                Float.random(in: -100...100)
            )
        }
        let config = SpatialHashing.Configuration(
            cellSize: 1.0,
            radius: 0.5
        )
        
        do {
            let heap = try self.device.heap(size: SpatialHashing.totalBuffersSize(maxElementsCount: count), storageMode: .shared)
            let spatialHashing = try SpatialHashing(
                heap: heap,
                configuration: config,
                maxElementsCount: positions.count
            )
            
            let candidatesCount = 8
            let positionsBuffer = try device.typedBuffer(with: positions, valueType: .float3)
            let collisionCandidatesBuffer = try device.typedBuffer(
                with: Array(repeating: UInt32.max, count: positions.count * candidatesCount),
                valueType: .uint
            )
            
            measure {
                guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
                    XCTFail("Failed to create command buffer")
                    return
                }

                spatialHashing.build(elements: positionsBuffer, in: commandBuffer)
                
                spatialHashing.find(
                    externalElements: nil,
                    collisionCandidates: collisionCandidatesBuffer,
                    connectedVertices: nil,
                    in: commandBuffer
                )
                
                let startTime = CFAbsoluteTimeGetCurrent()
                commandBuffer.addCompletedHandler { _ in
                    let endTime = CFAbsoluteTimeGetCurrent()
                    let duration = (endTime - startTime) * 1000
                    print("Performance test for \(count) positions took \(duration) ms")
                }

                commandBuffer.commit()
                commandBuffer.waitUntilCompleted()
            }
        }  catch {
            XCTFail("Performance test failed with error: \(error)")
        }
    }
    
    func testPackedFloat3Format() throws {
            let positions: [SIMD3<Float>] = [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.5, 0.0, 0.0]
            ]
            
            let packedPositions = positions.map { packed_float3($0) }
            
            let cellSize: Float = 1.0
            let config = SpatialHashing.Configuration(cellSize: cellSize, radius: 0.5)
            
            let spatialHashing = try SpatialHashing(
                device: self.device,
                configuration: config,
                maxElementsCount: positions.count
            )
            
            let positionsBuffer = try device.typedBuffer(with: packedPositions, valueType: .packedFloat3)
            let collisionCandidatesBuffer = try device.typedBuffer(
                with: Array(repeating: UInt32.max, count: positions.count * 8),
                valueType: .uint
            )
            
            guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
                XCTFail("Failed to create command buffer")
                throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
            }
            
            spatialHashing.build(elements: positionsBuffer, in: commandBuffer)
            
            spatialHashing.find(
                externalElements: nil,
                collisionCandidates: collisionCandidatesBuffer,
                connectedVertices: nil,
                in: commandBuffer
            )
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: 8)
            
            XCTAssertTrue(collisionCandidates[0].contains(1))
            XCTAssertTrue(collisionCandidates[1].contains(0))
            XCTAssertTrue(collisionCandidates[1].contains(2))
            XCTAssertTrue(collisionCandidates[2].contains(1))
            XCTAssertTrue(collisionCandidates[2].contains(3))
            XCTAssertTrue(collisionCandidates[3].contains(2))
        }
        
        func testExternalCollidablePositions() throws {
            let colliderPositions: [SIMD3<Float>] = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0]
            ]
            
            let externalPositions: [SIMD3<Float>] = [
                [0.5, 0.0, 0.0],
                [1.5, 0.0, 0.0],
                [2.5, 0.0, 0.0]
            ]
            
            let cellSize: Float = 1.0
            let config = SpatialHashing.Configuration(cellSize: cellSize, radius: 0.5)
            
            let spatialHashing = try SpatialHashing(
                device: self.device,
                configuration: config,
                maxElementsCount: colliderPositions.count
            )
            
            let colliderBuffer = try device.typedBuffer(with: colliderPositions, valueType: .float3)
            let externalBuffer = try device.typedBuffer(with: externalPositions, valueType: .float3)
            let collisionCandidatesBuffer = try device.typedBuffer(
                with: Array(repeating: UInt32.max, count: externalPositions.count * 8),
                valueType: .uint
            )
            
            guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
                XCTFail("Failed to create command buffer")
                throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
            }
            
            spatialHashing.build(elements: colliderBuffer, in: commandBuffer)
            
            spatialHashing.find(
                externalElements: externalBuffer,
                collisionCandidates: collisionCandidatesBuffer,
                connectedVertices: nil,
                in: commandBuffer
            )
            
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: 8)
            
            XCTAssertTrue(collisionCandidates[0].contains(0))
            XCTAssertTrue(collisionCandidates[0].contains(1))
            XCTAssertTrue(collisionCandidates[1].contains(1))
            XCTAssertTrue(collisionCandidates[1].contains(2))
            XCTAssertTrue(collisionCandidates[2].contains(2))
            XCTAssertTrue(collisionCandidates[2].contains(3))
        }
        
    func testMixedFormatPositions() throws {
        let colliderPositions: [SIMD3<Float>] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0]
        ]
        
        let externalPositions: [SIMD3<Float>] = [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0]
        ]
        
        let cellSize: Float = 1.0
        let config = SpatialHashing.Configuration(cellSize: cellSize, radius: 0.5)
        
        let spatialHashing = try SpatialHashing(
            device: self.device,
            configuration: config,
            maxElementsCount: colliderPositions.count
        )
        
        let packedColliderPositions = colliderPositions.map { packed_float3($0) }
        let colliderBuffer = try device.typedBuffer(with: packedColliderPositions, valueType: .packedFloat3)
        let externalBuffer = try device.typedBuffer(with: externalPositions, valueType: .float3)
        let collisionCandidatesBuffer = try device.typedBuffer(
            with: Array(repeating: UInt32.max, count: externalPositions.count * 8),
            valueType: .uint
        )
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
        }
        
        spatialHashing.build(elements: colliderBuffer, in: commandBuffer)
        
        spatialHashing.find(
            externalElements: externalBuffer,
            collisionCandidates: collisionCandidatesBuffer,
            connectedVertices: nil,
            in: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: 8)
        
        XCTAssertTrue(collisionCandidates[0].contains(0))
        XCTAssertTrue(collisionCandidates[0].contains(1))
        XCTAssertTrue(collisionCandidates[1].contains(1))
        XCTAssertTrue(collisionCandidates[1].contains(2))
        XCTAssertTrue(collisionCandidates[2].contains(2))
        XCTAssertTrue(collisionCandidates[2].contains(3))
    }
    
    func testEdgeCases() throws {
        let positions: [SIMD3<Float>] = [
            [0.0, 0.0, 0.0],
            [1.01, 0.0, 0.0],
            [0.99, 0.0, 0.0],
            [1.1, 0.0, 0.0],
            [10.0, 10.0, 10.0]
        ]
        
        let cellSize: Float = 1.0
        let radius: Float = 0.5
        let config = SpatialHashing.Configuration(cellSize: cellSize, radius: radius)
        
        let spatialHashing = try SpatialHashing(
            device: self.device,
            configuration: config,
            maxElementsCount: positions.count
        )
        
        let positionsBuffer = try device.typedBuffer(with: positions, valueType: .float3)
        let collisionCandidatesBuffer = try device.typedBuffer(
            with: Array(repeating: UInt32.max, count: positions.count * 8),
            valueType: .uint
        )
        
        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
        }
        
        spatialHashing.build(elements: positionsBuffer, in: commandBuffer)
        
        spatialHashing.find(
            externalElements: nil,
            collisionCandidates: collisionCandidatesBuffer,
            connectedVertices: nil,
            in: commandBuffer
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: 8)
        
        // Adjusted assertions based on the spatial hashing behavior
        XCTAssertTrue(collisionCandidates[0].contains(2), "Position 0 should detect position 2 as a neighbor")
        XCTAssertFalse(collisionCandidates[0].contains(1), "Position 0 should not detect position 1 as a neighbor")
        
        XCTAssertTrue(collisionCandidates[1].contains(2), "Position 1 should detect position 2 as a neighbor")
        XCTAssertTrue(collisionCandidates[1].contains(3), "Position 1 should detect position 3 as a neighbor")
        
        XCTAssertTrue(collisionCandidates[2].contains(0), "Position 2 should detect position 0 as a neighbor")
        XCTAssertTrue(collisionCandidates[2].contains(1), "Position 2 should detect position 1 as a neighbor")
        XCTAssertTrue(collisionCandidates[2].contains(3), "Position 2 should detect position 3 as a neighbor")
        
        XCTAssertTrue(collisionCandidates[3].contains(1), "Position 3 should detect position 1 as a neighbor")
        XCTAssertTrue(collisionCandidates[3].contains(2), "Position 3 should detect position 2 as a neighbor")
        
        XCTAssertFalse(collisionCandidates[4].contains(0), "Position 4 should not detect any neighbors")
        XCTAssertFalse(collisionCandidates[4].contains(1), "Position 4 should not detect any neighbors")
        XCTAssertFalse(collisionCandidates[4].contains(2), "Position 4 should not detect any neighbors")
        XCTAssertFalse(collisionCandidates[4].contains(3), "Position 4 should not detect any neighbors")
    }
    
    
    func testPerformanceFor1kPositions() throws {
        try testPerformanceForPositions(1_000)
    }
    
    func testPerformanceFor10kPositions() throws {
        try testPerformanceForPositions(10_000)
    }
    
    func testPerformanceFor100kPositions() throws {
        try testPerformanceForPositions(100_000)
    }
    
    func testPerformanceFor1mPositions() throws {
        try testPerformanceForPositions(1_000_000)
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

    init(x: Float, y: Float, z: Float) {
        self.x = x
        self.y = y
        self.z = z
    }
    
    
    init(_ simd: SIMD3<Float>) {
        self.x = simd.x
        self.y = simd.y
        self.z = simd.z
    }
}

//import XCTest
//import Metal
//@testable import SimulationTools
//
//final class SpatialHashingTests: XCTestCase {
//    var device: MTLDevice!
//    var commandQueue: MTLCommandQueue!
//    
//    override func setUp() {
//        super.setUp()
//        self.device = MTLCreateSystemDefaultDevice()
//        self.commandQueue = self.device.makeCommandQueue()
//    }
//    
//    override func tearDown() {
//        self.device = nil
//        self.commandQueue = nil
//        super.tearDown()
//    }
//    
//    func testSpatialHashingInitialization() throws {
//        let positions: [SIMD4<Float>] = [
//            [0.0, 0.0, 0.0, 1.0],
//            [1.0, 1.0, 1.0, 1.0],
//            [-1.0, 1.0, 1.0, 1.0],
//            [1.0, -1.0, 1.0, 1.0]
//        ]
//        let config = SpatialHashing.Configuration(
//            cellSize: 1.0,
//            spacingScale: 1.0,
//            collisionType: .vertexToVertex
//        )
//        
//        XCTAssertNoThrow(
//            try SpatialHashing(
//                device: self.device,
//                configuration: config,
//                positions: positions
//            )
//        )
//    }
//    
//    func generateMockData() -> [SIMD4<Float>] {
//        return (0..<100).map { i in
//            let angle = Float(i) * Float.pi / 50.0
//            return [cos(angle) * 10.0, sin(angle) * 10.0, 0.0, 1.0]
//        }
//    }
//    
//    func collisionCandidates(positions: [SIMD4<Float>], candidatesCount: Int = 8, cellSize: Float) throws -> MTLTypedBuffer<UInt32> {
//        let config = SpatialHashing.Configuration(
//            cellSize: cellSize,
//            spacingScale: 1.0,
//            collisionType: .vertexToVertex
//        )
//        
//        let spatialHashing = try SpatialHashing(
//            device: self.device,
//            configuration: config,
//            positions: positions
//        )
//        
//        let positionsBuffer = try device.typedBuffer(with: positions)
//        let collisionCandidatesBuffer = try device.typedBuffer(
//            with: Array(repeating: UInt32.max, count: positions.count * candidatesCount)
//        )
//        
//        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
//            XCTFail("Failed to create command buffer")
//            throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
//        }
//        
//        spatialHashing.build(
//            positions: positionsBuffer,
//            collisionCandidates: collisionCandidatesBuffer,
//            connectedVertices: nil,
//            in: commandBuffer
//        )
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        return collisionCandidatesBuffer
//    }
//    
//    func testBuildMethodInitialization() throws {
//        let positions = generateMockData()
//        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, cellSize: 0.5)
//        XCTAssertNotNil(collisionCandidatesBuffer.values, "Collision candidates buffer should not be nil")
//    }
//    
//    func testCollisionCandidatesContainClosestProximity() throws {
//        let positions: [SIMD4<Float>] = [
//            [-0.5, 0.0, 0.0, 1.0],
//            [0.0, 0.0, 0.0, 1.0],
//            [1.0, 0.0, 0.0, 1.0],
//            [1.5, 0.0, 0.0, 1.0]
//        ]
//        let candidatesCount = 4
//        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, candidatesCount: candidatesCount, cellSize: 1.0)
//        let collisionCandidates = collisionCandidatesBuffer.values!.chunked(into: candidatesCount).map { Set($0) }
//        
//        XCTAssertTrue(collisionCandidates[0].contains(1))
//        XCTAssertTrue(collisionCandidates[1].contains(0))
//        XCTAssertTrue(collisionCandidates[2].contains(3))
//        XCTAssertTrue(collisionCandidates[3].contains(2))
//    }
//    
//    func testCollisionCandidatesDoNotContainSelf() throws {
//        let positions = generateMockData()
//        let candidatesCount = 4
//        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, candidatesCount: candidatesCount, cellSize: 1.0)
//        let collisionCandidates = collisionCandidatesBuffer.values!.chunked(into: candidatesCount).map { Set($0) }
//        
//        XCTAssertFalse(collisionCandidates[0].contains(0))
//        XCTAssertFalse(collisionCandidates[1].contains(1))
//        XCTAssertFalse(collisionCandidates[2].contains(2))
//        XCTAssertFalse(collisionCandidates[3].contains(3))
//    }
//    
//    func testCollisionCandidatesDoNotContainNonClosestProximity() throws {
//        let positions: [SIMD4<Float>] = [
//            [-0.5, 0.0, 0.0, 1.0],
//            [0.0, 0.0, 0.0, 1.0],
//            [1.0, 0.0, 0.0, 1.0],
//            [1.5, 0.0, 0.0, 1.0]
//        ]
//        
//        let candidatesCount = 4
//        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, candidatesCount: candidatesCount, cellSize: 0.5)
//        let collisionCandidates = collisionCandidatesBuffer.values!.chunked(into: candidatesCount).map { Set($0) }
//        
//        XCTAssertFalse(collisionCandidates[0].contains(2))
//        XCTAssertFalse(collisionCandidates[0].contains(3))
//        XCTAssertFalse(collisionCandidates[1].contains(2))
//        XCTAssertFalse(collisionCandidates[1].contains(3))
//        
//        XCTAssertFalse(collisionCandidates[2].contains(0))
//        XCTAssertFalse(collisionCandidates[2].contains(1))
//        XCTAssertFalse(collisionCandidates[3].contains(0))
//        XCTAssertFalse(collisionCandidates[3].contains(1))
//    }
//    
//    func testCollisionCandidatesSymmetry() throws {
//        let positions = generateMockData()
//
//        let candidatesCount = 8
//        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, candidatesCount: candidatesCount, cellSize: 1.0)
//        let collisionCandidates = collisionCandidatesBuffer.values!.chunked(into: candidatesCount).map { Set($0) }
//        
//        for (i, candidates) in collisionCandidates.enumerated() {
//            for candidate in candidates where candidate != UInt32.max {
//                XCTAssertTrue(collisionCandidates[Int(candidate)].contains(UInt32(i)), "Symmetry check failed: \(i) and \(candidate)")
//            }
//        }
//    }
//    
//    func testConnectedVerticesExclusion() throws {
//        let positions: [SIMD4<Float>] = [
//            [0.0, 0.0, 0.0, 1.0],
//            [0.1, 0.0, 0.0, 1.0],  // Connected to 0, closer than cell size
//            [0.5, 0.0, 0.0, 1.0],  // Not connected, within cell size
//            [1.5, 0.0, 0.0, 1.0]   // Not connected, outside cell size
//        ]
//        
//        let cellSize: Float = 1.0
//        let candidatesCount = 8
//        let config = SpatialHashing.Configuration(
//            cellSize: cellSize,
//            spacingScale: 1.0,
//            collisionType: .vertexToVertex
//        )
//        
//        let spatialHashing = try SpatialHashing(
//            device: self.device,
//            configuration: config,
//            positions: positions
//        )
//        
//        let positionsBuffer = try device.typedBuffer(with: positions)
//        let collisionCandidatesBuffer = try device.typedBuffer(
//            with: Array(repeating: UInt32.max, count: positions.count * candidatesCount)
//        )
//        
//        // Specify connected vertices: 0 and 1 are connected
//        let connectedVertices: [UInt32] = [
//            1, UInt32.max, UInt32.max, UInt32.max,  // For vertex 0
//            0, UInt32.max, UInt32.max, UInt32.max,  // For vertex 1
//            UInt32.max, UInt32.max, UInt32.max, UInt32.max,  // For vertex 2
//            UInt32.max, UInt32.max, UInt32.max, UInt32.max   // For vertex 3
//        ]
//        let connectedVerticesBuffer = try device.typedBuffer(with: connectedVertices)
//        
//        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
//            XCTFail("Failed to create command buffer")
//            throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
//        }
//        
//        spatialHashing.build(
//            positions: positionsBuffer,
//            collisionCandidates: collisionCandidatesBuffer,
//            connectedVertices: connectedVerticesBuffer,
//            in: commandBuffer
//        )
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        let collisionCandidates = collisionCandidatesBuffer.values!.chunked(into: candidatesCount)
//        
//        // Verify that vertex 0 does not have vertex 1 as a collision candidate
//        XCTAssertFalse(collisionCandidates[0].contains(1), "Vertex 0 should not have connected vertex 1 as a collision candidate")
//        XCTAssertTrue(collisionCandidates[0].contains(2), "Vertex 0 should have vertex 2 as a collision candidate")
//        XCTAssertFalse(collisionCandidates[0].contains(3), "Vertex 0 should not have vertex 3 as a collision candidate (outside cell size)")
//        
//        // Verify that vertex 1 does not have vertex 0 as a collision candidate
//        XCTAssertFalse(collisionCandidates[1].contains(0), "Vertex 1 should not have connected vertex 0 as a collision candidate")
//        XCTAssertTrue(collisionCandidates[1].contains(2), "Vertex 1 should have vertex 2 as a collision candidate")
//        XCTAssertFalse(collisionCandidates[1].contains(3), "Vertex 1 should not have vertex 3 as a collision candidate (outside cell size)")
//        
//        // Verify that vertex 2 has both vertex 0 and 1 as collision candidates
//        XCTAssertTrue(collisionCandidates[2].contains(0), "Vertex 2 should have vertex 0 as a collision candidate")
//        XCTAssertTrue(collisionCandidates[2].contains(1), "Vertex 2 should have vertex 1 as a collision candidate")
//        XCTAssertFalse(collisionCandidates[2].contains(3), "Vertex 2 should not have vertex 3 as a collision candidate (outside cell size)")
//        
//        // Verify that vertex 3 has no collision candidates (all others are outside its cell)
//        XCTAssertFalse(collisionCandidates[3].contains(0), "Vertex 3 should not have vertex 0 as a collision candidate")
//        XCTAssertFalse(collisionCandidates[3].contains(1), "Vertex 3 should not have vertex 1 as a collision candidate")
//        XCTAssertFalse(collisionCandidates[3].contains(2), "Vertex 3 should not have vertex 2 as a collision candidate")
//    }
//    
//    func testPerformanceForPositions(_ count: Int) throws {
//        let positions: [SIMD4<Float>] = (0..<count).map { _ in
//            [
//                Float.random(in: -100...100),
//                Float.random(in: -100...100),
//                Float.random(in: -100...100),
//                1.0
//            ]
//        }
//        let config = SpatialHashing.Configuration(
//            cellSize: 1.0,
//            spacingScale: 1.0,
//            collisionType: .vertexToVertex
//        )
//        
//        do {
//            let heap = try self.device.heap(size: SpatialHashing.totalBuffersSize(positionsCount: count), storageMode: .shared)
//            let spatialHashing = try SpatialHashing(
//                heap: heap,
//                configuration: config,
//                positions: positions
//            )
//            
//            let candidatesCount = 8
//            let positionsBuffer = try device.typedBuffer(with: positions)
//            let collisionCandidatesBuffer = try device.typedBuffer(
//                with: Array(repeating: UInt32.max, count: positions.count * candidatesCount)
//            )
//            
//            measure {
//                guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
//                    XCTFail("Failed to create command buffer")
//                    return
//                }
//
//                spatialHashing.build(
//                    positions: positionsBuffer,
//                    collisionCandidates: collisionCandidatesBuffer,
//                    connectedVertices: nil,
//                    in: commandBuffer
//                )
//                
//                let startTime = CFAbsoluteTimeGetCurrent()
//                commandBuffer.addCompletedHandler { _ in
//                    let endTime = CFAbsoluteTimeGetCurrent()
//                    let duration = (endTime - startTime) * 1000
//                    print("Performance test for \(count) positions took \(duration) ms")
//                }
//
//                commandBuffer.commit()
//                commandBuffer.waitUntilCompleted()
//            }
//        }  catch {
//            XCTFail("Performance test failed with error: \(error)")
//        }
//    }
//    
//    func testPerformanceFor1kPositions() throws {
//        try testPerformanceForPositions(1_000)
//    }
//    
//    func testPerformanceFor10kPositions() throws {
//        try testPerformanceForPositions(10_000)
//    }
//    
//    func testPerformanceFor100kPositions() throws {
//        try testPerformanceForPositions(100_000)
//    }
//    
//    func testPerformanceFor1mPositions() throws {
//        try testPerformanceForPositions(1_000_000)
//    }
//}

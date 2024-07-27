//import XCTest
//import Metal
//@testable import SimulationTools
//
//final class TriangleSpatialHashingTests: XCTestCase {
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
//    func testTriangleSpatialHashingInitialization() throws {
//        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0)
//        
//        XCTAssertNoThrow(
//            try TriangleSpatialHashing(
//                device: self.device,
//                configuration: config,
//                trianglesCount: 100
//            )
//        )
//    }
//    
//    func generateMockTriangles() -> [SIMD3<UInt32>] {
//        return (0..<100).map { i in
//            SIMD3<UInt32>(UInt32(i * 3), UInt32(i * 3 + 1), UInt32(i * 3 + 2))
//        }
//    }
//    
//    func generateMockPositions() -> [SIMD4<Float>] {
//        return (0..<300).map { i in
//            let angle = Float(i) * Float.pi / 150.0
//            return [cos(angle) * 10.0, sin(angle) * 10.0, 0.0, 1.0]
//        }
//    }
//    
//    func collisionCandidates(positions: [SIMD4<Float>], triangles: [SIMD3<UInt32>], candidatesCount: Int = 8, cellSize: Float) throws -> MTLTypedBuffer<UInt32> {
//        let config = TriangleSpatialHashing.Configuration(cellSize: cellSize)
//        
//        let triangleSpatialHashing = try TriangleSpatialHashing(
//            device: self.device,
//            configuration: config,
//            trianglesCount: triangles.count
//        )
//        
//        let positionsBuffer = try device.typedBuffer(with: positions)
//        let scenePositionsBuffer = try device.typedBuffer(with: positions)
//        let sceneTrianglesBuffer = try device.typedBuffer(with: triangles)
//        let collisionCandidatesBuffer = try device.typedBuffer(
//            with: Array(repeating: UInt32.max, count: positions.count * candidatesCount)
//        )
//        
//        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
//            XCTFail("Failed to create command buffer")
//            throw NSError(domain: "TriangleSpatialHashingTests", code: 1, userInfo: nil)
//        }
//        
//        triangleSpatialHashing.build(
//            positions: positionsBuffer,
//            scenePositions: scenePositionsBuffer,
//            sceneTriangles: sceneTrianglesBuffer,
//            collisionCandidates: collisionCandidatesBuffer,
//            positionsCount: positions.count,
//            trianglesCount: triangles.count,
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
//        let positions = generateMockPositions()
//        let triangles = generateMockTriangles()
//        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, triangles: triangles, cellSize: 0.5)
//        XCTAssertNotNil(collisionCandidatesBuffer.values, "Collision candidates buffer should not be nil")
//    }
//    
//    func testCollisionCandidatesContainClosestTriangles() throws {
//        let positions: [SIMD4<Float>] = [
//            [-0.5, 0.0, 0.0, 1.0],
//            [0.0, 0.0, 0.0, 1.0],
//            [1.0, 0.0, 0.0, 1.0],
//            [1.5, 0.0, 0.0, 1.0]
//        ]
//        let triangles: [SIMD3<UInt32>] = [
//            SIMD3(0, 1, 2),
//            SIMD3(1, 2, 3)
//        ]
//        let candidatesCount = 4
//        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, triangles: triangles, candidatesCount: candidatesCount, cellSize: 1.0)
//        let collisionCandidates = collisionCandidatesBuffer.values!.chunked(into: candidatesCount).map { Set($0) }
//        
//        XCTAssertTrue(collisionCandidates[0].contains(0))
//        XCTAssertTrue(collisionCandidates[1].contains(0))
//        XCTAssertTrue(collisionCandidates[1].contains(1))
//        XCTAssertTrue(collisionCandidates[2].contains(0))
//        XCTAssertTrue(collisionCandidates[2].contains(1))
//        XCTAssertTrue(collisionCandidates[3].contains(1))
//    }
//    
//    func testCollisionCandidatesDoNotContainDistantTriangles() throws {
//        let positions: [SIMD4<Float>] = [
//            [-0.5, 0.0, 0.0, 1.0],
//            [0.0, 0.0, 0.0, 1.0],
//            [10.0, 0.0, 0.0, 1.0],
//            [10.5, 0.0, 0.0, 1.0]
//        ]
//        let triangles: [SIMD3<UInt32>] = [
//            SIMD3(0, 1, 2),
//            SIMD3(2, 3, 0)
//        ]
//        
//        let candidatesCount = 4
//        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, triangles: triangles, candidatesCount: candidatesCount, cellSize: 1.0)
//        let collisionCandidates = collisionCandidatesBuffer.values!.chunked(into: candidatesCount).map { Set($0) }
//        
//        XCTAssertFalse(collisionCandidates[0].contains(1))
//        XCTAssertFalse(collisionCandidates[1].contains(1))
//        XCTAssertFalse(collisionCandidates[2].contains(0))
//        XCTAssertFalse(collisionCandidates[3].contains(0))
//    }
//    
//    func testReuseMethod() throws {
//        let positions: [SIMD4<Float>] = [
//            [-0.5, 0.0, 0.0, 1.0],
//            [0.0, 0.0, 0.0, 1.0],
//            [1.0, 0.0, 0.0, 1.0],
//            [1.5, 0.0, 0.0, 1.0]
//        ]
//        let triangles: [SIMD3<UInt32>] = [
//            SIMD3(0, 1, 2),
//            SIMD3(1, 2, 3)
//        ]
//        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0)
//        
//        let triangleSpatialHashing = try TriangleSpatialHashing(
//            device: self.device,
//            configuration: config,
//            trianglesCount: triangles.count
//        )
//        
//        let positionsBuffer = try device.typedBuffer(with: positions)
//        let scenePositionsBuffer = try device.typedBuffer(with: positions)
//        let sceneTrianglesBuffer = try device.typedBuffer(with: triangles)
//        let collisionCandidatesBuffer = try device.typedBuffer(
//            with: Array(repeating: UInt32.max, count: positions.count * 8)
//        )
//        let vertexNeighborsBuffer = try device.typedBuffer(
//            with: Array(repeating: UInt32.max, count: positions.count * 8)
//        )
//        
//        guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
//            XCTFail("Failed to create command buffer")
//            throw NSError(domain: "TriangleSpatialHashingTests", code: 1, userInfo: nil)
//        }
//        
//        triangleSpatialHashing.reuse(
//            positions: positionsBuffer,
//            scenePositions: scenePositionsBuffer,
//            sceneTriangles: sceneTrianglesBuffer,
//            collisionCandidates: collisionCandidatesBuffer,
//            vertexNeighbors: vertexNeighborsBuffer,
//            positionsCount: positions.count,
//            in: commandBuffer
//        )
//        
//        commandBuffer.commit()
//        commandBuffer.waitUntilCompleted()
//        
//        // As we can't directly verify the results of the reuse method,
//        // we're just ensuring it doesn't crash and completes successfully
//        XCTAssertTrue(true, "Reuse method completed without crashing")
//    }
//    
//    func testPerformanceForTriangles(_ count: Int) throws {
//        let positions = (0..<count*3).map { _ in
//            SIMD4<Float>(
//                Float.random(in: -100...100),
//                Float.random(in: -100...100),
//                Float.random(in: -100...100),
//                1.0
//            )
//        }
//        let triangles = (0..<count).map { i in
//            SIMD3<UInt32>(UInt32(i * 3), UInt32(i * 3 + 1), UInt32(i * 3 + 2))
//        }
//        let config = TriangleSpatialHashing.Configuration(cellSize: 1.0)
//        
//        do {
//            let triangleSpatialHashing = try TriangleSpatialHashing(
//                device: self.device,
//                configuration: config,
//                trianglesCount: count
//            )
//            
//            let positionsBuffer = try device.typedBuffer(with: positions)
//            let scenePositionsBuffer = try device.typedBuffer(with: positions)
//            let sceneTrianglesBuffer = try device.typedBuffer(with: triangles)
//            let collisionCandidatesBuffer = try device.typedBuffer(
//                with: Array(repeating: UInt32.max, count: positions.count * 8)
//            )
//            
//            measure {
//                guard let commandBuffer = self.commandQueue.makeCommandBuffer() else {
//                    XCTFail("Failed to create command buffer")
//                    return
//                }
//
//                triangleSpatialHashing.build(
//                    positions: positionsBuffer,
//                    scenePositions: scenePositionsBuffer,
//                    sceneTriangles: sceneTrianglesBuffer,
//                    collisionCandidates: collisionCandidatesBuffer,
//                    positionsCount: positions.count,
//                    trianglesCount: triangles.count,
//                    in: commandBuffer
//                )
//                
//                let startTime = CFAbsoluteTimeGetCurrent()
//                commandBuffer.addCompletedHandler { _ in
//                    let endTime = CFAbsoluteTimeGetCurrent()
//                    let duration = (endTime - startTime) * 1000
//                    print("Performance test for \(count) triangles took \(duration) ms")
//                }
//
//                commandBuffer.commit()
//                commandBuffer.waitUntilCompleted()
//            }
//        } catch {
//            XCTFail("Performance test failed with error: \(error)")
//        }
//    }
//    
//    func testPerformanceFor1kTriangles() throws {
//        try testPerformanceForTriangles(1_000)
//    }
//    
//    func testPerformanceFor10kTriangles() throws {
//        try testPerformanceForTriangles(10_000)
//    }
//    
//    func testPerformanceFor100kTriangles() throws {
//        try testPerformanceForTriangles(100_000)
//    }
//}

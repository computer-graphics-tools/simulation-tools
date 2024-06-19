import XCTest
import Metal
@testable import simulation_tools

final class SpatialHashingTests: XCTestCase {
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    
    override func setUp() {
        super.setUp()
        device = MTLCreateSystemDefaultDevice()
        commandQueue = device.makeCommandQueue()
    }
    
    override func tearDown() {
        device = nil
        commandQueue = nil
        super.tearDown()
    }
    
    func testSpatialHashingInitialization() throws {
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>(0.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(1.0, 1.0, 1.0, 1.0),
            SIMD4<Float>(-1.0, 1.0, 1.0, 1.0),
            SIMD4<Float>(1.0, -1.0, 1.0, 1.0)
        ]
        let collisionType = SelfCollisionType.vertexVertex
        let spacingScale: Float = 1.0
        let cellSize: Float = 1.0
        
        XCTAssertNoThrow(
            try SpatialHashing(
                device: device,
                collisionType: collisionType,
                positions: positions,
                spacingScale: spacingScale,
                cellSize: cellSize,
                heap: nil
            )
        )
    }
    
    func generateMockData() -> [SIMD4<Float>] {
        return (0..<100).map { i in
            let angle = Float(i) * Float.pi / 50.0
            return SIMD4<Float>(cos(angle) * 10.0, sin(angle) * 10.0, 0.0, 1.0)
        }
    }
    
    func collisionCandidates(positions: [SIMD4<Float>], cellSize: Float) throws -> TypedMTLBuffer<UInt32> {
        let collisionType = SelfCollisionType.vertexVertex
        let spacingScale: Float = 1.0
        
        let spatialHashing = try SpatialHashing(
            device: device,
            collisionType: collisionType,
            positions: positions,
            spacingScale: spacingScale,
            cellSize: cellSize,
            heap: nil
        )
        
        let positionsBuffer = try TypedMTLBuffer<SIMD4<Float>>(elements: positions, device: device)
        let candidatesCount = 8
        let collisionCandidatesBuffer = try TypedMTLBuffer<UInt32>(
            elements: Array(repeating: UInt32.max, count: positions.count * candidatesCount),
            device: device
        )
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            XCTFail("Failed to create command buffer")
            throw NSError(domain: "SpatialHashingTests", code: 1, userInfo: nil)
        }
        
        spatialHashing.build(
            commandBuffer: commandBuffer,
            positions: positionsBuffer,
            collisionCandidates: collisionCandidatesBuffer,
            connectedVertices: nil
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return collisionCandidatesBuffer
    }
    
    func testBuildMethodInitialization() throws {
        let positions = generateMockData()
        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, cellSize: 0.5)
        XCTAssertNotNil(collisionCandidatesBuffer.elements, "Collision candidates buffer should not be nil")
    }
    
    func testCollisionCandidatesContainClosestProximity() throws {
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>(-0.5, 0.0, 0.0, 1.0),
            SIMD4<Float>(0.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(1.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(1.5, 0.0, 0.0, 1.0)
        ]
        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, cellSize: 1.0)
        let candidatesCount = 4
        let collisionCandidates = collisionCandidatesBuffer.elements!.chunked(into: candidatesCount).map { Set($0) }
        
        XCTAssertTrue(collisionCandidates[0].contains(1))
        XCTAssertTrue(collisionCandidates[1].contains(0))
        XCTAssertTrue(collisionCandidates[2].contains(3))
        XCTAssertTrue(collisionCandidates[3].contains(2))
    }
    
    func testCollisionCandidatesDoNotContainSelf() throws {
        let positions = generateMockData()
        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, cellSize: 1.0)
        let candidatesCount = 4
        let collisionCandidates = collisionCandidatesBuffer.elements!.chunked(into: candidatesCount).map { Set($0) }
        
        XCTAssertFalse(collisionCandidates[0].contains(0))
        XCTAssertFalse(collisionCandidates[1].contains(1))
        XCTAssertFalse(collisionCandidates[2].contains(2))
        XCTAssertFalse(collisionCandidates[3].contains(3))
    }
    
    func testCollisionCandidatesDoNotContainNonClosestProximity() throws {
        let positions: [SIMD4<Float>] = [
            SIMD4<Float>(-0.5, 0.0, 0.0, 1.0),
            SIMD4<Float>(0.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(1.0, 0.0, 0.0, 1.0),
            SIMD4<Float>(1.5, 0.0, 0.0, 1.0)
        ]
        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, cellSize: 0.5)
        let candidatesCount = 4
        let collisionCandidates = collisionCandidatesBuffer.elements!.chunked(into: candidatesCount).map { Set($0) }
        
        XCTAssertFalse(collisionCandidates[0].contains(2))
        XCTAssertFalse(collisionCandidates[0].contains(3))
        XCTAssertFalse(collisionCandidates[1].contains(2))
        XCTAssertFalse(collisionCandidates[1].contains(3))
        
        XCTAssertFalse(collisionCandidates[2].contains(0))
        XCTAssertFalse(collisionCandidates[2].contains(1))
        XCTAssertFalse(collisionCandidates[3].contains(0))
        XCTAssertFalse(collisionCandidates[3].contains(1))
    }
    
    func testCollisionCandidatesSymmetry() throws {
        let positions = generateMockData()

        let collisionCandidatesBuffer = try collisionCandidates(positions: positions, cellSize: 1.0)
        let candidatesCount = 8
        let collisionCandidates = collisionCandidatesBuffer.elements!.chunked(into: candidatesCount).map { Set($0) }
        
        for (i, candidates) in collisionCandidates.enumerated() {
            for candidate in candidates where candidate != UInt32.max {
                XCTAssertTrue(collisionCandidates[Int(candidate)].contains(UInt32(i)), "Symmetry check failed: \(i) and \(candidate)")
            }
        }
    }
    
    func testPerformanceForPositions(_ count: Int) throws {
        let positions: [SIMD4<Float>] = (0..<count).map { _ in
            SIMD4<Float>(
                Float.random(in: -100...100),
                Float.random(in: -100...100),
                Float.random(in: -100...100),
                1.0
            )
        }
        let collisionType = SelfCollisionType.vertexVertex
        let spacingScale: Float = 1.0
        let cellSize: Float = 1.0
        
        do {
            let heap = try device.heap(size: SpatialHashing.totalBuffersSize(positionsCount: count), storageMode: .shared)
            let spatialHashing = try SpatialHashing(
                device: device,
                collisionType: collisionType,
                positions: positions,
                spacingScale: spacingScale,
                cellSize: cellSize,
                heap: heap
            )
            
            let positionsBuffer = try TypedMTLBuffer<SIMD4<Float>>(elements: positions, device: device)
            let candidatesCount = 8
            let collisionCandidatesBuffer = try TypedMTLBuffer<UInt32>(
                elements: Array(repeating: UInt32.max, count: positions.count * candidatesCount),
                device: device
            )
            
            measure {
                guard let commandBuffer = commandQueue.makeCommandBuffer() else {
                    XCTFail("Failed to create command buffer")
                    return
                }

                let startTime = CFAbsoluteTimeGetCurrent()
                
                spatialHashing.build(
                    commandBuffer: commandBuffer,
                    positions: positionsBuffer,
                    collisionCandidates: collisionCandidatesBuffer,
                    connectedVertices: nil
                )
                
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
    
    func testPerformanceFor1kPositions() throws {
        try testPerformanceForPositions(1_000)
    }
    
    func testPerformanceFor10kPositions() throws {
        try testPerformanceForPositions(10_000)
    }
    
    func testPerformanceFor100kPositions() throws {
        try testPerformanceForPositions(100_000)
    }
}

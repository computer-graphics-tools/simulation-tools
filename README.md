# SimulationTools

## Description

SimulationTools is a GPU-accelerated library using Metal for various high-performance algorithms in 3D simulations and computational geometry.

## Key Components

### Spatial Hashing

Spatial hashing divides 3D space into a grid, enabling efficient neighbor searches and collision detection.

#### Algorithm Details:
1. Hash Computation: Each vertex is assigned to a cell based on its position.
2. Bitonic Sort: Hash-index pairs are sorted to resolve collisions and build cell buckets.
3. Cell Boundary Identification: Start and end indices for each cell are determined.
4. Collision Candidate Search: Potential colliders are identified within the same and adjacent cells.
5. Insertion Sort Storage: Collision candidates are stored using an insertion sort mechanism, maintaining a sorted list of the closest candidates.

#### Key Features:
- Efficient for both self-collision and external collision detection
- Insertion sort storage allows for temporal and structural reuse of candidates, improving performance in scenarios with coherent motion or stable structures

### Triangle Spatial Hashing

Optimized for triangle meshes, this variant enables efficient point-triangle collision detection.

#### Algorithm Details:
1. Triangle Hashing: Triangles are hashed into multiple cells they overlap.
2. Bucket Storage: Each cell maintains a fixed-size bucket of triangle indices.
3. Collision Candidate Search: For each query point, nearby triangles are identified through cell lookups.
4. Candidate Refinement: Distance computations are performed to sort and store the closest triangle candidates.

#### Key Features:
- Specialized for triangle mesh collision detection
- Supports both self-collision and external collision queries
- Efficient spatial reuse of collision information for improved performance in temporally coherent scenarios

## Performance Considerations

- GPU-optimized using Metal for scalability with large numbers of elements
- Insertion sort storage of candidates enables efficient updates in scenarios with small position changes
- Triangle spatial hashing optimizes memory usage through fixed-size buckets per cell
- Performance is sensitive to cell size: smaller cells increase precision but may reduce efficiency for large objects

## Usage Examples

### Spatial Hashing

Here are two examples demonstrating the use of SpatialHashing for both self-collision and external collision scenarios:

#### Self-Collision Example

```swift
import Metal
import SimulationTools

// Initialize Metal device and command queue
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

// Create a set of test positions
let positions: [SIMD3<Float>] = [
    [0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.5, 0.0],
    [1.0, 1.0, 1.0],
    [1.1, 1.1, 1.1]
]

// Configure spatial hashing
let config = SpatialHashing.Configuration(cellSize: 1.0, radius: 0.5)

// Create SpatialHashing instance
let spatialHashing = try! SpatialHashing(
    device: device,
    configuration: config,
    maxElementsCount: positions.count
)

// Create buffers
let positionsBuffer = try! device.makeBuffer(bytes: positions, length: MemoryLayout<SIMD3<Float>>.stride * positions.count, options: [])
let typedPositionsBuffer = try! device.typedBuffer(with: positions, valueType: .float3)
let collisionCandidatesBuffer = try! device.typedBuffer(
    with: Array(repeating: UInt32.max, count: positions.count * 8),
    valueType: .uint
)

// Create and execute command buffer
let commandBuffer = commandQueue.makeCommandBuffer()!
spatialHashing.build(elements: typedPositionsBuffer, in: commandBuffer)
spatialHashing.find(
    externalElements: nil,
    collisionCandidates: collisionCandidatesBuffer,
    connectedVertices: nil,
    in: commandBuffer
)
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// Process results
let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: 8)
for (index, candidates) in collisionCandidates.enumerated() {
    print("Collision candidates for position \(index): \(candidates.filter { $0 != .max })")
}
```

#### External Collision Example

```swift
import Metal
import SimulationTools

// Initialize Metal device and command queue
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

// Create a set of mesh positions and external query positions
let meshPositions: [SIMD3<Float>] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
]
let queryPositions: [SIMD3<Float>] = [
    [0.5, 0.5, 0.0],
    [1.5, 1.5, 0.0]
]

// Configure spatial hashing
let config = SpatialHashing.Configuration(cellSize: 1.0, radius: 0.5)

// Create SpatialHashing instance
let spatialHashing = try! SpatialHashing(
    device: device,
    configuration: config,
    maxElementsCount: max(meshPositions.count, queryPositions.count)
)

// Create buffers
let meshPositionsBuffer = try! device.typedBuffer(with: meshPositions, valueType: .float3)
let queryPositionsBuffer = try! device.typedBuffer(with: queryPositions, valueType: .float3)
let collisionCandidatesBuffer = try! device.typedBuffer(
    with: Array(repeating: UInt32.max, count: queryPositions.count * 8),
    valueType: .uint
)

// Create and execute command buffer
let commandBuffer = commandQueue.makeCommandBuffer()!
spatialHashing.build(elements: meshPositionsBuffer, in: commandBuffer)
spatialHashing.find(
    externalElements: queryPositionsBuffer,
    collisionCandidates: collisionCandidatesBuffer,
    connectedVertices: nil,
    in: commandBuffer
)
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// Process results
let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: 8)
for (index, candidates) in collisionCandidates.enumerated() {
    print("Collision candidates for query position \(index): \(candidates.filter { $0 != .max })")
}
```

### Triangle Spatial Hashing

Here's an example demonstrating the use of TriangleSpatialHashing for collision detection with a triangle mesh:

```swift
import Metal
import SimulationTools

// Initialize Metal device and command queue
let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

// Create a simple triangle mesh
let meshPositions: [SIMD3<Float>] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0]
]
let triangles: [SIMD3<UInt32>] = [
    [0, 1, 2],
    [1, 3, 2]
]

// Create query positions
let queryPositions: [SIMD3<Float>] = [
    [0.5, 0.5, 0.0],
    [1.5, 1.5, 0.0]
]

// Configure triangle spatial hashing
let config = TriangleSpatialHashing.Configuration(cellSize: 1.0, bucketSize: 8)

// Create TriangleSpatialHashing instance
let triangleSpatialHashing = try! TriangleSpatialHashing(
    device: device,
    configuration: config,
    trianglesCount: triangles.count
)

// Create buffers
let meshPositionsBuffer = try! device.typedBuffer(with: meshPositions, valueType: .float3)
let trianglesBuffer = try! device.typedBuffer(with: triangles, valueType: .uint3)
let queryPositionsBuffer = try! device.typedBuffer(with: queryPositions, valueType: .float3)
let collisionCandidatesBuffer = try! device.typedBuffer(
    with: Array(repeating: UInt32.max, count: queryPositions.count * 8),
    valueType: .uint
)

// Create and execute command buffer
let commandBuffer = commandQueue.makeCommandBuffer()!
triangleSpatialHashing.build(
    elements: meshPositionsBuffer,
    indices: trianglesBuffer,
    in: commandBuffer
)
triangleSpatialHashing.find(
    externalElements: queryPositionsBuffer,
    elements: meshPositionsBuffer,
    indices: trianglesBuffer,
    collisionCandidates: collisionCandidatesBuffer,
    in: commandBuffer
)
commandBuffer.commit()
commandBuffer.waitUntilCompleted()

// Process results
let collisionCandidates: [[UInt32]] = collisionCandidatesBuffer.values()!.chunked(into: 8)
for (index, candidates) in collisionCandidates.enumerated() {
    print("Collision candidates for query position \(index): \(candidates.filter { $0 != .max })")
}
```

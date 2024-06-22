# SimulationTools

## Description

SimulationTools is a set of GPU algorithms designed for various simulation tasks, currently focusing on spatial hashing for efficient collision detection.

## Algorithms

### Spatial Hashing

Spatial hashing is a technique used to divide space into a grid to quickly locate and manage vertices or other entities. This method reduces the complexity of finding neighboring vertices by organizing them into a structured grid.

**Overview:**

- **Hashing**: Each vertex is assigned to a cell based on its position in the space.
  
- **Bitonic Sorting**: Once the vertices are hashed into cells, a bitonic sort is performed to order the hash-index pairs. Sorting helps resolve hash collisions and build cell buckets that can contain multiple vertices.

- **Cell Bounds Identification**: After sorting, the start and end indices for each cell in the grid are identified. These indices describe the range of vertices within each cell, allowing for efficient access and iteration over the vertices in any given cell.

- **Collision Detection**:
  - **Vertex-Vertex**: For each vertex, potential collider vertices are identified within the same or adjacent cells. Collision candidates are then processed to determine actual collisions.

## Example Usage

Here's a basic example of how to use the SpatialHashing class:

```swift
import Metal
import simulation_tools

// Initialize Metal device
guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("Metal is not supported on this device")
}

// Create a command queue
guard let commandQueue = device.makeCommandQueue() else {
    fatalError("Failed to create command queue")
}

// Define positions for vertices
let positions: [SIMD4<Float>] = [
    SIMD4<Float>(0.0, 0.0, 0.0, 1.0),
    SIMD4<Float>(1.0, 1.0, 1.0, 1.0),
    SIMD4<Float>(-1.0, 1.0, 1.0, 1.0),
    SIMD4<Float>(1.0, -1.0, 1.0, 1.0)
]

// Configure spatial hashing
let config = SpatialHashing.Configuration(
    cellSize: 1.0,
    spacingScale: 1.0,
    collisionType: .vertexVertex
)

do {
    // Create SpatialHashing instance
    let spatialHashing = try SpatialHashing(
        device: device,
        configuration: config,
        positions: positions,
        heap: nil
    )

    // Create buffers
    let positionsBuffer = try TypedMTLBuffer<SIMD4<Float>>(values: positions, device: device)
    let collisionCandidatesBuffer = try TypedMTLBuffer<UInt32>(
        values: Array(repeating: UInt32.max, count: positions.count * 8),
        device: device
    )

    // Create command buffer
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        fatalError("Failed to create command buffer")
    }

    // Build spatial hash and find collision candidates
    spatialHashing.build(
        commandBuffer: commandBuffer,
        positions: positionsBuffer,
        collisionCandidates: collisionCandidatesBuffer,
        connectedVertices: nil
    )

    // Commit and wait for completion
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    // Process collision candidates
    if let collisionCandidates = collisionCandidatesBuffer.values {
        for i in 0..<positions.count {
            print("Collision candidates for vertex \(i):")
            for j in 0..<8 {
                let candidate = collisionCandidates[i * 8 + j]
                if candidate != UInt32.max {
                    print("  - Vertex \(candidate)")
                }
            }
        }
    }
} catch {
    print("An error occurred: \(error)")
}

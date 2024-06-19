# SimulationTools

## Description

SimulationTools is going to be a set of GPU algorithms designed for various simulation tasks.

## Algorithms

### Spatial Hashing

Spatial hashing is a technique used to divide space into a grid to quickly locate and manage vertexs or other entities. This method reduces the complexity of finding neighboring vertexs by organizing them into a structured grid.

**Overview:**

- **Hashing**: Each vertex or vertex is assigned to a cell based on its position in the space.
  
- **Bitonic Sorting**: Once the vertexs are hashed into cells, a bitonic sort is performed to order the hash-index pairs. Sorting helps resolve hash collisions and build cell buckets that can contain multiple vertexs.

- **Cell Bounds Identification**: After sorting, the start and end indices for each cell in the grid are identified. These indices describe the range of vertexs within each cell, allowing for efficient access and iteration over the vertexs in any given cell.

- **Collision Detection**:
  - **Vertex-Vertex**: For each vertex, potential collider vertexs are identified within the same or adjacent cells. Collision candidates are then processed to determine actual collisions.

## License

SimulationTools is licensed under [MIT license](LICENSE).

# SimulationTools

## Description

SimulationTools is going to be a set of GPU algorithms designed for various simulation tasks.

## Algorithms

### Spatial Hashing

Spatial hashing is a technique used to divide space into a grid to quickly locate and manage particles or other entities. This method reduces the complexity of finding neighboring particles by organizing them into a structured grid.

**Overview:**

- **Hashing**: Each particle or vertex is assigned to a cell based on its position in the space.
  
- **Bitonic Sorting**: Once the particles are hashed into cells, a bitonic sort is performed to order the hash-index pairs. Sorting helps resolve hash collisions and build cell buckets that can contain multiple particles.

- **Cell Bounds Identification**: After sorting, the start and end indices for each cell in the grid are identified. These indices describe the range of particles within each cell, allowing for efficient access and iteration over the particles in any given cell.

- **Collision Detection**:
  - **Vertex-Vertex**: For each particle, potential collider particles are identified within the same or adjacent cells. Collision candidates are then processed to determine actual collisions.
  - **Vertex-Triangle**: Similar to vertex-vertex, but involves checking each triangle against particles that fall within the triangle's bounding box.

- **Temporal Reuse**: In dynamic simulations, entities move, and their positions change over time. Using insertion sort helps maintain the sorted order of pairs (by distance), thus optimizing the collision detection process by improving temporal coherence and lookup performance.

- **Spatial Reuse**: As an optional step, we also can reuse collision pairs structurally, but looking up collision information of adjacent elements (currently only supports for Vertex-Triangle pairs). 

## License

SimulationTools is licensed under [MIT license](LICENSE).

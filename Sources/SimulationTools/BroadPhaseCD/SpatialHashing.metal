#include "../Common/SharedTypes.h"
#include "../Common/BroadPhaseCommon.h"

kernel void storeHalfPositions(
   constant float4 *positions [[ buffer(0) ]],
   device half4 *outPositions [[ buffer(1) ]],
   uint gid [[thread_position_in_grid]])
{
    outPositions[gid] = half4(positions[gid]);
}

kernel void computeParticleHash(
    device const half4* positions [[ buffer(0) ]],
    device uint2* hashTable [[ buffer(1) ]],
    constant uint& hashTableCapacity [[ buffer(2) ]],
    constant float& gridSpacing [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]])
{
    float3 position = float3(positions[id].xyz);
    uint hash = getHash(hashCoord(position, gridSpacing), hashTableCapacity);
    hashTable[id] = uint2(hash, id);
}

kernel void findCellStart(
    device uint* cellStart [[ buffer(0) ]],
    device uint* cellEnd [[ buffer(1) ]],
    device const uint2* hashTable [[ buffer(2) ]],
    constant uint& positionsCount [[ buffer(3) ]],
    uint id [[ thread_position_in_grid ]],
    uint threadIdx [[ thread_position_in_threadgroup ]],
    threadgroup uint* sharedHash [[ threadgroup(0) ]])
{
    uint2 hashIndex = hashTable[id];
    uint hash = hashIndex.x;
    sharedHash[threadIdx + 1] = hash;
    if (id > 0 && threadIdx == 0) {
        sharedHash[0] = hashTable[id - 1].x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (id == 0 || hash != sharedHash[threadIdx]) {
        cellStart[hash] = id;

        if (id > 0) {
            cellEnd[sharedHash[threadIdx]] = id;
        }
    }

    if (id == positionsCount - 1) {
        cellEnd[hash] = id + 1;
    }
}

kernel void cacheCollisions(
    device Indices8* collisionPairs [[ buffer(0) ]],
    constant uint2* hashTable [[ buffer(1) ]],
    constant uint* cellStart [[ buffer(2) ]],
    constant uint* cellEnd [[ buffer(3) ]],
    constant half4* positions [[ buffer(4) ]],
    constant Indices8* vertexNeighborhood [[buffer(5)]],
    constant uint& hashTableCapacity [[ buffer(6) ]],
    constant float& spacingScale [[ buffer(7) ]],
    constant float& gridSpacing [[ buffer(8) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uint id = hashTable[gid].y;
    if (id == UINT_MAX) { return; }

    float3 position = float3(positions[id].xyz);
    int3 hashPosition = hashCoord(position, gridSpacing);
    int ix = hashPosition.x;
    int iy = hashPosition.y;
    int iz = hashPosition.z;

    Indices8 vertexNeighbors = vertexNeighborhood[id];
    uint4 vertexNeighbors0 = uint4(vertexNeighbors.values[0], vertexNeighbors.values[1], vertexNeighbors.values[2], vertexNeighbors.values[3]);
    uint4 vertexNeighbors1 = uint4(vertexNeighbors.values[4], vertexNeighbors.values[5], vertexNeighbors.values[6], vertexNeighbors.values[7]);
    
    const float proximity = gridSpacing * spacingScale;
    
    CollisionCandidates collisionCandidates;
    initializeCollisionCandidates(collisionPairs, positions, id, collisionCandidates, position);

    for (int x = ix - 1; x <= ix + 1; x++) {
        for (int y = iy - 1; y <= iy + 1; y++) {
            for (int z = iz - 1; z <= iz + 1; z++) {
                uint h = getHash(int3(x, y, z), hashTableCapacity);
                uint start = cellStart[h];
                if (start == UINT_MAX) { continue; }
                uint end = min(cellEnd[h], start + INDICES_8_COUNT);
                
                for (uint i = start; i < end; i++) {
                    uint neighbor = hashTable[i].y;
                    if (neighbor == UINT_MAX) { break; }
                    if (any(vertexNeighbors0 == neighbor) || any(vertexNeighbors1 == neighbor) || neighbor == id) { continue; }
                    
                    float3 neighborPosition = float3(positions[neighbor].xyz);
                    float3 diff = position - neighborPosition;
                    float distanceSQ = length_squared(diff);
                    float errorSQ = distanceSQ - pow(proximity, 2.0);
                    if (errorSQ >= 0.0 || distanceSQ > collisionCandidates.candidates[7].distance) { continue; }

                    insertSeed(collisionCandidates, neighbor, distanceSQ);
                }
            }
        }
    }
    
    for (int i = 0; i <INDICES_8_COUNT; i++) {
        collisionPairs[id].values[i] = collisionCandidates.candidates[i].index;
    }
}

kernel void cacheTriangleCollisions(
    device Indices8* collisionPairs [[ buffer(0) ]],
    constant uint2* hashTable [[ buffer(1) ]],
    constant uint* cellStart [[ buffer(2) ]],
    constant uint* cellEnd [[ buffer(3) ]],
    constant half4* positions [[ buffer(4) ]],
    constant Indices8* vertexNeighborhood [[buffer(5)]],
    constant uint& hashTableCapacity [[ buffer(6) ]],
    constant float& spacingScale [[ buffer(7) ]],
    constant float& gridSpacing [[ buffer(8) ]],
    constant uint3 *triangles [[ buffer(9) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uint triangleIndex = gid;
    uint3 triangleVertices = triangles[triangleIndex];
    Triangle triangle = createTriangle(triangleVertices, positions);

    float3 minPos = float3(
                           min(min(triangle.a.x, triangle.b.x), triangle.c.x),
                           min(min(triangle.a.y, triangle.b.y), triangle.c.y),
                           min(min(triangle.a.z, triangle.b.z), triangle.c.z)
                           );
                      
    float3 maxPos = float3(
                           max(max(triangle.a.x, triangle.b.x), triangle.c.x),
                           max(max(triangle.a.y, triangle.b.y), triangle.c.y),
                           max(max(triangle.a.z, triangle.b.z), triangle.c.z)
                           );
    
    int3 minHashPos = hashCoord(minPos, gridSpacing);
    int3 maxHashPos = hashCoord(maxPos, gridSpacing);
    float proximity = gridSpacing * spacingScale;
    
    CollisionCandidates collisionCandidates;
    initializeCollisionCandidates(collisionPairs, positions, triangleIndex, collisionCandidates, triangle);


    for (int x = minHashPos.x; x <= maxHashPos.x; ++x) {
        for (int y = minHashPos.y; y <= maxHashPos.y; ++y) {
            for (int z = minHashPos.z; z <= maxHashPos.z; ++z) {
                int3 hashPos = int3(x, y, z);
                uint h = getHash(hashPos, hashTableCapacity);
                
                uint start = cellStart[h];
                if (start == UINT_MAX) { continue; }
                uint end = min(cellEnd[h], start + INDICES_8_COUNT);
                
                for (uint i = start; i < end; i++) {
                    uint colliderIndex = hashTable[i].y;
                    if (colliderIndex == UINT_MAX) { continue; }
                    
                    Indices8 vertexNeighbors = vertexNeighborhood[colliderIndex];
                    uint4 vertexNeighbors0 = uint4(vertexNeighbors.values[0], vertexNeighbors.values[1], vertexNeighbors.values[2], vertexNeighbors.values[3]);
                    uint4 vertexNeighbors1 = uint4(vertexNeighbors.values[4], vertexNeighbors.values[5], vertexNeighbors.values[6], vertexNeighbors.values[7]);
                    
                    if (any(vertexNeighbors0 == triangleVertices.x) || any(vertexNeighbors1 == triangleVertices.x)) { continue; }
                    if (any(vertexNeighbors0 == triangleVertices.y) || any(vertexNeighbors1 == triangleVertices.y)) { continue; }
                    if (any(vertexNeighbors0 == triangleVertices.z) || any(vertexNeighbors1 == triangleVertices.z)) { continue; }
                    if (any(triangleVertices == colliderIndex)) { continue; }

                    float3 position = float3(positions[colliderIndex].xyz);
                    float distanceSQ = usdTriangle(position, triangle.a.xyz, triangle.b.xyz, triangle.c.xyz);
                    float errorSQ = distanceSQ - pow(proximity, 2.0);
                    if (errorSQ >= 0.0 || distanceSQ > collisionCandidates.candidates[7].distance) { continue; }

                    insertSeed(collisionCandidates, colliderIndex, distanceSQ);
                }
            }
        }
    }
    
    for (int i = 0; i <INDICES_8_COUNT; i++) {
        collisionPairs[triangleIndex].values[i] = collisionCandidates.candidates[i].index;
    }
}

kernel void reuseTrinaglesCache(
    device Indices8* collisionPairs [[ buffer(0) ]],
    constant uint2* hashTable [[ buffer(1) ]],
    constant uint* cellStart [[ buffer(2) ]],
    constant uint* cellEnd [[ buffer(3) ]],
    constant float4* positions [[ buffer(4) ]],
    constant Indices8* vertexNeighborhood [[buffer(5)]],
    constant uint3 *triangles [[ buffer(6) ]],
    constant uint3* triangleNeighborhood [[buffer(7)]],
    constant uint& hashTableCapacity [[ buffer(8) ]],
    constant float& spacingScale [[ buffer(9) ]],
    constant float& gridSpacing [[ buffer(10) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uint triangleIndex = gid;
    uint3 triangleVertices = triangles[triangleIndex];
    Triangle triangle = createTriangle(triangleVertices, positions);

    float gridCellSpacing = gridSpacing;
    float proximity = gridCellSpacing * spacingScale;
    
    CollisionCandidates collisionCandidates;
    initializeCollisionCandidates(collisionPairs, positions, triangleIndex, collisionCandidates, triangle);

    uint3 triangleNeighbors = triangleNeighborhood[triangleIndex];
    for (int i = 0; i < 3; i++) {
        uint triangleNeighbor = triangleNeighbors[i];
        if (triangleNeighbor == UINT_MAX) { continue; }
        Indices8 neighborCollisionPairs = collisionPairs[triangleNeighbor];
        for (int j = 0; j < 2; j++) {
            uint vertexCollisionIndex = neighborCollisionPairs.values[j];
            if (vertexCollisionIndex == UINT_MAX) { continue; }
            Indices8 vertexNeighbors = vertexNeighborhood[vertexCollisionIndex];
        
            uint4 vertexNeighbors0 = uint4(vertexNeighbors.values[0], vertexNeighbors.values[1], vertexNeighbors.values[2], vertexNeighbors.values[3]);
            uint4 vertexNeighbors1 = uint4(vertexNeighbors.values[4], vertexNeighbors.values[5], vertexNeighbors.values[6], vertexNeighbors.values[7]);
            if (any(vertexNeighbors0 == triangleVertices.x) || any(vertexNeighbors1 == triangleVertices.x)) { continue; }
            if (any(vertexNeighbors0 == triangleVertices.y) || any(vertexNeighbors1 == triangleVertices.y)) { continue; }
            if (any(vertexNeighbors0 == triangleVertices.z) || any(vertexNeighbors1 == triangleVertices.z)) { continue; }
            if (any(triangleVertices == vertexCollisionIndex)) { continue; }
            
            float3 collider = positions[vertexCollisionIndex].xyz;
            float distanceSQ = usdTriangle(collider, triangle.a.xyz, triangle.b.xyz, triangle.c.xyz);
            float errorSQ = distanceSQ - pow(proximity, 2.0);
            if (errorSQ >= 0) { continue; }
            
            insertSeed(collisionCandidates, vertexCollisionIndex, distanceSQ);
        }
    }

    for (int i = 0; i <INDICES_8_COUNT; i++) {
        collisionPairs[triangleIndex].values[i] = collisionCandidates.candidates[i].index;
    }
}

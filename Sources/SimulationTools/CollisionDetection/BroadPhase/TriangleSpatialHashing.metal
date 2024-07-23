#include "../../Common/BroadPhaseCommon.h"
#include "../../Common/Definitions.h"

#define BUCKET_SIZE 8

kernel void hashTriangles(
    constant const packed_float3* positions [[ buffer(0) ]],
    device uint* hashTable [[ buffer(1) ]],
    device atomic_uint* hashTableCounter [[ buffer(2) ]],
    constant float& cellSize [[ buffer(3) ]],
    constant packed_uint3* triangles [[ buffer(4) ]],
    constant uint& trianglesCount [[ buffer(5) ]],
    constant uint& step [[ buffer(6) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= trianglesCount) { return; }
    uint gid = (step + id) % trianglesCount;
    uint3 triangle = triangles[gid];
    Triangle trianglePositions = createTriangle(triangle, positions);
    
    float3 minPos = min(min(trianglePositions.a, trianglePositions.b), trianglePositions.c);
    float3 maxPos = max(max(trianglePositions.a, trianglePositions.b), trianglePositions.c);
    
    int3 minCell = int3(floor(minPos / cellSize));
    int3 maxCell = int3(ceil(maxPos / cellSize));
    
    for (int x = minCell.x; x <= maxCell.x; x++) {
        for (int y = minCell.y; y <= maxCell.y; y++) {
            for (int z = minCell.z; z <= maxCell.z; z++) {
                uint hash = getHash(int3(x, y, z), trianglesCount);
                uint index = atomic_fetch_add_explicit(&hashTableCounter[hash], 1, memory_order_relaxed);
                if (index < BUCKET_SIZE) {
                    hashTable[hash * BUCKET_SIZE + index] = gid;
                }
            }
        }
    }
}

kernel void findTriangleCandidates(
    device uint* collisionCandidates [[ buffer(0) ]],
    constant float4* positions [[ buffer(1) ]],
    constant packed_float3* scenePositions [[ buffer(2) ]],
    constant uint* triangleHashTable [[ buffer(3) ]],
    constant packed_uint3* triangles [[ buffer(4) ]],
    constant uint& hashTableCapacity [[ buffer(5) ]],
    constant float& cellSize [[ buffer(6) ]],
    constant uint& maxCollisionCandidatesCount [[ buffer(7) ]],
    constant uint& gridSize [[ buffer(8) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= gridSize) { return; }
    float3 vertexPosition = positions[gid].xyz;
    int3 hashPosition = int3(floor(vertexPosition / cellSize));
        
    SortedCollisionCandidates sortedCollisionCandidates;
    initializeTriangleCollisionCandidates(
        collisionCandidates,
        scenePositions,
        triangles,
        gid,
        vertexPosition,
        sortedCollisionCandidates,
        maxCollisionCandidatesCount
    );

    uint h = getHash(hashPosition, hashTableCapacity);
    for (uint i = 0; i < BUCKET_SIZE; i++) {
        uint triangleIndex = triangleHashTable[h * BUCKET_SIZE + i];
        if (triangleIndex == UINT_MAX) { continue; }
        uint3 triangle = triangles[triangleIndex];
        
        Triangle trianglePositions = createTriangle(triangle, scenePositions);
        float distanceSQ = usdTriangle(vertexPosition, trianglePositions.a, trianglePositions.b, trianglePositions.c);
        if (distanceSQ > sortedCollisionCandidates.candidates[maxCollisionCandidatesCount - 1].distance) { continue; }
        
        insertSeed(sortedCollisionCandidates, triangleIndex, distanceSQ, maxCollisionCandidatesCount);
    }

    for (int i = 0; i < int(maxCollisionCandidatesCount); i++) {
        collisionCandidates[gid * maxCollisionCandidatesCount + i] = sortedCollisionCandidates.candidates[i].index;
    }
}

kernel void reuseTrianglesCache(
    device uint* vertexNeighbors [[ buffer(0) ]],
    device uint* sceneCollisionCandidates [[ buffer(1) ]],
    constant float4* positions [[ buffer(2) ]],
    constant packed_float3* scenePositions [[ buffer(3) ]],
    constant packed_uint3 *sceneTriangles [[ buffer(4) ]],
    constant uint& positionsCount [[ buffer(5) ]],
    constant uint& maxCollisionCandidatesCount [[ buffer(6) ]],
    constant uint& vertexNeighborsCount [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= positionsCount) { return; }
    float3 vertexPosition = positions[gid].xyz;

    SortedCollisionCandidates sortedCollisionCandidates;
    initializeTriangleCollisionCandidates(
        sceneCollisionCandidates,
        scenePositions,
        sceneTriangles,
        gid,
        vertexPosition,
        sortedCollisionCandidates,
        maxCollisionCandidatesCount
    );
    
    const int neighborsReuseCount = 4;
    for (int i = 0; i < min(neighborsReuseCount, int(vertexNeighborsCount)); i++) {
        uint neighborIndex = vertexNeighbors[gid * vertexNeighborsCount + i];
        for (int j = 0; j < 1; j++) {
            uint triangleIndex = sceneCollisionCandidates[neighborIndex * maxCollisionCandidatesCount + j];
            uint3 triangle = sceneTriangles[triangleIndex];
            Triangle trianglePositions = createTriangle(triangle, scenePositions);
            float distanceSQ = usdTriangle(vertexPosition, trianglePositions.a, trianglePositions.b, trianglePositions.c);
            if (distanceSQ > sortedCollisionCandidates.candidates[maxCollisionCandidatesCount - 1].distance) { continue; }
            
            insertSeed(sortedCollisionCandidates, triangleIndex, distanceSQ, maxCollisionCandidatesCount);
        }
    }
    
    for (int i = 0; i < int(maxCollisionCandidatesCount); i++) {
        sceneCollisionCandidates[gid * maxCollisionCandidatesCount + i] = sortedCollisionCandidates.candidates[i].index;
    }
}

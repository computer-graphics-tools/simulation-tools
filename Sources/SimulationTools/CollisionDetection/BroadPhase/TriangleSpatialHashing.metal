#include "../../Common/BroadPhaseCommon.h"
#include "../../Common/Definitions.h"
#include "../../../SimulationToolsSharedTypes/SimulationToolsSharedTypes.h"

kernel void hashTriangles(
    constant const void* positions [[ buffer(0) ]],
    device uint* hashTable [[ buffer(1) ]],
    device atomic_uint* hashTableCounter [[ buffer(2) ]],
    constant float& cellSize [[ buffer(3) ]],
    constant void* triangles [[ buffer(4) ]],
    constant uint& trianglesCount [[ buffer(5) ]],
    constant uint& bucketSize [[ buffer(6) ]],
    constant uint& step [[ buffer(7) ]],
    constant bool& usePackedColliderPositions [[ buffer(8) ]],
    constant bool& usePackedIndices [[ buffer(9) ]],
    uint id [[ thread_position_in_grid ]]
) {
    if (id >= trianglesCount) { return; }
    
    GetTriangleFunc getTriangle = usePackedIndices ? getPackedIndex : getIndex;
    GetPositionFunc getColliderPosition = usePackedColliderPositions ? getPackedPosition : getPosition;
    
    uint gid = (step + id) % trianglesCount;
    
    uint3 triangle = getTriangle(gid, triangles);
    Triangle trianglePositions = createTriangle(triangle, getColliderPosition, positions);
    
    float3 minPos = min(min(trianglePositions.a, trianglePositions.b), trianglePositions.c);
    float3 maxPos = max(max(trianglePositions.a, trianglePositions.b), trianglePositions.c);
    
    int3 minCell = int3(floor(minPos / cellSize));
    int3 maxCell = int3(ceil(maxPos / cellSize));
    
    for (int x = minCell.x; x <= maxCell.x; x++) {
        for (int y = minCell.y; y <= maxCell.y; y++) {
            for (int z = minCell.z; z <= maxCell.z; z++) {
                uint hash = getHash(int3(x, y, z), trianglesCount);
                uint index = atomic_fetch_add_explicit(&hashTableCounter[hash], 1, memory_order_relaxed);
                if (index < bucketSize) {
                    hashTable[hash * bucketSize + index] = gid;
                }
            }
        }
    }
}

kernel void findTriangleCandidates(
    device uint* collisionCandidates [[ buffer(0) ]],
    constant void* positions [[ buffer(1) ]],
    constant void* colliderPositions [[ buffer(2) ]],
    constant void* triangles [[ buffer(3) ]],
    constant uint* triangleHashTable [[ buffer(4) ]],
    constant uint* connectedVertices [[buffer(5)]],
    constant TriangleSHParameters& params [[ buffer(6) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.gridSize) { return; }
    
    GetTriangleFunc getTriangle = params.usePackedIndices ? getPackedIndex : getIndex;
    GetPositionFunc getCollidablePosition = params.usePackedCollidablePositions ? getPackedPosition : getPosition;
    GetPositionFunc getColliderPosition = params.usePackedColliderPositions ? getPackedPosition : getPosition;
    
    const bool handlingSelfCollision = !params.useExternalCollidable;
    const float3 vertexPosition = getCollidablePosition(gid, positions);
    const int3 hashPosition = int3(floor(vertexPosition / params.cellSize));
    
    SortedCollisionCandidates sortedCollisionCandidates;
    initializeTriangleCollisionCandidates(
        collisionCandidates,
        getColliderPosition,
        colliderPositions,
        getTriangle,
        triangles,
        gid,
        vertexPosition,
        sortedCollisionCandidates,
        params.maxCollisionCandidatesCount
    );

    uint hash = getHash(hashPosition, params.hashTableCapacity);
    for (uint i = 0; i < params.bucketSize; i++) {
        uint triangleIndex = triangleHashTable[hash * params.bucketSize + i];
        if (triangleIndex == UINT_MAX) { continue; }

        const uint3 triangle = getTriangle(triangleIndex, triangles);
        if (handlingSelfCollision && any(triangle == gid)) { continue; }
        
        bool isConnected = false;
        for (uint j = 0; j < params.connectedVerticesCount; j++) {
            isConnected = any(triangle == connectedVertices[gid * params.connectedVerticesCount + j]);
            if (isConnected) { break; }
        }
        if (isConnected) { continue; }

        const Triangle trianglePositions = createTriangle(triangle, getColliderPosition, colliderPositions);
        const float distanceSQ = usdTriangle(vertexPosition, trianglePositions.a, trianglePositions.b, trianglePositions.c);
        if (distanceSQ > sortedCollisionCandidates.candidates[params.maxCollisionCandidatesCount - 1].distance) { continue; }
        
        insertSeed(sortedCollisionCandidates, triangleIndex, distanceSQ, params.maxCollisionCandidatesCount);
    }

    for (int i = 0; i < int(params.maxCollisionCandidatesCount); i++) {
        collisionCandidates[gid * params.maxCollisionCandidatesCount + i] = sortedCollisionCandidates.candidates[i].index;
    }
}

kernel void reuseTrianglesCache(
    device uint* collisionCandidates [[ buffer(0) ]],
    constant void* positions [[ buffer(1) ]],
    constant void* colliderPositions [[ buffer(2) ]],
    constant void* triangles [[ buffer(3) ]],
    constant uint* vertexNeighbors [[ buffer(4) ]],
    constant uint* connectedVertices [[buffer(5)]],
    constant uint3* triangleNeighbors [[ buffer(6) ]],
    constant uint& vertexNeighborsCount [[ buffer(7) ]],
    constant TriangleSHParameters& params [[ buffer(8) ]],
    constant bool& enableTriangleReuse [[ buffer(9) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (gid >= params.gridSize) { return; }
    
    GetTriangleFunc getTriangle = params.usePackedIndices ? getPackedIndex : getIndex;
    GetPositionFunc getCollidablePosition = params.usePackedCollidablePositions ? getPackedPosition : getPosition;
    GetPositionFunc getColliderPosition = params.usePackedColliderPositions ? getPackedPosition : getPosition;

    const bool handlingSelfCollision = !params.useExternalCollidable;
    const float3 vertexPosition = getCollidablePosition(gid, positions);
    
    SortedCollisionCandidates sortedCollisionCandidates;
    initializeTriangleCollisionCandidates(
        collisionCandidates,
        getColliderPosition,
        colliderPositions,
        getTriangle,
        triangles,
        gid,
        vertexPosition,
        sortedCollisionCandidates,
        params.maxCollisionCandidatesCount
    );
    
    const int neighborsReuseCount = 4;
    for (int i = 0; i < min(neighborsReuseCount, int(vertexNeighborsCount)); i++) {
        uint neighborIndex = vertexNeighbors[gid * vertexNeighborsCount + i];
        if (neighborIndex == UINT_MAX) { continue; }
        for (int j = 0; j < 1; j++) {
            uint triangleIndex = collisionCandidates[neighborIndex * params.maxCollisionCandidatesCount + j];
            if (triangleIndex == UINT_MAX) { continue; }
    
            uint3 triangle = getTriangle(triangleIndex, triangles);
            if (handlingSelfCollision && any(triangle == gid)) { continue; }

            bool isConnected = false;
            for (uint j = 0; j < params.connectedVerticesCount; j++) {
                isConnected = any(triangle == connectedVertices[gid * params.connectedVerticesCount + j]);
                if (isConnected) { break; }
            }
            if (isConnected) { continue; }

            Triangle trianglePositions = createTriangle(triangle, getColliderPosition, colliderPositions);
            float distanceSQ = usdTriangle(vertexPosition, trianglePositions.a, trianglePositions.b, trianglePositions.c);
            if (distanceSQ > sortedCollisionCandidates.candidates[params.maxCollisionCandidatesCount - 1].distance) { continue; }
            
            insertSeed(sortedCollisionCandidates, triangleIndex, distanceSQ, params.maxCollisionCandidatesCount);
        }
    }
    
    if (enableTriangleReuse) {
        for (int j = 0; j < 1; j++) {
            uint closestIndex = collisionCandidates[gid * params.maxCollisionCandidatesCount + j];
            if (closestIndex == UINT_MAX) { continue; }
            for (int i = 0; i < 3; i++) {
                uint triangleIndex = triangleNeighbors[closestIndex][i];
                if (triangleIndex == UINT_MAX) { continue; }
                uint3 triangle = getTriangle(triangleIndex, triangles);
                Triangle trianglePositions = createTriangle(triangle, getColliderPosition, colliderPositions);
                float distanceSQ = usdTriangle(vertexPosition, trianglePositions.a, trianglePositions.b, trianglePositions.c);
                if (distanceSQ > sortedCollisionCandidates.candidates[params.maxCollisionCandidatesCount - 1].distance) { continue; }
                
                insertSeed(sortedCollisionCandidates, triangleIndex, distanceSQ, params.maxCollisionCandidatesCount);
            }
        }
    }
    
    for (int i = 0; i < int(params.maxCollisionCandidatesCount); i++) {
        collisionCandidates[gid * params.maxCollisionCandidatesCount + i] = sortedCollisionCandidates.candidates[i].index;
    }
}

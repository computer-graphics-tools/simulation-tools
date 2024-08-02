#include "../../Common/BroadPhaseCommon.h"
#include "../../Common/Definitions.h"
#include "../../Common/DistanceFunctions.h"

kernel void computeHashAndIndexState(
    device const half4* positions [[ buffer(0) ]],
    device uint2* hashTable [[ buffer(1) ]],
    constant uint& hashTableCapacity [[ buffer(2) ]],
    constant float& cellSize [[ buffer(3) ]],
    constant uint& gridSize [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    float3 position = float3(positions[gid].xyz);
    uint hash = getHash(hashCoord(position, cellSize), hashTableCapacity);
    hashTable[gid] = uint2(hash, gid);
}

kernel void computeCellBoundaries(
    device uint* cellStart [[ buffer(0) ]],
    device uint* cellEnd [[ buffer(1) ]],
    device const uint2* hashTable [[ buffer(2) ]],
    constant uint& hashTableSize [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]],
    uint threadIdx [[ thread_position_in_threadgroup ]],
    threadgroup uint* sharedHash [[ threadgroup(0) ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= hashTableSize) { return; }
    uint2 hashIndex = hashTable[gid];
    uint hash = hashIndex.x;
    sharedHash[threadIdx + 1] = hash;
    if (gid > 0 && threadIdx == 0) {
        sharedHash[0] = hashTable[gid - 1].x;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (gid == 0 || hash != sharedHash[threadIdx]) {
        cellStart[hash] = gid;
        if (gid > 0) {
            cellEnd[sharedHash[threadIdx]] = gid;
        }
    }

    if (gid == hashTableSize - 1) {
        cellEnd[hash] = gid + 1;
    }
}

template <typename T>
METAL_FUNC void convertToHalfPrecisionGeneric(
    constant T* positions,
    device half3* halfPositions,
    uint gid
) {
    float3 position = getPosition(positions[gid]);
    halfPositions[gid] = half3(position);
}

kernel void convertToHalfPrecisionPacked(
    constant packed_float3* positions [[ buffer(0) ]],
    device half3* halfPositions [[ buffer(1) ]],
    constant uint& gridSize [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    convertToHalfPrecisionGeneric(positions, halfPositions, gid);
}

kernel void convertToHalfPrecisionUnpacked(
    constant float3* positions [[ buffer(0) ]],
    device half3* halfPositions [[ buffer(1) ]],
    constant uint& gridSize [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    convertToHalfPrecisionGeneric(positions, halfPositions, gid);
}

kernel void reorderHalfPrecision(
    device const half3* halfPositions [[ buffer(0) ]],
    device half3* sortedHalfPositions [[ buffer(1) ]],
    device const uint2* hashTable [[ buffer(2) ]],
    constant uint& gridSize [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    uint2 hashAndIndex = hashTable[gid];
    sortedHalfPositions[gid] = halfPositions[hashAndIndex.y];
}

template <typename T>
METAL_FUNC void findCollisionCandidates(
    device uint* collisionCandidates,
    constant uint2 *hashTable,
    constant uint* cellStart,
    constant uint* cellEnd,
    constant half3* colliderPositions,
    thread T& collidablePosition,
    constant uint* connectedVertices,
    constant uint& hashTableCapacity,
    constant float& radius,
    constant float& cellSize,
    constant uint& maxCollisionCandidatesCount,
    constant uint& connectedVerticesCount,
    bool handlingSelfCollision,
    uint index
) {
    float3 position = getPosition(collidablePosition);
    int3 hashPosition = hashCoord(position, cellSize);
    
    uint4 simdConnectedVertices[MAX_CONNECTED_VERTICES / 4];
    uint simdConnectedVerticesCount = connectedVerticesCount / 4;

    for (uint i = 0; i < simdConnectedVerticesCount; i++) {
        uint baseIndex = index * connectedVerticesCount + i * 4;
        
        simdConnectedVertices[i] = uint4(
            connectedVertices[baseIndex],
            connectedVertices[baseIndex + 1],
            connectedVertices[baseIndex + 2],
            connectedVertices[baseIndex + 3]
        );
    }

    SortedCollisionCandidates sortedCollisionCandidates;
    initializeCollisionCandidates(
        collisionCandidates,
        colliderPositions,
        sortedCollisionCandidates,
        index,
        half3(position),
        maxCollisionCandidatesCount
      );
    
    const float squaredDiameter = pow(radius * 2, 2);
    for (int x = hashPosition.x - 1; x <= hashPosition.x + 1; x++) {
        for (int y = hashPosition.y - 1; y <= hashPosition.y + 1; y++) {
            for (int z = hashPosition.z - 1; z <= hashPosition.z + 1; z++) {
                const float3 cellCenter = float3(x, y, z) * cellSize + cellSize * 0.5;
                if (sdsBox(cellCenter - float3(position), float3(cellSize * 0.5)) > squaredDiameter) {
                    continue;
                }

                const uint hash = getHash(int3(x, y, z), hashTableCapacity);
                const uint start = cellStart[hash];
                if (start == UINT_MAX) { continue; }
                const uint end = min(cellEnd[hash], start + MAX_COLLISION_CANDIDATES);
                
                for (uint i = start; i < end; i++) {
                    uint collisionCandidate = hashTable[i].y;
                    if (collisionCandidate == UINT_MAX) { break; }
                    if (handlingSelfCollision && collisionCandidate == index) { continue; }

                    bool isConnected = false;
                    for (uint j = 0; j < simdConnectedVerticesCount; j++) {
                        isConnected = any(simdConnectedVertices[j] == collisionCandidate);
                        if (isConnected) { break; }
                    }
                    if (isConnected) { continue; }
                    
                    const half3 candidatePosition = colliderPositions[i].xyz;
                    float distanceSq = length_squared(position - float3(candidatePosition));
                    if (distanceSq > sortedCollisionCandidates.candidates[maxCollisionCandidatesCount - 1].distance) { continue; }
                    if (distanceSq > squaredDiameter) { continue; }
                    
                    insertSeed(sortedCollisionCandidates, collisionCandidate, distanceSq, maxCollisionCandidatesCount);
                }
            }
        }
    }

    for (int i = 0; i < int(maxCollisionCandidatesCount); i++) {
        collisionCandidates[index * maxCollisionCandidatesCount + i] = sortedCollisionCandidates.candidates[i].index;
    }
}

kernel void findCollisionCandidatesFloat3(
    device uint* collisionCandidates [[ buffer(0) ]],
    constant uint2* hashTable [[ buffer(1) ]],
    constant uint* cellStart [[ buffer(2) ]],
    constant uint* cellEnd [[ buffer(3) ]],
    constant half3* colliderPositions [[ buffer(4) ]],
    constant float3* collidablePositions [[ buffer(5) ]],
    constant uint* connectedVertices [[buffer(6)]],
    constant uint& hashTableCapacity [[ buffer(7) ]],
    constant float& radius [[ buffer(8) ]],
    constant float& cellSize [[ buffer(9) ]],
    constant uint& maxCollisionCandidatesCount [[ buffer(10) ]],
    constant uint& connectedVerticesCount [[ buffer(11) ]],
    constant uint& collidableCount [[ buffer(12) ]],
    constant uint& gridSize [[ buffer(13) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    if (collidableCount == 0) {
        uint index = hashTable[gid].y;
        if (index == UINT_MAX) { return; }
           
        findCollisionCandidates(
           collisionCandidates,
           hashTable,
           cellStart,
           cellEnd,
           colliderPositions,
           colliderPositions[gid],
           connectedVertices,
           hashTableCapacity,
           radius,
           cellSize,
           maxCollisionCandidatesCount,
           connectedVerticesCount,
           true,
           index
        );
    } else {
        findCollisionCandidates(
           collisionCandidates,
           hashTable,
           cellStart,
           cellEnd,
           colliderPositions,
           collidablePositions[gid],
           connectedVertices,
           hashTableCapacity,
           radius,
           cellSize,
           maxCollisionCandidatesCount,
           connectedVerticesCount,
           false,
           gid
        );
    }
}

kernel void findCollisionCandidatesPackedFloat3(
    device uint* collisionCandidates [[ buffer(0) ]],
    constant uint2* hashTable [[ buffer(1) ]],
    constant uint* cellStart [[ buffer(2) ]],
    constant uint* cellEnd [[ buffer(3) ]],
    constant half3* colliderPositions [[ buffer(4) ]],
    constant packed_float3* collidablePositions [[ buffer(5) ]],
    constant uint* connectedVertices [[buffer(6)]],
    constant uint& hashTableCapacity [[ buffer(7) ]],
    constant float& radius [[ buffer(8) ]],
    constant float& cellSize [[ buffer(9) ]],
    constant uint& maxCollisionCandidatesCount [[ buffer(10) ]],
    constant uint& connectedVerticesCount [[ buffer(11) ]],
    constant uint& collidableCount [[ buffer(12) ]],
    constant uint& gridSize [[ buffer(13) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    if (collidableCount == 0) {
        uint index = hashTable[gid].y;
        if (index == UINT_MAX) { return; }
           
        findCollisionCandidates(
           collisionCandidates,
           hashTable,
           cellStart,
           cellEnd,
           colliderPositions,
           colliderPositions[gid],
           connectedVertices,
           hashTableCapacity,
           radius,
           cellSize,
           maxCollisionCandidatesCount,
           connectedVerticesCount,
           true,
           index
        );
    } else {
        findCollisionCandidates(
           collisionCandidates,
           hashTable,
           cellStart,
           cellEnd,
           colliderPositions,
           collidablePositions[gid],
           connectedVertices,
           hashTableCapacity,
           radius,
           cellSize,
           maxCollisionCandidatesCount,
           connectedVerticesCount,
           false,
           gid
        );
    }
}
    

kernel void reuseCollisionCandidates(
    device uint* collisionCandidates [[ buffer(0) ]],
    constant float3* positions [[ buffer(1) ]],
    constant uint* connectedVertices [[buffer(2)]],
    constant float& spacingScale [[ buffer(3) ]],
    constant float& cellSize [[ buffer(4) ]],
    constant uint& maxCollisionCandidatesCount [[ buffer(5) ]],
    constant uint& connectedVerticesCount [[ buffer(6) ]],
    constant uint& gridSize [[ buffer(7) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    float3 position = float3(positions[gid].xyz);
    const uint index = gid;
    const float proximity = cellSize * spacingScale;
    const float proximitySq = pow(proximity, 2.0);
    
    SortedCollisionCandidates sortedCollisionCandidates;
    initializeCollisionCandidates(
        collisionCandidates,
        positions,
        sortedCollisionCandidates,
        index,
        position,
        maxCollisionCandidatesCount
    );
    
    
    for (uint i = 0; i < min(maxCollisionCandidatesCount, 4u); i++) {
        uint candidateIndex = sortedCollisionCandidates.candidates[i].index;
        uint candidateStart = candidateIndex * maxCollisionCandidatesCount;
        uint candidateEnd = candidateStart + min(maxCollisionCandidatesCount, 4u);
        
        for (uint j = candidateStart; j < candidateEnd; j++) {
            uint potentialCandidate = collisionCandidates[j];
            if (potentialCandidate == UINT_MAX || potentialCandidate == index) continue;
            
            float3 candidatePosition = float3(positions[potentialCandidate].xyz);
            float3 diff = position - candidatePosition;
            float distanceSq = length_squared(diff);
            float errorSq = distanceSq - proximitySq;
            if (errorSq >= 0.0) continue;
            
            insertSeed(sortedCollisionCandidates, potentialCandidate, distanceSq, maxCollisionCandidatesCount);
        }
    }
    
    for (int i = 0; i < int(maxCollisionCandidatesCount); i++) {
        collisionCandidates[index * maxCollisionCandidatesCount + i] = sortedCollisionCandidates.candidates[i].index;
    }
}


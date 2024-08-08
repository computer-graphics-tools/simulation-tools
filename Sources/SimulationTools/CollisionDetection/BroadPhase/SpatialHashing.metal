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

kernel void convertToHalf(
    constant void* positions [[ buffer(0) ]],
    device half3* halfPositions [[ buffer(1) ]],
    constant uint& gridSize [[ buffer(2) ]],
    constant bool& usePackedPositions [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    GetPositionFunc getCollidablePosition = usePackedPositions ? getPackedPosition : getPosition;

    float3 position = getCollidablePosition(gid, positions);
    halfPositions[gid] = half3(position);
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

kernel void findCollisionCandidates(
    device uint* collisionCandidates [[ buffer(0) ]],
    constant uint2* hashTable [[ buffer(1) ]],
    constant uint* cellStart [[ buffer(2) ]],
    constant uint* cellEnd [[ buffer(3) ]],
    constant half3* colliderPositions [[ buffer(4) ]],
    constant void* collidablePositions [[ buffer(5) ]],
    constant uint* connectedVertices [[buffer(6)]],
    constant uint& hashTableCapacity [[ buffer(7) ]],
    constant float& radius [[ buffer(8) ]],
    constant float& cellSize [[ buffer(9) ]],
    constant uint& collisionCandidatesCount [[ buffer(10) ]],
    constant uint& connectedVerticesCount [[ buffer(11) ]],
    constant uint& collidableCount [[ buffer(12) ]],
    constant uint& gridSize [[ buffer(13) ]],
    constant bool& usePackedPositions [[ buffer(14) ]],
    uint gid [[ thread_position_in_grid ]]
) {
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    const bool handlingSelfCollision = collidableCount == 0;
    const uint index = handlingSelfCollision ? hashTable[gid].y : gid;
    if (index == UINT_MAX) { return; }
    GetPositionFunc getCollidablePosition = usePackedPositions ? getPackedPosition : getPosition;
    
    const float3 position = handlingSelfCollision ? float3(colliderPositions[gid].xyz) : getCollidablePosition(gid, collidablePositions);
    const int3 hashPosition = hashCoord(position, cellSize);

    SortedCollisionCandidates sortedCollisionCandidates;
    initializeCollisionCandidates(
        collisionCandidates,
        colliderPositions,
        sortedCollisionCandidates,
        index,
        half3(position),
        collisionCandidatesCount
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
                    for (uint j = 0; j < connectedVerticesCount; j++) {
                        isConnected = connectedVertices[index * connectedVerticesCount + j] == collisionCandidate;
                        if (isConnected) { break; }
                    }
                    if (isConnected) { continue; }
                    
                    const half3 candidatePosition = colliderPositions[i].xyz;
                    float distanceSq = length_squared(position - float3(candidatePosition));
                    if (distanceSq > sortedCollisionCandidates.candidates[collisionCandidatesCount - 1].distance) { continue; }
                    if (distanceSq > squaredDiameter) { continue; }
                    
                    insertSeed(sortedCollisionCandidates, collisionCandidate, distanceSq, collisionCandidatesCount);
                }
            }
        }
    }

    for (int i = 0; i < int(collisionCandidatesCount); i++) {
        collisionCandidates[index * collisionCandidatesCount + i] = sortedCollisionCandidates.candidates[i].index;
    }
}

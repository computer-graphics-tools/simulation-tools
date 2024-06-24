#include "../../Common/BroadPhaseCommon.h"
#include "../../Common/Definitions.h"


kernel void convertToHalfPrecisionPositions(
   constant float4 *positions [[ buffer(0) ]],
   device half4 *outPositions [[ buffer(1) ]],
   constant uint& gridSize [[ buffer(2) ]],
   uint gid [[thread_position_in_grid]]
) {
    if (!deviceSupportsNonuniformThreadgroups && gid >= gridSize) { return; }
    outPositions[gid] = half4(positions[gid]);
}

kernel void reorderHalfPrecisionPositions(
   constant half4 *positions [[ buffer(0) ]],
   device half4 *outPositions [[ buffer(1) ]],
   constant uint2* hashTable [[ buffer(2) ]],
   constant uint& gridSize [[ buffer(3) ]],
   uint gid [[thread_position_in_grid]])
{
    if (deviceDoesntSupportNonuniformThreadgroups && gid >= gridSize) { return; }
    uint2 hashAndIndex = hashTable[gid];
    half3 position = half3(positions[hashAndIndex.y].xyz);
    outPositions[gid] = half4(position, 1.0);
}

kernel void computeVertexHashAndIndex(
    device const half4* positions [[ buffer(0) ]],
    device uint2* hashTable [[ buffer(1) ]],
    constant uint& hashTableCapacity [[ buffer(2) ]],
    constant float& cellSize [[ buffer(3) ]],
    constant uint& gridSize [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (!deviceSupportsNonuniformThreadgroups && gid >= gridSize) { return; }
    float3 position = float3(positions[gid].xyz);
    uint hash = getHash(hashCoord(position, cellSize), hashTableCapacity);
    hashTable[gid] = uint2(hash, gid);
}

kernel void computeCellBoundaries(
    device uint* cellStart [[ buffer(0) ]],
    device uint* cellEnd [[ buffer(1) ]],
    device const uint2* hashTable [[ buffer(2) ]],
    constant uint& gridSize [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]],
    uint threadIdx [[ thread_position_in_threadgroup ]],
    threadgroup uint* sharedHash [[ threadgroup(0) ]])
{
    if (!deviceSupportsNonuniformThreadgroups && gid >= gridSize) { return; }
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

    if (gid == gridSize - 1) {
        cellEnd[hash] = gid + 1;
    }
}

kernel void findCollisionCandidates(
    device uint* collisionCandidates [[ buffer(0) ]],
    constant uint2* hashTable [[ buffer(1) ]],
    constant uint* cellStart [[ buffer(2) ]],
    constant uint* cellEnd [[ buffer(3) ]],
    constant half4* sortedPositions [[ buffer(4) ]],
    constant uint* connectedVertices [[buffer(5)]],
    constant uint& hashTableCapacity [[ buffer(6) ]],
    constant float& spacingScale [[ buffer(7) ]],
    constant float& cellSize [[ buffer(8) ]],
    constant uint& maxCollisionCandidatesCount [[ buffer(9) ]],
    constant uint& connectedVerticesCount [[ buffer(10) ]],
    constant uint& gridSize [[ buffer(11) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (!deviceSupportsNonuniformThreadgroups && gid >= gridSize) { return; }
    uint index = hashTable[gid].y;
    if (index == UINT_MAX) { return; }

    float3 position = float3(sortedPositions[gid].xyz);
    int3 hashPosition = hashCoord(position, cellSize);
    int ix = hashPosition.x;
    int iy = hashPosition.y;
    int iz = hashPosition.z;
    
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
        
    const float proximity = cellSize * spacingScale;
    
    uint candidatesCount = 0;
    
    for (int x = ix - 1; x <= ix + 1; x++) {
        for (int y = iy - 1; y <= iy + 1; y++) {
            for (int z = iz - 1; z <= iz + 1; z++) {
                uint hash = getHash(int3(x, y, z), hashTableCapacity);
                uint start = cellStart[hash];
                if (start == UINT_MAX) { continue; }
                uint end = min(cellEnd[hash], start + maxCollisionCandidatesCount);
                
                for (uint i = start; i < end; i++) {
                    uint collisionCandidate = hashTable[i].y;
                    if (collisionCandidate == UINT_MAX) { break; }
                    if (collisionCandidate == index) { continue; }

                    bool isConnected = false;
                    for (uint j = 0; j < simdConnectedVerticesCount; j++) {
                        isConnected = any(simdConnectedVertices[j] == collisionCandidate);
                        if (isConnected) { break; }
                    }
                    if (isConnected) { continue; }

                    float3 candidatePosition = float3(sortedPositions[i].xyz);
                    float3 diff = position - candidatePosition;
                    float distanceSq = length_squared(diff);
                    float errorSq = distanceSq - pow(proximity, 2.0);
                    if (errorSq >= 0.0) { continue; }

                    collisionCandidates[index * maxCollisionCandidatesCount + candidatesCount] = collisionCandidate;
                    candidatesCount += 1;
                    if (candidatesCount >= maxCollisionCandidatesCount) { return; }
                }
            }
        }
    }
    
    if (candidatesCount < maxCollisionCandidatesCount) {
        collisionCandidates[index * maxCollisionCandidatesCount + candidatesCount] = UINT_MAX;
    }
}

#ifndef BroadPhaseCommon_h
#define BroadPhaseCommon_h

#include <metal_stdlib>
using namespace metal;

#define MAX_CONNECTED_VERTICES 32

METAL_FUNC int3 hashCoord(float3 position, float gridSpacing) {
    int x = floor(position.x / gridSpacing);
    int y = floor(position.y / gridSpacing);
    int z = floor(position.z / gridSpacing);
        
    return int3(x, y, z);
}

METAL_FUNC int computeHash(int3 position) {
    int x = position.x;
    int y = position.y;
    int z = position.z;
    
    int hash = (x * 92837111) ^ (y * 689287499) ^ (z * 283923481);
    
    return hash;
}

METAL_FUNC uint getHash(int3 position, uint hashTableCapacity) {
    int hash = computeHash(position);
    return uint(abs(hash % hashTableCapacity));
}

struct CollisionCandidate {
    uint index;
    float distance;
};

struct SortedCollisionCandidates {
    CollisionCandidate candidates[MAX_CONNECTED_VERTICES];
};

METAL_FUNC void initializeCollisionCandidates(
    device uint* candidates,
    constant const half4* positions,
    thread SortedCollisionCandidates &sortedCandidates,
    uint index,
    float3 position,
    uint count
) {
    for (int i = 0; i < int(count); i++) {
        uint colliderIndex = candidates[index * count + i];
        sortedCandidates.candidates[i].index = colliderIndex;
        if (colliderIndex != UINT_MAX) {
            float3 collider = float3(positions[colliderIndex].xyz);
            sortedCandidates.candidates[i].distance = length_squared(position - collider);
        } else {
            sortedCandidates.candidates[i].distance = FLT_MAX;
        }
    }
}

METAL_FUNC void insertSeed(
    thread SortedCollisionCandidates &candidates,
    uint index,
    float distance,
    uint count
) {
    int insertPosition = -1;
    int duplicateIndex = -1;

    for (int i = 0; i < int(count); i++) {
        if (distance <= candidates.candidates[i].distance && insertPosition == -1) {
            insertPosition = i;
        }
        
        if (index == candidates.candidates[i].index) {
            duplicateIndex = i;
            break;
        }
    }

    if (insertPosition != -1) {
        int start = duplicateIndex == -1 ? count - 1 : duplicateIndex;
        for (int j = start; j > insertPosition; j--) {
            candidates.candidates[j] = candidates.candidates[j - 1];
        }
    
        candidates.candidates[insertPosition] = { .index = index, .distance = distance };
    }
}

#endif /* BroadPhaseCommon_h */

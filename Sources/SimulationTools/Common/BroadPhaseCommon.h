#ifndef BroadPhaseCommon_h
#define BroadPhaseCommon_h

#include "SharedTypes.h"

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

struct CollisionCandidates {
    CollisionCandidate candidates[INDICES_8_COUNT];
};

METAL_FUNC void initializeCollisionCandidates(
    device Indices8* collisionPairs,
    constant const half4* positions,
    uint index,
    thread CollisionCandidates &collisionCandidates,
    float3 position)
{
    for (int i = 0; i < INDICES_8_COUNT; i++) {
        uint colliderIndex = collisionPairs[index].values[i];
        collisionCandidates.candidates[i].index = colliderIndex;
        if (colliderIndex != UINT_MAX) {
            float3 collider = float3(positions[colliderIndex].xyz);
            collisionCandidates.candidates[i].distance = length_squared(position - collider);
        } else {
            collisionCandidates.candidates[i].distance = FLT_MAX;
        }
    }
}

template <typename T>
enable_if_t<is_floating_point_v<T>, void>
METAL_FUNC initializeCollisionCandidates(
    device Indices8* collisionPairs,
    constant const vec<T, 4>* positions,
    uint index,
    thread CollisionCandidates &collisionCandidates,
    Triangle triangle)
{
    for (int i = 0; i < INDICES_8_COUNT; i++) {
        uint colliderIndex = collisionPairs[index].values[i];
        collisionCandidates.candidates[i].index = colliderIndex;
        if (index != UINT_MAX) {
            float3 collider = float3(positions[colliderIndex].xyz);
            collisionCandidates.candidates[i].distance = usdTriangle(collider, triangle.a.xyz, triangle.b.xyz, triangle.c.xyz);
        } else {
            collisionCandidates.candidates[i].distance = FLT_MAX;
        }
    }
}

METAL_FUNC void insertSeed(thread CollisionCandidates &candidates, uint index, float distance) {
    int insertPosition = -1;
    int duplicateIndex = -1;

    for (int i = 0; i < INDICES_8_COUNT; i++) {
        if (distance <= candidates.candidates[i].distance && insertPosition == -1) {
            insertPosition = i;
        }
        
        if (index == candidates.candidates[i].index) {
            duplicateIndex = i;
            break;
        }
    }

    if (insertPosition != -1) {
        int start = duplicateIndex == -1 ? INDICES_8_COUNT - 1 : duplicateIndex;
        for (int j = start; j > insertPosition; j--) {
            candidates.candidates[j] = candidates.candidates[j - 1];
        }
    
        candidates.candidates[insertPosition] = { .index = index, .distance = distance };
    }
}

#endif /* BroadPhaseCommon_h */

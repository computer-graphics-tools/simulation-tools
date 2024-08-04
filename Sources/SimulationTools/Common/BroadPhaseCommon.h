#ifndef BroadPhaseCommon_h
#define BroadPhaseCommon_h

#include <metal_stdlib>
using namespace metal;

#include "DistanceFunctions.h"
 #define MAX_CONNECTED_VERTICES 32
#define MAX_COLLISION_CANDIDATES 32

typedef float3 (*GetPositionFunc)(uint, constant void*);
typedef uint3 (*GetTriangleFunc)(uint, constant void*);

METAL_FUNC float3 getPosition(uint index, constant void* data) {
    constant float3* positions = (constant float3*)data;
    return positions[index];
}

METAL_FUNC uint3 getIndex(uint index, constant void* data) {
    constant uint3* positions = (constant uint3*)data;
    return positions[index];
}

METAL_FUNC float3 getPackedPosition(uint index, constant void* data) {
    constant packed_float3* positions = (constant packed_float3*)data;
    return positions[index];
}

METAL_FUNC uint3 getPackedIndex(uint index, constant void* data) {
    constant packed_uint3* positions = (constant packed_uint3*)data;
    return positions[index];
}

struct Triangle {
    float3 a;
    float3 b;
    float3 c;
};

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
    CollisionCandidate candidates[MAX_COLLISION_CANDIDATES];
};

METAL_FUNC Triangle createTriangle(uint3 triangleVertices,
                                   GetPositionFunc getPosition,
                                   constant void* positionData
                                   ) {
    return Triangle {
        getPosition(triangleVertices.x, positionData),
        getPosition(triangleVertices.y, positionData),
        getPosition(triangleVertices.z, positionData)
    };
}

template <typename T>
enable_if_t<is_floating_point_v<T>, void>
METAL_FUNC initializeCollisionCandidates(
    device uint* candidates,
    constant const vec<T, 3>* positions,
    thread SortedCollisionCandidates &sortedCandidates,
    uint index,
    const vec<T, 3> position,
    uint count
) {
    for (int i = 0; i < int(count); i++) {
        uint colliderIndex = candidates[index * count + i];
        sortedCandidates.candidates[i].index = colliderIndex;
        if (colliderIndex != UINT_MAX) {
            vec<T, 3> collider = positions[colliderIndex].xyz;
            sortedCandidates.candidates[i].distance = float(length_squared(position - collider));
        } else {
            sortedCandidates.candidates[i].distance = FLT_MAX;
        }
    }
}

METAL_FUNC void initializeCollisionCandidates(
    device uint* candidates,
    GetPositionFunc getPosition,
    constant void* positions,
    thread SortedCollisionCandidates &sortedCandidates,
    uint index,
    const float3 position,
    uint count
) {
    for (int i = 0; i < int(count); i++) {
        uint colliderIndex = candidates[index * count + i];
        sortedCandidates.candidates[i].index = colliderIndex;
        if (colliderIndex != UINT_MAX) {
            float3 collider = getPosition(colliderIndex, positions);
            sortedCandidates.candidates[i].distance = length_squared(position - collider);
        } else {
            sortedCandidates.candidates[i].distance = FLT_MAX;
        }
    }
}

void METAL_FUNC initializeTriangleCollisionCandidates(
    device uint* candidates,
    GetPositionFunc getPosition,
    constant void* positions,
    GetTriangleFunc getTriangle,
    constant void* triangleData,
    uint index,
    float3 position,
    thread SortedCollisionCandidates &collisionCandidates,
    uint count
) {
    for (int i = 0; i < int(count); i++) {
        uint colliderIndex = candidates[index * count + i];
        collisionCandidates.candidates[i].index = colliderIndex;
        if (colliderIndex != UINT_MAX) {
            uint3 triangleIndices = getTriangle(colliderIndex, triangleData);
            float3 a = getPosition(triangleIndices.x, positions);
            float3 b = getPosition(triangleIndices.y, positions);
            float3 c = getPosition(triangleIndices.z, positions);
            collisionCandidates.candidates[i].distance = usdTriangle(position, a, b, c);
        } else {
            collisionCandidates.candidates[i].distance = FLT_MAX;
        }
    }
}

template <typename T>
enable_if_t<is_floating_point_v<T>, void>
METAL_FUNC insertSeed(
    thread SortedCollisionCandidates &candidates,
    uint index,
    T distance,
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

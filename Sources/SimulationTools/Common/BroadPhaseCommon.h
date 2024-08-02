#ifndef BroadPhaseCommon_h
#define BroadPhaseCommon_h

#include <metal_stdlib>
using namespace metal;

#include "DistanceFunctions.h"
 #define MAX_CONNECTED_VERTICES 32
#define MAX_COLLISION_CANDIDATES 32

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

METAL_FUNC Triangle createTriangle(uint3 triangleVertices, constant float3* positions) {
    return Triangle {
        positions[triangleVertices.x],
        positions[triangleVertices.y],
        positions[triangleVertices.z]
    };
}

METAL_FUNC Triangle createTriangle(uint3 triangleVertices, constant packed_float3* positions) {
    return Triangle {
        positions[triangleVertices.x],
        positions[triangleVertices.y],
        positions[triangleVertices.z]
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

void METAL_FUNC initializeTriangleCollisionCandidates(
    device uint* candidates,
    constant packed_float3* positions,
    constant packed_uint3* triangles,
    uint index,
    float3 position,
    thread SortedCollisionCandidates &collisionCandidates,
    uint count
) {
    for (int i = 0; i < int(count); i++) {
        uint colliderIndex = candidates[index * count + i];
        collisionCandidates.candidates[i].index = colliderIndex;
        if (colliderIndex != UINT_MAX) {
            Triangle triangle = createTriangle(triangles[colliderIndex], positions);
            collisionCandidates.candidates[i].distance = usdTriangle(position, triangle.a.xyz, triangle.b.xyz, triangle.c.xyz);
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

template <typename T>
static inline float3 getPosition(T element);

template <>
inline float3 getPosition(float3 element) {
    return element;
}

template <>
inline float3 getPosition(half3 element) {
    return float3(element);
}

template <>
inline float3 getPosition(packed_float3 element) {
    return float3(element);
}


#endif /* BroadPhaseCommon_h */

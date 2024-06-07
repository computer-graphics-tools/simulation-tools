#ifndef Collision_h
#define Collision_h

#include "SDFMath.h"

#define INDICES_8_COUNT 8
struct Indices8 {
    uint values[8];
};

struct Triangle {
    float3 a;
    float3 b;
    float3 c;
};

template <typename T>
enable_if_t<is_floating_point_v<T>, Triangle>
METAL_FUNC createTriangle(uint3 triangleVertices, constant vec<T, 4> *positions) {
    return Triangle {
        .a = float3(positions[triangleVertices.x].xyz),
        .b = float3(positions[triangleVertices.y].xyz),
        .c = float3(positions[triangleVertices.z].xyz)
    };
}

#endif /* Collision_h */

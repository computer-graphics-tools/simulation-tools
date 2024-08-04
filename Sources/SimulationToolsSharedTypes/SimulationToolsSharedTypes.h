#ifndef SimulationToolsSharedTypes_h
#define SimulationToolsSharedTypes_h

#if __METAL_VERSION__

#include <metal_stdlib>

using namespace metal;
#define NS_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define metal_enum_t metal::int32_t

#else

#include <simd/simd.h>
#import <Foundation/Foundation.h>
#import <stdbool.h>
typedef int32_t metal_enum_t;

#endif

typedef struct {
    uint hashTableCapacity;
    float cellSize;
    uint maxCollisionCandidatesCount;
    uint connectedVerticesCount;
    uint bucketSize;
    uint gridSize;
    bool useExternalCollidable;
    bool usePackedCollidablePositions;
    bool usePackedColliderPositions;
    bool usePackedIndices;
} TriangleSHParameters;

#endif /* SimulationToolsSharedTypes_h */

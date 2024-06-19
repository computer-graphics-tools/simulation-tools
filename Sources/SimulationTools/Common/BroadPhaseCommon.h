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

#endif /* BroadPhaseCommon_h */

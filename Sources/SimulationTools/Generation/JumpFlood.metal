#include <metal_stdlib>
using namespace metal;

void JFAIter(texture3d<half, access::read_write> voxels,
             uint offset, uint3 id, uint dispatchCubeSide) {
    half3 idF = half3(id);
    half4 closest = voxels.read(id);
    half closestDist = HALF_MAX;
    uint3 bounds(dispatchCubeSide);
    
    for (uint i = 0; i < 3; i++) {
        for (uint j = 0; j < 3; j++) {
            for (uint k = 0; k < 3; k++) {
                int3 at = int3(i - 1, j - 1, k - 1) * int(offset) + int3(id);
                if (any(at < 0) || any(at >= int3(bounds))) continue;
                half4 voxel = voxels.read(uint3(at));
                if (voxel.w == 0.0h) continue;
                half voxelDist = distance(idF, voxel.xyz);
                if (voxelDist < closestDist) {
                    closestDist = voxelDist;
                    closest = voxel;
                }
            }
        }
    }
    voxels.write(closest, id);
}

kernel void jumpFlood(texture3d<half, access::read_write> voxels [[texture(0)]],
                      constant uint &dispatchCubeSide [[buffer(0)]],
                      constant uint &samplingOffset [[buffer(1)]],
                      uint3 id [[thread_position_in_grid]]) {
    JFAIter(voxels, samplingOffset, id, dispatchCubeSide);
}

kernel void jumpFloodToSDF(texture3d<half, access::read_write> voxels [[texture(0)]],
                              constant uint &dispatchCubeSide [[buffer(0)]],
                              constant half &postProcessThickness [[buffer(1)]],
                              uint3 id [[thread_position_in_grid]]) {
    half3 seedPos = voxels.read(id).xyz;
    half dist = (distance(seedPos, half3(id)) / half(dispatchCubeSide)) - postProcessThickness;
    voxels.write(half4(half3(dist), 1.0h), id);
}

struct MDVertex {
    packed_float3 position;
    packed_float3 normal;
    float2 uv;
};

kernel void meshToVoxel(texture3d<half, access::write> voxels [[texture(0)]],
                              const device MDVertex *vertices [[buffer(0)]],
                              const device int *indices [[buffer(1)]],
                              constant uint &trianglesCount [[buffer(2)]],
                              constant int &numSamples [[buffer(3)]],
                              constant float &scale [[buffer(4)]],
                              constant float3 &offset [[buffer(5)]],
                              constant uint &voxelSide [[buffer(6)]],
                              uint id [[thread_position_in_grid]]) {
    uint triID = id * 3;
    if (triID >= trianglesCount) return;
    
    half3 a = half3(vertices[indices[triID + 0]].position) * scale + half3(offset);
    half3 b = half3(vertices[indices[triID + 1]].position) * scale + half3(offset);
    half3 c = half3(vertices[indices[triID + 2]].position) * scale + half3(offset);
    
    half3 AB = b - a;
    half3 AC = c - a;
    
    float side = float(voxelSide);
    
    for (int i = 0; i < numSamples; i++) {
        half2 s = half2(fract(0.7548776662466927 * i), fract(0.5698402909980532 * i));
        s = s.x + s.y > 1.0 ? 1.0 - s : s;
        half3 pointOnTri = a + s.x * AB + s.y * AC;
        half3 scaled = pointOnTri * side;
        uint3 voxelIdx = uint3(floor(scaled));
        
        if (all(voxelIdx < voxelSide)) {
            float distFromCenter = 1.0 - length(fract(scaled) - 0.5);
            voxels.write(half4(half3(voxelIdx), distFromCenter), voxelIdx);
        }
    }
}

#include <metal_stdlib>
using namespace metal;

struct MDVertex {
    packed_float3 position;
    packed_float3 normal;
    float2 uv;
};

kernel void jumpFlood(
    texture3d<half, access::read_write> voxels [[texture(0)]],
    constant uint &dispatchCubeSide [[buffer(0)]],
    constant uint &samplingOffset [[buffer(1)]],
    uint3 id [[thread_position_in_grid]]
) {
    half3 idF = half3(id);
    half4 closest = voxels.read(id);
    half closestDist = HALF_MAX;
    uint3 bounds(dispatchCubeSide);
    
    for (uint i = 0; i < 3; i++) {
        for (uint j = 0; j < 3; j++) {
            for (uint k = 0; k < 3; k++) {
                int3 position = int3(i - 1, j - 1, k - 1) * int(samplingOffset) + int3(id);
                if (any(position < 0) || any(position >= int3(bounds))) continue;
                half4 voxel = voxels.read(uint3(position));
                if (voxel.w == 0.0h) continue;
                half voxelDist = distance(idF, voxel.xyz);

                if (voxelDist < closestDist) {
                    closestDist = voxelDist;
                    closest = voxel;
                }
            }
        }
    }
    voxels.write(closest, id);}

kernel void jumpFloodToSDF(texture3d<half, access::read_write> voxels [[texture(0)]],
                           constant uint &dispatchCubeSide [[buffer(0)]],
                           constant float &postProcessThickness [[buffer(1)]],
                           const device MDVertex *vertices [[buffer(2)]],
                           const device int *indices [[buffer(3)]],
                              uint3 id [[thread_position_in_grid]]) {
    half3 seedPos = voxels.read(id).xyz;
    uint triangleIndex = uint(as_type<ushort>(voxels.read(id).w));
    float3 n0 = vertices[indices[triangleIndex + 0]].normal;
    float3 n1 = vertices[indices[triangleIndex + 1]].normal;
    float3 n2 = vertices[indices[triangleIndex + 2]].normal;
    float3 normal = (n0 + n1 + n2) / 3.0;

    half dist = distance(seedPos, half3(id)); //- voxels.read(id).w;
    half normalizedDist = (dist / half(dispatchCubeSide)) - half(postProcessThickness);
    
    voxels.write(half4(normalizedDist, half3(normal)), id);
}

kernel void meshToVoxel(texture3d<half, access::write> voxels [[texture(0)]],
                        const device MDVertex *vertices [[buffer(0)]],
                        const device int *indices [[buffer(1)]],
                        constant uint &trianglesCount [[buffer(2)]],
                        constant int &numSamples [[buffer(3)]],
                        constant uint &voxelSide [[buffer(4)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= trianglesCount) return;
    uint triangleIndex = id * 3;
    
    float3 a = float3(vertices[indices[triangleIndex + 0]].position);
    float3 b = float3(vertices[indices[triangleIndex + 1]].position);
    float3 c = float3(vertices[indices[triangleIndex + 2]].position);
    float side = float(voxelSide);

    for (int i = 0; i < numSamples; i++) {
        float2 r = fract(float2(i) * float2(0.754877666246692760049508896358532874940835564978799543103, 0.569840290998053265911399958119574964216147658520394151385));
        float3 pointOnTri = a + r.x * (1 - r.y) * (b - a) + r.x * r.y * (c - a);
        
        float3 scaled = pointOnTri * side;
        uint3 voxelIdx = uint3(floor(scaled));
        
        if (all(voxelIdx < voxelSide)) {
//            float distFromCenter = 1.0 - length(fract(scaled) - float3(0.5, 0.5, 0.5));
            voxels.write(half4(half3(voxelIdx), as_type<half>(ushort(triangleIndex))), voxelIdx);
        }
    }
}

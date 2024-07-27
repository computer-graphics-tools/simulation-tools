#ifndef DistanceFunctions_h
#define DistanceFunctions_h

struct Triangle {
    float3 a;
    float3 b;
    float3 c;
};

METAL_FUNC Triangle createTriangle(uint3 triangleVertices, constant packed_float3 *positions) {
    return Triangle {
        .a = positions[triangleVertices.x],
        .b = positions[triangleVertices.y],
        .c = positions[triangleVertices.z]
    };
}

template <typename T>
enable_if_t<is_floating_point_v<T>, float>
METAL_FUNC usdTriangle(vec<T, 3> p, vec<T, 3> a, vec<T, 3> b, vec<T, 3> c) {
    vec<T, 3> ba = b - a;
    vec<T, 3> pa = p - a;
    vec<T, 3> cb = c - b;
    vec<T, 3> pb = p - b;
    vec<T, 3> ac = a - c;
    vec<T, 3> pc = p - c;
    vec<T, 3> nor = cross(ba, ac);

    return
        (sign(dot(cross(ba, nor), pa)) + sign(dot(cross(cb, nor), pb)) + sign(dot(cross(ac, nor), pc)) < 2.0)
        ?
        min(min(
            length_squared(ba * saturate(dot(ba, pa) / length_squared(ba)) - pa),
            length_squared(cb * saturate(dot(cb, pb) / length_squared(cb)) - pb)),
            length_squared(ac * saturate(dot(ac, pc) / length_squared(ac)) - pc))
        :
        dot(nor, pa) * dot(nor, pa) / length_squared(nor)
    ;
}

METAL_FUNC float3 closestPointTriangle(float3 p0, float3 p1, float3 p2, float3 p, thread float3& uvw) {
    float b0 = 1.0 / 3.0;
    float b1 = b0;
    float b2 = b0;
    
    float3 d1 = p1 - p0;
    float3 d2 = p2 - p0;
    float3 pp0 = p - p0;
    float a = length_squared(d1);
    float b = dot(d2, d1);
    float c = dot(pp0, d1);
    float d = b;
    float e = length_squared(d2);
    float f = dot(pp0, d2);
    float det = a * e - b * d;
    
    if (det != 0.0) {
        float s = (c * e - b * f) / det;
        float t = (a * f - c * d) / det;
        b0 = 1.0 - s - t; // inside triangle
        b1 = s;
        b2 = t;
        if (b0 < 0.0) { // on edge 1-2
            float3 d = p2 - p1;
            float d2 = length_squared(d);
            float t = (d2 == 0.0) ? 0.5 : dot(d, p - p1) / d2;
            t = saturate(t);

            b0 = 0.0;
            b1 = (1.0 - t);
            b2 = t;
        }
        else if (b1 < 0.0) { // on edge 2-0
            float3 d = p0 - p2;
            float d2 = length_squared(d);
            float t = (d2 == 0.0) ? 0.5 : dot(d, p - p2) / d2;
            t = saturate(t);

            b1 = 0.0;
            b2 = (1.0 - t);
            b0 = t;
        }
        else if (b2 < 0.0) { // on edge 0-1
            float3 d = p1 - p0;
            float d2 = length_squared(d);
            float t = (d2 == 0.0) ? 0.5 : dot(d, (p - p0)) / d2;
            t = saturate(t);

            b2 = 0.0;
            b0 = (1.0 - t);
            b1 = t;
        }
    }
    
    uvw = float3(b0, b1, b2);

    return b0 * p0 + b1 * p1 + b2 * p2;
}

METAL_FUNC float sdsBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
  return length_squared(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

#endif /* DistanceFunctions_h */

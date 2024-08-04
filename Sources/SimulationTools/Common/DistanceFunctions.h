#ifndef DistanceFunctions_h
#define DistanceFunctions_h

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

METAL_FUNC float sdsBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
  return length_squared(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

#endif /* DistanceFunctions_h */

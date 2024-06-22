#include <metal_stdlib>
using namespace metal;

#define SORT(F,L,R) {           \
    const auto v = sort(F,L,R); \
    (L) = uint2(v.x, v.y);      \
    (R) = uint2(v.z, v.w);      \
}                               \

static constexpr int genLeftIndex(const uint position,
                                  const uint blockSize) {
    const uint32_t blockMask = blockSize - 1;
    const auto no = position & blockMask; // comparator No. in block
    return ((position & ~blockMask) << 1) | no;
}

static uint4 sort(const bool reverse,
                  uint2 left,
                  uint2 right) {
    const bool lt = left.x < right.x;
    const bool swap = !lt ^ reverse;
    const bool4 dir = bool4(swap, swap, !swap, !swap); // (lt, gte) or (gte, lt)
    const uint4 v = select(uint4(left.x, left.y, left.x, left.y),
                           uint4(right.x, right.y, right.x, right.y),
                               dir);
    return v;
}

static void loadShared(const uint threadGroupSize,
                       const uint indexInThreadgroup,
                       const uint position,
                       device uint2* data,
                       threadgroup uint2* shared) {
    const auto index = genLeftIndex(position, threadGroupSize);
    shared[indexInThreadgroup] = data[index];
    shared[indexInThreadgroup | threadGroupSize] = data[index | threadGroupSize];
}

static void storeShared(const uint threadGroupSize,
                        const uint indexInThreadgroup,
                        const uint position,
                        device uint2* data,
                        threadgroup uint2* shared) {
    const auto index = genLeftIndex(position, threadGroupSize);
    data[index] = shared[indexInThreadgroup];
    data[index | threadGroupSize] = shared[indexInThreadgroup | threadGroupSize];
}

kernel void bitonicSortFirstPass(device uint2* data [[ buffer(0) ]],
                                 constant uint& gridSize [[ buffer(1) ]],
                                 threadgroup uint2* shared [[ threadgroup(0) ]],
                                 const uint threadgroupSize [[ threads_per_threadgroup ]],
                                 const uint indexInThreadgroup [[ thread_index_in_threadgroup ]],
                                 const uint position [[ thread_position_in_grid ]]) {
    loadShared(threadgroupSize, indexInThreadgroup, position, data, shared);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint unitSize = 1; unitSize <= threadgroupSize; unitSize <<= 1) {
        const bool reverse = (position & (unitSize)) != 0;    // to toggle direction
        for (uint blockSize = unitSize; 0 < blockSize; blockSize >>= 1) {
            const auto left = genLeftIndex(indexInThreadgroup, blockSize);
            SORT(reverse, shared[left], shared[left | blockSize]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    storeShared(threadgroupSize, indexInThreadgroup, position, data, shared);
}

kernel void bitonicSortGeneralPass(device uint2* data [[ buffer(0) ]],
                                   constant uint& gridSize [[ buffer(1) ]],
                                   constant uint2& params [[ buffer(2) ]],
                                   const uint position [[ thread_position_in_grid ]]) {
    const bool reverse = (position & (params.x >> 1)) != 0; // to toggle direction
    const uint blockSize = params.y; // size of comparison sets
    const auto left = genLeftIndex(position, blockSize);
    SORT(reverse, data[left], data[left | blockSize]);
}

kernel void bitonicSortFinalPass(device uint2* data,
                                 constant uint& gridSize [[ buffer(1) ]],
                                 constant uint2& params [[ buffer(2) ]],
                                 threadgroup uint2* shared [[ threadgroup(0) ]],
                                 const uint threadgroupSize [[ threads_per_threadgroup ]],
                                 const uint indexInThreadgroup [[ thread_index_in_threadgroup ]],
                                 const uint position [[ thread_position_in_grid ]]) {
    loadShared(threadgroupSize, indexInThreadgroup, position, data, shared);
    const auto unitSize = params.x;
    const auto blockSize = params.y;
    const auto num = 10 + 1;
    // Toggle direction.
    const bool reverse = (position & (unitSize >> 1)) != 0;
    for (uint i = 0; i < num; ++i) {
        const auto width = blockSize >> i;
        const auto left = genLeftIndex(indexInThreadgroup, width);
        SORT(reverse, shared[left], shared[left | width]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    storeShared(threadgroupSize, indexInThreadgroup, position, data, shared);
}

#undef SORT

import Metal

/// `MTLBufferProvider` is a utility to create `MTLBuffer` instances on either a provided `MTLHeap` or directly on the device.
/// It tries to initialize the buffer on the heap if provided, otherwise, it falls back to initializing on the device.
struct MTLBufferProvider {
    let device: MTLDevice
    let heap: MTLHeap?

    init(device: MTLDevice, heap: MTLHeap? = nil) {
        self.device = device
        self.heap = heap
    }

    func buffer<T>(with array: [T]) throws -> MTLBuffer {
        let buffer = try self.buffer(for: T.self, count: array.count)
        try buffer.put(array)
        
        return buffer
    }

    func buffer<T>(for type: T.Type, count: Int) throws -> MTLBuffer {
        return try (heap?.buffer(for: type, count: count, options: heap?.resourceOptions ?? []) ?? device.buffer(for: type, count: count))
    }
}

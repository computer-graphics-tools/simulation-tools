import MetalTools

public class MTLTypedBuffer<Value> {
    public let buffer: MTLBuffer
    public let count: Int
    public var values: [Value]? {
        return buffer.array(of: Value.self, count: count)
    }
    
    init(
        count: Int,
        options: MTLResourceOptions = .cpuCacheModeWriteCombined,
        bufferAllocator: MTLBufferAllocator
    ) throws {
        self.count = count
        buffer = try bufferAllocator.buffer(for: Value.self, count: count, options: options)
    }
    
    init(
        values: [Value],
        options: MTLResourceOptions = .cpuCacheModeWriteCombined,
        bufferAllocator: MTLBufferAllocator
    ) throws {
        count = values.count
        buffer = try bufferAllocator.buffer(with: values)
    }
    
    init(
        buffer: MTLBuffer,
        count: Int
    ) throws {
        self.count = count
        self.buffer = buffer
    }
    
    public func put(values: [Value]) throws {
        try buffer.put(values)
    }
}

public extension MTLDevice {
    func typedBuffer<T>(with array: [T], options: MTLResourceOptions = .cpuCacheModeWriteCombined) throws -> MTLTypedBuffer<T> {
        try MTLTypedBuffer(values: array, bufferAllocator: .init(type: .device(self)))
    }
    
    func typedBuffer<T>(for type: T.Type, count: Int) throws -> MTLTypedBuffer<T> {
        try MTLTypedBuffer(count: count, bufferAllocator: .init(type: .device(self)))
    }
    
    func typedBuffer<T>(with buffer: MTLBuffer, count: Int) throws -> MTLTypedBuffer<T> {
        try MTLTypedBuffer(buffer: buffer, count: count)
    }
}

public extension MTLHeap {
    func typedBuffer<T>(with array: [T], options: MTLResourceOptions = .cpuCacheModeWriteCombined) throws -> MTLTypedBuffer<T> {
        try MTLTypedBuffer(values: array, bufferAllocator: .init(type: .heap(self)))
    }
    
    func typedBuffer<T>(for type: T.Type, count: Int) throws -> MTLTypedBuffer<T> {
        try MTLTypedBuffer(count: count, bufferAllocator: .init(type: .heap(self)))
    }
    
    func typedBuffer<T>(with buffer: MTLBuffer, count: Int) throws -> MTLTypedBuffer<T> {
        try MTLTypedBuffer(buffer: buffer, count: count)
    }
}

import MetalTools

public class MTLTypedBuffer<Value> {
    public let buffer: MTLBuffer
    public let count: Int
    public var values: [Value]? {
        return buffer.array(of: Value.self, count: count)
    }
    
    public init(
        count: Int,
        options: MTLResourceOptions = [.storageModeShared],
        device: MTLDevice,
        heap: MTLHeap? = nil
    ) throws {
        self.count = count
        buffer = try (heap?.buffer(for: Value.self, count: count, options: options) ?? device.buffer(for: Value.self, count: count, options: options))
    }
    
    public init(
        values: [Value],
        options: MTLResourceOptions = [.storageModeShared],
        device: MTLDevice,
        heap: MTLHeap? = nil
    ) throws {
        count = values.count
        buffer = try (heap?.buffer(with: values) ?? device.buffer(with: values, options: options))
    }
    
    public func put(values: [Value]) throws {
        try buffer.put(values)
    }
}

public extension MTLDevice {
    func typedBuffer<T>(with array: [T], heap: MTLHeap? = nil) throws -> MTLTypedBuffer<T> {
        try MTLTypedBuffer(values: array, device: self, heap: heap)
    }
    
    func typedBuffer<T>(for type: T.Type, count: Int, heap: MTLHeap? = nil) throws -> MTLTypedBuffer<T> {
        try MTLTypedBuffer(count: count, device: self)
    }
}

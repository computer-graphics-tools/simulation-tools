import MetalTools

public class TypedMTLBuffer<Value> {
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

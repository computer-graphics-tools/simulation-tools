import Metal
import MetalTools

public enum MTLBufferValueType {
    case float, float2, float3, float4
    case half, half2, half3, half4
    case packedFloat3
    case uint, uint2, uint3, uint4
    case packedUInt3
    
    var stride: Int {
        switch self {
        case .float: return MemoryLayout<Float>.stride
        case .float2: return MemoryLayout<SIMD2<Float>>.stride
        case .float3: return MemoryLayout<SIMD3<Float>>.stride
        case .float4: return MemoryLayout<SIMD4<Float>>.stride
        case .half: return MemoryLayout<Float16>.stride
        case .half2: return MemoryLayout<SIMD2<Float16>>.stride
        case .half3: return MemoryLayout<SIMD3<Float16>>.stride
        case .half4: return MemoryLayout<SIMD4<Float16>>.stride
        case .packedFloat3: return MemoryLayout<Float>.stride * 3
        case .uint: return MemoryLayout<UInt32>.stride
        case .uint2: return MemoryLayout<SIMD2<UInt32>>.stride
        case .uint3: return MemoryLayout<SIMD3<UInt32>>.stride
        case .uint4: return MemoryLayout<SIMD4<UInt32>>.stride
        case .packedUInt3: return MemoryLayout<UInt32>.stride * 3
        }
    }
}

public struct MTLTypedBufferDescriptor {
    public var valueType: MTLBufferValueType
    public var count: Int
    
    public init(valueType: MTLBufferValueType, count: Int) {
        self.valueType = valueType
        self.count = count
    }
}

public class MTLTypedBuffer {
    public let buffer: MTLBuffer
    public let descriptor: MTLTypedBufferDescriptor
    
    init(descriptor: MTLTypedBufferDescriptor, options: MTLResourceOptions = .cpuCacheModeWriteCombined, bufferAllocator: MTLBufferAllocator) throws {
        self.descriptor = descriptor
        self.buffer = try bufferAllocator.buffer(length: descriptor.count * descriptor.valueType.stride, options: options)
    }
    
    init<T>(values: [T], valueType: MTLBufferValueType, options: MTLResourceOptions = .cpuCacheModeWriteCombined, bufferAllocator: MTLBufferAllocator) throws {
        self.descriptor = MTLTypedBufferDescriptor(valueType: valueType, count: values.count)
        self.buffer = try bufferAllocator.buffer(with: values)
    }
    
    init(buffer: MTLBuffer, descriptor: MTLTypedBufferDescriptor) {
        assert(buffer.length >= descriptor.count * descriptor.valueType.stride, "Buffer is too small for the specified count and value type")
        self.buffer = buffer
        self.descriptor = descriptor
    }
    
    public func values<T>() -> [T]? {
        return buffer.array(of: T.self, count: descriptor.count)
    }
}

public extension MTLDevice {
    func typedBuffer<T>(with array: [T], valueType: MTLBufferValueType, options: MTLResourceOptions = .cpuCacheModeWriteCombined) throws -> MTLTypedBuffer {
        try MTLTypedBuffer(values: array, valueType: valueType, bufferAllocator: .init(type: .device(self)))
    }
    
    func typedBuffer(descriptor: MTLTypedBufferDescriptor, options: MTLResourceOptions = .cpuCacheModeWriteCombined) throws -> MTLTypedBuffer {
        try MTLTypedBuffer(descriptor: descriptor, bufferAllocator: .init(type: .device(self)))
    }
    
    func typedBuffer(with buffer: MTLBuffer, valueType: MTLBufferValueType) -> MTLTypedBuffer {
        let descriptor = MTLTypedBufferDescriptor(valueType: valueType, count: buffer.length / valueType.stride)
        return MTLTypedBuffer(buffer: buffer, descriptor: descriptor)
    }
}

public extension MTLHeap {
    func typedBuffer<T>(with array: [T], valueType: MTLBufferValueType, options: MTLResourceOptions = .cpuCacheModeWriteCombined) throws -> MTLTypedBuffer {
        try MTLTypedBuffer(values: array, valueType: valueType, bufferAllocator: .init(type: .heap(self)))
    }
    
    func typedBuffer(descriptor: MTLTypedBufferDescriptor, options: MTLResourceOptions = .cpuCacheModeWriteCombined) throws -> MTLTypedBuffer {
        try MTLTypedBuffer(descriptor: descriptor, bufferAllocator: .init(type: .heap(self)))
    }
    
    func typedBuffer(with buffer: MTLBuffer, valueType: MTLBufferValueType) throws -> MTLTypedBuffer {
        guard buffer.heap === self else {
            throw MTLTypedBufferError.bufferNotInHeap
        }
        let descriptor = MTLTypedBufferDescriptor(valueType: valueType, count: buffer.length / valueType.stride)
        return MTLTypedBuffer(buffer: buffer, descriptor: descriptor)
    }
}

extension MTLBufferAllocator {
    func typedBuffer(
        descriptor: MTLTypedBufferDescriptor,
        options: MTLResourceOptions = .cpuCacheModeWriteCombined
    ) throws -> MTLTypedBuffer {
        switch self.type {
        case let .device(device):
            return try device.typedBuffer(descriptor: descriptor, options: options)
        case let .heap(heap):
            return try heap.typedBuffer(descriptor: descriptor, options: heap.resourceOptions)
        }
    }
    
    func typedBuffer<T>(
        with array: [T],
        valueType: MTLBufferValueType,
        options: MTLResourceOptions = .cpuCacheModeWriteCombined
    ) throws -> MTLTypedBuffer {
        switch self.type {
        case let .device(device):
            return try device.typedBuffer(with: array, valueType: valueType, options: options)
        case let .heap(heap):
            return try heap.typedBuffer(with: array, valueType: valueType, options: options)
        }
    }
}

public enum MTLTypedBufferError: Error {
    case bufferNotInHeap
}

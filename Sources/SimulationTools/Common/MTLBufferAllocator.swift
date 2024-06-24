import MetalTools

final class MTLBufferAllocator {
    enum `Type` {
        case device(MTLDevice)
        case heap(MTLHeap)
    }

    var device: MTLDevice {
        switch type {
        case let .device(device): return device
        case let .heap(heap): return heap.device
        }
    }

    lazy var syncQueue: MTLCommandQueue? = device.makeCommandQueue()

    private let type: `Type`

    init(type: Type) {
        self.type = type
    }

    func buffer<T>(
        for type: T.Type,
        count: Int = 1,
        options: MTLResourceOptions = .cpuCacheModeWriteCombined
    ) throws -> MTLBuffer {
        switch self.type {
        case let .device(device):
            return try device.buffer(
                for: type,
                count: count,
                options: options
            )
        case let .heap(heap):
            return try heap.buffer(
                for: type,
                count: count,
                options: heap.resourceOptions
            )
        }
    }

    func buffer<T>(
        with value: T,
        options: MTLResourceOptions = .cpuCacheModeWriteCombined
    ) throws -> MTLBuffer {
        switch type {
        case let .device(device):
            return try device.buffer(
                with: value,
                options: options
            )
        case let .heap(heap):
            return try heap.buffer(
                with: value,
                options: heap.resourceOptions
            )
        }
    }

    func buffer<T>(
        with values: [T],
        options: MTLResourceOptions = .cpuCacheModeWriteCombined
    ) throws -> MTLBuffer {
        switch type {
        case let .device(device):
            return try device.buffer(
                with: values,
                options: options
            )
        case let .heap(heap):
            return try heap.buffer(
                with: values,
                options: heap.resourceOptions
            )
        }
    }

    func buffer(
        length: Int,
        options: MTLResourceOptions = .cpuCacheModeWriteCombined
    ) throws -> MTLBuffer {
        switch type {
        case let .device(device):
            guard let buffer = device.makeBuffer(length: length, options: options)
            else { throw MetalError.MTLDeviceError.bufferCreationFailed }
            return buffer
        case let .heap(heap):
            guard let buffer = heap.makeBuffer(length: length, options: heap.resourceOptions)
            else { throw MetalError.MTLDeviceError.bufferCreationFailed }
            return buffer
        }
    }
}

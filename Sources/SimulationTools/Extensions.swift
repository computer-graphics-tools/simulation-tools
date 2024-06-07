import Metal

extension Array {
    var dataLength: Int { self.count * MemoryLayout<Element>.stride }
}

extension Collection {
    subscript(safe index: Index) -> Element? {
        self.indices.contains(index) ? self[index] : nil
    }
    
    var isNotEmpty: Bool { !isEmpty }
}

extension Array {
    subscript(index: UInt32) -> Element {
        return self[Int(index)]
    }
}

extension SIMD4 where Scalar == Float {
    var xyz: SIMD3<Float> {
        .init(x: x, y: y, z: z)
    }
}

extension MTLDevice {
    func buffer<T>(with array: [T], heap: MTLHeap?) throws -> MTLBuffer {
        guard let heap else { return try self.buffer(with: array) }
        let length = array.count * MemoryLayout<T>.stride
        guard let buffer = heap.makeBuffer(length: length, options: []) else {
            throw NSError(domain: "MTLBufferError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create MTLBuffer from heap"])
        }
        let pointer = buffer.contents().bindMemory(to: T.self, capacity: array.count)
        array.withUnsafeBufferPointer { srcPointer in
            pointer.update(from: srcPointer.baseAddress!, count: array.count)
        }
        return buffer
    }

    func buffer<T>(for type: T.Type, count: Int, heap: MTLHeap?) throws -> MTLBuffer {
        guard let heap else { return try self.buffer(for: type, count: count) }
        let length = count * MemoryLayout<T>.stride
        guard let buffer = heap.makeBuffer(length: length, options: []) else {
            throw NSError(domain: "MTLBufferError", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to create MTLBuffer from heap"])
        }
        return buffer
    }
}

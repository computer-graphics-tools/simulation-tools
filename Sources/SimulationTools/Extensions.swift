import Metal

extension Array {
    var dataLength: Int { self.count * MemoryLayout<Element>.stride }
}

extension Array {
    func chunked(into size: Int) -> [[Element]] {
        var chunks: [[Element]] = []
        var index = 0
        
        while index < self.count {
            let chunk = Array(self[index..<Swift.min(index + size, self.count)])
            chunks.append(chunk)
            index += size
        }
        
        return chunks
    }
}

extension Array {
    subscript(index: UInt32) -> Element {
        return self[Int(index)]
    }
}

extension Collection {
    subscript(safe index: Index) -> Element? {
        self.indices.contains(index) ? self[index] : nil
    }
    
    var isNotEmpty: Bool { !isEmpty }
}

extension MTLDevice {
    func typedBuffer<T>(with array: [T], heap: MTLHeap? = nil) throws -> TypedMTLBuffer<T> {
        try TypedMTLBuffer(values: array, device: self, heap: heap)
    }
    
    func typedBuffer<T>(for type: T.Type, count: Int, heap: MTLHeap? = nil) throws -> TypedMTLBuffer<T> {
        try TypedMTLBuffer(count: count, device: self)
    }

    func buffer<T>(with array: [T], heap: MTLHeap?) throws -> MTLBuffer {
        let buffer = try buffer(for: T.self, count: array.count, heap: heap)
        try buffer.put(array)
        
        return buffer
    }

    func buffer<T>(for type: T.Type, count: Int, heap: MTLHeap?) throws -> MTLBuffer {
        return try (heap?.buffer(for: type, count: count, options: heap?.resourceOptions ?? []) ?? self.buffer(for: type, count: count))
    }
}

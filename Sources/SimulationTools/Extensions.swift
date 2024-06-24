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

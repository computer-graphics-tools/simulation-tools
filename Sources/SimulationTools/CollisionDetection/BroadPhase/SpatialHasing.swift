import Metal
import MetalTools

public final class SpatialHashing {
    public struct Configuration {
        let cellSize: Float
        let radius: Float
        
        public init(cellSize: Float, radius: Float) {
            self.cellSize = cellSize
            self.radius = radius
        }
    }

    public let configuration: Configuration

    private let computeHashAndIndexState: MTLComputePipelineState
    private let computeCellBoundariesState: MTLComputePipelineState
    private let convertToHalfPrecisionPackedState: MTLComputePipelineState
    private let convertToHalfPrecisionUnpackedState: MTLComputePipelineState
    private let reorderHalfPrecisionState: MTLComputePipelineState
    private let findCollisionCandidatesFloat3State: MTLComputePipelineState
    private let findCollisionCandidatesPackedFloat3State: MTLComputePipelineState
    private let bitonicSort: BitonicSort

    private let cellStart: MTLBuffer
    private let cellEnd: MTLBuffer
    let hashTable: (buffer: MTLBuffer, paddedCount: Int)
    let halfPositions: MTLBuffer
    let sortedHalfPositions: MTLTypedBuffer
     let hashTableCapacity: Int

    public convenience init(
        heap: MTLHeap,
        configuration: Configuration,
        maxElementsCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .heap(heap)),
            configuration: configuration,
            maxElementsCount: maxElementsCount
        )
    }
    
    public convenience init(
        device: MTLDevice,
        configuration: Configuration,
        maxElementsCount: Int
    ) throws {
        try self.init(
            bufferAllocator: .init(type: .device(device)),
            configuration: configuration,
            maxElementsCount: maxElementsCount
        )
    }

    private init(
        bufferAllocator: MTLBufferAllocator,
        configuration: Configuration,
        maxElementsCount: Int
    ) throws {
        let library = try bufferAllocator.device.makeDefaultLibrary(bundle: .module)
        let deviceSupportsNonuniformThreadgroups = bufferAllocator.device.supports(feature: .nonUniformThreadgroups)

        let constantValues = MTLFunctionConstantValues()
        constantValues.set(deviceSupportsNonuniformThreadgroups, at: 0)

        self.configuration = configuration
        self.computeHashAndIndexState = try library.computePipelineState(function: "computeHashAndIndexState", constants: constantValues)
        self.computeCellBoundariesState = try library.computePipelineState(function: "computeCellBoundaries", constants: constantValues)
        self.convertToHalfPrecisionPackedState = try library.computePipelineState(function: "convertToHalfPrecisionPacked", constants: constantValues)
        self.convertToHalfPrecisionUnpackedState = try library.computePipelineState(function: "convertToHalfPrecisionUnpacked", constants: constantValues)
        self.reorderHalfPrecisionState = try library.computePipelineState(function: "reorderHalfPrecision", constants: constantValues)
        self.findCollisionCandidatesFloat3State = try library.computePipelineState(function: "findCollisionCandidatesFloat3", constants: constantValues)
        self.findCollisionCandidatesPackedFloat3State = try library.computePipelineState(function: "findCollisionCandidatesPackedFloat3", constants: constantValues)
        self.bitonicSort = try .init(library: library)
        
        self.hashTableCapacity = maxElementsCount * 2
        self.hashTable = try BitonicSort.buffer(count: maxElementsCount, bufferAllocator: bufferAllocator)
        self.cellStart = try bufferAllocator.buffer(for: UInt32.self, count: self.hashTableCapacity)
        self.cellEnd = try bufferAllocator.buffer(for: UInt32.self, count: self.hashTableCapacity)
        self.halfPositions = try bufferAllocator.buffer(for: SIMD3<Float16>.self, count: maxElementsCount)
        self.sortedHalfPositions = try bufferAllocator.typedBuffer(descriptor: .init(valueType: .half3, count: maxElementsCount))
    }
    
    public func build(
        elements: MTLTypedBuffer,
        in commandBuffer: MTLCommandBuffer
    ) {
        commandBuffer.compute { encoder in
            encoder.setBuffer(elements.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.halfPositions, offset: 0, index: 1)
            encoder.setValue(UInt32(elements.descriptor.count), at: 2)
            
            let state = isPacked(elements) ? self.convertToHalfPrecisionPackedState : self.convertToHalfPrecisionUnpackedState
            encoder.dispatch1d(state: state, exactlyOrCovering: elements.descriptor.count)
            
            encoder.setBuffer(self.halfPositions, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 1)
            encoder.setValue(UInt32(self.hashTableCapacity), at: 2)
            encoder.setValue(self.configuration.cellSize, at: 3)
            encoder.setValue(UInt32(elements.descriptor.count), at: 4)
            
            encoder.dispatch1d(state: self.computeHashAndIndexState, exactlyOrCovering: elements.descriptor.count)
        }
        
        self.bitonicSort.encode(data: self.hashTable.buffer, count: self.hashTable.paddedCount, in: commandBuffer)
        
        commandBuffer.compute { encoder in
            encoder.setBuffer(self.halfPositions, offset: 0, index: 0)
            encoder.setBuffer(self.sortedHalfPositions.buffer, offset: 0, index: 1)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 2)
            encoder.setValue(UInt32(elements.descriptor.count), at: 3)
            encoder.dispatch1d(state: self.reorderHalfPrecisionState, exactlyOrCovering: elements.descriptor.count)
        }
        
        commandBuffer.compute { encoder in
            let threadgroupWidth = 256
            encoder.setBuffer(self.cellStart, offset: 0, index: 0)
            encoder.setBuffer(self.cellEnd, offset: 0, index: 1)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 2)
            encoder.setValue(UInt32(elements.descriptor.count), at: 3)
            encoder.setThreadgroupMemoryLength((threadgroupWidth + 16) * MemoryLayout<UInt32>.size, index: 0)
            encoder.dispatch1d(state: self.computeCellBoundariesState, exactlyOrCovering: elements.descriptor.count, threadgroupWidth: threadgroupWidth)
        }
    }
    
    public func findCollisionCandidates(
        extrnalElements: MTLTypedBuffer? = nil,
        collisionCandidates: MTLTypedBuffer,
        connectedVertices: MTLTypedBuffer?,
        in commandBuffer: MTLCommandBuffer
    ) {
        let elements = extrnalElements ?? self.sortedHalfPositions
        let maxCandidatesCount = collisionCandidates.descriptor.count / elements.descriptor.count

        commandBuffer.compute { encoder in
            encoder.setBuffer(collisionCandidates.buffer, offset: 0, index: 0)
            encoder.setBuffer(self.hashTable.buffer, offset: 0, index: 1)
            encoder.setBuffer(self.cellStart, offset: 0, index: 2)
            encoder.setBuffer(self.cellEnd, offset: 0, index: 3)
            encoder.setBuffer(self.sortedHalfPositions.buffer, offset: 0, index: 4)
            encoder.setBuffer(elements.buffer, offset: 0, index: 5)
            if let connectedVertices {
                encoder.setBuffer(connectedVertices.buffer, offset: 0, index: 6)
            } else {
                encoder.setValue([UInt32.zero], at: 6)
            }
            encoder.setValue(UInt32(self.hashTableCapacity), at: 7)
            encoder.setValue(self.configuration.radius, at: 8)
            encoder.setValue(self.configuration.cellSize, at: 9)
            encoder.setValue(UInt32(maxCandidatesCount), at: 10)
            encoder.setValue(UInt32((connectedVertices?.descriptor.count ?? 0) / elements.descriptor.count), at: 11)
            encoder.setValue(UInt32(extrnalElements?.descriptor.count ?? 0), at: 12)
            encoder.setValue(UInt32(elements.descriptor.count), at: 13)

            let state: MTLComputePipelineState
            state = isPacked(elements) ? self.findCollisionCandidatesPackedFloat3State : self.findCollisionCandidatesFloat3State

            encoder.dispatch1d(state: state, exactlyOrCovering: elements.descriptor.count)
        }
    }
    
    private func isPacked(_ buffer: MTLTypedBuffer) -> Bool {
        return buffer.descriptor.valueType == .packedFloat3 || buffer.descriptor.valueType == .packedUInt3
    }
}

public extension SpatialHashing {
    static func totalBuffersSize(maxElementsCount: Int) -> Int {
        let cellStartSize = maxElementsCount * MemoryLayout<UInt32>.stride * 2
        let cellEndSize = maxElementsCount * MemoryLayout<UInt32>.stride * 2
        let hashTableSize = maxElementsCount * MemoryLayout<SIMD2<UInt32>>.stride * 2
        let halfPositionsSize = maxElementsCount * MemoryLayout<SIMD3<Float16>>.stride * 2
        
        return cellStartSize + cellEndSize + hashTableSize + halfPositionsSize
    }
}

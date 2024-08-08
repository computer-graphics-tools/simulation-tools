import MetalTools

public final class JumpFlood3D {
    private let pipelineState: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
        let constantValues = MTLFunctionConstantValues()
        let deviceSupportsNonuniformThreadgroups = library.device.supports(feature: .nonUniformThreadgroups)
        constantValues.set(deviceSupportsNonuniformThreadgroups, at: 0)
        pipelineState = try library.computePipelineState(function: "jumpFlood", constants: constantValues)
    }

    public func encode(
        voxelTexture: MTLTexture,
        in commandBuffer: MTLCommandBuffer
    ) {
        commandBuffer.compute { encoder in
            var offset = voxelTexture.width / 2
            encoder.setTexture(voxelTexture, index: 0)
            encoder.setValue(UInt32(voxelTexture.size.width), at: 0)
        
            while offset >= 1 {
                encoder.setValue(UInt32(offset), at: 1)
                encoder.dispatch3d(state: pipelineState, exactlyOrCovering: voxelTexture.size)
                offset /= 2
            }
        }
    }
}

public final class MeshToVoxel {
    private let pipelineState: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
        let constantValues = MTLFunctionConstantValues()
        let deviceSupportsNonuniformThreadgroups = library.device.supports(feature: .nonUniformThreadgroups)
        constantValues.set(deviceSupportsNonuniformThreadgroups, at: 0)
        pipelineState = try library.computePipelineState(function: "meshToVoxel", constants: constantValues)
    }
    
    public func encode(
        voxelTexture: MTLTexture,
        positions: MTLTypedBuffer,
        indices: MTLTypedBuffer,
        numSamples: Int = 16,
        scale: Float = 1.0,
        offset: SIMD3<Float> = .zero,
        in commandBuffer: MTLCommandBuffer
    ) {
        commandBuffer.compute { encoder in
            encoder.setTexture(voxelTexture, index: 0)
            encoder.setBuffer(positions.buffer, offset: 0, index: 0)
            encoder.setBuffer(indices.buffer, offset: 0, index: 1)
            encoder.setValue(UInt32(indices.descriptor.count / 3), at: 2)
            encoder.setValue(UInt32(numSamples), at: 3)
            encoder.setValue(UInt32(voxelTexture.size.width), at: 4)
            encoder.dispatch1d(state: self.pipelineState, exactlyOrCovering: indices.descriptor.count / 3)
        }
    }
}

public final class JumpFloodToSDF {
    private let pipelineState: MTLComputePipelineState

    public init(device: MTLDevice) throws {
        let library = try device.makeDefaultLibrary(bundle: .module)
        let constantValues = MTLFunctionConstantValues()
        let deviceSupportsNonuniformThreadgroups = library.device.supports(feature: .nonUniformThreadgroups)
        constantValues.set(deviceSupportsNonuniformThreadgroups, at: 0)
        pipelineState = try library.computePipelineState(function: "jumpFloodToSDF", constants: constantValues)
    }
    
    public func encode(
        voxelTexture: MTLTexture,
        thickness: Float = 0.01,
        mesh: MTKMesh,
        in commandBuffer: MTLCommandBuffer
    ) {
        commandBuffer.compute { encoder in
            encoder.setTexture(voxelTexture, index: 0)
            encoder.setValue(UInt32(voxelTexture.size.width), at: 0)
            encoder.setValue(thickness, at: 1)
            encoder.setBuffer(mesh.vertexBuffers[0].buffer, offset: 0, index: 2)
            encoder.setBuffer(mesh.submeshes[0].indexBuffer.buffer, offset: 0, index: 3)
            encoder.dispatch3d(state: pipelineState, exactlyOrCovering: voxelTexture.size)
        }
    }
}

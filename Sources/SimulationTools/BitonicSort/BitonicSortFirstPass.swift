import Metal

extension BitonicSort {
    final class FirstPass {
        // MARK: - Properties

        let pipelineState: MTLComputePipelineState
        private let deviceSupportsNonuniformThreadgroups: Bool

        // MARK: - Init

        init(
            library: MTLLibrary
        ) throws {
            self.deviceSupportsNonuniformThreadgroups = library.device
                .supports(feature: .nonUniformThreadgroups)

            let constantValues = MTLFunctionConstantValues()
            constantValues.set(
                self.deviceSupportsNonuniformThreadgroups,
                at: 0
            )

            self.pipelineState = try library.computePipelineState(
                function: "bitonicSortFirstPass",
                constants: constantValues
            )
        }

        // MARK: - Encode

        func encode(
            data: MTLBuffer,
            elementStride: Int,
            gridSize: Int,
            unitSize: Int,
            in commandBuffer: MTLCommandBuffer
        ) {
            commandBuffer.compute { encoder in
                encoder.label = "Bitonic Sort First Pass"
                self.encode(
                    data: data,
                    elementStride: elementStride,
                    gridSize: gridSize,
                    unitSize: unitSize,
                    using: encoder
                )
            }
        }

        func encode(
            data: MTLBuffer,
            elementStride: Int,
            gridSize: Int,
            unitSize: Int,
            using encoder: MTLComputeCommandEncoder
        ) {
            encoder.setBuffers(data)
            encoder.setValue(UInt32(gridSize), at: 1)
            encoder.setThreadgroupMemoryLength(
                (elementStride * unitSize) << 1,
                index: 0
            )

            if self.deviceSupportsNonuniformThreadgroups {
                encoder.dispatch1d(
                    state: self.pipelineState,
                    exactly: gridSize,
                    threadgroupWidth: unitSize
                )
            } else {
                encoder.dispatch1d(
                    state: self.pipelineState,
                    covering: gridSize,
                    threadgroupWidth: unitSize
                )
            }
        }
    }
}

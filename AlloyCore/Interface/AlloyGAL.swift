import Metal
import simd
import QuartzCore

public class AlloyGAL {
    private var vertexPool: [Float] = []
    private var indexPool: [UInt32] = []
    private var vertexBuffers: [(offset: UInt32, count: UInt32)] = []
    private var indexBuffers: [(offset: UInt32, count: UInt32)] = []
    private var indexBufferVertexPoolOffsets: [UInt32] = []
    private var pipelines: [AlloyPipelineDescriptor?] = []
    private var computePipelines: [AlloyComputePipelineDescriptor?] = []
    public var computeTime: Float = 0

    private let device: MTLDevice
    private var textures: [MTLTexture?] = []

    private var frameActive = false
    private var frameCommandBuffer: [UInt32] = []
    public private(set) var poolVersion: UInt32 = 0

    public init() {
        self.device = MTLCreateSystemDefaultDevice()!
    }

    public func beginFrame() {
        frameActive = true
        frameCommandBuffer.removeAll(keepingCapacity: true)
    }

    public func endFrame() {
        frameActive = false
    }

    public func createVertexBuffer(data: [Float]) -> AlloyBufferHandle {
        let offset = UInt32(vertexPool.count / 12)
        let count = UInt32(data.count / 12)
        let handle = UInt32(vertexBuffers.count)
        vertexBuffers.append((offset, count))
        vertexPool.append(contentsOf: data)
        poolVersion &+= 1
        return handle
    }

    public func createIndexBuffer(data: [UInt32], vertexHandle: AlloyBufferHandle) -> AlloyBufferHandle {
        guard Int(vertexHandle) < vertexBuffers.count else { return 0xFFFFFFFF }
        let vertexPoolOffset = vertexBuffers[Int(vertexHandle)].offset
        let offset = UInt32(indexPool.count)
        let count = UInt32(data.count)
        let handle = UInt32(indexBuffers.count)
        indexBuffers.append((offset, count))
        indexBufferVertexPoolOffsets.append(vertexPoolOffset)
        for idx in data {
            indexPool.append(idx + vertexPoolOffset)
        }
        poolVersion &+= 1
        return handle
    }

    public func createPipeline(_ desc: AlloyPipelineDescriptor) -> AlloyPipelineHandle {
        let handle = UInt32(pipelines.count)
        pipelines.append(desc)
        return handle
    }

    public func destroyPipeline(_ handle: AlloyPipelineHandle) {
        guard Int(handle) < pipelines.count else { return }
        pipelines[Int(handle)] = nil
    }
    
    public func createComputePipeline(_ desc: AlloyComputePipelineDescriptor) -> AlloyComputePipelineHandle {
        let handle = UInt32(computePipelines.count)
        computePipelines.append(desc)
        return handle
    }

    public func destroyComputePipeline(_ handle: AlloyComputePipelineHandle) {
        guard Int(handle) < computePipelines.count else { return }
        computePipelines[Int(handle)] = nil
    }

    public func clearColor(r: Float, g: Float, b: Float, a: Float) {
        frameCommandBuffer.append(0x02)
        frameCommandBuffer.append(r.bitPattern)
        frameCommandBuffer.append(g.bitPattern)
        frameCommandBuffer.append(b.bitPattern)
        frameCommandBuffer.append(a.bitPattern)
    }

    public func bindPipeline(_ handle: AlloyPipelineHandle) {
        guard Int(handle) < pipelines.count,
              let desc = pipelines[Int(handle)] else { return }
        frameCommandBuffer.append(0x03)
        frameCommandBuffer.append(desc.depthTestEnabled ? 1 : 0)
        frameCommandBuffer.append(desc.cullMode.rawValue)
        frameCommandBuffer.append(desc.blendEnabled ? 1 : 0)
        frameCommandBuffer.append(desc.shaderID)
    }

    public func setViewport(width: Int, height: Int) {
        frameCommandBuffer.append(0x04)
        frameCommandBuffer.append(Float(width).bitPattern)
        frameCommandBuffer.append(Float(height).bitPattern)
    }

    public func setTransform(matrix: [Float]) {
        guard matrix.count == 16 else { return }
        frameCommandBuffer.append(0x06)
        for f in matrix { frameCommandBuffer.append(f.bitPattern) }
    }

    public func drawIndexed(iboHandle: AlloyBufferHandle,
                            indexCount: UInt32,
                            firstIndex: UInt32,
                            textureID: UInt32) {
        guard Int(iboHandle) < indexBuffers.count else { return }
        let globalStart = indexBuffers[Int(iboHandle)].offset + firstIndex
        frameCommandBuffer.append(0x01)
        frameCommandBuffer.append(globalStart)
        frameCommandBuffer.append(indexCount)
        frameCommandBuffer.append(textureID)
    }
    
    public func bindComputePipeline(_ handle: AlloyComputePipelineHandle) {
        guard Int(handle) < computePipelines.count,
              computePipelines[Int(handle)] != nil else { return }
        frameCommandBuffer.append(AlloyOpcode.bindComputePipeline.rawValue)
        frameCommandBuffer.append(handle)
    }

    public func bindComputeBuffer(slot: Int, handle: AlloyBufferHandle, byteOffsetFloats: Int = 0) {
        guard Int(handle) < vertexBuffers.count else { return }
        let poolOffsetFloats = Int(vertexBuffers[Int(handle)].offset) * 12
        frameCommandBuffer.append(AlloyOpcode.bindComputeVertexPool.rawValue)
        frameCommandBuffer.append(UInt32(slot))
        frameCommandBuffer.append(UInt32(poolOffsetFloats))
        frameCommandBuffer.append(UInt32(byteOffsetFloats))
    }

    public func dispatchCompute(groups: SIMD3<UInt32>) {
        frameCommandBuffer.append(AlloyOpcode.computeDispatch.rawValue)
        frameCommandBuffer.append(groups.x)
        frameCommandBuffer.append(groups.y)
        frameCommandBuffer.append(groups.z)
    }

    public func updateVertexBuffer(_ handle: AlloyBufferHandle,
                                   data: [Float],
                                   offset: Int = 0) {
        guard Int(handle) < vertexBuffers.count, !data.isEmpty else { return }
        let vb = vertexBuffers[Int(handle)]
        let poolStart = Int(vb.offset) * 12
        let poolEnd   = poolStart + Int(vb.count) * 12

        let writeStart = poolStart + offset
        let writeEnd   = writeStart + data.count
        let clipStart  = max(poolStart, min(writeStart, poolEnd))
        let clipEnd    = min(poolEnd, max(writeEnd, poolStart))

        guard clipStart < clipEnd else {
            #if DEBUG
            assertionFailure("updateVertexBuffer: range out of handle bounds")
            #endif
            return
        }

        let localStart = clipStart - writeStart
        let localEnd   = localStart + (clipEnd - clipStart)
        let slice = Array(data[localStart..<localEnd])

        for (i, v) in slice.enumerated() {
            vertexPool[clipStart + i] = v
        }
AlloyLog.log("updVB h=\(handle) off=\(clipStart) n=\(slice.count)")
        frameCommandBuffer.append(0x09)
        frameCommandBuffer.append(UInt32(clipStart))
        frameCommandBuffer.append(UInt32(slice.count))
        for f in slice {
            frameCommandBuffer.append(f.bitPattern)
        }
    }

    public func updateIndexBuffer(_ handle: AlloyBufferHandle,
                                  data: [UInt32],
                                  offset: Int = 0) {
        guard Int(handle) < indexBuffers.count,
              Int(handle) < indexBufferVertexPoolOffsets.count,
              !data.isEmpty else { return }
        let ib = indexBuffers[Int(handle)]
        let vboOffset = indexBufferVertexPoolOffsets[Int(handle)]
        let poolStart = Int(ib.offset)
        let poolEnd   = poolStart + Int(ib.count)

        let writeStart = poolStart + offset
        let writeEnd   = writeStart + data.count
        let clipStart  = max(poolStart, min(writeStart, poolEnd))
        let clipEnd    = min(poolEnd, max(writeEnd, poolStart))

        guard clipStart < clipEnd else {
            #if DEBUG
            assertionFailure("updateIndexBuffer: range out of handle bounds")
            #endif
            return
        }

        let localStart = clipStart - writeStart
        let localEnd   = localStart + (clipEnd - clipStart)
        let absolute = data[localStart..<localEnd].map { $0 + vboOffset }

        for (i, v) in absolute.enumerated() {
            indexPool[clipStart + i] = v
        }

        frameCommandBuffer.append(0x0A)
        frameCommandBuffer.append(UInt32(clipStart))
        frameCommandBuffer.append(UInt32(absolute.count))
        for v in absolute {
            frameCommandBuffer.append(v)
        }
    }

    // MARK: - 纹理（v0.3.0 地基）

    public func createTexture(_ desc: AlloyTextureDescriptor) -> AlloyTextureHandle {
        let mtlDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: desc.width,
            height: desc.height,
            mipmapped: false)
        mtlDesc.usage = [.shaderRead]
        guard let tex = device.makeTexture(descriptor: mtlDesc) else { return 0xFFFFFFFF }
        if let data = desc.data {
            tex.replace(region: MTLRegionMake2D(0, 0, desc.width, desc.height),
                        mipmapLevel: 0,
                        withBytes: data,
                        bytesPerRow: desc.width * 4)
        }
        let handle = UInt32(textures.count)
        textures.append(tex)
        return handle
    }

    public func destroyTexture(_ handle: AlloyTextureHandle) {
        guard Int(handle) < textures.count else { return }
        textures[Int(handle)] = nil
    }

    public func getTexture(_ handle: AlloyTextureHandle) -> MTLTexture? {
        guard Int(handle) < textures.count else { return nil }
        return textures[Int(handle)]
    }

    public func getVertexPool() -> [Float] { vertexPool }
    public func getIndexPool() -> [UInt32] { indexPool }

    public func submit(to renderer: AlloyRenderer,
                       drawable: CAMetalDrawable) -> MTLCommandBuffer? {
        if renderer.cachedPoolVersion != poolVersion {
            renderer.uploadGeometry(vertexData: vertexPool, indexData: indexPool)
            renderer.cachedPoolVersion = poolVersion
        }
        return renderer.render(
            drawable: drawable,
            textures: textures,
            computePipelines: computePipelines,
            computeTime: computeTime,
            rawCommands: frameCommandBuffer
        )
    }
}
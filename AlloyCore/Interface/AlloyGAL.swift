import Metal
import simd
import QuartzCore

public class AlloyGAL {
    private var vertexPool: [Float] = []
    private var indexPool: [UInt32] = []
    private var vertexBuffers: [(offset: UInt32, count: UInt32)] = []
    private var indexBuffers: [(offset: UInt32, count: UInt32)] = []
    private var currentIndexBufferOffset: UInt32 = 0
    private var pipelines: [AlloyPipelineDescriptor?] = []

    private var frameActive = false
    private var frameCommandBuffer: [UInt32] = []

    public init() {}

    public func beginFrame() {
        if frameActive { print("AlloyGAL: frame already active") }
        frameActive = true
        frameCommandBuffer.removeAll(keepingCapacity: true)
    }

    public func endFrame() {
        if !frameActive { print("AlloyGAL: no active frame") }
        frameActive = false
    }

    public func createVertexBuffer(data: [Float]) -> AlloyBufferHandle {
        let offset = UInt32(vertexPool.count / 12)
        let count = UInt32(data.count / 12)
        let handle = UInt32(vertexBuffers.count)
        vertexBuffers.append((offset, count))
        vertexPool.append(contentsOf: data)
        return handle
    }

    public func createIndexBuffer(data: [UInt32], vertexHandle: AlloyBufferHandle) -> AlloyBufferHandle {
        guard Int(vertexHandle) < vertexBuffers.count else { return 0xFFFFFFFF }
        let vertexPoolOffset = vertexBuffers[Int(vertexHandle)].offset
        let offset = UInt32(indexPool.count)
        let count = UInt32(data.count)
        let handle = UInt32(indexBuffers.count)
        indexBuffers.append((offset, count))
        for idx in data {
            indexPool.append(idx + vertexPoolOffset)
        }
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

    public func bindVertexBuffer(_ handle: AlloyBufferHandle) {
        guard Int(handle) < vertexBuffers.count else { return }
        frameCommandBuffer.append(0x07)
        frameCommandBuffer.append(vertexBuffers[Int(handle)].offset)
    }

    public func bindIndexBuffer(_ handle: AlloyBufferHandle) {
        guard Int(handle) < indexBuffers.count else { return }
        currentIndexBufferOffset = indexBuffers[Int(handle)].offset
        frameCommandBuffer.append(0x08)
        frameCommandBuffer.append(currentIndexBufferOffset)
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

    public func drawIndexed(indexCount: UInt32, startIndex: UInt32, textureID: UInt32) {
        let globalStart = currentIndexBufferOffset + startIndex
        frameCommandBuffer.append(0x01)
        frameCommandBuffer.append(globalStart)
        frameCommandBuffer.append(indexCount)
        frameCommandBuffer.append(textureID)
    }

    public func getVertexPool() -> [Float] { vertexPool }
    public func getIndexPool() -> [UInt32] { indexPool }

    public func submit(to renderer: AlloyRenderer,
                       drawable: CAMetalDrawable,
                       texture0: MTLTexture,
                       texture1: MTLTexture) -> MTLCommandBuffer? {
        return renderer.render(
            drawable: drawable,
            texture0: texture0,
            texture1: texture1,
            rawCommands: frameCommandBuffer
        )
    }
}
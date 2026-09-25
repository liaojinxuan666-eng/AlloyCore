import Metal
import simd
import QuartzCore

public class AlloyGAL {
    private var vertexPool: [Float] = []
    private var indexPool: [UInt32] = []
    private var pipelines: [AlloyPipelineDescriptor] = []
    private var commandBuffer: [UInt32] = []

    public init() {}

    public func beginFrame() {
        commandBuffer.removeAll(keepingCapacity: true)
    }

    public func resetResources() {
        vertexPool.removeAll()
        indexPool.removeAll()
        pipelines.removeAll()
    }

    // === 资源 ===

    public func createVertexBuffer(data: [Float]) -> AlloyBufferHandle {
        let offset = UInt32(vertexPool.count / 12)
        vertexPool.append(contentsOf: data)
        return offset
    }

    public func createIndexBuffer(data: [UInt32], vertexBaseOffset: UInt32) -> AlloyBufferHandle {
        let offset = UInt32(indexPool.count)
        for idx in data {
            indexPool.append(idx + vertexBaseOffset)
        }
        return offset
    }

    public func createPipeline(_ desc: AlloyPipelineDescriptor) -> AlloyPipelineHandle {
        let handle = UInt32(pipelines.count)
        pipelines.append(desc)
        return handle
    }

    public func getPipeline(_ handle: AlloyPipelineHandle) -> AlloyPipelineDescriptor? {
        guard Int(handle) < pipelines.count else { return nil }
        return pipelines[Int(handle)]
    }

    // === 状态 ===

    public func clearColor(r: Float, g: Float, b: Float, a: Float) {
        commandBuffer.append(0x02)
        commandBuffer.append(r.bitPattern)
        commandBuffer.append(g.bitPattern)
        commandBuffer.append(b.bitPattern)
        commandBuffer.append(a.bitPattern)
    }

    public func bindPipeline(_ handle: AlloyPipelineHandle) {
        guard let desc = getPipeline(handle) else { return }
        commandBuffer.append(0x03)
        commandBuffer.append(desc.depthTestEnabled ? 1 : 0)
        commandBuffer.append(desc.cullMode.rawValue)
        commandBuffer.append(desc.blendEnabled ? 1 : 0)
        commandBuffer.append(desc.shaderID)
    }

    public func setViewport(width: Int, height: Int) {
        commandBuffer.append(0x04)
        commandBuffer.append(Float(width).bitPattern)
        commandBuffer.append(Float(height).bitPattern)
    }

    public func setTransform(matrix: [Float]) {
        guard matrix.count == 16 else { return }
        commandBuffer.append(0x06)
        for f in matrix {
            commandBuffer.append(f.bitPattern)
        }
    }

    // === 绘制 ===

    public func drawIndexed(indexCount: UInt32, startIndex: UInt32, textureID: UInt32) {
        commandBuffer.append(0x01)
        commandBuffer.append(startIndex)
        commandBuffer.append(indexCount)
        commandBuffer.append(textureID)
    }

    // === 提交 ===

    public func getVertexPool() -> [Float] { vertexPool }
    public func getIndexPool() -> [UInt32] { indexPool }

    public func submit(to renderer: AlloyRenderer, drawable: CAMetalDrawable, texture0: MTLTexture, texture1: MTLTexture) -> MTLCommandBuffer? {
        return renderer.render(
            drawable: drawable,
            texture0: texture0,
            texture1: texture1,
            rawCommands: commandBuffer
        )
    }
}
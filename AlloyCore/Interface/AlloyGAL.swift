import Metal
import simd
import QuartzCore

public class AlloyGAL {
    // === 资源池 ===
    private var vertexPool: [Float] = []
    private var indexPool: [UInt32] = []
    private var pipelines: [AlloyPipelineDescriptor?] = []

    // === 帧状态 ===
    private var frameActive = false
    private var frameCommandBuffer: [UInt32] = []

    public init() {}

    // === 帧生命周期 ===

    public func beginFrame() {
        if frameActive {
            print("AlloyGAL: beginFrame called while a frame is already active")
        }
        frameActive = true
        frameCommandBuffer.removeAll(keepingCapacity: true)
    }

    public func endFrame() {
        if !frameActive {
            print("AlloyGAL: endFrame called with no active frame")
        }
        frameActive = false
    }

    // === 资源创建 ===

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

    // === 资源销毁 ===

    public func destroyPipeline(_ handle: AlloyPipelineHandle) {
        guard Int(handle) < pipelines.count else { return }
        pipelines[Int(handle)] = nil
    }

    // === 命令录制 ===

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
        for f in matrix {
            frameCommandBuffer.append(f.bitPattern)
        }
    }

    public func drawIndexed(indexCount: UInt32, startIndex: UInt32, textureID: UInt32) {
        frameCommandBuffer.append(0x01)
        frameCommandBuffer.append(startIndex)
        frameCommandBuffer.append(indexCount)
        frameCommandBuffer.append(textureID)
    }

    // === 数据访问 ===

    public func getVertexPool() -> [Float] { vertexPool }
    public func getIndexPool() -> [UInt32] { indexPool }

    // === 提交 ===

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
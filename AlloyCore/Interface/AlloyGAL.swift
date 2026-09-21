import Metal
import simd
import QuartzCore

public struct AlloyPipelineDescriptor {
    public var depthTestEnabled: Bool = true
    public var cullMode: Int = 0
    public var blendEnabled: Bool = false
    public var shaderID: UInt32 = 0
    public init() {}
}

public class AlloyGAL {
    private var commandBuffer: [UInt32] = []
    public var vertexData: [Float] = []   // 每个顶点 12 个 float
    public var indexData: [UInt32] = []
    
    public init() {}
    
    // 顶点布局（12 个 float）：
    // [0-1] position, [2-5] color, [6-7] uv, [8] invZ, [9-11] normal
    public func addVertex(
        position: SIMD2<Float>,
        color: SIMD4<Float>,
        uv: SIMD2<Float>,
        invZ: Float,
        normal: SIMD3<Float>
    ) -> UInt32 {
        let idx = UInt32(vertexData.count / 12)
        vertexData.append(contentsOf: [
            position.x, position.y,
            color.x, color.y, color.z, color.w,
            uv.x, uv.y,
            invZ,
            normal.x, normal.y, normal.z
        ])
        return idx
    }
    
    public func clearColor(r: Float, g: Float, b: Float, a: Float) {
        commandBuffer.append(0x02)
        commandBuffer.append(r.bitPattern)
        commandBuffer.append(g.bitPattern)
        commandBuffer.append(b.bitPattern)
        commandBuffer.append(a.bitPattern)
    }
    
    public func bindPipeline(_ desc: AlloyPipelineDescriptor) {
        commandBuffer.append(0x03)
        commandBuffer.append(desc.depthTestEnabled ? 1 : 0)
        commandBuffer.append(UInt32(desc.cullMode))
        commandBuffer.append(desc.blendEnabled ? 1 : 0)
        commandBuffer.append(desc.shaderID)
    }
    
    public func setViewport(width: Int, height: Int) {
        commandBuffer.append(0x04)
        commandBuffer.append(Float(width).bitPattern)
        commandBuffer.append(Float(height).bitPattern)
    }
    
    // 指令 0x01：画一个三角形
    // [0x01] [indexStart] [indexCount] [texID] = 4 个 uint
    public func drawIndexed(v0: UInt32, v1: UInt32, v2: UInt32, textureID: UInt32) {
        let indexStart = UInt32(indexData.count)
        indexData.append(contentsOf: [v0, v1, v2])
        
        commandBuffer.append(0x01)
        commandBuffer.append(indexStart)
        commandBuffer.append(3) // 1 个三角形有 3 个索引
        commandBuffer.append(textureID)
    }
    
    public func submit(
        to renderer: AlloyRenderer,
        drawable: CAMetalDrawable,
        texture0: MTLTexture,
        texture1: MTLTexture
    ) -> MTLCommandBuffer? {
        return renderer.render(
            drawable: drawable,
            texture0: texture0,
            texture1: texture1,
            rawCommands: commandBuffer,
            vertexData: vertexData,
            indexData: indexData
        )
    }
}
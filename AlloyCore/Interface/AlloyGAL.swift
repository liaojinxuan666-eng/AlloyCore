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
    public var vertexData: [Float] = []
    public var indexData: [UInt32] = []
    
    public init() {}
    
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
    
    // 画一个三角形
    public func drawIndexed(v0: UInt32, v1: UInt32, v2: UInt32, textureID: UInt32) {
        let indexStart = UInt32(indexData.count)
        indexData.append(contentsOf: [v0, v1, v2])
        
        commandBuffer.append(0x01)
        commandBuffer.append(indexStart)
        commandBuffer.append(3)
        commandBuffer.append(textureID)
    }
    
    // 一次性绘制大段索引（比如球体）
    public func drawIndexedRange(startIndex: UInt32, indexCount: UInt32, textureID: UInt32) {
        commandBuffer.append(0x01)
        commandBuffer.append(startIndex)
        commandBuffer.append(indexCount)
        commandBuffer.append(textureID)
    }
    
    public func submit(to renderer: AlloyRenderer, drawable: CAMetalDrawable, texture0: MTLTexture, texture1: MTLTexture) -> MTLCommandBuffer? {
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
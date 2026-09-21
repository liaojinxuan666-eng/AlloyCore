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
    
    // 🔥 修复：drawTriangle 直接携带 textureID
    public func drawTriangle(
        p0: SIMD2<Float>, p1: SIMD2<Float>, p2: SIMD2<Float>,
        color: SIMD4<Float>,
        z0: Float, z1: Float, z2: Float,
        textureID: UInt32, // 🔥 新增
        uv0: SIMD2<Float> = .zero, uv1: SIMD2<Float> = .zero, uv2: SIMD2<Float> = .zero,
        n0: SIMD3<Float> = .zero, n1: SIMD3<Float> = .zero, n2: SIMD3<Float> = .zero
    ) {
        commandBuffer.append(0x01)
        
        let floats: [Float] = [
            p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
            color.x, color.y, color.z, color.w,
            color.x, color.y, color.z, color.w,
            color.x, color.y, color.z, color.w,
            uv0.x, uv0.y, uv1.x, uv1.y, uv2.x, uv2.y,
            z0, z1, z2,
            n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, n2.x, n2.y, n2.z,
            Float(textureID) // 🔥 将 textureID 追加到末尾
        ]
        
        for f in floats {
            commandBuffer.append(f.bitPattern)
        }
    }
    
    public func submit(to renderer: AlloyRenderer, drawable: CAMetalDrawable, texture0: MTLTexture, texture1: MTLTexture) -> MTLCommandBuffer? {
        return renderer.render(drawable: drawable, texture0: texture0, texture1: texture1, rawCommands: commandBuffer)
    }
}
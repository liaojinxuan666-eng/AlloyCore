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
    
    // 清屏
    public func clearColor(r: Float, g: Float, b: Float, a: Float) {
        commandBuffer.append(0x02)
        commandBuffer.append(r.bitPattern)
        commandBuffer.append(g.bitPattern)
        commandBuffer.append(b.bitPattern)
        commandBuffer.append(a.bitPattern)
    }
    
    // 绑定管线状态
    public func bindPipeline(_ desc: AlloyPipelineDescriptor) {
        commandBuffer.append(0x03)
        commandBuffer.append(desc.depthTestEnabled ? 1 : 0)
        commandBuffer.append(UInt32(desc.cullMode))
        commandBuffer.append(desc.blendEnabled ? 1 : 0)
        commandBuffer.append(desc.shaderID)
    }
    
    // 🔥 新增：设置视口（1 opcode + 2 float = 3 uint）
    public func setViewport(width: Int, height: Int) {
        commandBuffer.append(0x04)
        commandBuffer.append(Float(width).bitPattern)
        commandBuffer.append(Float(height).bitPattern)
    }
    
    // 🔥 新增：绑定纹理（1 opcode + 1 uint = 2 uint）
    public func bindTexture(textureID: UInt32) {
        commandBuffer.append(0x05)
        commandBuffer.append(textureID)
    }
    
    // 绘制三角形
    public func drawTriangle(
        p0: SIMD2<Float>, p1: SIMD2<Float>, p2: SIMD2<Float>,
        color: SIMD4<Float>,
        z0: Float, z1: Float, z2: Float,
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
            n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, n2.x, n2.y, n2.z
        ]
        
        for f in floats {
            commandBuffer.append(f.bitPattern)
        }
    }
    
    public func submit(to renderer: AlloyRenderer, drawable: CAMetalDrawable, texture: MTLTexture) -> MTLCommandBuffer? {
        return renderer.render(drawable: drawable, texture: texture, rawCommands: commandBuffer)
    }
}
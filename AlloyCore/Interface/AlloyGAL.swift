import Metal
import simd

// 管线状态描述符（模拟 DX/Vulkan 的 PSO）
public struct AlloyPipelineDescriptor {
    public var depthTestEnabled: Bool = true
    public var cullMode: Int = 0 // 0: none, 1: back, 2: front
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
    
    // 绘制三角形
    public func drawTriangle(
        p0: SIMD2<Float>, p1: SIMD2<Float>, p2: SIMD2<Float>,
        color: SIMD4<Float>, z: Float,
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
            z,
            n0.x, n0.y, n0.z, n1.x, n1.y, n1.z, n2.x, n2.y, n2.z
        ]
        
        for f in floats {
            commandBuffer.append(f.bitPattern)
        }
    }
    
    // 提交给底层引擎
    public func submit(to renderer: AlloyRenderer, drawable: CAMetalDrawable, texture: MTLTexture) -> MTLCommandBuffer? {
        return renderer.render(drawable: drawable, texture: texture, rawCommands: commandBuffer)
    }
}
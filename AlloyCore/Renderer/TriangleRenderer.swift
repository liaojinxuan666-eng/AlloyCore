import Metal
import simd
import QuartzCore // 需要引入 CAMetalDrawable

// 与 Metal 内存对齐的结构体
public struct Vertex {
    public var position: SIMD2<Float>
    public var color: SIMD4<Float>
    
    public init(position: SIMD2<Float>, color: SIMD4<Float>) {
        self.position = position
        self.color = color
    }
}

public class TriangleRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelineState: MTLComputePipelineState!
    
    public init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = commandQueue
        
        let bundle = Bundle(for: TriangleRenderer.self)
        guard let library = try? device.makeDefaultLibrary(bundle: bundle),
              let kernel = library.makeFunction(name: "rasterize_triangle") else {
            print("AlloyCore 初始化失败：无法加载 Metal 库")
            return nil
        }
        
        do {
            pipelineState = try device.makeComputePipelineState(function: kernel)
        } catch {
            print("创建 Pipeline 失败: \(error)")
            return nil
        }
    }
    
    public func render(drawable: CAMetalDrawable) {
        let texture = drawable.texture
        let w = Float(texture.width)
        let h = Float(texture.height)
        
        // 动态适配屏幕的三角形顶点
        let vertices = [
            Vertex(position: SIMD2<Float>(w / 2, 100), color: SIMD4<Float>(1, 0, 0, 1)), // 顶部红
            Vertex(position: SIMD2<Float>(100, h - 100), color: SIMD4<Float>(0, 1, 0, 1)), // 左下绿
            Vertex(position: SIMD2<Float>(w - 100, h - 100), color: SIMD4<Float>(0, 0, 1, 1))  // 右下蓝
        ]
        
        let vertexBuffer = device.makeBuffer(bytes: vertices,
                                             length: MemoryLayout<Vertex>.stride * vertices.count,
                                             options: .storageModeShared)
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setTexture(texture, index: 0)
        
        let threadW = pipelineState.threadExecutionWidth
        let threadH = pipelineState.maxTotalThreadsPerThreadgroup / threadW
        let threadsPerThreadgroup = MTLSize(width: threadW, height: threadH, depth: 1)
        let threadsPerGrid = MTLSize(width: texture.width, height: texture.height, depth: 1)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

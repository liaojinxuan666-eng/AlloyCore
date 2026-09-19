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
        
        // 从 AlloyCore 这个 Framework 的 Bundle 里加载 Metal 库
        let bundle = Bundle(for: TriangleRenderer.self)
        guard let library = try? device.makeDefaultLibrary(bundle: bundle),
              let kernel = library.makeFunction(name: "rasterize_triangle") else {
            print("AlloyCore 初始化失败：无法加载 Metal 库或找到 rasterize_triangle 函数")
            return nil
        }
        
        do {
            pipelineState = try device.makeComputePipelineState(function: kernel)
        } catch {
            print("创建 Pipeline 失败: \(error)")
            return nil
        }
    }
    
    // 注意：这里参数改成了 CAMetalDrawable
    public func render(drawable: CAMetalDrawable) {
        let texture = drawable.texture
        
        // 1. 定义三角形顶点 (假设画布是 1024x1024)
        let vertices = [
            Vertex(position: SIMD2<Float>(512, 100), color: SIMD4<Float>(1, 0, 0, 1)), // 红
            Vertex(position: SIMD2<Float>(100, 900), color: SIMD4<Float>(0, 1, 0, 1)), // 绿
            Vertex(position: SIMD2<Float>(900, 900), color: SIMD4<Float>(0, 0, 1, 1))  // 蓝
        ]
        
        let vertexBuffer = device.makeBuffer(bytes: vertices,
                                             length: MemoryLayout<Vertex>.stride * vertices.count,
                                             options: .storageModeShared)
        
        // 2. 创建 Command Buffer 和 Compute Encoder
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setTexture(texture, index: 0)
        
        // 3. 多点协作调度核心
        let w = pipelineState.threadExecutionWidth
        let h = pipelineState.maxTotalThreadsPerThreadgroup / w
        let threadsPerThreadgroup = MTLSize(width: w, height: h, depth: 1)
        let threadsPerGrid = MTLSize(width: texture.width, height: texture.height, depth: 1)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        // 🔥 核心修复：提交并呈现！
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

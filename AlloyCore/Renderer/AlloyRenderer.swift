import Metal
import simd
import QuartzCore

public struct Vertex {
    public var position: SIMD2<Float>
    public var color: SIMD4<Float>
    
    public init(position: SIMD2<Float>, color: SIMD4<Float>) {
        self.position = position
        self.color = color
    }
}

public class AlloyRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelineState: MTLComputePipelineState!
    
    public init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = commandQueue
        
        let bundle = Bundle(for: AlloyRenderer.self)
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
    
    // 🔥 引擎接口升级：接收一个三角形列表
    public func render(drawable: CAMetalDrawable, triangles: [[Vertex]]) {
        let texture = drawable.texture
        guard !triangles.isEmpty else { return }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setTexture(texture, index: 0)
        
        let threadW = pipelineState.threadExecutionWidth
        let threadH = pipelineState.maxTotalThreadsPerThreadgroup / threadW
        let threadsPerThreadgroup = MTLSize(width: threadW, height: threadH, depth: 1)
        let threadsPerGrid = MTLSize(width: texture.width, height: texture.height, depth: 1)
        
        // 🔥 核心：循环提交每一个三角形
        // 这实际上模拟了 GPU 的多次 Draw Call
        for triangle in triangles {
            let vertexBuffer = device.makeBuffer(bytes: triangle,
                                                 length: MemoryLayout<Vertex>.stride * triangle.count,
                                                 options: .storageModeShared)
            encoder.setBuffer(vertexBuffer, offset: 0, index: 0)
            encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        }
        
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

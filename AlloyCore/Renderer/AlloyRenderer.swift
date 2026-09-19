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

// 🔥 指令结构体
public struct DrawTriangleCommand {
    public var v0: Vertex
    public var v1: Vertex
    public var v2: Vertex
    public var z: Float
    
    public init(v0: Vertex, v1: Vertex, v2: Vertex, z: Float) {
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.z = z
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
              let kernel = library.makeFunction(name: "process_commands") else {
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
    
    // 🔥 引擎接口：接收指令流
    public func render(drawable: CAMetalDrawable, commands: [DrawTriangleCommand]) {
        let texture = drawable.texture
        guard !commands.isEmpty else { return }
        
        // 将所有指令打包进一个大的 MTLBuffer
        let commandBuffer = device.makeBuffer(bytes: commands,
                                              length: MemoryLayout<DrawTriangleCommand>.stride * commands.count,
                                              options: .storageModeShared)
        
        var commandCount = UInt32(commands.count)
        
        guard let cmdQueueBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdQueueBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(commandBuffer, offset: 0, index: 0)
        encoder.setBytes(&commandCount, length: MemoryLayout<UInt32>.size, index: 1) // 传递指令数量
        encoder.setTexture(texture, index: 0)
        
        let threadW = pipelineState.threadExecutionWidth
        let threadH = pipelineState.maxTotalThreadsPerThreadgroup / threadW
        let threadsPerThreadgroup = MTLSize(width: threadW, height: threadH, depth: 1)
        let threadsPerGrid = MTLSize(width: texture.width, height: texture.height, depth: 1)
        
        // 🔥 只派发一次！让 GPU 自己去循环遍历指令流
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        cmdQueueBuffer.present(drawable)
        cmdQueueBuffer.commit()
    }
}

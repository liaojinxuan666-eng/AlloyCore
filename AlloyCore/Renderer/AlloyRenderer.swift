import Metal
import simd
import QuartzCore

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
    
    public func render(drawable: CAMetalDrawable, texture: MTLTexture, rawCommands: [Float]) {
        let drawableTexture = drawable.texture
        guard !rawCommands.isEmpty else { return }
        
        let commandBuffer = device.makeBuffer(bytes: rawCommands,
                                              length: rawCommands.count * MemoryLayout<Float>.size,
                                              options: .storageModeShared)
        
        // 🔥 步长从 25 变成 27
        var commandCount = UInt32(rawCommands.count / 27)
        
        guard let cmdQueueBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdQueueBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(commandBuffer, offset: 0, index: 0)
        encoder.setBytes(&commandCount, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setTexture(drawableTexture, index: 0)
        encoder.setTexture(texture, index: 1)
        
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadsPerGrid = MTLSize(width: drawableTexture.width, height: drawableTexture.height, depth: 1)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        cmdQueueBuffer.present(drawable)
        cmdQueueBuffer.commit()
    }
}

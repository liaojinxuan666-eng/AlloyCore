import Metal
import simd
import QuartzCore

public class AlloyRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelineState: MTLComputePipelineState!
    let tileSize: Int = 16
    
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
    
    public func render(
        drawable: CAMetalDrawable,
        texture0: MTLTexture,
        texture1: MTLTexture,
        rawCommands: [UInt32],
        vertexData: [Float],
        indexData: [UInt32]
    ) -> MTLCommandBuffer? {
        let outputTexture = drawable.texture
        guard !rawCommands.isEmpty else { return nil }
        
        let commandBuffer = device.makeBuffer(bytes: rawCommands,
                                              length: rawCommands.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        
        let vertexBuffer = device.makeBuffer(bytes: vertexData,
                                             length: vertexData.count * MemoryLayout<Float>.size,
                                             options: .storageModeShared)
        
        let indexBuffer = device.makeBuffer(bytes: indexData,
                                            length: indexData.count * MemoryLayout<UInt32>.size,
                                            options: .storageModeShared)
        
        var commandCount = UInt32(rawCommands.count)
        
        guard let cmdQueueBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdQueueBuffer.makeComputeCommandEncoder() else { return nil }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(commandBuffer, offset: 0, index: 0)
        encoder.setBytes(&commandCount, length: MemoryLayout<UInt32>.size, index: 1)
        encoder.setBuffer(vertexBuffer, offset: 0, index: 2)
        encoder.setBuffer(indexBuffer, offset: 0, index: 3)
        encoder.setTexture(outputTexture, index: 0)
        encoder.setTexture(texture0, index: 1)
        encoder.setTexture(texture1, index: 2)
        
        let threadsPerThreadgroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let threadsPerGrid = MTLSize(width: outputTexture.width, height: outputTexture.height, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return cmdQueueBuffer
    }
}
import Metal
import simd
import QuartzCore

public class AlloyRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelineState: MTLComputePipelineState!
    
    // 公开给外界访问的低分辨率纹理（超分用，暂时关闭）
    public var lowResTexture: MTLTexture?
    
    // 🔥 恢复默认 1.0（直接渲染到屏幕），暂时关闭超分
    public var renderScale: Float = 1.0
    
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
    
    // 🔥 接收 drawable 和 texture，根据 renderScale 决定渲染目标
    public func render(drawable: CAMetalDrawable, texture: MTLTexture, rawCommands: [UInt32]) -> MTLCommandBuffer? {
        let drawableTexture = drawable.texture
        guard !rawCommands.isEmpty else { return nil }
        
        let fullWidth = drawableTexture.width
        let fullHeight = drawableTexture.height
        
        // 🔥 根据 renderScale 决定输出纹理
        let outputTexture: MTLTexture
        if renderScale < 1.0 {
            let lowWidth = Int(Float(fullWidth) * renderScale)
            let lowHeight = Int(Float(fullHeight) * renderScale)
            if lowResTexture == nil || lowResTexture!.width != lowWidth || lowResTexture!.height != lowHeight {
                let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: lowWidth, height: lowHeight, mipmapped: false)
                desc.usage = [.shaderRead, .shaderWrite]
                lowResTexture = device.makeTexture(descriptor: desc)
            }
            outputTexture = lowResTexture!
        } else {
            outputTexture = drawableTexture
        }
        
        let commandBuffer = device.makeBuffer(bytes: rawCommands,
                                              length: rawCommands.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        
        var commandCount = UInt32(rawCommands.count)
        
        guard let cmdQueueBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdQueueBuffer.makeComputeCommandEncoder() else { return nil }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(commandBuffer, offset: 0, index: 0)
        encoder.setBytes(&commandCount, length: MemoryLayout<UInt32>.size, index: 1)
        
        encoder.setTexture(outputTexture, index: 0)
        encoder.setTexture(texture, index: 1)
        
        let threadsPerThreadgroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let threadsPerGrid = MTLSize(width: outputTexture.width, height: outputTexture.height, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return cmdQueueBuffer
    }
}
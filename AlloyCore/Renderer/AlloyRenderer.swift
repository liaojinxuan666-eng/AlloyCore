import Metal
import simd
import QuartzCore

public class AlloyRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var pipelineState: MTLComputePipelineState!
    
    // 🔥 改为高分辨率纹理（SSAA 用）
    public var highResTexture: MTLTexture?
    public var renderScale: Float = 2.0 // 2倍分辨率渲染！
    
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
    
    // 🔥 直接渲染到高分辨率纹理
    public func render(drawable: CAMetalDrawable, texture: MTLTexture, rawCommands: [UInt32]) -> MTLCommandBuffer? {
        let drawableTexture = drawable.texture
        guard !rawCommands.isEmpty else { return nil }
        
        let fullWidth = drawableTexture.width
        let fullHeight = drawableTexture.height
        
        // SSAA: 渲染到 2倍尺寸纹理
        let ssaaWidth = Int(Float(fullWidth) * renderScale)
        let ssaaHeight = Int(Float(fullHeight) * renderScale)
        
        if highResTexture == nil || highResTexture!.width != ssaaWidth || highResTexture!.height != ssaaHeight {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: ssaaWidth, height: ssaaHeight, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            highResTexture = device.makeTexture(descriptor: desc)
        }
        
        guard let ssaaTex = highResTexture else { return nil }
        
        let commandBuffer = device.makeBuffer(bytes: rawCommands,
                                              length: rawCommands.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        
        var commandCount = UInt32(rawCommands.count)
        
        guard let cmdQueueBuffer = commandQueue.makeCommandBuffer(),
              let encoder = cmdQueueBuffer.makeComputeCommandEncoder() else { return nil }
        
        encoder.setComputePipelineState(pipelineState)
        encoder.setBuffer(commandBuffer, offset: 0, index: 0)
        encoder.setBytes(&commandCount, length: MemoryLayout<UInt32>.size, index: 1)
        
        encoder.setTexture(ssaaTex, index: 0) // 写入高分辨率纹理
        encoder.setTexture(texture, index: 1)
        
        let threadsPerThreadgroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let threadsPerGrid = MTLSize(width: ssaaWidth, height: ssaaHeight, depth: 1)
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
        
        return cmdQueueBuffer
    }
}
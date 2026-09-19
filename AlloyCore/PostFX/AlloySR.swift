import Metal
import simd

public class AlloySR {
    let device: MTLDevice
    var upscalePipelineState: MTLComputePipelineState!
    
    public init?(device: MTLDevice) {
        self.device = device
        
        let bundle = Bundle(for: AlloySR.self) // 🔥 注意这里拿的是 PostFX 的 Bundle
        guard let library = try? device.makeDefaultLibrary(bundle: bundle),
              let kernel = library.makeFunction(name: "upscale_pass") else {
            print("AlloySR 初始化失败：无法加载 PostFX 的 Metal 库")
            return nil
        }
        
        do {
            upscalePipelineState = try device.makeComputePipelineState(function: kernel)
        } catch {
            print("创建 AlloySR Pipeline 失败: \(error)")
            return nil
        }
    }
    
    // 🔥 超分核心：把低分辨率纹理放大到全屏
    public func upscale(lowRes: MTLTexture, highRes: MTLTexture, commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(upscalePipelineState)
        encoder.setTexture(lowRes, index: 0)
        encoder.setTexture(highRes, index: 1)
        
        let tileSize = 16
        let threadsPerThreadgroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let threadsPerGrid = MTLSize(width: highRes.width, height: highRes.height, depth: 1)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}
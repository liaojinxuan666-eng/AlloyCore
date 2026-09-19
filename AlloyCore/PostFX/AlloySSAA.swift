import Metal
import simd

public class AlloySSAA {
    let device: MTLDevice
    var downsamplePipelineState: MTLComputePipelineState!
    
    public init?(device: MTLDevice) {
        self.device = device
        
        let bundle = Bundle(for: AlloySSAA.self)
        guard let library = try? device.makeDefaultLibrary(bundle: bundle),
              let kernel = library.makeFunction(name: "downsample_pass") else {
            print("AlloySSAA 初始化失败：无法加载 Metal 库")
            return nil
        }
        
        do {
            downsamplePipelineState = try device.makeComputePipelineState(function: kernel)
        } catch {
            print("创建 AlloySSAA Pipeline 失败: \(error)")
            return nil
        }
    }
    
    // 把高分辨率纹理降采样到输出纹理
    public func downsample(highRes: MTLTexture, lowRes: MTLTexture, commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(downsamplePipelineState)
        encoder.setTexture(highRes, index: 0)
        encoder.setTexture(lowRes, index: 1)
        
        let tileSize = 16
        let threadsPerThreadgroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let threadsPerGrid = MTLSize(width: lowRes.width, height: lowRes.height, depth: 1)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}
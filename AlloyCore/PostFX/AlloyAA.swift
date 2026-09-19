import Metal
import simd

public class AlloyAA {
    let device: MTLDevice
    var aaPipelineState: MTLComputePipelineState!
    
    public init?(device: MTLDevice) {
        self.device = device
        
        let bundle = Bundle(for: AlloyAA.self)
        guard let library = try? device.makeDefaultLibrary(bundle: bundle),
              let kernel = library.makeFunction(name: "aa_pass") else {
            print("AlloyAA 初始化失败：无法加载 Metal 库")
            return nil
        }
        
        do {
            aaPipelineState = try device.makeComputePipelineState(function: kernel)
        } catch {
            print("创建 AlloyAA Pipeline 失败: \(error)")
            return nil
        }
    }
    
    // 接收一个纹理，在 CommandBuffer 里对它进行抗锯齿处理
    public func applyAA(texture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        
        encoder.setComputePipelineState(aaPipelineState)
        encoder.setTexture(texture, index: 0)
        encoder.setTexture(texture, index: 1) // 原地读写，注意只是演示，实际工程可能会用双缓冲
        
        let tileSize = 16
        let threadsPerThreadgroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
        let threadsPerGrid = MTLSize(width: texture.width, height: texture.height, depth: 1)
        
        encoder.dispatchThreads(threadsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        encoder.endEncoding()
    }
}
import Metal
import simd
import QuartzCore

public class AlloyRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var geometryPipeline: MTLComputePipelineState!
    var rasterPipeline: MTLComputePipelineState!
    var upscalePipeline: MTLComputePipelineState!
    let tileSize: Int = 16
    
    // 半分辨率渲染
    public var renderScale: Float = 0.5
    var lowResTexture: MTLTexture?
    
    var cachedVertexBuffer: MTLBuffer?
    var cachedScreenVertexBuffer: MTLBuffer?
    var cachedIndexBuffer: MTLBuffer?
    var cachedVertexCount: Int = 0
    
    public init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = commandQueue
        
        let bundle = Bundle(for: AlloyRenderer.self)
        guard let library = try? device.makeDefaultLibrary(bundle: bundle),
              let geoKernel = library.makeFunction(name: "geometry_pass"),
              let rasKernel = library.makeFunction(name: "process_commands"),
              let upKernel = library.makeFunction(name: "upscale_pass") else {
            print("AlloyCore 初始化失败：无法加载 Metal 库")
            return nil
        }
        
        do {
            geometryPipeline = try device.makeComputePipelineState(function: geoKernel)
            rasterPipeline = try device.makeComputePipelineState(function: rasKernel)
            upscalePipeline = try device.makeComputePipelineState(function: upKernel)
        } catch {
            print("创建 Pipeline 失败: \(error)")
            return nil
        }
    }
    
    public func uploadGeometry(vertexData: [Float], indexData: [UInt32]) {
        cachedVertexBuffer = device.makeBuffer(bytes: vertexData,
                                               length: vertexData.count * MemoryLayout<Float>.size,
                                               options: .storageModeShared)
        cachedScreenVertexBuffer = device.makeBuffer(length: vertexData.count * MemoryLayout<Float>.size,
                                                     options: .storageModePrivate)
        cachedIndexBuffer = device.makeBuffer(bytes: indexData,
                                              length: indexData.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        cachedVertexCount = vertexData.count / 12
    }
    
    private func extractTransform(from rawCommands: [UInt32]) -> simd_float4x4 {
        var matrix: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        var i = 0
        while i < rawCommands.count {
            let op = rawCommands[i]
            if op == 0x01 { i += 4 }
            else if op == 0x02 { i += 5 }
            else if op == 0x03 { i += 5 }
            else if op == 0x04 { i += 3 }
            else if op == 0x06 {
                for k in 0..<16 {
                    matrix[k] = Float(bitPattern: rawCommands[i + 1 + k])
                }
                i += 17
            } else { break }
        }
        return simd_float4x4(
            SIMD4<Float>(matrix[0], matrix[1], matrix[2], matrix[3]),
            SIMD4<Float>(matrix[4], matrix[5], matrix[6], matrix[7]),
            SIMD4<Float>(matrix[8], matrix[9], matrix[10], matrix[11]),
            SIMD4<Float>(matrix[12], matrix[13], matrix[14], matrix[15])
        )
    }
    
    public func render(
        drawable: CAMetalDrawable,
        texture0: MTLTexture,
        texture1: MTLTexture,
        rawCommands: [UInt32]
    ) -> MTLCommandBuffer? {
        guard let inputVBO = cachedVertexBuffer,
              let outputVBO = cachedScreenVertexBuffer,
              let indexBuffer = cachedIndexBuffer,
              !rawCommands.isEmpty else { return nil }
        
        let drawableTexture = drawable.texture
        let fullWidth = drawableTexture.width
        let fullHeight = drawableTexture.height
        
        // 半分辨率纹理
        let lowWidth = Int(Float(fullWidth) * renderScale)
        let lowHeight = Int(Float(fullHeight) * renderScale)
        
        if lowResTexture == nil || lowResTexture!.width != lowWidth || lowResTexture!.height != lowHeight {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: lowWidth, height: lowHeight, mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            lowResTexture = device.makeTexture(descriptor: desc)
        }
        
        guard let lowRes = lowResTexture else { return nil }
        
        var matrix = extractTransform(from: rawCommands)
        var screenSize = SIMD2<Float>(Float(lowWidth), Float(lowHeight))
        var vertexCount = UInt32(cachedVertexCount)
        
        let commandBuffer = device.makeBuffer(bytes: rawCommands,
                                              length: rawCommands.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        var commandCount = UInt32(rawCommands.count)
        
        guard let cmdQueueBuffer = commandQueue.makeCommandBuffer() else { return nil }
        
        // Pass 1: 几何处理（按低分辨率屏幕尺寸做投影）
        if let geoEncoder = cmdQueueBuffer.makeComputeCommandEncoder() {
            geoEncoder.setComputePipelineState(geometryPipeline)
            geoEncoder.setBuffer(inputVBO, offset: 0, index: 0)
            geoEncoder.setBuffer(outputVBO, offset: 0, index: 1)
            geoEncoder.setBytes(&vertexCount, length: MemoryLayout<UInt32>.size, index: 2)
            geoEncoder.setBytes(&matrix, length: MemoryLayout<simd_float4x4>.size, index: 3)
            geoEncoder.setBytes(&screenSize, length: MemoryLayout<SIMD2<Float>>.size, index: 4)
            
            let w = geometryPipeline.threadExecutionWidth
            let geoThreads = MTLSize(width: w, height: 1, depth: 1)
            let geoGrid = MTLSize(width: cachedVertexCount, height: 1, depth: 1)
            geoEncoder.dispatchThreads(geoGrid, threadsPerThreadgroup: geoThreads)
            geoEncoder.endEncoding()
        }
        
        // Pass 2: 光栅化到低分辨率纹理
        if let rasEncoder = cmdQueueBuffer.makeComputeCommandEncoder() {
            rasEncoder.setComputePipelineState(rasterPipeline)
            rasEncoder.setBuffer(commandBuffer, offset: 0, index: 0)
            rasEncoder.setBytes(&commandCount, length: MemoryLayout<UInt32>.size, index: 1)
            rasEncoder.setBuffer(outputVBO, offset: 0, index: 2)
            rasEncoder.setBuffer(indexBuffer, offset: 0, index: 3)
            rasEncoder.setTexture(lowRes, index: 0)
            rasEncoder.setTexture(texture0, index: 1)
            rasEncoder.setTexture(texture1, index: 2)
            
            let threadsPerGroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
            let grid = MTLSize(width: lowWidth, height: lowHeight, depth: 1)
            rasEncoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerGroup)
            rasEncoder.endEncoding()
        }
        
        // Pass 3: 放大到全屏
        if let upEncoder = cmdQueueBuffer.makeComputeCommandEncoder() {
            upEncoder.setComputePipelineState(upscalePipeline)
            upEncoder.setTexture(lowRes, index: 0)
            upEncoder.setTexture(drawableTexture, index: 1)
            
            let threadsPerGroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
            let grid = MTLSize(width: fullWidth, height: fullHeight, depth: 1)
            upEncoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerGroup)
            upEncoder.endEncoding()
        }
        
        return cmdQueueBuffer
    }
}
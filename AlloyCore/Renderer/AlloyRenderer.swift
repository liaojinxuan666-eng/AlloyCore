import Metal
import simd
import QuartzCore

public class AlloyRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var geometryPipeline: MTLComputePipelineState!
    var binningPipeline: MTLComputePipelineState!
    var rasterizePipeline: MTLComputePipelineState!
    var upscalePipeline: MTLComputePipelineState!
    
    let tileSize: Int = 16
    let maxTrianglesPerTile: Int = 256
    
    public var renderScale: Float = 0.5
    var lowResTexture: MTLTexture?
    
    var cachedVertexBuffer: MTLBuffer?
    var cachedScreenVertexBuffer: MTLBuffer?
    var cachedIndexBuffer: MTLBuffer?
    var cachedVertexCount: Int = 0
    var cachedTriangleCount: Int = 0
    
    var binCountsBuffer: MTLBuffer?
    var binDataBuffer: MTLBuffer?
    var binTileCountX: UInt32 = 0
    var binTileCountY: UInt32 = 0
    
    public init?() {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else { return nil }
        self.device = device
        self.commandQueue = commandQueue
        
        let bundle = Bundle(for: AlloyRenderer.self)
        guard let library = try? device.makeDefaultLibrary(bundle: bundle),
              let geoKernel = library.makeFunction(name: "geometry_pass"),
              let binKernel = library.makeFunction(name: "binning_pass"),
              let rasKernel = library.makeFunction(name: "rasterize_pass"),
              let upKernel = library.makeFunction(name: "upscale_pass") else {
            print("AlloyCore 初始化失败：无法加载 Metal 库")
            return nil
        }
        
        do {
            geometryPipeline = try device.makeComputePipelineState(function: geoKernel)
            binningPipeline = try device.makeComputePipelineState(function: binKernel)
            rasterizePipeline = try device.makeComputePipelineState(function: rasKernel)
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
        cachedTriangleCount = indexData.count / 3
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
              !rawCommands.isEmpty,
              cachedVertexCount > 0,
              cachedshTriangleCount > 0 else { return nil }
        
ader        let drawableTexture = drawable.texture
Write        let fullWidth = drawableTexture.width
        let]
 fullHeight = drawableTexture.height
        
        let low           Width = Int(Float(fullWidth) * render lowScale)
        let lowHeight = Int(Float(fullHeight) * renderScale)
        
        // 半分辨率纹理
        if lowResTexture == nil || lowResTexture!.width != lowWidth || lowResTexture!.height != lowHeight {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: lowWidth, height: lowHeight, mipmapped: false)
            desc.usage = [.shaderRead, .ResTexture = device.makeTexture(descriptor: desc)
        }
        guard let lowRes = lowResTexture else { return nil }
        
        // Bin buffer 分配（分辨率变化时重新分配）
        let tileCountX = UInt32((lowWidth + tileSize - 1) / tileSize)
        let tileCountY = UInt32((lowHeight + tileSize - 1) / tileSize)
        let numTiles = Int(tileCountX) * Int(tileCountY)
        
        if binCountsBuffer == nil || binTileCountX != tileCountX || binTileCountY != tileCountY {
            binCountsBuffer = device.makeBuffer(length: numTiles * MemoryLayout<UInt32>.size,
                                                options: .storageModePrivate)
            binDataBuffer = device.makeBuffer(length: numTiles * maxTrianglesPerTile * MemoryLayout<UInt32>.size,
                                              options: .storageModePrivate)
            binTileCountX = tileCountX
            binTileCountY = tileCountY
        }
        
        guard let binCounts = binCountsBuffer, let binData = binDataBuffer else { return nil }
        
        var matrix = extractTransform(from: rawCommands)
        var lowScreenSize = SIMD2<Float>(Float(lowWidth), Float(lowHeight))
        var vertexCount = UInt32(cachedVertexCount)
        var triangleCount = UInt32(cachedTriangleCount)
        var screenTileCounts = SIMD2<UInt32>(tileCountX, tileCountY)
        var screenTileCountX = tileCountX
        var depthTestEnabled: UInt32 = 1
        
        guard let cmdBuffer = commandQueue.makeCommandBuffer() else { return nil }
        
        // Pass 1: 几何变换
        if let geoEnc = cmdBuffer.makeComputeCommandEncoder() {
            geoEnc.setComputePipelineState(geometryPipeline)
            geoEnc.setBuffer(inputVBO, offset: 0, index: 0)
            geoEnc.setBuffer(outputVBO, offset: 0, index: 1)
            geoEnc.setBytes(&vertexCount, length: MemoryLayout<UInt32>.size, index: 2)
            geoEnc.setBytes(&matrix, length: MemoryLayout<simd_float4x4>.size, index: 3)
            geoEnc.setBytes(&lowScreenSize, length: MemoryLayout<SIMD2<Float>>.size, index: 4)
            
            let w = geometryPipeline.threadExecutionWidth
            geoEnc.dispatchThreads(MTLSize(width: cachedVertexCount, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            geoEnc.endEncoding()
        }
        
        // Pass 2: 清零 bin counts
        if let blit = cmdBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: binCounts, range: 0..<binCounts.length, value: 0)
            blit.endEncoding()
        }
        
        // Pass 3: 三角形分箱
        if let binEnc = cmdBuffer.makeComputeCommandEncoder() {
            binEnc.setComputePipelineState(binningPipeline)
            binEnc.setBuffer(outputVBO, offset: 0, index: 0)
            binEnc.setBuffer(indexBuffer, offset: 0, index: 1)
            binEnc.setBuffer(binCounts, offset: 0, index: 2)
            binEnc.setBuffer(binData, offset: 0, index: 3)
            binEnc.setBytes(&triangleCount, length: MemoryLayout<UInt32>.size, index: 4)
            binEnc.setBytes(&screenTileCounts, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
            
            let w = binningPipeline.threadExecutionWidth
            binEnc.dispatchThreads(MTLSize(width: cachedTriangleCount, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            binEnc.endEncoding()
        }
        
        // Pass 4: 光栅化（每个 tile 只读自己的 bin）
        if let rasEnc = cmdBuffer.makeComputeCommandEncoder() {
            rasEnc.setComputePipelineState(rasterizePipeline)
            rasEnc.setBuffer(outputVBO, offset: 0, index: 0)
            rasEnc.setBuffer(indexBuffer, offset: 0, index: 1)
            rasEnc.setBuffer(binCounts, offset: 0, index: 2)
            rasEnc.setBuffer(binData, offset: 0, index: 3)
            rasEnc.setBytes(&screenTileCountX, length: MemoryLayout<UInt32>.size, index: 4)
            rasEnc.setBytes(&depthTestEnabled, length: MemoryLayout<UInt32>.size, index: 5)
            rasEnc.setTexture(lowRes, index: 0)
            rasEnc.setTexture(texture0, index: 1)
            rasEnc.setTexture(texture1, index: 2)
            
            let threadsPerGroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
            let tileGroups = MTLSize(width: Int(tileCountX), height: Int(tileCountY), depth: 1)
            rasEnc.dispatchThreadgroups(tileGroups, threadsPerThreadgroup: threadsPerGroup)
            rasEnc.endEncoding()
        }
        
        // Pass 5: 放大到全屏
        if let upEnc = cmdBuffer.makeComputeCommandEncoder() {
            upEnc.setComputePipelineState(upscalePipeline)
            upEnc.setTexture(lowRes, index: 0)
            upEnc.setTexture(drawableTexture, index: 1)
            
            let threadsPerGroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
            upEnc.dispatchThreads(MTLSize(width: fullWidth, height: fullHeight, depth: 1),
                                  threadsPerThreadgroup: threadsPerGroup)
            upEnc.endEncoding()
        }
        
        return cmdBuffer
    }
}
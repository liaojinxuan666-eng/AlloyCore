import Metal
import simd
import QuartzCore

public class AlloyRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var geometryPipeline: MTLComputePipelineState!
    var rasterPipeline: MTLComputePipelineState!
    let tileSize: Int = 16
    
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
              let rasKernel = library.makeFunction(name: "process_commands") else {
            print("AlloyCore 初始化失败：无法加载 Metal 库")
            return nil
        }
        
        do {
            geometryPipeline = try device.makeComputePipelineState(function: geoKernel)
            rasterPipeline = try device.makeComputePipelineState(function: rasKernel)
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
    
    // 从指令流里扫描最后一个 0x06 矩阵
    private func extractTransform(from rawCommands: [UInt32]) -> [Float] {
        var matrix: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
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
        return matrix
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
        
        let outputTexture = drawable.texture
        let screenWidth = Float(outputTexture.width)
        let screenHeight = Float(outputTexture.height)
        
        // 提取矩阵
        let matrixArray = extractTransform(from: rawCommands)
        var matrix = simd_float4x4(
            SIMD4<Float>(matrixArray[0], matrixArray[1], matrixArray[2], matrixArray[3]),
            SIMD4<Float>(matrixArray[4], matrixArray[5], matrixArray[6], matrixArray[7]),
            SIMD4<Float>(matrixArray[8], matrixArray[9], matrixArray[10], matrixArray[11]),
            SIMD4<Float>(matrixArray[12], matrixArray[13], matrixArray[14], matrixArray[15])
        )
        var screenSize = SIMD2<Float>(screenWidth, screenHeight)
        var vertexCount = UInt32(cachedVertexCount)
        
        // 上传指令流
        let commandBuffer = device.makeBuffer(bytes: rawCommands,
                                              length: rawCommands.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        var commandCount = UInt32(rawCommands.count)
        
        guard let cmdQueueBuffer = commandQueue.makeCommandBuffer() else { return nil }
        
        // Pass 1: geometry_pass
        if let geoEncoder = cmdQueueBuffer.makeComputeCommandEncoder() {
            geoEncoder.setComputePipelineState(geometryPipeline)
            geoEncoder.setBuffer(inputVBO, offset: 0, index: 0)
            geoEncoder.setBuffer(outputVBO, offset: 0, index: 1)
            geoEncoder.setBytes(&vertexCount, length: MemoryLayout<UInt32>.size, index: 2)
            geoEncoder.setBytes(&matrix, length: MemoryLayout<simd_float4x4>.size, index: 3)
            geoEncoder.setBytes(&screenSize, length: MemoryLayout<SIMD2<Float>>.size, index: 4)
            
            let w = geometryPipeline.threadExecutionWidth
            let geoThreadsPerGroup = MTLSize(width: w, height: 1, depth: 1)
            let geoGrid = MTLSize(width: cachedVertexCount, height: 1, depth: 1)
            geoEncoder.dispatchThreads(geoGrid, threadsPerThreadgroup: geoThreadsPerGroup)
            geoEncoder.endEncoding()
        }
        
        // Pass 2: process_commands (光栅化)
        if let rasEncoder = cmdQueueBuffer.makeComputeCommandEncoder() {
            rasEncoder.setComputePipelineState(rasterPipeline)
            rasEncoder.setBuffer(commandBuffer, offset: 0, index: 0)
            rasEncoder.setBytes(&commandCount, length: MemoryLayout<UInt32>.size, index: 1)
            rasEncoder.setBuffer(outputVBO, offset: 0, index: 2)
            rasEncoder.setBuffer(indexBuffer, offset: 0, index: 3)
            rasEncoder.setTexture(outputTexture, index: 0)
            rasEncoder.setTexture(texture0, index: 1)
            rasEncoder.setTexture(texture1, index: 2)
            
            let threadsPerGroup = MTLSize(width: tileSize, height: tileSize, depth: 1)
            let grid = MTLSize(width: outputTexture.width, height: outputTexture.height, depth: 1)
            rasEncoder.dispatchThreads(grid, threadsPerThreadgroup: threadsPerGroup)
            rasEncoder.endEncoding()
        }
        
        return cmdQueueBuffer
    }
}
import Metal
import simd
import QuartzCore

public class AlloyRenderer {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    var geometryPipeline: MTLComputePipelineState!
    var clipProjectPipeline: MTLComputePipelineState!
    var binningPipeline: MTLComputePipelineState!
    var rasterizePipeline: MTLComputePipelineState!
    var upscalePipeline: MTLComputePipelineState!
    let tileSize: Int = 16
    let maxTrianglesPerTile: Int = 256
    public var renderScale: Float = 0.5
    var lowResTexture: MTLTexture?
    var cachedVertexBuffer: MTLBuffer?
    var cachedClipSpaceBuffer: MTLBuffer?
    var cachedClipOutputVerts: MTLBuffer?
    var cachedClipOutputIndices: MTLBuffer?
    var cachedIndexBuffer: MTLBuffer?
    var cachedVertexCount: Int = 0
    var cachedTriangleCount: Int = 0
    var cachedMaxOutputTriangles: Int = 0
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
              let clipKernel = library.makeFunction(name: "clip_project_pass"),
              let binKernel = library.makeFunction(name: "binning_pass"),
              let rasKernel = library.makeFunction(name: "rasterize_pass"),
              let upKernel = library.makeFunction(name: "upscale_pass") else {
            return nil
        }
        do {
            geometryPipeline = try device.makeComputePipelineState(function: geoKernel)
            clipProjectPipeline = try device.makeComputePipelineState(function: clipKernel)
            binningPipeline = try device.makeComputePipelineState(function: binKernel)
            rasterizePipeline = try device.makeComputePipelineState(function: rasKernel)
            upscalePipeline = try device.makeComputePipelineState(function: upKernel)
        } catch {
            return nil
        }
    }

    public func uploadGeometry(vertexData: [Float], indexData: [UInt32]) {
        cachedVertexBuffer = device.makeBuffer(bytes: vertexData,
                                               length: vertexData.count * MemoryLayout<Float>.size,
                                               options: .storageModeShared)
        let vertexCount = vertexData.count / 12
        let triangleCount = indexData.count / 3
        cachedClipSpaceBuffer = device.makeBuffer(length: vertexCount * 13 * MemoryLayout<Float>.size,
                                                  options: .storageModePrivate)
        // 每个输入三角形最多 4 个输出顶点，最多 2 个输出三角形
        cachedMaxOutputTriangles = triangleCount * 2
        cachedClipOutputVerts = device.makeBuffer(length: triangleCount * 4 * 13 * MemoryLayout<Float>.size,
                                                  options: .storageModePrivate)
        cachedClipOutputIndices = device.makeBuffer(length: triangleCount * 2 * 3 * MemoryLayout<UInt32>.size,
                                                    options: .storageModePrivate)
        cachedIndexBuffer = device.makeBuffer(bytes: indexData,
                                              length: indexData.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        cachedVertexCount = vertexCount
        cachedTriangleCount = triangleCount
    }

    private func extractTransform(from rawCommands: [UInt32]) -> simd_float4x4 {
        var m: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        var i = 0
        while i < rawCommands.count {
            let op = rawCommands[i]
            if op == 0x01 { i += 4 }
            else if op == 0x02 { i += 5 }
            else if op == 0x03 { i += 5 }
            else if op == 0x04 { i += 3 }
            else if op == 0x06 {
                for k in 0..<16 { m[k] = Float(bitPattern: rawCommands[i + 1 + k]) }
                i += 17
            } else { break }
        }
        return simd_float4x4(
            SIMD4<Float>(m[0], m[1], m[2], m[3]),
            SIMD4<Float>(m[4], m[5], m[6], m[7]),
            SIMD4<Float>(m[8], m[9], m[10], m[11]),
            SIMD4<Float>(m[12], m[13], m[14], m[15])
        )
    }

    public func render(drawable: CAMetalDrawable,
                       texture0: MTLTexture,
                       texture1: MTLTexture,
                       rawCommands: [UInt32]) -> MTLCommandBuffer? {
        guard let inputVBO = cachedVertexBuffer,
              let clipSpaceBuf = cachedClipSpaceBuffer,
              let outVerts = cachedClipOutputVerts,
              let outIndices = cachedClipOutputIndices,
              let indexBuffer = cachedIndexBuffer,
              !rawCommands.isEmpty,
              cachedVertexCount > 0,
              cachedTriangleCount > 0 else { return nil }

        let drawableTexture = drawable.texture
        let fullWidth = drawableTexture.width
        let fullHeight = drawableTexture.height
        let lowWidth = Int(Float(fullWidth) * renderScale)
        let lowHeight = Int(Float(fullHeight) * renderScale)

        if lowResTexture == nil || lowResTexture!.width != lowWidth || lowResTexture!.height != lowHeight {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm,
                                                                width: lowWidth,
                                                                height: lowHeight,
                                                                mipmapped: false)
            desc.usage = [.shaderRead, .shaderWrite]
            lowResTexture = device.makeTexture(descriptor: desc)
        }
        guard let lowRes = lowResTexture else { return nil }

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
        var screenSize = SIMD2<Float>(Float(lowWidth), Float(lowHeight))
        var vertexCount = UInt32(cachedVertexCount)
        var triangleCount = UInt32(cachedTriangleCount)
        var maxOutputTriangles = UInt32(cachedMaxOutputTriangles)
        var screenTileCounts = SIMD2<UInt32>(tileCountX, tileCountY)
        var screenTileCountX = tileCountX

        var depthTestEnabled: UInt32 = 1
        var cullMode: UInt32 = 0
        var ci = 0
        while ci < rawCommands.count {
            let op = rawCommands[ci]
            if op == 0x01 { ci += 4 }
            else if op == 0x02 { ci += 5 }
            else if op == 0x03 {
                depthTestEnabled = (rawCommands[ci+1] == 0) ? 0 : 1
                cullMode = rawCommands[ci+2]
                ci += 5
            }
            else if op == 0x04 { ci += 3 }
            else if op == 0x06 { ci += 17 }
            else { break }
        }

        guard let cmdBuffer = commandQueue.makeCommandBuffer() else { return nil }

        // Pass 1: 模型空间 → 裁剪空间
        if let geoEnc = cmdBuffer.makeComputeCommandEncoder() {
            geoEnc.setComputePipelineState(geometryPipeline)
            geoEnc.setBuffer(inputVBO, offset: 0, index: 0)
            geoEnc.setBuffer(clipSpaceBuf, offset: 0, index: 1)
            geoEnc.setBytes(&vertexCount, length: MemoryLayout<UInt32>.size, index: 2)
            geoEnc.setBytes(&matrix, length: MemoryLayout<simd_float4x4>.size, index: 3)
            let w = geometryPipeline.threadExecutionWidth
            geoEnc.dispatchThreads(MTLSize(width: cachedVertexCount, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            geoEnc.endEncoding()
        }

        // Pass 2: 近平面裁剪 + 投影
        if let clipEnc = cmdBuffer.makeComputeCommandEncoder() {
            clipEnc.setComputePipelineState(clipProjectPipeline)
            clipEnc.setBuffer(clipSpaceBuf, offset: 0, index: 0)
            clipEnc.setBuffer(indexBuffer, offset: 0, index: 1)
            clipEnc.setBuffer(outVerts, offset: 0, index: 2)
            clipEnc.setBuffer(outIndices, offset: 0, index: 3)
            clipEnc.setBytes(&triangleCount, length: MemoryLayout<UInt32>.size, index: 4)
            clipEnc.setBytes(&screenSize, length: MemoryLayout<SIMD2<Float>>.size, index: 5)
            let w = clipProjectPipeline.threadExecutionWidth
            clipEnc.dispatchThreads(MTLSize(width: cachedTriangleCount, height: 1, depth: 1),
                                    threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            clipEnc.endEncoding()
        }

        // Pass 3: 清零 bin counts
        if let blit = cmdBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: binCounts, range: 0..<binCounts.length, value: 0)
            blit.endEncoding()
        }

        // Pass 4: 三角形分箱
        if let binEnc = cmdBuffer.makeComputeCommandEncoder() {
            binEnc.setComputePipelineState(binningPipeline)
            binEnc.setBuffer(outVerts, offset: 0, index: 0)
            binEnc.setBuffer(outIndices, offset: 0, index: 1)
            binEnc.setBuffer(binCounts, offset: 0, index: 2)
            binEnc.setBuffer(binData, offset: 0, index: 3)
            binEnc.setBytes(&maxOutputTriangles, length: MemoryLayout<UInt32>.size, index: 4)
            binEnc.setBytes(&screenTileCounts, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
            let w = binningPipeline.threadExecutionWidth
            binEnc.dispatchThreads(MTLSize(width: cachedMaxOutputTriangles, height: 1, depth: 1),
                                   threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            binEnc.endEncoding()
        }

        // Pass 5: 光栅化
        if let rasEnc = cmdBuffer.makeComputeCommandEncoder() {
            rasEnc.setComputePipelineState(rasterizePipeline)
            rasEnc.setBuffer(outVerts, offset: 0, index: 0)
            rasEnc.setBuffer(outIndices, offset: 0, index: 1)
            rasEnc.setBuffer(binCounts, offset: 0, index: 2)
            rasEnc.setBuffer(binData, offset: 0, index: 3)
            rasEnc.setBytes(&screenTileCountX, length: MemoryLayout<UInt32>.size, index: 4)
            rasEnc.setBytes(&depthTestEnabled, length: MemoryLayout<UInt32>.size, index: 5)
            rasEnc.setBytes(&cullMode, length: MemoryLayout<UInt32>.size, index: 6)
            rasEnc.setTexture(lowRes, index: 0)
            rasEnc.setTexture(texture0, index: 1)
            rasEnc.setTexture(texture1, index: 2)
            let tg = MTLSize(width: tileSize, height: tileSize, depth: 1)
            let groups = MTLSize(width: Int(tileCountX), height: Int(tileCountY), depth: 1)
            rasEnc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
            rasEnc.endEncoding()
        }

        // Pass 6: 上采样到全屏
        if let upEnc = cmdBuffer.makeComputeCommandEncoder() {
            upEnc.setComputePipelineState(upscalePipeline)
            upEnc.setTexture(lowRes, index: 0)
            upEnc.setTexture(drawableTexture, index: 1)
            let tg = MTLSize(width: tileSize, height: tileSize, depth: 1)
            upEnc.dispatchThreads(MTLSize(width: fullWidth, height: fullHeight, depth: 1),
                                  threadsPerThreadgroup: tg)
            upEnc.endEncoding()
        }

        return cmdBuffer
    }
}
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
              let g = library.makeFunction(name: "geometry_pass"),
              let c = library.makeFunction(name: "clip_project_pass"),
              let b = library.makeFunction(name: "binning_pass"),
              let r = library.makeFunction(name: "rasterize_pass"),
              let u = library.makeFunction(name: "upscale_pass") else {
            return nil
        }
        do {
            geometryPipeline = try device.makeComputePipelineState(function: g)
            clipProjectPipeline = try device.makeComputePipelineState(function: c)
            binningPipeline = try device.makeComputePipelineState(function: b)
            rasterizePipeline = try device.makeComputePipelineState(function: r)
            upscalePipeline = try device.makeComputePipelineState(function: u)
        } catch {
            return nil
        }
    }

    public func uploadGeometry(vertexData: [Float], indexData: [UInt32]) {
        cachedVertexBuffer = device.makeBuffer(bytes: vertexData,
                                               length: vertexData.count * MemoryLayout<Float>.size,
                                               options: .storageModeShared)
        let vc = vertexData.count / 12
        let tc = indexData.count / 3
        cachedClipSpaceBuffer = device.makeBuffer(length: vc * 13 * MemoryLayout<Float>.size,
                                                  options: .storageModePrivate)
        cachedClipOutputVerts = device.makeBuffer(length: tc * 4 * 13 * MemoryLayout<Float>.size,
                                                  options: .storageModePrivate)
        cachedClipOutputIndices = device.makeBuffer(length: tc * 2 * 3 * MemoryLayout<UInt32>.size,
                                                    options: .storageModePrivate)
        cachedIndexBuffer = device.makeBuffer(bytes: indexData,
                                              length: indexData.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        cachedVertexCount = vc
        cachedTriangleCount = tc
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
            else if op == 0x07 { i += 2 }
            else if op == 0x08 { i += 2 }
            else if op == 0x06 {
                for k in 0..<16 { m[k] = Float(bitPattern: rawCommands[i + 1 + k]) }
                i += 17
            }
            else if op == 0x09, i + 3 <= rawCommands.count { i += 3 + Int(rawCommands[i + 2]) }
            else if op == 0x0A, i + 3 <= rawCommands.count { i += 3 + Int(rawCommands[i + 2]) }
            else { break }
        }
        return simd_float4x4(
            SIMD4<Float>(m[0], m[1], m[2], m[3]),
            SIMD4<Float>(m[4], m[5], m[6], m[7]),
            SIMD4<Float>(m[8], m[9], m[10], m[11]),
            SIMD4<Float>(m[12], m[13], m[14], m[15])
        )
    }

    public func render(drawable: CAMetalDrawable,
                       textures: [MTLTexture?],
                       rawCommands: [UInt32]) -> MTLCommandBuffer? {
        guard let inputVBO = cachedVertexBuffer,
              let clipSpaceBuf = cachedClipSpaceBuffer,
              let outVerts = cachedClipOutputVerts,
              let outIndices = cachedClipOutputIndices,
              let indexBuffer = cachedIndexBuffer else {
            AlloyLog.log("render skip: buffer nil")
            return nil
        }
        guard !rawCommands.isEmpty else {
            AlloyLog.log("render skip: cmds empty")
            return nil
        }
        guard cachedVertexCount > 0, cachedTriangleCount > 0 else {
            AlloyLog.log("render skip: vc=\(cachedVertexCount) tc=\(cachedTriangleCount)")
            return nil
        }
        guard !textures.isEmpty, let tex0 = textures[0] else {
            AlloyLog.log("render skip: tex=\(textures.count)")
            return nil
        }

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
        var screenTileCounts = SIMD2<UInt32>(tileCountX, tileCountY)
        var screenTileCountX = tileCountX
        var depthTestEnabled: UInt32 = 1
        var cullMode: UInt32 = 0

        var ci = 0
        while ci < rawCommands.count {
            let op = rawCommands[ci]
            if op == 0x01 {
                ci += 4
            } else if op == 0x02 {
                ci += 5
            } else if op == 0x03 {
                depthTestEnabled = (rawCommands[ci+1] == 0) ? 0 : 1
                cullMode = rawCommands[ci+2]
                ci += 5
            } else if op == 0x04 {
                ci += 3
            } else if op == 0x06 {
                ci += 17
            } else if op == 0x07 {
                ci += 2
            } else if op == 0x08 {
                ci += 2
            } else if op == 0x09 {
                 guard ci + 3 <= rawCommands.count else { break }
                 let poolOffset = Int(rawCommands[ci + 1])
                 let count      = Int(rawCommands[ci + 2])
                 guard count >= 0, ci + 3 + count <= rawCommands.count else { break }
                 AlloyLog.log("rx0x09 off=\(poolOffset) n=\(count)")
                let capacity   = inputVBO.length / MemoryLayout<Float>.size
                if poolOffset >= 0, poolOffset + count <= capacity {
                    let ptr = inputVBO.contents().bindMemory(to: Float.self, capacity: capacity)
                    for k in 0..<count {
                        ptr[poolOffset + k] = Float(bitPattern: rawCommands[ci + 3 + k])
                    }
                }
                ci += 3 + count
            } else if op == 0x0A {
                 guard ci + 3 <= rawCommands.count else { break }
                 let poolOffset = Int(rawCommands[ci + 1])
                 let count      = Int(rawCommands[ci + 2])
                 guard count >= 0, ci + 3 + count <= rawCommands.count else { break }
                 AlloyLog.log("rx0x0A off=\(poolOffset) n=\(count)")
                 let capacity   = indexBuffer.length / MemoryLayout<UInt32>.size
                 if poolOffset >= 0, poolOffset + count <= capacity {
                    let ptr = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: capacity)
                    for k in 0..<count {
                        ptr[poolOffset + k] = rawCommands[ci + 3 + k]
                    }
                }
                ci += 3 + count
            } else {
                break
            }
        }

        guard let cmdBuffer = commandQueue.makeCommandBuffer() else { return nil }

        if let enc = cmdBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(geometryPipeline)
            enc.setBuffer(inputVBO, offset: 0, index: 0)
            enc.setBuffer(clipSpaceBuf, offset: 0, index: 1)
            enc.setBytes(&vertexCount, length: MemoryLayout<UInt32>.size, index: 2)
            enc.setBytes(&matrix, length: MemoryLayout<simd_float4x4>.size, index: 3)
            let w = geometryPipeline.threadExecutionWidth
            enc.dispatchThreads(MTLSize(width: cachedVertexCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
        }

        if let enc = cmdBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(clipProjectPipeline)
            enc.setBuffer(clipSpaceBuf, offset: 0, index: 0)
            enc.setBuffer(indexBuffer, offset: 0, index: 1)
            enc.setBuffer(outVerts, offset: 0, index: 2)
            enc.setBuffer(outIndices, offset: 0, index: 3)
            enc.setBytes(&triangleCount, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&screenSize, length: MemoryLayout<SIMD2<Float>>.size, index: 5)
            let w = clipProjectPipeline.threadExecutionWidth
            enc.dispatchThreads(MTLSize(width: cachedTriangleCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
        }

        if let blit = cmdBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: binCounts, range: 0..<binCounts.length, value: 0)
            blit.endEncoding()
        }

        var totalOutputSlots = UInt32(cachedTriangleCount * 2)
        if let enc = cmdBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(binningPipeline)
            enc.setBuffer(outVerts, offset: 0, index: 0)
            enc.setBuffer(outIndices, offset: 0, index: 1)
            enc.setBuffer(binCounts, offset: 0, index: 2)
            enc.setBuffer(binData, offset: 0, index: 3)
            enc.setBytes(&totalOutputSlots, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&screenTileCounts, length: MemoryLayout<SIMD2<UInt32>>.size, index: 5)
            let w = binningPipeline.threadExecutionWidth
            let slotCount = cachedTriangleCount * 2
            enc.dispatchThreads(MTLSize(width: slotCount, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: 1, depth: 1))
            enc.endEncoding()
        }

        if let enc = cmdBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(rasterizePipeline)
            enc.setBuffer(outVerts, offset: 0, index: 0)
            enc.setBuffer(outIndices, offset: 0, index: 1)
            enc.setBuffer(binCounts, offset: 0, index: 2)
            enc.setBuffer(binData, offset: 0, index: 3)
            enc.setBytes(&screenTileCountX, length: MemoryLayout<UInt32>.size, index: 4)
            enc.setBytes(&depthTestEnabled, length: MemoryLayout<UInt32>.size, index: 5)
            enc.setBytes(&cullMode, length: MemoryLayout<UInt32>.size, index: 6)
            enc.setTexture(lowRes, index: 0)
            enc.setTexture(tex0, index: 1)
            let tg = MTLSize(width: tileSize, height: tileSize, depth: 1)
            let groups = MTLSize(width: Int(tileCountX), height: Int(tileCountY), depth: 1)
            enc.dispatchThreadgroups(groups, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        if let enc = cmdBuffer.makeComputeCommandEncoder() {
            enc.setComputePipelineState(upscalePipeline)
            enc.setTexture(lowRes, index: 0)
            enc.setTexture(drawableTexture, index: 1)
            let tg = MTLSize(width: tileSize, height: tileSize, depth: 1)
            let grid = MTLSize(width: fullWidth, height: fullHeight, depth: 1)
            enc.dispatchThreads(grid, threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        return cmdBuffer
    }
}
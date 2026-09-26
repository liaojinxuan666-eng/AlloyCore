import Metal
import simd
import QuartzCore

public struct AlloyPassTiming {
    public let name: String
    public let ms: Double
    public init(name: String, ms: Double) {
        self.name = name
        self.ms = ms
    }
}

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
    var cachedTriTexIDs: MTLBuffer?
    var cachedVertexOriginBuffer: MTLBuffer?
    var cachedVertexCount: Int = 0
    var cachedPoolVersion: UInt32 = 0xFFFFFFFF
    var cachedTriangleCount: Int = 0
    var binCountsBuffer: MTLBuffer?
    var binDataBuffer: MTLBuffer?
    var binTileCountX: UInt32 = 0
    var binTileCountY: UInt32 = 0
    let library: MTLLibrary
    private var computePipelineCache: [String: MTLComputePipelineState] = [:]
    private var currentComputeHandle: UInt32 = 0xFFFFFFFF
    private var pendingDispatches: [(handle: UInt32, groups: SIMD3<UInt32>, bindings: [(slot: Int, floatOffset: Int)])] = []
    private var pendingComputeBindings: [(slot: Int, floatOffset: Int)] = []
    public private(set) var passTimings: [AlloyPassTiming] = []

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
        self.library = library
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
        cachedVertexOriginBuffer = device.makeBuffer(bytes: vertexData,
                                                     length: vertexData.count * MemoryLayout<Float>.size,
                                                     options: .storageModeShared)
        cachedIndexBuffer = device.makeBuffer(bytes: indexData,
                                              length: indexData.count * MemoryLayout<UInt32>.size,
                                              options: .storageModeShared)
        cachedTriTexIDs = device.makeBuffer(length: tc * MemoryLayout<UInt32>.size,
                                                options: .storageModeShared)
        cachedVertexCount = vc
        cachedTriangleCount = tc
    }

    private func extractTransform(from rawCommands: [UInt32]) -> simd_float4x4 {
        var m: [Float] = [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        var i = 0
        while i < rawCommands.count {
            guard let len = AlloyOpcodeLength.of(rawCommands, at: i) else { break }
            if rawCommands[i] == AlloyOpcode.setTransform.rawValue {
                for k in 0..<16 { m[k] = Float(bitPattern: rawCommands[i + 1 + k]) }
            }
            i += len
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
                       computePipelines: [AlloyComputePipelineDescriptor?],
                       computeTime: Float,
                       rawCommands: [UInt32]) -> MTLCommandBuffer? {
        guard let inputVBO = cachedVertexBuffer,
              let clipSpaceBuf = cachedClipSpaceBuffer,
              let outVerts = cachedClipOutputVerts,
              let outIndices = cachedClipOutputIndices,
              let indexBuffer = cachedIndexBuffer,
              let triTexIDs = cachedTriTexIDs else {
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
            guard let op = AlloyOpcode(rawValue: rawCommands[ci]) else { break }
            guard let len = AlloyOpcodeLength.of(rawCommands, at: ci) else { break }

            switch op {
            case .drawIndexed:
                let globalStart = Int(rawCommands[ci + 1])
                let count       = Int(rawCommands[ci + 2])
                let tid         = rawCommands[ci + 3]
                if count > 0 {
                    let triStart = globalStart / 3
                    let triCount = count / 3
                    let cap = triTexIDs.length / MemoryLayout<UInt32>.size
                    if triStart >= 0 && triStart + triCount <= cap {
                        let ptr = triTexIDs.contents().bindMemory(to: UInt32.self, capacity: cap)
                        for t in 0..<triCount { ptr[triStart + t] = tid }
                    }
                }

            case .bindPipeline:
                depthTestEnabled = (rawCommands[ci + 1] == 0) ? 0 : 1
                cullMode         = rawCommands[ci + 2]

            case .updateVertexBuffer:
                let poolOffset = Int(rawCommands[ci + 1])
                let count      = Int(rawCommands[ci + 2])
                AlloyLog.log("rx0x09 off=\(poolOffset) n=\(count)")
                let capacity   = inputVBO.length / MemoryLayout<Float>.size
                if poolOffset >= 0, poolOffset + count <= capacity {
                    let ptr = inputVBO.contents().bindMemory(to: Float.self, capacity: capacity)
                    for k in 0..<count {
                        ptr[poolOffset + k] = Float(bitPattern: rawCommands[ci + 3 + k])
                    }
                }

            case .updateIndexBuffer:
                let poolOffset = Int(rawCommands[ci + 1])
                let count      = Int(rawCommands[ci + 2])
                AlloyLog.log("rx0x0A off=\(poolOffset) n=\(count)")
                let capacity   = indexBuffer.length / MemoryLayout<UInt32>.size
                if poolOffset >= 0, poolOffset + count <= capacity {
                    let ptr = indexBuffer.contents().bindMemory(to: UInt32.self, capacity: capacity)
                    for k in 0..<count {
                        ptr[poolOffset + k] = rawCommands[ci + 3 + k]
                    }
                }

            case .bindComputePipeline:
                currentComputeHandle = rawCommands[ci + 1]

            case .bindComputeVertexPool:
                let slot = Int(rawCommands[ci + 1])
                let poolOffsetFloats = Int(rawCommands[ci + 2])
                let byteOffsetFloats = Int(rawCommands[ci + 3])
                pendingComputeBindings.append((slot: slot, floatOffset: poolOffsetFloats + byteOffsetFloats))

           case .computeDispatch:
                let gx = rawCommands[ci + 1]
                let gy = rawCommands[ci + 2]
                let gz = rawCommands[ci + 3]
                pendingDispatches.append((currentComputeHandle, SIMD3<UInt32>(gx, gy, gz), pendingComputeBindings))
                pendingComputeBindings.removeAll(keepingCapacity: true)

            case .clearColor, .setViewport, .setTransform,
                 .bindVertexBuffer, .bindIndexBuffer:
                break
            }

            ci += len
        }

        guard let cmdBuffer = commandQueue.makeCommandBuffer() else { return nil }
        passTimings.removeAll(keepingCapacity: true)
        
        let tCompute = CACurrentMediaTime()
        if !pendingDispatches.isEmpty {
            if let enc = cmdBuffer.makeComputeCommandEncoder() {
                for disp in pendingDispatches {
                    guard Int(disp.handle) < computePipelines.count,
                          let desc = computePipelines[Int(disp.handle)] else { continue }
                    let state: MTLComputePipelineState
                    if let cached = computePipelineCache[desc.shaderName] {
                        state = cached
                    } else {
                        guard let fn = library.makeFunction(name: desc.shaderName),
                              let newState = try? device.makeComputePipelineState(function: fn) else {
                            AlloyLog.log("compute: shader \(desc.shaderName) not found")
                            continue
                        }
                        computePipelineCache[desc.shaderName] = newState
                        state = newState
                    }
                    enc.setComputePipelineState(state)
                    let baseOffset: UInt32 = disp.bindings.first.map { UInt32($0.floatOffset) } ?? 0
                    let totalThreads: UInt32 = disp.groups.x
                    enc.setComputePipelineState(state)
                    enc.setBuffer(inputVBO, offset: 0, index: 0)
                    var t = computeTime
                    enc.setBytes(&t, length: MemoryLayout<Float>.size, index: 1)
                    var dt: Float = 0.02
                    enc.setBytes(&dt, length: MemoryLayout<Float>.size, index: 2)
                    var bo = baseOffset
                    enc.setBytes(&bo, length: MemoryLayout<UInt32>.size, index: 3)
                    var vc = totalThreads
                    enc.setBytes(&vc, length: MemoryLayout<UInt32>.size, index: 4)
                    if let origin = cachedVertexOriginBuffer {
                     enc.setBuffer(origin, offset: 0, index: 5)
                    }
                    let tg = MTLSize(width: Int(desc.threadsPerThreadgroup.x),
                                     height: Int(desc.threadsPerThreadgroup.y),
                                     depth: Int(desc.threadsPerThreadgroup.z))
                    let threads = MTLSize(width: Int(totalThreads), height: 1, depth: 1)
                    enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
                }
                enc.endEncoding()
            }
            pendingDispatches.removeAll(keepingCapacity: true)
            passTimings.append(AlloyPassTiming(name: "compute", ms: (CACurrentMediaTime() - tCompute) * 1000))
        }
        
        let tGeom = CACurrentMediaTime()
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
        passTimings.append(AlloyPassTiming(name: "geometry", ms: (CACurrentMediaTime() - tGeom) * 1000))

        if let enc = cmdBuffer.makeComputeCommandEncoder() {
           let tClip = CACurrentMediaTime()
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
        passTimings.append(AlloyPassTiming(name: "clip_project", ms: (CACurrentMediaTime() - tClip) * 1000))
        
        let tBinning = CACurrentMediaTime()
        if let blit = cmdBuffer.makeBlitCommandEncoder() {
            blit.fill(buffer: binCounts, range: 0..<binCounts.length, value: 0)
            blit.endEncoding()
        }
        passTimings.append(AlloyPassTiming(name: "binning", ms: (CACurrentMediaTime() - tBinning) * 1000))

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
            enc.setBuffer(triTexIDs, offset: 0, index: 7)
            if textures.count > 1, let tex1 = textures[1] {
                enc.setTexture(tex1, index: 2)
            } else {
                enc.setTexture(tex0, index: 2)
            }
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
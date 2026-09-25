import SwiftUI
import MetalKit
import AlloyCore

struct MetalView: UIViewRepresentable {
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60

        guard let renderer = AlloyRenderer() else { return view }
        context.coordinator.renderer = renderer

        let blue = TextureHelper.makeCheckerboardPixels(isRed: false)
        _ = context.coordinator.gal.createTexture(
            AlloyTextureDescriptor(width: blue.width, height: blue.height, data: blue.pixels))
        let red = TextureHelper.makeCheckerboardPixels(isRed: true)
        _ = context.coordinator.gal.createTexture(
            AlloyTextureDescriptor(width: red.width, height: red.height, data: red.pixels))

        context.coordinator.buildScene()
        context.coordinator.uploadScene(to: renderer)

        view.delegate = context.coordinator
        return view
    }

    func updateUIView(_ uiView: MTKView, context: Context) {}

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    class Coordinator: NSObject, MTKViewDelegate {
        var renderer: AlloyRenderer?
        var time: Float = 0.0
        var middleVerts: [Float] = []

        let gal = AlloyGAL()
        var pipelineHandle: AlloyPipelineHandle = 0
        var meshRanges: [(vbo: AlloyBufferHandle, ibo: AlloyBufferHandle, count: UInt32, texID: UInt32)] = []

        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

        func buildScene() {
            let offsets: [Float] = [-3.0, 0.0, 3.0]

            for (i, xOffset) in offsets.enumerated() {
                let grid = buildCubeGrid(offsetX: xOffset)
                let vbo = gal.createVertexBuffer(data: grid.vertices)
                let ibo = gal.createIndexBuffer(data: grid.indices, vertexHandle: vbo)
                meshRanges.append((vbo: vbo, ibo: ibo, count: UInt32(grid.indices.count), texID: UInt32(i % 2)))
                if i == 1 { middleVerts = grid.vertices }
            }

            var pso = AlloyPipelineDescriptor()
            pso.depthTestEnabled = true
            pso.cullMode = .back
            pipelineHandle = gal.createPipeline(pso)
        }

        func buildCubeGrid(offsetX: Float) -> (vertices: [Float], indices: [UInt32]) {
            var vertices: [Float] = []
            var indices: [UInt32] = []

            let s: Float = 0.15
            let gridN = 3
            let spacing: Float = 2.0 / Float(gridN)

            let baseVerts: [SIMD3<Float>] = [
                SIMD3<Float>(-s, -s, -s), SIMD3<Float>(s, -s, -s), SIMD3<Float>(s, s, -s), SIMD3<Float>(-s, s, -s),
                SIMD3<Float>(-s, -s, s),  SIMD3<Float>(s, -s, s),  SIMD3<Float>(s, s, s),  SIMD3<Float>(-s, s, s)
            ]

            let faceIdxList: [[Int]] = [
                [0, 1, 2, 3], [1, 5, 6, 2], [5, 4, 7, 6],
                [4, 0, 3, 7], [3, 2, 6, 7], [4, 5, 1, 0]
            ]

            let faceNormals: [SIMD3<Float>] = [
                SIMD3<Float>(0, 0, -1), SIMD3<Float>(1, 0, 0),
                SIMD3<Float>(0, 0, 1),  SIMD3<Float>(-1, 0, 0),
                SIMD3<Float>(0, 1, 0),  SIMD3<Float>(0, -1, 0)
            ]

            let uvs: [SIMD2<Float>] = [
                SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)
            ]

            for ix in 0..<gridN {
                for iy in 0..<gridN {
                    for iz in 0..<gridN {
                        let offset = SIMD3<Float>(
                            Float(ix) * spacing - 1.0 + spacing * 0.5 + offsetX,
                            Float(iy) * spacing - 1.0 + spacing * 0.5,
                            Float(iz) * spacing - 1.0 + spacing * 0.5
                        )

                        for (faceIdx, indices4) in faceIdxList.enumerated() {
                            let n = faceNormals[faceIdx]
                            let baseIndex = UInt32(vertices.count / 12)

                            for k in 0..<4 {
                                let p = baseVerts[indices4[k]] + offset
                                vertices.append(contentsOf: [
                                    p.x, p.y, p.z,
                                    1, 1, 1, 1,
                                    uvs[k].x, uvs[k].y,
                                    n.x, n.y, n.z
                                ])
                            }

                            indices.append(contentsOf: [baseIndex, baseIndex + 1, baseIndex + 2])
                            indices.append(contentsOf: [baseIndex, baseIndex + 2, baseIndex + 3])
                        }
                    }
                }
            }

            return (vertices, indices)
        }

        func uploadScene(to renderer: AlloyRenderer) {
            renderer.uploadGeometry(vertexData: gal.getVertexPool(),
                                    indexData: gal.getIndexPool())
            PerformanceMonitor.shared.setTriangleCount(gal.getIndexPool().count / 3)
        }

        func draw(in view: MTKView) {
            guard let renderer = renderer,
                  let drawable = view.currentDrawable else { return }

            let width = Float(drawable.texture.width)
            let height = Float(drawable.texture.height)
            time += 0.02

            let ax = time * 0.6
            let ay = time * 0.8
            let cosAY = cos(ay); let sinAY = sin(ay)
            let cosAX = cos(ax); let sinAX = sin(ax)

            let rotY = simd_float4x4(
                SIMD4<Float>(cosAY, 0, -sinAY, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(sinAY, 0, cosAY, 0),
                SIMD4<Float>(0, 0, 0, 1)
            )
            let rotX = simd_float4x4(
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, cosAX, sinAX, 0),
                SIMD4<Float>(0, -sinAX, cosAX, 0),
                SIMD4<Float>(0, 0, 0, 1)
            )

            let zNear: Float = 0.1
            let zFar: Float = 100.0
            let aspect = width / height
            let fovY: Float = 45.0 * Float.pi / 180.0
            let f = 1.0 / tan(fovY / 2.0)

            let persp = simd_float4x4(
                SIMD4<Float>(f / aspect, 0, 0, 0),
                SIMD4<Float>(0, f, 0, 0),
                SIMD4<Float>(0, 0, (zFar + zNear) / (zNear - zFar), -1),
                SIMD4<Float>(0, 0, (2 * zFar * zNear) / (zNear - zFar), 0)
            )

            let translation = simd_float4x4(
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, -8, 1)
            )

            let finalMatrix = persp * translation * rotX * rotY

            var matrixArray: [Float] = []
            for col in 0..<4 {
                for row in 0..<4 {
                    matrixArray.append(finalMatrix[col][row])
                }
            }

            gal.beginFrame()
            gal.clearColor(r: 0.1, g: 0.1, b: 0.15, a: 1.0)
            gal.setViewport(width: Int(width), height: Int(height))
            gal.bindPipeline(pipelineHandle)
            gal.setTransform(matrix: matrixArray)

            var modified = middleVerts
            let wobble = Float(sin(time * 5.0)) * 0.3
            for i in stride(from: 1, to: modified.count, by: 12) { modified[i] += wobble }
                gal.updateVertexBuffer(meshRanges[1].vbo, data: modified, offset: 0)

            for mesh in meshRanges {
                gal.drawIndexed(iboHandle: mesh.ibo,
                                indexCount: mesh.count,
                                firstIndex: 0,
                                textureID: mesh.texID)
            }

            gal.endFrame()

            if let cmdBuffer = gal.submit(to: renderer, drawable: drawable) {
                cmdBuffer.present(drawable)
                cmdBuffer.commit()
            }

            PerformanceMonitor.shared.markFrame()
        }
    }
}

struct PerformanceHUD: View {
    @StateObject var perf = PerformanceMonitor.shared

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(String(format: "FPS: %.1f", perf.fps))
                .font(.system(size: 14, weight: .bold, design: .monospaced))
            Text(String(format: "Frame: %.2f ms", perf.frameTimeMs))
                .font(.system(size: 12, design: .monospaced))
            Text("Tris: \(perf.triangleCount)")
                .font(.system(size: 12, design: .monospaced))
            ForEach(Array(AlloyLog.snapshot().enumerated()), id: \.offset) { _, s in
                 Text(s).font(.system(size: 10, design: .monospaced))
            }
        }
        .padding(10)
        .background(Color.black.opacity(0.6))
        .foregroundColor(.green)
        .cornerRadius(8)
    }
}

struct ContentView: View {
    var body: some View {
        ZStack(alignment: .topLeading) {
            MetalView()
                .ignoresSafeArea()

            PerformanceHUD()
                .padding(.top, 60)
                .padding(.leading, 16)
        }
    }
}
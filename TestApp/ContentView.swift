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
        
        let renderer = AlloyRenderer()
        context.coordinator.renderer = renderer
        
        if let device = view.device {
            context.coordinator.texture0 = TextureHelper.createCheckerboardTexture(device: device, isRed: false)
            context.coordinator.texture1 = TextureHelper.createCheckerboardTexture(device: device, isRed: true)
        }
        
        view.delegate = context.coordinator
        return view
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, MTKViewDelegate {
        var renderer: AlloyRenderer?
        var texture0: MTLTexture?
        var texture1: MTLTexture?
        var time: Float = 0.0
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func draw(in view: MTKView) {
            guard let renderer = renderer,
                  let texture0 = texture0,
                  let texture1 = texture1,
                  let drawable = view.currentDrawable else { return }
            
            let width = Float(drawable.texture.width)
            let height = Float(drawable.texture.height)
            time += 0.02
            
            // 立方体模型空间顶点
            let cubeVertices: [SIMD3<Float>] = [
                SIMD3<Float>(-1, -1, -1), SIMD3<Float>( 1, -1, -1),
                SIMD3<Float>( 1,  1, -1), SIMD3<Float>(-1,  1, -1),
                SIMD3<Float>(-1, -1,  1), SIMD3<Float>( 1, -1,  1),
                SIMD3<Float>( 1,  1,  1), SIMD3<Float>(-1,  1,  1)
            ]
            
            let faceIndices: [[Int]] = [
                [0, 1, 2, 3], [1, 5, 6, 2], [5, 4, 7, 6],
                [4, 0, 3, 7], [3, 2, 6, 7], [4, 5, 1, 0]
            ]
            
            let faceNormals: [SIMD3<Float>] = [
                SIMD3<Float>(0, 0, -1), SIMD3<Float>(1, 0, 0),
                SIMD3<Float>(0, 0, 1),  SIMD3<Float>(-1, 0, 0),
                SIMD3<Float>(0, 1, 0),  SIMD3<Float>(0, -1, 0)
            ]
            
            let faceUVs: [[SIMD2<Float>]] = [
                [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)],
                [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)],
                [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)],
                [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)],
                [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)],
                [SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)]
            ]
            
            let ax = time * 0.6
            let ay = time * 0.8
            
            func rotate(_ v: SIMD3<Float>) -> SIMD3<Float> {
                let x1 = v.x * cos(ay) - v.z * sin(ay)
                let z1 = v.x * sin(ay) + v.z * cos(ay)
                let y2 = v.y * cos(ax) - z1 * sin(ax)
                let z2 = v.y * sin(ax) + z1 * cos(ax)
                return SIMD3<Float>(x1, y2, z2)
            }
            
            func project(_ v: SIMD3<Float>) -> (SIMD2<Float>, Float) {
                let fov: Float = 800.0
                let z = max(v.z + 4.0, 0.1)
                let x = v.x * fov / z + width / 2
                let y = -v.y * fov / z + height / 2
                return (SIMD2<Float>(x, y), 1.0 / z)
            }
            
            let gal = AlloyGAL()
            gal.clearColor(r: 0.1, g: 0.1, b: 0.15, a: 1.0)
            gal.setViewport(width: Int(width), height: Int(height))
            
            var pso = AlloyPipelineDescriptor()
            pso.depthTestEnabled = true
            pso.cullMode = 1
            gal.bindPipeline(pso)
            
            // 🔥 关键改动：给每个面添加 4 个顶点，然后发 2 个 drawIndexed 指令
            for (faceIdx, indices) in faceIndices.enumerated() {
                let texID: UInt32 = faceIdx < 3 ? 0 : 1
                let n = rotate(faceNormals[faceIdx])
                let uvs = faceUVs[faceIdx]
                
                // 添加这个面的 4 个顶点到 VBO
                var faceVertIndices: [UInt32] = []
                for k in 0..<4 {
                    let worldPos = rotate(cubeVertices[indices[k]])
                    let (screenPos, invZ) = project(worldPos)
                    
                    let idx = gal.addVertex(
                        position: screenPos,
                        color: SIMD4<Float>(1, 1, 1, 1),
                        uv: uvs[k],
                        invZ: invZ,
                        normal: n
                    )
                    faceVertIndices.append(idx)
                }
                
                // 发 2 个 drawIndexed 指令（每个面 = 2 个三角形）
                gal.drawIndexed(v0: faceVertIndices[0], v1: faceVertIndices[1], v2: faceVertIndices[2], textureID: texID)
                gal.drawIndexed(v0: faceVertIndices[0], v1: faceVertIndices[2], v2: faceVertIndices[3], textureID: texID)
            }
            
            if let cmdBuffer = gal.submit(to: renderer, drawable: drawable, texture0: texture0, texture1: texture1) {
                cmdBuffer.present(drawable)
                cmdBuffer.commit()
            }
        }
    }
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .ignoresSafeArea()
    }
}
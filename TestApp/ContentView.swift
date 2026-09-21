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
            // 🔥 生成两张纹理
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
        var texture0: MTLTexture? // 蓝色棋盘格
        var texture1: MTLTexture? // 红色棋盘格
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
            
            let vertices3D: [SIMD3<Float>] = [
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
            
            let colors: [SIMD4<Float>] = [
                SIMD4<Float>(1, 1, 1, 1), SIMD4<Float>(1, 1, 1, 1),
                SIMD4<Float>(1, 1, 1, 1), SIMD4<Float>(1, 1, 1, 1),
                SIMD4<Float>(1, 1, 1, 1), SIMD4<Float>(1, 1, 1, 1)
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
            
            for (faceIdx, indices) in faceIndices.enumerated() {
                // 🔥 核心：根据不同的面，绑定不同的纹理！
                // 前 3 个面用蓝色棋盘格，后 3 个面用红色棋盘格
                if faceIdx < 3 {
                    gal.bindTexture(textureID: 0) // 蓝色
                } else {
                    gal.bindTexture(textureID: 1) // 红色
                }
                
                let v0 = rotate(vertices3D[indices[0]])
                let v1 = rotate(vertices3D[indices[1]])
                let v2 = rotate(vertices3D[indices[2]])
                let v3 = rotate(vertices3D[indices[3]])
                
                let n = rotate(faceNormals[faceIdx])
                
                let (p0, z0) = project(v0)
                let (p1, z1) = project(v1)
                let (p2, z2) = project(v2)
                let (p3, z3) = project(v3)
                
                let color = colors[faceIdx]
                let uvs = faceUVs[faceIdx]
                
                gal.drawTriangle(
                    p0: p0, p1: p1, p2: p2,
                    color: color,
                    z0: z0, z1: z1, z2: z2,
                    uv0: uvs[0], uv1: uvs[1], uv2: uvs[2],
                    n0: n, n1: n, n2: n
                )
                
                gal.drawTriangle(
                    p0: p0, p1: p2, p2: p3,
                    color: color,
                    z0: z0, z1: z2, z2: z3,
                    uv0: uvs[0], uv1: uvs[2], uv2: uvs[3],
                    n0: n, n1: n, n2: n
                )
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
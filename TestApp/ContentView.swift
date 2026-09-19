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
            context.coordinator.texture = TextureHelper.createCheckerboardTexture(device: device)
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
        var texture: MTLTexture?
        var time: Float = 0.0
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func draw(in view: MTKView) {
            guard let renderer = renderer,
                  let texture = texture,
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
            
            // 🔥 构建真正的二进制指令流
            var rawData: [UInt32] = []
            
            // 指令 0x02：设置背景色（深灰蓝）
            rawData.append(0x02)
            rawData.append(Float(0.1).bitPattern)
            rawData.append(Float(0.1).bitPattern)
            rawData.append(Float(0.15).bitPattern)
            
            func appendFloats(_ floats: [Float]) {
                for f in floats {
                    rawData.append(f.bitPattern)
                }
            }
            
            for (faceIdx, indices) in faceIndices.enumerated() {
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
                
                // 指令 0x01：画三角形
                rawData.append(0x01)
                appendFloats([
                    p0.x, p0.y, p1.x, p1.y, p2.x, p2.y,
                    color.x, color.y, color.z, color.w,
                    color.x, color.y, color.z, color.w,
                    color.x, color.y, color.z, color.w,
                    uvs[0].x, uvs[0].y, uvs[1].x, uvs[1].y, uvs[2].x, uvs[2].y,
                    z0, z1, z2,
                    n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z
                ])
                
                // 指令 0x01：画三角形（第二个）
                rawData.append(0x01)
                appendFloats([
                    p0.x, p0.y, p2.x, p2.y, p3.x, p3.y,
                    color.x, color.y, color.z, color.w,
                    color.x, color.y, color.z, color.w,
                    color.x, color.y, color.z, color.w,
                    uvs[0].x, uvs[0].y, uvs[2].x, uvs[2].y, uvs[3].x, uvs[3].y,
                    z0, z2, z3,
                    n.x, n.y, n.z, n.x, n.y, n.z, n.x, n.y, n.z
                ])
            }
            
            renderer.render(drawable: drawable, texture: texture, rawCommands: rawData)
        }
    }
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .ignoresSafeArea()
    }
}
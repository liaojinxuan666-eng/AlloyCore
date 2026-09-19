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
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func draw(in view: MTKView) {
            guard let renderer = renderer,
                  let drawable = view.currentDrawable else { return }
            
            let width = Float(drawable.texture.width)
            let height = Float(drawable.texture.height)
            time += 0.02
            
            // === 3D 立方体数据 ===
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
            
            let colors: [SIMD4<Float>] = [
                SIMD4<Float>(1, 0, 0, 1), SIMD4<Float>(0, 1, 0, 1),
                SIMD4<Float>(0, 0, 1, 1), SIMD4<Float>(1, 1, 0, 1),
                SIMD4<Float>(1, 0, 1, 1), SIMD4<Float>(0, 1, 1, 1)
            ]
            
            // === 旋转与投影数学 ===
            let ax = time * 0.6
            let ay = time * 0.8
            
            func rotate(_ v: SIMD3<Float>) -> SIMD3<Float> {
                let x1 = v.x * cos(ay) - v.z * sin(ay)
                let z1 = v.x * sin(ay) + v.z * cos(ay)
                let y2 = v.y * cos(ax) - z1 * sin(ax)
                let z2 = v.y * sin(ax) + z1 * cos(ax)
                return SIMD3<Float>(x1, y2, z2)
            }
            
            func project(_ v: SIMD3<Float>) -> SIMD2<Float> {
                let fov: Float = 800.0
                let z = max(v.z + 4.0, 0.1) // 平移 Z 轴防止除以零
                let x = v.x * fov / z + width / 2
                let y = -v.y * fov / z + height / 2
                return SIMD2<Float>(x, y)
            }
            
            // === 组装三角形并按深度排序（画家算法） ===
            var triangleList: [(vertices: [SIMD3<Float>], color: SIMD4<Float>, avgZ: Float)] = []
            
            for (faceIdx, indices) in faceIndices.enumerated() {
                let v0 = rotate(vertices3D[indices[0]])
                let v1 = rotate(vertices3D[indices[1]])
                let v2 = rotate(vertices3D[indices[2]])
                let v3 = rotate(vertices3D[indices[3]])
                
                // 每个面拆成两个三角形
                let z1 = (v0.z + v1.z + v2.z) / 3.0
                triangleList.append((vertices: [v0, v1, v2], color: colors[faceIdx], avgZ: z1))
                
                let z2 = (v0.z + v2.z + v3.z) / 3.0
                triangleList.append((vertices: [v0, v2, v3], color: colors[faceIdx], avgZ: z2))
            }
            
            // 按深度从远到近排序
            triangleList.sort { $0.avgZ < $1.avgZ }
            
            // 投影到 2D 并打包给引擎
            var finalTriangles: [[Vertex]] = []
            for tri in triangleList {
                let p0 = project(tri.vertices[0])
                let p1 = project(tri.vertices[1])
                let p2 = project(tri.vertices[2])
                
                finalTriangles.append([
                    Vertex(position: p0, color: tri.color),
                    Vertex(position: p1, color: tri.color),
                    Vertex(position: p2, color: tri.color)
                ])
            }
            
            // 丢给 AlloyCore 引擎！
            renderer.render(drawable: drawable, triangles: finalTriangles)
        }
    }
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .ignoresSafeArea()
    }
}

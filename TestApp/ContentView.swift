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
            
            let ax = time * 0.6
            let ay = time * 0.8
            
            func rotate(_ v: SIMD3<Float>) -> SIMD3<Float> {
                let x1 = v.x * cos(ay) - v.z * sin(ay)
                let z1 = v.x * sin(ay) + v.z * cos(ay)
                let y2 = v.y * cos(ax) - z1 * sin(ax)
                let z2 = v.y * sin(ax) + z1 * cos(ax)
                return SIMD3<Float>(x1, y2, z2)
            }
            
            // 投影函数现在返回 2D 坐标和用于深度测试的 Z 值
            func project(_ v: SIMD3<Float>) -> (SIMD2<Float>, Float) {
                let fov: Float = 800.0
                let z = max(v.z + 4.0, 0.1)
                let x = v.x * fov / z + width / 2
                let y = -v.y * fov / z + height / 2
                // 将 Z 值取倒数作为深度值，越大越近
                return (SIMD2<Float>(x, y), 1.0 / z)
            }
            
            var commands: [DrawTriangleCommand] = []
            
            for (faceIdx, indices) in faceIndices.enumerated() {
                let v0 = rotate(vertices3D[indices[0]])
                let v1 = rotate(vertices3D[indices[1]])
                let v2 = rotate(vertices3D[indices[2]])
                let v3 = rotate(vertices3D[indices[3]])
                
                let (p0, z0) = project(v0)
                let (p1, z1) = project(v1)
                let (p2, z2) = project(v2)
                let (p3, z3) = project(v3)
                
                let color = colors[faceIdx]
                
                // 构建指令流：每个面拆成两个三角形指令
                commands.append(DrawTriangleCommand(
                    v0: Vertex(position: p0, color: color),
                    v1: Vertex(position: p1, color: color),
                    v2: Vertex(position: p2, color: color),
                    z: (z0 + z1 + z2) / 3.0
                ))
                
                commands.append(DrawTriangleCommand(
                    v0: Vertex(position: p0, color: color),
                    v1: Vertex(position: p2, color: color),
                    v2: Vertex(position: p3, color: color),
                    z: (z0 + z2 + z3) / 3.0
                ))
            }
            
            // 提交指令流给引擎！
            renderer.render(drawable: drawable, commands: commands)
        }
    }
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .ignoresSafeArea()
    }
}

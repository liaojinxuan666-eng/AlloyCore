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
        var time: Float = 0.0 // 用于动画
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func draw(in view: MTKView) {
            guard let renderer = renderer,
                  let drawable = view.currentDrawable else { return }
            
            let texture = drawable.texture
            let w = Float(texture.width)
            let h = Float(texture.height)
            let centerX = w / 2
            let centerY = h / 2
            
            // 时间步进，让三角形转起来
            time += 0.02
            
            // 定义基础三角形（未旋转）
            let baseVertices = [
                SIMD2<Float>(0, -200), // 顶部
                SIMD2<Float>(-173, 100), // 左下
                SIMD2<Float>(173, 100)   // 右下
            ]
            
            // 简单的 2D 旋转矩阵
            let cosT = cos(time)
            let sinT = sin(time)
            
            // 在 TestApp 层计算旋转后的顶点，然后丢给引擎！
            let transformedVertices = [
                Vertex(position: SIMD2<Float>(
                    centerX + baseVertices[0].x * cosT - baseVertices[0].y * sinT,
                    centerY + baseVertices[0].x * sinT + baseVertices[0].y * cosT),
                       color: SIMD4<Float>(1, 0, 0, 1)),
                Vertex(position: SIMD2<Float>(
                    centerX + baseVertices[1].x * cosT - baseVertices[1].y * sinT,
                    centerY + baseVertices[1].x * sinT + baseVertices[1].y * cosT),
                       color: SIMD4<Float>(0, 1, 0, 1)),
                Vertex(position: SIMD2<Float>(
                    centerX + baseVertices[2].x * cosT - baseVertices[2].y * sinT,
                    centerY + baseVertices[2].x * sinT + baseVertices[2].y * cosT),
                       color: SIMD4<Float>(0, 0, 1, 1))
            ]
            
            // 把计算好的顶点丢给引擎！
            renderer.render(drawable: drawable, vertices: transformedVertices)
        }
    }
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .ignoresSafeArea()
    }
}
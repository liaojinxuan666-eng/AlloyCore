import SwiftUI
import MetalKit
import AlloyCore

struct MetalView: UIViewRepresentable {
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false // 我们用 Compute 直接写纹理
        
        // 🔥 确保渲染循环开启！
        view.isPaused = false
        view.enableSetNeedsDisplay = false
        view.preferredFramesPerSecond = 60
        
        let renderer = TriangleRenderer()
        context.coordinator.renderer = renderer
        view.delegate = context.coordinator
        
        return view
    }
    
    func updateUIView(_ uiView: MTKView, context: Context) {}
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator: NSObject, MTKViewDelegate {
        var renderer: TriangleRenderer?
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func draw(in view: MTKView) {
            guard let renderer = renderer,
                  let drawable = view.currentDrawable else { return }
            
            // 直接传入 drawable，让渲染器自己提交
            renderer.render(drawable: drawable)
        }
    }
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .ignoresSafeArea()
    }
}

import SwiftUI
import MetalKit
import AlloyCore // 重要：导入核心库

struct MetalView: UIViewRepresentable {
    func makeUIView(context: Context) -> MTKView {
        let view = MTKView()
        view.device = MTLCreateSystemDefaultDevice()
        view.colorPixelFormat = .bgra8Unorm
        view.framebufferOnly = false // 我们用 Compute 直接写纹理
        
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
                  let drawable = view.currentDrawable,
                  let texture = drawable.texture else { return }
            
            // 直接让我们的虚拟 GPU 写入这块纹理！
            renderer.render(to: texture)
        }
    }
}

struct ContentView: View {
    var body: some View {
        MetalView()
            .ignoresSafeArea()
    }
}
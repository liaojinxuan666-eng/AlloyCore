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
        
        if let device = view.device {
            context.coordinator.texture0 = TextureHelper.createCheckerboardTexture(device: device, isRed: false)
            context.coordinator.texture1 = TextureHelper.createCheckerboardTexture(device: device, isRed: true)
        }
        
        context.coordinator.buildCubeGeometry()
        context.coordinator.uploadGeometry(to: renderer)
        
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
        
        var vertexData: [Float] = []
        var indexData: [UInt32] = []
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func buildCubeGeometry() {
            let faces: [(normal: SIMD3<Float>, verts: [SIMD3<Float>])] = [
                (SIMD3<Float>(0, 0, -1), [SIMD3<Float>(-1, -1, -1), SIMD3<Float>(1, -1, -1), SIMD3<Float>(1, 1, -1), SIMD3<Float>(-1, 1, -1)]),
                (SIMD3<Float>(1, 0, 0),  [SIMD3<Float>(1, -1, -1), SIMD3<Float>(1, -1, 1), SIMD3<Float>(1, 1, 1), SIMD3<Float>(1, 1, -1)]),
                (SIMD3<Float>(0, 0, 1),  [SIMD3<Float>(1, -1, 1), SIMD3<Float>(-1, -1, 1), SIMD3<Float>(-1, 1, 1), SIMD3<Float>(1, 1, 1)]),
                (SIMD3<Float>(-1, 0, 0), [SIMD3<Float>(-1, -1, 1), SIMD3<Float>(-1, -1, -1), SIMD3<Float>(-1, 1, -1), SIMD3<Float>(-1, 1, 1)]),
                (SIMD3<Float>(0, 1, 0),  [SIMD3<Float>(-1, 1, -1), SIMD3<Float>(1, 1, -1), SIMD3<Float>(1, 1, 1), SIMD3<Float>(-1, 1, 1)]),
                (SIMD3<Float>(0, -1, 0), [SIMD3<Float>(-1, -1, 1), SIMD3<Float>(1, -1, 1), SIMD3<Float>(1, -1, -1), SIMD3<Float>(-1, -1, -1)])
            ]
            
            let uvs: [SIMD2<Float>] = [
                SIMD2<Float>(0, 0), SIMD2<Float>(1, 0), SIMD2<Float>(1, 1), SIMD2<Float>(0, 1)
            ]
            
            vertexData.removeAll()
            indexData.removeAll()
            
            for face in faces {
                let baseIndex = UInt32(vertexData.count / 12)
                
                for k in 0..<4 {
                    let p = face.verts[k]
                    let n = face.normal
                    vertexData.append(contentsOf: [
                        p.x, p.y, p.z,
                        1, 1, 1, 1,
                        uvs[k].x, uvs[k].y,
                        n.x, n.y, n.z
                    ])
                }
                
                indexData.append(contentsOf: [baseIndex, baseIndex + 1, baseIndex + 2])
                indexData.append(contentsOf: [baseIndex, baseIndex + 2, baseIndex + 3])
            }
        }
        
        func uploadGeometry(to renderer: AlloyRenderer) {
            renderer.uploadGeometry(vertexData: vertexData, indexData: indexData)
        }
        
        func draw(in view: MTKView) {
            guard let renderer = renderer,
                  let texture0 = texture0,
                  let texture1 = texture1,
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
            // 🔥 用 45 度 fov，让立方体尺寸更温和
            let fovY: Float = 45.0 * Float.pi / 180.0
            let f = 1.0 / tan(fovY / 2.0)
            
            let persp = simd_float4x4(
                SIMD4<Float>(f / aspect, 0, 0, 0),
                SIMD4<Float>(0, f, 0, 0),
                SIMD4<Float>(0, 0, (zFar + zNear) / (zNear - zFar), -1),
                SIMD4<Float>(0, 0, (2 * zFar * zNear) / (zNear - zFar), 0)
            )
            
            // 🔥 把立方体推到 z = -6，摄像机离它更远
            let translation = simd_float4x4(
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 1, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, 0, -6, 1)
            )
            
            let finalMatrix = persp * translation * rotX * rotY
            
            var matrixArray: [Float] = []
            for col in 0..<4 {
                for row in 0..<4 {
                    matrixArray.append(finalMatrix[col][row])
                }
            }
            
            let gal = AlloyGAL()
            gal.clearColor(r: 0.1, g: 0.1, b: 0.15, a: 1.0)
            gal.setViewport(width: Int(width), height: Int(height))
            
            var pso = AlloyPipelineDescriptor()
            pso.depthTestEnabled = true
            // 🔥 暂时关掉背面剔除，先让立方体正常显示
            pso.cullMode = 0
            gal.bindPipeline(pso)
            
            gal.setTransform(matrix: matrixArray)
            gal.drawIndexedRange(startIndex: 0, indexCount: UInt32(indexData.count), textureID: 0)
            
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
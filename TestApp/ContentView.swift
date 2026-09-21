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
        
        context.coordinator.generateSphere()
        
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
        
        var sphereVertices: [SIMD3<Float>] = []
        var sphereUVs: [SIMD2<Float>] = []
        var sphereNormals: [SIMD3<Float>] = []
        var sphereIndices: [UInt32] = []
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func generateSphere() {
            let latBands = 20
            let lonBands = 20
            
            for lat in 0...latBands {
                let theta = Float(lat) * Float.pi / Float(latBands)
                let sinTheta = sin(theta)
                let cosTheta = cos(theta)
                
                for lon in 0...lonBands {
                    let phi = Float(lon) * 2.0 * Float.pi / Float(lonBands)
                    let sinPhi = sin(phi)
                    let cosPhi = cos(phi)
                    
                    let x = cosPhi * sinTheta
                    let y = cosTheta
                    let z = sinPhi * sinTheta
                    
                    sphereVertices.append(SIMD3<Float>(x, y, z))
                    sphereUVs.append(SIMD2<Float>(Float(lon)/Float(lonBands), Float(lat)/Float(latBands)))
                    sphereNormals.append(SIMD3<Float>(x, y, z))
                }
            }
            
            for lat in 0..<latBands {
                for lon in 0..<lonBands {
                    let first = UInt32(lat * (lonBands + 1) + lon)
                    let second = first + UInt32(lonBands + 1)
                    
                    sphereIndices.append(contentsOf: [first, second, first + 1])
                    sphereIndices.append(contentsOf: [second, second + 1, first + 1])
                }
            }
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
            
            var finalVertexData: [Float] = []
            for i in 0..<sphereVertices.count {
                let worldPos = rotate(sphereVertices[i])
                let (screenPos, invZ) = project(worldPos)
                let n = rotate(sphereNormals[i])
                
                finalVertexData.append(contentsOf: [
                    screenPos.x, screenPos.y,
                    1, 1, 1, 1,
                    sphereUVs[i].x, sphereUVs[i].y,
                    invZ,
                    n.x, n.y, n.z
                ])
            }
            
            gal.vertexData = finalVertexData
            gal.indexData = sphereIndices
            gal.drawIndexedRange(startIndex: 0, indexCount: UInt32(sphereIndices.count), textureID: 0)
            
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
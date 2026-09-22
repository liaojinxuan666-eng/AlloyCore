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
        
        var sphereVertices: [SIMD3<Float>] = [] // 模型空间
        var sphereUVs: [SIMD2<Float>] = []
        var sphereNormals: [SIMD3<Float>] = []
        var sphereIndices: [UInt32] = []
        
        func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
        
        func generateSphere() {
            // 🔥 直接把顶点分辨率拉满！20x20 = 800 个三角形，CPU 也不再是瓶颈
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
            
            // 🔥 每帧只算 1 次三角函数，生成 4x4 矩阵
            let cosAY = cos(ay); let sinAY = sin(ay)
            let cosAX = cos(ax); let sinAX = sin(ax)
            
            // 构建旋转矩阵（Y 轴旋转 * X 轴旋转）
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
            // 注意：这里简化为旋转矩阵，真正的透视投影矩阵我们后续在 Shader 里做
            // 但为了保留透视效果，我们可以手动构造一个简单的透视矩阵。
            // 这里为了演示，直接传旋转矩阵，Shader 里简化处理。
            
            // 为简单起见，在 CPU 生成一个包含透视投影的矩阵
            let fov: Float = 800.0
            let zNear: Float = 0.1
            let zFar: Float = 100.0
            let aspect = width / height
            
            // 透视矩阵 (OpenGL 风格，NDC z 范围 [-1, 1])
            let f = 1.0 / tan(Float.pi / 4.0) // 90度视场角的一半
            let persp = simd_float4x4(
                SIMD4<Float>(f / aspect, 0, 0, 0),
                SIMD4<Float>(0, f, 0, 0),
                SIMD4<Float>(0, 0, (zFar + zNear) / (zNear - zFar), -1),
                SIMD4<Float>(0, 0, (2 * zFar * zNear) / (zNear - zFar), 0)
            )
            
            // 最后将矩阵压扁成 16 个 float 发给引擎
            var finalMatrix = persp * rotX * rotY
            // 扩大一点，让它看起来近一点
            finalMatrix.columns.3.z = -4.0
            
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
            pso.cullMode = 1
            gal.bindPipeline(pso)
            
            // 🔥 发送矩阵指令
            gal.setTransform(matrix: matrixArray)
            
            // 🔥 顶点数据不再做任何 CPU 运算，原样打包
            var finalVertexData: [Float] = []
            for i in 0..<sphereVertices.count {
                finalVertexData.append(contentsOf: [
                    sphereVertices[i].x, sphereVertices[i].y, sphereVertices[i].z, // 位置
                    1, 1, 1, 1,                                                   // 颜色
                    sphereUVs[i].x, sphereUVs[i].y,                               // UV
                    sphereNormals[i].x, sphereNormals[i].y, sphereNormals[i].z    // 法线
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
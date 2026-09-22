import Foundation
import QuartzCore
import Combine

class PerformanceMonitor: ObservableObject {
    static let shared = PerformanceMonitor()
    
    @Published var fps: Double = 0
    @Published var frameTimeMs: Double = 0
    @Published var gpuTimeMs: Double = 0
    @Published var triangleCount: Int = 0
    
    private var lastTimestamp: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var accumulatedTime: Double = 0
    private var accumulatedGpu: Double = 0
    
    func markFrame(gpuTime: Double) {
        let now = CACurrentMediaTime()
        if lastTimestamp == 0 { lastTimestamp = now; return }
        
        let dt = now - lastTimestamp
        lastTimestamp = now
        
        frameCount += 1
        accumulatedTime += dt
        accumulatedGpu += gpuTime
        
        if accumulatedTime >= 1.0 {
            DispatchQueue.main.async {
                self.fps = Double(self.frameCount) / self.accumulatedTime
                self.frameTimeMs = (self.accumulatedTime / Double(self.frameCount)) * 1000
                self.gpuTimeMs = (self.accumulatedGpu / Double(self.frameCount)) * 1000
            }
            frameCount = 0
            accumulatedTime = 0
            accumulatedGpu = 0
        }
    }
    
    func setTriangleCount(_ count: Int) {
        DispatchQueue.main.async {
            self.triangleCount = count
        }
    }
}
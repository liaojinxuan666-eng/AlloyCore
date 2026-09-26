import Foundation
import QuartzCore
import Combine
import AlloyCore

class PerformanceMonitor: ObservableObject {
    static let shared = PerformanceMonitor()
    
    @Published var fps: Double = 0
    @Published var frameTimeMs: Double = 0
    @Published var triangleCount: Int = 0
    @Published var passTimings: [AlloyPassTiming] = []
    
    private var lastTimestamp: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var accumulatedTime: Double = 0
    
    func markFrame() {
        let now = CACurrentMediaTime()
        if lastTimestamp == 0 {
            lastTimestamp = now
            return
        }
        
        let dt = now - lastTimestamp
        lastTimestamp = now
        
        frameCount += 1
        accumulatedTime += dt
        
        if accumulatedTime >= 0.5 {
            let fpsValue = Double(frameCount) / accumulatedTime
            let frameMs = (accumulatedTime / Double(frameCount)) * 1000
            DispatchQueue.main.async {
                self.fps = fpsValue
                self.frameTimeMs = frameMs
            }
            frameCount = 0
            accumulatedTime = 0
        }
    }
    
    func setTriangleCount(_ count: Int) {
        DispatchQueue.main.async {
            self.triangleCount = count
        }
        
        
    func setPassTimings(_ t: [AlloyPassTiming]) {
        DispatchQueue.main.async {
            self.passTimings = t
        }
    }
}
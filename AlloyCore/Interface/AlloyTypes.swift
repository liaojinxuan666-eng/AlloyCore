import simd
import QuartzCore

public typealias AlloyBufferHandle = UInt32
public typealias AlloyTextureHandle = UInt32
public typealias AlloyPipelineHandle = UInt32
public typealias AlloyComputePipelineHandle = UInt32

public enum AlloyPixelFormat: UInt32 {
    case rgba8Unorm = 0
    case bgra8Unorm = 1
    case depth32Float = 2
}

public enum AlloyCullMode: UInt32 {
    case none  = 0
    case back  = 1
    case front = 2
}

public struct AlloyPipelineDescriptor {
    public var depthTestEnabled: Bool = true
    public var depthWriteEnabled: Bool = true
    public var cullMode: AlloyCullMode = .back
    public var blendEnabled: Bool = false
    public var shaderID: UInt32 = 0
    public init() {}
}

public struct AlloyTextureDescriptor {
    public var width: Int
    public var height: Int
    public var format: AlloyPixelFormat
    public var data: [UInt8]?
    public init(width: Int, height: Int, format: AlloyPixelFormat = .rgba8Unorm, data: [UInt8]? = nil) {
        self.width = width
        self.height = height
        self.format = format
        self.data = data
    }
}
public final class AlloyLog {
    private static let lock = NSLock()
    private static var lines: [String] = []
    private static let maxLines = 8
    private static var lastFlush: CFTimeInterval = 0
    private static let logURL: URL? = {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
            .first?.appendingPathComponent("alloy.log")
    }()

    public static func log(_ s: String) {
        let line = s
        lock.lock()
        lines.append(line)
        if lines.count > maxLines { lines.removeFirst() }
        let now = CACurrentMediaTime()
        let flush = (now - lastFlush) > 0.1
        if flush { lastFlush = now }
        let content = lines.joined(separator: "\n") + "\n"
        lock.unlock()
        if flush, let url = logURL {
            try? content.write(to: url, atomically: true, encoding: .utf8)
        }
    }

    public static func snapshot() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        return lines
    }
}
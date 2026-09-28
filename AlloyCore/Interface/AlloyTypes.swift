import simd
import QuartzCore

public typealias AlloyBufferHandle = UInt32
public typealias AlloyTextureHandle = UInt32
public typealias AlloyPipelineHandle = UInt32
public typealias AlloyComputePipelineHandle = UInt32
public typealias AlloySamplerHandle = UInt32

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

public enum AlloyCompareFunc: UInt32 {
    case never        = 0
    case less         = 1
    case equal        = 2
    case lessEqual    = 3
    case greater      = 4
    case notEqual     = 5
    case greaterEqual = 6
    case always       = 7
}

public enum AlloyBlendFactor: UInt32 {
    case zero                  = 0
    case one                   = 1
    case srcColor              = 2
    case oneMinusSrcColor      = 3
    case dstColor              = 4
    case oneMinusDstColor      = 5
    case srcAlpha              = 6
    case oneMinusSrcAlpha      = 7
    case dstAlpha              = 8
    case oneMinusDstAlpha      = 9
    case constantColor         = 10
    case oneMinusConstantColor = 11
}

public enum AlloyBlendOp: UInt32 {
    case add             = 0
    case subtract        = 1
    case reverseSubtract = 2
    case min             = 3
    case max             = 4
}

public enum AlloyFilterMode: UInt32 {
    case nearest = 0
    case linear  = 1
}

public enum AlloyAddressMode: UInt32 {
    case clampToEdge = 0
    case repeatMode  = 1
    case mirror      = 2
}

public struct AlloySamplerDescriptor {
    public var magFilter: AlloyFilterMode = .linear
    public var minFilter: AlloyFilterMode = .linear
    public var addressU: AlloyAddressMode = .clampToEdge
    public var addressV: AlloyAddressMode = .clampToEdge
    public var addressW: AlloyAddressMode = .clampToEdge
    public init() {}
}

public struct AlloyVertexLayout {
    public var stride: UInt32
    public var positionOffset: Int32
    public var uvOffset: Int32
    public var normalOffset: Int32
    public var colorOffset: Int32

    public init(stride: UInt32 = 48,
                positionOffset: Int32 = 0,
                uvOffset: Int32 = 28,
                normalOffset: Int32 = 36,
                colorOffset: Int32 = 12) {
        self.stride = stride
        self.positionOffset = positionOffset
        self.uvOffset = uvOffset
        self.normalOffset = normalOffset
        self.colorOffset = colorOffset
    }
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

public struct AlloyPipelineDescriptor {
    public var depthTestEnabled: Bool = true
    public var depthWriteEnabled: Bool = true
    public var depthCompareFunc: AlloyCompareFunc = .less
    public var cullMode: AlloyCullMode = .back
    public var blendEnabled: Bool = false
    public var srcColorBlend: AlloyBlendFactor = .one
    public var dstColorBlend: AlloyBlendFactor = .zero
    public var colorBlendOp: AlloyBlendOp = .add
    public var srcAlphaBlend: AlloyBlendFactor = .one
    public var dstAlphaBlend: AlloyBlendFactor = .zero
    public var alphaBlendOp: AlloyBlendOp = .add
    public var shaderID: UInt32 = 0
    public init() {}
}

public struct AlloyComputePipelineDescriptor {
    public var shaderName: String
    public var threadsPerThreadgroup: SIMD3<UInt32>

    public init(shaderName: String,
                threadsPerThreadgroup: SIMD3<UInt32> = SIMD3<UInt32>(1, 1, 1)) {
        self.shaderName = shaderName
        self.threadsPerThreadgroup = threadsPerThreadgroup
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

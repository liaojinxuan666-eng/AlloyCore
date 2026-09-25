import simd
import QuartzCore

public typealias AlloyBufferHandle = UInt32
public typealias AlloyTextureHandle = UInt32
public typealias AlloyPipelineHandle = UInt32

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
import Foundation
import simd

/// NVN 前端——用于 Switch 游戏
///
/// 目标：把 NVN 调用翻译成 GAL 的中立调用。
/// 当前版本只支持"单 vertex buffer + interleaved attributes"。
/// 多 buffer / 分离 attribute 留到后续版本。
public final class NVNFrontend {

    // MARK: - 资源映射

    private var bufferMap: [UInt32: AlloyBufferHandle] = [:]
    private var nextBufferId: UInt32 = 1

    private var boundVertexBufferId: UInt32 = 0xFFFFFFFF
    private var boundIndexBufferId: UInt32 = 0xFFFFFFFF

    /// 每个 texture unit 绑定一个 GAL slot（0 或 1）。
    private var boundTextureSlots: [UInt32] = [0xFFFFFFFF, 0xFFFFFFFF]

    private weak var gal: AlloyGAL?

    public init() {}

    public func attach(to gal: AlloyGAL) {
        self.gal = gal
    }

    // MARK: - 资源创建

    /// 对应 nvnBufferCreate。分配空 buffer。
    public func createBuffer(sizeBytes: Int) -> UInt32 {
        guard let gal = gal else { return 0xFFFFFFFF }
        let floatCount = max(1, sizeBytes / 4)
        let zeros = [Float](repeating: 0, count: floatCount)
        let galHandle = gal.createVertexBuffer(data: zeros)
        let nvnId = nextBufferId
        nextBufferId += 1
        bufferMap[nvnId] = galHandle
        return nvnId
    }

    /// 注册一个已经在 GAL 里创建好的 buffer，返回 NVN 层 id。
    /// 用于测试场景：TestApp 用 GAL 建好，再包一层 NVN 前端。
    public func registerBuffer(galHandle: AlloyBufferHandle) -> UInt32 {
        let nvnId = nextBufferId
        nextBufferId += 1
        bufferMap[nvnId] = galHandle
        return nvnId
    }

    /// 对应 nvnBufferMap + 写入 + nvnBufferUnmap。
    public func uploadBuffer(_ nvnId: UInt32, data: [Float], offsetFloats: Int = 0) {
        guard let gal = gal, let galHandle = bufferMap[nvnId] else { return }
        gal.updateVertexBuffer(galHandle, data: data, offset: offsetFloats)
    }

    public func destroyBuffer(_ nvnId: UInt32) {
        bufferMap.removeValue(forKey: nvnId)
    }

    // MARK: - 命令录制

    public func bindVertexBuffer(_ nvnId: UInt32) {
        boundVertexBufferId = nvnId
    }

    public func bindIndexBuffer(_ nvnId: UInt32) {
        boundIndexBufferId = nvnId
    }

    /// 对应 nvnCommandBufferBindTexture。
    /// 简化版：`slot` 现在是 GAL 的 texture slot（0=tex0, 1=tex1）。
    /// 真正的 NVN texture → GAL slot 映射在后续版本。
    public func bindTextureSlot(_ slot: UInt32, unit: Int) {
        guard unit >= 0, unit < boundTextureSlots.count else { return }
        boundTextureSlots[unit] = slot
    }

    public func setViewport(x: Int, y: Int, width: Int, height: Int) {
        gal?.setViewport(width: width, height: height)
    }

    public func setScissor(x: Int, y: Int, width: Int, height: Int) {
        gal?.setScissor(x: x, y: y, width: width, height: height)
    }

    public func setDepthTestEnable(_ enable: Bool) {
        gal?.setDepthTestEnabled(enable)
    }

    public func setDepthWriteEnable(_ enable: Bool) {
        gal?.setDepthWriteEnabled(enable)
    }

    public func setDepthFunc(_ f: AlloyCompareFunc) {
        gal?.setDepthCompareFunc(f)
    }

    public func setCullMode(enable: Bool, face: AlloyCullMode) {
        if !enable {
            gal?.setCullMode(.none)
        } else {
            gal?.setCullMode(face)
        }
    }

    public func setBlendEnable(_ enable: Bool) {
        gal?.setBlendEnabled(enable)
    }

    public func setBlendFunc(srcColor: AlloyBlendFactor,
                             dstColor: AlloyBlendFactor,
                             colorOp: AlloyBlendOp,
                             srcAlpha: AlloyBlendFactor,
                             dstAlpha: AlloyBlendFactor,
                             alphaOp: AlloyBlendOp) {
        gal?.setBlendFactors(srcColor: srcColor,
                             dstColor: dstColor,
                             colorOp: colorOp,
                             srcAlpha: srcAlpha,
                             dstAlpha: dstAlpha,
                             alphaOp: alphaOp)
    }

    public func setVertexLayout(stride: UInt32,
                                positionOffset: Int32,
                                uvOffset: Int32,
                                normalOffset: Int32,
                                colorOffset: Int32) {
        let layout = AlloyVertexLayout(stride: stride,
                                       positionOffset: positionOffset,
                                       uvOffset: uvOffset,
                                       normalOffset: normalOffset,
                                       colorOffset: colorOffset)
        gal?.setVertexLayout(layout)
    }

    public func drawElements(indexCount: UInt32,
                             firstIndex: UInt32,
                             textureUnit: Int = 0) {
        guard let gal = gal,
              let galIbo = bufferMap[boundIndexBufferId] else { return }
        let slot: UInt32
        if textureUnit >= 0, textureUnit < boundTextureSlots.count {
            slot = boundTextureSlots[textureUnit]
        } else {
            slot = 0
        }
        gal.drawIndexed(iboHandle: galIbo,
                        indexCount: indexCount,
                        firstIndex: firstIndex,
                        textureID: slot)
    }

    // MARK: - 帧生命周期

    public func beginFrame() {
        boundVertexBufferId = 0xFFFFFFFF
        boundIndexBufferId = 0xFFFFFFFF
        boundTextureSlots = [0xFFFFFFFF, 0xFFFFFFFF]
    }

    public func endFrame() {
        // 暂无操作
    }
}

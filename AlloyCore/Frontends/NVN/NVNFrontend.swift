import Foundation
import simd

/// NVN 前端——用于 Switch 游戏
///
/// 目标：把 NVN 调用翻译成 GAL 的中立调用。
///
/// 当前版本（v0.7.0 Step 3a）只定义接口，**未被 TestApp 调用**。
/// Step 3b 会在 TestApp 里加开关，走这条路重建一个立方体，
/// 验证接口映射正确。
///
/// 关于 NVN 的真实签名（参考 Ryujinx / yuzu 的 Nvn 实现）：
///
///   void nvnCommandBufferBindVertexBuffer(cmbuf, slot, buffer, offset)
///   void nvnCommandBufferBindVertexAttrib(cmbuf, attrib, bufferSlot, size, fmt, stride, offset)
///   void nvnCommandBufferBindIndexBuffer(cmbuf, buffer, offset, indexType)
///   void nvnCommandBufferDrawElements(cmbuf, draw, indexType, count, indexOffset)
///   void nvnCommandBufferSetViewport(cmbuf, x, y, w, h)
///   void nvnCommandBufferSetScissor(cmbuf, x, y, w, h)
///   void nvnCommandBufferSetDepthTestEnable(cmbuf, bool)
///   void nvnCommandBufferSetDepthFunc(cmbuf, NVNdepthFunc)
///   void nvnCommandBufferSetCullFaceEnable(cmbuf, bool)
///   void nvnCommandBufferSetCullFace(cmbuf, NVNface)
///   void nvnCommandBufferSetBlendEnable(cmbuf, target, bool)
///   void nvnCommandBufferSetBlendFunc(cmbuf, target, srcRGB, dstRGB, srcA, dstA)
///
/// 这里简化成"单 vertex buffer + interleaved attributes"——最常见的用法。
/// 多 buffer / 分离 attribute 留到后续版本。
public final class NVNFrontend {

    // MARK: - 资源映射

    /// 前端持有的句柄 → GAL 句柄
    private var bufferMap: [UInt32: AlloyBufferHandle] = [:]
    private var textureMap: [UInt32: AlloyTextureHandle] = [:]
    private var nextBufferId: UInt32 = 1
    private var nextTextureId: UInt32 = 1

    /// 当前帧绑定的顶点/索引 buffer（NVN 层句柄）
    private var boundVertexBufferId: UInt32 = 0xFFFFFFFF
    private var boundIndexBufferId: UInt32 = 0xFFFFFFFF

    /// 当前帧绑定的纹理 slot（NVN 层句柄）
    /// NVN 有 16 个 texture unit，先支持 2 个（和 GAL 现在一致）
    private var boundTextures: [UInt32] = [0xFFFFFFFF, 0xFFFFFFFF]

    /// GAL 引用（弱引用，防止循环）
    private weak var gal: AlloyGAL?

    public init() {}

    public func attach(to gal: AlloyGAL) {
        self.gal = gal
    }

    // MARK: - 资源创建

    /// 对应 nvnBufferCreate + nvnBufferReserve。
    /// sizeBytes 必须是 4 的倍数（float 对齐）。
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

    /// 对应 nvnBufferMap + 写入 + nvnBufferUnmap。
    public func uploadBuffer(_ nvnId: UInt32, data: [Float], offsetFloats: Int = 0) {
        guard let gal = gal, let galHandle = bufferMap[nvnId] else { return }
        gal.updateVertexBuffer(galHandle, data: data, offset: offsetFloats)
    }

    public func destroyBuffer(_ nvnId: UInt32) {
        bufferMap.removeValue(forKey: nvnId)
    }

    /// 对应 nvnTextureBuilder + nvnTextureInitialize。
    /// 简化版：只支持 RGBA8。
    public func createTexture(width: Int, height: Int, pixels: [UInt8]) -> UInt32 {
        guard let gal = gal else { return 0xFFFFFFFF }
        let desc = AlloyTextureDescriptor(width: width, height: height, data: pixels)
        let galHandle = gal.createTexture(desc)
        let nvnId = nextTextureId
        nextTextureId += 1
        textureMap[nvnId] = galHandle
        return nvnId
    }

    // MARK: - 命令录制（NVN 风格）

    /// 对应 nvnCommandBufferBindVertexBuffer。
    public func bindVertexBuffer(_ nvnId: UInt32) {
        boundVertexBufferId = nvnId
    }

    /// 对应 nvnCommandBufferBindIndexBuffer。
    public func bindIndexBuffer(_ nvnId: UInt32) {
        boundIndexBufferId = nvnId
    }

    /// 对应 nvnCommandBufferBindTexture(unit, tex)。
    public func bindTexture(_ nvnId: UInt32, unit: Int) {
        guard unit >= 0, unit < boundTextures.count else { return }
        boundTextures[unit] = nvnId
    }

    /// 对应 nvnCommandBufferSetViewport(x, y, w, h)。
    /// GAL 现在只接受 (w, h)——x/y 暂忽略。
    public func setViewport(x: Int, y: Int, width: Int, height: Int) {
        gal?.setViewport(width: width, height: height)
    }

    /// 对应 nvnCommandBufferSetScissor(x, y, w, h)。
    public func setScissor(x: Int, y: Int, width: Int, height: Int) {
        gal?.setScissor(x: x, y: y, width: width, height: height)
    }

    /// 对应 nvnCommandBufferSetDepthTestEnable。
    public func setDepthTestEnable(_ enable: Bool) {
        gal?.setDepthTestEnabled(enable)
    }

    /// 对应 nvnCommandBufferSetDepthWriteEnable。
    public func setDepthWriteEnable(_ enable: Bool) {
        gal?.setDepthWriteEnabled(enable)
    }

    /// 对应 nvnCommandBufferSetDepthFunc。
    public func setDepthFunc(_ f: AlloyCompareFunc) {
        gal?.setDepthCompareFunc(f)
    }

    /// 对应 nvnCommandBufferSetCullFaceEnable + SetCullFace。
    /// NVN 用 (enable: Bool, face: NVNface)。GAL 用三态 enum。
    public func setCullMode(enable: Bool, face: AlloyCullMode) {
        if !enable {
            gal?.setCullMode(.none)
        } else {
            gal?.setCullMode(face)
        }
    }

    /// 对应 nvnCommandBufferSetBlendEnable。
    public func setBlendEnable(_ enable: Bool) {
        gal?.setBlendEnabled(enable)
    }

    /// 对应 nvnCommandBufferSetBlendFunc。
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

    /// 对应 nvnCommandBufferBindVertexAttrib（简化版：一次性给出全部偏移）。
    /// 真实 NVN 是逐 attrib 绑定，这里简化成 interleaved。
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

    /// 对应 nvnCommandBufferDrawElements。
    public func drawElements(indexCount: UInt32,
                             firstIndex: UInt32,
                             textureUnit: Int = 0) {
        guard let gal = gal,
              let galIbo = bufferMap[boundIndexBufferId] else { return }
        let texId: UInt32
        if textureUnit >= 0, textureUnit < boundTextures.count {
            texId = boundTextures[textureUnit]
        } else {
            texId = 0xFFFFFFFF
        }
        gal.drawIndexed(iboHandle: galIbo,
                        indexCount: indexCount,
                        firstIndex: firstIndex,
                        textureID: texId)
    }

    // MARK: - 帧生命周期

    public func beginFrame() {
        boundVertexBufferId = 0xFFFFFFFF
        boundIndexBufferId = 0xFFFFFFFF
        boundTextures = [0xFFFFFFFF, 0xFFFFFFFF]
    }

    public func endFrame() {
        // 暂无操作
    }
}

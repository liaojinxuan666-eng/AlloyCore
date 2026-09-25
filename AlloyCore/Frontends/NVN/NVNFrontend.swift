import Foundation

/// NVN 前端——用于 Switch 游戏
/// 未来：从 NVN 调用截获数据，翻译成 GAL 调用
public final class NVNFrontend: AlloyFrontend {
    public static var name: String { "NVN" }

    private static weak var gal: AlloyGAL?

    public static func attach(to gal: AlloyGAL) {
        Self.gal = gal
    }

    public static func beginFrame() {}
    public static func endFrame() {}

    // === 未来实现的接口（示例占位）===
    //
    // nvnCommandBufferDrawElements(...)  →  gal.drawIndexed(...)
    // nvnTextureBuilderSetSize2D(...)    →  gal.createTexture(...)
    // nvnWindowBuilderSetTextures(...)   →  gal.setViewport(...)
    //
    // 每个 NVN 调用都会被这个前端翻译成 GAL 的中立接口。
}
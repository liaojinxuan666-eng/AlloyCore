import Metal
import simd

class TextureHelper {
    // 新方法：返回像素数据，让 GAL 建纹理
    static func makeCheckerboardPixels(isRed: Bool = false) -> (pixels: [UInt8], width: Int, height: Int) {
        let width = 64
        let height = 64
        var pixelData = [UInt8](repeating: 0, count: width * height * 4)

        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * 4
                let isWhite = (x / 8 + y / 8) % 2 == 0

                if isWhite {
                    pixelData[offset]     = 255
                    pixelData[offset + 1] = 255
                    pixelData[offset + 2] = 255
                    pixelData[offset + 3] = 255
                } else if isRed {
                    pixelData[offset]     = 200
                    pixelData[offset + 1] = 50
                    pixelData[offset + 2] = 50
                    pixelData[offset + 3] = 255
                } else {
                    pixelData[offset]     = 50
                    pixelData[offset + 1] = 50
                    pixelData[offset + 2] = 150
                    pixelData[offset + 3] = 255
                }
            }
        }
        return (pixelData, width, height)
    }

    // 旧方法保留（其他地方可能还在用）
    static func createCheckerboardTexture(device: MTLDevice, isRed: Bool = false) -> MTLTexture? {
        let (pixels, width, height) = makeCheckerboardPixels(isRed: isRed)
        let bytesPerRow = width * 4
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false)
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else { return nil }
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: pixels,
            bytesPerRow: bytesPerRow)
        return texture
    }
}
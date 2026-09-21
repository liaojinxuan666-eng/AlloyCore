import Metal
import simd

class TextureHelper {
    // 蓝白棋盘格（现有的）
    static func createCheckerboardTexture(device: MTLDevice, isRed: Bool = false) -> MTLTexture? {
        let width = 64
        let height = 64
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let dataSize = width * height * bytesPerPixel
        
        var pixelData = [UInt8](repeating: 0, count: dataSize)
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = (y * width + x) * bytesPerPixel
                let isWhite = (x / 8 + y / 8) % 2 == 0
                
                // 根据参数生成蓝色或红色的棋盘格
                let color: (UInt8, UInt8, UInt8, UInt8)
                if isWhite {
                    color = (255, 255, 255, 255)
                } else {
                    color = isRed ? (200, 50, 50, 255) : (50, 50, 150, 255)
                }
                
                pixelData[offset] = color.0
                pixelData[offset + 1] = color.1
                pixelData[offset + 2] = color.2
                pixelData[offset + 3] = color.3
            }
        }
        
        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm,
            width: width,
            height: height,
            mipmapped: false
        )
        
        guard let texture = device.makeTexture(descriptor: textureDescriptor) else { return nil }
        
        texture.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: pixelData,
            bytesPerRow: bytesPerRow
        )
        
        return texture
    }
}
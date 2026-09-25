import Foundation

public protocol AlloyFrontend: AnyObject {
    static var name: String { get }
    static func attach(to gal: AlloyGAL)
    static func beginFrame()
    static func endFrame()
}
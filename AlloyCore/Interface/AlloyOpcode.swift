import Foundation

// MARK: - AlloyCore Command Protocol
//
// Single source of truth for the GAL <-> Renderer command stream.
// Every opcode and its payload length are defined here.
//
// Stream format: [opcode: UInt32][payload: UInt32...]
// All payloads are UInt32; floats are encoded via .bitPattern.
//
// Opcode table (v0.4.0):
//   0x01 DRAW_INDEXED              [globalIndexStart][indexCount][textureID]        4
//   0x02 CLEAR_COLOR               [r][g][b][a]                                     5
//   0x03 BIND_PIPELINE             [depthTest][cullMode][blend][shaderID]           5
//   0x04 SET_VIEWPORT              [w][h]                                           3
//   0x05 (unused)
//   0x06 SET_TRANSFORM             [16 floats as bitPattern]                       17
//   0x07 BIND_VERTEX_BUFFER        [poolOffset]                                     2
//   0x08 BIND_INDEX_BUFFER         [poolOffset]                                     2
//   0x09 UPDATE_VB                 [poolOffset][count][data0..dataN]            3 + N
//   0x0A UPDATE_IB                 [poolOffset][count][data0..dataN]            3 + N
//   0x10 BIND_COMPUTE_PIPELINE     [handle]                                         2
//   0x11 BIND_COMPUTE_VERTEX_POOL  [slot][byteOffset][byteLength]                   4
//   0x12 COMPUTE_DISPATCH          [gx][gy][gz]                                     4

public enum AlloyOpcode: UInt32 {
    case drawIndexed             = 0x01
    case clearColor              = 0x02
    case bindPipeline            = 0x03
    case setViewport             = 0x04
    case setTransform            = 0x06
    case bindVertexBuffer        = 0x07
    case bindIndexBuffer         = 0x08
    case updateVertexBuffer      = 0x09
    case updateIndexBuffer       = 0x0A
    case bindComputePipeline     = 0x10
    case bindComputeVertexPool   = 0x11
    case computeDispatch         = 0x12
}

public enum AlloyOpcodeLength {

    /// Returns the total length (in UInt32 words) of the command starting at
    /// index `i`, or nil if the stream is malformed / truncated.
    public static func of(_ cmd: [UInt32], at i: Int) -> Int? {
        guard i >= 0, i < cmd.count else { return nil }
        guard let op = AlloyOpcode(rawValue: cmd[i]) else { return nil }

        switch op {
        case .drawIndexed:
            return 4

        case .clearColor:
            return 5

        case .bindPipeline:
            return 5

        case .setViewport:
            return 3

        case .setTransform:
            return 17

        case .bindVertexBuffer:
            return 2

        case .bindIndexBuffer:
            return 2

        case .updateVertexBuffer:
            guard i + 3 <= cmd.count else { return nil }
            let n = Int(cmd[i + 2])
            let total = 3 + n
            guard i + total <= cmd.count else { return nil }
            return total

        case .updateIndexBuffer:
            guard i + 3 <= cmd.count else { return nil }
            let n = Int(cmd[i + 2])
            let total = 3 + n
            guard i + total <= cmd.count else { return nil }
            return total

        case .bindComputePipeline:
            return 2

        case .bindComputeVertexPool:
            return 4

        case .computeDispatch:
            return 4
        }
    }

    /// True if the opcode exists in the protocol.
    public static func isKnown(_ op: UInt32) -> Bool {
        return AlloyOpcode(rawValue: op) != nil
    }

    /// Human-readable name (useful for logging / debugging).
    public static func name(of op: UInt32) -> String {
        guard let o = AlloyOpcode(rawValue: op) else { return "UNKNOWN(\(op))" }
        switch o {
        case .drawIndexed:           return "DRAW_INDEXED"
        case .clearColor:            return "CLEAR_COLOR"
        case .bindPipeline:          return "BIND_PIPELINE"
        case .setViewport:           return "SET_VIEWPORT"
        case .setTransform:          return "SET_TRANSFORM"
        case .bindVertexBuffer:      return "BIND_VERTEX_BUFFER"
        case .bindIndexBuffer:       return "BIND_INDEX_BUFFER"
        case .updateVertexBuffer:    return "UPDATE_VB"
        case .updateIndexBuffer:     return "UPDATE_IB"
        case .bindComputePipeline:   return "BIND_COMPUTE_PIPELINE"
        case .bindComputeVertexPool: return "BIND_COMPUTE_VERTEX_POOL"
        case .computeDispatch:       return "COMPUTE_DISPATCH"
        }
    }
}
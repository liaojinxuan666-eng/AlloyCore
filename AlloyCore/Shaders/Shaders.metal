#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float4 color;
};

// 🔥 新的指令结构体：包含 Z 值用于深度测试
struct DrawTriangleCommand {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    float z;
};

kernel void process_commands(
    device const DrawTriangleCommand* commands [[buffer(0)]],
    constant uint& commandCount [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float2 pixel_pos = float2(gid) + 0.5;
    float4 finalColor = float4(0.0, 0.0, 0.0, 1.0); // 默认黑色背景
    float closestZ = -1e9; // 假设 Z 值越大越近

    // 🔥 核心：遍历所有指令！
    for (uint i = 0; i < commandCount; i++) {
        DrawTriangleCommand cmd = commands[i];
        
        float2 p0 = cmd.v0.position;
        float2 p1 = cmd.v1.position;
        float2 p2 = cmd.v2.position;
        
        // 计算重心坐标
        float2 v0 = p1 - p0;
        float2 v1 = p2 - p0;
        float2 v2 = pixel_pos - p0;
        
        float dot00 = dot(v0, v0);
        float dot01 = dot(v0, v1);
        float dot02 = dot(v0, v2);
        float dot11 = dot(v1, v1);
        float dot12 = dot(v1, v2);
        
        float invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
        float u = (dot11 * dot02 - dot01 * dot12) * invDenom;
        float v = (dot00 * dot12 - dot01 * dot02) * invDenom;
        float w = 1.0 - u - v;
        
        if (u >= 0.0 && v >= 0.0 && w >= 0.0) {
            // 深度测试：只有更近的才覆盖
            if (cmd.z > closestZ) {
                closestZ = cmd.z;
                finalColor = w * cmd.v0.color + u * cmd.v1.color + v * cmd.v2.color;
            }
        }
    }
    
    output.write(finalColor, gid);
}
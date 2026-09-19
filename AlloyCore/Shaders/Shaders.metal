#include <metal_stdlib>
using namespace metal;

// 每个指令包含 19 个 float，固定步长
constant int STRIDE = 19;

kernel void process_commands(
    device const float* rawCommands [[buffer(0)]],
    constant uint& commandCount [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tileOrigin [[threadgroup_position_in_grid]],
    uint2 tileSize [[threads_per_threadgroup]]
) {
    uint2 tileMin = tileOrigin * tileSize;
    uint2 tileMax = tileMin + tileSize;
    
    float2 pixel_pos = float2(gid) + 0.5;
    float4 finalColor = float4(0.0, 0.0, 0.0, 1.0);
    float closestZ = -1e9;

    for (uint i = 0; i < commandCount; i++) {
        uint offset = i * STRIDE;
        
        // 解析位置数据 (float2)
        float2 p0 = float2(rawCommands[offset + 0], rawCommands[offset + 1]);
        float2 p1 = float2(rawCommands[offset + 2], rawCommands[offset + 3]);
        float2 p2 = float2(rawCommands[offset + 4], rawCommands[offset + 5]);
        
        // 解析颜色数据 (float4)
        float4 c0 = float4(rawCommands[offset + 6], rawCommands[offset + 7], rawCommands[offset + 8], rawCommands[offset + 9]);
        float4 c1 = float4(rawCommands[offset + 10], rawCommands[offset + 11], rawCommands[offset + 12], rawCommands[offset + 13]);
        float4 c2 = float4(rawCommands[offset + 14], rawCommands[offset + 15], rawCommands[offset + 16], rawCommands[offset + 17]);
        
        // 解析深度数据 (float)
        float z = rawCommands[offset + 18];
        
        // 简单剔除
        float minX = min(min(p0.x, p1.x), p2.x);
        float maxX = max(max(p0.x, p1.x), p2.x);
        float minY = min(min(p0.y, p1.y), p2.y);
        float maxY = max(max(p0.y, p1.y), p2.y);
        
        if (maxX < float(tileMin.x) || minX > float(tileMax.x) || 
            maxY < float(tileMin.y) || minY > float(tileMax.y)) {
            continue;
        }
        
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
            if (z > closestZ) {
                closestZ = z;
                finalColor = w * c0 + u * c1 + v * c2;
            }
        }
    }
    
    output.write(finalColor, gid);
}
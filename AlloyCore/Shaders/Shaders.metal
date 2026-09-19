#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float4 color;
};

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
    uint2 gid [[thread_position_in_grid]],
    uint2 tileOrigin [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint2 tileSize [[threads_per_threadgroup]]
) {
    // 当前 Tile 负责的屏幕区域范围
    uint2 tileMin = tileOrigin * tileSize;
    uint2 tileMax = tileMin + tileSize;
    
    // 当前线程负责的具体像素
    float2 pixel_pos = float2(gid) + 0.5;
    float4 finalColor = float4(0.0, 0.0, 0.0, 1.0);
    float closestZ = -1e9;

    // 线程组内的所有线程，一起遍历指令流
    for (uint i = 0; i < commandCount; i++) {
        DrawTriangleCommand cmd = commands[i];
        
        float2 p0 = cmd.v0.position;
        float2 p1 = cmd.v1.position;
        float2 p2 = cmd.v2.position;
        
        // 简单剔除：如果三角形的包围盒和当前 Tile 完全没有交集，跳过
        // (这是 Tile-based 渲染的最基本优化，我们让同一个线程组内的线程共享这个判断)
        float minX = min(min(p0.x, p1.x), p2.x);
        float maxX = max(max(p0.x, p1.x), p2.x);
        float minY = min(min(p0.y, p1.y), p2.y);
        float maxY = max(max(p0.y, p1.y), p2.y);
        
        if (maxX < float(tileMin.x) || minX > float(tileMax.x) || 
            maxY < float(tileMin.y) || minY > float(tileMax.y)) {
            continue; // 这个三角形跟当前 Tile 无关，整个线程组一起跳过
        }
        
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
            if (cmd.z > closestZ) {
                closestZ = cmd.z;
                finalColor = w * cmd.v0.color + u * cmd.v1.color + v * cmd.v2.color;
            }
        }
    }
    
    output.write(finalColor, gid);
}
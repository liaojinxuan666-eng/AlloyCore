#include <metal_stdlib>
using namespace metal;

// 与 Swift 的 SIMD 完美对齐
struct Vertex {
    float2 position; // 去掉 packed
    float4 color;    // 去掉 packed
};

kernel void rasterize_triangle(
    device const Vertex* vertices [[buffer(0)]],
    texture2d<float, access::write> output [[texture(0)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float2 pixel_pos = float2(gid) + 0.5;
    
    float2 p0 = float2(vertices[0].position);
    float2 p1 = float2(vertices[1].position);
    float2 p2 = float2(vertices[2].position);
    
    // 1. 计算重心坐标
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
    
    // 2. 如果像素在三角形内，插值出颜色
    if (u >= 0.0 && v >= 0.0 && w >= 0.0) {
        float4 color = w * float4(vertices[0].color) +
                       u * float4(vertices[1].color) +
                       v * float4(vertices[2].color);
        output.write(color, gid);
    } else {
        // 外部则输出黑色背景
        output.write(float4(0.0, 0.0, 0.0, 1.0), gid);
    }
}

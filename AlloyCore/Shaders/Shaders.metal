#include <metal_stdlib>
using namespace metal;

// 🔥 步长从 27 变成 36
constant int STRIDE = 36;

kernel void process_commands(
    device const float* rawCommands [[buffer(0)]],
    constant uint& commandCount [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float> texture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tileOrigin [[threadgroup_position_in_grid]],
    uint2 tileSize [[threads_per_threadgroup]]
) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    uint2 tileMin = tileOrigin * tileSize;
    uint2 tileMax = tileMin + tileSize;
    
    float2 pixel_pos = float2(gid) + 0.5;
    float4 finalColor = float4(0.0, 0.0, 0.0, 1.0);
    float closestInvZ = -1e9;

    // 简单光照方向（从右上方打光）
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));

    for (uint i = 0; i < commandCount; i++) {
        uint offset = i * STRIDE;
        
        float2 p0 = float2(rawCommands[offset + 0], rawCommands[offset + 1]);
        float2 p1 = float2(rawCommands[offset + 2], rawCommands[offset + 3]);
        float2 p2 = float2(rawCommands[offset + 4], rawCommands[offset + 5]);
        
        float4 c0 = float4(rawCommands[offset + 6], rawCommands[offset + 7], rawCommands[offset + 8], rawCommands[offset + 9]);
        float4 c1 = float4(rawCommands[offset + 10], rawCommands[offset + 11], rawCommands[offset + 12], rawCommands[offset + 13]);
        float4 c2 = float4(rawCommands[offset + 14], rawCommands[offset + 15], rawCommands[offset + 16], rawCommands[offset + 17]);
        
        float2 uv0 = float2(rawCommands[offset + 18], rawCommands[offset + 19]);
        float2 uv1 = float2(rawCommands[offset + 20], rawCommands[offset + 21]);
        float2 uv2 = float2(rawCommands[offset + 22], rawCommands[offset + 23]);
        
        float invZ0 = rawCommands[offset + 24];
        float invZ1 = rawCommands[offset + 25];
        float invZ2 = rawCommands[offset + 26];
        
        // 🔥 解析法线
        float3 n0 = float3(rawCommands[offset + 27], rawCommands[offset + 28], rawCommands[offset + 29]);
        float3 n1 = float3(rawCommands[offset + 30], rawCommands[offset + 31], rawCommands[offset + 32]);
        float3 n2 = float3(rawCommands[offset + 33], rawCommands[offset + 34], rawCommands[offset + 35]);
        
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
            float invZInterp = w * invZ0 + u * invZ1 + v * invZ2;
            
            if (invZInterp > closestInvZ) {
                closestInvZ = invZInterp;
                
                float2 uvInterp = (w * uv0 * invZ0 + u * uv1 * invZ1 + v * uv2 * invZ2) / invZInterp;
                float4 texColor = texture.sample(textureSampler, uvInterp);
                float4 vertexColor = w * c0 + u * c1 + v * c2;
                
                // 🔥 插值法线并归一化
                float3 normal = normalize(w * n0 + u * n1 + v * n2);
                
                // 🔥 计算 Lambertian 光照强度，环境光设为 0.2
                float intensity = max(dot(normal, lightDir), 0.2);
                
                finalColor = vertexColor * texColor * intensity;
            }
        }
    }
    
    output.write(finalColor, gid);
}

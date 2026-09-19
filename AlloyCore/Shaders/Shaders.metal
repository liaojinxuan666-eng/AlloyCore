#include <metal_stdlib>
using namespace metal;

kernel void process_commands(
    device const uint* rawCommands [[buffer(0)]],
    constant uint& commandCount [[buffer(1)]], // 现在是数组总长度
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
    
    // 🔥 初始背景色
    float4 finalColor = float4(0.0, 0.0, 0.0, 1.0);
    float closestInvZ = -1e9;
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));

    // 🔥 使用 while 循环，因为每条指令的长度是不固定的
    uint i = 0;
    while (i < commandCount) {
        uint opcode = rawCommands[i];
        
        if (opcode == 0x02) {
            // 指令 0x02：设置背景色 (3 个 float)
            finalColor = float4(
                as_type<float>(rawCommands[i+1]),
                as_type<float>(rawCommands[i+2]),
                as_type<float>(rawCommands[i+3]),
                1.0
            );
            i += 4;
        }
        else if (opcode == 0x01) {
            // 指令 0x01：绘制三角形 (36 个 float)
            // 数据从 i+1 开始
            uint offset = i + 1;
            
            float2 p0 = float2(as_type<float>(rawCommands[offset + 0]), as_type<float>(rawCommands[offset + 1]));
            float2 p1 = float2(as_type<float>(rawCommands[offset + 2]), as_type<float>(rawCommands[offset + 3]));
            float2 p2 = float2(as_type<float>(rawCommands[offset + 4]), as_type<float>(rawCommands[offset + 5]));
            
            float4 c0 = float4(as_type<float>(rawCommands[offset + 6]), as_type<float>(rawCommands[offset + 7]), as_type<float>(rawCommands[offset + 8]), as_type<float>(rawCommands[offset + 9]));
            float4 c1 = float4(as_type<float>(rawCommands[offset + 10]), as_type<float>(rawCommands[offset + 11]), as_type<float>(rawCommands[offset + 12]), as_type<float>(rawCommands[offset + 13]));
            float4 c2 = float4(as_type<float>(rawCommands[offset + 14]), as_type<float>(rawCommands[offset + 15]), as_type<float>(rawCommands[offset + 16]), as_type<float>(rawCommands[offset + 17]));
            
            float2 uv0 = float2(as_type<float>(rawCommands[offset + 18]), as_type<float>(rawCommands[offset + 19]));
            float2 uv1 = float2(as_type<float>(rawCommands[offset + 20]), as_type<float>(rawCommands[offset + 21]));
            float2 uv2 = float2(as_type<float>(rawCommands[offset + 22]), as_type<float>(rawCommands[offset + 23]));
            
            float invZ0 = as_type<float>(rawCommands[offset + 24]);
            float invZ1 = as_type<float>(rawCommands[offset + 25]);
            float invZ2 = as_type<float>(rawCommands[offset + 26]);
            
            float3 n0 = float3(as_type<float>(rawCommands[offset + 27]), as_type<float>(rawCommands[offset + 28]), as_type<float>(rawCommands[offset + 29]));
            float3 n1 = float3(as_type<float>(rawCommands[offset + 30]), as_type<float>(rawCommands[offset + 31]), as_type<float>(rawCommands[offset + 32]));
            float3 n2 = float3(as_type<float>(rawCommands[offset + 33]), as_type<float>(rawCommands[offset + 34]), as_type<float>(rawCommands[offset + 35]));
            
            // 包围盒剔除
            float minX = min(min(p0.x, p1.x), p2.x);
            float maxX = max(max(p0.x, p1.x), p2.x);
            float minY = min(min(p0.y, p1.y), p2.y);
            float maxY = max(max(p0.y, p1.y), p2.y);
            
            if (maxX >= float(tileMin.x) && minX <= float(tileMax.x) && 
                maxY >= float(tileMin.y) && minY <= float(tileMax.y)) {
                
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
                        
                        float3 normal = normalize(w * n0 + u * n1 + v * n2);
                        float intensity = max(dot(normal, lightDir), 0.2);
                        
                        finalColor = vertexColor * texColor * intensity;
                    }
                }
            }
            
            // 推进指针：1 个 opcode + 36 个数据
            i += 37;
        }
        else {
            // 未知指令，跳出循环，防止死循环
            break;
        }
    }
    
    output.write(finalColor, gid);
}
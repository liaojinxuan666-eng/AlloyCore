#include <metal_stdlib>
using namespace metal;

// 绘制指令长度：1 个 opcode + 36 个 float = 37 个 uint
constant int STRIDE = 37;
constant int MAX_TILE_TRIANGLES = 64;
constant int TILE_SIZE = 16;

kernel void process_commands(
    device const uint* rawCommands [[buffer(0)]],
    constant uint& commandCount [[buffer(1)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float> texture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tileOrigin [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint2 tileSize [[threads_per_threadgroup]]
) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    uint2 tileMin = tileOrigin * tileSize;
    uint2 tileMax = tileMin + tileSize;
    
    float2 pixel_pos = float2(gid) + 0.5;
    float4 finalColor = float4(0.0, 0.0, 0.0, 1.0);
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));

    // 共享内存
    threadgroup atomic_uint tileTriangleCount;
    threadgroup uint triangleOffsets[MAX_TILE_TRIANGLES];
    threadgroup uint triangleOffsetCount;
    threadgroup int tileTriangleIndices[MAX_TILE_TRIANGLES];
    threadgroup float tileDepthBuffer[256];
    uint pixelIndex = localId.y * TILE_SIZE + localId.x;
    
    if (localId.x == 0 && localId.y == 0) {
        atomic_store_explicit(&tileTriangleCount, 0, memory_order_relaxed);
        triangleOffsetCount = 0;
    }
    tileDepthBuffer[pixelIndex] = -1e9;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. 预解析指令流，提取所有三角形偏移量
    if (localId.x == 0 && localId.y == 0) {
        uint i = 0;
        while (i < commandCount && triangleOffsetCount < MAX_TILE_TRIANGLES) {
            uint opcode = rawCommands[i];
            if (opcode == 0x02) {
                i += 5; // 修复：clearColor = 1 opcode + 4 float = 5
            } else if (opcode == 0x03) {
                i += 5; // bindPipeline = 1 opcode + 4 uint = 5
            } else if (opcode == 0x01) {
                triangleOffsets[triangleOffsetCount] = i;
                triangleOffsetCount++;
                i += STRIDE; // STRIDE = 37
            } else {
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. 动态负载均衡筛选三角形
    uint tid = localId.y * TILE_SIZE + localId.x;
    for (uint t = tid; t < triangleOffsetCount; t += TILE_SIZE * TILE_SIZE) {
        uint offset = triangleOffsets[t] + 1;
        
        float2 p0 = float2(as_type<float>(rawCommands[offset + 0]), as_type<float>(rawCommands[offset + 1]));
        float2 p1 = float2(as_type<float>(rawCommands[offset + 2]), as_type<float>(rawCommands[offset + 3]));
        float2 p2 = float2(as_type<float>(rawCommands[offset + 4]), as_type<float>(rawCommands[offset + 5]));
        
        float minX = min(min(p0.x, p1.x), p2.x);
        float maxX = max(max(p0.x, p1.x), p2.x);
        float minY = min(min(p0.y, p1.y), p2.y);
        float maxY = max(max(p0.y, p1.y), p2.y);
        
        if (maxX >= float(tileMin.x) && minX <= float(tileMax.x) && 
            maxY >= float(tileMin.y) && minY <= float(tileMax.y)) {
            uint idx = atomic_fetch_add_explicit(&tileTriangleCount, 1, memory_order_relaxed);
            if (idx < MAX_TILE_TRIANGLES) {
                tileTriangleIndices[idx] = triangleOffsets[t];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    uint finalTriangleCount = atomic_load_explicit(&tileTriangleCount, memory_order_relaxed);
    if (finalTriangleCount > MAX_TILE_TRIANGLES) finalTriangleCount = MAX_TILE_TRIANGLES;

    // 3. 深度预通道
    for (uint t = 0; t < finalTriangleCount; t++) {
        uint offset = tileTriangleIndices[t] + 1;
        
        float2 p0 = float2(as_type<float>(rawCommands[offset + 0]), as_type<float>(rawCommands[offset + 1]));
        float2 p1 = float2(as_type<float>(rawCommands[offset + 2]), as_type<float>(rawCommands[offset + 3]));
        float2 p2 = float2(as_type<float>(rawCommands[offset + 4]), as_type<float>(rawCommands[offset + 5]));
        
        float invZ0 = as_type<float>(rawCommands[offset + 24]);
        float invZ1 = as_type<float>(rawCommands[offset + 25]);
        float invZ2 = as_type<float>(rawCommands[offset + 26]);
        
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
            if (invZInterp > tileDepthBuffer[pixelIndex]) {
                tileDepthBuffer[pixelIndex] = invZInterp;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. 着色通道
    for (uint t = 0; t < finalTriangleCount; t++) {
        uint offset = tileTriangleIndices[t] + 1;
        
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
            
            if (abs(invZInterp - tileDepthBuffer[pixelIndex]) < 0.0001) {
                float2 uvInterp = (w * uv0 * invZ0 + u * uv1 * invZ1 + v * uv2 * invZ2) / invZInterp;
                float4 texColor = texture.sample(textureSampler, uvInterp);
                float4 vertexColor = w * c0 + u * c1 + v * c2;
                
                float3 normal = normalize(w * n0 + u * n1 + v * n2);
                float intensity = max(dot(normal, lightDir), 0.2);
                
                finalColor = vertexColor * texColor * intensity;
            }
        }
    }
    
    output.write(finalColor, gid);
}
#include <metal_stdlib>
using namespace metal;

constant int MAX_TILE_TRIANGLES = 128;
constant int TILE_SIZE = 16;

struct ScreenVertex {
    float2 position;
    float2 uv;
    float invZ;
    float3 normal;
    float4 color;
};

struct TriangleRef {
    uint v0, v1, v2;
    uint texID;
};

kernel void geometry_pass(
    device const float* inputVBO [[buffer(0)]],
    device float* outputVBO [[buffer(1)]],
    constant uint& vertexCount [[buffer(2)]],
    constant float4x4& transform [[buffer(3)]],
    constant float2& screenSize [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vertexCount) return;
    
    uint src = gid * 12;
    float3 pos = float3(inputVBO[src], inputVBO[src+1], inputVBO[src+2]);
    float4 color = float4(inputVBO[src+3], inputVBO[src+4], inputVBO[src+5], inputVBO[src+6]);
    float2 uv = float2(inputVBO[src+7], inputVBO[src+8]);
    float3 nrm = float3(inputVBO[src+9], inputVBO[src+10], inputVBO[src+11]);
    
    float4 clip = transform * float4(pos, 1.0);
    
    float2 screenPos;
    float invZ;
    if (clip.w <= 0.0001) {
        screenPos = float2(-10000.0, -10000.0);
        invZ = -1e9;
    } else {
        float2 ndc = clip.xy / clip.w;
        screenPos = float2((ndc.x + 1.0) * 0.5 * screenSize.x, (1.0 - ndc.y) * 0.5 * screenSize.y);
        invZ = 1.0 / clip.w;
    }
    
    float3 viewNormal = (transform * float4(nrm, 0.0)).xyz;
    
    outputVBO[src+0] = screenPos.x;
    outputVBO[src+1] = screenPos.y;
    outputVBO[src+2] = uv.x;
    outputVBO[src+3] = uv.y;
    outputVBO[src+4] = invZ;
    outputVBO[src+5] = viewNormal.x;
    outputVBO[src+6] = viewNormal.y;
    outputVBO[src+7] = viewNormal.z;
    outputVBO[src+8] = color.r;
    outputVBO[src+9] = color.g;
    outputVBO[src+10] = color.b;
    outputVBO[src+11] = color.a;
}

inline ScreenVertex loadScreenVertex(device const float* vd, uint index) {
    uint off = index * 12;
    ScreenVertex v;
    v.position = float2(vd[off], vd[off+1]);
    v.uv = float2(vd[off+2], vd[off+3]);
    v.invZ = vd[off+4];
    v.normal = float3(vd[off+5], vd[off+6], vd[off+7]);
    v.color = float4(vd[off+8], vd[off+9], vd[off+10], vd[off+11]);
    return v;
}

kernel void process_commands(
    device const uint* rawCommands [[buffer(0)]],
    constant uint& commandCount [[buffer(1)]],
    device const float* screenVertexData [[buffer(2)]],
    device const uint* indexData [[buffer(3)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float> texture0 [[texture(1)]],
    texture2d<float> texture1 [[texture(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tileOrigin [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint2 tileSize [[threads_per_threadgroup]]
) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    uint2 tileMin = tileOrigin * tileSize;
    uint2 tileMax = tileMin + tileSize;
    
    float2 pixel_pos = float2(gid) + 0.5;
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));

    threadgroup atomic_uint triCount;
    threadgroup TriangleRef triList[MAX_TILE_TRIANGLES];
    threadgroup float4 sharedClearColor;
    threadgroup uint sharedDepthTestEnabled;
    threadgroup uint sharedCullMode;
    
    if (localId.x == 0 && localId.y == 0) {
        atomic_store_explicit(&triCount, 0, memory_order_relaxed);
        sharedClearColor = float4(0.0, 0.0, 0.0, 1.0);
        sharedDepthTestEnabled = 1;
        sharedCullMode = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (localId.x == 0 && localId.y == 0) {
        uint i = 0;
        while (i < commandCount) {
            uint opcode = rawCommands[i];
            if (opcode == 0x02) {
                sharedClearColor = float4(
                    as_type<float>(rawCommands[i+1]),
                    as_type<float>(rawCommands[i+2]),
                    as_type<float>(rawCommands[i+3]),
                    1.0
                );
                i += 5;
            } else if (opcode == 0x03) {
                sharedDepthTestEnabled = rawCommands[i+1];
                sharedCullMode = rawCommands[i+2];
                i += 5;
            } else if (opcode == 0x04) {
                i += 3;
            } else if (opcode == 0x06) {
                i += 17;
            } else if (opcode == 0x01) {
                uint indexStart = rawCommands[i+1];
                uint indexCount = rawCommands[i+2];
                uint texID = rawCommands[i+3];
                
                for (uint t = 0; t < indexCount; t += 3) {
                    if (atomic_load_explicit(&triCount, memory_order_relaxed) >= MAX_TILE_TRIANGLES) break;
                    
                    uint v0 = indexData[indexStart + t];
                    uint v1 = indexData[indexStart + t + 1];
                    uint v2 = indexData[indexStart + t + 2];
                    
                    uint idx = atomic_fetch_add_explicit(&triCount, 1, memory_order_relaxed);
                    if (idx < MAX_TILE_TRIANGLES) {
                        triList[idx].v0 = v0;
                        triList[idx].v1 = v1;
                        triList[idx].v2 = v2;
                        triList[idx].texID = texID;
                    }
                }
                i += 4;
            } else {
                break;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint finalTriCount = atomic_load_explicit(&triCount, memory_order_relaxed);
    if (finalTriCount > MAX_TILE_TRIANGLES) finalTriCount = MAX_TILE_TRIANGLES;

    // 单一通道：直接算出最近的深度，然后着色
    float closestInvZ = -1e9;
    float4 bestColor = sharedClearColor;
    float bestInvZ = -1e9;
    
    for (uint t = 0; t < finalTriCount; t++) {
        TriangleRef ref = triList[t];
        ScreenVertex v0 = loadScreenVertex(screenVertexData, ref.v0);
        ScreenVertex v1 = loadScreenVertex(screenVertexData, ref.v1);
        ScreenVertex v2 = loadScreenVertex(screenVertexData, ref.v2);
        
        float2 s0 = v0.position;
        float2 s1 = v1.position;
        float2 s2 = v2.position;
        
        float minX = min(min(s0.x, s1.x), s2.x);
        float maxX = max(max(s0.x, s1.x), s2.x);
        float minY = min(min(s0.y, s1.y), s2.y);
        float maxY = max(max(s0.y, s1.y), s2.y);
        
        if (maxX < float(tileMin.x) || minX > float(tileMax.x) ||
            maxY < float(tileMin.y) || minY > float(tileMax.y)) continue;
        
        float cross2D = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x);
        if (sharedCullMode == 1 && cross2D >= 0.0) continue;
        if (sharedCullMode == 2 && cross2D <= 0.0) continue;
        
        float2 e0 = s1 - s0;
        float2 e1 = s2 - s0;
        float2 e2 = pixel_pos - s0;
        
        float d00 = dot(e0, e0);
        float d01 = dot(e0, e1);
        float d02 = dot(e0, e2);
        float d11 = dot(e1, e1);
        float d12 = dot(e1, e2);
        
        float invDenom = 1.0 / (d00 * d11 - d01 * d01);
        float u = (d11 * d02 - d01 * d12) * invDenom;
        float v = (d00 * d12 - d01 * d02) * invDenom;
        float wBary = 1.0 - u - v;
        
        if (u >= 0.0 && v >= 0.0 && wBary >= 0.0) {
            float invZ = wBary * v0.invZ + u * v1.invZ + v * v2.invZ;
            
            bool passesDepth;
            if (sharedDepthTestEnabled == 1) {
                passesDepth = (invZ > closestInvZ);
            } else {
                passesDepth = true;
            }
            
            if (passesDepth) {
                closestInvZ = invZ;
                
                float invZ0 = v0.invZ;
                float invZ1 = v1.invZ;
                float invZ2 = v2.invZ;
                
                float2 uvInterp = (wBary * v0.uv * invZ0 + u * v1.uv * invZ1 + v * v2.uv * invZ2) / invZ;
                float4 colorInterp = wBary * v0.color + u * v1.color + v * v2.color;
                
                float3 n0 = normalize(v0.normal);
                float3 n1 = normalize(v1.normal);
                float3 n2 = normalize(v2.normal);
                float3 normal = normalize(wBary * n0 + u * n1 + v * n2);
                
                float4 texColor;
                if (ref.texID == 0) {
                    texColor = texture0.sample(textureSampler, uvInterp);
                } else {
                    texColor = texture1.sample(textureSampler, uvInterp);
                }
                
                float intensity = max(dot(normal, lightDir), 0.2);
                bestColor = colorInterp * texColor * intensity;
                bestInvZ = invZ;
            }
        }
    }
    
    output.write(bestColor, gid);
}
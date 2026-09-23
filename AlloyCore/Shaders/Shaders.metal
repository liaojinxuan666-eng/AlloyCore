#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 16;
constant int MAX_PER_TILE = 256;

struct ScreenVertex {
    float2 position;
    float2 uv;
    float invZ;
    float3 normal;
    float4 color;
};

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

kernel void binning_pass(
    device const float* screenVertexData [[buffer(0)]],
    device const uint* indexData [[buffer(1)]],
    device atomic_uint* binCounts [[buffer(2)]],
    device uint* binData [[buffer(3)]],
    constant uint& triangleCount [[buffer(4)]],
    constant uint2& screenTileCounts [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= triangleCount) return;
    
    uint v0 = indexData[gid * 3];
    uint v1 = indexData[gid * 3 + 1];
    uint v2 = indexData[gid * 3 + 2];
    
    ScreenVertex sv0 = loadScreenVertex(screenVertexData, v0);
    ScreenVertex sv1 = loadScreenVertex(screenVertexData, v1);
    ScreenVertex sv2 = loadScreenVertex(screenVertexData, v2);
    
    float minX = min(min(sv0.position.x, sv1.position.x), sv2.position.x);
    float maxX = max(max(sv0.position.x, sv1.position.x), sv2.position.x);
    float minY = min(min(sv0.position.y, sv1.position.y), sv2.position.y);
    float maxY = max(max(sv0.position.y, sv1.position.y), sv2.position.y);
    
    float screenW = float(screenTileCounts.x * TILE_SIZE);
    float screenH = float(screenTileCounts.y * TILE_SIZE);
    
    if (maxX < 0.0 || minX >= screenW) return;
    if (maxY < 0.0 || minY >= screenH) return;
    
    uint tx0 = uint(max(0.0, floor(minX / float(TILE_SIZE))));
    uint tx1 = uint(min(float(screenTileCounts.x - 1), floor(maxX / float(TILE_SIZE))));
    uint ty0 = uint(max(0.0, floor(minY / float(TILE_SIZE))));
    uint ty1 = uint(min(float(screenTileCounts.y - 1), floor(maxY / float(TILE_SIZE))));
    
    for (uint ty = ty0; ty <= ty1; ty++) {
        for (uint tx = tx0; tx <= tx1; tx++) {
            uint tileIdx = ty * screenTileCounts.x + tx;
            uint slot = atomic_fetch_add_explicit(&binCounts[tileIdx], 1, memory_order_relaxed);
            if (slot < MAX_PER_TILE) {
                binData[tileIdx * MAX_PER_TILE + slot] = gid;
            }
        }
    }
}

kernel void rasterize_pass(
    device const float* screenVertexData [[buffer(0)]],
    device const uint* indexData [[buffer(1)]],
    device const uint* binCounts [[buffer(2)]],
    device const uint* binData [[buffer(3)]],
    constant uint& screenTileCountX [[buffer(4)]],
    constant uint& depthTestEnabled [[buffer(5)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float> texture0 [[texture(1)]],
    texture2d<float> texture1 [[texture(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tileOrigin [[threadgroup_position_in_grid]],
    uint2 localId [[thread_position_in_threadgroup]],
    uint2 tileSize [[threads_per_threadgroup]]
) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    
    uint tileIdx = tileOrigin.y * screenTileCountX + tileOrigin.x;
    uint count = binCounts[tileIdx];
    if (count > MAX_PER_TILE) count = MAX_PER_TILE;
    
    float2 pixel_pos = float2(gid) + 0.5;
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));
    
    float4 bestColor = float4(0.1, 0.1, 0.15, 1.0);
    float closestInvZ = -1e9;
    
    for (uint t = 0; t < count; t++) {
        uint triIdx = binData[tileIdx * MAX_PER_TILE + t];
        uint v0 = indexData[triIdx * 3];
        uint v1 = indexData[triIdx * 3 + 1];
        uint v2 = indexData[triIdx * 3 + 2];
        
        ScreenVertex sv0 = loadScreenVertex(screenVertexData, v0);
        ScreenVertex sv1 = loadScreenVertex(screenVertexData, v1);
        ScreenVertex sv2 = loadScreenVertex(screenVertexData, v2);
        
        float2 s0 = sv0.position;
        float2 s1 = sv1.position;
        float2 s2 = sv2.position;
        
        float2 e0 = s1 - s0;
        float2 e1 = s2 - s0;
        float2 e2 = pixel_pos - s0;
        
        float d00 = dot(e0, e0);
        float d01 = dot(e0, e1);
        float d02 = dot(e0, e2);
        float d11 = dot(e1, e1);
        float d12 = dot(e1, e2);
        
        float det = d00 * d11 - d01 * d01;
        if (abs(det) < 1e-9) continue;
        float invDenom = 1.0 / det;
        float u = (d11 * d02 - d01 * d12) * invDenom;
        float v = (d00 * d12 - d01 * d02) * invDenom;
        float wBary = 1.0 - u - v;
        
        if (u >= 0.0 && v >= 0.0 && wBary >= 0.0) {
            float invZ = wBary * sv0.invZ + u * sv1.invZ + v * sv2.invZ;
            
            bool passesDepth;
            if (depthTestEnabled == 1) {
                passesDepth = (invZ > closestInvZ);
            } else {
                passesDepth = true;
            }
            
            if (passesDepth) {
                closestInvZ = invZ;
                
                float invZ0 = sv0.invZ;
                float invZ1 = sv1.invZ;
                float invZ2 = sv2.invZ;
                
                float2 uvInterp = (wBary * sv0.uv * invZ0 + u * sv1.uv * invZ1 + v * sv2.uv * invZ2) / invZ;
                float4 colorInterp = wBary * sv0.color + u * sv1.color + v * sv2.color;
                
                float3 n0 = normalize(sv0.normal);
                float3 n1 = normalize(sv1.normal);
                float3 n2 = normalize(sv2.normal);
                float3 normal = normalize(wBary * n0 + u * n1 + v * n2);
                
                float4 texColor = texture0.sample(textureSampler, uvInterp);
                
                float intensity = max(dot(normal, lightDir), 0.2);
                bestColor = colorInterp * texColor * intensity;
            }
        }
    }
    
    output.write(bestColor, gid);
}

kernel void upscale_pass(
    texture2d<float, access::sample> lowRes [[texture(0)]],
    texture2d<float, access::write> highRes [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    float2 uv = (float2(gid) + 0.5) / float2(highRes.get_width(), highRes.get_height());
    float4 color = lowRes.sample(s, uv);
    highRes.write(color, gid);
}
#include <metal_stdlib>
using namespace metal;

constant int MAX_TILE_TRIANGLES = 128;
constant int TILE_SIZE = 16;
constant int VERTEX_STRIDE = 12; // 每个顶点 12 个 float

struct Vertex {
    float2 position;
    float4 color;
    float2 uv;
    float invZ;
    float3 normal;
};

struct TriangleRef {
    uint v0, v1, v2;
    uint texID;
};

inline Vertex loadVertex(device const float* vd, uint index) {
    uint off = index * VERTEX_STRIDE;
    Vertex v;
    v.position = float2(vd[off], vd[off+1]);
    v.color = float4(vd[off+2], vd[off+3], vd[off+4], vd[off+5]);
    v.uv = float2(vd[off+6], vd[off+7]);
    v.invZ = vd[off+8];
    v.normal = float3(vd[off+9], vd[off+10], vd[off+11]);
    return v;
}

kernel void process_commands(
    device const uint* rawCommands [[buffer(0)]],
    constant uint& commandCount [[buffer(1)]],
    device const float* vertexData [[buffer(2)]],
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
    threadgroup float tileDepthBuffer[256];
    threadgroup float4 sharedClearColor;
    threadgroup uint sharedDepthTestEnabled;
    threadgroup uint sharedCullMode;
    
    uint pixelIndex = localId.y * TILE_SIZE + localId.x;
    
    if (localId.x == 0 && localId.y == 0) {
        atomic_store_explicit(&triCount, 0, memory_order_relaxed);
        sharedClearColor = float4(0.0, 0.0, 0.0, 1.0);
        sharedDepthTestEnabled = 1;
        sharedCullMode = 0;
    }
    tileDepthBuffer[pixelIndex] = -1e9;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============ 第 1 步：预解析，收集三角形 ============
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
            } else if (opcode == 0x01) {
                uint indexStart = rawCommands[i+1];
                uint indexCount = rawCommands[i+2];
                uint texID = rawCommands[i+3];
                
                for (uint t = 0; t < indexCount; t += 3) {
                    if (atomic_load_explicit(&triCount, memory_order_relaxed) >= MAX_TILE_TRIANGLES) break;
                    
                    uint v0 = indexData[indexStart + t];
                    uint v1 = indexData[indexStart + t + 1];
                    uint v2 = indexData[indexStart + t + 2];
                    
                    Vertex vert0 = loadVertex(vertexData, v0);
                    Vertex vert1 = loadVertex(vertexData, v1);
                    Vertex vert2 = loadVertex(vertexData, v2);
                    
                    float minX = min(min(vert0.position.x, vert1.position.x), vert2.position.x);
                    float maxX = max(max(vert0.position.x, vert1.position.x), vert2.position.x);
                    float minY = min(min(vert0.position.y, vert1.position.y), vert2.position.y);
                    float maxY = max(max(vert0.position.y, vert1.position.y), vert2.position.y);
                    
                    if (maxX >= float(tileMin.x) && minX <= float(tileMax.x) &&
                        maxY >= float(tileMin.y) && minY <= float(tileMax.y)) {
                        uint idx = atomic_fetch_add_explicit(&triCount, 1, memory_order_relaxed);
                        if (idx < MAX_TILE_TRIANGLES) {
                            triList[idx].v0 = v0;
                            triList[idx].v1 = v1;
                            triList[idx].v2 = v2;
                            triList[idx].texID = texID;
                        }
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

    // ============ 第 2 步：深度预通道 ============
    for (uint t = 0; t < finalTriCount; t++) {
        TriangleRef ref = triList[t];
        Vertex v0 = loadVertex(vertexData, ref.v0);
        Vertex v1 = loadVertex(vertexData, ref.v1);
        Vertex v2 = loadVertex(vertexData, ref.v2);
        
        float2 p0 = v0.position;
        float2 p1 = v1.position;
        float2 p2 = v2.position;
        
        float cross2D = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        if (sharedCullMode == 1 && cross2D >= 0.0) continue;
        if (sharedCullMode == 2 && cross2D <= 0.0) continue;
        
        float2 e0 = p1 - p0;
        float2 e1 = p2 - p0;
        float2 e2 = pixel_pos - p0;
        
        float d00 = dot(e0, e0);
        float d01 = dot(e0, e1);
        float d02 = dot(e0, e2);
        float d11 = dot(e1, e1);
        float d12 = dot(e1, e2);
        
        float invDenom = 1.0 / (d00 * d11 - d01 * d01);
        float u = (d11 * d02 - d01 * d12) * invDenom;
        float v = (d00 * d12 - d01 * d02) * invDenom;
        float w = 1.0 - u - v;
        
        if (u >= 0.0 && v >= 0.0 && w >= 0.0) {
            float invZ = w * v0.invZ + u * v1.invZ + v * v2.invZ;
            if (sharedDepthTestEnabled == 1) {
                if (invZ > tileDepthBuffer[pixelIndex]) {
                    tileDepthBuffer[pixelIndex] = invZ;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ============ 第 3 步：着色通道 ============
    float4 finalColor = sharedClearColor;
    
    for (uint t = 0; t < finalTriCount; t++) {
        TriangleRef ref = triList[t];
        Vertex v0 = loadVertex(vertexData, ref.v0);
        Vertex v1 = loadVertex(vertexData, ref.v1);
        Vertex v2 = loadVertex(vertexData, ref.v2);
        
        float2 p0 = v0.position;
        float2 p1 = v1.position;
        float2 p2 = v2.position;
        
        float cross2D = (p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x);
        if (sharedCullMode == 1 && cross2D >= 0.0) continue;
        if (sharedCullMode == 2 && cross2D <= 0.0) continue;
        
        float2 e0 = p1 - p0;
        float2 e1 = p2 - p0;
        float2 e2 = pixel_pos - p0;
        
        float d00 = dot(e0, e0);
        float d01 = dot(e0, e1);
        float d02 = dot(e0, e2);
        float d11 = dot(e1, e1);
        float d12 = dot(e1, e2);
        
        float invDenom = 1.0 / (d00 * d11 - d01 * d01);
        float u = (d11 * d02 - d01 * d12) * invDenom;
        float v = (d00 * d12 - d01 * d02) * invDenom;
        float w = 1.0 - u - v;
        
        if (u >= 0.0 && v >= 0.0 && w >= 0.0) {
            float invZ = w * v0.invZ + u * v1.invZ + v * v2.invZ;
            
            bool shouldDraw = true;
            if (sharedDepthTestEnabled == 1) {
                if (abs(invZ - tileDepthBuffer[pixelIndex]) >= 0.0001) shouldDraw = false;
            }
            
            if (shouldDraw) {
                float2 uvInterp = (w * v0.uv * v0.invZ + u * v1.uv * v1.invZ + v * v2.uv * v2.invZ) / invZ;
                float4 colorInterp = w * v0.color + u * v1.color + v * v2.color;
                float3 normal = normalize(w * v0.normal + u * v1.normal + v * v2.normal);
                
                float4 texColor;
                if (ref.texID == 0) {
                    texColor = texture0.sample(textureSampler, uvInterp);
                } else {
                    texColor = texture1.sample(textureSampler, uvInterp);
                }
                
                float intensity = max(dot(normal, lightDir), 0.2);
                finalColor = colorInterp * texColor * intensity;
            }
        }
    }
    
    output.write(finalColor, gid);
}
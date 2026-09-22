#include <metal_stdlib>
using namespace metal;

constant int MAX_TILE_TRIANGLES = 128;
constant int TILE_SIZE = 16;
constant int VERTEX_STRIDE = 12; // position(3) + color(4) + uv(2) + normal(3) = 12 floats

struct Vertex {
    float3 position;  // 模型空间坐标
    float4 color;
    float2 uv;
    float3 normal;    // 模型空间法线
};

struct TriangleRef {
    uint v0, v1, v2;
    uint texID;
};

inline Vertex loadVertex(device const float* vd, uint index) {
    uint off = index * VERTEX_STRIDE;
    Vertex v;
    v.position = float3(vd[off], vd[off+1], vd[off+2]);
    v.color = float4(vd[off+3], vd[off+4], vd[off+5], vd[off+6]);
    v.uv = float2(vd[off+7], vd[off+8]);
    v.normal = float3(vd[off+9], vd[off+10], vd[off+11]);
    return v;
}

// 4x4 矩阵乘法
inline float4 mulMatrix(float4x4 m, float4 v) {
    return float4(
        dot(m[0], v),
        dot(m[1], v),
        dot(m[2], v),
        dot(m[3], v)
    );
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
    threadgroup float4x4 sharedTransform;
    
    uint pixelIndex = localId.y * TILE_SIZE + localId.x;
    
    if (localId.x == 0 && localId.y == 0) {
        atomic_store_explicit(&triCount, 0, memory_order_relaxed);
        sharedClearColor = float4(0.0, 0.0, 0.0, 1.0);
        sharedDepthTestEnabled = 1;
        sharedCullMode = 0;
        sharedTransform = float4x4(1.0); // 单位矩阵
    }
    tileDepthBuffer[pixelIndex] = -1e9;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 1. 预解析，收集三角形，读取变换矩阵
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
                // 🔥 读取 4x4 矩阵
                sharedTransform = float4x4(
                    float4(as_type<float>(rawCommands[i+1]), as_type<float>(rawCommands[i+2]), as_type<float>(rawCommands[i+3]), as_type<float>(rawCommands[i+4])),
                    float4(as_type<float>(rawCommands[i+5]), as_type<float>(rawCommands[i+6]), as_type<float>(rawCommands[i+7]), as_type<float>(rawCommands[i+8])),
                    float4(as_type<float>(rawCommands[i+9]), as_type<float>(rawCommands[i+10]), as_type<float>(rawCommands[i+11]), as_type<float>(rawCommands[i+12])),
                    float4(as_type<float>(rawCommands[i+13]), as_type<float>(rawCommands[i+14]), as_type<float>(rawCommands[i+15]), as_type<float>(rawCommands[i+16]))
                );
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
                    
                    triList[atomic_fetch_add_explicit(&triCount, 1, memory_order_relaxed)].v0 = v0;
                    triList[atomic_fetch_add_explicit(&triCount, 1, memory_order_relaxed)].v1 = v1;
                    triList[atomic_fetch_add_explicit(&triCount, 1, memory_order_relaxed)].v2 = v2;
                    // 简化处理，纹理ID所有三角形统一
                    triList[atomic_load_explicit(&triCount, memory_order_relaxed) - 1].texID = texID;
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

    // 2. 深度预通道（在 GPU 里做矩阵变换）
    for (uint t = 0; t < finalTriCount; t++) {
        TriangleRef ref = triList[t];
        Vertex v0 = loadVertex(vertexData, ref.v0);
        Vertex v1 = loadVertex(vertexData, ref.v1);
        Vertex v2 = loadVertex(vertexData, ref.v2);
        
        // 🔥 在 GPU 里把模型空间坐标投影到屏幕空间
        float4 clip0 = mulMatrix(sharedTransform, float4(v0.position, 1.0));
        float4 clip1 = mulMatrix(sharedTransform, float4(v1.position, 1.0));
        float4 clip2 = mulMatrix(sharedTransform, float4(v2.position, 1.0));
        
        float2 p0 = clip0.xy / clip0.w;
        float2 p1 = clip1.xy / clip1.w;
        float2 p2 = clip2.xy / clip2.w;
        
        // 转换到屏幕坐标
        float w = float(output.get_width());
        float h = float(output.get_height());
        float2 screen0 = float2((p0.x + 1.0) * 0.5 * w, (1.0 - p0.y) * 0.5 * h);
        float2 screen1 = float2((p1.x + 1.0) * 0.5 * w, (1.0 - p1.y) * 0.5 * h);
        float2 screen2 = float2((p2.x + 1.0) * 0.5 * w, (1.0 - p2.y) * 0.5 * h);
        
        float cross2D = (screen1.x - screen0.x) * (screen2.y - screen0.y) - (screen1.y - screen0.y) * (screen2.x - screen0.x);
        if (sharedCullMode == 1 && cross2D >= 0.0) continue;
        if (sharedCullMode == 2 && cross2D <= 0.0) continue;
        
        float2 e0 = screen1 - screen0;
        float2 e1 = screen2 - screen0;
        float2 e2 = pixel_pos - screen0;
        
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
            float invZ = wBary * (1.0/clip0.w) + u * (1.0/clip1.w) + v * (1.0/clip2.w);
            if (sharedDepthTestEnabled == 1) {
                if (invZ > tileDepthBuffer[pixelIndex]) {
                    tileDepthBuffer[pixelIndex] = invZ;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. 着色通道
    float4 finalColor = sharedClearColor;
    
    for (uint t = 0; t < finalTriCount; t++) {
        TriangleRef ref = triList[t];
        Vertex v0 = loadVertex(vertexData, ref.v0);
        Vertex v1 = loadVertex(vertexData, ref.v1);
        Vertex v2 = loadVertex(vertexData, ref.v2);
        
        float4 clip0 = mulMatrix(sharedTransform, float4(v0.position, 1.0));
        float4 clip1 = mulMatrix(sharedTransform, float4(v1.position, 1.0));
        float4 clip2 = mulMatrix(sharedTransform, float4(v2.position, 1.0));
        
        float2 p0 = clip0.xy / clip0.w;
        float2 p1 = clip1.xy / clip1.w;
        float2 p2 = clip2.xy / clip2.w;
        
        float w = float(output.get_width());
        float h = float(output.get_height());
        float2 screen0 = float2((p0.x + 1.0) * 0.5 * w, (1.0 - p0.y) * 0.5 * h);
        float2 screen1 = float2((p1.x + 1.0) * 0.5 * w, (1.0 - p1.y) * 0.5 * h);
        float2 screen2 = float2((p2.x + 1.0) * 0.5 * w, (1.0 - p2.y) * 0.5 * h);
        
        float cross2D = (screen1.x - screen0.x) * (screen2.y - screen0.y) - (screen1.y - screen0.y) * (screen2.x - screen0.x);
        if (sharedCullMode == 1 && cross2D >= 0.0) continue;
        if (sharedCullMode == 2 && cross2D <= 0.0) continue;
        
        float2 e0 = screen1 - screen0;
        float2 e1 = screen2 - screen0;
        float2 e2 = pixel_pos - screen0;
        
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
            float invZ0 = 1.0 / clip0.w;
            float invZ1 = 1.0 / clip1.w;
            float invZ2 = 1.0 / clip2.w;
            float invZ = wBary * invZ0 + u * invZ1 + v * invZ2;
            
            bool shouldDraw = true;
            if (sharedDepthTestEnabled == 1) {
                if (abs(invZ - tileDepthBuffer[pixelIndex]) >= 0.0001) shouldDraw = false;
            }
            
            if (shouldDraw) {
                float2 uvInterp = (wBary * v0.uv * invZ0 + u * v1.uv * invZ1 + v * v2.uv * invZ2) / invZ;
                float4 colorInterp = wBary * v0.color + u * v1.color + v * v2.color;
                
                // 法线变换（忽略平移）
                float3 n0 = normalize(mulMatrix(sharedTransform, float4(v0.normal, 0.0)).xyz);
                float3 n1 = normalize(mulMatrix(sharedTransform, float4(v1.normal, 0.0)).xyz);
                float3 n2 = normalize(mulMatrix(sharedTransform, float4(v2.normal, 0.0)).xyz);
                float3 normal = normalize(wBary * n0 + u * n1 + v * n2);
                
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
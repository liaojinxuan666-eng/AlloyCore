#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 16;
constant int MAX_PER_TILE = 256;
constant float NEAR_W = 0.001;

// 裁剪空间顶点布局（13 个 float）：
// [0-3] clip.xyzw  [4-5] uv  [6-8] normal  [9-12] color

// 屏幕空间顶点布局（13 个 float）：
// [0-1] screen.xy  [2-3] uv  [4] invZ  [5] clipW  [6-8] normal  [9-12] color

struct ScreenVertex {
    float2 position;
    float2 uv;
    float invZ;
    float clipW;
    float3 normal;
    float4 color;
};

inline ScreenVertex loadScreenVertex(device const float* vd, uint index) {
    uint off = index * 13;
    ScreenVertex v;
    v.position = float2(vd[off], vd[off+1]);
    v.uv = float2(vd[off+2], vd[off+3]);
    v.invZ = vd[off+4];
    v.clipW = vd[off+5];
    v.normal = float3(vd[off+6], vd[off+7], vd[off+8]);
    v.color = float4(vd[off+9], vd[off+10], vd[off+11], vd[off+12]);
    return v;
}

kernel void geometry_pass(
    device const float* inputVBO [[buffer(0)]],
    device float* outputVBO [[buffer(1)]],
    constant uint& vertexCount [[buffer(2)]],
    constant float4x4& transform [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= vertexCount) return;
    uint src = gid * 12;
    float3 pos = float3(inputVBO[src], inputVBO[src+1], inputVBO[src+2]);
    float4 color = float4(inputVBO[src+3], inputVBO[src+4], inputVBO[src+5], inputVBO[src+6]);
    float2 uv = float2(inputVBO[src+7], inputVBO[src+8]);
    float3 nrm = float3(inputVBO[src+9], inputVBO[src+10], inputVBO[src+11]);
    
    float4 clip = transform * float4(pos, 1.0);
    float3 viewNormal = (transform * float4(nrm, 0.0)).xyz;
    
    uint dst = gid * 13;
    outputVBO[dst+0] = clip.x;
    outputVBO[dst+1] = clip.y;
    outputVBO[dst+2] = clip.z;
    outputVBO[dst+3] = clip.w;
    outputVBO[dst+4] = uv.x;
    outputVBO[dst+5] = uv.y;
    outputVBO[dst+6] = viewNormal.x;
    outputVBO[dst+7] = viewNormal.y;
    outputVBO[dst+8] = viewNormal.z;
    outputVBO[dst+9] = color.r;
    outputVBO[dst+10] = color.g;
    outputVBO[dst+11] = color.b;
    outputVBO[dst+12] = color.a;
}

// 从裁剪空间插值到屏幕空间顶点
inline ScreenVertex makeScreenVertex(
    float4 clip, float2 uv, float3 nrm, float4 color, float2 screenSize)
{
    ScreenVertex v;
    if (clip.w > NEAR_W) {
        float2 ndc = clip.xy / clip.w;
        v.position = float2((ndc.x + 1.0) * 0.5 * screenSize.x, (1.0 - ndc.y) * 0.5 * screenSize.y);
        v.invZ = 1.0 / clip.w;
    } else {
        v.position = float2(0.0, 0.0);
        v.invZ = 0.0;
    }
    v.uv = uv;
    v.clipW = clip.w;
    v.normal = nrm;
    v.color = color;
    return v;
}

// Sutherland-Hodgman 近平面裁剪 + 投影，每个线程处理一个三角形
kernel void clip_project_pass(
    device const float* inVerts [[buffer(0)]],
    device const uint* inIndices [[buffer(1)]],
    device float* outVerts [[buffer(2)]],
    device uint* outIndices [[buffer(3)]],
    constant uint& triangleCount [[buffer(4)]],
    constant float2& screenSize [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= triangleCount) return;
    
    uint i0 = inIndices[gid * 3];
    uint i1 = inIndices[gid * 3 + 1];
    uint i2 = inIndices[gid * 3 + 2];
    
    float4 c0 = float4(inVerts[i0*13], inVerts[i0*13+1], inVerts[i0*13+2], inVerts[i0*13+3]);
    float4 c1 = float4(inVerts[i1*13], inVerts[i1*13+1], inVerts[i1*13+2], inVerts[i1*13+3]);
    float4 c2 = float4(inVerts[i2*13], inVerts[i2*13+1], inVerts[i2*13+2], inVerts[i2*13+3]);
    
    float2 uv0 = float2(inVerts[i0*13+4], inVerts[i0*13+5]);
    float2 uv1 = float2(inVerts[i1*13+4], inVerts[i1*13+5]);
    float2 uv2 = float2(inVerts[i2*13+4], inVerts[i2*13+5]);
    
    float3 n0 = float3(inVerts[i0*13+6], inVerts[i0*13+7], inVerts[i0*13+8]);
    float3 n1 = float3(inVerts[i1*13+6], inVerts[i1*13+7], inVerts[i1*13+8]);
    float3 n2 = float3(inVerts[i2*13+6], inVerts[i2*13+7], inVerts[i2*13+8]);
    
    float4 col0 = float4(inVerts[i0*13+9], inVerts[i0*13+10], inVerts[i0*13+11], inVerts[i0*13+12]);
    float4 col1 = float4(inVerts[i1*13+9], inVerts[i1*13+10], inVerts[i1*13+11], inVerts[i1*13+12]);
    float4 col2 = float4(inVerts[i2*13+9], inVerts[i2*13+10], inVerts[i2*13+11], inVerts[i2*13+12]);
    
    // 每个三角形最多 4 个输出顶点，最多 2 个输出三角形
    uint baseVert = gid * 4 * 13;
    uint baseTri = gid * 2 * 3;
    
    // 默认全部标记为空
    outIndices[baseTri]     = 0xFFFFFFFF;
    outIndices[baseTri + 1] = 0xFFFFFFFF;
    outIndices[baseTri + 2] = 0xFFFFFFFF;
    outIndices[baseTri + 3] = 0xFFFFFFFF;
    outIndices[baseTri + 4] = 0xFFFFFFFF;
    outIndices[baseTri + 5] = 0xFFFFFFFF;
    
    // 内部/外部判定
    bool in0 = c0.w > NEAR_W;
    bool in1 = c1.w > NEAR_W;
    bool in2 = c2.w > NEAR_W;
    int insideCount = (in0 ? 1 : 0) + (in1 ? 1 : 0) + (in2 ? 1 : 0);
    
    if (insideCount == 0) return;
    
    if (insideCount == 3) {
        // 全部在内部：直接输出 1 个三角形
        ScreenVertex sv0 = makeScreenVertex(c0, uv0, n0, col0, screenSize);
        ScreenVertex sv1 = makeScreenVertex(c1, uv1, n1, col1, screenSize);
        ScreenVertex sv2 = makeScreenVertex(c2, uv2, n2, col2, screenSize);
        
        device float* p0 = outVerts + baseVert;
        device float* p1 = outVerts + baseVert + 13;
        device float* p2 = outVerts + baseVert + 26;
        
        p0[0]=sv0.position.x; p0[1]=sv0.position.y; p0[2]=sv0.uv.x; p0[3]=sv0.uv.y; p0[4]=sv0.invZ; p0[5]=sv0.clipW;
        p0[6]=sv0.normal.x; p0[7]=sv0.normal.y; p0[8]=sv0.normal.z;
        p0[9]=sv0.color.r; p0[10]=sv0.color.g; p0[11]=sv0.color.b; p0[12]=sv0.color.a;
        
        p1[0]=sv1.position.x; p1[1]=sv1.position.y; p1[2]=sv1.uv.x; p1[3]=sv1.uv.y; p1[4]=sv1.invZ; p1[5]=sv1.clipW;
        p1[6]=sv1.normal.x; p1[7]=sv1.normal.y; p1[8]=sv1.normal.z;
        p1[9]=sv1.color.r; p1[10]=sv1.color.g; p1[11]=sv1.color.b; p1[12]=sv1.color.a;
        
        p2[0]=sv2.position.x; p2[1]=sv2.position.y; p2[2]=sv2.uv.x; p2[3]=sv2.uv.y; p2[4]=sv2.invZ; p2[5]=sv2.clipW;
        p2[6]=sv2.normal.x; p2[7]=sv2.normal.y; p2[8]=sv2.normal.z;
        p2[9]=sv2.color.r; p2[10]=sv2.color.g; p2[11]=sv2.color.b; p2[12]=sv2.color.a;
        
        outIndices[baseTri + 0] = 0;
        outIndices[baseTri + 1] = 1;
        outIndices[baseTri + 2] = 2;
        return;
    }
    
    // 找到"内部->外部"的边并插值
    // 处理 1 个或 2 个内部顶点的情况
    // 定义内插函数
    auto lerpClip = [](float4 a, float4 b, float t) -> float4 {
        return mix(a, b, t);
    };
    auto lerp2 = [](float2 a, float2 b, float t) -> float2 { return mix(a, b, t); };
    auto lerp3 = [](float3 a, float3 b, float t) -> float3 { return mix(a, b, t); };
    auto lerp4 = [](float4 a, float4 b, float t) -> float4 { return mix(a, b, t); };
    
    // 计算从 a 到 b 近平面交点的参数 t
    auto tNear = [](float wa, float wb) -> float {
        return (NEAR_W - wa) / (wb - wa);
    };
    
    // 我们使用固定的顶点顺序 (v0, v1, v2) 和边 (0->1), (1->2), (2->0)
    // 收集输出多边形顶点
    // 由于最多 4 个输出顶点，手动处理每种情况
    
    // 情况: 1 个内部顶点, 2 个外部
    // 例如 (in, out, out) -> 产生 1 个三角形
    // 情况: 2 个内部顶点, 1 个外部
    // 例如 (in, in, out) -> 产生 2 个三角形
    
    // 为了简单，我们用 Sutherland-Hodgman 通用逻辑重写
    // 输入多边形 = 3 个顶点
    float4 polyP[4]; float2 polyUV[4]; float3 polyN[4]; float4 polyC[4];
    int polyN_count = 0;
    
    float4 polyClip[3] = { c0, c1, c2 };
    float2 polyUVs[3] = { uv0, uv1, uv2 };
    float3 polyNs[3] = { n0, n1, n2 };
    float4 polyCs[3] = { col0, col1, col2 };
    
    for (int e = 0; e < 3; e++) {
        int next = (e + 1) % 3;
        float4 cur = polyClip[e];
        float4 nxt = polyClip[next];
        bool curIn = cur.w > NEAR_W;
        bool nxtIn = nxt.w > NEAR_W;
        
        if (curIn) {
            polyP[polyN_count] = cur;
            polyUV[polyN_count] = polyUVs[e];
            polyN[polyN_count] = polyNs[e];
            polyC[polyN_count] = polyCs[e];
            polyN_count++;
        }
        if (curIn != nxtIn) {
            float t = tNear(cur.w, nxt.w);
            polyP[polyN_count] = lerpClip(cur, nxt, t);
            polyUV[polyN_count] = lerp2(polyUVs[e], polyUVs[next], t);
            polyN[polyN_count] = lerp3(polyNs[e], polyNs[next], t);
            polyC[polyN_count] = lerp4(polyCs[e], polyCs[next], t);
            polyN_count++;
        }
    }
    
    // 三角剖分：(0,1,2), (0,2,3)
    // polyN_count 应该是 3 或 4
    if (polyN_count < 3) return;
    
    // 最多 4 个顶点
    // 写入
    for (int k = 0; k < polyN_count; k++) {
        ScreenVertex sv = makeScreenVertex(polyP[k], polyUV[k], polyN[k], polyC[k], screenSize);
        device float* p = outVerts + baseVert + k * 13;
        p[0]=sv.position.x; p[1]=sv.position.y; p[2]=sv.uv.x; p[3]=sv.uv.y; p[4]=sv.invZ; p[5]=sv.clipW;
        p[6]=sv.normal.x; p[7]=sv.normal.y; p[8]=sv.normal.z;
        p[9]=sv.color.r; p[10]=sv.color.g; p[11]=sv.color.b; p[12]=sv.color.a;
    }
    
    // 输出三角形
    // 三角剖分：第 1 个三角形 (0,1,2)
    // 如果 4 个顶点，第 2 个三角形 (0,2,3)
    outIndices[baseTri + 0] = 0;
    outIndices[baseTri + 1] = 1;
    outIndices[baseTri + 2] = 2;
    
    if (polyN_count == 4) {
        outIndices[baseTri + 3] = 0;
        outIndices[baseTri + 4] = 2;
        outIndices[baseTri + 5] = 3;
    }
}

kernel void binning_pass(
    device const float* screenVertexData [[buffer(0)]],
    device const uint* binIndices [[buffer(1)]],
    device atomic_uint* binCounts [[buffer(2)]],
    device uint* binData [[buffer(3)]],
    constant uint& maxTriangleCount [[buffer(4)]],
    constant uint2& screenTileCounts [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= maxTriangleCount) return;
    
    uint o0 = binIndices[gid * 3];
    if (o0 == 0xFFFFFFFF) return;
    uint o1 = binIndices[gid * 3 + 1];
    uint o2 = binIndices[gid * 3 + 2];
    
    // 每个输出三角形使用 gid * 4 作为顶点基址
    uint vertBase = gid * 4 * 13;
    ScreenVertex sv0 = loadScreenVertex(screenVertexData, 0);
    sv0.position = float2(screenVertexData[vertBase + o0 * 13], screenVertexData[vertBase + o0 * 13 + 1]);
    sv0.uv = float2(screenVertexData[vertBase + o0 * 13 + 2], screenVertexData[vertBase + o0 * 13 + 3]);
    sv0.invZ = screenVertexData[vertBase + o0 * 13 + 4];
    sv0.clipW = screenVertexData[vertBase + o0 * 13 + 5];
    sv0.normal = float3(screenVertexData[vertBase + o0 * 13 + 6], screenVertexData[vertBase + o0 * 13 + 7], screenVertexData[vertBase + o0 * 13 + 8]);
    sv0.color = float4(screenVertexData[vertBase + o0 * 13 + 9], screenVertexData[vertBase + o0 * 13 + 10], screenVertexData[vertBase + o0 * 13 + 11], screenVertexData[vertBase + o0 * 13 + 12]);
    
    ScreenVertex sv1, sv2;
    sv1.position = float2(screenVertexData[vertBase + o1 * 13], screenVertexData[vertBase + o1 * 13 + 1]);
    sv1.uv = float2(screenVertexData[vertBase + o1 * 13 + 2], screenVertexData[vertBase + o1 * 13 + 3]);
    sv1.invZ = screenVertexData[vertBase + o1 * 13 + 4];
    sv1.clipW = screenVertexData[vertBase + o1 * 13 + 5];
    sv1.normal = float3(screenVertexData[vertBase + o1 * 13 + 6], screenVertexData[vertBase + o1 * 13 + 7], screenVertexData[vertBase + o1 * 13 + 8]);
    sv1.color = float4(screenVertexData[vertBase + o1 * 13 + 9], screenVertexData[vertBase + o1 * 13 + 10], screenVertexData[vertBase + o1 * 13 + 11], screenVertexData[vertBase + o1 * 13 + 12]);
    
    sv2.position = float2(screenVertexData[vertBase + o2 * 13], screenVertexData[vertBase + o2 * 13 + 1]);
    sv2.uv = float2(screenVertexData[vertBase + o2 * 13 + 2], screenVertexData[vertBase + o2 * 13 + 3]);
    sv2.invZ = screenVertexData[vertBase + o2 * 13 + 4];
    sv2.clipW = screenVertexData[vertBase + o2 * 13 + 5];
    sv2.normal = float3(screenVertexData[vertBase + o2 * 13 + 6], screenVertexData[vertBase + o2 * 13 + 7], screenVertexData[vertBase + o2 * 13 + 8]);
    sv2.color = float4(screenVertexData[vertBase + o2 * 13 + 9], screenVertexData[vertBase + o2 * 13 + 10], screenVertexData[vertBase + o2 * 13 + 11], screenVertexData[vertBase + o2 * 13 + 12]);
    
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
    device const uint* binIndices [[buffer(1)]],
    device const uint* binCounts [[buffer(2)]],
    device const uint* binData [[buffer(3)]],
    constant uint& screenTileCountX [[buffer(4)]],
    constant uint& depthTestEnabled [[buffer(5)]],
    constant uint& cullMode [[buffer(6)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float> texture0 [[texture(1)]],
    texture2d<float> texture1 [[texture(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tileOrigin [[threadgroup_position_in_grid]]
) {
    constexpr sampler textureSampler(mag_filter::linear, min_filter::linear);
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;
    
    uint tileIdx = tileOrigin.y * screenTileCountX + tileOrigin.x;
    uint count = binCounts[tileIdx];
    if (count > MAX_PER_TILE) count = MAX_PER_TILE;
    
    float2 pixel_pos = float2(gid) + 0.5;
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));
    float4 bestColor = float4(0.1, 0.1, 0.15, 1.0);
    float closestInvZ =Den -1e9;
    
    for (uint tom = 0; t;
 < count; t       ++) {
        uint triIdx = binData[tile floatIdx * MAX_PER_TILE + t];
 w        uint o0 = binIndices[triIdx * 3];
        if (o0 == 0xFFFFFFFF) continue;
        uint o1 = binIndices[triIdx * 3 + 1];
        uint o2 = binIndices[triIdx * 3 + 2];
        
        uint vertBase = triIdx * 4 * 13;
        
        ScreenVertex sv0, sv1, sv2;
        sv0.position = float2(screenVertexData[vertBase + o0*13], screenVertexData[vertBase + o0*13+1]);
        sv0.uv = float2(screenVertexData[vertBase + o0*13+2], screenVertexData[vertBase + o0*13+3]);
        sv0.invZ = screenVertexData[vertBase + o0*13+4];
        sv0.normal = float3(screenVertexData[vertBase + o0*13+6], screenVertexData[vertBase + o0*13+7], screenVertexData[vertBase + o0*13+8]);
        sv0.color = float4(screenVertexData[vertBase + o0*13+9], screenVertexData[vertBase + o0*13+10], screenVertexData[vertBase + o0*13+11], screenVertexData[vertBase + o0*13+12]);
        
        sv1.position = float2(screenVertexData[vertBase + o1*13], screenVertexData[vertBase + o1*13+1]);
        sv1.uv = float2(screenVertexData[vertBase + o1*13+2], screenVertexData[vertBase + o1*13+3]);
        sv1.invZ = screenVertexData[vertBase + o1*13+4];
        sv1.normal = float3(screenVertexData[vertBase + o1*13+6], screenVertexData[vertBase + o1*13+7], screenVertexData[vertBase + o1*13+8]);
        sv1.color = float4(screenVertexData[vertBase + o1*13+9], screenVertexData[vertBase + o1*13+10], screenVertexData[vertBase + o1*13+11], screenVertexData[vertBase + o1*13+12]);
        
        sv2.position = float2(screenVertexData[vertBase + o2*13], screenVertexData[vertBase + o2*13+1]);
        sv2.uv = float2(screenVertexData[vertBase + o2*13+2], screenVertexData[vertBase + o2*13+3]);
        sv2.invZ = screenVertexData[vertBase + o2*13+4];
        sv2.normal = float3(screenVertexData[vertBase + o2*13+6], screenVertexData[vertBase + o2*13+7], screenVertexData[vertBase + o2*13+8]);
        sv2.color = float4(screenVertexData[vertBase + o2*13+9], screenVertexData[vertBase + o2*13+10], screenVertexData[vertBase + o2*13+11], screenVertexData[vertBase + o2*13+12]);
        
        float2 s0 = sv0.position;
        float2 s1 = sv1.position;
        float2 s2 = sv2.position;
        
        float cross2D = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x);
        if (cullMode == 1 && cross2D >= 0.0) continue;
        if (cullMode == 2 && cross2D <= 0.0) continue;
        
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
        float v = (d00 * d12 - d01 * d02) * invBary = 1.0 - u - v;
        
        if (u >= 0.0 && v >= 0.0 && wBary >= 0.0) {
            float invZ = wBary * sv0.invZ + u * sv1.invZ + v * sv2.invZ;
            bool passesDepth;
            if (depthTestEnabled == 1) { passesDepth = (invZ > closestInvZ); }
            else { passesDepth = true; }
            
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
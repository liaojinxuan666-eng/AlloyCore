#include <metal_stdlib>
using namespace metal;

constant int TILE_SIZE = 16;
constant int MAX_PER_TILE = 256;
constant float NEAR_W = 0.001;

struct ScreenVertex {
    float2 position;
    float2 uv;
    float invZ;
    float clipW;
    float3 normal;
    float4 color;
};

inline ScreenVertex loadFromSlot(device const float* verts, uint inputTriIdx, uint localVertIdx) {
    uint base = inputTriIdx * 4 * 13;
    uint off = base + localVertIdx * 13;
    ScreenVertex v;
    v.position = float2(verts[off], verts[off+1]);
    v.uv = float2(verts[off+2], verts[off+3]);
    v.invZ = verts[off+4];
    v.clipW = verts[off+5];
    v.normal = float3(verts[off+6], verts[off+7], verts[off+8]);
    v.color = float4(verts[off+9], verts[off+10], verts[off+11], verts[off+12]);
    return v;
}

inline ScreenVertex makeScreenVertex(float4 clip, float2 uv, float3 nrm, float4 color, float2 screenSize) {
    ScreenVertex v;
    float w = max(clip.w, 0.0001);
    float2 ndc = clip.xy / w;
    v.position = float2((ndc.x + 1.0) * 0.5 * screenSize.x, (1.0 - ndc.y) * 0.5 * screenSize.y);
    v.invZ = 1.0 / w;
    v.uv = uv;
    v.clipW = clip.w;
    v.normal = nrm;
    v.color = color;
    return v;
}

inline void storeScreenVertex(device float* p, ScreenVertex v) {
    p[0]  = v.position.x;
    p[1]  = v.position.y;
    p[2]  = v.uv.x;
    p[3]  = v.uv.y;
    p[4]  = v.invZ;
    p[5]  = v.clipW;
    p[6]  = v.normal.x;
    p[7]  = v.normal.y;
    p[8]  = v.normal.z;
    p[9]  = v.color.r;
    p[10] = v.color.g;
    p[11] = v.color.b;
    p[12] = v.color.a;
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
    float4 col = float4(inputVBO[src+3], inputVBO[src+4], inputVBO[src+5], inputVBO[src+6]);
    float2 uv = float2(inputVBO[src+7], inputVBO[src+8]);
    float3 nrm = float3(inputVBO[src+9], inputVBO[src+10], inputVBO[src+11]);

    float4 clip = transform * float4(pos, 1.0);
    float3 vN = (transform * float4(nrm, 0.0)).xyz;

    uint dst = gid * 13;
    outputVBO[dst+0]  = clip.x;
    outputVBO[dst+1]  = clip.y;
    outputVBO[dst+2]  = clip.z;
    outputVBO[dst+3]  = clip.w;
    outputVBO[dst+4]  = uv.x;
    outputVBO[dst+5]  = uv.y;
    outputVBO[dst+6]  = vN.x;
    outputVBO[dst+7]  = vN.y;
    outputVBO[dst+8]  = vN.z;
    outputVBO[dst+9]  = col.r;
    outputVBO[dst+10] = col.g;
    outputVBO[dst+11] = col.b;
    outputVBO[dst+12] = col.a;
}

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

    float4 c[3];
    float2 uv[3];
    float3 n[3];
    float4 col[3];

    c[0] = float4(inVerts[i0*13], inVerts[i0*13+1], inVerts[i0*13+2], inVerts[i0*13+3]);
    c[1] = float4(inVerts[i1*13], inVerts[i1*13+1], inVerts[i1*13+2], inVerts[i1*13+3]);
    c[2] = float4(inVerts[i2*13], inVerts[i2*13+1], inVerts[i2*13+2], inVerts[i2*13+3]);

    uv[0] = float2(inVerts[i0*13+4], inVerts[i0*13+5]);
    uv[1] = float2(inVerts[i1*13+4], inVerts[i1*13+5]);
    uv[2] = float2(inVerts[i2*13+4], inVerts[i2*13+5]);

    n[0] = float3(inVerts[i0*13+6], inVerts[i0*13+7], inVerts[i0*13+8]);
    n[1] = float3(inVerts[i1*13+6], inVerts[i1*13+7], inVerts[i1*13+8]);
    n[2] = float3(inVerts[i2*13+6], inVerts[i2*13+7], inVerts[i2*13+8]);

    col[0] = float4(inVerts[i0*13+9], inVerts[i0*13+10], inVerts[i0*13+11], inVerts[i0*13+12]);
    col[1] = float4(inVerts[i1*13+9], inVerts[i1*13+10], inVerts[i1*13+11], inVerts[i1*13+12]);
    col[2] = float4(inVerts[i2*13+9], inVerts[i2*13+10], inVerts[i2*13+11], inVerts[i2*13+12]);

    uint baseTri = gid * 6;
    outIndices[baseTri + 0] = 0xFFFFFFFF;
    outIndices[baseTri + 1] = 0xFFFFFFFF;
    outIndices[baseTri + 2] = 0xFFFFFFFF;
    outIndices[baseTri + 3] = 0xFFFFFFFF;
    outIndices[baseTri + 4] = 0xFFFFFFFF;
    outIndices[baseTri + 5] = 0xFFFFFFFF;

    float4 polyC[4];
    float2 polyUV[4];
    float3 polyN[4];
    float4 polyCol[4];
    int polyCount = 0;

    for (int e = 0; e < 3; e++) {
        int nx = (e + 1) % 3;
        bool curIn = c[e].w > NEAR_W;
        bool nxtIn = c[nx].w > NEAR_W;

        if (curIn) {
            polyC[polyCount] = c[e];
            polyUV[polyCount] = uv[e];
            polyN[polyCount] = n[e];
            polyCol[polyCount] = col[e];
            polyCount++;
        }
        if (curIn != nxtIn) {
            float t = (NEAR_W - c[e].w) / (c[nx].w - c[e].w);
            polyC[polyCount] = mix(c[e], c[nx], t);
            polyUV[polyCount] = mix(uv[e], uv[nx], t);
            polyN[polyCount] = mix(n[e], n[nx], t);
            polyCol[polyCount] = mix(col[e], col[nx], t);
            polyCount++;
        }
    }

    if (polyCount < 3) return;

    uint baseVert = gid * 4 * 13;
    for (int k = 0; k < polyCount; k++) {
        ScreenVertex sv = makeScreenVertex(polyC[k], polyUV[k], polyN[k], polyCol[k], screenSize);
        device float* p = outVerts + baseVert + k * 13;
        storeScreenVertex(p, sv);
    }

    outIndices[baseTri + 0] = 0;
    outIndices[baseTri + 1] = 1;
    outIndices[baseTri + 2] = 2;

    if (polyCount == 4) {
        outIndices[baseTri + 3] = 0;
        outIndices[baseTri + 4] = 2;
        outIndices[baseTri + 5] = 3;
    }
}

kernel void binning_pass(
    device const float* outVerts [[buffer(0)]],
    device const uint* outIndices [[buffer(1)]],
    device atomic_uint* binCounts [[buffer(2)]],
    device uint* binData [[buffer(3)]],
    constant uint& totalOutputSlots [[buffer(4)]],
    constant uint2& screenTileCounts [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= totalOutputSlots) return;

    uint o0 = outIndices[gid * 3];
    if (o0 == 0xFFFFFFFF) return;
    uint o1 = outIndices[gid * 3 + 1];
    uint o2 = outIndices[gid * 3 + 2];

    uint inputTriIdx = gid / 2;
    uint base = inputTriIdx * 4 * 13;

    float2 p0 = float2(outVerts[base + o0*13], outVerts[base + o0*13 + 1]);
    float2 p1 = float2(outVerts[base + o1*13], outVerts[base + o1*13 + 1]);
    float2 p2 = float2(outVerts[base + o2*13], outVerts[base + o2*13 + 1]);

    float minX = min(min(p0.x, p1.x), p2.x);
    float maxX = max(max(p0.x, p1.x), p2.x);
    float minY = min(min(p0.y, p1.y), p2.y);
    float maxY = max(max(p0.y, p1.y), p2.y);

    float sw = float(screenTileCounts.x * TILE_SIZE);
    float sh = float(screenTileCounts.y * TILE_SIZE);
    if (maxX < 0.0 || minX >= sw) return;
    if (maxY < 0.0 || minY >= sh) return;

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
            } else {
                atomic_fetch_sub_explicit(&binCounts[tileIdx], 1, memory_order_relaxed);
            }
        }
    }
}

kernel void rasterize_pass(
    device const float* outVerts [[buffer(0)]],
    device const uint* outIndices [[buffer(1)]],
    device const uint* binCounts [[buffer(2)]],
    device const uint* binData [[buffer(3)]],
    constant uint& screenTileCountX [[buffer(4)]],
    constant uint& depthTestEnabled [[buffer(5)]],
    constant uint& cullMode [[buffer(6)]],
    device const uint* triTexIDs [[buffer(7)]],
    texture2d<float, access::write> output [[texture(0)]],
    texture2d<float> tex0 [[texture(1)]],
    texture2d<float> tex1 [[texture(2)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tileOrigin [[threadgroup_position_in_grid]]
) {
    constexpr sampler texSampler(mag_filter::linear, min_filter::linear);
    if (gid.x >= output.get_width() || gid.y >= output.get_height()) return;

    uint tileIdx = tileOrigin.y * screenTileCountX + tileOrigin.x;
    uint count = binCounts[tileIdx];
    if (count > MAX_PER_TILE) count = MAX_PER_TILE;

    float2 pixel = float2(gid) + 0.5;
    float3 lightDir = normalize(float3(0.5, 1.0, 0.5));
    float4 bestColor = float4(0.1, 0.1, 0.15, 1.0);
    float closestInvZ = -1e9;

    for (uint t = 0; t < count; t++) {
        uint slotIdx = binData[tileIdx * MAX_PER_TILE + t];
        uint o0 = outIndices[slotIdx * 3];
        if (o0 == 0xFFFFFFFF) continue;
        uint o1 = outIndices[slotIdx * 3 + 1];
        uint o2 = outIndices[slotIdx * 3 + 2];

        uint inputTriIdx = slotIdx / 2;
        ScreenVertex v0 = loadFromSlot(outVerts, inputTriIdx, o0);
        ScreenVertex v1 = loadFromSlot(outVerts, inputTriIdx, o1);
        ScreenVertex v2 = loadFromSlot(outVerts, inputTriIdx, o2);

        float2 s0 = v0.position;
        float2 s1 = v1.position;
        float2 s2 = v2.position;

        float cross2D = (s1.x - s0.x) * (s2.y - s0.y) - (s1.y - s0.y) * (s2.x - s0.x);
        if (cullMode == 1 && cross2D <= 0.0) continue;
        if (cullMode == 2 && cross2D >= 0.0) continue;

        float2 e0 = s1 - s0;
        float2 e1 = s2 - s0;
        float2 e2 = pixel - s0;

        float d00 = dot(e0, e0);
        float d01 = dot(e0, e1);
        float d02 = dot(e0, e2);
        float d11 = dot(e1, e1);
        float d12 = dot(e1, e2);

        float det = d00 * d11 - d01 * d01;
        if (abs(det) < 1e-9) continue;
        float invDen = 1.0 / det;
        float u = (d11 * d02 - d01 * d12) * invDen;
        float v = (d00 * d12 - d01 * d02) * invDen;
        float w = 1.0 - u - v;

        if (u >= 0.0 && v >= 0.0 && w >= 0.0) {
            float iz0 = v0.invZ;
            float iz1 = v1.invZ;
            float iz2 = v2.invZ;
            float invZ = w * iz0 + u * iz1 + v * iz2;
            bool passes = (depthTestEnabled == 0) || (invZ > closestInvZ);
            if (passes) {
                closestInvZ = invZ;
                float2 uvI = (w * v0.uv * iz0 + u * v1.uv * iz1 + v * v2.uv * iz2) / invZ;
                float4 colI = (w * v0.color * iz0 + u * v1.color * iz1 + v * v2.color * iz2) / invZ;
                float3 nrm = normalize(w * normalize(v0.normal) + u * normalize(v1.normal) + v * normalize(v2.normal));
                uint tid = triTexIDs[inputTriIdx];
                float4 texColor = (tid == 1) ? tex1.sample(texSampler, uvI) : tex0.sample(texSampler, uvI);
                float intensity = max(dot(nrm, lightDir), 0.2);
                bestColor = colI * texColor * intensity;
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
    highRes.write(lowRes.sample(s, uv), gid);
}
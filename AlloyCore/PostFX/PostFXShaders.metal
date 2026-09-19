#include <metal_stdlib>
using namespace metal;

// AlloySR：超分辨率放大 + 锐化
kernel void upscale_pass(
    texture2d<float, access::sample> lowRes [[texture(0)]],
    texture2d<float, access::write> highRes [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    float2 uv = (float2(gid) + 0.5) / float2(highRes.get_width(), highRes.get_height());
    float4 center = lowRes.sample(sampler(mag_filter::linear, min_filter::linear), uv);
    
    float2 texelSize = 1.0 / float2(lowRes.get_width(), lowRes.get_height());
    
    float4 up = lowRes.sample(sampler(mag_filter::linear), uv + float2(0, -texelSize.y));
    float4 down = lowRes.sample(sampler(mag_filter::linear), uv + float2(0, texelSize.y));
    float4 left = lowRes.sample(sampler(mag_filter::linear), uv + float2(-texelSize.x, 0));
    float4 right = lowRes.sample(sampler(mag_filter::linear), uv + float2(texelSize.x, 0));
    
    float4 neighbors = (up + down + left + right) * 0.25;
    float4 edge = center - neighbors;
    
    float sharpness = 0.8;
    float4 finalColor = center + edge * sharpness;
    
    highRes.write(finalColor, gid);
}

// 🔥 新增：边缘平滑抗锯齿（FXAA 简化版）
kernel void aa_pass(
    texture2d<float, access::read> inputTex [[texture(0)]],
    texture2d<float, access::write> outputTex [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // 获取上下左右相邻像素
    float3 center = inputTex.read(gid).rgb;
    float3 top    = inputTex.read(gid + uint2(0, 1)).rgb;
    float3 bottom = inputTex.read(gid - uint2(0, 1)).rgb;
    float3 left   = inputTex.read(gid - uint2(1, 0)).rgb;
    float3 right  = inputTex.read(gid + uint2(1, 0)).rgb;
    
    // 计算周围像素平均值
    float3 neighbors = (top + bottom + left + right) * 0.25;
    
    // 计算边缘强度
    float edgeStrength = length(center - neighbors);
    
    // 如果边缘强度超过阈值，进行平滑处理
    if (edgeStrength > 0.15) {
        float3 smoothed = (center + neighbors) * 0.5;
        outputTex.write(float4(smoothed, 1.0), gid);
    } else {
        outputTex.write(float4(center, 1.0), gid);
    }
}
#include <metal_stdlib>
using namespace metal;

// AlloySR：超分辨率放大 + 锐化
kernel void upscale_pass(
    texture2d<float, access::sample> lowRes [[texture(0)]], // 🔥 改为 access::sample
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
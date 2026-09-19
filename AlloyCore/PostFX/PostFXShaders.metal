#include <metal_stdlib>
using namespace metal;

// 降采样：把 2x 分辨率纹理完美缩回 1x 屏幕
kernel void downsample_pass(
    texture2d<float, access::sample> highRes [[texture(0)]],
    texture2d<float, access::write> outputTex [[texture(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // 计算当前低分辨率像素对应的高分辨率 UV 坐标
    float2 uv = (float2(gid) + 0.5) / float2(outputTex.get_width(), outputTex.get_height());
    
    // 采样高分辨率纹理，使用线性插值自动混合周围 4 个像素
    constexpr sampler s(mag_filter::linear, min_filter::linear);
    float4 color = highRes.sample(s, uv);
    
    // 也可以手动做 2x2 平均（这里用线性插值其实已经包含了平均效果）
    outputTex.write(color, gid);
}
# AlloyCore

在 iOS 上用 Metal Compute Shader 构建的**虚拟 GPU**（Virtual GPU）。目标：为 Switch（NVN）、Windows（D3D11/12）、Vulkan 等图形 API 提供一个中立的图形抽象层（GAL），让它们只需翻译到 GAL，而无需各自实现 Metal 后端。

项目始于 2026-09-19，Apache-2.0 协议。

> **注意**：仓库名与 [alloy-rs/core](https://github.com/alloy-rs/core)（Rust 以太坊库）和 Disney 的 alloy-core（Smithy）无关，本项目是 iOS 图形技术。

---

## 项目定位

传统做法：
 
D3D11 游戏 → D3DMetal → Metal（苹果官方，闭源）
Vulkan 应用 → MoltenVK → Metal（Khronos，翻译层）
Switch 游戏 → MeloNX → MoltenVK → Metal（多层翻译）

AlloyCore 的做法：
 
D3D11 / NVN / Vulkan 前端
│
▼
[AlloyGAL]      ← 中立的图形抽象层
│
▼
[AlloyRenderer] ← Metal Compute 后端
│
▼
Apple GPU

**核心思想**：所有图形 API 都翻译到 GAL 的同一套中立接口。加新 API 支持只需在 `Frontends/XXX/` 加文件，不动核心。

---

## 架构总览
 
AlloyCore/
├── Interface/                    GAL 中立接口层（地基，稳定）
│   ├── AlloyTypes.swift          资源句柄、描述符、AlloyLog
│   ├── AlloyOpcode.swift         协议单份定义（opcode + 长度表）
│   ├── AlloyGAL.swift            GAL 主体
│   └── AlloyFrontend.swift       前端接入协议
├── Renderer/
│   └── AlloyRenderer.swift       Metal Compute 后端
├── Shaders/
│   └── Shaders.metal             八个 Compute Kernel
├── Frontends/                    各 API 前端（可插拔）
│   └── NVN/
│       └── NVNFrontend.swift     NVN 前端骨架（v0.7.0，接口侧）
└── PostFX/                       AA/SR/SSAA（未接线，死代码）

`TestApp/` 是宿主 iOS 应用，展示如何调用 GAL。

---

## 渲染管线（五级 Compute Pass）
 
[模型空间顶点]
│ 1. geometry_pass
▼
[裁剪空间 clip.xyzw + viewNormal]
│ 2. clip_project_pass (Sutherland-Hodgman 近平面裁剪)
▼
[屏幕空间三角形 + 每个输入三角形最多 2 个输出三角形]
│ 3. binning_count_pass → binning_offset_pass → binning_fill_pass
│    (Tile Binning，三趟；无每 tile 容量上限)
▼
[][每个 tile 的三角形索引列表]
│ 4. rasterize_pass (每线程组处理一个 tile)
│    含 8-way depth compare、Scissor 剔除
▼
[低分辨率纹理 (renderScale = 0.5)]
│ 5. upscale_pass
▼
[全屏纹理]

**关键特性**：
- 透视校正插值（UV 和颜色都用 invZ 加权）
- 近平面裁剪（避免 w≤0 除零和三角形翻转）
- Tile Binning（O(triangles) 而非 O(tiles × triangles)）
- 三趟 binning：无每 tile 容量上限（v0.5.0）
- 8 种深度比较函数（v0.6.0）
- Scissor Rect（v0.6.0）
- 完整 Blend 状态编码（v0.6.0，待 Render Pass 时启用）
- Edge-function 光栅化（v0.6.1）
- GPU Compute 顶点动画（v0.4.0）
- 背面剔除
- Early-Z（逐像素深度测试）

---

## Compute Dispatch（v0.4.0）

GAL 支持在渲染前uv派Off发通用 Compute Kernel——"我们的 CUDA"。

**接口**（状态机模型，对应][ D3D11 / Vulkan 的 Bind → Dispatch）：

n```swift
// 创建（一次性）
gal.createComputePipeline(AlloyComputePipelineDescriptor(
    shaderName: "vertex_animate_pass",
    threadsPerThreadgroup: SIMD3<UInt32>(64, 1, 1)
)) -> AlloyComputePipelineHandle

// 帧内（每帧）
gal.computeTime = time             // 传给 kernel 的时间 uniform
gal.bindComputePipeline(handle)
gal.bindComputeBuffer(slot: 0, handle: vertexBufferHandle)
gal.dispatchCompute(groups: SIMD3<UInt32>(threadCount, 1, 1))
 
执行时序：所有 compute dispatch 在 geometry_pass 之前 一次性执行。
适用于"GPU 顶点动画"这类"compute 改顶点 → 渲染读顶点"的场景。
真正的"draw 之间插 compute"是未来的事。
 
协议 opcode：
• 0x10 BIND_COMPUTE_PIPELINE [handle]
• 0x11 BIND_COMPUTE_VERTEX_POOL [slot][poolOffsetFloats][byteOffsetFloats]
• 0x12 COMPUTE_DISPATCH [threadCount][1][1] 
 
指令流协议（opcode）
 
GAL 与 Renderer 之间的私有协议，[UInt32] 数组。唯一真相源是 AlloyOpcode.swift。
Opcode 名称 参数 总长度
0x01 DRAW_INDEXED [globalIndexStart][indexCount][textureID] 4
0x02 CLEAR_COLOR [r][g][b][a] (bitPattern) 5
0x03 BIND_PIPELINE [depthTest][cullMode][blend][shaderID][depthFunc][depthWrite][srcCol][dstCol][colOp][srcAlpha][dstAlpha][alphaOp] 13
0x04 SET_VIEWPORT [w][h] (bitPattern) 3
0x05 (未使用) — —
0x06 SET_TRANSFORM [16 floats as bitPattern] 17
0x07 BIND_VERTEX_BUFFER [poolOffset] 2
0x08 BIND_INDEX_BUFFER [poolOffset] 2
0x09 UPDATE_VB [poolOffset][count][data0..dataN] 3+N
0x0A UPDATE_IB [poolOffset][count][data0..dataN] 3+N
0x10 BIND_COMPUTE_PIPELINE [handle] 2
0x11 BIND_COMPUTE_VERTEX_POOL [slot][poolOffsetFloats][byteOffsetFloats] 4
0x12 COMPUTE_DISPATCH [threadCount][1][1] 4
0x13 SET_SCISSOR [x][y][w][h] 5
0x14 SET_VERTEX_LAYOUT [stride][posOffrmOff][colOff] 6

 
修改协议时两处必须同步：
1. AlloyOpcode.swift 的 AlloyOpcode 枚举 + AlloyOpcodeLength.of
2. AlloyRenderer.swift 主 while 的 switch case 
extractTransform 不需要再手改——它已改为调用 AlloyOpcodeLength.of。 
 
GAL 接口（当前已实现）
 
帧生命周期：
beginFrame()
endFrame()
 
资源创建：
createVertexBuffer(data: [Float]) -> AlloyBufferHandle
createIndexBuffer(data: [UInt32], vertexHandle: AlloyBufferHandle) -> AlloyBufferHandle
createPipeline(desc: AlloyPipelineDescriptor) -> AlloyPipelineHandle
createTexture(desc: AlloyTextureDescriptor) -> AlloyTextureHandle
createComputePipeline(desc: AlloyComputePipelineDescriptor) -> AlloyComputePipelineHandle
 
资源销毁：
destroyPipeline(handle)
destroyTexture(handle)
destroyComputePipeline(handle)
 
命令录制：
clearColor(r, g, b, a)
bindPipeline(handle)
setViewport(width, height)
setScissor(x, y, width, height)             // v0.6.0
setVertexLayout(layout)                     // v0.7.0
setTransform(matrix: [Float])
 
分散状态 setter（v0.7.0）：
setDepthTestEnabled(bool)
setDepthWriteEnabled(bool)
setDepthCompareFunc(AlloyCompareFunc)
setCullMode(AlloyCullMode)
setBlendEnabled(bool)
setBlendFactors(srcColor, dstColor, colorOp, srcAlpha, dstAlpha, alphaOp)
setShaderID(UInt32)
 
动态 buffer 更新：
updateVertexBuffer(_ handle:, data:, offset:)
updateIndexBuffer(_ handle:, data:, offset:)
 
绘制：
drawIndexed(iboHandle:, indexCount:, firstIndex:, textureID:)
 
Compute：
var computeTime: Float
bindComputePipeline(handle)
bindComputeBuffer(slot:, handle:, byteOffsetFloats:)
dispatchCompute(groups: SIMD3<UInt32>)
 
数据访问：
getVertexPool() -> [Float]
getIndexPool() -> [UInt32]
getTexture(handle) -> MTLTexture?
 
提交：
submit(to: renderer, drawable) -> MTLCommandBuffer?
 
 
顶点格式
 
模型空间输入：
不再固定为"12 个 float"。由 AlloyVertexLayout 决定——stride + 4 个偏移（position / uv / normal / color）。偏移 -1 表示该属性缺失。
 
默认 layout（匹配 TestApp 当前顶点顺序）：
stride = 48
positionOffset = 0
uvOffset       = 28
normalOffset   = 36
colorOffset    = 12
 
裁剪空间中间（每顶点 13 个 float，固定）：
[0-3]  clip.xyzw
[4-5]  uv.xy
[6-8]  viewNormal.xyz
[9-12] color.rgba
 
屏幕空间输出（每顶点 13 个 float，固定）：
[0-1]  screenPosition.xy
[2-3]  uv.xy
[4]    invZ (1/clip.w)
[5]    clipW
[6-8]  viewNormal.xyz
[9-12] color.rgba
 
 
Shader Kernel 签名（Swift ↔ Metal buffer 索引必须严格对齐）
 
geometry_pass（v0.7.0 改为 raw-byte + layout 驱动）：
buffer(0): device const uchar* inputVBO
buffer(1): device float* outputVBO
buffer(2): constant uint& vertexCount
buffer(3): constant float4x4& transform
buffer(4): constant uint& vertexStride
buffer(5): constant int& positionOffset
buffer(6): constant int& uvOffset
buffer(7): constant int& normalOffset
buffer(8): constant int& colorOffset
 
clip_project_pass：
buffer(0): device const float* inVerts
buffer(1): device const uint* inIndices
buffer(2): device float* outVerts
buffer(3): device uint* outIndices
buffer(4): constant uint& triangleCount
buffer(5): constant float2& screenSize
 
binning_count_pass（v0.5.0）：
buffer(0): device const float* outVerts
buffer(1): device const uint* outIndices
buffer(2): device atomic_uint* binCounts
buffer(3): constant uint& totalOutputSlots
buffer(4): constant uint2& screenTileCounts
 
binning_offset_pass（v0.5.0）：
buffer(0): device uint* binCounts
buffer(1): device uint* binStarts
buffer(2): constant uint& numTiles
 
binning_fill_pass（v0.5.0）：
buffer(0): device const float* outVerts
buffer(1): device const uint* outIndices
buffer(2): device atomic_uint* binCounts
buffer(3): device uint* binData
buffer(4): device const uint* binStarts
buffer(5): constant uint& totalOutputSlots
buffer(6): constant uint2& screenTileCounts
buffer(7): constant uint& binDataCapacity
 
rasterize_pass（v0.6.0 扩展）：
buffer(0): device const float* outVerts
buffer(1): device const uint* outIndices
buffer(2): device const uint* binCounts
buffer(3): device const uint* binData
buffer(4): constant uint& screenTileCountX
buffer(5): constant uint& depthTestEnabled
buffer(6): constant uint& cullMode
buffer(7): device const uint* triTexIDs
buffer(8): device const uint* binStarts
buffer(9): constant uint& depthCompareFunc
buffer(10): constant uint4& scissor
texture(0): texture2d<float, access::write> output
texture(1): texture2d<float> tex0
texture(2): texture2d<float> tex1
 
upscale_pass：
texture(0): texture2d<float, access::sample> lowRes
texture(1): texture2d<float, access::write> highRes
 
vertex_animate_pass（v0.4.0）：
buffer(0): device float* verts
buffer(1): constant float& t
buffer(2): constant float& dt
buffer(3): constant uint& baseOffset
buffer(4): constant uint& count
buffer(5): device const float* origin
 
⚠️ 变更 shader buffer 索引时，Swift 端 setBuffer 的 index: 参数必须同步。 
 
NVN 前端（v0.7.0）
 
Frontends/NVN/NVNFrontend.swift 是 NVN 风格的接口层，不是游戏运行时。
 
当前做到的：
• 用 NVN 风格的 API（setDepthFunc / setCullFace / bindVertexBuffer / drawElements 等）重建一个立方体
• 渲染结果与直接调 GAL 完全一致
• 证明了"NVN 风格调用 → GAL → Metal Compute"的翻译链路是通的 
没做到的：
• 不拦截真实 NVN 调用——没有 dylib hook、没有符号替换、没有进程注入
• 不能跑 Switch 游戏——只支持约 15 个 NVN 调用，真实 NVN 有 200+ 个
• 只支持"单 vertex buffer + interleaved attributes"，多 attrib / 多 buffer 未做 
当前支持的 NVN 调用（映射到 GAL）：
NVN 风格 GAL
createBuffer(sizeBytes:) createVertexBuffer(data:)（先分配空 buffer）
uploadBuffer(id, data, offsetFloats:) updateVertexBuffer
bindVertexBuffer(id) 记录状态
bindIndexBuffer(id) 记录状态
bindTextureSlot(slot, unit:) 记录 texture slot
setViewport(x, y, w, h) setViewport(w, h)（x/y 暂忽略）
setScissor(x, y, w, h) setScissor
setDepthTestEnable(bool) setDepthTestEnabled
setDepthWriteEnable(bool) setDepthWriteEnabled
setDepthFunc(f) setDepthCompareFunc
setCullMode(enable, face) setCullMode（enable=false → .none）
setBlendEnable(bool) setBlendEnabled
setBlendFunc(...) setBlendFactors(...)
setVertexLayout(...) setVertexLayout(AlloyVertexLayout(...))
drawElements(indexCount, firstIndex, textureUnit) drawIndexed(...)

 
为什么是"接口侧"而非"拦截侧"：拦截真实 NVN 需要 Switch 内核模拟、ARM 二进制加载、符号替换——那是模拟器内核的工作，不在渲染层范围内。AlloyCore 的定位是"给别的模拟器用的后端"。
 
TestApp 的开关：useNVNForMesh0（默认 false）。true 时第一组立方体走 NVN 前端，其余走直接 GAL。两条路径画面等价。 
 
关键设计决策
 
1. 为什么用 [UInt32] 指令流而不是 [Float]？
 
统一类型，避免 Swift struct 和 Metal struct 的内存对齐问题。浮点通过 bitPattern 转换。
 
2. 为什么 binData 存 UInt32 而不是 uint2？
 
之前尝试过带 texID 的 uint2，但因为 Swift 和 Metal 的 buffer 索引错位，导致整个 pass 崩掉。目前版本移除了 texID，回归最简单的 UInt32。
 
3. 为什么 drawIndexed 接收 iboHandle 而不是 bind + startIndex？
 
无状态设计，避免"全局 currentIndexBufferOffset"这类容易出错的隐式状态。Vulkan/D3D12 精神。
 
4. 为什么 binning 从单趟改成三趟？（v0.5.0）
 
旧实现用 atomic_fetch_add 分配槽位，超过 MAX_PER_TILE = 256 就 atomic_fetch_sub 回退——溢出的三角形被静默丢弃。这在密集场景（大三角形横跨多个 tile）会丢几何。
 
新实现分三趟：
• binning_count_pass：只累加每个 tile 的计数器
• binning_offset_pass：单线程串行前缀和，算出每个 tile 在 binData 里的起点，顺便把 binCounts 清零
• binning_fill_pass：按前缀和偏移 + 每 tile 计数器原子递增，把三角形索引写入 binData 
收益：
• 每 tile 无 256 上限，唯一约束是 binData 总容量
• fill 阶段没有回退分支，写路径更干净
• 代价是两次额外的 encoder 切换（v0.5.0 实测 HUD 里 binning 从 0.03ms → 0.10ms，绝对值仍可忽略） 
5. 为什么 compute 不走指令流而走 Swift 对象？（v0.4.0）
 
Compute dispatch 结构复杂（shader 名、threadgroup 尺寸、多 buffer 绑定），编到 [UInt32] 流里会很丑。改为 GAL 存对象数组、submit 时传给 Renderer、Renderer 循环执行。和纹理数组同一模式。
 
6. 为什么 GPU 顶点动画读 origin 快照，而不是原地累加？（v0.4.0）
 
原地累加会漂移：sin(t) 的数值积分误差随时间放大，几十秒后几何炸开。正确做法是每帧 verts[i] = origin[i] + sin(t) * A——绝对值写入，永远不漂移。
 
7. 为什么深度比较用 invZ > closestInvZ 而不是 depth < closestDepth？（v0.6.0）
 
invZ = 1/w，越大越近。深度空间里"更近 = 深度值更小"，所以 D3D/Vulkan 的 .less（新片元更近）对应到我们的 invZ 空间就是 invZ > closestInvZ。映射表：
AlloyCompareFunc 逆 Z 空间判据
.never false
.less invZ > closestInvZ
.equal abs(invZ - closestInvZ) < ε
.lessEqual invZ >= closestInvZ
.greater invZ < closestInvZ
.notEqual abs(invZ - closestInvZ) >= ε
.greaterEqual invZ <= closestInvZ
.always true

 
8. 为什么 Scissor 默认用"哨兵值"而不是直接用全屏？（v0.6.0）
 
setScissor 是可选命令。GAL 默认编码 (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF) 作为哨兵，Renderer 检测到哨兵后填入全屏。这样"没调过 setScissor"和"调了全屏"在指令流里能区分——未来做 Render Pass 时有用。
 
9. 为什么 edge function 和 2×2 行列式解是等价的？（v0.6.1）
 
对三角形 s0, s1, s2 和点 P：
• area = cross(s1 - s0, s2 - s0)
• 三个 edge function E0(P), E1(P), E2(P) 对应三条边
• 内部点时：E0 + E1 + E2 = area，且同号
• 除以 absArea 归一化后，三个权重和恒为 1 
用 edge function 而不是行列式，是因为 E(P + dx) = E(P) + const——scanline 光栅器可以增量更新，不用每个像素重算。v0.6.1 只换了数学，还没做增量遍历。
 
10. 为什么 GAL 用"集中 PSO"而不是"分散状态机"？（v0.4.0）
 
集中式更贴近 D3D12 / Vulkan——一个 PSO 对象封装所有状态。但 D3D11 / NVN 是分散式的（SetDepthFunc 各一条）。
 
v0.7.0 加了内部状态缓存：分散 setter 更新 currentPipeline，draw 前 flush 成 0x03 指令。这样两种前端都能翻译——前端只看到自己习惯的风格，指令流始终是集中的。 
 
性能基线（iPhone，iOS 17+）
场景 三角形 FPS
单立方体 12 60
27 立方体阵列 324 60
3×27 立方体（当前 TestApp） 972 60
1000 立方体（10000+ 三角形） 12000 32

 
v0.4.0 变化：顶点动画从 CPU 侧 7776 次循环 / 帧 → GPU 内 648 线程 compute。
CPU 每帧仅编码 4 条 compute 指令。
 
v0.5.0 变化：HUD 显示 6 个 pass 的 CPU 编码耗时（compute / geometry / clip_project / binning / rasterize / upscale）。总编码耗时 < 0.2ms，60fps 预算的 1% 左右。
 
注意：截屏瞬间 FPS 会跌到 50 左右，是 iOS 系统合成开销，稳定状态 60。 
 
构建与 CI
• XcodeGen：project.yml 描述工程，xcodegen generate 生成 .xcodeproj
• GitHub Actions：每次 push 触发，产出未签名 IPA artifact
• 安装：用 TrollStore 或开发者证书侧载 
Actions 关键配置（.github/workflows/build.yml）：
• runner: macos-14
• Xcode 15.4（注意 objectVersion 强制降到 56）
• CODE_SIGNING_ALLOWED=NO 跳过签名
• 打包成 IPA 上传 artifact 
 
手机端编辑注意事项
 
不要用手机浏览器直接编辑长文件。 粘贴 200+ 行的 Metal 文件时，浏览器有极高概率丢字符、断行、插入乱码（历史失败案例：else if4)、outInd、poly floatCount、ci ras += 2 等）。
 
推荐方案：
1. GitHub Codespaces（浏览器打开仓库 → Code → Codespaces → Create）
2. Working Copy（iOS 上的 Git 客户端）
3. 电脑 + git clone 
中文注释是最高危因素——iOS 剪贴板处理多字节字符时容易出错。 
 
日志与调试
 
AlloyLog 提供环形日志缓冲（8 行），每 100ms 刷写到 Documents/alloy.log。日志通过 Info.plist 的 UIFileSharingEnabled 暴露到文件 App。
 
查看方式：「文件」App → 浏览 → 我的 iPhone → AlloyCore → alloy.log
 
TestApp 的 HUD 也会实时显示最近 8 行。 
 
Roadmap
 
v0.3.0 地基扩展 ✅
[x] 动态 buffer 更新（updateVertexBuffer / updateIndexBuffer）
[x] 纹理进 GAL（createTexture / bindTexture）
[x] 多纹理支持（per-triangle texID）
[x] 指令流越界保护
[x] 资源生命周期（poolVersion）
[x] 帧节流（semaphore = 2） 
v0.4.0 Compute & Protocol ✅
[x] 协议单份定义（AlloyOpcode.swift）
[x] Renderer 表驱动（消灭双份 while）
[x] Compute dispatch 接口（0x10/0x11/0x12）
[x] GPU 顶点动画（"我们的 CUDA"第一个用例） 
v0.5.0 Profiling & Binning ✅
[x] Per-pass CPU 计时（HUD 显示 6 个 pass）
[x] 三趟 binning（count / offset / fill）
[x] 移除 MAX_PER_TILE = 256 硬上限 
v0.6.0 Depth & Blend ✅
[x] 深度比较函数（8 种 AlloyCompareFunc）
[x] 深度写入开关（depthWriteEnabled 编码）
[x] 完整混合模式（AlloyBlendFactor × 12、AlloyBlendOp × 5）
[x] Scissor Rect（0x13 opcode）
[x] BIND_PIPELINE 从 5 词扩到 13 词 
v0.6.1 Edge-Function Rasterization ✅
[x] 用 edge function 替代 2×2 行列式解
[x] 数学等价、画面不变
[x] 为未来的 scanline 光栅器打基础 
v0.7.0 NVN Frontend（接口侧）✅
[x] 顶点格式进 GAL（AlloyVertexLayout + 0x14 opcode）
[x] 分散状态机（setDepthFunc / setCullMode / setBlendFactors 等）
[x] NVNFrontend 接口定义（15 个 NVN 风格调用）
[x] 翻译链路验证：NVN → GAL → Metal，画面一致 
v0.8.0 NVN Frontend（资源侧）
[ ] 多 vertex buffer / 分离 attrib
[ ] 多 texture slot（NVN 有 16 个，GAL 现在 2 个）
[ ] Sampler 资源（AlloySampler + createSampler）
[ ] Uniform buffer（bindUniformBuffer） 
v0.9.0 Render Pass
[ ] 渲染到纹理（多 render target）
[ ] 真正的深度缓冲区（让 depthWriteEnabled 生效）
[ ] 真正的 Blend（读 backbuffer）
[ ] 模板测试 
v1.0.0 着色器虚拟机
[ ] DXBC / DXIL 解析
[ ] SPIR-V 解析
[ ] 虚拟 ISA
[ ] 着色器缓存 
远期
[ ] D3D11 前端
[ ] Vulkan 前端
[ ] GPU 侧 per-pass 计时（MTLCounterSampleBuffer）
[ ] 性能基线自动化（CI 跑 benchmark，超阈值就红） 
 
已知问题
 
✅ 已修复
• binning 溢出：atomic_fetch_add 后判断 slot，溢出时 atomic_fetch_sub 回退（v0.3.0）
• 颜色插值未透视校正：colI 现在也用 iz0/iz1/iz2 加权（v0.3.0）
• cullMode 符号反了：cross2D <= 0.0 剔除背面（v0.3.0）
• geometry_pass 写错 dst+110 丢蓝通道（v0.4.0）
• 指令流无边界检查（越界崩溃）（v0.4.0）
• addCompletedHandler 在 commit() 之后调用（Metal 断言崩溃）（v0.4.0）
• updateVertexBuffer 缺失 count + 数据循环（流错位）（v0.4.0）
• MAX_PER_TILE = 256 硬上限（v0.5.0，三趟 binning 根治）
• 顶点格式硬编码 12 float（v0.7.0，改为 AlloyVertexLayout 驱动） 
⚠️ 待处理
• 命令协议单份解析：已部分解决（AlloyOpcodeLength.of），但主 while 里的 switch 仍需手改
• 资源只增不减：vertexPool / indexPool 只 append，无 destroy
• 接收但未实现的状态：clearColor、blendEnabled、depthWriteEnabled、viewport 被编码但部分被忽略（等 Render Pass）
• PostFX 是死代码：AlloyAA / AlloySR / AlloySSAA 未接入 renderer
• NVNFrontend 是接口侧：支持 15 个 NVN 调用，不拦截真实调用，不跑 Switch 游戏
• bindings.first 忽略 slot（v0.4.0 遗留）：多 buffer 绑定时会塌缩到第一个
• dispatchCompute(groups:) 名不副实：实际传的是线程数，不是 group 数
• dt 参数冗余：vertex_animate_pass 收到但未使用
• poolVersion bump 后动画跳回原点：uploadGeometry 重传后 origin 快照更新，compute 需重新同步
• MAX_PER_TILE warning：v0.5.0 后 kernel 不再使用，暂时保留作 binData 容量估算；未来 binData 动态分配时移除
• Scattered state 的 flushPipelineIfNeeded 只在 drawIndexed 调用：其他可能改变状态的地方未来也要记得 flush 
 
终极目标
[Windows x64 游戏]              [Switch 游戏]           [Vulkan 应用]
       │                              │                      │
       ▼                              ▼                      ▼
   Box64/FEX-Emu                  NVN 调用              Vulkan 调用
       │                              │                      │
       ▼                              │                      │
   Wine (Win32 API)                   │                      │
       │                              │                      │
       ▼                              │                      │
   D3D11 前端                         │                      │
       │                              │                      │
       └──────────────┬───────────────┴──────────────────────┘
                      ▼
              [AlloyGAL 中立抽象层]
                      │
                      ▼
              [AlloyCore 虚拟 GPU]
                      │
                      ▼
              [Metal Compute Backend]
                      │
                      ▼
              [Apple GPU]
 
iOS 上第一个用自建虚拟 GPU（而非 MoltenVK / D3DMetal）跑通 D3D11 / NVN / Vulkan 游戏的图形栈。这是 AlloyCore 的终极定位。 
 
联系方式 / 贡献
 
当前是个人研究项目，尚未对外开放贡献。任何 API 前端的实现都应满足：
1. 不改 Interface/ 的公开接口（除非确实需要通用能力）
2. 不改 Renderer/ 和 Shaders/（除非修复 bug）
3. 只加 Frontends/XXX/ 下的文件 
如果加前端时发现需要改核心，说明那是地基缺失的通用能力，应该先补地基。

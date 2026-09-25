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
│   ├── AlloyTypes.swift          资源句柄、描述符定义
│   ├── AlloyGAL.swift            GAL 主体
│   └── AlloyFrontend.swift       前端接入协议
├── Renderer/
│   └── AlloyRenderer.swift       Metal Compute 后端
├── Shaders/
│   └── Shaders.metal             五个 Compute Kernel
├── Frontends/                    各 API 前端（可插拔）
│   └── NVN/
│       └── NVNFrontend.swift     NVN 骨架（未填实）
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
│ 3. binning_pass (Tile Binning，每 16×16 tile 收集三角形)
▼
[每个 tile 的三角形索引列表]
│ 4. rasterize_pass (每线程组处理一个 tile)
▼
[低分辨率纹理 (renderScale = 0.5)]
│ 5. upscale_pass
▼
[全屏纹理]

**关键特性**：
- 透视校正插值（UV 和颜色都用 invZ 加权）
- 近平面裁剪（避免 w≤0 除零和三角形翻转）
- Tile Binning（O(triangles) 而非 O(tiles × triangles)）
- Early-Z（逐像素深度测试）
- 背面剔除

---

## 指令流协议（opcode）

GAL 与 Renderer 之间的私有协议，`[UInt32]` 数组。**修改协议时三处必须同步：**
1. `AlloyGAL.swift` 的 `frameCommandBuffer.append`
2. `AlloyRenderer.swift` 的 `while` 解析循环
3. `Shaders.metal` 的 `if opcode == ...` 分支

| Opcode | 名称 | 参数 | 总长度 |
|--------|------|------|--------|
| `0x01` | DRAW_INDEXED | `[globalIndexStart][indexCount][textureID]` | 4 uint |
| `0x02` | CLEAR_COLOR | `[r][g][b][a]` (bitPattern) | 5 uint |
| `0x03` | BIND_PIPELINE | `[depthTest][cullMode][blend][shaderID]` | 5 uint |
| `0x04` | SET_VIEWPORT | `[w][h]` (bitPattern) | 3 uint |
| `0x06` | SET_TRANSFORM | `[16 floats as bitPattern]` | 17 uint |
| `0x07` | BIND_VERTEX_BUFFER | `[poolOffset]` | 2 uint |
| `0x08` | BIND_INDEX_BUFFER | `[poolOffset]` | 2 uint |

**注意**：`0x05` 未使用，`0x07` / `0x08` 目前 GAL 已不再生成（改用 iboHandle 直接查表），Renderer 保留跳过分支。

---

## GAL 接口（当前已实现）

```swift
// 帧生命周期
beginFrame()
endFrame()

// 资源创建
createVertexBuffer(data: [Float]) -> AlloyBufferHandle
createIndexBuffer(data: [UInt32], vertexHandle: AlloyBufferHandle) -> AlloyBufferHandle
createPipeline(desc: AlloyPipelineDescriptor) -> AlloyPipelineHandle

// 资源销毁
destroyPipeline(handle)

// 命令录制
clearColor(r, g, b, a)
bindPipeline(handle)
setViewport(width, height)
setTransform(matrix: [Float])

// 绘制
drawIndexed(iboHandle: AlloyBufferHandle,
            indexCount: UInt32,
            firstIndex: UInt32,
            textureID: UInt32)

// 数据访问（供 Renderer 上传）
getVertexPool() -> [Float]
getIndexPool() -> [UInt32]

// 提交
submit(to: renderer, drawable, texture0, texture1) -> MTLCommandBuffer?
 
 
顶点格式
 
模型空间输入（每顶点 12 个 float）：
[0-2]  position.xyz
[3-6]  color.rgba
[7-8]  uv.xy
[9-11] normal.xyz
 
裁剪空间中间（每顶点 13 个 float）：
[0-3]  clip.xyzw
[4-5]  uv.xy
[6-8]  viewNormal.xyz
[9-12] color.rgba
 
屏幕空间输出（每顶点 13 个 float）：
[0-1]  screenPosition.xy
[2-3]  uv.xy
[4]    invZ (1/clip.w)
[5]    clipW
[6-8]  viewNormal.xyz
[9-12] color.rgba
 
 
Shader Kernel 签名（Swift ↔ Metal buffer 索引必须严格对齐）
 
geometry_pass
buffer(0): device const float* inputVBO
buffer(1): device float* outputVBO
buffer(2): constant uint& vertexCount
buffer(3): constant float4x4& transform
 
clip_project_pass
buffer(0): device const float* inVerts
buffer(1): device const uint* inIndices
buffer(2): device float* outVerts
buffer(3): device uint* outIndices
buffer(4): constant uint& triangleCount
buffer(5): constant float2& screenSize
 
binning_pass
buffer(0): device const float* outVerts
buffer(1): device const uint* outIndices
buffer(2): device atomic_uint* binCounts
buffer(3): device uint* binData
buffer(4): constant uint& totalOutputSlots
buffer(5): constant uint2& screenTileCounts
 
rasterize_pass
buffer(0): device const float* outVerts
buffer(1): device const uint* outIndices
buffer(2): device const uint* binCounts
buffer(3): device const uint* binData
buffer(4): constant uint& screenTileCountX
buffer(5): constant uint& depthTestEnabled
buffer(6): constant uint& cullMode
texture(0): texture2d<float, access::write> output
texture(1): texture2d<float> tex0
 
upscale_pass
texture(0): texture2d<float, access::sample> lowRes
texture(1): texture2d<float, access::write> highRes
 
⚠️ 变更 shader buffer 索引时，Swift 端 setBuffer 的 index: 参数必须同步。 
 
关键设计决策
 
1. 为什么用 [UInt32] 指令流而不是 [Float]？
 
统一类型，避免 Swift struct 和 Metal struct 的内存对齐问题。浮点通过 bitPattern 转换。
 
2. 为什么 binData 存 UInt32 而不是 uint2？
 
之前尝试过带 texID 的 uint2，但因为 Swift 和 Metal 的 buffer 索引错位，导致整个 pass 崩掉。目前版本移除了 texID，回归最简单的 UInt32。
 
3. 为什么 drawIndexed 接收 iboHandle 而不是 bind + startIndex？
 
无状态设计，避免"全局 currentIndexBufferOffset"这类容易出错的隐式状态。Vulkan/D3D12 精神。
 
4. 为什么 MAX_PER_TILE 是 256？
 
每个 tile 16×16 = 256 像素，平均下来 256 个三角形足够覆盖。溢出时 atomic_fetch_sub 回退计数。 
 
已知 Bug 与陷阱
 
✅ 已修复
• binning 溢出：atomic_fetch_add 后判断 slot，溢出时 atomic_fetch_sub 回退。
• 颜色插值未透视校正：colI 现在也用 iz0/iz1/iz2 加权。
• cullMode 符号反了：cross2D <= 0.0 剔除背面。 
⚠️ 未修复 / 未来处理
• 命令协议双份解析：extractTransform 和 render 里的两个 while 循环维护同一协议，改动需同步三处。
• 资源只增不减：vertexPool / indexPool 只 append，无 destroy。长时间运行会膨胀。
• 接收但未实现的状态：clearColor、blendEnabled、depthWriteEnabled、texture1、viewport 被编码但部分被忽略。
• PostFX 是死代码：AlloyAA / AlloySR / AlloySSAA 未接入 renderer。
• NVNFrontend 是占位：纯注释骨架。 
 
性能基线（iPhone，iOS 17+）
场景 三角形 FPS
单立方体 12 60
27 立方体阵列 324 60
3×27 立方体（当前 TestApp） 972 60
1000 立方体（10000+ 三角形） 12000 32
无 Tile Binning 时 324 三角形 324 10

 
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
 
Roadmap
 
v0.3.0 地基扩展（当前阶段）
[ ] 动态 buffer 更新：updateVertexBuffer(handle, data, offset)
[ ] 纹理进 GAL：createTexture / bindTexture
[ ] Sampler 资源：createSampler
[ ] 索引格式：支持 UInt16 
v0.4.0 命令协议重构
[ ] 单份协议定义（opcode 表）
[ ] 带长度字段的指令
[ ] 资源销毁：destroyBuffer / destroyTexture 
v0.5.0 深度状态
[ ] 深度比较函数（less/greater/equal）
[ ] 深度写入开关
[ ] 模板测试 
v0.6.0 NVN 前端
[ ] NVN 调用翻译到 GAL
[ ] 动态顶点数据流 
远期
[ ] D3D11 前端
[ ] 着色器虚拟机（DXBC/SPIR-V 解释器）
[ ] 渲染到纹理（Render Pass） 
 
联系方式 / 贡献
 
当前是个人研究项目，尚未对外开放贡献。任何 API 前端的实现都应满足：
1. 不改 Interface/ 的公开接口（除非确实需要通用能力）
2. 不改 Renderer/ 和 Shaders/（除非修复 bug）
3. 只加 Frontends/XXX/ 下的文件 
如果加前端时发现需要改核心，说明那是地基缺失的通用能力，应该先补地基。

---

把这份 README 放到仓库根目录，新对话开始时让 AI 读它，它就能接上全部上下文。

新对话里你还应该同时附上当前能编译过的 `Shaders.metal`（因为我没法保证你手里的版本没被手机端搞乱），告诉 AI："这是当前状态，先确认能编译，然后继续做动态 buffer 更新。"
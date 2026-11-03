
| 对象 | 中文名 | English name | 说明 |
|---|---|---|---|
| 文章标题 | **SALT-GS：面向水下高斯泼溅的场景感知光传输** | **SALT-GS: Scene-Aware Light Transport for Underwater Gaussian Splatting** | **SALT** 表示 *Scene-Aware Light Transport*，**GS** 表示 Gaussian Splatting。总标题用“场景感知光传输”概括统一机制：M1 提供水体–场景归属证据，M2 在该证据支持下建模空间异质水体的路径传输；它是独立的方法名，不是两个模块缩写的拼接。 |
| M1 | **多视角证据引导的水体–场景解耦** | **Multi-View Evidence-Guided Water–Scene Separation (MV-WSS)** | 在同一组高斯上累计跨视角责任证据，以连续份额分配表面与介质辐射。 |
| M2 | **空间异质水体路径积分传输** | **Path-Integrated Transport in Spatially Varying Water (SVW-PI)** | 用有界三维残差调制方向相关的水体光学参数，并沿相机到高斯及背景路径积分。 |

这些名称强调实现中能直接核实的操作。这里的“解耦/分离”指算法估计的场景/介质责任及其辐射份额，不意味着已证明真实物理材料身份可唯一辨识；“空间异质”描述场的表达能力，也不意味着完整反问题有唯一解。

## 机制总览

| 项目 | 按代码修正后的表述 |
|---|---|
| 故事 | 水下 3DGS 同时要拟合观测图像、保留真实物体，并解释沿光路累积的水体衰减和散射。远处低纹理物体容易和自由水外观混淆，均匀水体参数又不足以表达空间差异。本方法一方面把二维软证据经遮挡感知的 splat 贡献积累到三维高斯上，以连续份额区分表面与介质辐射；另一方面在方向相关的全局水体模型上增加低自由度、受支持域门控的空间残差，并用低分辨率前缀近似路径积分。 |
| 难点一 | 颜色接近水背景、纹理弱的远景并不必然是水体；单视角阈值式判断会在水雾残留和误伤真实结构之间摇摆。因此责任要结合深度/纹理/颜色/边缘/水背景拟合证据，并通过遮挡感知的二维贡献投影到高斯，再跨视角积累。 |
| 难点二 | 三维空间水体场自由度过大时可能与高斯外观相互解释误差。代码通过 16³、四通道场、平滑和减去空间均值、`tanh` 有界映射、由高斯归属导出的自由水支持，以及分块交替更新，限制该竞争。 |
| 难点三 | 清晰输出不能把水体高斯留作黑色遮挡，也不能为追求清晰而改变水下观测分支的几何占据。代码因此让主水下渲染继续使用总不透明度，并仅在评估期另外执行表面不透明度光栅化。 |
| 创新一 | **MV-WSS：多视角证据引导的水体–场景解耦。** 通过停止梯度的二维责任证据、基于真实 splat 足迹和遮挡的 GPU 责任归因、视角窗口累计与后验式状态缓冲，估计同一组高斯的表面份额 `q` 与介质份额 `1-q`。`q` 用于表面/介质辐射分流，光学厚度按份额拆分以供评估期表面渲染。它不创建两套独立高斯，也不等于一个端到端学习的材料分类器。 |
| 创新二 | **SVW-PI：空间异质水体路径积分传输。** 方向条件 WPP 和水体校准器提供全局基线，四通道三维网格只调制密度、相对光谱类型、散射比例和环境光；沿视锥的粗分辨率射线段建立直接透射与后向散射前缀，供高斯路径和背景查询。场的空间支持由 MV-WSS 产生的归属证据更新，因此两项机制有明确的数据依赖。 |
| 配套优化 | 默认训练对水下 RGB、前景加权 L1/DSSIM、伪深度视差相关性及水体/场景责任相关项联合优化；空间场只在固定时间窗的每四步之一单独更新。代码没有空间场专属责任加权 L1，也没有作用于空间场参数的显式三维 TV/幅度损失。 |

## M1：多视角证据引导的水体–场景解耦（MV-WSS）

### 1. 二维证据不是网络分类输出

默认损失路径先从训练伪深度、输入梯度/纹理、与水背景颜色的距离、边缘保护以及内禀外观反证构造停止梯度的纯水、过渡水、附着水幕、场景核心和不确定边界证据。空水与附着水幕分别进入后续责任组合，场景结构、深度边缘和水-only 拟合优势用于保护真实内容。它们是依据图像和当前渲染构造的软证据，不是一个单独训练的像素级材料分割网络，也不是四种独立三维实体。

跨视角三维积累时，代码组合水色/低结构上下文形成 medium evidence，组合深度边缘/场景核心形成 surface evidence，并从两类证据得到 uncertainty。另有几何反证：投影处伪深度、渲染期望深度一致性、深度边缘和水-only 假设差异；该支路以比例混入责任，而不是替代观测责任。

### 2. 遮挡感知的高斯责任和跨视角缓冲

责任不是简单在每个高斯中心采样。GPU kernel 使用投影 splat 的 footprint、opacity、前向透射率和归一化软责任，将像素责任累计到高斯贡献质量。实现使用独立责任归因 kernel；默认主颜色渲染的 `collect_contributions` 关闭。

载体从 step 4000 起、每 100 step 更新。窗口内对真实训练视图 ID 去重，要求方向变化代理有效且具有观测质量，累计介质/表面证据、观测质量、方向和 novelty。窗口接受的视角覆盖数被累计为 `lifetime_coverage`；该计数不是无限保存所有历史相机 ID 的严格唯一视角集合。默认窗口长度由训练视图数和 100-step 更新率推导并裁剪在 2000 到 6000 step；窗口认证至少需要 `min(3, configured_min_views)` 个窗口视角、足够观测质量和累计覆盖。默认需两个有效候选窗口才能认证；非决定性窗口不会强制清零候选连续计数。方向差用于近似角度/视差新颖性，不是独立的光流或几何视差监督。

高斯状态以 buffer 维护，初始 medium evidence 为 0.5，surface evidence 为 `0.5 + 1.5 × SfM reliability`，观测质量初始为 0。可靠视图数和重投影误差生成 SfM 可靠度。后验中位数与稳健离散度形成自适应介质阈值，邻域光学厚度支持以置信门控的 logit 修正责任。状态更新位于 `no_grad` 路径：代码实现的是带迟滞的在线证据估计，而不是对高斯属性端到端反传的概率分类器。

### 3. 连续分流及清晰表面渲染

记 `q_i` 为表面份额、`1-q_i` 为介质份额。实现把证据质量、角覆盖置信度、medium 后验和 SfM 保护组合为软介质份额，并取其补数为 `q_i`；已认证表面/介质再分别钳到 1/0。对总 opacity `α_i` 转成光学厚度 `τ_i = -log(1-α_i)`，然后分配：

```text
α_surface = 1 - exp(-q_i τ_i)
α_medium  = 1 - exp(-(1-q_i) τ_i)
```

在基元 opacity 层面，两者的透过率乘积等于总透过率；这不构成考虑不同投影 footprint、遮挡次序后的逐像素严格两层分解。默认水下主渲染保留原总 opacity（正常 carrier 路径中几何 gate 为 1），并把高斯颜色组合为 `q × (J × T_D + B) + (1-q) × B = q × J × T_D + B`。随后通过总 alpha 合成像素并加上剩余透射权重下的水背景。

主渲染还以总 alpha 权重渲染 raw intrinsic `J_raw` 与期望深度。**独立表面 opacity 的 clear raster 只在 `not training` 且 `torch.no_grad()` 下执行**：它用 `α_surface` 重新计算表面可见性，不叠加水背景。训练期 `intrinsic_color_render` 是 `J_raw`，清晰评估输出才来自该额外光栅化。训练期 carrier 归属诊断图返回常量占位，不应描述成训练期间已渲染出的真实像素级归属图。

carrier lifecycle 默认在 step 4000 之后可阻止已认证介质高斯继续生长；该路径返回空 prune mask，不执行基于该归属的删点。其他 3DGS 增密/常规裁剪策略仍按各自配置运行，所以不能把它表述为“完全不删点”或“执行归属清理”。

## M2：空间异质水体路径积分传输（SVW-PI）

### 1. 受约束的空间残差

默认空间场是单个 `1 × 4 × 16 × 16 × 16` 参数网格，零初始化；四通道对应介质密度、水体光谱类型、相对后向散射比例和环境光残差。空间查询先对网格做 replicate padding 的 3³ 均值平滑，再从每通道减去空间均值；结果有低频偏置，但并非严格带限，且均值移除发生在 `tanh` 前，不能据此保证最终光学参数残差严格零均值。

三维自由水支持从零开始。每 500 step 的支持更新使用 carrier 的软表面/介质份额、不确定度置信度和总 opacity 光学厚度，在高斯端点及相机到高斯线段的五个采样点累计介质/表面质量。介质证据经平均池化扩散，表面端点经最大池化形成排除项，二者形成目标支持并做 EMA；无观测区域保留原支持。默认执行链因此是 **MV-WSS 归属 → 自由水支持 → SVW-PI 空间传输场**，而不是独立于归属估计的统一支持域。

设平滑去均值网格在位置 `x` 处的采样为 `v(x)`、支持为 `F(x)`，实现先计算 `u = tanh(F(x) ⊙ v(x))`。用前三维构造有界密度、光谱类型和散射比例乘子：

```text
m_D      = exp(0.20 u_1) · exp(0.20 u_2 [ 0.65, 0, -0.65])
m_B      = exp(0.20 u_1) · exp(0.20 u_3) · exp(0.20 u_2 [-0.35, 0, 0.35])
A        = A_0 + 0.04 u_4 (A_ref - A_0)
β_D(x)  = β_D,0 ⊙ m_D
β_B(x)  = β_B,0 ⊙ m_B
```

上式中 `u` 的调制范围由 `tanh` 限定；RGB 斜率是固定设计，不能据此宣称最终 RGB 光学系数始终严格遵守某个物理波长排序。环境光是沿全局基线 `A_0` 与水色锚点 `A_ref` 的有界加性残差方向调整，`tanh` 允许正负值，不是独立 RGB 空间颜色场，也不保证是二者的凸组合。高置信自由水观测的输入颜色通过无梯度 EMA 更新 `A_ref`。

### 2. 沿路径构造传输前缀

WPP/方向编码器和水体形成校准器查询方向相关基线。空间网格的查询只依赖坐标；空间场参数与 WPP/校准器参数是并行维护、通过光学参数融合和阶段更新协作，并非用 WPP 的中间输出串联预测空间网格。

启用空间场时，默认 prefix renderer 在分辨率约为原图 `1/8` 的视锥网格上，对 AABB 内射线做 8 段中点采样。它分别累积直接透射 `T_D` 和后向散射的前向透射/辐射 `B`，形成可以在高斯投影坐标和路径距离处查询的前缀残差；路径离开 AABB 后用全局基线延拓。尺度由归一化场景中的欧氏相机—高斯距离和 `depth_scale=10` 换算，基线距离被裁剪到 `far_distance=4`，不是未经归一化的米制水下路径长度。背景也用 `integrate_background_from_base` 将空间 prefix 的末端残差上采样后加入方向相关环境光；这是默认实现，不能用代码中的备用 `render_background` 描述默认流程。

前缀采用空间异质结果相对于均匀基线的残差形式；零初始化空间场时应退化回方向相关基线。对于单个表面高斯，路径传输给出直达项 `J × T_D` 与累积后向散射 `B`，再与 MV-WSS 的介质/表面份额组合。

### 3. 分块更新

默认 30,000 步训练中，step 0–14,999 固定空间体素场，优化高斯及方向水体基线；15,000–22,999 每四步安排一个空间场更新步，此步只开放空间场参数并冻结高斯、WPP、校准器、相机优化和可选 bilateral grid，其余三步冻结空间场并更新这些外观/场景参数；23,000 步起固定空间场继续稳定化其余参数。场更新步绕过常规高斯 density-control 后处理；其他训练状态（例如 no-grad carrier/support 证据缓冲）不因此自动冻结。

这种设计是交替优化和优化器参数隔离，不代表误差来源已经完全解耦，也不构成唯一解保证。零初始化且无可靠自由水支持时，局部场可保持接近全局水体模型。

## 默认训练目标与监督边界

| 训练项 | 默认实现及实际边界 |
|---|---|
| 水下图像重建 | 前景感知权重 L1 与 DSSIM，按配置系数组合；方向水体背景在前 15,000 步接受背景监督。 |
| 伪深度 | 归一化伪深度与 `1/(10D+1)` 形式的渲染轴向深度代理做 Pearson 相关损失，默认权重 0.1。它是粗粒度深度约束，不是 M1 后验的精确深度标签。 |
| 可信纯水/空水 | 从 step 500 起启用相关区域损失；纯水区域约束总累积 alpha（权重 0.08）、raw intrinsic 零贡献（0.02）、水下 RGB 向水背景归属（0.02）以及辅助 layered-clean 空水黑底（0.045）。 |
| 过渡水 | alpha 低于目标 0.15 的惩罚默认权重 **0**，因此不能说默认强制过渡水区低不透明度。区域掩码还会做严格阈值和每图比例限制。 |
| 残余水色和场景保护 | raw intrinsic 蓝/青残留权重 0.01；附着 veil 的辅助清理色度权重 0.030、细节保护 0.012；场景/veil/边界水下观测一致性权重分别为 0.035/0.018/0.008。辅助 clean 是由 raw intrinsic 与 detach 的水背景/区域证据代数组合得到，并非独立 surface-opacity 光栅。 |
| calibrator | 方向水体校准残差 L2，权重 0.0006；这不是空间体素场的幅度正则。 |
| 名义上的 veil 平滑/层互斥项 | loss 字典中有 0.004/0.006 权重，但对应责任与 veil 区域由停止梯度证据生成，对训练参数没有梯度。因此不能当作有效的可学习空间平滑或互斥约束来宣传。 |
| 空间水体场专属目标 | 默认 `get_loss_dict` 没有场专属责任加权 L1，也没有显式对三维场施加 TV 或幅度损失。空间场通过同一默认渲染损失获得梯度，并依靠低容量、平滑/去均值、支持门控、有界调制和分块冻结限制。 |

`water_null_start_step=500` 是相关区域损失的开始步；carrier 证据 warm-up 为 4,000 步；空间场从 15,000 步开始训练。这三个时间点对应不同机制，不能互换。默认配置还使用 `sh_degree=0`、antialiased rasterizer 和 30,000 总步数；通用可配置的 SH 逻辑不改变默认实验颜色为 sigmoid DC RGB 的事实。

## 两项机制的关系与可主张边界

MV-WSS 与 SVW-PI 有显式联动：二维证据经 CUDA footprint 责任归因更新高斯份额，份额再更新 SVW-PI 的自由水支持；支持对局部水体场开放的位置进行门控；两者共同进入默认水下颜色形成。它们不是独立的、互不相干的两个模块，但其更新并非一个端到端联合可微材料分解过程。

在论文中较稳妥的机制主张是：**多视角观测质量加权的三维高斯表面/介质软归属，以及归属支持域约束下的有界空间水体路径积分。** 应避免写成：已物理证明的唯一/可辨识材料身份；两套独立表面与水体高斯；训练和推理一致的可微清晰表面分支；与归属无关的统一自由水支持；空间场专属责任加权监督或有效 TV/幅度正则；不受归一化限制的真实米制传输；完全禁止高斯删除。

代码目录里存在 next-generation 和其他候选实现，但 `_uses_nextgen_water_model()` 在该提交中恒返回 `False`，因此上面的结论仅采用默认 `get_outputs()` / `get_loss_dict()` 路径。代码文件或类名中出现 “physics-identifiable” 不应被单独当作论文的数学可辨识性结论。

## 原机理表的关键修正

| 原表述/隐含主张 | 核对结果及应替换为 |
|---|---|
| 存在训练与评估一致的独立表面渲染分支，并对其直接优化多项表面损失。 | **不成立。** 清晰表面 opacity 光栅化只在 eval/no-grad 执行；训练以总 opacity 渲染 raw intrinsic，另外使用辅助 clean 代数组合和水下 RGB 监督。 |
| 多项区域表面损失都有效回传梯度。 | **不成立。** veil 平滑和层互斥的输入证据停止梯度；这些条目可能有标量数值，但不优化区域生成器或场参数。 |
| M2 使用独立统一支持，不使用 M1 归属后验。 | **与默认实现相反。** 每 500 步的场支持更新读取 carrier 的介质/表面份额及其不确定度置信度，并沿相机—高斯线段投票。 |
| M2 有专属责任加权 L1 和显式 3D 平滑/场幅度损失。 | **不成立。** 空间场使用相同总训练目标；代码中没有这些场专属损失。平滑与减均值是参数化前向操作，`tanh` 和支持门控也是机制限制。 |
| M1 只在空间场更新步之外执行。 | **不成立。** 默认 `get_loss_dict` 在全部训练步执行；归属缓冲由各自的 warm-up/100-step 更新条件控制。field step 冻结可训练参数并跳过 density control，不意味着 carrier 状态停止更新。 |
| 多视角覆盖表示全生命周期严格不同相机 ID；认证必须每个相邻自然窗口一致。 | **过强。** 窗口内 ID 去重，累计 coverage 是被接受窗口视角的计数；窗口长度是按 step 构造的。认证保留非决定性窗口之前的候选状态，因此不等价于严格相邻自然窗口连续一致。 |
| M1 一旦判断介质就会执行高斯清理/删点。 | **不成立。** carrier 生命周期默认只对已认证介质提供 no-grow mask，不返回 carrier prune mask。 |
| 低频网格等于严格 band-limited、最终光学残差空间均值为零，并保证物理 RGB 排序。 | **不成立。** 3³ 均值平滑只能偏向低频；去均值发生在非线性映射前；固定斜率不保证组合后的 RGB 系数具有严格物理排序。 |

## 代码依据（固定提交）

以下链接对应提交 `f18be11763c282a8597808e96f7b189069e6967b`，可用于论文方法复核：

- 默认实现与配置：[`seafree_config.py`](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_config.py#L18-L42)、[`seafree_model.py` 默认分支](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L608-L621)。
- 二维证据、区域合成、三维责任更新与几何反证：[`seafree_model.py` water evidence](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L1448-L1668)、[`seafree_model.py` null responsibilities](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L1924-L2240)、[`seafree_model.py` geometric counterfactual](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L2410-L2501)、[`seafree_model.py` lifecycle/support update](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L2501-L2635)。
- GPU splat-footprint responsibility：[`rasterize_to_pixels_fwd.cu`](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/third_party/gsplat/gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu#L371-L475)。
- Carrier 初始化、证据窗口、认证、软份额、opacity 分配和生命周期：[`aquanull3d.py`](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/aquanull3d.py#L1-L110)、[窗口及认证](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/aquanull3d.py#L363-L466)、[归属和分流](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/aquanull3d.py#L498-L592)、[窗口内视角累计](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/aquanull3d.py#L635-L772)。
- 水下渲染分流、总 alpha、clear eval-only raster 和 prefix 查询：[`seafree_model.py` outputs](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L3351-L3802)。
- 空间场结构、支持更新、参数调制、路径前缀和背景：[`path_integrated_water.py`](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/path_integrated_water.py#L1-L58)、[平滑/减均值](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/path_integrated_water.py#L122-L130)、[carrier 支持网格](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/path_integrated_water.py#L164-L260)、[有界光学调制](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/path_integrated_water.py#L294-L369)、[前缀与路径查询](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/path_integrated_water.py#L472-L797)、[默认背景积分](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/path_integrated_water.py#L933-L984)。
- loss 数值、有效监督和 detach 证据边界：[`seafree_model.py` loss/field defaults](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L313-L491)、[`seafree_model.py` default losses](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L4110-L4550)。
- 分块优化与 field-step 后处理：[`seafree_model.py` optimizer schedule](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L1059-L1091)、[`seafree_model.py` density control](https://github.com/guoyize409-sketch/seafree2/blob/f18be11763c282a8597808e96f7b189069e6967b/seafree_gs/seafree_model.py#L964-L1038)。

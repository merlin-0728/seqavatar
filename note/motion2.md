# SeqAvatar Point Notes

## 2026-09-05 temporal hidden 与 Part hidden

`temporal_hidden` 可以直接理解为时间专家 MLP 生成的形变中间特征。它还不是
最终的 `d_xyz/d_rotation/d_scaling`，而是时间 branch 读取原始 Gaussian
features 后提取的高维表示；时间输出头可以据此预测最终形变。

Part expert 也会产生自己的中间量：它接收 `temporal_hidden`，再执行
`part_hidden = body(temporal_hidden)`，然后用 Part expert 自己的 xyz、rotation、
scaling heads 预测部件形变。因此两者是前后串联而非平行关系：

```text
原始 features -> 时间专家 -> temporal_hidden
                              -> Part expert body -> part_hidden
                                                  -> Part 形变输出
```

`temporal_hidden` 主要编码当前时间区间的形变状态；`part_hidden` 是在该时间状态
基础上进一步针对某个人体部件重新编码后的特征。当前实现中 Part expert 不从头
重新处理原始 features，而是以 temporal hidden 为条件。

补充：两个 hidden 不是直接相加、拼接后再统一预测形变。实际是同一个
`temporal_hidden` 分成两条输出路径：

```text
temporal_hidden
    ├─ 共享 temporal xyz/rotation/scaling heads -> temporal_output
    └─ Part expert body -> part_hidden
                           -> Part heads -> part_output
```

代码先分别对 `temporal_output` 和 `part_output` 做时间 branch 的 soft blending，
再使用 `part_gate` 融合两个“形变输出”而不是两个 hidden：

```text
final_output = (1 - part_gate) * temporal_output
              + part_gate * part_output
```

因此 `temporal_hidden` 同时影响两条路径，`part_hidden` 只存在于 Part 路径内部。
在 Part warmup 期间两者逐渐混合；warmup 前主要使用 temporal output，warmup 后
主要使用 Part output。unknown Gaussian 的 Part output 使用 global Part expert，
但它仍然以 temporal hidden 为输入。

## 2026-09-05 GPU memory holder 诊断

命令行出现单独的 `Killed`，表示该次 holder 进程被 `SIGKILL`（signal 9）
终止，不是 Python 收到 SIGINT/SIGTERM 后的正常退出。常见原因是系统 OOM
killer、管理员或外部脚本执行 `kill -9`；仅凭这行不能区分具体原因。

后续检查发现 GPU1 当前仍有一个相同命令的 holder：PID `156639`，启动于
`2026-09-05 12:33:28`，状态为 sleeping，实际占用约 `19386 MiB`。因此它是
仍存活的后续实例，不能把之前的 `Killed` 误认为当前实例也已退出。

GPU holder 脚本已重命名为 `note/gpu/train.py`，并同步更新
`note/gpu/monitor_gpu_then_hold.sh` 的脚本引用。已运行进程不受重命名影响。

## 2026-09-05 独立 part_time_moe 正式实验

### 代码实现分析

补充说明：`temporal branch 预测`和`temporal hidden`不是两次独立的预测。
branch 是被时间 `pose_id` 选中或加权的完整 deformation MLP；该 MLP 的中间
输出就是 `temporal_hidden`，随后它再经过 xyz/rotation/scaling 输出头得到最终
形变。`temporal_hidden` 是每个 Gaussian 的高维中间特征，不是帧编号，也不是
额外拼接的时间编码。

进一步澄清：`temporal hidden` 是时间 branch 的中间特征；在
`TemporalConditionedFullPartExpert` 内部还会生成另一个中间特征
`h = body(temporal_hidden)`。因此 Part expert 不是只做一个固定权重操作，而是
先用自己的 `W->W + ReLU` 对 temporal hidden 再编码，再通过自己的 xyz、
rotation、scaling 输出头产生 Part 形变。前者描述当前时间段的运动状态，后者
描述在这个时间状态下某个身体部件的专门形变。

### 时间分组和 Part 分组的层级关系

这两个标签确实对应两个维度：

```text
时间维度：pose_id -> temporal branch
空间维度：Gaussian 的 part_label -> Part expert
```

但当前 `part_time_moe` 不是 28 个独立专家的扁平并行结构。实际结构是：

```text
原始 Gaussian features
        |
        +--> 时间 branch 0/1/2/3（先选一个，边界处最多混合两个）
                    |
                    +--> temporal_hidden
                              |
                              +--> 共享的 Part expert 0/1/2/3/4/5/6
                                      （按 Gaussian 的 part_label 选一个）
```

所以一个 Gaussian 的路由可以写成一个二元组合：
`(temporal_branch_id, part_id)`，例如 `(branch_2, left_leg)`，但这个组合
没有对应一套独立参数。四个时间 branch 共享同一组七个
`TemporalConditionedFullPartExpert`；代码中的 Part expert 列表只创建一次，
见 `nets/mlp_delta_non_rigid.py:1700-1711`，然后对每个时间 branch 的
`temporal_hidden` 重复使用，见 `nets/mlp_delta_non_rigid.py:2116-2125`。

当前实际参数组件是：

```text
4 个 temporal branch 的 MLP（branch-specific hidden）
+ 共享的 temporal 输出头
+ 7 个跨时间复用的 temporal-conditioned Part expert
```

不是：

```text
4 个时间区间 × 7 个部件 = 28 个独立完整 MLP
```

需要特别注意组合 forward 的代码细节：`_mapo_hidden_for_leaf()` 只调用
非 root temporal branch 的 `.mlp(features)`；随后
`forward_temporal_conditioned_part()` 使用 deformer 共享的
`self.gaussian_warp/rotation/scaling` 生成 temporal output，并把同一个
`temporal_hidden` 送入 Part expert。因此，虽然初始化 `TemporalDeformationExpert`
时也 deep-copy 了三个输出头，但在 `part_time_moe` 的组合路径中这些 branch 自己
的复制输出头没有被调用；它们会在 standalone `l2 soft` 的 `forward_mapo()` 中
使用。这个区别见 `nets/mlp_delta_non_rigid.py:2011-2019,2116-2125`。

而且 `TemporalConditionedFullPartExpert` 名称中的 `FullPart` 指它能完整输出
xyz/rotation/scaling 三种形变，并不表示它复制了 standalone Part-MoE 的完整
原始 deformation MLP。它实际包含一个 `W->W` body 和三个输出头，定义在
`nets/mlp_delta_non_rigid.py:1160-1182`。

standalone `part_moe_leg` 使用的是另一类 `PartNonrigidExpert`：它对原始
`self.mlp` 和三个输出头都做 deep copy，七个 Part expert 更接近七套完整网络，
见 `nets/mlp_delta_non_rigid.py:1117-1127`。因此 `part_time_moe` 是功能上的
Part 路由 + 时间路由结合，但不是把 standalone `part_moe_leg` 的七套完整 MLP
再复制到每个时间区间里。

真正的 28 专家结构需要显式创建类似
`experts[temporal_id][part_id]` 的二维 ModuleList，并让每个二元路由拥有自己
的一整套参数；当前代码没有这样做。

入口在 `scripts/exps_dnarendering.sh:275-288`：该模式同时打开
`use_mapo_all_dynamic`、`mapo_max_partition_level=2`、`mapo_soft_routing`、
`use_part_moe` 和 `use_temporal_conditioned_part_moe`。没有打开 robust label、
confidence route、partial sharing、dynamic score 或额外 loss。

时间分支部分：

- `NonrigidDeformer` 先创建一个原始 `self.mlp + 3 个输出头`，并额外创建
  3 个 `TemporalDeformationExpert`；它们是完整 MLP 的深拷贝，代码在
  `nets/mlp_delta_non_rigid.py:1694-1727`。
- 开始训练时 `GaussianModel.configure_mapo_training()` 把 active level 设为 0，
  因此 0-4999 step 只使用 root branch。
- 在每个 iteration 开头，`scene/gaussian_model.py:561-585` 根据迭代数检查分层：
  5000 step 激活 level 1，得到 2 个时间区间；10000 step 激活 level 2，得到
  4 个时间区间。新增 branch 从上一级 branch 的当前参数快照继承，并清除其
  optimizer state 后开始独立更新，具体复制逻辑在
  `nets/mlp_delta_non_rigid.py:1841-1930`。
- `mapo_num_frames=100`，所以 level 1 对应 pose_id 0-49/50-99，level 2 对应
  0-24/25-49/50-74/75-99。`_mapo_temporal_weights()` 在每个边界前后 4 帧内
  用 smoothstep 混合相邻两个 branch；代码在
  `nets/mlp_delta_non_rigid.py:1989-2009`。

Part 部分：

- 10000 step 的 `PartMoeController.after_iteration()` 先保存当前 Gaussian，
  按 Gaussian canonical 坐标到 SMPL-X canonical 顶点的最近邻生成标签，代码在
  `ablations/part_moe_controller.py:65-77,92-151`。
- 这次 `part_label_robust=false`，所以虽然命令传入 `part_label_knn=8`，实际
  `requested_k` 会被强制为 1；标签是单个最近 SMPL 顶点的 `part_moe_leg` 标签。
  标签 0 表示 unknown/global，1-6 表示 6 个身体部件。
- 10000 step 后由 `init_temporal_conditioned_part_moe()` 只打开运行时 active
  标志。7 个 `TemporalConditionedFullPartExpert` 在模型初始化时已经创建，代码在
  `nets/mlp_delta_non_rigid.py:1160-1182,1700-1711`；每个包含一个 `W->W` body
  和 xyz/rotation/scaling 三个输出头，并从原始输出头初始化。

真正的组合顺序在 `forward_temporal_conditioned_part()`：

```text
features
  -> root/temporal branch MLP
  -> temporal hidden
  -> 同一组 7 个 Part predictor 按 part_label 选择
  -> temporal branch 的 soft blending
  -> temporal 输出与 Part 输出的渐进融合
  -> d_xyz, d_rotation, d_scaling
```

代码先对每个有非零时间权重的 branch 计算 temporal hidden；靠近边界时只计算
相邻两个 branch，并分别得到 temporal output 和 Part output，再按 temporal 权重
相加，见 `nets/mlp_delta_non_rigid.py:2105-2137`。因此这里不是 4 组互相独立的
Part expert，而是 7 个 Part expert 跨 4 个 temporal branch 复用。

Part 输出内部的组合是：对已知 part，`global_keep=0.1` 的 global expert 加上
`0.9` 的对应 part expert；unknown label 0 使用 global expert。外层 Part 路由
从 10000 step 后开始 warmup 1000 steps：`compute_part_moe_alpha()` 在
`train.py:712-725` 产生 0 到 0.9 的权重，`fusion_mode=replace` 在
`nets/mlp_delta_non_rigid.py:2138-2165` 实际计算

```text
output = (1 - part_gate) * temporal_output + part_gate * part_output
```

所以 10001 step 左右仍主要是 temporal 输出，11000 step 后基本完全采用 Part
输出；这里的 `global_keep=0.1` 是 Part 输出内部保留 global expert 的比例，
不是保留 temporal 输出的比例。

训练和梯度：

- Gaussian 的 xyz、颜色、opacity 等仍由原训练流程优化；非刚性网络由
  `mlp_optimizer` 优化，`scene/gaussian_model.py:774-806` 会把整个 deformer
  参数加入 optimizer。
- 这次 temporal-conditioned Part expert 从初始化时就在 deformer 参数中，
  10000 step 前没有进入 forward，因而没有有效梯度；10000 step 激活后开始从
  图像重建损失学习。
- 没有增加 AIAP、边界一致性或 Part 监督 loss；所有改进只能通过时间分支和 Part
  路由改变 `d_xyz/d_rotation/d_scaling`，最终仍由原始 RGB、mask、SSIM、LPIPS
  损失训练。

和单模块的本质区别：

- `l2 soft` 只有时间分支：`forward_mapo()` 直接选择或 soft blend temporal
  branch，见 `nets/mlp_delta_non_rigid.py:2298-2352`。
- standalone `part_moe_leg` 使用 `PartNonrigidExpert` 深拷贝完整 root MLP 和
  三个输出头，再按 part label 路由，见 `nets/mlp_delta_non_rigid.py:1117-1127`
  和 `2727-2761`。
- `part_time_moe` 则是 temporal branch 先产生 hidden，随后由跨时间复用的
  7 个 `TemporalConditionedFullPartExpert` 对这个 hidden 做 Part-specific
  预测。因此它是“时间条件化的 Part-MoE”，不是两个单模块输出结果的后处理相加。

目标：只直接组合原始 `part_moe_leg` 与 `l2 soft`，不加入 confidence、
robust label、unknown route、partial sharing 或其他优化，验证简单组合能否超过单模块。

统一配置：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
GPU=1（六序列串行）
IMAGE_DATA_DEVICE=cpu
```

模块配置：

```text
l2 soft: 5000 step 激活 2 个完整时间 branch，10000 step 激活 4 个，soft blend width=4
part_moe_leg: 10000 step 激活，7 个完整 Part expert，global_keep=0.1，warmup=1000
融合：Part 输出直接替换对应 Gaussian 的 temporal 输出
标签：原始 1-NN（part_label_robust=false）
```

正式输出版本：`20260905_part_time_moe_plain_cpu`。

0044 已核实：
- 5000 step 日志出现 `activated level=1 branches=2 intervals=50`；
- 10000 step 日志出现 `activated level=2 branches=4 intervals=25`；
- 同步出现 `Build labels at iteration 10000` 和
  `full part predictors activated`；
- 两个模块确实同时生效，组合后训练速度约 7.7 it/s，CPU 图像驻留下未 OOM；
- 六序列已完成，六个结果文件均已生成；无 OOM 或异常退出。

`part_time_moe` 六序列 novel-view 25000 指标：

```text
0044_11: PSNR 32.99698181152344, SSIM 0.97817276318868, LPIPS*1000 21.089035171704985
0051_09: PSNR 28.802151950200397, SSIM 0.9720080539584159, LPIPS*1000 30.48575938834498
0206_04: PSNR 31.531845585505167, SSIM 0.9705860803524653, LPIPS*1000 32.62539912636081
0813_05: PSNR 36.13070635795593, SSIM 0.9870904376109441, LPIPS*1000 18.118568595188357
0007_04: PSNR 29.60045822461446, SSIM 0.9589263752102851, LPIPS*1000 43.61092095884184
0019_10: PSNR 35.39065691630046, SSIM 0.9812868545452753, LPIPS*1000 20.753366321635742

mean: PSNR 32.40880014101665, SSIM 0.9746784274776776, LPIPS*1000 27.780508260346117
```

均值对比（LPIPS 越低越好）：

```text
original:     PSNR 32.31045727729797, SSIM 0.9742256858282619, LPIPS*1000 28.405228959551703
part_moe_leg: PSNR 32.41623780992296, SSIM 0.9748385775420401, LPIPS*1000 27.595275557703445
l2_soft:      PSNR 32.44707352585262, SSIM 0.974839673191309,  LPIPS*1000 27.793325547180658
part_time_moe:PSNR 32.40880014101665, SSIM 0.9746784274776776, LPIPS*1000 27.780508260346117
```

判断：`part_time_moe` 比 original 提升，但 PSNR/SSIM 均值低于两个单模块；
LPIPS 高于 `part_moe_leg`，略低于 `l2_soft`。因此简单的 Part 输出替换与
时间分支 soft routing 存在冲突，不能归因成两个模块增益相加。

## 2026-09-03 Part-MoE + l2 soft v5 运行状态

- 已读取并接续检查正式实验 `20260903_fullconf_v5`。
- 当前仅使用空闲的 GPU3 串行训练，未占用其他用户使用的 GPU0/1/2。
- `0044_11` 已完成：PSNR `33.005066108703616`，SSIM `0.978235730032126`，LPIPS*1000 `21.268079934331278`。
- `0044_11` 的 PSNR 已超过单模块 `part_moe_leg` 的 `32.98199820518494`，但尚未在三项指标上整体超过 `l2 soft`；需等待六序列完成后按均值判断。
- `0051_09` 正在训练，正式进程 PID 为 `2283827`；继续监督，不使用失败的 v3/v4 结果。

## 2026-09-03 Part-MoE + l2 soft 修复进行中

目标：修复 `part_moe_leg + l2 soft` 组合实验，并在统一的
`ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`、`seed=0` 配置下，
与单模块 `part_moe_leg` 和 `mapo_all_dynamic_l2_soft` 严格比较。

已确认并修复：
- 训练主路由原来只检查 `part_moe_active`，导致
  `temporal_conditioned_part_active=True` 时组合分支仍可能被跳过；
  现在两种 active 状态都可进入组合路由。
- `render.py` 加载 checkpoint 后原来没有恢复 temporal-conditioned Part
  的运行时 active 状态；现在最终评估会显式激活组合预测器。
- 修复后 0044 在 10000 step 真正启用组合图时 OOM。原因是一次前向同时构建
  4 个 temporal branch x 7 个 Part expert 的全量 Gaussian 计算图。
- 保持数学路由不变，改为只计算 temporal 权重非零的 branch：普通帧 1 个，
  soft boundary 最多相邻 2 个。
- confidence fusion 下只对 `known && confidence > threshold` 的 Gaussian
  计算 Part expert，其余点严格保留 temporal 输出。0044 在 10000 step 的标签中，
  该有效子集约占 `0.7764982957773149`。

验证：
- `compileall`、`bash -n`、`git diff --check` 通过。
- CPU 前向 smoke 通过，四分支、soft routing、Part expert 和 confidence gate
  都实际参与输出。
- 当前正式目录：
  `mapo_temporal_conditioned_part_full_confidence/20260903_fullconf_v5`。
- v5 早期训练 GPU3 显存约 19.3 GB，低于修复前约 23.8 GB；仍需确认
  10000 step 组合激活后能否稳定通过，并等待最终六序列指标。

## Scope
只保留 `point` 相关实验记录。

已放弃继续推进：
- `part_moe / part_score_route / tri / time / state` 相关线
- 以后新增内容只补 `point` 主线

## 1. 主线定义

`point_anchor_tb`:

```text
point_anchor + temporal persistence + boundary prior + delayed replacement
```

`point_update`:

```text
fixed-budget point redistribution
```

核心是：
- 先在高误差区域补点
- 再把总点数拉回 original baseline 的最终点数
- 训练全过程允许点位变化，但最终预算必须公平

## 2. 已验证结论

### point_update

结论：
- 已实现“最终点数公平”
- 但提升很弱，主要只有极小 PSNR 正向信号
- SSIM / LPIPS 没有稳定同步改善

四序列均值：

```text
point_update:
    PSNR 32.27229080994924
    SSIM 0.9764028407633305
    LPIPS*1000 26.281662310551233
```

相对 original:

```text
ΔPSNR  +0.014937970373195666
ΔSSIM  -0.0000151835382199
ΔLPIPS +0.04483989368742617
```

判断：
- 能说明 fixed-budget point redistribution 跑通了
- 不能说明它稳定优于 original

### point_update_soft

结论：
- 没超过 `point_update`
- 只是局部序列有小收益

四序列均值：

```text
PSNR 32.23563768863678
SSIM 0.9763126400609811
LPIPS*1000 26.498774448797725
```

### point_update_perf

结论：
- 没超过 `point_update`
- 延长 spawn / 放宽中期点数 / 延迟删除，不是稳定收益方向

四序列均值：

```text
PSNR 32.21909255186716
SSIM 0.9762630087633928
LPIPS*1000 26.536896772449836
```

### point_update_edge

结论：
- edge prior 只加在加点侧
- 效果和 `point_update` 基本持平，略弱
- 说明边缘信息能微调 anchor 选择，但不足以改写预算迁移主方向

### point_update_nonrigid

结论：
- `d_xyz_norm` 这类 nonrigid prior 也只带来弱信号
- 有相关性，但不稳定，不足以单独作为主改进

## 3. 当前最可靠的判断

当前这条线真正有价值的不是“把点加更多”，而是：
- 更准地选哪里加
- 更准地选哪里删
- 更准地保护哪里不删

也就是把重点从“补点强度”转到“预算迁移质量”。

## 4. 下一步只看 point 线时的优先级

1. `point_update_keep`
   - 继续保持 final points = original
   - 强化保护策略
   - 不删 boundary / high-error / high-motion / long-visible / newly spawned 点
   - 做空间均衡删除，避免局部挖空

2. 优先改删除策略，而不是继续加 spawn 强度

3. 如果再加新先验，只能放在 add side 或 protect side
   - 不要把新信号直接塞进全局删点规则里

## 5. 当前可复用的表达

```text
基于 original baseline 的最终点数公平约束，
在固定 Gaussian 预算下进行高误差区域的点重分配。
当前结果说明机制可行，但收益仍弱，
后续应重点优化删点与保护策略，而不是继续扩大补点强度。
```

## 6. 2026-08-26 part_point 混合分支结果

说明：
- `part_point = point_tem_mul + part_moe_leg_unknown_route_strong`
- 这是 point 与 part route 的交叉实验，不作为当前纯 point 主线继续展开
- 记录指标是为了后续查数方便

结果来源：

```text
output/DNA-Rendering/*/part_point/20260825_213900/metrics/results_novelview_25000.json
```

DNA-Rendering 六序列 novel-view 25000：

```text
0007_04: PSNR 29.648974974950153, SSIM 0.959372294942538,  LPIPS*1000 42.80584648561974
0019_10: PSNR 35.373106416066484, SSIM 0.981357941031456,  LPIPS*1000 20.199782859223586
0044_11: PSNR 32.996869738896685, SSIM 0.9783144583304723, LPIPS*1000 21.01858698297292
0051_09: PSNR 28.84291869799296,  SSIM 0.9720176935195922, LPIPS*1000 29.840894090011716
0206_04: PSNR 31.49399100939433,  SSIM 0.9704646954933802, LPIPS*1000 32.85064056205253
0813_05: PSNR 36.21438485781351,  SSIM 0.9873889123400053, LPIPS*1000 17.690769241501886

mean: PSNR 32.42837428251902, SSIM 0.9748193326095741, LPIPS*1000 27.40108670356373
```

简短判断：
- 相比单纯调 point 强度，这条混合分支没有形成明显 PSNR / SSIM 优势
- LPIPS 有一定正向信号，但因为混入 part route，不适合作为纯 point 主贡献归因

## 7. part_point 的分层起点

启动配置里两段起点都要分开看：

```text
point_tb_start_iter = 800
part_moe_start_iter = 10000
```

所以：
- `point_anchor_tb` 这类 point 分层/增删点，从 800 step 就开始
- `part_moe_leg` 这类 part expert 分层，从 10000 step 才开始
- 这次拿来对比的 `part_moe_leg` 基线也是同样的 `10000` 起步，不是别的版本

## 8. part_point 相对 part_moe_leg

DNA 六序列均值对比：

```text
part_moe_leg:
    PSNR 32.424867553181123
    SSIM 0.9746271777484150
    LPIPS*1000 28.091741072334

part_point:
    PSNR 32.42837428251902
    SSIM 0.9748193326095741
    LPIPS*1000 27.40108670356373
```

差值：

```text
ΔPSNR  +0.003507
ΔSSIM  +0.000192
ΔLPIPS -0.690654
```

简短判断：
- 比 `part_moe_leg` 略好
- 但幅度很小，属于弱正向结果
- 因为它混了 `point_tem_mul` 和 `part_moe_leg_unknown_route_strong`，不能把提升完全归到 point 侧

## 9. 归因结论

补充确认：
- 纯 `point_tem_mul` 实验已经跑过
- 结果来源：

```text
output/DNA-Rendering/*/point_tem_mul/*/metrics/results_novelview_25000.json
```

DNA-Rendering 六序列 novel-view 25000：

```text
0007_04: PSNR 29.540740807851154, SSIM 0.9586394329865774, LPIPS*1000 43.813573118920125
0019_10: PSNR 35.27489086786906,  SSIM 0.9810092474023501, LPIPS*1000 20.551023740942277
0044_11: PSNR 32.95373956362406,  SSIM 0.9780667945742607, LPIPS*1000 21.137755969539285
0051_09: PSNR 28.639523553848267, SSIM 0.9714013422528902, LPIPS*1000 31.283746302748717
0206_04: PSNR 31.38172345161438,  SSIM 0.9702898239096006, LPIPS*1000 33.043038845062256
0813_05: PSNR 36.086021709442136, SSIM 0.9869752869009971, LPIPS*1000 18.333750722619396

mean: PSNR 32.31277332570818, SSIM 0.974396988004446, LPIPS*1000 28.027148116638674
```

## 10. part_point 的 densify 公平性

最新确认：
- 这次 `part_point` 不是 `densify_until_iter=1500`
- `scripts/exps_dnarendering.sh` 里默认先写 `1500`
- 但只要开了 `point_anchor_tb`，且没有显式传 `DENSIFY_UNTIL_ITER`，脚本会自动抬到 `1801`
- 这次 `part_point` 的配置里，`POINT_TB_END_ITER=1800`
- `train.py` 的全局增删点条件是 `if iteration < opt.densify_until_iter`

所以这次实际是把 point 分支的增删点窗口拉到了 1800 步附近，不是和 baseline / `part_moe` 完全同预算。

判断：
- 这次对比不完全公平
- `part_point` 的小幅提升，至少有一部分可能来自更长的 densify 窗口
- 若要严格比较，必须把所有方法统一到同一个 `densify_until_iter`

补充：
- 你说的 0044 `original` 在 `densify_until_iter=1800` 会 OOM，这意味着 1800 这个窗口对原版 baseline 不是可运行设置
- 这样一来，这次 `part_point` 和 `original` 的差异不只是“预算更长”，还包含“原版根本跑不稳”
- 所以不能把 `part_point` 在 1800 跑通直接当成纯性能提升，最多只能说它在这套配置下可运行并有弱收益

`part_point - point_tem_mul` 六序列均值差：

```text
ΔPSNR  +0.115601
ΔSSIM  +0.000422
ΔLPIPS -0.626061
```

当前这次 `part_point` 相对 `part_moe_leg` 的提升，**不能**说明都是 `point_tem_mul` 带来的。

原因：
- `part_point` 不是纯 point 实验
- 里面同时包含 `part_moe_leg_unknown_route_strong`

## 11. part_moe_leg 的 Part 误分类机制

严格结论：当前普通 `part_moe_leg` 没有真正学习式地降低 Part 分类错误率的机制，只有降低错误标签影响的防错机制。

当前标签生成流程：

```text
Gaussian canonical xyz
    -> 找最近的 SMPL-X canonical 顶点
    -> 继承该顶点的 Part label
    -> 距离超过 part_max_smpl_dist 则设为 unknown
```

相关实现位于 `ablations/part_moe_controller.py`。当前 `gaussian_part_vote.npy` 只是全 0 初始化，没有真正执行多帧/多视角投票、邻域平滑或依据渲染误差纠正标签。

已有保护机制：

- `part_max_smpl_dist` 距离阈值：距离太远的 Gaussian 拒绝硬分配，改为 `unknown`。
- `unknown` 使用 global/unknown expert，避免直接进入错误的具体 Part expert。
- confidence fusion 中，低置信度或 unknown Gaussian 保留 temporal 分支输出，减少错误 Part label 的破坏。

因此这些机制只能做到“少用不可靠标签”，不能做到“把错误 label 改正确”。当前没有 Part ground-truth 或 confusion matrix，不能宣称 Part 错误率已经下降。

后续若继续做 Part 方向，优先考虑：多邻近 SMPL 顶点的距离加权投票、Gaussian 邻域 label consistency smoothing、低置信度 soft label，以及结合多视角可见性/渲染误差的 label reweighting，并单独统计标签一致性或 confusion matrix。

## 11. 2026-09-02 Temporal-Conditioned Full Part-MoE

说明：
- 这是独立于 point 主线的 MAPO/Part-MoE 组合消融；本节只记录本次用户明确要求的实验结果。
- 方案为：shared trunk -> temporal branch -> 7 个跨时间复用的完整 part predictor。
- 不是 28 个独立的“时间 branch + part” MLP；时间 branch 负责时间段特征，part expert 负责完整的 d_xyz、d_rotation、d_scaling 预测。
- SMPL/LBS 初始化保持不变，Gaussian 点不复制；`part_moe_start_iter=10000` 后激活完整 part predictor，并使用时间边界 soft routing。

固定配置：
```text
iterations=25000
densify_until_iter=1800
seed=0
mapo_max_partition_level=2  # 4 temporal branches
mapo_soft_routing=true
mapo_partial_sharing=true
part_label_schema=part_moe_leg
num_parts=7
```

运行标识和结果根目录：
```text
RUN_TIME=20260902_tcpm
output/DNA-Rendering/<sequence>/mapo_temporal_conditioned_part/20260902_tcpm/
```

六序列 novel-view 25000 指标：
```text
0007_04: PSNR 29.542285839716595, SSIM 0.9585386931896209, LPIPS*1000 44.24141476241251
0019_10: PSNR 35.29382384618123,  SSIM 0.981044007341067,  LPIPS*1000 21.128202369436622
0044_11: PSNR 32.98410018285116,  SSIM 0.9780986537535985, LPIPS*1000 21.233410116595526
0051_09: PSNR 28.72421719233195,  SSIM 0.971628422041734,  LPIPS*1000 30.948133231140672
0206_04: PSNR 31.513268407185873, SSIM 0.9705617929498355, LPIPS*1000 33.00029526775082
0813_05: PSNR 36.13396037419637,  SSIM 0.9870684981346131,  LPIPS*1000 18.174226279370487

mean: PSNR 32.36527712874942, SSIM 0.9744901013871035, LPIPS*1000 28.120668342388754
```

完整性确认：六个序列均存在 `Training complete`、`point_cloud/iteration_25000/point_cloud.ply` 和 `metrics/results_novelview_25000.json`。

## 11. 2026-08-30 VGGT garment strict budget 对比

本轮只运行 `0007_04` 和 `0206_04`，训练设置统一为：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
VGGT candidate cache = corrected candidates.npz
VGGT replacement window = 800-1500, interval=100
strict budget = enabled
final evaluation = iteration 25000 novel-view
```

对比方法：

```text
vggt_garment_strict:
    SMPL 初始化 + VGGT 衣物候选 + 固定预算替换

vggt_garment_strict_high_error:
    上述方法 + 当前 render-GT 高误差区域参与候选筛选
    + 高误差 Gaussian 和候选父点保护
```

输出路径：

```text
0007 strict:
output/DNA-Rendering/0007_04/vggt_garment_strict/20260830_vggt_strict_0007/
0007 strict + high-error:
output/DNA-Rendering/0007_04/vggt_garment_strict_high_error/20260830_vggt_higherr_0007/
0206 strict:
output/DNA-Rendering/0206_04/vggt_garment_strict/20260830_vggt_strict3_0206/
0206 strict + high-error:
output/DNA-Rendering/0206_04/vggt_garment_strict_high_error/20260830_vggt_higherr_0206b/
```

最终点数检查：

```text
0007 original target = 35540
0007 strict final PLY = 35540
0007 strict + high-error final PLY = 35540

0206 original target = 58936
0206 strict final PLY = 58936
0206 strict + high-error final PLY = 58936
```

最终预算校正日志：

```text
0007 strict: before=35244, pruned=0, padded=296, final=35540
0007 strict + high-error: before=35912, pruned=372, padded=0, final=35540
0206 strict: before=58661, pruned=0, padded=275, final=58936
0206 strict + high-error: before=59712, pruned=776, padded=0, final=58936
```

每个替换事件均检查到 `replaced == spawned`。高误差版本确实执行了高误差筛选和保护：

```text
0007 iter=800: high_error_points=2899, selected_high_error=194,
              filtered_high_error=337, replaced=512, spawned=512
0206 iter=800: high_error_points=3982, selected_high_error=342,
              filtered_high_error=424, replaced=512, spawned=512
```

最终 novel-view 指标（LPIPS 已乘 1000，原始精度保留）：

| 实验 | 序列 | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| original | 0007_04 | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 |
| vggt_garment_strict | 0007_04 | 29.547481075922647 | 0.9586173683404923 | 44.405649369582534 |
| vggt_garment_strict_high_error | 0007_04 | 29.536748838424682 | 0.9585092023015022 | 44.61549075009922 |
| original | 0206_04 | 31.31386383374532 | 0.9695543631911278 | 33.811808843165636 |
| vggt_garment_strict | 0206_04 | 31.282401498158773 | 0.9691947018106778 | 34.135206726690136 |
| vggt_garment_strict_high_error | 0206_04 | 31.395742559432982 | 0.969868399699529 | 33.67418895165126 |

相对 original 的差值（PSNR/SSIM 越大越好，LPIPS*1000 越小越好）：

```text
0007 strict:
    ΔPSNR=+0.01614810625711982
    ΔSSIM=+0.00018004079659783567
    ΔLPIPS*1000=-0.6052370027949394

0007 strict + high-error:
    ΔPSNR=+0.005415868759154563
    ΔSSIM=+7.18747576077261e-05
    ΔLPIPS*1000=-0.3953956222782565

0206 strict:
    ΔPSNR=-0.03146233558654643
    ΔSSIM=-0.00035966138044996043
    ΔLPIPS*1000=+0.32339788352449966

0206 strict + high-error:
    ΔPSNR=+0.08187872568766252
    ΔSSIM=+0.00031403650840122754
    ΔLPIPS*1000=-0.13761989151437476
```

结论：
- 严格预算约束已实现，最终点数与 original 完全一致，新增数和删除数在每次替换中严格相等。
- `0007_04` 的 strict 和 strict + high-error 都略优于 original，但 strict-only 的提升更大；高误差策略在该序列没有继续提升。
- `0206_04` 的 strict-only 低于 original，而加入高误差筛选和保护后，三个指标均优于 original，说明高误差策略对该序列有效。
- 高误差策略不是普遍稳定增益：两个序列中仅 `0206_04` 明显受益，`0007_04` 的收益弱于 strict-only。当前更合理的判断是高误差保护具有序列依赖性，候选质量和删除区域仍需进一步分析。
- 这次对比验证了方法和预算公平性，但不能仅凭两个序列宣称稳定优于 original；下一步应扩大序列覆盖，并统计被保护点、被删除点与最终渲染误差的空间对应关系。

## 12. 1500 步公平对比的归因判断

现象：
- `part_moe_leg_unknown_route_strong` 在 `densify_until_iter=1500` 下，整体仍比 `part_point` 稍强
- 说明两个模块直接叠加后，并不保证比单独的 `part_moe` 加强分支更好

当前判断：
- 两个模块不是严格可加的独立收益
- `point_tem_mul` 更像弱先验，主要提供局部修补信号
- `part_moe_leg_unknown_route_strong` 本身已经吃到了更主要的收益
- 叠加后可能出现优化目标重叠、梯度互相稀释、点位/路由决策互相干扰

简短结论：
- 这不是“两个模块各自有效，所以加起来必然更强”
- 更像“主增益来自 part 侧，point 侧只是补充，组合后没有形成协同”

## 11. 2026-08-27 original 0044_11 / densify_until_iter=1800

结果来源：

```text
output/DNA-Rendering/0044_11/original/20260827_original_0044_1800/metrics/results_novelview_25000.json
```

novel-view 25000：

```text
0044_11: PSNR 32.950079933802286, SSIM 0.9779646729429563, LPIPS*1000 21.350521423543493
```
- 纯 `point_tem_mul` 单独结果低于 `part_point`，说明混合后的额外收益里很可能也包含 part route 的贡献

更严谨的表述是：
- `point_tem_mul` 有独立实验记录，但单独不等于 `part_point`
- 但 `part route` 也是共同变量
- 目前只能说 `point_tem_mul + part route strong` 组合优于 `part_moe_leg`，不能说提升全由 point 侧带来

## 10. part_point 相对 original 的代码级改动

`original` 配置：
- `use_part_moe=False`
- 没有启用 `use_point_anchor_tb`
- `num_parts=5`
- `part_label_schema=anatomy5`
- 非刚性变形走单个 shared `NonrigidDeformer` MLP

`part_point` 配置：
- `use_part_moe=True`
- `use_part_point=True`
- `use_point_anchor_tb=True`
- `use_part_score_route=True`
- `part_moe_start_iter=10000`
- `part_moe_warmup=1000`
- `part_moe_global_keep=0.1`
- `num_parts=7`
- `part_label_schema=part_moe_leg`
- `part_score_route_mode=delta`
- `part_score_route_alpha=1.5`
- `part_score_route_gate_bias=0.0`

底层改动分三块：

1. point 侧：
   - 在 800-1800 step，每 100 step 选择高误差 patch
   - score = render error * temporal history * boundary prior
   - 把高分 patch 附近可见 Gaussian 作为 anchor
   - 从 anchor 复制属性并随机偏移生成子 Gaussian
   - `part_point` 不启用 `use_point_update`，因此这里不是 fixed-budget 删除/替换主线

2. part MoE 侧：
   - 10000 step 生成 Gaussian part label
   - schema 从 `anatomy5` 改成 `part_moe_leg`
   - 7 类为 unknown/body/left_hand/right_hand/face/left_leg_foot/right_leg_foot
   - label 由 canonical Gaussian 到 SMPL-X canonical 顶点最近邻得到
   - 到达 start_iter 后，把 shared non-rigid MLP 复制成多个 part expert
   - 冻结原 shared 分支，后续训练 part experts
   - forward 时输出 global expert 与对应 part expert 的加权混合，`global_keep=0.1`

3. unknown route strong 侧：
   - 在 `forward_part_moe` 内额外启用 `PartScoreRouteAdapter`
   - 输入包括 base features、boundary score、motion score、unknown score、query xyz
   - `delta` 模式会按 route gate 调整 part_weight
   - 对 label=unknown 的点，不只走 unknown/global expert，而是用 learned expert logits 在 1-6 号 part experts 间 soft routing

简短判断：
- 相对 original，`part_point` 不是只改点，也改了非刚性变形结构
- 它同时加入了早期误差/时序/边界引导补点，以及后期 part-aware non-rigid expert routing
- 因此和 original 的差异是“Gaussian 分布优化 + 非刚性 deformation capacity 分部化”的组合

## 11. 为什么是 800-1800

这个窗口更像是**工程上固定下来的 densification 阶段**，不是理论推出来的最优解。

代码里的依据是：
- `point_start_iter = 800`
- `point_anchor_start_iter = 800`
- `point_anchor_end_iter = 1800`
- `point_tb_start_iter = 800`
- `point_tb_end_iter = 1800`
- `densify_until_iter` 默认是 `1500`
- 但只要开了 point 类模块，脚本会把它自动抬到 `1801`

也就是说：
- 800 之前先让基础形状和非刚性分支稳定一点
- 800-1800 这段专门给点补充/重分配
- 1800 之后基本不再让这条线继续大规模动点，避免和后面的 part MoE 阶段互相干扰

所以它的“依据”主要是：
- 和现有训练日程对齐
- 和默认 point / densify 日程一致
- 经验上把补点窗口放在早中期，而不是太早或太晚

更准确地说：
- 这是一个合理的默认窗口
- 但不是已经被严格消融证明“800-1800 最优”

## 12. point_tb 用到的先验到底有哪些

严格说不只三个。

当前 `select_point_anchor_tb_anchors` 里真正参与打分/筛选的有：
- `error_map`：当前 render 和 gt 的像素误差
 - `history_score`：历史误差的指数滑动平均，表示“这块区域是不是反复出错”
- `boundary_map`：GT 前景 mask 的边界带
- `edge_map`：GT 图像的边缘图，但只有 `edge_beta > 0` 时才启用
- `nonrigid_norm`：非刚性位移强度，但只有 `nonrigid_beta > 0` 时才启用
- `cov_patch` / `min_persistence` / `visibility`：这是筛选门控，不是主先验

就 `part_point` 这次配置来说：
- `boundary_beta=0.5`
- `temporal_alpha=0.5`
- `edge_beta=0.0`
- `nonrigid_beta=0.0`

所以实际在这次实验里，核心就是：
- 误差
- 时序历史
- 边界

它们的来源也很直接：
- 误差：当前图像和 GT 的差
- 时序：前几轮同一 patch 的误差 EMA
- 边界：GT 前景 mask 做形态学膨胀/腐蚀后得到的边界带

这里的 EMA 就是指数滑动平均：
- 新一次的 patch 误差不会直接完全覆盖旧记忆
- 而是 `history = momentum * old_history + (1 - momentum) * current_error`
- `momentum=0.8` 时，旧历史占 80%，当前误差占 20%

它表达的时序含义是：
- 如果一个 patch 连续几轮都高误差，history 会慢慢变高
- 下次选点时，这块区域更容易再次被选中
- 这不是显式时间编码，而是“跨迭代的误差记忆”

这个 `current_error` 的基准就是：
- 当前 render 和 GT 的逐像素绝对差
- 先乘前景 mask，只看目标区域
- 再按 patch 做平均，得到 `err_patch`

训练时是在线算的：
- 每次迭代都会随机取一个训练相机
- 用当前模型参数渲染出一张新图
- 再拿这张 render 和该视角对应的 GT 图像做差

所以这里的误差不是“缓存图像差”，而是“当前迭代模型输出 vs 当前 GT 视角”的即时误差。

## 13. 如果拉长到 1000-3000 会怎样

这件事**不能直接说一定更好**，更像一个要单独做消融的问题。

可能的收益：
- 让补点持续覆盖更久，可能对慢收敛序列有帮助
- 对一些后期才显露的问题区域，可能还能继续修正

主要风险：
- 后期模型已经更稳定，继续频繁增删点可能引入抖动
- 误差会越来越“跟着当前模型状态走”，容易追逐局部噪声
- 更长窗口会让点结构一直变化，和已有训练日程的稳定阶段冲突

所以更像是：
- 1000-3000 不排除能涨
- 但它不是当前代码里默认更优的设置
- 真要试，应该把它当成单独窗口消融，而不是直接假设越长越好

## 14. 0044 更稳妥的 point 窗口候选

对 0044 这条序列，若目标是让实验更贴近 original 的可运行预算，当前更合理的候选是：

```text
densify_until_iter = 1500
point_tb_start_iter = 500
point_tb_end_iter = 1500
```

判断：
- 这比 `800-1800` 更稳
- 可以避免原版在 `1800` 附近的 OOM 风险
- 也更符合“在 original 可跑预算内做 point 优化”的目标

但要注意：
- 这只修正 point 窗口，不会自动把 `part_point` 变成纯 point 实验
- 如果还保留 `use_part_moe=True` 和 `use_part_score_route=True`，归因仍然是混合的
- 所以它适合做更公平的 ablation，不适合直接拿来宣称“纯 point 提升”
- 由于 `part_point` 这条线实际启用的是 `point_anchor_tb`，`point_anchor_start/end` 也建议同步改成 `500/1500`

## 15. 2026-08-26 计划中的公平窗口 part_point

这次要补跑的配置是：

```text
densify_until_iter = 1500
point_anchor_start_iter = 500
point_anchor_end_iter = 1500
point_tb_start_iter = 500
point_tb_end_iter = 1500
```

执行方式：
- GPU2 跑 `0044_11 0051_09 0206_04`
- GPU3 跑 `0813_05 0007_04 0019_10`

目的：
- 把 `part_point` 的 point 窗口压回 original 可运行预算
- 避免再让 `point` 侧比 baseline 多出一段 1500 之后的 densify 时间
- 后面只用这组结果判断公平窗口下的 `part_point` 是否还有稳定收益

启动注意：
- 这条线不能直接用 `/home/anaconda3/bin/python`
- 该解释器缺 `torch`，需要改用 `seqavatar` 环境里的 python
- 后续补跑都统一显式传 `PYTHON_BIN=/media/coding/ckx/.conda/envs/seqavatar/bin/python`

## 16. 2026-08-26 正式启动记录

正式补跑已启动，当前配置：

```text
RUN_TIME = 20260826_155500_pp500_1500
PYTHON_BIN = /media/coding/ckx/.conda/envs/seqavatar/bin/python
GPU2 = 0044_11 0051_09 0206_04
GPU3 = 0813_05 0007_04 0019_10
```

说明：
- 运行方式改成 `tmux` 托管，避免外层会话退出影响训练
- 这次才是有效的 `densify_until_iter=1500 + point_tb_start/end=500/1500` 公平窗口

## 17. 2026-08-27 当前结果摘要

已完成的 5 个序列结果如下：

```text
0007_04: PSNR 29.654420185089112, SSIM 0.9591877604524295, LPIPS*1000 43.112088833004235
0019_10: PSNR 35.446163177490234, SSIM 0.9816510011752446, LPIPS*1000 20.21330191443364
0044_11: PSNR 32.98979956309,     SSIM 0.9781737715005875, LPIPS*1000 21.071360058461625
0051_09: PSNR 28.767481962839764, SSIM 0.9720098808407783, LPIPS*1000 30.172928733130294
0813_05: PSNR 36.18334201176961,  SSIM 0.9872277051210403, LPIPS*1000 18.196872373421987

mean over 5 finished seqs:
    PSNR 32.608241380055745
    SSIM 0.9756500238180161
    LPIPS*1000 26.55331038249036
```

补充：
- `0206_04` 在训练中报了 `CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`
- 所以这轮还不是完整 6 序列均值，当前只能先记 5 序列完成结果

### 0206_04 重跑完成

这次 `0206_04` 已重跑成功，novel-view 25000 指标为：

```text
0206_04: PSNR 31.53406097094218, SSIM 0.9704986770947774, LPIPS*1000 33.128568048899375
```

六序列均值更新为：

```text
PSNR 32.42921131187015
SSIM 0.9747914660308096
LPIPS*1000 27.649186660225194
```

## 18. 评价指标输出约定

以后你问评价指标时，统一用表格输出：
- 列为 `Sequence`, `PSNR`, `SSIM`, `LPIPS*1000`
- 不做四舍五入，保留原始值
- 若有序列未完成，单独标注 `未完成`

## 19. 2026-08-27 0206 重跑记录

`0206_04` 的公平窗口重跑已单独挂到 GPU2：

```text
RUN_TIME = 20260827_0206_retry_pp500_1500
GPU_id = 2
SEQUENCES_OVERRIDE = 0206_04
PYTHON_BIN = /media/coding/ckx/.conda/envs/seqavatar/bin/python
```

仍然保持同一组参数：
- `densify_until_iter = 1500`
- `point_anchor_start/end = 500/1500`
- `point_tb_start/end = 500/1500`

结果已完成：

```text
0206_04: PSNR 31.53406097094218, SSIM 0.9704986770947774, LPIPS*1000 33.128568048899375
```

## 20. part_point 的来源拆分

`part_point` 这条线的组合方式是：

```text
part_point = point_tem_mul + part_moe_leg_unknown_route_strong
```

这里的 `part_moe_leg_unknown_route` 部分，和 `part_moe_leg_unknown_route_strong` 的实验设置是一致的，核心参数都对齐：

```text
use_part_moe=True
part_moe_start_iter=10000
part_moe_warmup=1000
part_moe_global_keep=0.1
num_parts=7
part_label_schema=part_moe_leg
use_part_score_route=True
part_score_route_hidden_dim=128
part_score_route_alpha=1.5
part_score_route_gate_bias=0.0
part_score_route_mode='delta'
part_score_route_use_route_gate=1
part_score_route_use_unknown_mix=1
part_score_route_signal_mode='full'
```

所以更准确地说：
- `part_point` 不是接了普通版 `part_moe_leg_unknown_route`
- 它接的是 `part_moe_leg_unknown_route_strong` 这套增强路由做法
- point 侧再叠加的是 `point_tem_mul`

## 21. 2026-08-27 strong baseline 公平重跑

为了和 `part_point` 做公平对照，单独重跑 `part_moe_leg_unknown_route_strong`：

```text
densify_until_iter = 1500
```

说明：
- `part_moe_leg_unknown_route_strong` 本身不启用 point_anchor_tb
- 所以这次公平性主要体现在和 `part_point` 统一 `densify_until_iter=1500`
- point 侧保持关闭，只看纯 strong route baseline

启动计划：
- GPU1 跑一半序列
- GPU2 跑一半序列

结果已完成：

```text
0007_04: PSNR 29.64151717821757,  SSIM 0.9591088935732841, LPIPS*1000 43.48686106192569
0019_10: PSNR 35.44831476211548,  SSIM 0.9814966107408205, LPIPS*1000 20.612236477124192
0044_11: PSNR 33.014883295694986, SSIM 0.9782224794228872, LPIPS*1000 21.213388745673
0051_09: PSNR 28.798695230484007, SSIM 0.9719721600413322, LPIPS*1000 30.178756763537724
0206_04: PSNR 31.579859765370685, SSIM 0.9707841078440348, LPIPS*1000 33.07823079327742
0813_05: PSNR 36.20627508163452,  SSIM 0.9872405921419461, LPIPS*1000 17.88949832941095

mean: PSNR 32.448257552252876
SSIM 0.9748041406273842
LPIPS*1000 27.743162028491497
```

和 `part_point (500-1500)` 的均值相比：

```text
ΔPSNR  +0.018852896160556
ΔSSIM  +0.0000036804212463
ΔLPIPS +0.110205575927267
```

## 22. 两次 part_point 对比

两次 `part_point` 分别是：

```text
800-1800: 20260825_213900
500-1500: 20260827_0206_retry_pp500_1500
```

六序列均值对比：

```text
800-1800:
    PSNR 32.42837428251902
    SSIM 0.9748193326095741
    LPIPS*1000 27.40108670356373

500-1500:
    PSNR 32.42940465609232
    SSIM 0.9748004602061378
    LPIPS*1000 27.63295645256423
```

差值（500-1500 相对 800-1800）：

```text
ΔPSNR  +0.0010303735733019
ΔSSIM  -0.0000188724034363
ΔLPIPS +0.2318697489995
```

简短判断：
- 500-1500 在 PSNR 上只略高一点
- SSIM 和 LPIPS 没有同步变好
- 这说明把窗口压到原本公平预算后，优势没有变强，整体还是弱差异

## 23. 9000-10000 点窗口是否可行

结论：
- **可行，但不能精确做到 10000 这一轮**
- 因为训练里点增删分支都包在 `if iteration < opt.densify_until_iter` 里
- 又因为 `use_part_moe=True` 时要求 `densify_until_iter <= part_moe_start_iter`
- 还因为点分支本身是按 `interval=100` 触发，不会在 9999 这种非整百步执行

所以如果设：

```text
part_moe_start_iter = 10000
densify_until_iter = 10000
```

那么：
- 点增删窗口最晚只能跑到 `9900`
- `10000` 这一轮会直接跳出 densify 分支，转去做 part label / expert 初始化
- `9999` 虽然小于 `10000`，但不是 `100` 的倍数，所以不会触发点增删

更稳妥的做法是：
- 把点窗口放成 `9000-9900`
- 或者 `8500-9500`
- 留出和 `part_moe_start_iter=10000` 的缓冲

补充：
- 训练参数仍会每步更新 Gaussian 的连续属性
- 但“结构性增删点”只会在 densify 窗口里发生

## 24. 2026-08-27 original 0044_11 1800 重跑

为了补一个更长 densify 窗口下的 `original` 对照，单独重跑：

```text
MODE = original
SEQUENCES_OVERRIDE = 0044_11
DENSIFY_UNTIL_ITER = 1800
GPU_id = 1
```

目的：
- 看 `original` 在 1800 窗口下的 `0044_11` 指标
- 以后和 `part_point`、`part_moe_leg_unknown_route_strong` 继续做公平比较

目的：
- 只补回之前在 0206 上失败的那一个序列
- 不改实验要求，只换成单序列重跑，直到拿到最终评价指标

## 25. 2026-08-27 DNA 六序列 1800 重跑

本次重新启动两组六序列实验，统一 `densify_until_iter=1800`，其余设置保持论文基线一致。

启动信息：

```text
original
    GPU_id = 1
    RUN_TIME = 20260827_dna6_original_1800_tmux
    MODE = original

part_moe_leg
    GPU_id = 2
    RUN_TIME = 20260827_dna6_part_moe_leg_1800_tmux
    MODE = part_moe_leg
```

当前状态：
- 两个 `tmux` 会话已启动
- 已进入 `train.py`
- 结果待收集

### 0044_11 已完成

`densify_until_iter=1800` 下的 0044_11 结果：

```text
original
    PSNR 32.98733657201131
    SSIM 0.9780929783980051
    LPIPS*1000 21.279019598538677

part_moe_leg
    PSNR 32.98198075294495
    SSIM 0.9782935485243798
    LPIPS*1000 20.876424588883915
```

当前已进入下一条序列 `0051_09`。

### 0051_09 original 已完成

```text
original
    PSNR 28.57111225128174
    SSIM 0.9706683094302814
    LPIPS*1000 31.55909333533297
```

当前状态：
- `original` 已开始下一条 `0206_04`
- `part_moe_leg` 仍在 `0051_09`

### 0051_09 part_moe_leg 已完成

```text
part_moe_leg
    PSNR 28.62366002400716
    SSIM 0.9711981738607088
    LPIPS*1000 31.169041828252374
```

当前状态：
- 两组都已进入 `0206_04`

### 0206_04 original 已完成

```text
original
    PSNR 31.313099559148153
    SSIM 0.9695186073581378
    LPIPS*1000 33.84117659491797
```

当前状态：
- `original` 已开始下一条 `0813_05`
- `part_moe_leg` 仍在 `0206_04`

### 0206_04 part_moe_leg 已完成

```text
part_moe_leg
    PSNR 31.53108598391215
    SSIM 0.9707265749573708
    LPIPS*1000 32.78309289986889
```

当前状态：
- `original` 仍在 `0813_05`
- `part_moe_leg` 已开始下一条 `0813_05`

### 0813_05 original 已完成

```text
original
    PSNR 36.10658038457235
    SSIM 0.9869976962606112
    LPIPS*1000 18.244266610903045
```

当前状态：
- `part_moe_leg` 仍在 `0813_05`

### 0813_05 part_moe_leg 已完成

```text
part_moe_leg
    PSNR 36.24788769086202
    SSIM 0.9874844372272491
    LPIPS*1000 17.522351994800072
```

## 26. 2026-08-27 DNA 六序列 1800 最终结果

两组都已跑完，`tmux` 会话和训练进程都结束了。

### original

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0007_04 | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 |
| 0019_10 | 35.257725365956624 | 0.9808832183480263 | 21.035196678712964 |
| 0044_11 | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 |
| 0051_09 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 |
| 0206_04 | 31.31386383374532 | 0.9695543631911278 | 33.811808843165636 |
| 0813_05 | 36.10658038457235 | 0.9869976962606112 | 18.244266610903047 |

mean:
- PSNR 32.310457277297975
- SSIM 0.9742256858282618
- LPIPS*1000 28.405228959551705

### part_moe_leg

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0007_04 | 29.60400875409444 | 0.959126636882623 | 43.3049468168368 |
| 0019_10 | 35.41057926813761 | 0.9814167340596517 | 20.47669793634365 |
| 0044_11 | 32.98199820518494 | 0.9782939051588376 | 20.875872555188835 |
| 0051_09 | 28.71769100824992 | 0.9719194740056991 | 30.67279694757114 |
| 0206_04 | 31.53526193300883 | 0.9707902779181798 | 32.718987095480166 |
| 0813_05 | 36.24788769086202 | 0.9874844372272491 | 17.522351994800072 |

mean:
- PSNR 32.41623780992296
- SSIM 0.9748385775420401
- LPIPS*1000 27.595275557703445

### 对比

- `part_moe_leg` 比 `original` 略好
- 六序列均值差：
  - `ΔPSNR +0.105780532624985`
  - `ΔSSIM +0.0006128917137783`
  - `ΔLPIPS -0.8099534018482599`

## 27. 代码推送准备

本地仓库状态：
- 当前分支：`newstart`
- 最新提交：`c3338bd`
- 提交信息：`0827新服务器`
- `origin` 已切到 SSH：
  - `git@github.com:merlin-0728/seqavatar.git`

SSH 公钥：

```text
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOBopyOUZl85Ws/NxJgaEPRS9GfAxN347Qat6pQKPjQD merlin-0728@seqavatar
```

## 28. 代码已推送

当前状态：
- 本地分支：`newstart`
- 远端分支：`origin/newstart`
- 最新提交：`2d6518f`
- 工作区：clean

说明：
- 这次已经把 `newstart` 上的已提交代码推到 GitHub
- `.gitignore` 排除的 `output/`、`logs/`、数据集、编译产物、模型二进制没有进入仓库

## 29. 提交区别说明

`c3338bd` 和 `2d6518f` 不是两套不同代码逻辑：
- `c3338bd`：`0827新服务器`，是先前已经完成的代码提交
- `2d6518f`：`Update motion2 with latest experiment notes`，只是后来补写 `note/motion2.md`

所以这次 push 的实际内容是：
- 代码主提交 `c3338bd`
- 以及一条只改笔记的补充提交 `2d6518f`

## 30. DynOMo C 当前状态

- 这是单独的 `original` 轨道实验，不并入 point 主线
- 当前实现方向：canonical kNN + 特征相似度加权 + SMPL part gate
- 训练侧已经接入到 `full_aiap_loss` 的替代分支，位置/运动/旋转项的梯度会回到 non-rigid 主干；但 affinity head 本身被权重 detach，当前正则不能直接训练它
- 四序列评测已经完成，但实际 densify budget 是 1500，不能与 `original=1800` 做严格公平比较

## 31. DynOMo C 这次报错与修复

- 这次失败的根因是 `train.py` 里把 `d_xyz / d_rotation` 先做了 `norm(dim=-1)`，再送进 `weighted_aiap_loss`
- 这样 `motion_obs / rotation_obs` 变成了 `[1, N]`，后面按点邻域索引时会把 batch 维当点维，用错维度
- `utils/loss_utils.py` 已改成直接做邻域一致性，不再用 `smooth_l1_loss` 的展开版本
- `train.py` 已改成直接传入 `d_xyz / d_rotation`，后续重新跑四序列

## 32. DynOMo C 四序列结果（2026-08-28）

实验设置：
- 基线：`original`
- 改进：DynOMo 版本 C，`use_dynomo_c=True`
- 训练步数：25000
- 实际 `densify_until_iter=1500`，不是 1800
- 评估：DNA-Rendering novel-view，25000 step，120 views
- 对比基线路径：`output/DNA-Rendering/*/original/20260827_dna6_original_1800_tmux/metrics/results_novelview_25000.json`
- DynOMo C 路径：`output/DNA-Rendering/*/dynomo_c/20260828_dynomo_c/metrics/results_novelview_25000.json`

### 逐序列指标

| Sequence | original PSNR | DynOMo C PSNR | ΔPSNR | original SSIM | DynOMo C SSIM | ΔSSIM | original LPIPS*1000 | DynOMo C LPIPS*1000 | ΔLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0044_11 | 32.98732282320658 | 32.96389880180359 | -0.023424021402995265 | 0.9780931328733762 | 0.9778484349449476 | -0.0002446979284286277 | 21.27879182808101 | 21.520101302303374 | +0.24130947422236204 |
| 0051_09 | 28.66591828664144 | 28.71858377456665 | +0.05266548792521064 | 0.9713883767525355 | 0.9714174206058184 | +0.00002904385328283965 | 31.05042342407008 | 30.836422624997795 | -0.2140007990722843 |
| 0206_04 | 31.31386383374532 | 31.34501315752665 | +0.031149323781331617 | 0.9695543631911278 | 0.9694132258494695 | -0.00014113734165821512 | 33.811808843165636 | 34.67745290448268 | +0.8656440613170417 |
| 0813_05 | 36.10658038457235 | 36.075470733642575 | -0.03110965092977125 | 0.9869976962606112 | 0.9868646492560704 | -0.0001330470045407539 | 18.244266610903047 | 18.4226265642792 | +0.17835995337615174 |

### 四序列均值

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| original | 32.26842133204142 | 0.9765083922694127 | 26.09632267655494 |
| DynOMo C | 32.275741616884865 | 0.9763859326640765 | 26.36415084901576 |
| Δ（DynOMo C - original） | +0.007320284843444824 | -0.00012245960533618927 | +0.26782817246082047 |

### 结论

- 没有形成整体指标提升。
- PSNR：2/4 序列提升，均值仅 `+0.007320284843444824`，属于弱信号。
- SSIM：4 序列中只有 `0051_09` 略升，均值下降 `-0.00012245960533618927`。
- LPIPS：只有 `0051_09` 变好，均值变差 `+0.26782817246082047`；LPIPS 越低越好。
- 当前结果只能说明版本 C 已跑通并可能对个别序列有帮助，不能支持“DynOMo C 优于 original”的结论。
- 由于版本 C 同时引入了 feature affinity、SMPL part gate、motion consistency 和 rotation consistency，当前四序列结果也不能归因到其中某一个子模块；下一步应做 feature-only、part-only、feature+part、再逐步加入 motion/rotation 的消融。

## 33. DynOMo C 相对 original 的代码级分析与下一步方向（2026-08-28）

### 当前真正改动的内容

这次 `dynomo_c` 是独立的 original-baseline ablation，不启用 point、Part-MoE、tri、time 或其他模块。相对 `original`，实际改动有四层：

1. 在 `NonrigidDeformer` 的 shared MLP 后增加 `dynomo_affinity_head`：
   - 输入是 MLP 中间特征 `h`
   - 结构为 `Linear(512, 256) -> ReLU -> Linear(256, 32)`
   - 输出做 L2 normalize，作为 Gaussian motion affinity feature
   - 该特征只用于训练 loss 的邻域权重，不改变 Gaussian 的增删和渲染结构

2. 增加 SMPL part pseudo-instance 标签：
   - 每个 canonical Gaussian 找最近的 SMPL-X canonical vertex
   - 用 SMPL-X 顶点的 dominant LBS joint 映射到 `anatomy5`
   - 标签为 `unknown/body/left_hand/right_hand/face`
   - `dynomo_c_label_iter=0`，初始就生成标签；Gaussian 数量变化后重新生成，保证标签长度匹配
   - 因为 `use_part_moe=False`，这些标签只进入 DynOMo loss，不会让 NonrigidDeformer 切换成多个 part expert

3. 用 `weighted_aiap_loss` 替换原来的 `full_aiap_loss` 的位置约束：
   - canonical Gaussian 上做 K=5 的 kNN
   - 原始 AIAP 项是 canonical/deformed 邻域距离差的 L1
   - 每条邻边的权重为 `relu(cosine(affinity_i, affinity_j)) * part_gate`
   - 同 part 权重 1.0，相邻 part 权重 0.1，其他 part 权重 0
   - 权重只用于位置 AIAP；covariance AIAP 仍然调用原始的无权 `aiap_loss`

4. 增加两个形变一致性项：
   - `L_motion`：相邻 Gaussian 的 `d_xyz` 做逐分量 L1 一致性
   - `L_rotation`：相邻 Gaussian 的 `d_rotation` 做逐分量 L1 一致性
   - 当前权重均为 `0.01`
   - 这不是严格的旋转几何距离，而是网络输出 rotation delta 的分量差

RGB、mask、SSIM、LPIPS、shared MLP 主体和 Gaussian densification 逻辑沿用 original。当前运行没有启用 point 或 Part-MoE。
但这次 DynOMo C 的 densify budget 实际是 1500，而对照 `original` 使用的是 1800；因此原先表格不能作为严格公平的同预算 ablation。

### 当前实现中需要修正的点

- `dynomo_c_part_same_w`、`dynomo_c_part_adj_w`、`dynomo_c_part_other_w` 虽然被解析并打印，但 `train.py` 调用 `build_part_neighbor_weight_matrix()` 时没有把它们传入；矩阵内部固定使用 1.0、0.1、0.0。因此后续调这些参数目前不会生效。
- `weighted_aiap_loss` 对 feature weight 做了 `detach()`。这可以防止模型通过把权重压到 0 来逃避正则，但也意味着 affinity head 没有来自该 loss 的直接梯度。它只能通过共享的 `h` 间接变化，新增 head 本身不会被一致性目标训练成“运动相似特征”。
- 当前 part gate 是硬过滤。跨非相邻部件的边全部置零，unknown 也只与 unknown 连接；如果 SMPL 最近邻标签有误，正确的空间邻居会被直接丢弃。
- `L_cov` 没有采用 feature/part 权重，当前版本实际上是“加权位置 AIAP + 原始 covariance AIAP”，不是所有形变属性都做 DynOMo 加权。
- `L_rotation` 是输出向量的分量 L1，不等价于旋转矩阵或四元数上的 geodesic consistency；它可能把不同尺度/参数化下的 rotation delta 过度拉近。
- 当前实验把 affinity、part gate、motion consistency、rotation consistency 一次性打开，无法判断是哪一项造成了 `0051_09` 的收益或 `0206_04` 的 LPIPS 退化。

### 对四序列结果的进一步解释

从 3000、10000、25000 step 的日志看，DynOMo C 在 `0044_11` 和 `0813_05` 从早期到最终都略低于 original；`0051_09` 在 3000 step 较差，但 10000 和 25000 step 反超；`0206_04` 从 3000 step 就略好 PSNR，但 SSIM/LPIPS 没有同步改善。这说明当前正则没有形成一致的收敛优势，更像是对不同序列的局部偏置：

- 结构较稳定的序列可能被硬 part gate 错误过滤邻居
- 某些序列的空间邻域与 SMPL part 边界更一致，因此能获得少量 PSNR 收益
- affinity 没有直接监督，不能保证它真的代表运动区域相似性
- motion/rotation 项权重较小，可能只提供很弱的附加约束；但即使提高，也可能进一步过平滑衣服褶皱

### 下一步推荐顺序

1. 先修正实现，不先扩大超参数搜索：
   - 让 part gate 权重参数真正生效
   - 给权重增加非零 floor，避免 hard zero 造成邻域断裂
   - 先移除随机 affinity head，直接使用已经被 RGB/形变任务训练过的 MLP 中间特征 `h.detach()` 做相似度，验证“已有运动特征 + part gate”是否有效

2. 做最小可归因消融，四个序列保持完全相同设置：
   - `C-part-only`：只启用 part gate，关闭 affinity、motion、rotation
   - `C-feature-only`：只启用 feature affinity，关闭 part gate 和额外 consistency
   - `C-feature+part`：启用 feature affinity 与 part gate
   - `C-full`：在上一个版本上再加入 motion/rotation

3. 如果 `feature+part` 仍无收益，再训练 affinity：
   - 用 detached 的 motion similarity 或 part compatibility 构造 pseudo target
   - 用单独的 affinity auxiliary loss 训练 affinity head
   - consistency 权重仍使用 stop-gradient，避免通过降权逃避约束

4. 最后再调正则强度：
   - 先固定 `L_motion=0`、`L_rotation=0`，避免把衣服细节过度平滑
   - 确认位置约束有稳定收益后，再分别测试 motion 和 rotation
   - rotation 优先改成归一化四元数/旋转矩阵的相对旋转距离，而不是直接分量 L1

当前最合理的阶段性结论是：代码已经把 DynOMo 思路接入了主干 loss，但版本 C 的 feature 分支尚未真正学会运动 affinity，且四序列结果没有整体优于 original。下一步重点应是修正可训练性和做可归因消融，而不是直接增大 consistency 权重。

## 34. DynOMo C 全量排查结论（2026-08-28）

本节是对本版本代码、启动脚本、输出目录、TensorBoard 和标签文件的逐项审计结果。

### A. 已确认的实现问题

1. **Affinity head 实际没有直接训练梯度。**
   - `nets/mlp_delta_non_rigid.py` 中 `dynomo_affinity_head(h)` 生成 affinity。
   - `utils/loss_utils.py` 在位置、motion、rotation 三项计算前对最终权重执行 `detach()`。
   - 数值梯度测试得到：共享位置输入有梯度，但 affinity 输入梯度为 `None`。
   - 因此新增 head 会被 RGB/AIAP 通过共享特征间接影响，但不会被当前 consistency loss 直接学成运动相似度特征；它的随机投影会长期参与邻域权重，设计目标没有真正实现。

2. **part weight 的命令行参数当前不生效。**
   - 脚本和参数解析器确实接收并打印 `same/adj/other` 三个权重。
   - 但 `train.py` 调用 `build_part_neighbor_weight_matrix(schema=...)` 时没有传入这三个值。
   - `part_label/common.py` 内部固定使用 `1.0/0.1/0.0`，修改环境变量不会改变实际矩阵。

3. **标签置信度被生成和加载，但没有进入 DynOMo loss。**
   - `gaussian_part_conf.npy` 会根据 Gaussian 到 SMPL 顶点的距离生成。
   - `GaussianModel` 保存了 `_part_conf`，但 `train.py` 只取 `get_part_label`，没有取 `get_part_conf`。
   - 低置信度的最近邻标签和高置信度标签在 gate 中被同等对待。

### B. 已确认的结构性风险

4. **硬 part gate 会切断跨部件邻居。**
   - `anatomy5` 实际只有 `unknown/body/left_hand/right_hand/face` 五类，非相邻部件权重为 0，unknown 只连接 unknown。
   - 在 iteration 1401 的实际标签上，四个序列的零权重边比例约为 `0.226468%` 到 `1.026564%`；所有 4 条邻边都为零的中心点约为 `0.057268%` 到 `0.088048%`。
   - 比例不大，但错误标签会让合法空间约束变成完全没有约束；这也是一个应修正的鲁棒性风险，而不是当前结果变差的唯一已证实原因。

5. **KNN 参数名与实际邻居数不一致。**
   - 当前 `dynomo_c_knn=5` 传给 `knn_points(K=5)`，再统一去掉第 0 个 self 邻居。
   - 所以每个 Gaussian 实际只使用 4 条非 self 邻边。原始 `aiap_loss` 也沿用同样的约定，属于两条轨道共有的实现约定，不是 DynOMo 单独引入的回归，但参数语义容易误读。

6. **Covariance 没有使用 DynOMo 权重。**
   - `loss_aiap_xyz`、`loss_motion`、`loss_rotation` 使用 feature/part 权重。
   - `loss_aiap_cov` 直接调用原始等权 `aiap_loss`。
   - 因此当前实现不是完整的 feature/part-weighted covariance consistency，而是“加权位置和运动项 + 等权 covariance 项”。

7. **rotation consistency 不是几何旋转距离。**
   - 现在比较的是 `d_rotation` 输出向量逐分量的 L1 差。
   - 它不是旋转矩阵或归一化四元数的相对旋转 geodesic distance，可能产生参数化相关的过平滑。

8. **全零权重时损失静默变成零。**
   - 分母用 `clamp_min(1e-6)`，但分子也全为零，因此位置、motion、rotation loss 都为 0，不会报错。
   - 对极少数被 gate 完全隔离的点，这意味着该正则完全失效。建议增加非零 floor 或回退到空间权重。

### C. 公平性与运行证据

9. **之前“DynOMo C 与 original 都是 1800”的记录不成立。**
   - 脚本默认 `densify_until_iter=1500`；DynOMo 模式不会触发 point 模式的自动 1801 逻辑。
   - DynOMo C 输出的标签最后到 `iteration_1401`，四个序列 TensorBoard 在 3000/10000/25000 的 `total_points` 已固定为：`0044_11=59961`、`0051_09=51385`、`0206_04=40887`、`0813_05=38416`。
   - 对照 original=1800 的对应最终点数为：`83909`、`68103`、`58936`、`50740`。
   - DynOMo C 的 `cfg_args` 没有保存 optimization 参数，但启动脚本默认值、标签刷新终点、最终点数和输出结构共同支持实际运行是 1500；不能把旧表格解读为同预算提升。
   - 现有 0044/0051/0206/0813 的 DynOMo C 与 original 指标仍可作为“不同 densify budget 下的现象记录”，不能作为干净的 DynOMo ablation 结论。

10. **没有找到可直接复用的 original=1500 novel-view 对照。**
    - `20260811_204020` original 目录有训练/点云/MLP 输出，但没有 `metrics/results_novelview_25000.json`。
    - 因此要获得公平指标，需要重新跑 original 四序列 `densify_until_iter=1500`，并使用新的唯一输出目录。

### D. 监控与工程问题

11. **当前日志无法恢复 DynOMo 正则项的实际量级。**
    - TensorBoard 只有训练 RGB/total loss、novel-view 指标、iter time 和 total_points。
    - 没有 `loss_aiap_xyz/cov/motion/rotation`、正权重边比例、平均权重或 gradient norm。
    - 所以不能从已完成实验判断正则是否主导训练、是否太弱或是否只在少数边上起作用。

12. **标签刷新正确但代价偏高，且元数据有误。**
    - Gaussian 数量变化时会重新建立 CPU `scipy.cKDTree`，写入磁盘，再读回 GPU；长度同步是正确的，但会产生大量磁盘 I/O。
    - DynOMo 复用了 `PartMoeController._build_prior_only_labels()`，所以元数据里的 `use_part_moe=true` 是错误标签；实际训练配置中 `use_part_moe=false`，不影响本次训练分支，但会误导实验审计。

### E. 当前结果应该如何表述

- 四序列 DynOMo C 的旧对比表不能作为“相对 original 提升”的证据，因为 densify budget 不一致。
- 即使暂时只看现象，DynOMo C 也没有形成稳定的三指标同步收益：它只在部分序列上改善 PSNR，SSIM 和 LPIPS 没有一致改善。
- 目前不能把结果归因到 feature affinity、part gate、motion consistency 或 rotation consistency 中的任何一个。

### F. 推荐修复和重跑顺序

1. 先记录并修正实际配置：在 `cfg_args` 或训练日志中显式写入 `densify_until_iter`，并为 DynOMo C 重跑 `original=1500` 对照。
2. 让 part 权重参数真正传入矩阵构造；接入 `part_conf`，对低置信度标签降低 gate 强度；对全零邻域保留空间回退权重。
3. 第一轮归因实验关闭随机 affinity head，使用 shared MLP 的 `h.detach()` 作为固定特征，先验证 feature/part 邻域本身是否有效。
4. 做 `part-only`、`feature-only`、`feature+part`，再分别加入 motion 和 rotation 的最小消融；所有实验统一 `densify_until_iter=1500`、随机种子、评估视角和输出配置。
5. 将 rotation consistency 改成旋转矩阵或四元数相对距离；在此之前不要直接增大 `rotation_w`。

最终判断：当前代码“接入了 DynOMo 形式的正则项”这一事实成立，但版本 C 尚未构成可信的 DynOMo 改进实验。最重要的两个实现缺陷是 affinity head 不可由当前正则直接训练、part 权重 CLI 参数不生效；最重要的实验缺陷是与 original 的 densify budget 不公平。下一步应先修复和重跑，再讨论指标提升。

## 35. DynOMo C 修复版重跑准备（2026-08-28）

本次按用户要求固定 `densify_until_iter=1800`，四个序列分别使用四张 GPU，并与已有同为 1800 的 original 结果比较。

### 已完成的修复

- affinity 权重不再在 loss 前 `detach()`，并设置 `affinity_floor=0.1`，保证 affinity head 可收到梯度且不会通过把所有权重压为 0 来关闭正则。
- `dynomo_c_part_same_w`、`dynomo_c_part_adj_w`、`dynomo_c_part_other_w` 已从优化参数传入 part 邻接矩阵，实际值不再固定。
- 标签 confidence 已用于调节邻边权重；硬 gate 导致某个中心没有有效邻边时，回退到空间/feature 权重。
- 输出目录新增 `train_cfg_args`，同时保存模型参数和优化参数；脚本日志显式打印 `DENSIFY_UNTIL_ITER`，便于复核公平性。

### 检查结果

- `bash -n scripts/exps_dnarendering.sh`：通过。
- `python -m py_compile`：通过。
- `git diff --check`：通过。
- affinity 数值梯度测试：非零；part 权重矩阵参数测试：生效；全零 gate 回退测试：损失非零且可反传。

### 重跑配置

```text
MODE = dynomo_c
ITERATIONS = 25000
DENSIFY_UNTIL_ITER = 1800
SEQUENCES = 0044_11, 0051_09, 0206_04, 0813_05
GPU = 0, 1, 2, 3（每卡一个序列）
评估 = novel-view, 25000 step, 120 views
对照 = original / 20260827_dna6_original_1800_tmux
```

当前状态：修复已通过检查。首次四卡启动因脚本默认 `/home/anaconda3/bin/python` 缺少 torch，在训练入口前退出；未产生训练结果。后续重跑显式使用已验证的 `/media/coding/ckx/.conda/envs/seqavatar/bin/python`。

## 36. DynOMo C 修复版 1800 公平重跑结果（2026-08-28）

### 完成与完整性检查

- 四个序列使用 GPU 0/1/2/3 并行完成，训练到 `iteration=25000`。
- DynOMo C 配置：`densify_until_iter=1800`、`dynomo_c_knn=5`、`same=1.0`、`adjacent=0.1`、`other=0.0`、`motion_w=0.01`、`rotation_w=0.01`。
- 每个序列均存在 `train_cfg_args`、`point_cloud/iteration_25000/point_cloud.ply` 和 `metrics/results_novelview_25000.json`。
- 四个 DynOMo 日志均以 `Training complete` 结束；错误扫描没有发现 `Traceback`、OOM、CUDA error、exception 或 NaN。
- 对照 original 日志在 1800 步的点数为 `0044_11=83909`、`0051_09=68103`、`0206_04=58936`、`0813_05=50740`，与本次比较使用同一 `densify_until_iter=1800`；其余训练和评估设置保持论文 baseline 配置。
- DynOMo C 在最终 25000 步的点数为 `0044_11=82600`、`0051_09=69115`、`0206_04=56603`、`0813_05=48814`。这是相同 densify 窗口下由不同梯度和点筛选结果导致的自然差异，不应表述为最终点数完全相同。

### Novel-view 25000 指标

| Sequence | DynOMo C PSNR | DynOMo C SSIM | DynOMo C LPIPS*1000 | original PSNR | original SSIM | original LPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| 0044_11 | 32.95318856239319 | 0.9780002603928248 | 21.27644782885909 | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 |
| 0051_09 | 28.692757670084635 | 0.9715872690081596 | 30.753879722518224 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 |
| 0206_04 | 31.281256516774494 | 0.9692579567432403 | 34.569685952737926 | 31.31386383374532 | 0.9695543631911278 | 33.811808843165636 |
| 0813_05 | 36.01941838264465 | 0.9867767910162608 | 18.31204507810374 | 36.10658038457235 | 0.9869976962606112 | 18.244266610903047 |
| mean | 32.23665528297424 | 0.9764055692901215 | 26.228014645554746 | 32.26842133204142 | 0.9765083922694127 | 26.09632267655494 |

### Difference: DynOMo C - original

| Sequence | Delta PSNR | Delta SSIM | Delta LPIPS*1000 | Better on all three |
|---|---:|---:|---:|---|
| 0044_11 | -0.034134260813395656 | -9.287248055134256e-05 | -0.002343999221922355 | no |
| 0051_09 | 0.02683938344319614 | 0.00019889225562408352 | -0.2965437015518546 | yes |
| 0206_04 | -0.032607316970825195 | -0.00029640644788742065 | 0.7578771095722927 | no |
| 0813_05 | -0.08716200192769463 | -0.00022090524435036674 | 0.06777846720069577 | no |
| mean | -0.031766049067179836 | -0.00010282297929126161 | 0.13169196899980287 | no |

### 结论

- 公平配置和实现修复已通过实际运行验证。
- 四序列平均没有提升：PSNR 和 SSIM 略低于 original，LPIPS*1000 略高（LPIPS 越低越好）。
- 只有 `0051_09` 三项同时改善；`0044_11` 的 LPIPS 有极小改善，但 PSNR/SSIM 下降。
- 因此当前修复版证明了 affinity、part 权重和回退逻辑能够稳定运行，但不能证明 DynOMo C 在这四个序列上带来整体性能提升。后续应先做 feature-only、part-only、feature+part 以及 motion/rotation 的逐项消融，并记录各正则项的实际量级。

## 37. DynOMo C 相对 original 的代码改动与未提升原因（2026-08-28）

### 相对 original 实际改动

1. **实验入口和配置**
   - `arguments/__init__.py` 增加 `use_dynomo_c`、affinity 维度、KNN 数量、part same/adjacent/other 权重、motion/rotation 权重。
   - `scripts/exps_dnarendering.sh` 增加 `dynomo_c` mode，并显式传入 `densify_until_iter=1800` 和上述参数。
   - DynOMo C 禁止和 `part_moe/tri/time/part_budget` 等其他分支组合，保证它是 original-based ablation。

2. **NonrigidDeformer 增加 affinity head**
   - 在原有 shared MLP 的中间特征 `h` 后增加 `Linear(512,256)-ReLU-Linear(256,32)`，再做 L2 normalize，得到每个 Gaussian 的 `dynomo_affinity`。
   - 这个特征只用于当前训练迭代的 loss 权重，不进入 RGB forward，也不改变 `d_xyz/d_rotation/d_scaling` 的生成结构。
   - affinity head 的参数和原 deformer 一起进入 `mlp_optimizer`，因此修复后可以收到梯度。

3. **训练 loss 分支**
   - original 使用等权 `full_aiap_loss`；DynOMo C 使用 `weighted_aiap_loss`。
   - 仍然在 canonical Gaussian 的空间 KNN 上计算局部等距误差，使用 affinity cosine、SMPL part 权重和 part confidence 对边加权。
   - 新增 `d_xyz` 邻域一致性和 `d_rotation` 邻域一致性；RGB、mask、SSIM、LPIPS 主损失保持不变。
   - covariance AIAP 仍然是原来的等权 `aiap_loss`，没有使用 affinity/part 权重。

4. **part 先验和动态刷新**
   - 通过 Gaussian 到 canonical SMPL-X 顶点的最近邻生成 `anatomy5` 伪 part label；距离超过 `0.08` 的点标为 unknown。
   - part confidence 由 `1 - distance / 0.08` 得到；Gaussian 数量发生变化时重新生成标签。
   - `anatomy5` 只有 `unknown/body/left_hand/right_hand/face` 五类，邻域矩阵使用 same=`1.0`、adjacent=`0.1`、other=`0.0`。
   - `gaussian_renderer/__init__.py` 只有在 `use_part_moe=True` 时才把 part label 传入 NonrigidDeformer；DynOMo C 明确使用 `use_part_moe=False`，因此 part 先验只参与 `train.py` 的正则权重，不参与形变网络的前向条件。

### 为什么没有改善

1. **改动主要是正则项重加权，不是主干表达能力提升。**
   - original 已经有 canonical KNN 的 AIAP position/covariance 约束。
   - DynOMo C 仍使用同一组空间 KNN，只是改变 position 边的权重并增加两个小权重一致性项，因此新增信息量有限，不能自动产生更准确的邻居。
   - 尤其是 part guide 没有进入 NonrigidDeformer 的输入，只能通过额外 loss 间接调整已有网络，实际改造强度比“part-conditioned deformation”弱。

2. **当前可学习 affinity 存在目标退化。**
   - `weighted_aiap_loss` 对权重做归一化平均；对某条边的权重求导，其方向近似由 `该边误差 - 当前加权平均误差` 决定。
   - 因此优化器会倾向于给高误差、难拟合的邻居更低权重，给低误差邻居更高权重。这是在“逃避困难约束”，不等于学到了真实运动相似性。
   - `affinity_floor=0.1` 只能防止完全关闭边，不能解决这个目标本身；代码中也没有 affinity 的对比学习、标签监督或跨帧一致性监督。
   - affinity 来自带 pose/sequence 条件的当前 `h`，每个训练迭代只服务当前姿态，未建立跨姿态的稳定邻域记忆。

3. **SMPL 最近邻 part 先验对衣服和边界不可靠。**
   - 该先验没有区分身体表面和衣服层；靠近身体的衣服点可能获得高 confidence 的错误 body label，远离 SMPL 的衣服点则进入 unknown。
   - 本次最终标签中 unknown 占比：`0044_11=7.406779661016949%`、`0051_09=11.490993272082761%`、`0206_04=9.748599897531934%`、`0813_05=0.3748924488876142%`。
   - `other=0.0` 会删除跨 part 的有效空间邻居；`adjacent=0.1` 又显著削弱边界邻居。对衣服褶皱、袖口和身体-衣服接触区域，这可能比 original 的等权约束更差。

4. **新增 motion/rotation 项可能与真实非刚性运动冲突。**
   - `loss_motion` 直接比较邻点 `d_xyz` 的逐分量 L1 差，会压平衣服褶皱、裙摆和袖口的合法局部差异。
   - `loss_rotation` 比较 raw `d_rotation` 向量的逐分量差，不是旋转矩阵或四元数的相对 geodesic 距离，存在参数化依赖。
   - 两项权重都只有 `0.01`，而原有 covariance 项的权重是 `ioscov_w=100.0`；由于当前没有记录各项实际数值，不能证明新增项足以改变训练方向，也不能排除局部过平滑。

5. **part/affinity 并没有作用到 covariance consistency。**
   - 位置和 motion/rotation 项使用了权重，但 covariance 仍由等权 `aiap_loss` 计算。
   - Gaussian 的位置、尺度和旋转是耦合的，只改 position 邻域而不一致地约束 covariance，可能让新增约束互相抵消。

6. **1800 窗口公平，但最终点数仍不同。**
   - 本次与 original 的 densify 窗口、训练步数和 baseline 参数一致，但正则改变了梯度和删点结果。
   - 最终 Gaussian 数量 DynOMo/original 分别为：`0044_11=82600/83909`、`0051_09=69115/68103`、`0206_04=56603/58936`、`0813_05=48814/50740`。
   - 所以这是“densify 配置公平”，不是“最终表示容量完全相同”；三个序列的 DynOMo 点数更少，会增加其达到 original 质量的难度。

7. **当前实验缺少能验证机制是否真正工作的诊断量。**
   - 日志没有 affinity 均值/方差、有效边比例、part gate 后边比例、四个正则项的实际 loss 和梯度范数。
   - 因此目前只能确认代码路径执行且最终指标略降，不能判断主要失败点是 affinity 退化、part 误 gate、motion 过平滑还是权重太弱。

### 结论与下一步

- 当前 DynOMo C 的事实贡献是：把 feature/part-guided local consistency 接入 SeqAvatar 的训练 loss，并完成了可运行、参数生效、窗口一致的实现。
- 当前结果不能支持“DynOMo C 提升 original”；四序列平均 Delta 为 PSNR `-0.031766049067179836`、SSIM `-0.00010282297929126161`、LPIPS*1000 `+0.13169196899980287`。
- 最优先的下一步是固定 affinity 特征后做 `feature-only`、`part-only`、`feature+part` 消融，并记录正则量级；不要直接增大 motion/rotation 权重。
- affinity 更合理的训练方式应使用 stop-gradient 特征加独立的 affinity/temporal supervision，或先用固定 `h.detach()` 验证邻居权重本身是否有效；part 侧应增加 cloth-aware label 或软 gate，并将 covariance 一并改成一致的加权约束。

## 38. DynOMo 方向是否值得继续（2026-08-28）

### 判断

- **方向本身有希望，当前 DynOMo C 组合没有足够证据。** DynOMo 的核心价值是让局部约束从“空间相邻”变成“运动相容的相邻”，这与 SeqAvatar 的非刚性形变问题是匹配的。
- 但当前 C 把 affinity、SMPL part gate、motion consistency、rotation consistency 一次性叠加，四序列平均反而下降，无法判断究竟是哪一项有效或有害。
- `0051_09` 三项同时改善说明该机制不是对所有序列都失效，但单个序列的弱正向不足以支持继续做大规模组合实验。

### 应保留和应放弃的部分

- 保留：**feature-guided local neighborhood / local isometry** 这个主线。
- 暂时放弃：当前无监督可学习 affinity 直接参与归一化权重、`other=0.0` 的硬 part gate、未经几何归一化的 raw rotation L1、和一开始就加入 motion/rotation 的版本 C。
- 当前版本应作为“实现验证失败、研究假设尚未证伪”的中间版本，不应作为最终方法结果。

### 最低成本验证方案

1. 先只跑 `feature-only`：使用 `h.detach()` 计算 affinity，关闭 part、motion、rotation；记录 affinity 分布和 weighted AIAP 相对 original 的变化。
2. 再跑 `part-only`：只使用 soft part confidence，不使用 `other=0` 的硬删除，验证 SMPL 先验本身是否有益。
3. 只有当其中一个单项在至少两个序列上稳定改善，再组合 `feature+part`；motion/rotation 放到最后单独加入。
4. 每次实验必须记录 `loss_aiap_xyz`、`loss_aiap_cov`、`loss_motion`、`loss_rotation`、有效邻边比例、平均 affinity 权重和各模块梯度范数。

### 停止标准

- 如果 `feature-only` 和 `part-only` 在至少两个代表性序列上都不能同时改善 PSNR/SSIM/LPIPS 中的主要趋势，就应停止 DynOMo 线，转回 point 主线。
- 如果 feature-only 有收益，再只围绕 affinity 的监督、温度和跨帧一致性优化；不要继续增加更多正则项。

### 最终结论

DynOMo 还有一次小规模、可证伪的验证机会；值得继续的是“运动特征引导邻域”这一点，不值得继续的是当前 C 的整体配方。下一步目标不是再跑四序列长实验，而是用最少的 feature-only/part-only 对照定位是否存在真实有效信号。

## 39. VGGT 衣物候选点新实验准备（2026-08-28）

当前新方向改为：

```text
保留 SMPL-X A-pose canonical Gaussian 初始化
-> 多视角当前姿态图像离线运行 VGGT
-> 前景、多视角一致性、SMPL 外侧距离和边界筛选衣物候选
-> 将当前姿态 VGGT 点的有限局部偏移转为 canonical 候选点
-> 用低贡献 Gaussian 等量替换，保持固定预算
-> 继续使用 SeqAvatar 原始 RGB / Mask / SSIM / LPIPS / AIAP loss
```

### 已完成

- 已确认 DNA-Rendering 每个序列有 30 个相机、多视角图像、mask、`RT/K` 和 SMPL-X `obs_xyz`。
- 已下载 VGGT 官方源码到本地 `third_party/vggt/`，但该目录已加入 `.gitignore`，不应提交第三方源码。
- 在 `scripts/vggt_generate_clothing_candidates.py` 增加离线候选生成器：支持关键帧、多视角、VGGT confidence、前景过滤、SMPL 外侧距离、多视角邻域一致性、边界加权、Sim(3) 对齐和 canonical 偏移转移。
- 在 `ablations/vggt_garment_controller.py` 增加独立的显式候选点加入与低贡献点等量替换接口；`scene/gaussian_model.py` 增加 `spawn_from_explicit_candidates()`。
- `scripts/exps_dnarendering.sh` 增加 `vggt_garment` 模式；默认不影响 original、point、part_point、DynOMo 等已有模式，并在参数提取后禁止与其他消融混用。
- `seqavatar` 环境已补充 `huggingface_hub`、`einops`、`safetensors`；Torch/Torchvision 仍保持原来的 `2.1.2/0.16.2`。

### 当前阻塞

- 官方权重 `facebook/VGGT-1B` 尚未下载成功。服务器访问 `huggingface.co` 建立连接超时，smoke test 在请求 `config.json` 阶段中止，没有生成候选缓存，也没有启动训练。
- 权重需要在能访问 Hugging Face 的机器下载后拷贝，或配置可用镜像；可用本地模型目录替代 `--model-name facebook/VGGT-1B`。
- 官方 requirements 指向 Torch `2.3.1/Torchvision 0.18.1`，当前 SeqAvatar 是 `2.1.2/0.16.2`。源码导入和图像预处理已通过，但正式推理前仍需单独确认兼容性；不应为此升级现有训练环境。

### 重要实现边界

- VGGT 只作为候选提议器，不作为 canonical 真值，也没有加入 VGGT 强几何 loss。
- 当前代码仍需在权重到位后验证：Sim(3) 对齐尺度、canonical 候选范围、候选与已有 Gaussian 的去重、候选加入后的固定预算和点数日志。
- 当前没有声称实验已跑通或指标已产生；在权重和 smoke test 通过前，不启动四/六序列长训练。

## 52. VGGT strict + high-error 六序列总结与下一步（2026-08-30）

本节汇总六个 DNA 序列上已经完成的 `vggt_garment_strict_high_error` 结果。六个序列统一使用：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
VGGT replacement window=800-1500
strict replacement budget=true
high-error candidate filtering/protection=true
```

最终 novel-view 指标（120 views，LPIPS 已乘以 1000，未四舍五入）：

| 实验 | 序列 | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| original | 0007_04 | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 |
| strict + high-error | 0007_04 | 29.536748838424682 | 0.9585092023015022 | 44.61549075009922 |
| original | 0019_10 | 35.257725365956624 | 0.9808832183480263 | 21.035196678712964 |
| strict + high-error | 0019_10 | 35.27290960947673 | 0.9809831599394481 | 20.927772112190723 |
| original | 0044_11 | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 |
| strict + high-error | 0044_11 | 32.95231997172038 | 0.9779256527622541 | 21.25621880404651 |
| original | 0051_09 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 |
| strict + high-error | 0051_09 | 28.690577745437622 | 0.9715634857614835 | 31.05802504966656 |
| original | 0206_04 | 31.31386383374532 | 0.9695543631911278 | 33.811808843165636 |
| strict + high-error | 0206_04 | 31.395742559432982 | 0.969868399699529 | 33.67418895165126 |
| original | 0813_05 | 36.10658038457235 | 0.9869976962606112 | 18.244266610903047 |
| strict + high-error | 0813_05 | 36.055868784586586 | 0.9869326864679654 | 18.269647215493023 |

六序列均值：

| 实验 | 序列 | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| original | Mean | 32.31045727729797 | 0.974225685828262 | 28.405228959551707 |
| strict + high-error | Mean | 32.31736125151316 | 0.9742970978220303 | 28.300223813857883 |

相对 original 的均值差值（实验减 original）：

```text
Delta PSNR       +0.006903974215191511
Delta SSIM       +7.141199376847762e-05
Delta LPIPS*1000 -0.10500514569382347
```

逐指标统计：

```text
PSNR 改善：4/6（0007_04、0019_10、0051_09、0206_04）
SSIM 改善：4/6（0007_04、0019_10、0051_09、0206_04）
LPIPS 改善：4/6（0007_04、0019_10、0044_11、0206_04）
三项同时改善：3/6（0007_04、0019_10、0206_04）
```

### 结果分析

1. **整体有弱正向信号，但幅度很小。** 六序列平均 PSNR 只增加 `0.006903974215191511`，SSIM 只增加 `0.0000714119937683666`，LPIPS*1000 只降低 `0.10500514569382347`。这可以说明当前接入没有整体破坏 original，但还不能作为稳定提升。

2. **序列依赖明显。** `0206_04` 的三项提升最大，是当前最支持 high-error 策略的序列；`0007_04`、`0019_10` 也三项改善。`0044_11` 只有 LPIPS 改善，`0051_09` 的 LPIPS 略退化，`0813_05` 三项均退化。

3. **预算公平性已经成立，但训练容量过程仍需更严格控制。** 每次替换事件满足新增数等于删除数，最终 PLY 也严格对齐 original；不过部分序列在训练结束前仍需要 `padded` 才回到目标点数。更干净的做法是在 densification 结束时就把点数校正到目标，并让后续 1800-25000 步始终保持该点数，避免最后一步才加入未充分训练的新点。

4. **当前结果不能单独证明 high-error 比 strict-only 更好。** `0007_04` 和 `0206_04` 已有 strict-only 对照，但另外四个序列目前只有 strict + high-error 结果。因此六序列可以和 original 做比较，却还不能完成 high-error 的六序列归因。

5. **和 part 线相比，不能直接排名。** 六序列 `part_moe_leg` 和 `part_point` 的均值更高，但它们的网络结构、point 窗口和部分 densify 设置不同；当前 VGGT 结果只能说明在 original-based、1800 窗口下有弱收益，不能据此断言 VGGT 已超过 part 方法。

### 下一步修改顺序

**第一步：先补齐严格消融矩阵，不立即叠加新机制。**

- 对 `0019_10`、`0044_11`、`0051_09`、`0813_05` 补跑 `vggt_garment_strict`，配置完全复制当前 high-error 实验，只关闭 high-error 筛选和保护。
- 这样六个序列都同时拥有 strict-only 与 strict + high-error，才能判断 high-error 本身是否贡献了提升。
- 先在 `0206_04` 和 `0813_05` 作为正负代表检查，再决定是否扩大修改。

**第二步：改进 high-error 统计方式。**

- 当前误差来自单次训练视角的 `abs(render-GT)`，容易受相机、遮挡和随机采样影响。
- 改为按 Gaussian/候选父点累计多视角、多帧的误差 EMA，并按 visibility 归一化。
- 只有在连续多个采样窗口中稳定高误差，才提升候选优先级或加入保护；避免一次异常视角决定增删点。
- 保留 boundary 与 high-error 两个独立分数，分别记录命中率，避免 high-error 分数吞掉边界先验。

**第三步：把固定预算从“末步校正”改成“窗口结束即对齐”。**

- 在 iter=1800 的最后一次结构变化后立即校正到 original 目标点数。
- 1800 之后禁止任何点数变化，保证所有方法在主要训练阶段拥有相同 Gaussian 容量。
- padding 不应使用未经训练的弱点；最好通过等量替换或从已有 Gaussian 复制属性完成预算对齐，并记录 `points_at_1800` 与 `points_at_25000`。

**第四步：提高候选点的可学习性，但单独做消融。**

- 候选初始 `opacity=0.04`、scale ratio=`0.65` 可能使正确 VGGT 候选在早期渲染贡献过弱。
- 先只测试父点属性复制、opacity warm-up 或稍大的初始 opacity，不能和 high-error 统计改动同时进行。
- 继续让 RGB/Mask/SSIM/LPIPS/AIAP 决定候选是否有效，不加入强 VGGT 几何监督。

**第五步：再考虑删除策略。**

- 目前应优先记录 low-opacity、low-gradient、low-visibility 点的删除分布，以及被保护点在最终图像误差中的贡献。
- 加入空间均衡约束，避免 512 个删除点集中在局部区域造成孔洞。
- high-error 保护不能无限扩大，否则会把真正低贡献点留在模型中，降低预算迁移效率。

### 阶段性结论

当前最合理的结论是：

```text
VGGT corrected candidate + strict budget + high-error protection 已经跑通；
六序列相对 original 有弱平均改善，但序列依赖明显；
当前首要问题不是继续增加 VGGT 点，而是补齐 strict-only 对照、稳定多视角误差统计、
并在 densification 结束时而不是训练末步完成点数对齐。
```

## 53. strict + high-error 最终点云位置

六个序列的最终 25000-step 点云均位于对应输出目录的
`point_cloud/iteration_25000/point_cloud.ply`：

```text
0007_04: output/DNA-Rendering/0007_04/vggt_garment_strict_high_error/20260830_vggt_higherr_0007/point_cloud/iteration_25000/point_cloud.ply
0019_10: output/DNA-Rendering/0019_10/vggt_garment_strict_high_error/20260830_vggt_higherr_0019_0044/point_cloud/iteration_25000/point_cloud.ply
0044_11: output/DNA-Rendering/0044_11/vggt_garment_strict_high_error/20260830_vggt_higherr_0019_0044/point_cloud/iteration_25000/point_cloud.ply
0051_09: output/DNA-Rendering/0051_09/vggt_garment_strict_high_error/20260830_vggt_higherr_0051_0813/point_cloud/iteration_25000/point_cloud.ply
0206_04: output/DNA-Rendering/0206_04/vggt_garment_strict_high_error/20260830_vggt_higherr_0206b/point_cloud/iteration_25000/point_cloud.ply
0813_05: output/DNA-Rendering/0813_05/vggt_garment_strict_high_error/20260830_vggt_higherr_0051_0813/point_cloud/iteration_25000/point_cloud.ply
```
- 正式公平配置应固定训练步数、densify 窗口和同一组 original 实验的最终 Gaussian 目标数；目标数可通过 `vggt_target_points_file` 显式提供。当前脚本不再默认使用上一轮 `densify_until_iter=1800` 的旧点数文件。

### 运行顺序

```text
1. 准备 VGGT-1B 本地权重或 Hugging Face 访问
2. 单序列、单关键帧、8 视角生成 candidates.npz
3. 检查候选数量、坐标范围、Sim(3) 和可视化投影
4. 用 original final point count 做 fixed-budget 小规模训练
5. 再扩展到四序列或六序列，并按约定报告 PSNR、SSIM、LPIPS*1000，不四舍五入
```

### Smoke test 更新

- 已通过 `HF_ENDPOINT=https://hf-mirror.com` 下载并缓存 `facebook/VGGT-1B` 权重。
- 0044_11 的 frame 0、8 个视角 smoke test 已完成：初筛 626 个候选，voxel 去重/排序后保存 32 个候选。
- 缓存字段包含 `canonical_xyz`、`reference_xyz`、`confidence`、`score`、来源视角、最近 SMPL-X 顶点、frame id 和初始化颜色。
- canonical 候选坐标范围约为 `[-0.7403, 0.3528]`，需要正式实验前继续做投影可视化和与 SeqAvatar Gaussian 分布的范围检查。
- smoke test 只验证了候选预处理，没有启动 SeqAvatar 长训练，也没有产生评价指标。

### 六序列候选缓存状态

已完成 5 个关键帧、8 个视角的正式候选缓存：

```text
0007_04: 1943
0019_10: 1837
0044_11: 1566
0051_09: 1336
0206_04: 1480
0813_05: 1818
```

候选文件位于 `output/vggt_garment_candidates/<sequence>/candidates.npz`，属于输出数据，不提交 GitHub。六个缓存的 canonical 坐标范围约为 `[-1.0102, 0.7763]`，正式训练前的静态检查已通过；目前尚未运行 `vggt_garment` 的 SeqAvatar 训练，因此没有评价指标。

## 40. 后续实验默认设置与六序列候选缓存（2026-08-29）

后续实验默认统一使用：

```bash
DENSIFY_UNTIL_ITER=1800
```

已同步修改：
- `scripts/exps_dnarendering.sh` 的 DNA 实验默认值为 `1800`
- `scripts/exps_zjumocap.sh` 的普通实验默认值为 `1800`
- `arguments/__init__.py` 的通用训练参数默认值为 `1800`
- 仍可通过显式环境变量覆盖，但正式对比实验必须在日志中确认实际值

说明：ZJU 的 Part-MoE 模式仍受其 `part_moe_start_iter=1000` 的专门约束，会使用该模式自己的 densify 逻辑；这不改变 DNA 六序列主线默认的 `1800`。

六个 DNA 序列的候选缓存不是六份训练真值，也不是六个序列共享的点云。它们是按序列独立生成的 VGGT 离线候选池：

```text
output/vggt_garment_candidates/<sequence>/candidates.npz
```

当前数量：

```text
0007_04: 1943
0019_10: 1837
0044_11: 1566
0051_09: 1336
0206_04: 1480
0813_05: 1818
```

缓存的用途：
1. 保存经过 VGGT 置信度、前景、多视角一致性、SMPL 外侧距离和边界筛选的衣物候选点。
2. 保存把当前姿态观测偏移近似转换到 SeqAvatar canonical space 后的 `canonical_xyz`。
3. 训练时只加载当前序列的缓存，按 score 提议新增 Gaussian；新增前等量删除低贡献 Gaussian，保持事件级固定预算。
4. 候选加入后不使用 VGGT 强监督，仍由 SeqAvatar 原始 RGB、Mask、SSIM、LPIPS 和 AIAP 损失决定这些点是否有效；后续再依据 opacity、gradient、visibility 等贡献淘汰无效点。

因此，候选缓存解决的是“衣物点从哪里开始初始化”的问题，不决定最终点是否保留，也不会把 VGGT 噪声直接当作 GT。缓存不能跨序列直接复用，因为每个 DNA 序列的相机、姿态、可见衣物区域和 canonical 偏移都不同。

## 41. 基线、论文与后续公平配置审计（2026-08-29）

结论：当前可以保证“同一仓库脚本下的 original-based 对比共享统一基础参数”，但不能声称当前配置与论文原始实验完全一致；`DENSIFY_UNTIL_ITER=1800` 是后续统一的实验协议，不是论文明确给出的复现值。

### 已对齐的 DNA 基础设置

- 6 个序列：`0007_04`、`0019_10`、`0044_11`、`0051_09`、`0206_04`、`0813_05`
- 训练视角：24 个（camera ID 0-47，步长 2）
- novel-view：6 个视角（camera ID 48-59，步长 2）
- 训练时间帧：100；novel-view 评估为 20 个时间帧，每 5 帧采样
- SMPL-X、neutral、序列长度 `8`
- DNA 脚本的序列条件：`seq_xyz_knn=8`、`time_step_num=3`、`max_time_step=3`、`minimal_time_step=1`
- MLP：3 层、宽度 512
- 颜色/SSIM/LPIPS 权重：`1.0/0.01/0.01`
- 评估：novel-view，最终训练步保存并计算指标
- 后续默认：`DENSIFY_UNTIL_ITER=1800`

### 尚不能称为论文完全复现的差异

1. 论文 PDF 的补充材料第 11 节 `Details of Optimization`（该 PDF 第 14 页）写的是 “We optimize 30k iterations”；但论文配套 GitHub DNA 脚本实际设置 `iter=25000`，当前脚本与官方 GitHub 脚本一致。
2. 论文说明使用 adaptive densification，但没有明确给出 `densify_until_iter=1800`；1800 是当前统一公平协议。
3. 论文补充材料写的 motion sampling scales 为 `{1, 3, 5}`，当前代码通过 `minimal_time_step=1` 到 `max_time_step=3` 生成 `{1, 2, 3}`，这一点需要以后单独确认是否是代码实现定义不同，不能直接宣称完全一致。
4. 当前代码的 mask loss 在总损失中固定乘 `0.1`；论文公式只写出 `Lmask`，没有在正文明确对应的数值系数。

### original 与 VGGT 的公平边界

- `original` 不读取 VGGT 缓存，也不启用 `use_vggt_garment`。
- `vggt_garment` 继承相同的 DNA 基础参数，只额外读取当前序列的候选缓存并做候选替换，这是方法变量。
- 当前候选控制器在每次加入候选前等量删除低贡献点，保证事件级 fixed budget。
- 若要保证最终 Gaussian 数量也完全一致，必须提供 original 在 `densify_until_iter=1800`、`iteration=25000` 的目标点数文件。已有六序列 original 目标数为：

```text
0007_04: 35540
0019_10: 42988
0044_11: 83909
0051_09: 68103
0206_04: 58936
0813_05: 50740
```

当前 `VGGT_TARGET_POINTS_FILE` 默认为空，所以正式 VGGT 对比前还不能保证最终表示容量完全相同；必须显式传入包含上述六个序列目标数的 JSON 文件，并在日志确认实际加载。

## 42. 与论文官方原始仓库的基线保护审计（2026-08-29）

参考仓库固定为 `/media/coding/ckx/SeqAvatar`，只读使用，未做任何修改；其 Git 状态为 `main...origin/main` 且工作区干净。

对比结果：
- 官方 DNA 脚本与当前 DNA 脚本的核心训练配置一致：`25000` iterations、`densify_until_iter=1800`、24/6 视角、100/20 时间帧、SMPL-X neutral、`seq_len=8`、`seq_xyz_knn=8`、`time_step_num=3`、`max_time_step=3`、`minimal_time_step=1`、loss 权重 `1.0/0.01/0.01`。
- 当前 `original` 模式的 `part_moe`、DynOMo、VGGT、point、time、tri 等新增开关均为关闭；训练仍走原始 `full_aiap_loss` 和原始 `densify_and_prune` 路径。
- 当前渲染函数增加了可选参数和 depth 返回，但 RGB、alpha、Gaussian covariance 和原始输入下的渲染分支保持一致。
- `scene/dataset_readers.py` 对 DNA 仅把条件字典构造提取成等价 helper；新增文件存在性检查只影响其他数据集分支。
- 参考库与当前库的 baseline `NonrigidDeformer` 在相同参数和随机种子下：state key、shape、初始化权重完全一致；随机输入前向输出最大绝对差为 `0.0`。
- 当前脚本的 `original` 中间评估/保存时点已调整为 `3000/15000/25000`，与官方 `train.py` 默认 schedule 对齐；最终训练和 novel-view 评估仍为 `25000`。

因此目前可以作出较强保证：**当前仓库没有把新增消融逻辑无条件注入 original 的模型前向、损失或增密路径，DNA original 与官方仓库的核心 baseline 设置一致。**

仍需区分两件事：
- `original` 与官方仓库一致，不代表 `vggt_garment` 与 original 完全相同；VGGT 候选替换就是该消融的唯一方法变量。
- VGGT 正式实验还必须显式提供六序列 original 的目标点数文件，才能把最终 Gaussian 容量也固定为同一预算；否则只能保证候选替换事件等量。

## 43. VGGT garment C 实验运行记录（2026-08-29）

本次实验按用户指定配置运行：

```text
MODE=vggt_garment
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
GPU 2: 0044_11, 0051_09
GPU 3: 0206_04, 0813_05
RUN_TIME=20260829_vggt_garment_1800
```

实际启动时显式传入：

```text
VGGT_TARGET_POINTS_FILE=note/vggt_target_points_1800_25000.json
```

这样每个序列使用自己的候选缓存和 original 目标点数；候选加入窗口仍为 `800-1500`，间隔 100，事件级等量删点后再加候选，训练损失保持 original。

已完成。最终指标统一读取各序列的
`metrics/results_novelview_25000.json`，不是训练/渲染日志中的在线均值：

| Sequence | VGGT PSNR | VGGT SSIM | VGGT LPIPS*1000 | Original PSNR | Original SSIM | Original LPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| 0044_11 | 32.95826457341512 | 0.9779684290289878 | 21.342927667622764 | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 |
| 0051_09 | 28.67866628964742 | 0.97150482883056 | 30.87858803725491 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 |
| 0206_04 | 31.346332931518553 | 0.9695774306853612 | 33.67907018400729 | 31.31386383374532 | 0.9695543631911278 | 33.811808843165636 |
| 0813_05 | 36.07834423383077 | 0.9869667783379554 | 18.178060899178185 | 36.10658038457235 | 0.9869976962606112 | 18.244266610903047 |
| Mean | 32.26540200710296 | 0.976504366720716 | 26.01966169701579 | 32.26842133204142 | 0.9765083922694127 | 26.09632267655494 |

相对 original 的均值差值（VGGT - original）为：

```text
PSNR:      -0.003019324938455803
SSIM:      -0.000004025548696551251
LPIPS*1000: -0.07666097953915561  # 越低越好
```

逐序列结论：`0051_09` 和 `0206_04` 三项指标均改善；`0044_11` 三项均略降；`0813_05` 的 LPIPS 改善，但 PSNR/SSIM 略降。因此这次四序列平均结果不能判定为整体提升：LPIPS 有小幅改善，PSNR/SSIM 基本持平且略低，说明 VGGT 候选提议在部分衣物/高误差区域有效，但当前候选筛选、canonical 偏移转移或固定预算替换仍不稳定。

注意：`LPIPS*1000` 按要求报告且不四舍五入；原始 JSON 中 LPIPS 没有乘 1000。四个 VGGT 实验及最终渲染均已结束。

公平性说明：四个运行在训练步数、`densify_until_iter`、视角/帧采样、损失和基础模型配置上与对应 original 对齐，且启动时显式传入了目标点数文件；但候选替换与运行时删点/增点的交互仍可能造成最终 Gaussian 数量存在少量差异。因此本次结论是“统一训练协议下的 VGGT 候选提议消融”，若要做严格的最终容量控制，还需在训练结束时再次核验并强制统一点数。

## 44. VGGT 候选点云缓存与可视化（2026-08-29）

候选缓存保存在：

```text
/media/coding/ckx/human/SeqAvatar/output/vggt_garment_candidates/<sequence>/candidates.npz
```

六个 DNA 序列都有缓存，每个缓存包含：

- `reference_xyz`：VGGT 观测点经过对齐后的 world-space 坐标；
- `canonical_xyz`：经过 SMPL/局部偏移转移后，实际提供给 SeqAvatar 新 Gaussian 的 canonical 坐标；
- `confidence`、`score`：VGGT 置信度和候选筛选分数；
- `colors`：候选点初始化时使用的 RGB；
- `source_view`、`frame_id`、`nearest_smpl_vertex`：来源视角、帧和最近 SMPL 顶点。

为便于直接查看，已在每个序列目录导出：

```text
vggt_reference.ply  # 对齐后的 VGGT 姿态观测点
vggt_canonical.ply # 实际新增 Gaussian 使用的 canonical 候选点
```

例如：

```text
/media/coding/ckx/human/SeqAvatar/output/vggt_garment_candidates/0044_11/vggt_reference.ply
/media/coding/ckx/human/SeqAvatar/output/vggt_garment_candidates/0044_11/vggt_canonical.ply
```

可以用 MeshLab、CloudCompare 或 Open3D 打开 `.ply`。当前导出的点按候选 RGB 着色，并保留 `score` 和 `confidence` 属性；`candidates.npz` 才是训练实际读取的原始缓存，`.ply` 只是可视化副本。

## 45. VGGT 四序列结果分析（2026-08-29）

这次结果说明 VGGT **确实被调用并产生了有限影响，但当前影响很弱，不能证明方法整体有效**：

- `0051_09`、`0206_04` 三项指标均改善；
- `0044_11` 三项指标均下降；
- `0813_05` 只有 LPIPS 改善，PSNR/SSIM 下降；
- 四序列平均 PSNR `-0.003019324938455803`、SSIM `-0.000004025548696551251`、LPIPS*1000 `-0.07666097953915561`。

更准确的判断是：**VGGT 信号不是完全无效，但候选点的利用方式和 canonical 转换限制了它的作用；同时新增候选的初始影响也偏弱。**

代码层面的证据：

1. `scripts/vggt_generate_clothing_candidates.py` 确实使用 VGGT 的 `world_points`、`world_points_conf` 和预测相机位姿，并结合前景 mask、跨视角近邻一致性、SMPL 表面距离和边界分数筛选候选。
2. 训练阶段 `VggtGarmentController` 只在 `800-1500` step 用候选点替换低贡献 Gaussian；没有加入 VGGT 几何 loss，最终仍由 original 的 RGB/mask/SSIM/LPIPS/AIAP 损失决定点是否有效。
3. 新点使用固定 opacity `0.04` 和父点 scale 的 `0.65`，初始渲染贡献较小，需要后续损失把它们学习起来；这会削弱 VGGT 的即时作用。
4. `canonicalize_points()` 中计算了 `canonical_rot`，但实际没有使用；当前 `offset_canonical` 使用的是 `target_rot`。由于该代码负责把姿态空间 VGGT 偏移转回 canonical 空间，这属于优先核验的实现问题，可能导致候选点位置偏差。
5. 当前候选是先做置信度百分位、SMPL 距离阈值、体素去重和 top-score 截断，候选点数量只有约 1336-1943；VGGT 的完整点云信息在进入 SeqAvatar 前已经被强烈压缩。

因此目前不能简单归因于“VGGT 作用太小”。优先级应为：先修正/验证 canonical 逆变换并检查候选点在姿态图像上的投影，再做 `reference-only`、`canonical-only`、无候选替换和不同 opacity/注入窗口的对照。只有在坐标转换正确且候选点确实落在衣物高误差区域后，才能判断 VGGT 本身的信息量是否不足。

## 46. VGGT 问题排查：第 1-2 步（2026-08-29）

### 第 1 步：canonical 逆变换

已修正 `scripts/vggt_generate_clothing_candidates.py` 的姿态空间到 canonical 空间偏移转换。修正基于 SeqAvatar `coarse_deform_c2source()` 的实际刚性链：

```text
offset_canonical = offset_world
                    @ global_R
                    @ inv(target_rot).T
                    @ canonical_rot.T
```

对真实 `0044_11` SMPL 变换做 round-trip：

```text
修正公式最大绝对误差: 2.384185791015625e-07
旧公式最大绝对误差:   1.2513480186462402
```

因此旧公式确实存在明显的姿态偏移逆变换误差，而不是单纯理论上的代码风格问题。已生成新缓存：
`output/vggt_garment_candidates_corrected/<sequence>/candidates.npz`；旧的 `output/vggt_garment_candidates/<sequence>/candidates.npz` 未覆盖。

### 第 2 步：候选点投影审计

审计工具为 `scripts/audit_vggt_candidates.py`，结果保存在
`output/vggt_candidate_audit_legacy/<sequence>/stats.json`，叠加图在各序列的 `overlays/` 下。

| Sequence | Projection valid | Foreground | Boundary 3px | High-error* | Outside SMPL threshold |
|---|---:|---:|---:|---:|---:|
| 0044_11 | 1.0 | 0.9961685823754789 | 0.009578544061302681 | 0.16338582677165353 | 1.0 |
| 0051_09 | 1.0 | 0.9169161676646707 | 0.08233532934131736 | 0.07112970711297072 | 0.999251497005988 |
| 0206_04 | 1.0 | 0.9743243243243244 | 0.06486486486486487 | 0.0871559633027523 | 0.9993243243243243 |
| 0813_05 | 1.0 | 0.9317931793179318 | 0.035753575357535754 | 0.13615023474178403 | 0.9988998899889989 |

`High-error*` 只在已有 baseline novel-view render 的候选上统计，阈值是对应图像误差的 top 20%，可评估候选数分别为 `508/239/218/213`。前景和 SMPL 外侧条件基本满足，但边界命中不足，高误差命中也不高，说明当前候选筛选并没有稳定地找到最值得补点的区域；同时“SMPL 外侧”不能等价于“衣物”。

第 1-2 步的中间结论：**旧实验结果偏弱至少部分来自 canonical 转换错误和候选区域与重建误差错位；VGGT 本身是否有增益，还需要固定预算随机对照验证。**

## 47. corrected VGGT 队列状态（2026-08-29）

当前主实验配置固定为：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
```

使用修正后的候选缓存：

```text
output/vggt_garment_candidates_corrected/{sequence}/candidates.npz
```

四序列 corrected 主实验由 GPU 2/3 串行运行：

```text
GPU 2: 0044_11 -> 0051_09
GPU 3: 0206_04 -> 0813_05
```

截至本次记录：

- 四序列 `0044_11`、`0051_09`、`0206_04`、`0813_05` 的训练最终评估文件已齐全；
- `0813_05` 的独立 `render.py` 曾因 GPU 3 被其他进程占用而在 LPIPS 阶段 OOM，但训练末次评估已保存，后续需在空闲显存上补完整渲染；
- 等待队列已正确放行另外两个序列：
  - GPU 2：`0007_04`，输出 `vggt_garment/20260829_vggt_corrected_1800_extra/`；
  - GPU 3：`0019_10`，输出 `vggt_garment/20260829_vggt_corrected_1800_extra/`；
- 两个新增序列已进入训练，配置仍为 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`，启动阶段未发现异常。

参考基线仓库 `/media/coding/ckx/SeqAvatar` 未修改。

## 48. corrected VGGT 六序列完成结果（2026-08-30）

统一配置：`ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`，指标均为 120 个 novel-view 的最终结果；LPIPS 已乘以 `1000`。

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0007_04 | 29.511724376678465 | 0.958197579284509 | 44.77907596156001 |
| 0019_10 | 35.25521783828735 | 0.9808408752083778 | 21.131248337527115 |
| 0044_11 | 32.97896426518758 | 0.9781139100591342 | 21.19630251545459 |
| 0051_09 | 28.630172872543334 | 0.9713909144202868 | 31.166374279807012 |
| 0206_04 | 31.444559049606323 | 0.9701333358883858 | 33.26642719718318 |
| 0813_05 | 36.0873927116394 | 0.9868349740902582 | 18.613297779423494 |
| Mean | 32.31800518565708 | 0.9742519314918253 | 28.358787678492565 |

相对同配置 original 的六序列均值：

```text
VGGT corrected: PSNR 32.31800518565708, SSIM 0.9742519314918253, LPIPS*1000 28.358787678492565
original:       PSNR 32.31045727729798, SSIM 0.9742256858282619, LPIPS*1000 28.405228959551703
delta:          PSNR +0.0075479083591, SSIM +0.0000262456636, LPIPS*1000 -0.0464412810591
```

逐序列结论：

- `0206_04` 三项均改善，是主要正向序列；
- `0044_11`、`0007_04` 的 LPIPS 改善，但 PSNR/SSIM 略降；
- `0019_10`、`0051_09`、`0813_05` 的整体变化很小或略退化；
- 六序列平均只有弱正向，尚不能说明 VGGT-Garment 稳定优于 original。

运行完整性：

- `0007_04` 和 `0019_10` 已在 GPU 2/3 完成训练、评估并打印 `All sequences finished`；
- `0813_05` 首次评估因 GPU 3 争用发生 LPIPS OOM，之后使用空闲 GPU 3 补跑 `120/120` 成功，最终结果为补跑后的 JSON；
- 当前工作仓库有 VGGT 相关代码和笔记改动，参考基线 `/media/coding/ckx/SeqAvatar` 的 `git status` 为空，未被修改。

## 49. corrected 变换与 VGGT 作用判断（2026-08-30）

### 变换是否改对

已改对，并且不是只做形式上的修改。当前
`scripts/vggt_generate_clothing_candidates.py:171-199` 使用的逆变换是：

```text
offset_world -> offset_smpl = offset_world @ global_R
offset_target_inverse = inv(target_rot) @ offset_smpl
offset_canonical = offset_target_inverse @ canonical_rot.T
```

它对应 SeqAvatar `scene/gaussian_model.py` 中的行向量变换约定。真实 SMPL 变换 round-trip 验证：

```text
corrected max abs error = 2.384185791015625e-07
legacy max abs error    = 1.2513480186462402
```

因此旧版本确实有坐标逆变换错误，corrected 缓存已经使用了修正公式；六个训练日志也明确加载了
`output/vggt_garment_candidates_corrected/{sequence}/candidates.npz`。

### VGGT 是否发挥作用

发挥了，但作用有限。证据是六个日志都出现了：

```text
[VGGT_GARMENT] loaded=...corrected... candidates=...
[VGGT_GARMENT] iter=800/900/1000/1100 proposals=... replaced=... spawned=...
```

这说明 VGGT 候选不是摆设，确实参与了 Gaussian 替换和新增。最终相对 original 六序列均值：

```text
PSNR       +0.0075479083591
SSIM       +0.0000262456636
LPIPS*1000 -0.0464412810591
```

但这是弱正向，不能称为稳定显著提升。只有 `0206_04` 三项都明显改善，其余序列大多是持平、单项改善或轻微退化。

### 主要不是变换仍然错误，而是使用方式偏弱且有实现缺口

1. VGGT 没有进入训练损失。当前 controller 只做“候选点替换/新增”，后续仍由 original RGB、Mask、SSIM、LPIPS、AIAP 训练；没有弱的局部 VGGT 几何约束，因此候选初始位置可能很快被普通重建梯度覆盖。
2. 新点初始影响偏弱：`opacity=0.04`、scale 为父点的 `0.65`。这会让正确的候选在早期渲染中贡献很小。
3. 当前 candidate score 实际使用 confidence、跨视角近邻一致性、SMPL 外侧距离和 boundary；没有真正使用训练阶段的 `high_error_overlap`。因此候选不一定落在当前最需要补点的区域。审计中 boundary 命中和 high-error 命中都偏低。
4. 当前删点只按长期 opacity/gradient/visibility 低贡献排序，没有保护已有 high-error、boundary、重要运动区域或候选邻域，可能在替换时删掉仍有价值的 Gaussian。
5. “固定预算”当前只在事件开始时先删同数量点，但新增候选会因 `min_distance` 去重而少于删除数。例如日志中 `replaced=512` 而 `spawned` 常为 `484-511`。controller 只在点数超过 target 时裁剪，没有在点数不足时补齐，因此最终点数并不总与 original 一样：

```text
0007_04 VGGT 34649 vs original 35540
0019_10 VGGT 42781 vs original 42988
0044_11 VGGT 84118 vs original 83909
0051_09 VGGT 67059 vs original 68103
0206_04 VGGT 57102 vs original 58936
0813_05 VGGT 50249 vs original 50740
```

这会影响严格公平比较，也解释了部分序列的波动。

### 当前判断

```text
变换：已修正并通过数值验证；
VGGT：确实生效，但目前只是弱候选提议器；
主要瓶颈：候选与高误差区域错位、初始 opacity 太弱、删点没有保护、最终点数未严格对齐；
结论：不是 VGGT 完全无效，也不能说它本身信息量不足，当前版本更准确地说是“正确接入但利用不充分且预算实现不严谨”。

## 50. VGGT 严格预算实验（2026-08-30，运行中）

本轮只在 `0007_04` 和 `0206_04` 运行两组新消融，统一：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
```

实验 A：`vggt_garment_strict`

- 使用 corrected VGGT candidate cache；
- 候选先按 canonical 空间距离去重，再确定实际新增数量；
- `n_spawn` 与 `n_delete` 在每次事件严格相等；
- 末步将 Gaussian 数量强制校正到 original 目标：
  - `0007_04`: `35540`
  - `0206_04`: `58936`

实验 B：`vggt_garment_strict_high_error`

- 包含实验 A 的严格预算约束；
- 使用当前训练视角的 `abs(render - GT)` 误差；
- 将高误差 Gaussian 及候选父点加入删点保护；
- 用高误差重合分数参与 VGGT 候选优先级。

当前状态：实验 A 已启动，GPU 2 运行 `0007_04`、GPU 3 运行 `0206_04`；已观察到 `replaced == spawned`。最终指标和实验 B 结果待训练完成后补充。
```

## 51. strict + high-error 四序列实验完成（2026-08-30）

本轮是用户要求的另外四个 DNA 序列上的 `strict + high-error` 实验，使用 corrected VGGT candidate cache：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
vggt_garment_start_iter=800
vggt_garment_end_iter=1500
vggt_garment_interval=100
vggt_garment_max_spawn=512
vggt_strict_budget=true
vggt_high_error_enabled=true
```

序列与输出：

```text
0019_10: output/DNA-Rendering/0019_10/vggt_garment_strict_high_error/20260830_vggt_higherr_0019_0044/
0044_11: output/DNA-Rendering/0044_11/vggt_garment_strict_high_error/20260830_vggt_higherr_0019_0044/
0051_09: output/DNA-Rendering/0051_09/vggt_garment_strict_high_error/20260830_vggt_higherr_0051_0813/
0813_05: output/DNA-Rendering/0813_05/vggt_garment_strict_high_error/20260830_vggt_higherr_0051_0813/
```

高误差候选筛选在首次替换事件（iter=800）确实生效：

```text
0019_10: high_error_points=4015, selected_high_error=240, filtered_high_error=357, replaced=512, spawned=512, target=42988
0044_11: high_error_points=4663, selected_high_error=191, filtered_high_error=226, replaced=512, spawned=512, target=83909
0051_09: high_error_points=4172, selected_high_error=344, filtered_high_error=392, replaced=512, spawned=512, target=68103
0813_05: high_error_points=3947, selected_high_error=204, filtered_high_error=434, replaced=512, spawned=512, target=50740
```

运行完整性和预算核验：

- 四个序列都打印 `Training complete`，没有发现 `Traceback`、OOM、CUBLAS 或 CUDA 错误。
- 首次高误差替换均满足 `replaced=spawned=512`。
- 最终 PLY 点数严格等于 original 目标：
  - `0019_10`: `element vertex 42988`
  - `0044_11`: `element vertex 83909`
  - `0051_09`: `element vertex 68103`
  - `0813_05`: `element vertex 50740`
- 最终预算校正日志：
  - `0044_11`: `before=83528 pruned=0 padded=381 target=83909 final=83909`
  - `0051_09`: `before=67919 pruned=0 padded=184 target=68103 final=68103`
  - `0813_05`: `before=50658 pruned=0 padded=82 target=50740 final=50740`
  - `0019_10` 的 PLY 已核验为 `42988`，训练日志对应的最终预算目标为 `42988`。

指标均为 `novelview`、120 views、25000 steps；LPIPS 已乘以 1000，未四舍五入。original 使用同一批 `20260827_dna6_original_1800_tmux` 结果：

| 实验 | 序列 | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| original | 0019_10 | 35.257725365956624 | 0.9808832183480263 | 21.035196678712964 |
| strict + high-error | 0019_10 | 35.27290960947673 | 0.9809831599394481 | 20.927772112190723 |
| original | 0044_11 | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 |
| strict + high-error | 0044_11 | 32.95231997172038 | 0.9779256527622541 | 21.25621880404651 |
| original | 0051_09 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 |
| strict + high-error | 0051_09 | 28.690577745437622 | 0.9715634857614835 | 31.05802504966656 |
| original | 0813_05 | 36.10658038457235 | 0.9869976962606112 | 18.244266610903047 |
| strict + high-error | 0813_05 | 36.055868784586586 | 0.9869326864679654 | 18.269647215493023 |

逐序列差值（实验减 original）：

```text
0019_10: ΔPSNR +0.01518424352010328, ΔSSIM +9.994159142179271e-05, ΔLPIPS*1000 -0.10742456652224064
0044_11: ΔPSNR -0.035002851486204634, ΔSSIM -0.00016748011112210914, ΔLPIPS*1000 -0.022573024034500122
0051_09: ΔPSNR +0.02465945879618303, ΔSSIM +0.00017510900894801562, ΔLPIPS*1000 +0.007601625596482364
0813_05: ΔPSNR -0.05071159998576036, ΔSSIM -6.50097926457871e-05, ΔLPIPS*1000 +0.025380604589976485
```

四序列均值：

```text
original:            PSNR 33.25438671509425, SSIM 0.9793406060586373, LPIPS*1000 22.902169635441773
strict + high-error: PSNR 33.24291902780533, SSIM 0.9793512462327878, LPIPS*1000 22.877915795349203
delta:               PSNR -0.01146768728891967, SSIM +1.0640174150478021e-05, LPIPS*1000 -0.024253840092570478
```

结论：

- 本轮 high-error 筛选和保护逻辑确实被执行，strict 替换预算和最终点数公平性也已经成立。
- 但从四序列平均指标看，不能称为整体性能提升：PSNR 略降，SSIM 和 LPIPS 仅有极弱改善。
- `0019_10`、`0051_09` 的 PSNR/SSIM 改善，`0044_11`、`0813_05` 退化，说明当前 high-error 先验的收益依赖序列，稳定性不足。
- 这更像是“预算约束和候选保护机制验证成功”，还不能作为 VGGT + high-error 策略有效提升重建质量的证据。

## 54. strict + high-error 点云清单与产生过程

本轮六个序列使用同一套设置：

```text
VGGT model: facebook/VGGT-1B
frames: 0, 20, 40, 60, 80
views: 0, 8, 16, 24, 32, 40, 48, 56
candidate replacement: iter 800-1500, interval 100, max 512/event
strict budget: replaced == spawned
high-error: current abs(render - GT), visible foreground Gaussian, quantile=0.75
training: iterations=25000, densify_until_iter=1800
```

### 1. VGGT 候选点云

这些文件是候选点云，不是训练后的完整 Gaussian：

| 序列 | 候选数 | canonical 候选 | reference 候选 | 缓存 |
|---|---:|---|---|---|
| `0007_04` | 1943 | `output/vggt_garment_candidates_corrected/0007_04/vggt_canonical.ply` | `output/vggt_garment_candidates_corrected/0007_04/vggt_reference.ply` | `output/vggt_garment_candidates_corrected/0007_04/candidates.npz` |
| `0019_10` | 1837 | `output/vggt_garment_candidates_corrected/0019_10/vggt_canonical.ply` | `output/vggt_garment_candidates_corrected/0019_10/vggt_reference.ply` | `output/vggt_garment_candidates_corrected/0019_10/candidates.npz` |
| `0044_11` | 1566 | `output/vggt_garment_candidates_corrected/0044_11/vggt_canonical.ply` | `output/vggt_garment_candidates_corrected/0044_11/vggt_reference.ply` | `output/vggt_garment_candidates_corrected/0044_11/candidates.npz` |
| `0051_09` | 1336 | `output/vggt_garment_candidates_corrected/0051_09/vggt_canonical.ply` | `output/vggt_garment_candidates_corrected/0051_09/vggt_reference.ply` | `output/vggt_garment_candidates_corrected/0051_09/candidates.npz` |
| `0206_04` | 1480 | `output/vggt_garment_candidates_corrected/0206_04/vggt_canonical.ply` | `output/vggt_garment_candidates_corrected/0206_04/vggt_reference.ply` | `output/vggt_garment_candidates_corrected/0206_04/candidates.npz` |
| `0813_05` | 1818 | `output/vggt_garment_candidates_corrected/0813_05/vggt_canonical.ply` | `output/vggt_garment_candidates_corrected/0813_05/vggt_reference.ply` | `output/vggt_garment_candidates_corrected/0813_05/candidates.npz` |

`vggt_canonical.ply` 中的点由多视角 VGGT 点经过 SMPL 当前姿态到 canonical 空间的逆变换得到；`vggt_reference.ply` 保留对齐后的 VGGT 参考姿态坐标。`candidates.npz` 还保存 confidence、score、来源视角、帧号、最近 SMPL 顶点和颜色等字段。

### 2. 最终训练 Gaussian 点云

下面才是 strict + high-error 训练完成后的完整点云，包含原始 SMPL 初始化点、普通训练点以及被接受的 VGGT 替换点：

```text
0007_04: output/DNA-Rendering/0007_04/vggt_garment_strict_high_error/20260830_vggt_higherr_0007/point_cloud/iteration_25000/point_cloud.ply   (35540 points)
0019_10: output/DNA-Rendering/0019_10/vggt_garment_strict_high_error/20260830_vggt_higherr_0019_0044/point_cloud/iteration_25000/point_cloud.ply   (42988 points)
0044_11: output/DNA-Rendering/0044_11/vggt_garment_strict_high_error/20260830_vggt_higherr_0019_0044/point_cloud/iteration_25000/point_cloud.ply   (83909 points)
0051_09: output/DNA-Rendering/0051_09/vggt_garment_strict_high_error/20260830_vggt_higherr_0051_0813/point_cloud/iteration_25000/point_cloud.ply   (68103 points)
0206_04: output/DNA-Rendering/0206_04/vggt_garment_strict_high_error/20260830_vggt_higherr_0206b/point_cloud/iteration_25000/point_cloud.ply   (58936 points)
0813_05: output/DNA-Rendering/0813_05/vggt_garment_strict_high_error/20260830_vggt_higherr_0051_0813/point_cloud/iteration_25000/point_cloud.ply   (50740 points)
```

### 3. 候选如何进入最终点云

流程是：

```text
VGGT 多帧多视角推理
-> 前景、confidence、跨视角一致性、SMPL 外侧距离和边界筛选
-> 当前姿态点的局部偏移逆变换到 canonical 坐标
-> 保存 candidates.npz 和候选 PLY
-> iter=800 开始按 VGGT score + high-error parent score 排序
-> 删除低 opacity/gradient/visibility 的旧 Gaussian
-> 等量加入候选 Gaussian
-> high-error Gaussian 和候选父点加入保护集合
-> iter=25000 保存完整 Gaussian PLY
```

首次替换事件的实际统计如下；`replaced` 与 `spawned` 完全相等：

| 序列 | iter=800 | high-error Gaussian | 选中的 high-error 候选 | 最终预算校正 |
|---|---|---:|---:|---|
| `0007_04` | replaced=512, spawned=512 | 2899 | 194 | before=35912 -> final=35540 |
| `0019_10` | replaced=512, spawned=512 | 4015 | 240 | before=43858 -> final=42988 |
| `0044_11` | replaced=512, spawned=512 | 4663 | 191 | before=83528 -> final=83909, padded=381 |
| `0051_09` | replaced=512, spawned=512 | 4172 | 344 | before=67919 -> final=68103, padded=184 |
| `0206_04` | replaced=512, spawned=512 | 3982 | 342 | before=59712 -> final=58936 |
| `0813_05` | replaced=512, spawned=512 | 3947 | 204 | before=50658 -> final=50740, padded=82 |
```

补充：各实验目录下的 `input.ply` 是训练开始时的 SMPL 初始化点云，不能当作 VGGT 候选；`0206_04` 的 `20260830_vggt_higherr_0206` 是早期未完成目录，查看结果应使用带 `b` 的 `20260830_vggt_higherr_0206b`。

## 55. 新方向：MAPo-AllDynamic 方案

后续停止推进 VGGT，改为参考 MAPo 的高动态 Gaussian 时间递归划分思想。用户指定的改动是：不再计算高低动态分数，直接把全部 SeqAvatar Gaussian 视为高动态 Gaussian，递归划分时间区间。

### 推荐首版：逻辑 Gaussian 复制 + 时间段 deformation branch

SeqAvatar 当前是 canonical Gaussian 加一个共享 `NonrigidDeformer`。首版不物理复制 Gaussian 属性，而是：

```text
所有 Gaussian
-> 逻辑上复制到每个时间子区间
-> 每个子区间拥有独立的 deformation MLP/warp/rotation/scaling branch
-> 当前 pose_id 只激活所属时间子区间的 branch
-> canonical xyz、SH、opacity、scale、rotation 仍共享
```

这相当于先验证 MAPo 最核心的“时间段专用变形容量”，同时保持：

- Gaussian 数量与 original 完全一致；
- SMPL canonical 初始化不变；
- `densify_until_iter=1800` 不变；
- 原始 RGB、Mask、SSIM、LPIPS、AIAP 损失不变；
- 不把 point、part、VGGT 或已有 `use_time` 分支混入实验。

这种实现不是 MAPo 的完整物理属性复制版，而是适合 SeqAvatar 的第一版迁移。物理复制 Gaussian 属性会同时改变点预算、存储和渲染容量，不能作为首个公平消融。

### 时间递归划分

DNA 序列训练帧为 `pose_id=0..99`，采用左闭右开区间：

```text
level 0: [0, 100)
level 1: [0, 50), [50, 100)
level 2: [0, 25), [25, 50), [50, 75), [75, 100)
```

首版建议 `max_partition_level=2`，得到 4 个 temporal deformation branches。所有 Gaussian 在每一级都参与划分，不使用动态阈值筛选。`pose_id=99` 归入最后一个区间，边界不重叠。

采用渐进式创建 branch，避免一开始让每个 branch 从随机参数学习：

```text
iter < 5000:  level 0，共享 deformation branch
iter = 5000:  复制 root branch，建立 level 1 的两个 branch
iter = 10000: 分别复制 level 1 branch，建立 level 2 的四个 branch
iter >= 10000: 按 pose_id 路由到四个 branch
```

新 branch 从父 branch 复制权重，并加入 `mlp_optimizer`；父 branch 在分裂后冻结，避免同一时间段同时更新旧、 新两套网络。共享的 SeqPose/SeqXYZ 条件编码器保持训练，只有 deformation MLP 和三个属性输出 head 按时间段复制。

### 边界一致性第二阶段

时间 branch 独立后可能在 `25/50/75` 帧边界产生跳变。首版先只验证 partition 本身；若首版有效，再加入 MAPo 的跨段一致性：

```text
L_current = || render(current branch, t) - render(neighbor branch, t) ||_1
L_gt      = || render(neighbor branch, t) - GT(t) ||_1
L_cross   = 0.5 * L_current + L_gt
```

只在距离边界不超过 `2` 个 frame 的训练样本启用，并将权重从很小值开始，例如 `0.01`。neighbor branch 使用当前相机、当前 SMPL pose 和当前条件，只更换 temporal branch；这样约束的是时间分段造成的变形不连续，不改变数据真值。

### 实验配置

```text
mode=mapo_all_dynamic
iterations=25000
densify_until_iter=1800
max_partition_level=2
partition_level1_iter=5000
partition_level2_iter=10000
cross_frame_consistency=off  # 首版
use_time=false
use_point=false
use_part_moe=false
use_vggt_garment=false
```

建议保留三组对照：

```text
original
mapo_all_dynamic, level=1, 2 个时间 branch
mapo_all_dynamic, level=2, 4 个时间 branch
```

如果 level=2 有稳定收益，再测试 level=3 的 8 个 branch。MAPo 论文报告 level 增大通常带来收益但存在递减，并且成本继续增加；SeqAvatar 中 level=3 每个 branch 获得的训练样本更少，不能直接假设一定更好。

### 代码落点

建议新增独立开关和模块，不修改 original 的行为：

```text
arguments/__init__.py              新增 mapo_all_dynamic 参数
nets/mlp_delta_non_rigid.py        新增 temporal branch、分裂和 pose_id 路由
scene/gaussian_model.py            管理 branch 创建、optimizer 参数组和冻结
gaussian_renderer/__init__.py      将 viewpoint_camera.pose_id 传给 deformer
train.py                            在 5000/10000 step 触发递归分裂
scripts/exps_dnarendering.sh       新增独立实验模式和参数
```

关键验证项：

- original 模式的 forward、参数和输出必须不变；
- `mapo_all_dynamic` 的 active branch 对每个 `pose_id` 唯一；
- 分裂前后 Gaussian 点数严格相同；
- 训练日志记录 level、branch 数、当前 pose 区间和参数量；
- 先做小规模 smoke test，再跑四序列，最后扩展六序列；
- 指标统一记录 `PSNR | SSIM | LPIPS*1000`，不四舍五入。

### 不建议首版采用

```text
1. 一开始物理复制全部 Gaussian 属性到 2^L 份
2. 同时加入 static partition
3. 同时加入 L_current/L_gt、point redistribution 或 part MoE
4. 继续沿用现有 use_time 的 scale attention 作为 MAPo partition
5. 直接使用历史动态分数筛选，因为本实验的定义是全部 Gaussian 都是高动态
```

该方案的研究问题可以准确表述为：

```text
在保持 canonical Gaussian 数量和 SeqAvatar 原始损失不变的条件下，
将共享的非刚性变形网络递归划分为时间段专用 branch，
能否缓解单一 deformation network 对跨时间运动模式的平均化问题？
```

### 55.1 MAPo-AllDynamic 应复制的具体参数

是的，首版核心就是复制时间段专用的 deformation expert 参数，但这里的 expert 是“一个时间段对应一个网络”，不是“每个 Gaussian 一个网络”。所有 Gaussian 共享当前时间段的 branch。

当前 SeqAvatar 的非刚性变形部分由以下模块组成：

```text
shared condition encoders:
    PoseEncoder
    SeqPoseEncoder
    SeqXYZEncoder

deformation trunk and output heads:
    self.mlp
    self.gaussian_warp       -> d_xyz
    self.gaussian_rotation   -> d_rotation
    self.gaussian_scaling    -> d_scaling
```

首版复制范围：

```text
复制：self.mlp
复制：self.gaussian_warp
复制：self.gaussian_rotation
复制：self.gaussian_scaling
```

首版共享范围：

```text
共享：PoseEncoder
共享：SeqPoseEncoder
共享：SeqXYZEncoder
共享：canonical Gaussian 的 xyz、SH、opacity、base scaling、base rotation
共享：SMPL/LBS coarse deformation
```

因此一次 forward 的逻辑是：

```text
canonical Gaussian 全部输入共享条件编码器
-> 得到同一批 pose/sequence condition features
-> 根据 pose_id 选择唯一 temporal deformation branch
-> branch.mlp(features)
-> branch.gaussian_warp/rotation/scaling
-> 对全部 Gaussian 输出非刚性变形
```

### 55.2 为什么不复制条件编码器和 Gaussian 属性

条件编码器的作用是把当前姿态和历史运动编码成条件特征。它们不直接承担“不同时间区间使用不同变形函数”的容量瓶颈，因此共享可以：

- 保持不同时间段看到相同的运动条件表示；
- 减少参数量和显存；
- 让提升主要归因于 temporal deformation branch；
- 避免每个 branch 重复学习相同的 SeqXYZ/SeqPose 编码。

Gaussian 的 `xyz`、SH、opacity、base scaling、base rotation 也不能复制。复制它们会直接增加渲染点数和 Adam 状态，导致点数、显存和表示容量都改变；这不再是只测试时间划分的公平消融。

### 55.3 与现有 PartNonrigidExpert 的对应关系

当前代码中的 `PartNonrigidExpert` 已经采用类似结构：它对 `mlp`、`gaussian_warp`、`gaussian_rotation` 和 `gaussian_scaling` 做 `copy.deepcopy`，再对同一批 Gaussian 输出对应属性的变形。因此 MAPo-AllDynamic 可以复用这一设计模式，但路由依据由 `part_label` 改为 `pose_id` 所属时间区间。

### 55.4 分裂时的参数处理

递归分裂时，父 branch 复制出子 branch 的初始权重：

```text
root [0,100)
-> left  [0,50)   继承 root 权重
-> right [50,100) 复制 root 权重
```

再递归：

```text
left  [0,50)   -> [0,25), [25,50)
right [50,100) -> [50,75), [75,100)
```

重要修正：分裂后父 branch 不应整体冻结。原 branch 可以作为左子区间继续训练，复制出来的 branch 作为右子区间训练；否则左侧时间区间会停留在分裂时的旧状态，削弱分裂的意义。实现上只需要：

```text
保留父 branch 作为第一个子区间的可训练参数
deepcopy 父 branch 作为第二个子区间的可训练参数
将新 branch 加入 mlp_optimizer
```

新 branch 的权重从父 branch 复制，Adam 的动量状态建议初始化为零，避免两个 branch 共享同一 optimizer state。父 branch 原有 optimizer state 保持不变。

### 55.5 参数规模与显存判断

若只复制 `mlp + 三个输出 head`，参数增量与时间 branch 数量线性相关：

```text
level 0: 1 套 deformation branch
level 1: 2 套 deformation branch
level 2: 4 套 deformation branch
level 3: 8 套 deformation branch
```

但 Gaussian 的点数不变，渲染和 Gaussian optimizer 的显存不随 `2^level` 倍增。首版使用 level 2 的 4 个 branch，通常比物理复制 Gaussian 稳定得多；仍需在 2-step smoke test 中检查显存和 optimizer 参数组。

### 55.6 首版不复制的附加模块

本实验关闭已有 part、tri、VGGT 和 use_time 分支，因此首版不处理这些模块。如果未来在已有分支上研究 temporal expert，需要明确选择：

```text
共享：TriPlane/TokenTriPlane 等空间条件特征
或复制：其 temporal-specific output adapter
```

不能在同一个消融中同时复制它们并修改路由，否则无法判断收益来自时间 branch 还是额外空间模块。

## 56. 2026-08-30 MAPo-AllDynamic 独立消融运行记录

本实验独立于 point、part、DynOMo 和 VGGT 线，当前只测试“全部 Gaussian 使用时间专用 deformation branch”：

```text
mapo_all_dynamic_l1: 全部 Gaussian，2 个 temporal branch
mapo_all_dynamic_l2: 全部 Gaussian，4 个 temporal branch
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
```

时间划分和激活设置：

```text
level 0: [0, 100)
level 1: [0, 50), [50, 100)     @ step 5000
level 2: [0, 25), [25, 50), [50, 75), [75, 100) @ step 10000
```

只复制 temporal deformation expert 的 `mlp`、`gaussian_warp`、`gaussian_rotation`、`gaussian_scaling`；PoseEncoder、SeqPoseEncoder、SeqXYZEncoder、Gaussian 属性和 SMPL/LBS 保持共享。没有复制物理 Gaussian，因此 Gaussian 点数仍由 original 的 densification 流程决定。

首批序列 `0007_04`、`0206_04` 的 l1/l2 均已完成并生成最终 checkpoint、PLY 和 novel-view 指标。`0007_04` 的 l2 首次运行在约 17030 step 遇到一次 `CUBLAS_STATUS_EXECUTION_FAILED`，没有作为结果使用，已使用独立 `mapo_20260830_l2_retry` 目录从头重跑；`0206_04` 的 l2 同样使用 retry 目录。l2 retry 已确认正确切换到 4 branches。

根据后续任务调整，本轮最终完成了六个序列的 l1；l2 只在 `0007_04`、`0206_04`、`0044_11` 上完成。曾误启动后停止的 `0019_10`、`0051_09`、`0813_05` 后来按要求重新完成 l1。

最新追加任务：在 `0044_11` 上补跑同一套 MAPo-AllDynamic 消融，l1 使用 GPU1，l2 使用 GPU2；run time 为 `mapo_20260830_0044`，配置仍为 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`。该序列的 l1/l2 均已独立完成并完成最终 render。

### 56.1 最终指标

指标为 iteration 25000 的 novel-view 结果；`LPIPS*1000` 按要求直接乘 1000，不四舍五入。

| 序列 | 配置 | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| 0007_04 | original | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 |
| 0007_04 | mapo_all_dynamic_l1, 2 branches | 29.531344922383624 | 0.9585795074701309 | 44.17762826196849 |
| 0007_04 | mapo_all_dynamic_l2, 4 branches | 29.265462080637615 | 0.9573791856567064 | 45.535211482395724 |
| 0206_04 | original | 31.31386383374532 | 0.9695543631911278 | 33.811808843165636 |
| 0206_04 | mapo_all_dynamic_l1, 2 branches | 31.45844578742981 | 0.970130612452825 | 33.743841495985784 |
| 0206_04 | mapo_all_dynamic_l2, 4 branches | 30.436640135447185 | 0.9651019300023714 | 37.86216468239824 |
| 0044_11 | original | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 |
| 0044_11 | mapo_all_dynamic_l1, 2 branches | 33.00728626251221 | 0.9781651362776757 | 21.23346662459274 |
| 0044_11 | mapo_all_dynamic_l2, 4 branches | 32.51228893597921 | 0.9757799352208774 | 23.22227171777437 |

三序列均值：

```text
original:            PSNR 31.27750654220581, SSIM 0.9686949412027994, LPIPS*1000 33.36716234787471
mapo_all_dynamic_l1: PSNR 31.332358990775216, SSIM 0.9689584187335439, LPIPS*1000 33.05164546084901
mapo_all_dynamic_l2: PSNR 30.738130384021336, SSIM 0.966087016959985, LPIPS*1000 35.53988262752278
```

相对 original 的均值变化：

```text
l1: PSNR +0.054852448569405, SSIM +0.0002634775307445, LPIPS*1000 -0.3155168870257
l2: PSNR -0.539376158184474, SSIM -0.0026079242428144, LPIPS*1000 +2.17272027964807
```

结论：2-branch l1 在三个序列上均未破坏 original，三个指标均值有弱正向变化；但 `0007_04` 的 PSNR 提升几乎为零，整体不能称为稳定显著提升。4-branch l2 在三个序列上均低于 original，说明当前递归划分过细或分支训练/数据量不足，增加 branch 数量没有带来收益。

结果目录：

```text
0007_04 l1: output/DNA-Rendering/0007_04/mapo_all_dynamic_l1/mapo_20260830_l1/
0007_04 l2: output/DNA-Rendering/0007_04/mapo_all_dynamic_l2/mapo_20260830_l2_retry/
0206_04 l1: output/DNA-Rendering/0206_04/mapo_all_dynamic_l1/mapo_20260830_l1/
0206_04 l2: output/DNA-Rendering/0206_04/mapo_all_dynamic_l2/mapo_20260830_l2_retry/
0044_11 l1: output/DNA-Rendering/0044_11/mapo_all_dynamic_l1/mapo_20260830_0044/
0044_11 l2: output/DNA-Rendering/0044_11/mapo_all_dynamic_l2/mapo_20260830_0044/
```

### 56.2 三序列汇总复核

已重新读取最终 JSON 并确认 `0007_04`、`0206_04`、`0044_11` 的 l1/l2 指标、对应 original 基线、LPIPS*1000 换算及最终目录均一致；当前没有残留训练或 render 进程。

### 56.3 其余三个序列的 l1

按后续任务要求，补跑剩余三个 DNA 序列的 `mapo_all_dynamic_l1`（2 个 temporal branches）：

```text
序列：0019_10、0051_09、0813_05
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
run time: mapo_20260830_l1_remaining
GPU：0019_10 -> 0，0051_09 -> 1，0813_05 -> 2
```

三组任务已启动，完成后补充最终指标及与 original 的对比；本轮不运行这三个序列的 l2。

三组 `mapo_all_dynamic_l1` 已完成训练和最终 render，均确认 `level=1 branches=2`；没有 OOM、traceback 或残留训练/render 进程。

新增三序列指标如下，均为 iteration 25000 的 novel-view 结果，`LPIPS*1000` 未四舍五入：

| 序列 | 配置 | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| 0019_10 | original | 35.257725365956624 | 0.9808832183480263 | 21.035196678712964 |
| 0019_10 | mapo_all_dynamic_l1, 2 branches | 35.29945011138916 | 0.9810570314526558 | 21.25862887284408 |
| 0051_09 | original | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 |
| 0051_09 | mapo_all_dynamic_l1, 2 branches | 28.71627737681071 | 0.9716942911346753 | 30.847213269832235 |
| 0813_05 | original | 36.10658038457235 | 0.9869976962606112 | 18.244266610903047 |
| 0813_05 | mapo_all_dynamic_l1, 2 branches | 36.184804662068686 | 0.9872430662314097 | 17.923949301863708 |

相对 original：

```text
0019_10: PSNR +0.04172474543253912, SSIM +0.0001738131046294944, LPIPS*1000 +0.22343219413111726
0051_09: PSNR +0.05035909016926965, SSIM +0.0003059143821397825, LPIPS*1000 -0.20321015423784416
0813_05: PSNR +0.07822427749633931, SSIM +0.00024536997079849243, LPIPS*1000 -0.3203173090393392
```

三序列均值：

```text
original:            PSNR 33.343408012390135, SSIM 0.9797564304537243, LPIPS*1000 23.4432955712287
mapo_all_dynamic_l1: PSNR 33.40017738342285, SSIM 0.9799981296062469, LPIPS*1000 23.343263814846676
```

判断：这三个新增序列上，l1 的 PSNR 和 SSIM 均提升；LPIPS 在两个序列改善、一个序列轻微变差。结合前面的三个序列，六序列目前均值为：

```text
original:            PSNR 32.31045727729798, SSIM 0.9742256858282619, LPIPS*1000 28.405228959551703
mapo_all_dynamic_l1: PSNR 32.36626818709903, SSIM 0.9744782741698953, LPIPS*1000 28.19745463784784
```

六序列均值变化为 `PSNR +0.055810909801060404`、`SSIM +0.0002525883416334788`、`LPIPS*1000 -0.20777432170386234`。因此 2-branch l1 在六序列上表现为一致的弱正向趋势，但增益幅度仍较小，不能表述为显著提升。结果目录：

```text
0019_10 l1: output/DNA-Rendering/0019_10/mapo_all_dynamic_l1/mapo_20260830_l1_remaining/
0051_09 l1: output/DNA-Rendering/0051_09/mapo_all_dynamic_l1/mapo_20260830_l1_remaining/
0813_05 l1: output/DNA-Rendering/0813_05/mapo_all_dynamic_l1/mapo_20260830_l1_remaining/
```

### 56.4 六序列 MAPo 汇总

六个 DNA 序列的 `mapo_all_dynamic_l1`（2 个 temporal branches）均已完成；`mapo_all_dynamic_l2`（4 个 temporal branches）仅完成 `0007_04`、`0206_04`、`0044_11`。六序列 l1 的 PSNR 和 SSIM 在六个序列上全部高于 original；LPIPS*1000 在 5 个序列下降，仅 `0019_10` 上升。

六序列 l1 均值对比：

```text
original:            PSNR 32.31045727729798, SSIM 0.9742256858282619, LPIPS*1000 28.405228959551703
mapo_all_dynamic_l1: PSNR 32.36626818709903, SSIM 0.9744782741698953, LPIPS*1000 28.19745463784784
变化:                PSNR +0.055810909801060404, SSIM +0.0002525883416334788, LPIPS*1000 -0.20777432170386234
```

结论：2-branch MAPo 在六序列上呈现一致但较弱的平均改善，说明时间分支专门化有一定作用，但提升幅度不足以称为显著提升。已有三序列 l2 结果均低于 original，当前证据支持优先保留 2-branch 方案。

## 57. 当前代码审计与后续方向（2026-08-30）

### 57.1 当前工作区实际改动

当前工作仓库为 `/media/coding/ckx/human/SeqAvatar`；参考基线 `/media/coding/ckx/SeqAvatar` 未修改。工作区同时保留了历史 point、part、DynOMo、VGGT 和 MAPo 实验代码，但各实验通过脚本模式和参数互斥独立运行。

MAPo 相关核心改动：

```text
arguments/__init__.py
    增加 use_mapo_all_dynamic、最大层数、分裂迭代、帧数等参数
    当前默认优化配置为 iterations=30000、densify_until_iter=1800；
    DNA 实验脚本显式覆盖为 ITERATIONS=25000、DENSIFY_UNTIL_ITER=1800

nets/mlp_delta_non_rigid.py
    增加 TemporalDeformationExpert
    复制 mlp、gaussian_warp、gaussian_rotation、gaussian_scaling
    共享 PoseEncoder、SeqPoseEncoder、SeqXYZEncoder 和 Gaussian 属性
    用 pose_id 硬路由到时间区间 branch
    在 5000/10000/15000 step 递归激活第 1/2/3 层

scene/gaussian_model.py
    创建并传递 MAPo deformer 参数
    在训练开始设置 level 0，每轮根据 iteration 更新分层
    deformer 全部参数（包括预创建的 temporal branches）加入 mlp_optimizer

gaussian_renderer/__init__.py
    将当前 viewpoint 的 pose_id 传入非刚性网络

train.py
    每轮训练前更新 MAPo 分层
    MAPo 模式仍使用 original 的 RGB、Mask、SSIM、LPIPS、AIAP 损失
    其他模式额外包含历史 point/DynOMo/VGGT 控制逻辑

render.py
    加载 checkpoint 后切换到最终 MAPo level，保证测试使用正确 branch

scripts/exps_dnarendering.sh
    增加 mapo_all_dynamic_l1/l2 模式、参数传递、独立输出目录
```

其他历史实验改动主要位于 `part_label/common.py`、`utils/loss_utils.py`、`train.py`、`scene/gaussian_model.py` 和脚本文件中，包括 part 路由、point 增删、DynOMo 加权 AIAP、VGGT 候选点等；这些不属于 MAPo l1/l2 的有效路径。`note/*.pdf`、候选点 JSON 和审计脚本属于资料/工具，不会改变 original 的训练逻辑。

### 57.2 MAPo 实际实现边界

当前代码没有计算 MAPo 论文定义的 Gaussian 历史位置动态分数，也没有按 Gaussian 动态分数选择并复制 Gaussian 或独立 Gaussian deformation。全部 Gaussian 只是共享同一套时间分支，并按 `pose_id` 所在时间段选择 branch。因此当前实验应命名为：

```text
all-Gaussian time-partitioned deformation experts
```

不能直接表述为已经复现了 MAPo 的 dynamic-score Gaussian duplication。

分支参数确实参与训练：`training_setup()` 将整个 `non_rigid_deformer.parameters()` 加入 `mlp_optimizer`；分裂时子 branch 从父 branch 复制权重，原 root 继续作为左区间训练，新增 branch 作为右区间训练。checkpoint 保存完整 deformer `state_dict`，render 会按最终 level 加载，因此 l1/l2 的 branch 路由是有效的。

Gaussian 点没有物理复制，点数和 densification 仍由 original 流程决定；MAPo 没有改 RGB/Mask/SSIM/LPIPS/AIAP 的损失项，也没有改变 Gaussian 的最终预算。严格公平的含义是：同一脚本、同一 `iterations=25000`、同一 `densify_until_iter=1800`、同一数据和随机种子下，只比较 deformation branch 结构。与论文仓库默认值逐项比较时，必须以实际命令行日志中的参数为准，不能仅看 Python 默认值。

### 57.3 当前结果能支持的结论

```text
六序列 l1：PSNR +0.055810909801060404
             SSIM +0.0002525883416334788
             LPIPS*1000 -0.20777432170386234

三序列 l2：PSNR -0.539376158184474
             SSIM -0.0026079242428144
             LPIPS*1000 +2.17272027964807
```

l1 六个序列的 PSNR、SSIM 均高于 original，LPIPS*1000 五个下降、一个上升；这是弱而一致的正向趋势，不足以称为显著提升。l2 已完成的三个序列均低于 original，说明直接递归到 4 个硬时间 branch 当前不稳。

### 57.4 下一步改动优先级

1. **先做 l1 稳定性复核**：固定数据划分、命令行参数和随机种子，重复 0007、0206、0051 或使用 3 个 seed。当前增益很小，先排除随机波动。

2. **优先改 l2 的硬路由**：在相邻时间边界使用局部 soft blending，输出为相邻 branch 的加权和，避免区间边界输出跳变。先保持 4 branch 和其他设置不变，单独验证 blending 是否挽回 l2。

3. **减少 branch 的独立参数量**：下一版采用 shared trunk + branch residual，只为每个时间段增加小型 residual warp/rotation/scaling head，不再复制完整 `mlp`。这样每个 branch 只学习时间特有变化，降低数据稀疏和过拟合风险。

4. **分裂后加入短期 parent-to-child distillation**：让 child 在 warmup 内贴近父 branch 输出，再逐渐放开；这可以保持时间连续性，且不改变 Gaussian 点数和主损失。

5. **最后才实现真正 dynamic score**：若需要更贴近 MAPo，记录每个 Gaussian 的变形位置或 `d_xyz` 的 EMA，计算历史动态分数，再决定哪些 Gaussian 获得 residual capacity。即使用户要求全部 Gaussian 参与，也可以采用“全部共享 branch、动态分数只调 residual 强度”的低风险版本，避免物理复制和显存变化。

当前推荐下一实验：保持 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`，只实现 **MAPo-l2 + boundary soft blending**，先跑 0007 和 0206；若仍低于 original，再转向 shared trunk + residual，而不是继续增加分层层数。

## 58. MAPo 顺序优化与实现复核（2026-08-31）

统一配置仍为：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
```

### 58.1 l1 随机种子复核

训练入口已增加 `--seed`，DNA 脚本增加 `SEED`；同一个 seed 同时设置 Python、NumPy、CPU PyTorch 和 CUDA PyTorch。默认 seed 为 0，因此不改变旧实验默认行为。

seed=1 已完成的同 seed 对照：

| 序列 | 配置 | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| 0007_04 | original, seed=1 | 29.459373807907102 | 0.9580940753221512 | 44.98024797067046 |
| 0007_04 | l1, seed=1 | 29.511163314183552 | 0.9582483366131782 | 44.36817062087357 |
| 0206_04 | original, seed=1 | 31.287377039591473 | 0.9694054087003072 | 33.708139679705106 |
| 0206_04 | l1, seed=1 | 31.368675486246744 | 0.9695801511406898 | 34.14183179847896 |

同 seed 差值：

```text
0007_04: PSNR +0.05178950627644907, SSIM +0.00015426129102658082, LPIPS*1000 -0.6120773497968908
0206_04: PSNR +0.08129844665527175, SSIM +0.00017474244038262565, LPIPS*1000 +0.4336921187738536
```

当前结论：三个序列在 seed=0 和 seed=1 下的 PSNR、SSIM 提升方向一致；LPIPS 在 `0206_04 seed=1` 反向，说明 l1 是弱的 PSNR/SSIM 稳定信号，不是三个指标都稳定改善。三个 seed=1 序列的平均变化为 `PSNR +0.08838190502590611`、`SSIM +0.0002376432220141019`、`LPIPS*1000 -0.11308169147620717`。因此 l1 的小幅 PSNR/SSIM 提升不是单次随机结果，但仍需避免夸大为显著提升。

### 58.2 l2 审计发现的实现错误

旧 `mapo_all_dynamic_l2` 结果不能继续作为有效实验结论。代码审计发现两个会直接破坏 level 2 的错误：

1. `update_mapo_partition()` 每轮依次调用 level 1、level 2；旧 `mapo_activate_level(1)` 会把已激活的 level 2 降回 level 1，随后再次升到 level 2，导致 10000 step 后四分支每轮都被父分支覆盖。
2. level 2 复制父 expert 时保存的是引用原参数的 `state_dict`，先覆盖 branch 1 后会改变后续 branch 2/3 使用的父快照，导致父子继承不正确。

修复内容：

```text
- 每轮先计算唯一 target_level，只调用一次 activate
- 已激活更高层时禁止降级
- 父 branch state 使用 deepcopy 快照
- 新激活/重写的 branch 清空对应 Adam optimizer state
- full expert 与 residual expert 均通过 level 0/1/2 路由和父子继承单测
```

因此此前记录的三序列 l2 明显退化，主要是实现错误污染，不能用来证明“4 branches 本身无效”或“硬路由一定失败”。

### 58.3 l2 soft 与后续版本

已实现独立模式：

```text
mapo_all_dynamic_l2_soft
    4 branches
    25/50/75 帧边界左右 4 帧使用 smoothstep 相邻 branch blending
    其他帧仍走单 branch

mapo_all_dynamic_l2_residual
    shared deformation trunk + 每个时间区间的 residual warp/rotation/scaling head
    residual 零初始化
    保留 soft boundary routing

mapo_all_dynamic_l2_residual_dynamic
    在 residual 版本上记录非刚性变形位置的历史 EMA 偏差
    动态分数只控制 residual 强度，不复制 Gaussian
```

动态分数定义为：

```text
position_ema_i <- m * position_ema_i + (1-m) * (x_i + d_xyz_i)
dynamic_score_i <- m * dynamic_score_i + (1-m) * ||(x_i + d_xyz_i) - position_ema_i||
```

这避免把随机采样的相邻训练 iteration 直接解释为连续帧速度；它衡量每个 Gaussian 的非刚性变形位置相对历史中心的变化幅度。训练日志每 1000 step 输出 mean、p50、p95、max，后续必须先确认分数非零且有分布，再评价动态门控是否有效。

已完成的公平对照记录：

```text
0206_04 fixed hard l2: mapo_l2_hard_fixed_0206
0206_04 fixed soft l2: mapo_l2_soft_fixed2_0206
```

比较顺序必须是：

```text
original -> fixed hard -> fixed soft
```

如果 fixed soft 仍低于 original，再运行 shared trunk + residual；如果 residual 仍不稳定，再运行 dynamic-score residual。不能再拿旧的错误 l2 指标直接判断 soft blending 的作用。

### 58.4 动态分数实现校正

进一步复核后，动态分数已改为按真实时间邻域更新，而不是把随机训练 iteration 当作时间序列。缓存每个 `pose_id` 的变形位置，只有得到相邻 pose 的位置后，才用相邻 pose 间的位移更新 Gaussian 的动态分数 EMA；没有相邻缓存时只缓存当前位置，不更新分数。

动态分数更新只在训练态生效，render 的 `torch.no_grad()` 阶段不会改变分数；checkpoint 同时保存和恢复 `mapo_dynamic_score`。动态门控使用归一化分数控制 residual 强度：低动态 Gaussian 的 residual 被压制，高动态 Gaussian 保留更强 residual，Gaussian 数量和 densification 均不改变。

单测已覆盖：相邻 pose 才更新、推理态不更新、分数非零、各层级/边界 forward 正常。

### 58.5 当前运行状态

修复后的公平对照已完成：

```text
0206_04 fixed hard l2: mapo_l2_hard_fixed_0206
0206_04 fixed soft l2: mapo_l2_soft_fixed2_0206
0044_11 l1 seed=1: mapo_seed1_l1_fixed_0044
```

这些任务均为从头训练，配置仍是 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`。soft/hard 已完成并读取最终 render 指标；soft l2 在已复核的三个序列上均恢复并超过 original，因此没有启动 residual 和 dynamic residual 训练。

`0044_11 seed=1` l1 已完成：PSNR `33.08728585243225`，SSIM `0.9780003895362218`，LPIPS*1000 `21.359907113946974`。对应同 seed original 为 PSNR `32.955228090286255`、SSIM `0.9776164636015892`、LPIPS*1000 `21.52076695735256`，三项均改善。

### 58.6 l2 修复后的 soft routing 最终结果

`mapo_all_dynamic_l2` 的修复版仍保持 4 个完整 deformation branches；`mapo_all_dynamic_l2_soft` 只在 25、50、75 帧边界附近的左右 4 帧用 smoothstep 对相邻 branch 输出加权，其余帧仍是单 branch。两者都不改变 Gaussian 数量、densification、RGB/Mask/SSIM/LPIPS/AIAP 损失或训练预算。

最终评测均为 iteration 25000 的 DNA novel-view，`LPIPS*1000` 未四舍五入：

| 序列 | original PSNR | original SSIM | original LPIPS*1000 | l2 fixed-soft PSNR | l2 fixed-soft SSIM | l2 fixed-soft LPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 | 29.58473817507426 | 0.958546108007431 | 44.202796959628664 |
| 0206_04 | 31.31386383374532 | 0.9695543631911278 | 33.81180884316563 | 31.633441146214803 | 0.971306781967481 | 32.46326390653849 |
| 0044_11 | 32.98732282320658 | 0.9780931328733762 | 21.278791828081012 | 33.00627454121908 | 0.9783195222417513 | 21.126633618647854 |

0206_04 的修复后 hard l2（仅作 `hard -> soft` 对照）为 PSNR `31.5245721022288`、SSIM `0.9707271307706833`、LPIPS*1000 `33.241529343649746`。0007 和 0044 的旧 hard 结果不纳入本节有效对比。

三序列均值：

```text
original: 31.27750654220581 | 0.9686949412027994 | 33.367162347874704
l2 soft:  31.408151287502715 | 0.969390804072221  | 32.59756482827167
变化:     PSNR +0.1306447452969041 | SSIM +0.0006958628694216559 | LPIPS*1000 -0.7695975196030377
```

三个序列的 soft l2 相对 original 都是 PSNR、SSIM 提升且 LPIPS*1000 下降；因此当前“l2 不稳定”的条件已不成立。soft blending 对已修复的 l2 硬路由有明确的正向作用，但样本数只有三个，结论仍应表述为阶段性有效。

### 58.7 顺序优化的停止点

本轮按既定顺序执行到第二步即可停止：

1. l1 seed=1 复核完成，确认 PSNR/SSIM 的小幅提升方向稳定。
2. l2 硬路由修复并加入边界 soft blending，三个已复核序列均优于 original。
3. shared trunk + branch residual head 已保留为独立模式 `mapo_all_dynamic_l2_residual`，但因 soft l2 已稳定，不启动训练。
4. dynamic score residual 已保留为独立模式 `mapo_all_dynamic_l2_residual_dynamic`，动态分数只控制 residual 强度、不复制 Gaussian；因前一条件未触发，不启动训练。

后续若继续验证，应优先把 soft l2 扩展到其余序列，保持 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`、seed 和测试划分一致；不要把未训练的 residual/dynamic 版本写成已验证收益。

### 58.8 代码验证

静态检查已通过：

```text
python -m py_compile arguments/__init__.py scene/__init__.py scene/gaussian_model.py nets/mlp_delta_non_rigid.py train.py render.py
bash -n scripts/exps_dnarendering.sh
git diff --check
```

在 `seqavatar` 环境下的 MAPo smoke test 也通过：完整 branch、soft boundary routing、shared trunk + residual、dynamic-score gate 均完成 level 0→1→2 路由并可反向传播；dynamic score 仅在存在相邻 `pose_id` 缓存时更新，测试得到非零分数。当前没有残留 SeqAvatar 训练或渲染进程。

### 58.9 l1/l2 相对 original 的代码改动与功能

#### original 的非刚性变形路径

参考 `/media/coding/ckx/SeqAvatar` 基线，原始 `NonrigidDeformer` 只有一套：

```text
位置 embedding + pose condition + sequential pose condition + sequential xyz condition
    -> 一个共享 MLP
    -> gaussian_warp / gaussian_rotation / gaussian_scaling
    -> d_xyz / d_rotation / d_scaling
```

Renderer 将 `d_xyz` 加到 canonical Gaussian 位置后，再执行 SMPL/LBS 刚性变换；`d_rotation` 和 `d_scaling` 用于更新 Gaussian 的协方差。original 的 Gaussian 数量、点属性、densification、训练损失和优化器机制均保持不变。

#### l1 的修改

1. `nets/mlp_delta_non_rigid.py` 增加 `TemporalDeformationExpert`。每个 expert 深拷贝一套 `mlp`、warp head、rotation head 和 scaling head，而不是复制 Gaussian 本身。
2. root branch 使用原来的 `self.mlp` 和三个输出 head；额外 branch 放在 `self.mapo_branches` 中。l1 配置 `mapo_max_partition_level=1`，因此总共 2 个时间 branch。
3. `scene/gaussian_model.py` 将 MAPo 参数传入 deformer，并在训练开始设为 level 0；所有 branch 参数从一开始就加入 `mlp_optimizer`，但新增 branch 在激活前不参与前向输出。
4. 训练到 `mapo_partition_level1_iter=5000` 时，新增 branch 从 root 的参数快照初始化，并清空新增参数的 Adam state，使 child 从父 branch 的函数开始继续学习，而不是随机初始化。
5. 前向时由当前相机的 `pose_id` 选择 branch：

```text
pose_id 0-49  -> branch 0
pose_id 50-99 -> branch 1
```

因此 l1 实现的是“前半段和后半段时间区间使用不同的非刚性 deformation function”。它让网络能为不同时间段学习不同的形变残差/映射，减轻单一 MLP 对不同姿态运动的平均化。

#### l2 的修改

1. l2 使用 `mapo_max_partition_level=2`，最终得到 4 个时间 branch。
2. 训练流程是递归分裂：5000 step 从 1 个 branch 变为 2 个，10000 step 再从 2 个变为 4 个。
3. 分裂采用父子继承：level 1 的两个 branch 分别覆盖 `[0,50)` 和 `[50,100)`；level 2 的四个 branch 覆盖 `[0,25)`、`[25,50)`、`[50,75)`、`[75,100)`。level 2 的新 child 从对应 level 1 parent 的深拷贝参数初始化。
4. `scene/gaussian_model.py` 每轮先计算唯一的 target level，再只执行一次激活；这修复了旧实现反复覆盖 level 2 branch 的问题。父状态使用 `copy.deepcopy`，避免后续加载时被引用修改。
5. 基础 `mapo_all_dynamic_l2` 是硬路由；当前修复版的 `mapo_all_dynamic_l2_soft` 在 25、50、75 帧边界左右 4 帧，对相邻 branch 的三个输出使用 smoothstep 加权：

```text
output = (1 - w) * branch_left_output + w * branch_right_output
```

边界外仍是单 branch。这个修改的目标是消除时间区间切换造成的 `d_xyz`、rotation 和 scaling 跳变，减少渲染时的时序不连续。

### 58.18 soft blending 中左右 branch 的含义

soft blending 里的“左 branch”和“右 branch”指时间轴上的前一个 branch 和后一个 branch，不是人体或图像空间中的左右方向。以边界 `pose_id=50` 为例，前半段 branch 是左 branch，后半段 branch 是右 branch。

在 soft blend window 内，两个 branch 会同时参与输出；离开这个窗口后，仍然会恢复为单 branch：左侧窗口外为 100% 左 branch，右侧窗口外为 100% 右 branch。因此并不是整个序列都保持两个 branch 混合。

权重和为 1，实际计算是：

```text
最终形变 = 左 branch 形变 * 左权重
         + 右 branch 形变 * 右权重
```

原笔记中若写成“减去右 branch”，应理解为排版/符号错误，正确操作是加权相加。

#### 相关调用链

```text
scripts/exps_dnarendering.sh
    选择 l1/l2 模式和 5000/10000 step 分裂配置
        -> arguments/__init__.py
            解析参数并禁止与其他 ablation 混用
                -> scene/gaussian_model.py
                    构造 deformer、更新 partition、管理 optimizer state
                        -> gaussian_renderer/__init__.py
                            传入 pose_id 和 Gaussian query
                                -> NonrigidDeformer.forward_mapo()
                                    输出当前时间 branch 的非刚性变形
```

训练循环在每次 iteration 前调用 `update_mapo_partition()`；renderer 也会在加载 checkpoint 后切换到最终 level，保证评测时使用和训练结束一致的 branch 数量。MAPo 分支的 deformation 参数会随 checkpoint 保存到 `non_rigid_deformer` state dict。

#### 没有改动的部分

当前 l1/l2 实验没有：

- 物理复制、删除或增加 Gaussian；
- 根据 Gaussian 动态分数筛选点；
- 修改 RGB、Mask、SSIM、LPIPS 或 AIAP 损失；
- 修改原始 densification 规则或最终 Gaussian 预算；
- 引入 part、point、DynOMo 或 VGGT 模块。

因此当前 `mapo_all_dynamic_l1/l2` 更准确的名称是 **all-Gaussian time-partitioned deformation experts**。它借鉴了 MAPo 的“时间递归专门化”思路，但还不是 MAPo 原论文中基于 Gaussian 动态分数复制 Gaussian 的完整复现。

#### 当前实验含义

- l1 验证了将全部 Gaussian 的 deformation network 按时间一分为二是否有益；六序列 seed=0 的平均 PSNR/SSIM/LPIPS*1000 变化为 `+0.055810909801060404 / +0.0002525883416334788 / -0.20777432170386234`。
- l2 验证了更细的四段时间专门化；修复并加入 soft boundary 后，已复核的三个序列平均变化为 `+0.1306447452969041 / +0.0006958628694216559 / -0.7695975196030377`。
- 这说明当前收益主要来自 temporal deformation specialization，以及 l2 的边界连续性处理；不能归因于 Gaussian 数量变化或动态点筛选。

### 58.10 l1/l2 soft 与 original 的公平性边界

结论：当前比较可以作为**训练协议和 Gaussian 预算固定下的结构消融**，但不能称为参数量、显存和 FLOPs 完全相同的严格公平比较。

公平的部分：

- 使用相同 DNA 数据、训练/测试视角和 novel-view 评测方式；
- 参考 DNA 脚本与当前 MAPo 脚本设计上都使用 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`、seq_len/seq_xyz_knn/time_step_num 等相同训练配置；MAPo 日志明确打印了这些值，但已有 original 日志没有完整打印 `DENSIFY_UNTIL_ITER`，因此该项应以实际启动命令或重新运行日志最终确认；
- original、l1、l2 soft 都使用原始 RGB、Mask、SSIM、LPIPS、AIAP 损失；
- Gaussian 初始化、Gaussian 属性、densification 和最终 Gaussian 数量没有因 MAPo 分支改变；
- l1/l2 没有启用 point、part、DynOMo、VGGT 或 dynamic-score Gaussian 筛选。

不完全相同的部分：

- l1 增加一套完整 temporal deformation branch；l2 增加到四套完整 branch，因此可训练参数量和前向计算量高于 original；
- l1 在 5000 step、l2 在 5000/10000 step 改变了网络结构的有效容量，属于有额外模型容量的架构消融；
- `l2 soft` 同时包含“四段时间 branch”和“边界相邻 branch soft blending”两个改动。因此它相对 original 的提升不能全部归因于四段 branch，必须用 fixed-hard l2 作为中间对照才能单独估计 soft blending 的贡献。

还需注意：参考原版目录 `/media/coding/ckx/SeqAvatar` 没有被修改，但现有 original 指标日志的运行入口是当前工作仓库的扩展 `scripts/exps_dnarendering.sh`，只是 original 模式下关闭了 MAPo 等 ablation 开关。因此当前结果可以称为“当前扩展仓库中关闭新增模块的 original 对照”，不能直接等同于“已经由参考原版代码生成的 original”。若要做最严格的论文级声明，应使用参考仓库原版 `train.py`/`render.py` 重新跑 original，并固定完全相同的命令行参数和 seed。

推荐的表述：

```text
l1/l2-soft 在相同数据、训练轮数、Gaussian 预算和损失函数下，
相对 original 获得了结构消融意义上的提升；
但 MAPo branch 增加了 deformation network 容量，
因此不是参数量或计算量严格匹配的公平比较。
```

若要进一步拆分归因，实验顺序应为：

```text
reference-original
    -> fixed-hard-l1
    -> fixed-hard-l2
    -> fixed-soft-l2
```

其中每个序列都使用同一 seed；目前 fixed-hard l2 只在 `0206_04` 有有效修复版对照，所以三序列 l2 soft 结果暂时只能支持“组合方案有效”，不能精确报告 soft blending 单独贡献了多少。

### 58.11 六序列 l2 soft 最终指标

实验配置：

```text
实验：mapo_all_dynamic_l2_soft
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
MAPo：全部 Gaussian，max_partition_level=2，即 4 个时间 branch
分层：5000 / 10000 step
路由：相邻时间 branch 在边界使用 soft blending，blend_width=4.0
评测：novelview，120 views，iteration=25000
```

对比基线均来自 `original/20260827_dna6_original_1800_tmux`，指标顺序固定为 `PSNR | SSIM | LPIPS*1000`，保留完整小数：

| 序列 | original PSNR | original SSIM | original LPIPS*1000 | l2 soft PSNR | l2 soft SSIM | l2 soft LPIPS*1000 | ΔPSNR | ΔSSIM | ΔLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 | 29.58473817507426 | 0.958546108007431 | 44.202796959628664 | +0.053405205408733281 | +0.000108780463536573 | -0.80808941274881363 |
| 0019_10 | 35.257725365956624 | 0.9808832183480263 | 21.035196678712964 | 35.41862014134725 | 0.9814029360810915 | 20.801884145475924 | +0.16089477539062358 | +0.00051971773306525026 | -0.23331253323704004 |
| 0044_11 | 32.98732282320658 | 0.9780931328733762 | 21.278791828081012 | 33.00627454121908 | 0.9783195222417513 | 21.126633618647855 | +0.018951718012495178 | +0.00022638936837515722 | -0.15215820943315705 |
| 0051_09 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 | 28.72958860397339 | 0.9713795269529025 | 30.845224314058822 | +0.063670317331951054 | -0.0000088497996330039186 | -0.20519911001125735 |
| 0206_04 | 31.31386383374532 | 0.9695543631911278 | 33.811808843165636 | 31.633441146214803 | 0.971306781967481 | 32.463263906538486 | +0.31957731246948384 | +0.0017524187763532373 | -1.3485449366271496 |
| 0813_05 | 36.10658038457235 | 0.9869976962606112 | 18.244266610903047 | 36.21615295410156 | 0.9873696744441985 | 17.816613769779604 | +0.10957256952921313 | +0.00037197818358736257 | -0.42765284112344304 |

六序列均值：

```text
original: 32.310457277297978 | 0.9742256858282619 | 28.405228959551703
l2 soft:  32.447073525852623 | 0.974839673191309 | 27.793325547180658
差值:     +0.13661624855464538 | +0.00061398736304718593 | -0.61190341237104562
```

六个序列的 PSNR 和 LPIPS*1000 均改善，SSIM 有 5 个序列改善、`0051_09` 轻微下降；其中 5 个序列三项指标同步改善。需要保持严谨的归因边界：这组结果支持“l2 soft 组合方案在当前协议下有效”，但 l2 soft 同时包含 4 个时间 branch 和边界 soft blending，不能仅凭该表把提升拆分为某一个改动单独带来的贡献。

### 58.12 后续论证实验：参数量匹配与 boundary consistency

#### 结论

这两个实验都有价值，但优先级不同：

1. **参数量匹配实验优先级高，建议必须做。** 当前 l1/l2 相对 original 增加了 deformation branch，因此总参数量和计算量更高。没有参数量匹配对照，只能说明“增加时间 branch 后有效”，不能排除收益主要来自模型容量变大。
2. **显式 boundary consistency loss 建议做，但放在参数量匹配之后。** 当前 `l2 soft` 是在边界附近对左右 branch 的输出做加权平均，它能缓解输出跳变，但本身没有约束两个 branch 的输出应该一致。因此如果想声称“学习到了跨时间段一致性”，需要额外的 consistency loss 对照。

#### 参数量匹配怎么做

最有解释力的是增加一个单 MLP 对照：

```text
original：1 个 width=512 的 MLP
l1：2 个原始规模的 temporal branch
l1-wide：1 个更宽的 MLP，总参数量匹配 l1
```

如果 `l1 > original` 且 `l1 > l1-wide`，才能说明时间划分本身优于单纯扩大网络容量。l2 同理：

```text
l2：4 个 temporal branch
l2-wide：1 个更宽的 MLP，总参数量匹配 l2
```

此外还可以补充一个严格固定原始预算的版本：把每个 branch 缩窄，使两个 branch 的总参数量约等于 original。这样可以回答“在相同总参数预算下，时间划分是否仍然有效”，但 branch 变窄会降低单 branch 表达能力，所以它应作为补充，不建议替代 `wide-single` 对照。

参数匹配不能只按 width 粗略估计，应该实际统计 `NonrigidDeformer` 的 trainable parameters，并保持以下配置不变：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
相同 seed、数据、视角、损失、Gaussian 初始化和 densification
```

#### 显式 boundary consistency loss 怎么做

建议只在时间边界附近计算，不要对整段序列强行平滑。对边界 `b` 的左右 branch，在 overlap window 内约束它们对同一个 Gaussian 的 deformation output 接近：

```text
L_boundary = Huber(d_xyz_left - d_xyz_right)
           + beta_r * Huber(d_rotation_left - d_rotation_right)
           + beta_s * Huber(d_scaling_left - d_scaling_right)
```

第一版建议：

- 先只约束 `d_xyz`，rotation/scaling 暂时不加或使用更小权重；
- 只在边界左右 2~4 个 pose 的 overlap 区间启用；
- 使用很小的 `lambda_boundary`，避免把真实的动作变化和衣服褶皱抹平；
- 训练初期不启用，等 branch 完成继承并稳定后再启用；
- 只对两个 branch 的输出计算，不改变 RGB、Mask、SSIM、LPIPS、AIAP 和 Gaussian 数量。

#### 推荐消融顺序

最低成本且能回答主要问题的顺序是：

```text
original
    -> l1
    -> l1-wide（参数量匹配的单宽 MLP）
    -> l2 hard
    -> l2 soft
    -> l2 soft + boundary consistency
```

其中：

- `l1` 与 `l1-wide` 的比较验证时间划分是否优于单纯增大网络；
- `l2 hard` 与 `l2 soft` 的比较验证 soft routing 是否有效；
- `l2 soft` 与 `l2 soft + boundary consistency` 的比较验证显式一致性损失是否在 soft blending 之外继续带来收益。

如果实验资源有限，先做 `l1-wide` 六序列，再做 `l2 soft + boundary consistency` 六序列。只有当参数匹配后仍然提升，MAPo 的时间划分才具有更有说服力的创新归因；只有当加 loss 后继续提升，才值得把“显式跨边界一致性”作为单独创新点。

### 58.13 soft blending 的直观解释

`l2` 将时间序列分成多个 branch。硬路由会在边界突然切换：边界前完全使用左 branch，边界后完全使用右 branch。`soft blending` 则在边界附近让两个 branch 一起工作，并按距离边界的远近混合它们预测的 `d_xyz`、rotation 和 scaling：

```text
靠近左侧：左 branch 权重大，右 branch 权重小
正好过渡：两个 branch 各占一部分
靠近右侧：右 branch 权重大，左 branch 权重小
```

它混合的是 deformation 输出，不是 Gaussian 点、网络参数或训练损失。作用是把原本“突然换模型”变成“逐渐换模型”，从而减少边界处形变跳变、闪烁和时序不连续。它本身不是显式的一致性 loss，也没有直接要求两个 branch 的输出必须相同。

### 58.14 当前 l1/l2 与 MAPo 原文的划分区别

当前 SeqAvatar 的 `mapo_all_dynamic_l1/l2` 确实是按 `pose_id` 划分时间窗口，但具体含义是：

```text
所有 Gaussian 共用同一套时间窗口
pose_id 0~49  -> branch 0
pose_id 50~99 -> branch 1       # l1
```

l2 则进一步划成四个固定区间。这里的 `5000/10000 step` 是网络 branch 开始分裂的训练 iteration，不是数据的时间戳；真正决定当前帧使用哪个 branch 的是 `pose_id`。当前 l1/l2 没有使用 Gaussian 动态分数，也没有物理复制 Gaussian，因此更准确地说是“全部 Gaussian 的时间分段 deformation experts”。

MAPo 原文不是把全部 Gaussian 按 `pose_id` 统一分组，而是两步：

```text
每个 Gaussian 根据历史位置计算自己的 dynamic score
    -> 只挑出高动态 Gaussian
    -> 对这些 Gaussian 的时间区间递归二分
    -> 为它们复制 Gaussian 属性和对应 deformation network
```

论文中的 dynamic score 由历史位置的最大位移和位置方差共同得到，再用 harmonic mean 融合。低动态 Gaussian 被视为 static，以减少 deformation network 的计算；高动态 Gaussian 才进入更细的时间 partition。每个高动态 Gaussian 的区间在当前层级按时间中点切开，随后递归处理。

因此两者的核心差异是：

```text
当前 l1/l2：先统一划时间，再让所有 Gaussian 使用对应 branch
MAPo 原文：先逐 Gaussian 判断动态程度，再只给高动态 Gaussian 划时间并复制
```

当前实验保留了 MAPo 的“时间划分、branch 继承和专门化建模”思想，但没有复现原文的 dynamic-score 筛选、Gaussian 复制/static Gaussian 分支，也没有加入原文的 cross-frame consistency loss。当前的 boundary soft blending 是为 SeqAvatar 做的适配，用输出插值减轻硬切换；MAPo 原文主要使用 cross-frame consistency loss 处理 partition boundary。

### 58.15 pose_id 与时间帧的对应关系

对当前 DNA-Rendering 实验，可以把 `pose_id` 直接理解成数据序列中的时间帧编号：训练集读取 `pose_id=0,1,...,99`，每个 `pose_id` 对应一个动作姿态；同一姿态下的多个相机视角共享这个 `pose_id`。

因此当前 l1 的：

```text
pose_id 0~49  -> 前 50 个时间帧
pose_id 50~99 -> 后 50 个时间帧
```

这里的 `5000 step` 仍然是训练 iteration，表示模型训练到第 5000 步时才启用这两个 branch；它不是第 5000 帧。对于其他数据集，`pose_id` 的数量和含义要以对应数据读取器为准，不能无条件把所有数据集都解释成 100 帧。

### 58.16 MAPo 原文的 branch 数量

MAPo 原文可以得到 2 段、4 段，甚至更多段，但不是所有 Gaussian 都固定使用相同段数。每个 Gaussian 是否继续划分，由它自己的 dynamic score 和当前层级阈值决定：

```text
level 0：不划分，1 个时间段
第一次满足高动态条件：二分为 2 段
下一层仍满足高动态条件：继续二分为 4 段
再继续满足：可以得到 8 段……
```

因此，如果最大递归层数为 `N`，单个高动态 Gaussian 最多可以对应 `2^N` 个时间段；低动态 Gaussian 可能保持 1 段并作为 static。不同 Gaussian 可以停在不同层级，所以同一时刻场景中可能同时存在 1 段、2 段和 4 段的 Gaussian。

这与当前 SeqAvatar 的 l1/l2 不同：当前 l1 是所有 Gaussian 统一使用 2 个 branch，l2 是所有 Gaussian 统一使用 4 个 branch；它们是固定的全局时间划分，而不是 MAPo 原文那种逐 Gaussian、自适应的递归划分。

### 58.17 l2 soft 后续优化优先级

当前 `l2 soft` 是最值得继续的主线，但下一步应先验证已有收益，而不是立即叠加更多机制。六序列结果的平均变化为：

```text
PSNR       +0.13661624855464538
SSIM       +0.00061398736304718593
LPIPS*1000 -0.61190341237104562
```

推荐顺序如下：

1. **先做 l2 soft 的 seed 复现。** 当前六序列主结果是 `seed=0`，应至少选 3 个代表性序列用 `seed=1` 重跑；资源允许时直接做六序列。配置保持 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`、4 branch、5000/10000 分层和相同 soft blend。这样可以确认小幅提升不是随机初始化造成的。
2. **做参数量匹配对照。** 增加一个参数量接近 l2 的单宽 MLP `l2-wide`。如果 `l2 soft > l2-wide`，才能说明收益不只是 deformation network 变大；这是当前论文论证中最重要的缺口。
3. **在 l2 soft 上加入显式边界一致性。** 第一版建议参考 MAPo 原文的 cross-frame rendering consistency：在 25/50/75 边界附近，让相邻 branch 对同一时间和视角的渲染接近，并用当前帧 GT 约束，避免只靠自一致性导致过度平滑。权重从小值开始，只作用于边界窗口。
4. **改成 shared trunk + branch residual head。** 当前 l2 是 4 套完整 deformation network，容量和计算量明显增加。共享主干、每个时间段只保留 residual head，可以降低参数量，同时保留时间专门化；它适合在参数匹配对照之后作为更公平、更实用的 l2 版本。
5. **最后再做 dynamic-score 自适应。** 当前 l2 是所有 Gaussian 都进入四个 branch。可以用历史位置的 dynamic score 控制 residual 强度，或只让高动态 Gaussian 使用更细的 branch，低动态 Gaussian 继续走共享分支。这个方向更接近 MAPo 原文，但会同时改变路由和计算量，应该放在前面几项验证之后。

暂时不建议优先做：同时改变 branch 数量、分层 iteration、soft blend 宽度、损失权重和 dynamic score。这样即使指标变化，也很难判断到底是哪一个因素起作用。当前最稳妥的主线是：

```text
l2 soft 复现
    -> l2-wide 参数量匹配
    -> l2 soft + 显式 boundary consistency
    -> shared trunk + residual head
    -> dynamic-score adaptive routing
```

## 58.18 l2 soft seed=1 与参数量匹配 l2-wide 验证

实验目的：

```text
l2 soft seed=1：在完全相同的 l2 soft 配置下更换随机 seed，检查原有提升是否稳定
l2-wide：取消时间 branch，只使用一个更宽的 deformation MLP，参数量匹配 l2 soft
```

统一配置：

```text
DNA 六序列：0007_04、0019_10、0044_11、0051_09、0206_04、0813_05
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
l2 soft：4 个时间 branch，5000/10000 step 分层，soft routing，seed=1
l2-wide：单个 deformation MLP，NON_RIGID_MLP_WIDTH=1078，seed=0
```

参数量检查：

```text
l2 soft NonrigidDeformer：2836424
l2-wide NonrigidDeformer：2834490
差值：-1934，相对差异：0.068184446331%
parameter_match_check=PASS
```

最终 novel-view 25000 指标，LPIPS 已乘 1000：

| 序列 | l2 soft seed=1 PSNR | l2 soft seed=1 SSIM | l2 soft seed=1 LPIPS*1000 | l2-wide PSNR | l2-wide SSIM | l2-wide LPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.54713142712911 | 0.9583725323279698 | 44.22046854160726 | 29.556156762441 | 0.9586041912436485 | 43.999754094208285 |
| 0019_10 | 35.39699440002441 | 0.9811475485563278 | 21.091785506966215 | 35.34987134933472 | 0.9813330958286921 | 20.700243945854407 |
| 0044_11 | 33.08908807436625 | 0.9781018992265066 | 21.389403683133423 | 32.971333471934 | 0.9780771628022193 | 21.33734816840539 |
| 0051_09 | 28.878650045394895 | 0.9720172156890233 | 30.77486330488076 | 28.743329270680746 | 0.9720953489343325 | 30.276678603452943 |
| 0206_04 | 31.426626873016357 | 0.970050651828448 | 33.618241315707564 | 31.5509516398112 | 0.9708731353282928 | 32.98892732709646 |
| 0813_05 | 36.133305342992145 | 0.9871709302067756 | 17.883521388284862 | 36.1226492245992 | 0.9870720833539962 | 18.25768536267181 |
| 均值 | 32.41196602715386 | 0.9744767963058418 | 28.163047290096678 | 32.38238195313348 | 0.9746758362485303 | 27.926772916948213 |

seed=1 相对原 seed=0 的变化：

```text
原 seed=0 均值：PSNR 32.44707352585262，SSIM 0.974839673191309，LPIPS*1000 27.793325547180654
seed=1 均值：PSNR 32.41196602715386，SSIM 0.9744767963058418，LPIPS*1000 28.163047290096678
变化：PSNR -0.03510749869876525，SSIM -0.0003628768854671005，LPIPS*1000 +0.36972174291602317
```

结论：

- l2 soft seed=1 相对 original 仍有正向均值变化：PSNR +0.10150874985588842，SSIM +0.0002511104775798634，LPIPS*1000 -0.24218166945502162。
- 但 seed=1 比原 seed=0 变差，说明原先的 l2 soft 小幅提升存在随机波动，不能只凭一次 seed 宣称稳定收益。
- 逐序列看，seed=1 相对 original 的 PSNR 六序列均为正，LPIPS*1000 五序列变好、0019_10 略变差；SSIM 六序列均为正，但幅度很小。
- l2-wide 相对 original 的均值变化为：PSNR +0.0719246758355047，SSIM +0.0004501504202683397，LPIPS*1000 -0.4784560426034859。它也有一定正向结果，但不包含时间划分。
- l2-wide 相对 l2 soft seed=1：PSNR -0.029584074020383728，SSIM +0.00019903994268838376，LPIPS*1000 -0.23627437314846397。三项指标不是同向，不能说 l2 soft 在本次 seed 上全面胜出。
- 因此“收益来自时间划分而不只是参数量增加”目前只能得到弱支持：l2 soft 的 PSNR 均值高于 l2-wide，但 SSIM 和 LPIPS*1000 反而是 l2-wide 更好；还需要多 seed 或统计检验才能作强结论。

结果路径：

```text
l2 soft seed=1：output/DNA-Rendering/<sequence>/mapo_all_dynamic_l2_soft/<run>/metrics/results_novelview_25000.json
l2-wide：output/DNA-Rendering/<sequence>/mapo_all_dynamic_l2_wide/mapo_l2_wide_<sequence-prefix>_20260831/metrics/results_novelview_25000.json
```

运行备注：

```text
0019_10 的 seed=1 在训练完成、并已写入最终 metrics/results_novelview_25000.json 后，额外 render 脚本的 LPIPS 阶段发生 OOM。
该 JSON 的写入时间早于 OOM，且训练阶段已输出 [ITER 25000] Evaluating novelview #120 和 Training complete，因此本次表格使用的最终指标有效；但不能把该条日志记为全流程零错误。
其余本轮目标实验的最终 JSON 均已生成。
```

下一步判断：优先保留 l2 soft 的时间划分思路，但不要继续扩大 branch 容量。应先用至少 3 个 seed 计算均值和标准差，再考虑 shared trunk + branch residual head；同时可加入很小权重的 boundary consistency，验证 soft blending 之外是否存在独立的边界收益。

## 58.19 seed 的含义

`seed` 是随机种子的编号。训练中有些过程带有随机性，例如网络参数的随机初始化、训练相机或姿态的随机抽取，以及 Gaussian 增密时可能涉及的随机选择。设置固定 seed 后，同一份代码和配置可以尽量复现同一条训练轨迹。

例如：

```text
seed=0：使用第 0 套随机序列
seed=1：使用第 1 套随机序列
```

“换 seed 重跑”指的是：代码、数据、损失、训练步数、GPU 配置等都不变，只把 `--seed 0` 改成 `--seed 1`，重新完整训练一次。

目的不是让 seed=1 一定更好，而是检查结果是否依赖某一次随机初始化：

```text
不同 seed 都提升 -> 方法的收益更可信
某个 seed 提升、另一个 seed 下降 -> 收益可能受随机波动影响
```

本轮 l2 soft 中，seed=0 均值高于 seed=1，因此已经观察到一定随机波动；后续最好使用至少 3 个 seed，报告均值和标准差。

## 58.20 l2 soft 的进一步优化方向：连续低秩时间调制

当前 l2 soft 的主要局限：

```text
原始 deformation MLP -> 复制成 4 套完整 branch -> 按 pose_id 区间路由 -> 边界插值
```

因此它的收益可能来自网络容量增加，而不是时间建模本身。下一版建议做 `Pose-conditioned Low-Rank Temporal Adapter`，简称 `PLTA`：

```text
一个共享 deformation trunk
    + 一个小的连续时间/姿态调制器
    + 低秩 temporal residual adapter
```

核心形式可以写成：

```text
h_l = shared_trunk_l(h_{l-1})
g_l = temporal_encoder(normalized_pose, pose_history, normalized_pose_id)
delta_h_l = U_l((V_l h_l) * g_l)
h'_l = h_l + alpha * delta_h_l
```

其中：

- `normalized_pose_id` 提供连续的序列相位，不再把时间切成 2 段或 4 段；
- `normalized_pose` 和已有的 pose/sequence-pose feature 提供真实姿态条件，避免只按帧号路由；
- `U_l`、`V_l` 是低秩投影，rank 可从 8 或 16 开始；
- adapter 最后一层零初始化，训练初期等价于 original，降低破坏基线的风险；
- `alpha` 从小值开始 warm-up，让共享 trunk 先学习稳定的主体形变，再逐渐引入时间调制。

建议把调制放在共享 MLP 的中间层或输出头前，而不是复制完整 MLP。这样衣服褶皱可以获得时间相关的残差，身体主体仍由共享 trunk 保持稳定。

它与 MAPO 的区别：

```text
MAPO：历史位置 dynamic score -> 选择高动态 Gaussian -> 递归切时间区间 -> 复制 Gaussian/branch
PLTA：所有 Gaussian 共享一个连续 deformation field -> 时间/姿态生成低秩调制 -> 输出连续变化
```

PLTA 不使用：

- Gaussian dynamic score 排序；
- 高动态 Gaussian 筛选；
- 递归时间区间划分；
- Gaussian 物理复制；
- 每个时间区间一套完整 deformation network。

因此它不是 MAPO 的简单改写，而是 SeqAvatar 中的连续 pose-time deformation modulation。

### 推荐实验顺序

```text
original
-> l2-wide（已有参数量对照）
-> PLTA-r8
-> PLTA-r16
-> PLTA-r8 + latent temporal smoothness
```

公平性要求：

1. 所有实验使用 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`、相同数据和损失。
2. PLTA 报告新增参数量，并优先把新增参数控制在 original 的小比例；不能只与 l2 soft 比参数量。
3. 每个配置至少运行 `seed=0/1/2`，报告均值和标准差。
4. 先做不加额外 loss 的 PLTA，确认收益来自连续调制结构；之后再加入很小权重的 latent temporal smoothness。

### 第二阶段的稳定性约束

如果第一版 PLTA 有正向结果，可以对 temporal modulation code 加二阶差分约束：

```text
L_temporal = || c(t+1) - 2c(t) + c(t-1) ||_1
```

这里约束的是低秩调制系数 `c`，不是强行约束 Gaussian 的绝对位置，也不是把相邻姿态渲染成一样。这样可以抑制时间抖动，同时保留由姿态变化引起的真实非刚性运动。权重应从 `1e-5` 或 `1e-4` 量级试起，并单独做 loss ablation。

### 当前最推荐的具体版本

```text
PLTA-r8：shared original MLP width=512
时间输入：normalized pose_id + 当前 pose/seq_pose feature
调制位置：最后一个 hidden layer 和 deformation output head 之间
adapter：rank=8，zero-init，alpha warm-up
不复制 Gaussian，不划分时间区间，不使用 dynamic score
```

这条路线的论文论证点是：

```text
在近似原始参数预算下，连续 pose-time 低秩调制能否比单纯加宽 MLP 更好地表达非刚性形变？
```

如果 `PLTA-r8` 在参数量明显低于 l2 soft 的情况下仍能稳定提升，论证会比当前 l2 soft 更有说服力；如果只提升 l2-wide 而不提升 PLTA，则说明当前问题主要是容量而不是时间建模。

## 58.21 对 PLTA 的修正：更强的主线应改为运动模式分解

PLTA 虽然参数少，但本质仍是：

```text
original deformation MLP + 一个小 temporal residual adapter
```

这个改动太像工程补丁，不适合作为当前 l2 soft 的主要创新点。下一步更推荐 `Pose-conditioned Multi-scale Motion Basis`，简称 `PMB`。

### PMB 的核心

不按时间段复制 branch，而是把 deformation 主干拆成多个“形变模式分支”：

```text
Gaussian/pose feature
        -> 多个完整的 motion-mode deformation path
        -> 当前姿态下的 mode router
        -> 多个模式共同预测并加权融合
```

第一版可以设置三个模式：

```text
global articulated mode：身体整体和骨骼相关的平滑运动
local bending mode：关节附近、袖口、衣服局部弯曲
high-frequency cloth mode：褶皱、裙摆、边界等局部快速变化
```

每个 mode 都是独立的非线性 deformation path，不是只有最后一层的 residual head。对每个 Gaussian 和每个 pose，三个 mode 都计算输出：

```text
d_i,t = a_i,t,0 * d_i,t,0
      + a_i,t,1 * d_i,t,1
      + a_i,t,2 * d_i,t,2
```

其中 `a_i,t` 由当前姿态、姿态变化量、已有 sequence pose/xyz feature 和 Gaussian 的 canonical feature 预测。它不根据固定 `pose_id` 区间选 branch，因此同一个 Gaussian 在相邻姿态中可以平滑地改变运动模式组合。

### 为什么比 l2 soft 更有意义

```text
l2 soft：branch 0/1/2/3 代表前后不同时间区间
PMB：branch 0/1/2 代表不同的形变模式，所有时间都共同使用
```

l2 soft 的边界问题来自时间硬分区；PMB 没有时间边界，也不需要在边界做 branch 插值。它学习的是“当前 Gaussian 在当前姿态下应该由哪几种形变共同组成”，而不是“当前帧属于哪一个时间窗口”。

### 与 MAPO 的区别

PMB 不使用 MAPO 的关键机制：

- 不计算历史位置 dynamic score；
- 不筛选高动态 Gaussian；
- 不递归划分时间区间；
- 不复制 Gaussian；
- 不为不同时间区间复制 deformation network。

PMB 的基本变量是 `motion mode`，不是 `temporal partition`。它的创新问题是：

```text
能否把 SeqAvatar 的非刚性形变分解为可组合的多尺度运动模式，
并由姿态和 Gaussian 局部状态进行连续融合？
```

### 防止三个分支塌缩成同一个网络

仅使用 RGB/Mask 损失时，三个 mode 可能学成相同的函数。因此需要两个很小的辅助约束：

```text
L_mode_balance：避免 router 长期只使用一个 mode
L_mode_diversity：鼓励不同 mode 的 deformation direction 不完全相同
```

可以使用 batch 内的 mode 使用率熵约束和输出方向的 cosine decorrelation。权重必须很小，主监督仍然是 SeqAvatar 原有的 RGB、Mask、SSIM、LPIPS 和原有正则，避免辅助损失主导训练。

为了让模式具有可解释性，可以再加入弱的多尺度先验：

```text
global mode：较强空间平滑
local/high-frequency mode：较弱空间平滑，允许保留衣物细节
```

这不是把衣物 Gaussian 预先标成某一类，而是通过形变场的空间频率约束，让不同 mode 自然承担不同尺度的运动。

### 公平配置

PMB 必须同时做两种参数预算：

```text
PMB-budget-original：总参数量接近 original
PMB-budget-l2：总参数量接近 l2 soft
```

推荐第一版使用较窄的三个 mode path，并通过参数统计严格匹配 `original` 或 `l2-wide`。只有在相同预算下 PMB 仍然有效，才能说明收益来自形变模式分解，而不是网络变大。

### 推荐消融

```text
original
l2-wide
PMB without mode regularization
PMB + mode balance
PMB + mode balance + mode diversity
PMB full + weak multi-scale prior
```

每个配置至少使用 `seed=0/1/2`，统一 `ITERATIONS=25000` 和 `DENSIFY_UNTIL_ITER=1800`，同时记录总参数量、各 mode 的平均路由权重和不同姿态下的路由变化。

当前建议：不要继续把 l2 soft 做成更多时间 branch。先把 PMB 做成独立于 MAPO 的新消融；如果 PMB 的参数量匹配版本仍然优于 l2-wide，论文主线会比当前 l2 soft 更有说服力。

## 58.22 PMB 中如何区分不同形变模式

重要判断：只复制多个 branch 并让 RGB loss 自己训练，不能保证 branch 自动学成不同模式。分支存在置换不确定性，也可能全部学成相同函数。因此 PMB 需要“架构上的输入隔离 + 弱的模式约束”，不能只依赖 branch 编号。

推荐使用多尺度形变分解：

```text
global branch：低频 canonical geometry + pose/SMPL motion
local branch：joint-relative geometry + sequence pose/xyz feature
detail branch：高频 positional feature + local neighborhood feature
```

三个 branch 的区别不是人为给 Gaussian 打衣物标签，而是让它们看到不同尺度的信号：

- global branch 主要表达身体整体和关节驱动的平滑变化；
- local branch 表达关节、袖口和局部弯曲；
- detail branch 保留褶皱、轮廓和小范围非刚性变化。

具体实现可以把现有 `x_emb` 的 positional encoding 分为低频和高频两组，并给 local/detail branch 增加 canonical kNN 的局部聚合特征。三条 path 都可以是完整的非线性网络，但输入通道不同，所以不会只是三份完全相同的 MLP。

训练约束建议如下：

```text
L_global_smooth = sum_(i,j in canonical kNN) w_ij ||d_i^global - d_j^global||^2
L_mode_balance = 防止 router 长期只使用一个 mode
L_mode_diversity = 降低不同 mode 形变方向的 cosine 相似度
```

global branch 使用较强的空间平滑权重，local branch 使用中等权重，detail branch 不使用强平滑约束。RGB/Mask/SSIM/LPIPS 仍然负责决定总形变是否有用，辅助约束只负责规定“不同 branch 应该承担什么尺度”。

更严格的版本可以使用固定的 canonical 图滤波器：

```text
P_low(d)：邻域平均得到低频形变
P_high(d)=d-P_low(d)：得到局部细节形变
```

再让 global/detail 输出分别接近总形变的低频/高频部分。这样模式区分有明确数学含义，但需要控制 kNN 聚合的显存和计算量，建议作为第二阶段，不要第一版就加入。

因此 PMB 的第一版应定义为：

```text
mode-specific input pathways
+ full nonlinear mode branches
+ router balance
+ weak global smoothness and mode diversity
```

这比“branch 0/1/2 直接复制并相加”更容易解释，也比按 `pose_id` 分时间段更不接近 MAPO。需要避免的表述是“模型自动发现了 global/cloth mode”；更准确的说法是“通过多尺度输入和弱正则，学习可组合的多尺度 deformation bases”。

## 58.23 保留时间分支的更强方案：Pose-Phase Overlapping Temporal Experts

如果继续沿用 l2 soft 的时间分支思想，推荐不要再增加 branch 数量，而是改进“时间坐标、分支覆盖和分支衔接”。方案简称 `PP-OTE`。

### 1. 从 pose_id 时间改成姿态运动阶段

当前 l2 soft 使用固定的 pose_id 四等分：

```text
0~24、25~49、50~74、75~99
```

但 pose_id 相邻帧不一定有相同的运动变化量。建议根据整个序列的 SMPL 姿态轨迹计算一个全局 phase：

```text
v_t = weighted_pose_distance(pose_t, pose_{t-1})
s_t = normalize(cumulative_sum(v_t))
```

其中 `v_t` 可以由关节旋转变化和关节位移变化组成，权重固定在序列层面。再按照 `s_t` 而不是原始 `pose_id` 放置时间专家中心。

这不是 MAPO 的 dynamic score：

```text
PP-OTE：每个序列只有一个全局姿态 phase，不区分 Gaussian
MAPO：每个 Gaussian 单独计算历史动态分数并递归划分时间区间
```

PP-OTE 不使用 render 历史，不根据 Gaussian 运动大小改变 branch 数量。

### 2. 四个 branch 使用重叠时间窗口

当前 l2 soft 只有靠近边界的少量 pose 才会混合相邻 branch，其他时间基本是硬路由。建议改成固定的连续 basis 权重：

```text
w_k(s) = normalized RBF/B-spline basis
sum_k w_k(s) = 1
d(s) = sum_k w_k(s) * F_k(features, local_phase_k)
```

每个 branch 额外接收自己的局部 phase：

```text
local_phase_k = (s - center_k) / window_width_k
```

这样 branch 学习的是“一个姿态阶段内部的形变规律”，而不是死记一组绝对帧号。相邻窗口有明确重叠区域，时间变化全程连续，不只在四个边界附近插值。

### 3. 约束形变速度，而不只是约束形变值

单纯 soft blending 只能让两个输出的数值平均，不能保证经过边界时运动速度连续。可以在重叠区域加入一阶 temporal consistency：

```text
L_vel = ||
  (F_k(t+delta)-F_k(t))/delta
  - (F_{k+1}(t+delta)-F_{k+1}(t))/delta
||_1
```

这个损失约束的是相邻 branch 对形变变化趋势的判断。建议优先作用于 `d_xyz` 和 rotation，不直接约束颜色或 Gaussian opacity。权重从很小值开始，并只在窗口重叠区域计算。

可选的数值连续性约束为：

```text
L_value = ||F_k(t)-F_{k+1}(t)||_1
```

但 `L_vel` 比单纯 `L_value` 更有价值，因为它能减少 branch 切换造成的速度突变，同时允许不同 branch 在具体形变幅度上保留差异。

### 4. 保留完整时间 branch，但增加 branch 间信息交换

第一版可以继续使用完整的 temporal deformation experts，不需要退回到小 residual head。区别在于：

```text
每个 branch 输出前读取相邻 branch 的 hidden summary
```

在重叠区域，可以使用一个很小的 cross-branch fusion layer：

```text
h'_k = h_k + A_k(s) * W_k([h_{k-1}, h_k, h_{k+1}])
```

它让相邻时间专家共享过渡信息，避免每个 branch 独立拟合后再到输出端才插值。训练和推理时使用相同的融合方式，不会产生额外的时间边界逻辑。

### 推荐的论文消融链

```text
original
l2 hard：固定 pose_id 四分段
l2 soft：固定 pose_id + boundary blending
PP-OTE-A：全局 pose phase + 固定四个 branch
PP-OTE-B：PP-OTE-A + overlapping basis routing
PP-OTE-C：PP-OTE-B + velocity consistency
PP-OTE-D：PP-OTE-C + cross-branch hidden fusion
```

每一步只增加一个因素，可以回答：

```text
收益来自姿态 phase，还是来自重叠路由？
收益来自输出插值，还是来自真正的运动趋势连续性？
收益是否来自 branch 间的信息交流？
```

### 参数公平性

由于 PP-OTE 仍然保留多个完整 branch，必须把 branch width 调小，使总参数量分别匹配：

```text
PP-OTE-original：匹配 original 参数量
PP-OTE-wide：匹配 l2-wide 参数量
```

不建议直接用四个 width=512 的完整 branch，否则仍然会被质疑只是扩大网络。每个版本都要保存参数统计、路由权重、phase 边界和 `L_vel` 数值。

### 与 MAPO 的最终边界

PP-OTE 是序列级、连续的、固定容量的时间专家模型；MAPO 是 Gaussian 级、动态分数驱动的、递归时间划分模型。PP-OTE 不改变 Gaussian 数量，也不为不同 Gaussian 建立不同 partition，因此可以作为独立于 MAPO 的时间形变网络优化。

## 58.24 用运动/误差决定时间专家强度

可以使用高误差区域、高敏感度 Gaussian 和细粒度顶点/Gaussian 速度，但三者含义不同，不能把它们简单当成同一种 dynamic score：

```text
运动速度：这个 Gaussian 的形变变化有多快
加速度：运动变化是否突然
高误差：当前模型在哪些区域重建失败
高敏感度：哪些 Gaussian 的形变变化最容易影响训练损失
```

其中速度/加速度是“运动复杂度”信号，高误差/敏感度是“模型是否需要更多容量”信号。高误差不一定代表动态，例如纹理、遮挡和初始化错误也会造成高误差，因此不建议单独用高误差分配时间专家。

### 推荐方案：Motion-Conditioned Temporal Capacity

保留固定数量的全局时间 experts，但让每个 Gaussian 根据运动复杂度决定时间专家参与强度：

```text
一个序列共享 4 个 temporal experts
    -> 所有 Gaussian 都可以使用这 4 个 experts
    -> 高运动/高敏感度 Gaussian 使用更强的时间专门化
    -> 低运动 Gaussian 保持接近共享形变
```

可以定义一个 detached 的容量分数：

```text
C_i = normalize(
    w_v * velocity_i
  + w_a * acceleration_i
  + w_e * error_ema_i
  + w_s * sensitivity_ema_i
)
```

再用它控制时间专家融合的锐度或专门化强度，而不是控制 Gaussian 数量：

```text
r_i = sigmoid(C_i)
d_i(t) = (1-r_i) * d_i^shared(t)
       + r_i * sum_k w_k(t) * d_i^expert_k(t)
```

高运动 Gaussian 会更多使用局部时间 experts，低运动 Gaussian 主要使用 shared deformation。四个 experts 的参数对所有 Gaussian 共享，不存在每个 Gaussian 单独复制网络的问题。

### 这些信号怎么得到

#### 1. SMPL 顶点速度

从 DNA 训练序列已有的 SMPL pose/vertices 得到：

```text
v_j,t = ||vertex_j,t - vertex_j,t-1|| / delta_t
a_j,t = ||v_j,t - v_j,t-1|| / delta_t
```

再把 Gaussian 映射到最近 SMPL 顶点或其 LBS 影响顶点，得到 Gaussian 的初始速度和加速度。这个信号不依赖 render，不引入 novel-view 信息，最稳定、最适合做第一版。

但速度大不一定表示非刚性复杂：整条手臂快速摆动也可能只是刚性 LBS 运动。因此更好的运动信号是：

```text
nonrigid_velocity = total Gaussian velocity - LBS-predicted velocity
```

第一版可以先用 SMPL/LBS 速度，第二版再加入训练 warm-up 后统计的 Gaussian non-rigid residual velocity。

#### 2. 高误差区域

当前训练每次迭代都有：

```text
rendered image - training GT image
```

可以把像素误差投影或归因到当前可见 Gaussian，再按 Gaussian 和 pose_id 做 EMA：

```text
error_ema_i <- m * error_ema_i + (1-m) * pixel_error_i
```

它适合表示“哪里还没有拟合好”，不适合直接表示“哪里运动最复杂”。建议只作为 velocity 的乘法增强或弱权重修正，并且只使用 training views。

#### 3. Gaussian 敏感度

可以使用形变相关梯度的 EMA，例如：

```text
sensitivity_i = EMA(
    ||dL / d(deformed_means3D_i)||
  + ||dL / d(deformed_rotation_i)||
)
```

如果直接保留 `deformed_means3D` 的梯度代价太高，可以先用当前 renderer 已有的 `viewspace_point_tensor.grad.norm()` 作为近似。敏感度必须使用上一轮或前几轮的 detached EMA，不能在同一轮用它反向改变自己的梯度，否则会出现路由和梯度互相放大的反馈。

### 推荐训练流程

```text
0~5000 step：单 shared deformation MLP warm-up
             同时统计 error/sensitivity EMA
             速度和加速度从 SMPL 轨迹预先计算

5000 step：  固定或缓慢冻结容量分数 C_i
             从同一个 checkpoint 初始化 4 个 temporal experts
             使用 C_i 控制每个 Gaussian 的时间专家强度

5000~25000：继续使用原始 RGB/Mask/SSIM/LPIPS 损失训练
             不改变 Gaussian 数量，不递归重新划分时间
```

5000 step 只是推荐起点，因为它位于原 l2 soft 的第一阶段 branch 激活位置，且已经晚于 `densify_until_iter=1800`，可以避免点数量变化干扰 Gaussian 索引和容量统计。

### 建议的独立消融

```text
l2 soft：固定 pose_id 四分段
motion-only：SMPL/LBS velocity + acceleration
motion-error：motion-only + error EMA
motion-sensitivity：motion-only + sensitivity EMA
motion-error-sensitivity：三类信号全部使用
```

每个版本都保持相同 branch 数量、width、训练步数和 `DENSIFY_UNTIL_ITER=1800`。另外要做一个参数量匹配版本，排除专家容量增加的影响。

### 与 MAPO 的边界

这个方向只有在以下约束下才真正区别于 MAPO：

```text
专家数量对整个序列固定
所有 Gaussian 共享同一组 temporal experts
运动/误差只控制融合强度或时间带宽
不按 Gaussian 递归切分时间区间
不复制 Gaussian
不使用 MAPO 的历史位置 dynamic score
```

因此论文表述应是“motion-conditioned temporal capacity allocation”，而不是“根据高动态 Gaussian 复制更多时间专家”。前者是固定容量 deformation network 的条件化，后者会直接落入 MAPO 的方法范式。

## 58.25 SMPL/LBS motion-temperature 消融实现记录

本次新增独立实验 `motion_temporal_temperature`，目标是验证：

```text
SMPL/LBS vertex velocity + acceleration
    -> per-Gaussian temporal weight temperature
    -> fixed four temporal branches
```

实现约束：
- 不启用 MAPO historical dynamic score；
- 不复制或删除 Gaussian；
- 不改变 4 个 temporal branch 的数量和 5000/10000 step 激活时序；
- 训练和评估仍使用原始 RGB/Mask/SSIM/LPIPS 损失；
- 默认 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`，与当前 DNA 对比配置一致。

信号来源：DNA 的每帧 `model/*.npz` 已提供 SMPL/LBS 后的 `obs_xyz` 顶点轨迹。对相邻姿态计算顶点速度，对速度差计算加速度，再通过 canonical Gaussian 到 SMPL 顶点的 kNN 映射得到 Gaussian 级别信号。速度和加速度只作为 detached routing 条件，不参与梯度回传。

温度含义：高运动/高加速度 Gaussian 使用更低温度，时间分支权重更集中；低运动 Gaussian 使用更高温度，跨相邻时间分支的权重更平滑。branch 数量固定，改变的是融合权重的尖锐程度，不是网络容量。

## 58.26 实现检查

静态检查和两个真实数据 smoke test 已通过：
- 100 帧 `obs_xyz` 能计算 SMPL/LBS velocity 和 acceleration；
- Gaussian 通过现有 canonical-to-SMPL kNN 获得运动先验；
- level 1/2 提前激活 smoke 中，2/4 branch temperature routing、checkpoint 和 120 个 novel-view 评估均成功；
- smoke 日志确认 `MAPO_DYNAMIC_SCORE=0`，因此本实验没有使用 MAPO 历史动态分数。

正式运行配置：`ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`、`MAPO_PARTITION_LEVEL1_ITER=5000`、`MAPO_PARTITION_LEVEL2_ITER=10000`，实验名为 `motion_temporal_temperature`。

## 58.27 0007_04 正式结果

本次只保留并完成 `0007_04`；脚本在完成该序列后误继续启动了 `0019_10`，后者已终止且没有可用 checkpoint/指标，不纳入实验结果。

正式输出目录：

```text
output/DNA-Rendering/0007_04/motion_temporal_temperature/20260831_193100_motiontemp/
```

验收记录：
- 训练完成至 25000 step；
- `densify_until_iter=1800`，最终 Gaussian 数量为 35578；
- 5000 step 激活 level 1 / 2 branches；
- 10000 step 激活 level 2 / 4 branches；
- 日志确认 temperature routing 在两个 level 均实际执行；
- `mapo_dynamic_score_enabled=False`，没有使用 MAPO 历史动态分数；
- 120 个 novel views 评测完成。

路由日志统计：

```text
level=1: velocity=0.271464, acceleration=0.289309,
         temperature mean/p50/p95=1.219613/1.290723/1.331323
level=2: velocity=0.271374, acceleration=0.289132,
         temperature mean/p50/p95=1.219747/1.291063/1.331310
```

指标来源：

```text
output/DNA-Rendering/0007_04/motion_temporal_temperature/20260831_193100_motiontemp/metrics/results_novelview_25000.json
```

```text
0007_04: PSNR 29.593663787841795, SSIM 0.9587396214405696, LPIPS*1000 43.88694238538543
```

## 58.28 六序列 motion-temperature 正式结果

本次实验完成六个 DNA 序列。统一配置：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
NON_RIGID_MLP_DEPTH=3
NON_RIGID_MLP_WIDTH=512
MAPO_PARTITION_LEVEL1_ITER=5000
MAPO_PARTITION_LEVEL2_ITER=10000
MAPO_MAX_PARTITION_LEVEL=2
MAPO_SOFT_ROUTING=1
MOTION_TEMPERATURE_MIN=0.50
MOTION_TEMPERATURE_MAX=1.50
MOTION_VELOCITY_WEIGHT=0.50
MOTION_ACCELERATION_WEIGHT=0.50
MAPO_DYNAMIC_SCORE=0
```

本实验不改变最终 Gaussian 数量和 temporal branch 结构：5000 step 后为 2 branches，10000 step 后为 4 branches。SMPL/LBS 顶点轨迹预先计算 velocity/acceleration，映射到 Gaussian 后仅控制 temporal softmax 的 temperature；没有使用 MAPO historical dynamic score，也没有复制 Gaussian。

正式结果路径和指标：

| 序列 | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0007_04 | 29.593663787841795 | 0.9587396214405696 | 43.88694238538543000 |
| 0019_10 | 35.25254360834757 | 0.9808352952202161 | 21.045152307488026000 |
| 0044_11 | 32.99877621332804 | 0.9780600354075432 | 21.31470260210335000 |
| 0051_09 | 28.692870235443113 | 0.9718522702654203 | 30.558966193348167000 |
| 0206_04 | 31.4665625890096 | 0.9704382076859474 | 33.285868183399236000 |
| 0813_05 | 36.147056833902994 | 0.9871393685539563 | 17.91053149693956000 |
| mean | 32.35857887797886 | 0.9745107997622755 | 28.00036052811062816666666667 |

采用的正式输出目录：

```text
0007_04/motion_temporal_temperature/20260831_193100_motiontemp
0019_10/motion_temporal_temperature/20260831_221000_motiontemp6
0044_11/motion_temporal_temperature/20260831_220500_motiontemp6
0051_09/motion_temporal_temperature/20260831_215500_motiontemp6
0206_04/motion_temporal_temperature/20260831_213000_motiontemp6
0813_05/motion_temporal_temperature/20260831_214500_motiontemp6
```

与同配置 `original` 六序列均值对比：

```text
original: PSNR 32.31045727729797, SSIM 0.974225685828262, LPIPS*1000 28.405228959551701000
delta:    PSNR +0.04812160068087993, SSIM +0.0002851139340135737, LPIPS*1000 -0.40486843144107283333333333
```

排除说明：`0019_10` 的 220500 目录和 `0051_09` 的 220500 目录分别因启动阶段/评估阶段 OOM，不作为正式结果；`0051_09` 已用 215500 目录完整重跑。所有正式结果均有 25000 step checkpoint、120-view novel-view 评估文件和 `Finished sequence` 日志。

## 58.29 motion-temperature 改了什么、为什么可能有效

### 相对 original 的实际改动

原版只有一个 shared `NonrigidDeformer`，每个 Gaussian 都由同一套 MLP 和输出头预测非刚性位移、旋转和缩放。当前实验增加了三层逻辑：

1. **固定时间专家**：5000 step 从 1 个 deformation branch 扩展到 2 个，10000 step 再扩展到 4 个。新 branch 从已有 branch 的参数快照继承后继续训练；不复制 Gaussian，不改变 Gaussian 数量。
2. **SMPL/LBS 运动统计**：读取 DNA 每帧的 `obs_xyz`，按 SMPL 顶点计算相邻帧位移长度作为 velocity，按速度差计算 acceleration，再用 95% 分位数做归一化。
3. **Gaussian 级温度路由**：每个 Gaussian 通过最近 SMPL 顶点获得 velocity/acceleration，二者按 0.5/0.5 合成为 complexity。complexity 高的点使用较低 temperature，complexity 低的点使用较高 temperature，再用这个 temperature 计算 2/4 个时间 branch 的 softmax 权重。

关键调用链是：

```text
Scene.__init__()
  -> GaussianModel.prepare_smpl_motion_stats()
  -> renderer 用 canonical Gaussian -> 最近 SMPL 顶点映射运动先验
  -> NonrigidDeformer.forward_mapo()
  -> _forward_motion_temperature()
  -> 融合各时间 branch 的 d_xyz/d_rotation/d_scaling
```

### 直观理解

这次没有让“运动大的点拥有更多网络”，而是让它在已有时间专家之间更明确地选择。可以把 temperature 理解成 branch 权重的“聚焦旋钮”：

- 低运动点：温度高，邻近时间 branch 融合更平滑，避免时间边界出现突变；
- 高运动点：温度低，权重更集中，减少多个时间 branch 平均化快速形变的现象。

因此它可能有效的原因是：同一帧、同一人体上，不同区域的运动复杂度并不一样。腿、手臂等快速变化区域不必和慢变化区域使用同样的时间融合方式，网络可以保留更局部的时间行为；而平稳区域继续共享相邻 branch 的信息，降低硬切换带来的不连续。

### 当前结果能证明什么

六序列均值相对同配置 original 为：

| 实验 | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| original | 32.31045727729797 | 0.974225685828262 | 28.405228959551701000 |
| motion-temperature | 32.35857887797886 | 0.9745107997622755 | 28.00036052811062816666666667 |
| delta | +0.04812160068087993 | +0.0002851139340135737 | -0.40486843144107283333333333 |

这说明组合方案有小幅、三个指标方向一致的提升。但当前实验同时打开了：

```text
L2 temporal branches + soft routing + motion-conditioned temperature
```

所以不能把全部提升严格归因于 velocity/acceleration。提升可能来自 branch 容量、时间划分、soft blending 和 motion temperature 的共同作用。另一个限制是当前 velocity/acceleration 对整段序列求平均，同一个 Gaussian 在不同 pose 基本使用同一个 temperature，还不是逐帧的 `T(i,t)`。

### 下一步修改顺序

1. **先做归因消融**：保持相同 seed、branch、width、训练步数和 `DENSIFY_UNTIL_ITER=1800`，比较 `l2_soft`、`velocity_only`、`acceleration_only`、`velocity+acceleration`。只有这样才能知道当前提升是否真正来自运动温度。
2. **改成逐帧运动条件**：把当前序列平均统计改成当前 pose 附近的局部 velocity/acceleration，得到 `T(i,t)`，避免一个点在整段序列中始终使用同一温度。
3. **优先使用非刚性残差运动**：SMPL 顶点速度包含肢体整体刚性运动，建议计算“观测 Gaussian 运动 - LBS 预测运动”的 residual，再控制 temperature，减少把快速但规则的刚性运动误判为复杂非刚性运动。
4. **改进 Gaussian-SMPL 映射**：当前是最近单个 SMPL 顶点，衣服点容易被错误绑定到身体。可改为多个邻近顶点的距离加权或 LBS 权重加权，得到更稳定的 Gaussian motion prior。
5. **最后再考虑可学习修正**：用 motion prior 初始化 temperature，再增加一个很小的 bounded residual head 学习修正，并限制温度上下界；不要一开始就让网络自由学习 temperature，否则容易退化成无约束路由。

当前最推荐的下一版命名为 `motion_temperature_at`：固定 4 个 branch，只把静态序列平均先验改为逐 pose 的局部运动先验，先不加入新 loss、不增加 Gaussian、不改变 branch 数量。这样变量最少，也最容易和当前结果公平比较。

## 58.30 严格归因消融与逐帧运动温度实现

### 本次实验设计

为区分时间分支容量和运动先验的贡献，四组实验统一：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
NON_RIGID_MLP_DEPTH=3
NON_RIGID_MLP_WIDTH=512
MAPO_PARTITION_LEVEL1_ITER=5000
MAPO_PARTITION_LEVEL2_ITER=10000
MAPO_MAX_PARTITION_LEVEL=2
MAPO_SOFT_ROUTING=1
MAPO_SOFT_BLEND_WIDTH=4.0
```

四组只改变运动温度先验：

```text
mapo_all_dynamic_l2_soft                  无 motion temperature，控制组
motion_temporal_temperature_velocity     只使用 velocity
motion_temporal_temperature_acceleration  只使用 acceleration
motion_temporal_temperature               velocity + acceleration
```

控制组不计算运动温度，保持原有 4-branch soft routing；后三组 branch 数量和 soft routing 完全相同，只改变温度输入权重：

```text
velocity-only:     velocity_weight=1.0, acceleration_weight=0.0
acceleration-only: velocity_weight=0.0, acceleration_weight=1.0
both:              velocity_weight=0.5, acceleration_weight=0.5
```

GPU0 当时被无关进程占用，因此本轮新增三组使用 GPU1/2/3 并行，控制组使用已经核对过的同配置正式结果。当前运行批次的输出时间标记为 `20260901_motiontemp_at`，每组覆盖六个 DNA 序列。

### 逐帧 `T(i,t)` 的改法

旧实现对整个序列先求平均：

```text
velocity[i]、acceleration[i]
```

所以同一个 Gaussian 在所有 pose 上近似使用同一个 temperature。新实现先对每一个 SMPL 顶点和每一个 pose 计算局部统计：

```text
velocity[t] = ||(x[t+1] - x[t-1]) / 2||
acceleration[t] = ||x[t+1] - 2*x[t] + x[t-1]||
```

首尾帧使用单边差分。然后按 `pose_id` 保存 `velocity[t]` 和 `acceleration[t]`，渲染时把当前 pose 的 SMPL 顶点统计通过原有 Gaussian-to-SMPL 多邻居映射转换为 Gaussian 统计，最终形成：

```text
T(i,t) = T_max - complexity(i,t) * (T_max - T_min)
```

其中 `complexity` 是 velocity 和 acceleration 的加权平均。该温度仍然 `detach`，只是路由条件，不会让模型通过改变先验数值逃避损失。

### 预期归因判读

- `l2_soft` 高于三个 motion 版本：主要收益来自时间分支和 soft routing，运动温度没有额外收益；
- velocity-only 高于 l2 soft：速度先验有独立贡献；
- acceleration-only 高于 l2 soft：加速度先验有独立贡献；
- both 高于两个单项：速度和加速度有互补作用；
- 四组差距都很小：当前温度调制幅度或 SMPL-to-Gaussian 映射仍然太弱。

最终结论仍需以六序列表格和均值为准，不能只看单个序列。

## 58.31 严格归因消融最终结果与逐帧运动温度结论

### 配置核对

四组实验均使用：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
NON_RIGID_MLP_DEPTH=3
NON_RIGID_MLP_WIDTH=512
5000 step: 2 temporal branches
10000 step: 4 temporal branches
MAPO_SOFT_ROUTING=1
MAPO_SOFT_BLEND_WIDTH=4.0
```

除运动温度输入外，四组的 Gaussian 初始化、训练步数、densification 上限、MLP 结构、branch 数和 soft routing 均一致。控制组 `l2_soft` 不启用 motion temperature；`velocity_only` 只用 velocity；`acceleration_only` 只用 acceleration；`velocity+acceleration` 使用二者各 0.5 的加权组合。三种新增方案的正式输出均位于 `20260901_motiontemp_at`，每个序列使用 120 个 novel-view 评估视角。

### 六序列指标

LPIPS 已按要求乘以 1000，表中保留评估文件的完整数值，不做四舍五入。

| 序列 | l2_soft PSNR | l2_soft SSIM | l2_soft LPIPS*1000 | velocity_only PSNR | velocity_only SSIM | velocity_only LPIPS*1000 | acceleration_only PSNR | acceleration_only SSIM | acceleration_only LPIPS*1000 | velocity+acceleration PSNR | velocity+acceleration SSIM | velocity+acceleration LPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.58473817507426 | 0.958546108007431 | 44.202796959628664 | 29.557210556666057 | 0.9586801821986835 | 44.058886480828126 | 29.593943055470785 | 0.9588120386004447 | 44.21589765697718 | 29.577786779403684 | 0.958802259961764 | 44.21073192109664 |
| 0019_10 | 35.41862014134725 | 0.9814029360810915 | 20.801884145475923 | 35.285772482554115 | 0.9810203219453494 | 20.991708734072745 | 35.264321645100914 | 0.9809681142369906 | 20.96781125292182 | 35.30071900685628 | 0.9810694346825282 | 20.91037973295897 |
| 0044_11 | 33.00627454121908 | 0.9783195222417513 | 21.126633618647854 | 32.98012215296427 | 0.9781256849567095 | 21.21035037562251 | 33.004920879999794 | 0.9781109675765037 | 21.239680820144713 | 32.97622383435567 | 0.9781111627817154 | 21.27437904321899 |
| 0051_09 | 28.823214197158812 | 0.9720930164059003 | 30.348760883013408 | 28.69639490445455 | 0.9716239914298057 | 30.787187279202044 | 28.73629055023193 | 0.971817018588384 | 30.60124587112417 | 28.619359795252482 | 0.9716940194368362 | 30.86309313463668 |
| 0206_04 | 31.633441146214803 | 0.971306781967481 | 32.46326390653849 | 31.41500237782796 | 0.969993582367897 | 33.67609859754642 | 31.46214288075765 | 0.9703313867251078 | 33.43087211251259 | 31.453617684046428 | 0.9704549113909403 | 33.2983731602629 |
| 0813_05 | 36.21615295410156 | 0.9873696744441985 | 17.816613769779603 | 36.10690474510193 | 0.987039215862751 | 17.958896040606003 | 36.18733695348104 | 0.9872412120302518 | 17.93670333766689 | 36.17004192670186 | 0.9871794094642004 | 17.81651907755683 |
| mean | 32.4470735258526275 | 0.97483967319130893333333333333333333333333333333333 | 27.793325547180657000 | 32.340234536594813666666666666666666666666666667 | 0.97441382979353268333333333333333333333333333333333 | 28.113854584646308000 | 32.3748259941736855 | 0.97454678962628043333333333333333333333333333333333 | 28.065368508557893833333333333333333333333333333333 | 32.349624837769400666666666666666666666666666667 | 0.97455186628633075 | 28.06224601162183500 |

### 相对 l2_soft 的均值差

| 实验 | ΔPSNR | ΔSSIM | ΔLPIPS*1000 |
|---|---:|---:|---:|
| velocity_only | -0.10683898925781383333333333333333333333333333333333 | -0.00042584339777625 | +0.320529037465651000 |
| acceleration_only | -0.072247531678942 | -0.0002928835650285 | +0.27204296137723683333333333333333333333333333333333 |
| velocity+acceleration | -0.097448688083226833333333333333333333333333333333 | -0.00028780690497818333333333333333333333333333333333333333333333 | +0.268920464441178000 |

### 归因结论

1. `l2_soft` 在六序列均值上最好：PSNR 最高、SSIM 最高、LPIPS*1000 最低。当前可确认主要收益来自 4 个 temporal branch 加 soft routing，而不是速度或加速度温度调制。
2. `acceleration_only` 比 `velocity_only` 更接近控制组，但仍然没有超过控制组。说明局部动作变化率可能比单纯运动速度更有用，不过现有温度映射的额外收益不足。
3. `velocity+acceleration` 没有超过两个单项，也没有体现互补收益；简单线性融合不适合作为当前版本的最终方案。
4. 逐帧 `T(i,t)` 已实现并生效：每个 pose 使用当前帧的局部 finite difference 统计，而不是全序列平均值。其形式为：

```text
velocity[t] = ||0.5 * (x[t+1] - x[t-1])||
acceleration[t] = ||x[t+1] - 2*x[t] + x[t-1]||
T(i,t) = T_max - complexity(i,t) * (T_max - T_min)
```

但“逐帧”本身没有带来可见提升，说明当前先验可能没有准确表示非刚性形变复杂度，或温度对 branch 权重的调制方向/范围仍不合适。

### 下一步建议

优先检查 `SMPL/LBS motion prior` 与 Gaussian 实际非刚性位移、训练误差之间的相关性。如果相关性弱，不应继续扩大温度范围，而应改用：

1. `Gaussian 总运动 - LBS 预测运动` 的 residual motion，降低刚性骨骼运动对温度的干扰；
2. Gaussian 对多个 SMPL 顶点的距离加权映射，避免最近单顶点绑定错误，尤其是衣服点；
3. 在 warm-up 后统计非刚性 residual，再用 bounded、低幅度的 temperature correction；
4. 先做 motion prior 与误差/非刚性位移的相关性可视化，确认先验真的能找到需要额外时间建模的区域。

当前不建议继续增加 branch 数量或继续扩大 temperature 范围，因为本轮结果已经表明简单 motion temperature 没有提供独立增益。

## 58.32 l2-wide 是如何匹配参数量的

`l2-wide` 的目的，是构造一个“只有一套 deformation MLP、但总参数量接近 l2 soft”的容量对照。它没有增加时间专家，也没有使用 `mapo_all_dynamic` 路由；只将唯一 MLP 的 hidden width 从 `512` 改为 `1078`：

```text
l2 soft：4 套完整 deformation predictor，depth=3，width=512
l2-wide：1 套完整 deformation predictor，depth=3，width=1078
```

代码路径在 `scripts/exps_dnarendering.sh` 的 `mapo_all_dynamic_l2_wide` 分支：该模式将 `mapo_l2_wide_enabled` 设为 1，并强制 `non_rigid_mlp_width=1078`；因为没有启用 `use_mapo_all_dynamic`，因此不会创建 temporal branches。`nets/mlp_delta_non_rigid.py` 中的所有 hidden `Linear` 层和三个 deformation 输出头都使用这个 `W`，所以 widening 同时扩大了各 hidden 层以及 `gaussian_warp`、`gaussian_rotation`、`gaussian_scaling` 的输入维度。

宽度 `1078` 不是随意取的，而是根据实例化后的可训练参数量选择的：

```text
l2 soft NonrigidDeformer：2836424
l2-wide NonrigidDeformer：2834490
差值：-1934
相对差异：0.068184446331%
```

因此这个实验回答的是：在几乎相同的 deformation-network 参数预算下，4 个按时间划分的专家是否优于 1 个更宽的 MLP。它不能证明 l2 soft 在所有指标上都一定更好，但可以排除“l2 soft 只是因为参数更多所以变好”的最直接解释。已有结果中，l2-wide 的 PSNR 均值低于 l2 soft seed=1，但 SSIM 和 LPIPS*1000 更好，所以当时只能得到弱支持，不能据此宣称时间划分全面优于单纯增宽。

## 58.33 降低 l2 soft 参数量的判断

### 仅做输出加权不等于减少参数

如果仍然保留 4 套完整 temporal deformation MLP，只把输出改成：

```text
d = w_shared * d_shared + sum_k w_k * d_expert_k
```

或者像 `part_moe_leg_unknown_route_strong` 一样，让 global/root expert 和 part experts 按权重融合，那么：

```text
存储的 trainable parameters：不变
前向计算量：通常也不变，因为仍需计算参与融合的专家
专家输出的实际贡献：改变
```

因此这种方法可以叫 soft routing、soft fusion 或 expert contribution control，但不能称为参数量降低。当前 `PartNonrigidExpert` 和 `TemporalDeformationExpert` 都通过 `copy.deepcopy` 复制完整 MLP 与三个输出头；`part_score_route` 只是预测 `part_weight` 和未知点的专家混合权重，并没有删除这些专家参数。

### 真正减少参数的推荐结构

更合适的压缩方式是：

```text
一个 shared deformation trunk + 一个 shared base output head
        + 每个时间段一个很小的 residual warp/rotation/scaling head
```

这正是代码中已保留的 `mapo_all_dynamic_l2_residual` 思路。共享 trunk 负责主体和稳定运动，时间分支只学习相对 shared 输出的时间差异：

```text
d_k = d_shared + alpha * residual_k
```

residual head 零初始化时，训练开始阶段等价于 original，随后各时间段逐渐学习自己的残差。以 `width=512`、4 个 branch 为例，l2 soft 的 `2836424` 个参数主要来自 4 套完整 predictor；shared trunk + 4 个三输出 residual head 约为 `729626` 个参数，理论上比 l2 soft 少约 `74.27655385795634%`。这里的精确值仍应以实际实例化和参数统计为准。

### 结论

“主干和专家按权重软发挥”本身不降低参数量；只有在专家不再保存完整 MLP，而改成小 residual、低秩 adapter、共享层加少量专属层，或直接减少专家数量时，才是真正的参数压缩。最推荐的实验顺序是：

```text
l2 soft：4 个完整 width=512 predictor
    -> l2 residual：1 个 shared width=512 predictor + 4 个小 residual heads
    -> 若仍需压缩，再做 low-rank temporal adapters
```

比较时应同时报告总参数量、训练显存、推理耗时和指标；不能只报告路由权重变小，就把它解释成模型参数减少。

## 58.34 shared trunk + 4 residual heads 四序列实验

实验名称：

```text
mapo_all_dynamic_l2_residual
```

实验配置：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
NON_RIGID_MLP_DEPTH=3
NON_RIGID_MLP_WIDTH=512
5000 step: 1 -> 2 temporal branches
10000 step: 2 -> 4 temporal branches
MAPO_SOFT_ROUTING=1
MAPO_SOFT_BLEND_WIDTH=4.0
MAPO_SHARED_TRUNK=1
MAPO_DYNAMIC_SCORE=0
```

结果来源：

```text
output/DNA-Rendering/*/mapo_all_dynamic_l2_residual/20260901_l2residual_retry/metrics/results_novelview_25000.json
```

四序列 novel-view 25000 指标（LPIPS 已乘 1000）：

```text
0007_04: PSNR 29.451569747924804, SSIM 0.9581858073671659, LPIPS*1000 44.93876087168852
0019_10: PSNR 35.13415552775065,  SSIM 0.9804926489790281, LPIPS*1000 21.485776399883132
0044_11: PSNR 32.886080201466875, SSIM 0.9776990845799446, LPIPS*1000 21.470638597384095
0051_09: PSNR 28.632415930430096, SSIM 0.9713772470752398, LPIPS*1000 31.192227670301993
```

四序列均值：

```text
original:
    PSNR 31.610574861367542
    SSIM 0.9722005138794582
    LPIPS*1000 29.593824575810384

l2_soft:
    PSNR 31.70821176369985
    SSIM 0.9725903956840435
    LPIPS*1000 29.120018901691466

l2_residual:
    PSNR 31.52605535189311
    SSIM 0.9719386970003446
    LPIPS*1000 29.771850884814434
```

相对均值差值：

```text
l2_residual - original:
    PSNR  -0.084519509474432
    SSIM  -0.0002618168791136
    LPIPS*1000 +0.17802630900405

l2_residual - l2_soft:
    PSNR  -0.18215641180674
    SSIM  -0.0006516986836989
    LPIPS*1000 +0.651831983122968
```

判断：
- 四个序列中，residual 版相对 original 的 PSNR 和 SSIM 都下降，LPIPS*1000 都上升。
- residual 版相对 l2_soft 的三项指标也全部变差，因此本次压缩结构没有保留 l2_soft 的收益。
- 这说明“共享 trunk + 小 residual head”确实显著减少参数，但当前 residual 容量、初始化或训练方式不足以表达四个完整 temporal MLP 的效果。
- 本次结果不能说明压缩思路没有价值，只能说明当前实现不能作为 l2_soft 的等价替代；后续应优先检查 residual head 的容量、学习率/优化器状态和 branch 激活后的训练时长。

## 58.35 residual head 的含义

在当前代码中，`residual head` 是接在共享 deformation MLP 输出后面的一个很小的“修正层”，不是另一套完整 MLP。每个时间 branch 有三个线性层，分别预测：

```text
warp residual:     3 维位移修正
rotation residual: 4 维旋转修正
scaling residual:  3 维缩放修正
```

最终输出为：

```text
完整输出 = shared MLP 输出 + residual_alpha * 当前时间 branch 修正
```

这些 residual head 初始时权重和偏置都是 0，所以训练开始时模型等价于共享 MLP；训练过程中每个时间段只学习相对共享结果的少量差异。它能减少参数，但表达能力也明显弱于每个时间段各自拥有完整 MLP 的 `l2_soft`。

## 58.36 不使用 residual head 时的降参与严格归因方案

需要区分两个目标：

```text
降参：让模型实际保存的 trainable parameters 变少
归因：证明 l2_soft 的收益来自时间划分，而不是单纯容量变大
```

### A. 保留完整时间分支，但共享前面的层

推荐结构：

```text
共享第 1、2 个 hidden layer
        -> 4 个时间分支各自的最后 hidden layer + 完整输出头
```

每个 branch 仍然直接预测完整的 warp、rotation、scaling，不是“共享输出加 residual”。因此时间分支仍有独立的形变表达能力，但最耗参数的前面层只保存一份。也可以只共享第 1 个 hidden layer，作为容量和表达能力之间的折中。

这个方案比 residual head 更接近原来的 `l2_soft`，是“不改变分支语义”的首选压缩方法。

### B. 4 个完整分支都保留，但降低每个分支的 width

将 `width=512` 的 4 个完整专家改成较窄的 4 个专家，例如 `width=230~256`，使四个分支的总参数量接近 original。这样仍然是四个独立的时间 MLP，只是每个 MLP 更小。

这组实验最适合回答：在与 original 参数量接近时，时间划分本身是否仍然有效。

### C. 减少完整分支数量

可以只保留 2 个完整时间专家，把 100 帧分成前后两个区间，或者先用 2 个分支验证收益是否已出现。它不是最强的模型，但能直接观察：收益是否在较小参数增量下已经存在。

### D. 用参数匹配实验削弱“模型变大”的解释

已有 `l2-wide` 是：

```text
1 个 width=1078 的 MLP
vs
4 个 width=512 的 temporal MLP
```

两者总参数量接近，但已有结果中 `l2-wide` 的 SSIM / LPIPS 与 `l2_soft` 的优劣并不完全一致，所以它只能提供部分证据，不能单独证明时间划分的收益。

更完整的归因实验应同时包含：

```text
original：1 个 width=512 MLP
l2-narrow：4 个窄的完整 temporal MLP，总参数量匹配 original
l2-soft：4 个 width=512 temporal MLP
l2-wide：1 个宽 MLP，总参数量匹配 l2-soft
```

四组保持相同 seed、depth、训练步数、densify 配置、loss、优化器和评估协议，并报告：

```text
总 trainable parameters
实际训练显存
前向 FLOPs / 每步耗时
PSNR、SSIM、LPIPS
```

判断规则：

- `l2-narrow > original`：支持“时间划分本身有收益”；
- `l2-soft > l2-wide`：支持“时间划分优于单纯增大网络”；
- 只有 `l2-soft > original`，但 `l2-narrow <= original` 且 `l2-soft <= l2-wide`：收益更可能主要来自参数容量；
- `l2-wide` 在 SSIM / LPIPS 更好时，不能只用 PSNR 宣称 temporal routing 全面有效。

当前最值得做的是 `l2-narrow` 和“共享前两层、分支独立最后一层”的 partial-sharing 实验；它们比单纯给输出加权更能保留 l2_soft 的时间专家含义，同时能降低或控制参数量。

## 58.37 l2 partial-sharing 四序列实验状态

实验模式：`mapo_all_dynamic_l2_partial`，共享前两层、四个分支独立后端，`iterations=25000`、`densify_until_iter=1800`、`seed=0`，序列为 `0007_04`、`0019_10`、`0044_11`、`0051_09`。

本次四个任务均正常启动并训练至约 18000--19500 步，但随后被外部任务会话同时终止；四个输出目录均没有 `iteration_25000/point_cloud.ply`、`results_novelview_25000.json` 或 `Training complete`。因此本次没有可用于对比的最终 PSNR、SSIM、LPIPS*1000，不能据此评价降参方案是否有效。

## 58.38 l2 partial-sharing 重跑状态

由于前一次训练会话被外部终止，已使用 `seqavatar` Python 环境重新启动四个序列，采用独立运行目录：`20260901_l2partial_retry2`（0007_04）、`20260901_l2partial_retry5`（0019_10）、`20260901_l2partial_live2`（0044_11）、`20260901_l2partial_live3`（0051_09）。四个训练进程当前均存活，分别使用 GPU 0--3；截至本次检查尚未生成最终 `results_novelview_25000.json`，所以仍没有可汇总的最终评价指标。

## 58.39 l2 partial-sharing 四序列最终指标

四个重跑均完成 `iteration=25000` 的 novel-view 评估，每个序列 `num_views=120`。partial-sharing 采用共享前两层、四个分支独立后端；配置为 `densify_until_iter=1800`、`seed=0`。

| 序列 | original PSNR | original SSIM | original LPIPS*1000 | l2 soft PSNR | l2 soft SSIM | l2 soft LPIPS*1000 | partial PSNR | partial SSIM | partial LPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 | 29.58473817507426 | 0.958546108007431 | 44.202796959628664 | 29.588104470570883 | 0.9586118206381797 | 44.36599009980758 |
| 0019_10 | 35.257725365956624 | 0.9808832183480263 | 21.035196678712964 | 35.41862014134725 | 0.9814029360810915 | 20.801884145475924 | 35.35039869944254 | 0.9811417390902837 | 20.85616036783904 |
| 0044_11 | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 | 33.00627454121908 | 0.9783195222417513 | 21.126633618647855 | 32.980789772669475 | 0.9780922984083493 | 21.39106597751379 |
| 0051_09 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 | 28.823214197158812 | 0.9720930164059003 | 30.34876088301341 | 28.665311702092488 | 0.9715624819199244 | 31.022088788449764 |
| Mean | 31.610574861367542 | 0.9722005138794582 | 29.593824575810384 | 31.70821176369985 | 0.9725903956840435 | 29.120018901691463 | 31.64615116119385 | 0.9723520850141844 | 29.408826308402546 |

相对 original，partial-sharing 均值变化为：PSNR `+0.03557629982630672`，SSIM `+0.00015157113472619166`，LPIPS*1000 `-0.1849982674078383`。四个序列中 0007_04、0019_10 的三项指标整体改善，0044_11 轻微变差，0051_09 的 PSNR 轻微变差但 SSIM 和 LPIPS 改善。

相对完整四分支 l2 soft，partial-sharing 均值为：PSNR `-0.06206060250600132`，SSIM `-0.00023831066985913196`，LPIPS*1000 `+0.28880740671108285`。结论：共享前两层确实保留了部分时间划分收益，同时降低了参数量，但性能未达到完整 l2 soft；因此后续应继续做参数量统计和共享层数/分支宽度的折中实验。

## 58.40 partial-sharing 补充序列实验状态

为补齐六个 DNA 序列，已按相同配置启动 `0206_04` 和 `0813_05`：`iterations=25000`、`densify_until_iter=1800`、`seed=0`、四个时间分支、soft routing、共享前两层。`0813_05` 使用 GPU1，输出目录为 `20260901_l2partial_extra0813`；`0206_04` 首次在约 5930 步、level-1 分支切换后的反向传播中出现 `CUBLAS_STATUS_EXECUTION_FAILED`，该次没有有效结果。最终结果见 58.41；0206 使用 GPU2 的 `retry3` 完整重跑成功。

## 58.41 partial-sharing 六序列最终指标

`0206_04` 的 retry2 在约 23500 步中断，没有最终文件；因此使用 GPU2 完整重跑 `retry3`。本次配置为 `iterations=25000`、`densify_until_iter=1800`、`seed=0`、四个时间分支、soft routing、共享前两层。训练、25000 步点云保存和 120 个 novel-view 的独立 render 均完成。

`0206_04` 最终文件：

```text
output/DNA-Rendering/0206_04/mapo_all_dynamic_l2_partial/20260901_l2partial_extra0206_retry3/metrics/results_novelview_25000.json
```

```text
PSNR 31.451732524236043
SSIM 0.970228873193264
LPIPS*1000 33.18014269073804
```

六个 DNA 序列的 novel-view 25000 指标如下。original、完整四分支 `l2_soft` 和 partial-sharing 使用各自已完成的同配置结果；LPIPS 已乘以 1000，全部保留 JSON 原始精度。

| 序列 | original PSNR | original SSIM | original LPIPS*1000 | l2 soft PSNR | l2 soft SSIM | l2 soft LPIPS*1000 | partial PSNR | partial SSIM | partial LPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 | 29.58473817507426 | 0.958546108007431 | 44.202796959628664 | 29.588104470570883 | 0.9586118206381797 | 44.36599009980758 |
| 0019_10 | 35.257725365956624 | 0.9808832183480263 | 21.035196678712964 | 35.41862014134725 | 0.9814029360810915 | 20.801884145475924 | 35.35039869944254 | 0.9811417390902837 | 20.85616036783904 |
| 0044_11 | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 | 33.00627454121908 | 0.9783195222417513 | 21.126633618647855 | 32.980789772669475 | 0.9780922984083493 | 21.39106597751379 |
| 0051_09 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 | 28.823214197158812 | 0.9720930164059003 | 30.34876088301341 | 28.665311702092488 | 0.9715624819199244 | 31.022088788449764 |
| 0206_04 | 31.31386383374532 | 0.9695543631911278 | 33.81180884316563 | 31.633441146214803 | 0.971306781967481 | 32.46326390653849 | 31.451732524236043 | 0.970228873193264 | 33.18014269073804 |
| 0813_05 | 36.10658038457235 | 0.9869976962606112 | 18.244266610903045 | 36.21615295410156 | 0.9873696744441985 | 17.816613769779603 | 36.16539355913798 | 0.9872171610593795 | 17.910796799696982 |
| Mean | 32.31045727729797 | 0.974225685828262 | 28.405228959551703 | 32.44707352585263 | 0.9748396731913088 | 27.793325547180658 | 32.36695512135824 | 0.9744757290515635 | 28.12104078734087 |

六序列 partial-sharing 相对 original 的均值变化为：PSNR `+0.056497844060267255`，SSIM `+0.000250043223301577`，LPIPS*1000 `-0.2841881722108326`。相对完整 l2 soft，partial-sharing 的均值变化为：PSNR `-0.08011840449439234`，SSIM `-0.00036394413974527584`，LPIPS*1000 `+0.32771524016021303`。因此降参方案仍保留了部分时间划分收益，但整体低于完整四分支 l2 soft；后续需要报告参数量并继续寻找容量与时间专家表达能力之间的折中。

## 58.42 partial-sharing 相对 original 的代码修改

本次 `mapo_all_dynamic_l2_partial` 对比中，original 使用原始的一套 NonrigidDeformer MLP；partial-sharing 只启用了 MAPO 时间分支和 soft routing，没有启用 point、part、DynOMo、VGGT、dynamic score 或额外 loss。两者共用相同的输入特征、Gaussian 参数、RGB/Mask/SSIM/LPIPS/AIAP 损失，以及本次实验的 `iterations=25000`、`densify_until_iter=1800`、`seed=0`。

### 1. 参数和开关

`arguments/__init__.py` 新增了 `use_mapo_all_dynamic`、`mapo_max_partition_level`、分层迭代步数、`mapo_num_frames`、`mapo_soft_routing`、`mapo_soft_blend_width` 和 `mapo_partial_sharing` 等参数。`scripts/exps_dnarendering.sh` 中，partial 模式实际传入：

```text
use_mapo_all_dynamic=True
mapo_max_partition_level=2
mapo_partition_level1_iter=5000
mapo_partition_level2_iter=10000
mapo_num_frames=100
mapo_soft_routing=True
mapo_soft_blend_width=4.0
mapo_partial_sharing=True
```

original 模式不传 `--use_mapo_all_dynamic`，因此保持单路径 MLP。

### 2. 网络结构

original 的形变网络是：

```text
Linear(input, 512) -> ReLU
-> Linear(512, 512) -> ReLU
-> Linear(512, 512) -> ReLU
-> warp/rotation/scaling 三个输出头
```

partial-sharing 在 `nets/mlp_delta_non_rigid.py:1632-1639` 把前四个 Sequential 子层取出，即前两层 Linear 及其 ReLU，保存为共享 trunk；剩余的第三层 Linear/ReLU 保存为 `mapo_partial_suffix`。随后在 `:1641-1656` 为时间分支创建独立的 suffix 和三个输出头。

因此实际结构是：

```text
共享 Linear1 + Linear2
        -> branch_0: root suffix + root warp/rotation/scaling head
        -> branch_1: 独立 suffix + 独立三个输出头
        -> branch_2: 独立 suffix + 独立三个输出头
        -> branch_3: 独立 suffix + 独立三个输出头
```

这不是 residual head：每个分支后端都直接预测完整的位移、旋转和缩放输出；只是最前面的两层特征提取被四个分支共用。

按本次 DNA 设置 `input_ch=63+32+32+96=223`、`W=512`、`D=3` 估算，NonrigidDeformer 参数量约为：original `645130`，完整四分支 l2 soft `2580520`，partial-sharing `1448488`。所以 partial-sharing 仍比 original 大，但约为完整 l2 soft 的 `56.135%`，降参来自共享前两层而不是减少分支数量。

### 3. 时间分层和参数继承

`train.py:830-831` 每个训练 iteration 调用 `gaussians.update_mapo_partition(iteration)`；`scene/gaussian_model.py:541-565` 根据迭代步数触发分层：

```text
0-4999: 1 个 root branch，覆盖 100 帧
5000:   2 个 branch，每段 50 帧
10000:  4 个 branch，每段 25 帧
```

`mapo_activate_level()` 会把 root 或父 branch 的 suffix/head 参数快照复制给新子分支，然后清空新参数在 Adam 中的旧状态。它不是重新随机初始化，因此新分支从已有形变模型继续学习。所有分支参数在 `scene/gaussian_model.py:766-779` 初始化时已经加入 `mlp_optimizer`，只是未被当前路由使用的分支在激活前没有梯度。

### 4. forward 路由

`gaussian_renderer/__init__.py:95-113` 把当前相机的 `pose_id` 传给 NonrigidDeformer；`nets/mlp_delta_non_rigid.py:2027-2081` 根据 pose_id 选择时间分支。

当 pose 不在边界附近时，只使用对应 branch。每个 level 的边界附近 `4` 个 pose 范围内，同时计算左右 branch，并按 smoothstep 权重融合位移、旋转和缩放：

```text
output = (1 - blend) * left_output + blend * right_output
```

因此 partial-sharing 相对 original 增加的是“按 pose_id 划分的时间专家 + 边界 soft blending”，而不是新增 Gaussian 或改变监督信号。

### 5. 没有改动的部分

本次 partial-sharing 没有改变：

- SMPL/LBS 初始化和 Gaussian 数量逻辑；
- Gaussian 的位置、颜色、opacity、scale、rotation 优化方式；
- RGB、Mask、SSIM、LPIPS 和 AIAP 损失；
- 输入的 pose、sequence pose、sequence xyz 特征编码；
- densification 的时间范围和训练总步数。

所以它的性能差异主要来自网络容量重新分配和时间条件路由。partial-sharing 相对 original 的六序列均值有小幅改善，但相对完整 l2 soft 仍低，说明共享层确实压低了参数，同时也牺牲了一部分时间分支的独立表达能力。

## 58.43 l2 soft 与 partial-sharing 的直观区别

对 `l2_soft` 的理解基本正确，但 5000 和 10000 步都不是从零随机创建专家：

```text
0-4999:   一个 root MLP 学习 0-99 帧
5000:     root/branch 0 负责 0-49 帧，branch 1 负责 50-99 帧
10000:    branch 0/1/2/3 分别负责 0-24、25-49、50-74、75-99 帧
```

新 branch 会继承父 branch 的完整 MLP 和输出头参数，再继续训练。`l2_soft` 的 `TemporalDeformationExpert` 会复制完整的 `self.mlp`、warp head、rotation head 和 scaling head，因此最终相当于四套完整形变网络；soft blending 的边界处理与 partial-sharing 相同。

partial-sharing 不减少四个时间 branch，也不改变上述帧区间，而是在 `nets/mlp_delta_non_rigid.py:1632-1656` 中只共享 MLP 的前两层：

```text
l2_soft:
    branch 0 = prefix + suffix + 3 个输出头
    branch 1 = prefix + suffix + 3 个输出头
    branch 2 = prefix + suffix + 3 个输出头
    branch 3 = prefix + suffix + 3 个输出头

partial-sharing:
    共享 1 份 prefix
    branch 0/1/2/3 各自保留 suffix + 3 个输出头
```

因此 l2 soft 重复保存 4 份 prefix，而 partial-sharing 只保存 1 份；节省的就是这 3 份 prefix 参数。partial-sharing 的 branch 1 到 3 仍然有独立的最后一层和输出头，所以能保留时间专门化能力，但独立表达能力比 l2 soft 弱。

在本次设置下参数量为：original `645130`，l2 soft `2580520`，partial-sharing `1448488`。partial-sharing 约为 l2 soft 的 `56.135%`，但仍约为 original 的 `2.245` 倍。因此它是“保留四个时间专家、共享大部分前端特征提取”的降参方案，而不是把四个专家变成一个。

## 58.44 单 MLP 参数匹配对照实验

为验证 partial-sharing 的收益是否来自时间专家结构，而不是仅仅来自参数量增加，在脚本中新增模式 `mapo_single_mlp_partial_match`。该模式不传 `--use_mapo_all_dynamic`、不创建时间 branch、不做 soft routing，只把原始单个 NonrigidDeformer MLP 的宽度固定为 `794`。

本次 DNA 实验配置：

```text
iterations=25000
densify_until_iter=1800
seed=0
non_rigid_mlp_depth=3
non_rigid_mlp_width=794
单个 MLP，无新增专家、无时间分层、无额外 loss
```

在 `input_ch=223` 的当前 DNA 配置下，宽度 794 的单 MLP 参数量为 `1448266`，partial-sharing 为 `1448488`，两者只相差 `222` 个参数。该实验用于和六序列 partial-sharing 做近似等参数对比：若单 MLP 不能达到 partial-sharing，而 partial-sharing 更好，才能支持收益来自时间划分/专家结构，而不是单纯参数量增大。

## 58.45 参数匹配单 MLP 对比实验完成

为验证 partial-sharing 的收益是否来自时间划分，而不是单纯增加 MLP 参数量，已在六个 DNA 序列上完成 `mapo_single_mlp_partial_match`。该对照不启用 MAPO、不创建时间专家、不进行 soft routing，只把 original 的单个 deformation MLP 宽度设为 `794`。配置为：`iterations=25000`、`densify_until_iter=1800`、`seed=0`、`depth=3`。实验使用 GPU2 和 GPU3，六个序列均完成点云保存、`Training complete` 和 120-view novel-view 评估。

参数量为：

```text
original:                  645130
single MLP width=794:     1448266
partial-sharing:          1448488
single 与 partial 差异:       222
```

六序列最终指标如下，LPIPS 已乘以 1000，数值保持 JSON 原始精度。

| 序列 | original PSNR | original SSIM | original LPIPS*1000 | single MLP PSNR | single MLP SSIM | single MLP LPIPS*1000 | partial-sharing PSNR | partial-sharing SSIM | partial-sharing LPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.531332969665527 | 0.9584373275438944 | 45.01088637237748 | 29.542473951975506 | 0.9584286918242773 | 44.44322182486454 | 29.588104470570883 | 0.9586118206381797 | 44.36599009980758 |
| 0019_10 | 35.257725365956624 | 0.9808832183480263 | 21.035196678712964 | 35.32290760676066 | 0.9812357286612192 | 20.753958992039162 | 35.35039869944254 | 0.9811417390902837 | 20.85616036783904 |
| 0044_11 | 32.98732282320658 | 0.9780931328733762 | 21.27879182808101 | 32.98294825553894 | 0.9780519237120946 | 21.32591335879018 | 32.980789772669475 | 0.9780922984083493 | 21.39106597751379 |
| 0051_09 | 28.66591828664144 | 0.9713883767525355 | 31.05042342407008 | 28.6942893187205 | 0.9716430529952049 | 30.466375996669133 | 28.665311702092488 | 0.9715624819199244 | 31.022088788449764 |
| 0206_04 | 31.31386383374532 | 0.9695543631911278 | 33.811808843165636 | 31.379846874872843 | 0.9699982538819313 | 33.36410384314756 | 31.451732524236043 | 0.970228873193264 | 33.18014269073804 |
| 0813_05 | 36.10658038457235 | 0.9869976962606112 | 18.244266610903047 | 36.037984657287595 | 0.9869209880630175 | 18.331559103292722 | 36.16539355913798 | 0.9872171610593795 | 17.910796799696982 |
| Mean | 32.31045727729797 | 0.974225685828262 | 28.405228959551703 | 32.326741777526 | 0.9743797731896241 | 28.114188853133886 | 32.36695512135824 | 0.9744757290515635 | 28.12104078734087 |

参数匹配单 MLP 相对 original 的均值变化为：PSNR `+0.016284500228032073`，SSIM `+0.00015408736136213186`，LPIPS*1000 `-0.2910401064178174`。partial-sharing 相对该单 MLP 的均值变化为：PSNR `+0.04021334383223518`，SSIM `+0.00009595586193944516`，LPIPS*1000 `+0.006851934206984822`（LPIPS 数值越低越好）。因此，在几乎相同的参数量下，partial-sharing 的 PSNR 和 SSIM 更高，LPIPS 略差 `0.006851934206984822`，整体表现基本持平但偏向 partial-sharing；这支持时间划分/分支结构带来收益，但由于 LPIPS 未同步改善，不能宣称三项指标全面优于单纯增宽 MLP。

本次新实验输出目录为：

```text
output/DNA-Rendering/<sequence>/mapo_single_mlp_partial_match/20260901_single_mlp_partial_match/
```

## 58.46 三组实验指标拆分记录

为便于后续引用，58.45 的三组结果可分别按 original、参数匹配单 MLP 和 partial-sharing 三张六序列表格使用；指标均为 25000 步、120-view novel-view 评估，LPIPS 已乘以 1000，保留 JSON 原始精度。

## 58.47 partial-sharing 相对 original 与 l2 soft 的区别

相对 original，partial-sharing 增加了按 `pose_id` 划分的时间分支：5000 步后划分为两个时间区间，10000 步后划分为四个时间区间，并在区间边界使用 soft blending；同时保留 original 的 Gaussian、损失函数、输入特征和训练设置。它的六序列均值相对 original 为：PSNR `+0.056497844060267255`、SSIM `+0.000250043223301577`、LPIPS*1000 `-0.2841881722108326`。

相对 l2 soft，partial-sharing 保留相同的四分支时间划分、参数继承和边界 soft blending，但共享前两层 MLP，只让各分支独立学习第三层及 warp/rotation/scaling 输出头。因此参数量从 l2 soft 的 `2580520` 降至 `1448488`，减少 `1132032`，约减少 `43.865%`。代价是六序列均值相对 l2 soft：PSNR `-0.08011840449439234`、SSIM `-0.00036394413974527584`、LPIPS*1000 `+0.32771524016021303`；也就是参数效率提高，但精度略低于完整 l2 soft。

## 58.48 partial-sharing 后续优化策略

当前 partial-sharing 的主要限制是：所有 Gaussian 使用相同的 `pose_id` 路由权重；时间边界按帧数等宽固定；相邻分支只在推理输出上做 soft blending，没有显式约束两个分支在边界处的一致性；另外第三层和三个输出头仍然完全独立。建议按以下顺序优化：

### 1. Gaussian-adaptive temporal routing（最推荐）

保留四个时间分支，不增加完整专家，只让每个 Gaussian 根据共享 trunk 特征产生自己的四分支权重：

```text
h_i = shared_trunk(feature_i)
logits_i = router([h_i, pose_embedding])
w_i = softmax(logits_i / temperature_i)
output_i = sum_k w_i,k * branch_k(h_i)
```

这样同一帧中，身体、衣服、袖口和裙摆可以选择不同的时间专家，而不是所有点都按照同一个 `pose_id` 权重。router 应保持很小，并对 logits 使用温度或熵正则，避免所有 Gaussian 都退化为同一分支或平均使用四个分支。该方案仍然属于 partial-sharing 的时间专家路线，区别在于从“整帧路由”变成“Gaussian 级路由”。

### 2. 基于动作变化的非等宽时间区间

不要直接把 100 帧固定切成 `0-24/25-49/50-74/75-99`。先用 SMPL/LBS 顶点或关节轨迹计算每一帧的运动变化量，例如速度、加速度和速度变化率，再用变化峰值确定 3 个边界。动作变化快的区间分配更窄的 expert window，变化平缓的区间分配更宽的 window。

这不是 MAPO 的“根据高动态 Gaussian 递归复制时间分支”：这里仍然是序列级固定 4 个 branch，只改变时间边界，使 branch 与真实动作阶段对齐。应先离线计算边界再训练，避免训练过程中改变路由导致实验不稳定。

### 3. 相邻 branch 的 boundary consistency loss

在每个时间边界附近，对相邻 branch 使用同一批 Gaussian 特征，增加小权重一致性约束：

```text
L_boundary = |d_xyz_left - d_xyz_right|
           + beta_r * |d_rotation_left - d_rotation_right|
           + beta_s * |d_scaling_left - d_scaling_right|
```

只在边界附近计算，且 `beta_r`、`beta_s` 小于位移项，避免把衣服褶皱过度抹平。soft blending 只是最终输出混合，不能保证两个 branch 本身相近；这个 loss 可以减少分支切换处的局部不连续。需要用单独消融确认收益来自一致性约束，而不是额外正则。

### 4. 按输出类型选择性共享

当前四个 branch 都独立预测 warp、rotation 和 scaling。可以先只让最需要时间专门化的 `warp` 保持 branch-specific，而共享 rotation/scaling head；或者反过来只对高运动点开放独立 scaling。这样利用“位移变化通常比尺度和旋转更需要时间自由度”的先验，既能降参也能减少不必要的分支噪声。该方案适合做结构消融：`all-head independent`、`warp-only independent`、`warp+rotation independent`。

### 5. 改进 branch 激活和优化状态

当前 5000/10000 步激活新分支后会复制父分支参数并清空新参数的 Adam 状态，但四分支只剩最后 15000 步训练。可以在激活后的短 warmup 内逐渐增大新分支路由权重，同时让父分支保持一部分权重；或者给新分支使用短暂的 learning-rate warmup。这个方向主要解决“新 branch 学习时间不足”和激活瞬间扰动，不改变最终参数量，风险低但创新性弱于 Gaussian-adaptive routing。

### 6. 低秩 branch-specific suffix（压缩方向）

如果仍需降低参数，可将独立第三层改成共享权重加低秩更新：

```text
W_k = W_shared + U_k V_k^T
```

只给每个时间 branch 保存小秩 `U_k,V_k`，输出头也可采用同样方式。它比完全独立 suffix 更省参数，但属于参数压缩，不能单独作为主要创新；必须和参数匹配单 MLP、partial-sharing 做公平对比。

### 推荐实验顺序

先做 `Gaussian-adaptive routing`，因为它直接解决当前所有 Gaussian 共用时间权重的问题；然后做“动作变化边界”和 `boundary consistency loss` 的独立消融。最后再考虑 selective head sharing 或低秩 suffix。每次保持 `iterations=25000`、`DENSIFY_UNTIL_ITER=1800`、seed、branch 数和原始损失不变，并报告参数量与六序列指标。

## 58.49 partial-sharing 与 part_moe_leg 的组合方案

### 当前代码的直接组合问题

当前 `NonrigidDeformer.forward()` 中，Part-MoE 分支位于 MAPO/partial-sharing 分支之前，并在得到 Part-MoE 输出后直接 `return`。因此仅仅同时传入 `--use_part_moe --use_mapo_all_dynamic --mapo_partial_sharing`，实际不会得到两个模块串联，而是只执行 Part-MoE，partial-sharing 的 temporal branch 不会参与最终输出。

另外，Part-MoE 在 `part_moe_start_iter=10000` 初始化后会调用 `freeze_shared_after_part_moe()`。如果直接沿用这个行为，partial-sharing 的共享 trunk 和 root 输出头可能被冻结，破坏 temporal branch 的继续训练。因此组合模式必须使用独立的 forward 和 optimizer/freeze 逻辑。

### 推荐方案：Temporal-Part Factorized Residual MoE

推荐先实现“时间主分支 + 部件残差校正”：

```text
features
   -> shared trunk
   -> temporal branch k                  -> d_temporal
   -> small part residual head(label)    -> r_part

d_final = d_temporal + alpha_part * r_part
```

具体做法：

1. 继续使用 partial-sharing 的共享前两层、四个 temporal branch、5000/10000 步分层和 boundary soft blending。
2. temporal branch 先产生当前 Gaussian 的 hidden/output；part residual head 接收当前 temporal hidden，因此部件校正仍然可以随时间 branch 改变。
3. 为 `part_moe_leg` 的 7 个标签建立轻量 residual head：`part_0` 作为 global/unknown residual，`part_1` 到 `part_6` 对应人体部件。每个 head 只预测 warp、rotation、scaling 的残差，不复制完整 MLP。
4. 对已知部件使用对应 residual；未知部件使用 `part_0`。保留 `part_moe_global_keep=0.1`，也就是最终 part correction 中保留一小部分 global correction。
5. residual head 零初始化，保证 10000 步刚启用时 `d_final` 等于 partial-sharing 的输出；part 权重从 0 在 1000 步内 warm up 到 `1-global_keep`。
6. 不冻结 shared trunk、temporal suffix 或 temporal output head；所有 temporal 参数和 residual head 参数都加入同一个 `mlp_optimizer`。

该方案的关键不是让 part expert 重新预测完整形变，而是让 temporal branch 负责“这一时刻怎么动”，让 part head 负责“这个部件需要怎样修正”。因此它避免了两个完整预测器互相竞争，也避免了 4 temporal x 7 part 的笛卡尔积专家。

### 参数和对比关系

当前 partial-sharing 的 NonrigidDeformer 约为 `1448488` 参数。若直接追加现有 7 个 `PartNonrigidExpert`，会额外复制 prefix 和输出头，组合模型约达到 `4.1M` 量级，且 part expert 不接收 temporal branch hidden。推荐的 residual head 只增加约 `7 x (3+4+3) x 512` 级别的输出参数和少量路由参数，整体仍接近 partial-sharing，组合收益更容易归因。

### 推荐的公平实验矩阵

第一轮应至少包含以下四组，全部使用 `iterations=25000`、`densify_until_iter=1800`、`seed=0`、相同输入和原始损失：

```text
original
part_moe_leg
partial-sharing
partial-sharing + temporal-part residual
```

组合模式使用：

```text
mapo level 1: 5000
mapo level 2: 10000
part label/expert activation: 10000
part residual warmup: 1000 steps
mapo branches: 4
part labels: part_moe_leg, num_parts=7
```

10000 步同时发生 temporal level-2 激活和 part residual 启用，必须记录激活前后 loss、part residual norm 和 temporal/part 输出范数，确认没有突然爆炸。六序列正式实验中使用独立实验名和输出目录，不改变 original、partial-sharing 或 part_moe_leg 的现有模式。

### 可选的直接模块融合对照

如果需要证明“残差式组合”优于简单加权，可以另做一个高参数量对照：分别计算 partial-sharing 的 `d_temporal` 和现有 Part-MoE 的 `d_part`，再使用：

```text
d_final = d_temporal + alpha_part * (d_part - d_global)
```

这里减去 `d_global` 是为了只加入部件相对于 global expert 的修正，避免重复叠加两套完整形变。这个版本更接近原始 `part_moe_leg`，但参数量和显存明显更高，应作为 fusion ablation，不建议作为最终主方案。

不建议第一版做 28 个 `(temporal branch, part)` 独立专家：它会显著增加参数和显存，且很难判断收益来自时间建模、部件建模还是单纯容量增加。

## 58.50 不使用 part residual 的组合方案

如果不希望 Part expert 只预测部件残差，推荐使用 `Temporal-Conditioned Part-MoE`：Part expert 仍然直接预测完整的 `d_xyz`、`d_rotation` 和 `d_scaling`，partial-sharing temporal branch 只负责提供当前时间下的隐表示和路由权重。

### 推荐结构：时间条件化的完整部件专家

```text
raw features
    -> shared trunk H
    -> temporal suffix S_k(H) = h_k
    -> full part expert P_p(h_k)
    -> complete deformation output
```

其中 `k` 是 4 个时间 branch，`p` 是 `part_moe_leg` 的 7 个部件标签。每一个 `P_p` 都完整输出绝对形变，而不是输出 residual：

```text
d_{k,p} = P_p(h_k)
d_final = sum_k temporal_weight_k * d_{k,p}
```

时间边界附近仍使用 partial-sharing 的 soft blending，但融合的是同一个部件专家在相邻时间 hidden 上产生的两个完整预测。已知部件选择对应的 `P_p`，unknown 部件使用 `P_0`；如果保留 `part_moe_global_keep=0.1`，已知部件可使用 `0.1 * P_0 + 0.9 * P_p`。

这个设计的含义是：时间 branch 决定“当前属于哪种时间形变状态”，Part expert 决定“这个部件在该状态下的完整形变”。Part expert 不是对 temporal 输出做小修正，而是从时间条件重新预测完整形变。

### 为什么不需要 28 个专家

不建立 `P_{k,p}` 这 28 个独立网络，而是建立 4 个 temporal suffix 和 7 个 part predictor：

```text
4 temporal suffixes
        x
7 shared-across-time full part experts
```

7 个 Part expert 在所有 temporal branch 之间复用，因此时间和部件形成因子化组合，而不是为每一个时间区间复制一套部件 MLP。实现时可让 `TemporalPartialExpert` 返回 `h_k`，再由 `TemporalConditionedPartExpert` 接收 `h_k` 并输出完整形变；该组合模式不使用旧的 temporal output head，避免重复计算两套绝对输出。

### 三种可选组合方式

1. `Temporal-Conditioned Part-MoE`：推荐方案。Part expert 完整预测，temporal hidden 作为条件，组合关系最清晰。
2. `Full-output gated fusion`：分别得到 partial-sharing 的完整输出 `d_time` 和 part_moe_leg 的完整输出 `d_part`，再使用 `d = g*d_time + (1-g)*d_part`。实现简单，但两个模块互相看不到对方的隐特征，更多是输出集成。
3. `Part-conditioned temporal expert`：先由 part embedding 调制 temporal suffix，再由 temporal branch 完整输出。参数更省，但 Part-MoE 的独立完整专家含义会变弱。

不建议直接使用 `d_time + d_part`，因为两者都是绝对形变预测，直接相加会重复计算形变并可能导致位移、旋转和尺度幅值失控。若使用现有两个完整 predictor，至少应采用凸组合；若希望真正形成一个联合模型，则使用第一种时间条件化结构。

### 训练与公平设置

```text
0-4999:   一个 temporal branch
5000:     两个 temporal branch
10000:    四个 temporal branch，同时启用 part label routing
part warmup: 1000 steps
iterations=25000
densify_until_iter=1800
seed=0
num_parts=7
```

10000 步时不要冻结 partial-sharing 的 shared trunk 或 temporal suffix。Part expert 参数应加入 `mlp_optimizer`，新模式不要调用原 Part-MoE 的 `freeze_shared_after_part_moe()`。第一轮建议比较 `original`、`part_moe_leg`、`partial-sharing` 和 `Temporal-Conditioned Part-MoE`，同时报告完整参数量、10000 步前后 loss 和六序列 PSNR/SSIM/LPIPS*1000。

## 58.52 时间条件化完整 Part-MoE 的直观解释

该方案已经实现并完成六序列实验。它不是同时计算 28 个独立网络，而是把“时间”和“部件”拆成两步：先由 temporal branch 生成当前时间状态，再由部件专家根据这个状态完整预测形变。

对一个 Gaussian，可以理解为：

```text
原始特征
  -> shared trunk：提取通用特征
  -> temporal branch：提取当前时间状态 h_k
  -> 当前 part 对应的完整 Part expert：输出完整形变
```

例如某个 Gaussian 属于左臂，当前 pose 位于 branch 0 和 branch 1 的边界，则只使用同一个 `PartExpert_left_arm` 两次：一次输入 `h_0`，一次输入 `h_1`，最后按时间 soft 权重融合两次完整输出。不是建立 `PartExpert_branch0_left_arm` 和 `PartExpert_branch1_left_arm` 两个独立网络。

因此实际组合关系是：

```text
4 个 temporal suffix
        -> 产生 4 种时间状态
7 个 part predictor
        -> 被不同 temporal state 复用
```

数学上是先得到 `h_k=S_k(H)`，再得到 `d_{k,p}=P_p(h_k)`，最后计算 `d=sum_k w_k*d_{k,p}`。这里 `P_p` 对同一个部件在所有时间 branch 中共享参数，所以是“因子化组合”，不是 28 个完全独立的 `P_{k,p}`。

Part expert 仍然直接输出完整 `d_xyz/d_rotation/d_scaling`，不是 residual，也不是把 temporal 输出和 part 输出直接相加。时间 branch 负责提供时间条件，part expert 负责在该条件下重新预测完整形变。

## 58.51 partial-sharing 中“前两层”的含义

这里的“前两层”指 deformation MLP 的网络层，不是 10000 步以前的时间阶段。当前 `D=3` 时，MLP 结构可写成：

```text
Linear(input, 512) -> ReLU       第 1 层
Linear(512, 512)  -> ReLU        第 2 层
Linear(512, 512)  -> ReLU        第 3 层
```

partial-sharing 在模型初始化时把第 1 层和第 2 层作为 shared trunk，把第 3 层作为每个时间分支的独立 suffix；warp、rotation、scaling 三个输出头也保持分支独立。也就是说，从训练一开始 shared trunk 就只有一份，并不是 10000 步之前共享、10000 步之后才共享。

时间步只决定分支何时启用：

```text
0-4999:   只使用 root：shared trunk -> root suffix -> root heads
5000:     启用第二个时间分支：shared trunk -> branch_1 suffix -> branch_1 heads
10000:    启用四个时间分支，对应 0-24、25-49、50-74、75-99 帧
```

因此可以把 partial-sharing 理解为：所有时间分支先共用同一套基础特征提取器，再在 MLP 后端用不同 suffix/head 学习时间差异；5000 和 10000 步是时间分支的激活时刻，不是 shared trunk 的切换时刻。

## 58.54 part_moe_leg 与 l2 soft 六序列指标

本次整理使用最终 novel-view 25000 指标，采用 seed 0 主实验：
```text
part_moe_leg: 20260827_dna6_part_moe_leg_1800_tmux
l2 soft:      mapo_l2_soft_* 主实验目录（非 seed1 重跑）
```

六序列结果：
```text
0007_04:
    part_moe_leg PSNR 29.60400875409444, SSIM 0.959126636882623, LPIPS*1000 43.3049468168368
    l2 soft      PSNR 29.58473817507426, SSIM 0.958546108007431, LPIPS*1000 44.202796959628664
0019_10:
    part_moe_leg PSNR 35.41057926813761, SSIM 0.9814167340596517, LPIPS*1000 20.47669793634365
    l2 soft      PSNR 35.41862014134725, SSIM 0.9814029360810915, LPIPS*1000 20.801884145475923
0044_11:
    part_moe_leg PSNR 32.98199820518494, SSIM 0.9782939051588376, LPIPS*1000 20.875872555188836
    l2 soft      PSNR 33.00627454121908, SSIM 0.9783195222417513, LPIPS*1000 21.126633618647854
0051_09:
    part_moe_leg PSNR 28.71769100824992, SSIM 0.9719194740056991, LPIPS*1000 30.67279694757114
    l2 soft      PSNR 28.823214197158812, SSIM 0.9720930164059003, LPIPS*1000 30.348760883013408
0206_04:
    part_moe_leg PSNR 31.53526193300883, SSIM 0.9707902779181798, LPIPS*1000 32.71898709548016
    l2 soft      PSNR 31.633441146214803, SSIM 0.971306781967481, LPIPS*1000 32.46326390653849
0813_05:
    part_moe_leg PSNR 36.24788769086202, SSIM 0.9874844372272491, LPIPS*1000 17.52235199480007
    l2 soft      PSNR 36.21615295410156, SSIM 0.9873696744441985, LPIPS*1000 17.816613769779603

mean:
    part_moe_leg PSNR 32.41623780992296, SSIM 0.9748385775420401, LPIPS*1000 27.59527555770344
    l2 soft      PSNR 32.44707352585263, SSIM 0.9748396731913088, LPIPS*1000 27.793325547180658
```

## 58.53 Temporal-Conditioned Full Part-MoE 低于单独 l2 soft 的原因

六序列均值：
```text
l2_soft:
    PSNR 32.44707352585263
    SSIM 0.9748396731913088
    LPIPS*1000 27.793325547180658

Temporal-Conditioned Full Part-MoE:
    PSNR 32.36527712874942
    SSIM 0.9744901013871035
    LPIPS*1000 28.120668342388754
```

当前组合相对 l2_soft：
```text
PSNR -0.0817963971032114
SSIM -0.00034957180420536194
LPIPS*1000 +0.3273427952080965
```

代码层面的主要原因：
1. 当前组合继承的是 `partial-sharing`，不是 l2_soft 的四个完整 MLP。l2_soft 每个时间 branch 都有完整的 MLP、输出头；当前组合只让 temporal branch 保留 suffix/head，再额外加入跨时间共享的 part predictor。参数变少后，时间分支表达能力也变弱，因此不能直接期待超过完整四分支 l2_soft。
2. Part expert 不是在 temporal 输出上做一个小修正，而是对 `temporal_hidden` 重新输出完整 `d_xyz/d_rotation/d_scaling`。在 10000 步后的 warmup 中，最终输出从 temporal predictor 逐渐切换到 part predictor，这会改变原本已经学好的 l2_soft 输出分布，产生优化扰动。
3. `part_gate` 进入饱和后，已知部件的输出主要是 `0.9 * part expert + 0.1 * global expert`。这等价于大幅替换时间专家的预测，而不是保留两个模块各自的有效结果；时间专家的优点可能被 Part expert 覆盖。
4. 时间权重通常只有一个 branch 为 1，只有靠近 25/50/75 帧边界时才 soft blend。因此一次训练大多数时候只给当前时间 branch 梯度，4 个时间 branch 与共享 Part expert 并没有持续获得跨时间协同训练信号。
5. Part label 是由 Gaussian 到 SMPL 的映射得到的伪标签，不完全等于衣服/身体真实语义。错误映射会把 Gaussian 送入不合适的 part expert；l2_soft 不依赖这个标签，所以没有这类风险。
6. 当前 Part expert 用复制的 global output head 初始化，并在输入后增加了 `Linear + ReLU`。这个路径不是严格的恒等初始化，激活时即使 `part_gate` 很小，也可能改变输出分布。

因此这次结果不是说明两个想法理论上不能结合，而是说明当前组合方式存在容量压缩、输出替换、伪标签噪声和梯度稀疏几个问题。它仍略高于 original，但基本与 partial-sharing 持平，不能作为优于 l2_soft 的结果。

## 12. 2026-09-02 Part-MoE + L2 Soft Confidence Fusion

本次实验要求：
- `part_moe_leg + l2 soft`
- Part expert 与 temporal output 并行预测，再按置信度融合
- 低置信度 Gaussian 保留 temporal 分支输出
- `unknown Gaussian` 保留 temporal 分支输出

融合逻辑：
```text
confidence_mix = clamp((part_conf - 0.5) / (1 - 0.5), 0, 1)
known_mask = (part_label > 0)
confidence_gate = part_gate * confidence_mix * known_mask
final = temporal_output + confidence_gate * (part_output - temporal_output)
```

正式公平配置：
```text
iterations=25000
densify_until_iter=1800
seed=0
mapo_max_partition_level=2       # 4 temporal branches
mapo_soft_routing=true
mapo_partial_sharing=true
part_label_schema=part_moe_leg
num_parts=7
temporal_conditioned_part_conf_threshold=0.5
```

正式输出路径：
```text
output/DNA-Rendering/0007_04/mapo_temporal_conditioned_part_confidence/20260902_tcpm_conf/metrics/results_novelview_25000.json
output/DNA-Rendering/0019_10/mapo_temporal_conditioned_part_confidence/20260902_tcpm_conf/metrics/results_novelview_25000.json
output/DNA-Rendering/0044_11/mapo_temporal_conditioned_part_confidence/20260902_tcpm_conf_retry0044b/metrics/results_novelview_25000.json
output/DNA-Rendering/0051_09/mapo_temporal_conditioned_part_confidence/20260902_tcpm_conf/metrics/results_novelview_25000.json
output/DNA-Rendering/0206_04/mapo_temporal_conditioned_part_confidence/20260902_tcpm_conf_retry0206e/metrics/results_novelview_25000.json
output/DNA-Rendering/0813_05/mapo_temporal_conditioned_part_confidence/20260902_tcpm_conf/metrics/results_novelview_25000.json
```

六序列 novel-view 25000 指标：
```text
0007_04: PSNR 29.61887396176656, SSIM 0.9586502522230148, LPIPS*1000 44.20834138679008
0019_10: PSNR 35.34582462310791,  SSIM 0.9811905140678088, LPIPS*1000 20.929016390194496
0044_11: PSNR 32.955779536565146, SSIM 0.9781195312738419, LPIPS*1000 21.333993629862866
0051_09: PSNR 28.696412801742554, SSIM 0.9716278279821078, LPIPS*1000 30.891177717906733
0206_04: PSNR 31.498931709925333, SSIM 0.9707442184289297, LPIPS*1000 32.85510271477202
0813_05: PSNR 36.170053768157956, SSIM 0.9871846626202265, LPIPS*1000 18.00378963040809

mean: PSNR 32.38097940021090983333333333, SSIM 0.97458616776598825, LPIPS*1000 28.03690357832238083333333333
```

0206 重试记录：
- CUDA 图像配置的 `retry0206`、`retry0206b`、`retry0206c`、`retry0206d` 在约 4100 step 触发 `CUBLAS_STATUS_EXECUTION_FAILED`，不作为结果。
- 使用 `IMAGE_DATA_DEVICE=cpu` 和 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后，`retry0206e` 完成 25000 step 训练、checkpoint 保存和 120 个 novel-view 渲染。
- `retry0206e` 日志确认 `Training complete`，最终点云为 `point_cloud/iteration_25000/point_cloud.ply`。

## 13. Current Confidence Fusion Compared with Original / Part-MoE / L2 Soft

### 当前实验的实际设置

当前实验使用的是：
```text
SMPL/LBS canonical Gaussian initialization
4 temporal branches, 100 pose frames
level 0: 1 branch
5000 step: 2 branches, intervals 0-49 and 50-99
10000 step: 4 branches, intervals 0-24, 25-49, 50-74, 75-99
soft routing width=4 pose ids
shared temporal MLP prefix + independent temporal suffixes/heads
7 part predictors shared across all temporal branches
part label schema=part_moe_leg
part expert activation=10000 step
confidence threshold=0.5
```

代码对应关系：
- `nets/mlp_delta_non_rigid.py:1678-1685` 将 `D=3` 的 deformation MLP 拆成共享前缀和独立后缀。前缀是前两层 `Linear+ReLU`，后缀是第三层 `Linear+ReLU`。
- `nets/mlp_delta_non_rigid.py:1695-1704` 建立 7 个 `TemporalConditionedFullPartExpert`，这些 Part expert 在时间分支之间复用，不为每个时间段重复创建。
- `nets/mlp_delta_non_rigid.py:2054-2082` 对每个时间 branch 计算 temporal output 和 part output，并使用同一组时间权重分别融合。
- `nets/mlp_delta_non_rigid.py:2102-2109` 使用 part confidence 计算门控：置信度低于 0.5 时门控为 0，unknown label (`part_label=0`) 也强制为 0。
- `train.py:712-725` 让 Part-MoE 的最大作用权重从 10000 step 后经过 1000 step warmup 增长到 `1-global_keep=0.9`。
- `ablations/part_moe_controller.py:78-103` 用 Gaussian 到 SMPL-X canonical 顶点的最近距离生成 part label 和 confidence：`confidence=1-distance/0.08`，超出距离阈值的 Gaussian 标记为 unknown、confidence 为 0。

对一个 Gaussian，当前路径可以简化为：
```text
输入特征 H
  -> 共享前两层
  -> 当前时间 branch 的第三层和输出头
  -> temporal prediction

同一个 temporal hidden
  -> 当前 part 对应的 Part expert
  -> part prediction

最终输出 = temporal prediction
           + confidence_gate * (part prediction - temporal prediction)
```

因此它不是 28 个独立网络，也不是直接把 Part expert 替换 temporal expert，而是“时间条件 + 部件条件”的因子化组合。

### 与 original 的区别

`original`：
```text
1 个 shared deformation MLP
1 套 warp/rotation/scaling 输出头
不做时间分支
不做 part label 路由
不做 confidence fusion
```

当前实验在 original 基础上增加了三件事：
1. **时间分支**：5000 和 10000 step 分阶段激活 branch，把原来一套 MLP 拆成不同 pose 区间建模；时间边界附近用相邻 branch 的 smoothstep 权重混合。
2. **Part predictor**：建立 7 个部件 predictor，使 body、hand、face、leg/foot 等部件可以学习不同的完整形变输出。
3. **置信度保护**：Part label 可靠且 confidence 高时才逐渐使用 Part 输出；低置信度和 unknown Gaussian 继续使用 temporal 输出。

共同设置方面，四者都保留 SMPL/LBS 初始化、Gaussian 训练、RGB/Mask/SSIM/LPIPS 主损失和 `densify_until_iter=1800`；当前正式实验使用 `iterations=25000`、`seed=0`。

### 与 part_moe_leg 的区别

`part_moe_leg` 的核心流程是：
```text
0-9999: 1 个 shared deformation MLP
10000: 生成 7 类 Gaussian part label
10000: 从 shared MLP 深拷贝出 7 个完整 Part expert
之后冻结原 shared branch
已知 part: 0.1 * global/unknown expert + 0.9 * 对应 part expert
unknown: 使用 global/unknown expert
```

当前实验相比它增加/改变了：
- **增加 temporal hierarchy**：Part-MoE 之前和之后都建立在 4 个 temporal branch 上，时间区间为 25 帧一级；`part_moe_leg` 只有一套时间无关的 deformation MLP。
- **Part expert 的输入变了**：当前 Part expert 接收某个 temporal branch 产生的 `temporal_hidden`，所以同一 Part expert 可以根据不同时间状态输出不同形变；普通 `part_moe_leg` 直接接收 shared MLP 输入并独立预测。
- **融合策略变了**：普通 `part_moe_leg` 对已知 part 直接采用固定的 `global_keep=0.1` 与 `part_weight=0.9`；当前实验进一步乘上 Gaussian confidence gate。低置信度时不让 part 输出接管。
- **unknown 处理变了**：普通版本 unknown 走 global/unknown expert；当前 confidence 版本的 `known_mask` 让 unknown 保留 temporal output。
- **共享参数策略变了**：普通版本 10000 step 时复制完整 shared MLP，并冻结原 shared 分支；当前版本要求 temporal suffix 继续可训练，Part predictor 跨时间共享。

所以当前方法不是“part_moe_leg 加一个置信度数值”这么简单，而是把 Part-MoE 放到了 temporal branch 之后，并改变了 unknown/低置信度 Gaussian 的路由。

### 与 l2 soft 的区别

`l2 soft` 的实际结构是：
```text
root: 1 个完整 deformation MLP + 3 个完整 temporal branch
5000 step: 激活 2 branches
10000 step: 激活 4 branches
每个 branch 都有自己的完整 MLP 和 warp/rotation/scaling heads
边界附近仅对相邻 temporal branch 的输出做 soft blending
没有 part label、Part expert 和 confidence gate
```

当前实验相对 l2 soft 的主要区别：
- l2 soft 的 4 个 temporal branch 都保留完整的三层 MLP；当前实验共享前两层，每个 temporal branch 只保留第三层和输出头，因此 temporal 参数更少。
- 当前实验新增 7 个跨时间复用的完整 Part predictor；l2 soft 没有这部分参数。
- l2 soft 的输出完全由 temporal branch 决定；当前实验先得到 temporal fused 和 part fused，再由 confidence gate 在二者之间插值。
- l2 soft 对所有 Gaussian 使用同一时间路由；当前实验在时间路由之外还使用 Gaussian 的 pseudo-part label 和 SMPL 距离置信度。

需要特别注意：当前实验并不是参数量一定低于 l2 soft。按 `input_ch=223, W=512, D=3` 的结构估算：
```text
original 单个 deformation 网络约 645130 个参数
l2 soft 约 4 * 645130 = 2580520 个 deformation 参数
当前实验约 1448488 个 temporal/shared 参数
        + 7 * 267786 个 Part predictor 参数
        = 3322990 个 deformation 参数
```

因此当前 confidence fusion 的总 deformation 参数量反而约为 l2 soft 的 `1.29` 倍，不能把它称作严格的轻量化版本。它减少的是“时间分支的重复 MLP”，但新增的 7 个 Part predictor 又带回了较多参数。

### 公平性与结果含义

本次四个方法的训练步数和 Gaussian densification 上限统一为 `25000/1800`，但网络容量不完全匹配：
- `original` 是单 MLP；
- `part_moe_leg` 增加 7 个完整 Part expert；
- `l2 soft` 增加 4 套完整 temporal deformation 网络；
- 当前 confidence fusion 同时有 temporal partial-sharing 和 7 个 Part predictor。

所以这些实验可以作为“模块效果对比”，但不能单独证明当前方法优于其他方法且完全不依赖参数量。若论文要做严格归因，还需要报告 trainable/total deformation 参数量，并加入参数量匹配的 baseline。

当前六序列均值为：
```text
current confidence fusion:
    PSNR 32.38097940021090983333333333
    SSIM 0.97458616776598825
    LPIPS*1000 28.03690357832238083333333333

l2 soft:
    PSNR 32.44707352585263
    SSIM 0.9748396731913088
    LPIPS*1000 27.793325547180658
```

按当前记录，confidence fusion 仍低于 l2 soft，说明“避免错误 Part label”在机制上更稳妥，但现有 Part predictor/partial-sharing 的组合没有把 l2 soft 的完整时间分支能力保留下来。下一步应优先做：
1. 参数量匹配的 temporal-only 对照；
2. `confidence gate` 的 active ratio 和各 part 的错误率统计；
3. 只在高置信度 Gaussian 上训练/使用 Part predictor，避免大量低质量 pseudo-label 干扰；
4. 保留 temporal full branch 能力后，再测试 confidence fusion 是否真正带来收益。

## 11. 2026-09-03 Part-MoE 标签/流程错误风险审计

审计范围：`part_moe_leg` 的标签生成、训练激活、路由、checkpoint/render 加载和 DNA 六序列统计。

### 结论

- Python、shell 和 diff 检查通过：`compileall_ok`、`bash_syntax_ok`、`diff_check_ok`。
- 没有发现当前 `densify_until_iter=1800` 下标签数组因增删点错位的问题。代码要求 Part-MoE 的 densify 上限不超过 `part_moe_start_iter`，当前配置为 `1800 < 10000`。
- 但当前没有 Part ground-truth，因此不能报告真实分类错误率；以下是“标签不确定性/误分类风险比例”，不是 confusion matrix。

### 高风险点

1. **标签是单最近邻硬分配**。`part_moe_controller.py:78-95` 只用 Gaussian 到最近 SMPL-X 顶点的 `k=1` 查询，并继承该顶点的 dominant LBS label，没有多邻居投票、多帧投票、空间平滑或图像误差校正。
2. **confidence 不是分类正确率**。当前 `conf=1-distance/0.08` 只表示接近 SMPL 表面的程度；衣服点即使贴近身体，也可能被错误分到 body/leg，但仍获得较高 confidence。
3. **普通 `part_moe_leg` 没有用 `part_conf` 做防错路由**。`gaussian_renderer` 会传入 `part_conf`，但普通路径在 `nets/mlp_delta_non_rigid.py:3346-3351` 调 `forward_part_moe` 时没有传入它。普通路由随后按 `part_label` 直接选择对应 expert；`part_conf` 主要只在 tri/token/score-route 等扩展路径生效。
4. **标签只在 10000 step 生成一次**。`part_moe_controller.py:30-65` 在 start step 保存并加载一次标签，而 Gaussian canonical xyz 在之后仍继续被 RGB/Mask 等损失更新，所以后续存在 stale label 风险。
5. **没有 cloth 类别**。`part_moe_leg` 只有 `unknown/body/hands/face/left_leg_foot/right_leg_foot`。衣物只能被映射到附近身体部件或 unknown，衣服与身体相邻时的错误路由风险最高。
6. **所谓 vote 文件目前不是投票结果**。`part_moe_controller.py:96` 初始化全零矩阵后直接保存，不能用于判断标签一致性。

### 六序列风险统计

来源：`output/DNA-Rendering/*/part_moe_leg/20260827_dna6_part_moe_leg_1800_tmux/part_labels/iteration_10000/`。

| 序列 | Gaussian 数 | unknown | conf<0.5 | 已知且 conf<0.5 | 已知点平均 conf |
|---|---:|---:|---:|---:|---:|
| 0007_04 | 35362 | 0.166506 | 0.408885 | 0.242379 | 0.639697 |
| 0019_10 | 43194 | 0.021091 | 0.086193 | 0.065102 | 0.772417 |
| 0044_11 | 84205 | 0.073012 | 0.206817 | 0.133804 | 0.723388 |
| 0051_09 | 68725 | 0.121339 | 0.279214 | 0.157876 | 0.695080 |
| 0206_04 | 59182 | 0.085904 | 0.256463 | 0.170559 | 0.698945 |
| 0813_05 | 50898 | 0.005973 | 0.090554 | 0.084581 | 0.753679 |

风险最高的是 `0007_04`，其次是 `0051_09`、`0206_04` 和 `0044_11`；`0019_10`、`0813_05` 相对稳定。腿部点的距离置信度也普遍偏低，说明左右腿边界和 SMPL 外层点是重点风险区域。

### 当前机制能做什么

- unknown expert、`part_moe_global_keep=0.1` 和扩展版 score-route 可以降低错误标签对输出的破坏。
- 它们不改变标签本身，也不能证明标签错误率下降。
- 普通 `part_moe_leg` 的最主要防线仍是 global expert 的约 10% 混合和 unknown label 回退，保护力度有限。

### 推荐修复顺序

1. 用 SMPL 顶点 `k=8/16` 的距离加权投票替代 `k=1`，并保存真实 vote/count。
2. 用 top-1/top-2 label margin 加距离共同计算 confidence；低 margin 直接 soft route 或回退 global。
3. 对 Gaussian 空间邻域做 label smoothing，并统计邻域不一致率。
4. 10000 step 后定期重算标签，或冻结用于标签关联的 canonical 坐标并记录 label flip 数。
5. 若目标是衣物，增加 outer-layer/cloth 候选机制，否则单靠 SMPL part prior 无法可靠区分衣服和身体。

## 12. 2026-09-03 修复版组合实验运行记录

实验模式：`mapo_temporal_conditioned_part_full_confidence`，独立于之前的
VGGT、DynOMo、point 和其他实验。配置固定为：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
MAPO level 1=5000, level 2=10000, 4 temporal branches
soft temporal routing
part_moe_leg, 7 parts, robust kNN=8
Part label refresh=5000
confidence-gated Part fusion, threshold=0.5
```

本次修复包括：
- 主 forward 同时识别 `part_moe_active` 和
  `temporal_conditioned_part_active`，确保组合路径实际执行。
- render 加载 checkpoint 后显式恢复 temporal-conditioned Part predictor。
- temporal soft routing 只计算权重非零的 branch，边界最多计算相邻两个。
- Part expert 只对 `known && confidence > 0.5` 的 Gaussian 计算；unknown/低置信度
  点直接保留 temporal 输出。
- 修复 subset 路由的 confidence 广播和全局 expert 索引映射错误。

验证：`compileall`、`bash -n`、`git diff --check` 和 CPU forward smoke 均通过。
修复前 v2/v3/v4 分别属于排查过程，不能作为实验结果；v5 才是修复后正式运行。

0044 已完成 v5 训练和 render，正式指标如下：

| 序列 | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | 33.005066108703616 | 0.978235730032126 | 21.268079934331278 |

结果文件：
`output/DNA-Rendering/0044_11/mapo_temporal_conditioned_part_full_confidence/20260903_fullconf_v5/metrics/results_novelview_25000.json`

当前状态：脚本已自动进入 `0051_09`，其余序列仍在排队运行，暂不能据此给出六序列均值或最终优于某一组件的结论。

补充运行状态：0044、0051、0206、0813 已完成 v5 训练和最终 render；原始总脚本在
`0007_04` 训练到约 10650 step 时外部退出，日志没有 Python traceback、CUDA OOM 或
`Training complete`。该不完整目录不计入指标。随后尝试的后台重启 shell 未成功驻留，
因此 0007/0019 需要重新以独立会话启动，配置保持不变。

## 13. 2026-09-03 组合融合上限修复与重跑

对 v5/v6 结果的检查显示，原 confidence fusion 的最大 Part 权重为
`1 - part_moe_global_keep = 0.9`。高置信度 Gaussian 因而几乎完全采用 Part
输出，temporal branch 的有效贡献过小，组合结果退化为 Part-MoE 主导。

代码修复：
- 新增 `temporal_conditioned_part_max_mix`，默认值为 `0.75`。
- confidence gate 改为
  `0.75 * part_gate * confidence_mix * known_mask`。
- unknown 和低置信度 Gaussian 仍保持 temporal 输出；高置信度 Gaussian
  也至少保留 temporal 分支贡献。
- 旧实验目录不覆盖；新实验使用独立 run：
  `20260903_fullconf_cap75`。

检查：
- `compileall` 通过。
- `bash -n scripts/exps_dnarendering.sh` 通过。
- `git diff --check` 通过。
- 新训练命令确认包含 `--temporal_conditioned_part_max_mix 0.75`。

新实验配置保持：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
MAPO level 1=5000, level 2=10000
4 temporal branches, soft routing
part_moe_leg, 7 parts, robust kNN=8
```

运行目录：
`output/DNA-Rendering/<sequence>/mapo_temporal_conditioned_part_full_confidence/20260903_fullconf_cap75/`

当前使用空闲 GPU3 串行运行六个 DNA 序列；GPU0/1/2 当时被其他任务占用，未强行抢占。

当前进度补充：
- `0044_11` cap75 已完成训练和 120-view render：PSNR
  `33.01568686167399`，SSIM `0.9782331516345342`，LPIPS*1000
  `21.137160551734268`。
- `0051_09` cap75 已完成训练和 120-view render：PSNR
  `28.783620325724282`，SSIM `0.9720293283462524`，LPIPS*1000
  `30.721811639765898`。
- `0206_04` cap75 首次运行在 `2770` step 的 backward 处出现
  `CUBLAS_STATUS_EXECUTION_FAILED`，无有效 checkpoint/指标，不计入结果。
- 将使用独立 `retry0206` 目录重跑，并设置
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

## 14. 2026-09-03 cap75 修复版六序列完成结果

本轮继续使用 `mapo_temporal_conditioned_part_full_confidence`，配置为：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
MAPO level 1=5000, level 2=10000, 4 temporal branches, soft routing
part_moe_leg, 7 parts, robust kNN=8
temporal_conditioned_part_max_mix=0.75
```

首次 `0206_04` 在约 2770 step 出现 `CUBLAS_STATUS_EXECUTION_FAILED`，未计入结果。
随后在独立 `retry0206` 目录中使用 `expandable_segments:True` 重跑成功，并通过
10000 step 的组合模块启用阶段。补跑 `0813_05`、`0007_04`、`0019_10` 时确认
脚本使用 `/media/coding/ckx/.conda/envs/seqavatar/bin/python`；此前默认的
`/home/anaconda3/bin/python` 因没有 PyTorch 的启动尝试未进入训练，也不计入结果。

有效结果路径：

```text
0044_11/.../20260903_fullconf_cap75/metrics/results_novelview_25000.json
0051_09/.../20260903_fullconf_cap75/metrics/results_novelview_25000.json
0206_04/.../20260903_fullconf_cap75_retry0206/metrics/results_novelview_25000.json
0813_05/.../20260903_fullconf_cap75_cont0813/metrics/results_novelview_25000.json
0007_04/.../20260903_fullconf_cap75_cont0007/metrics/results_novelview_25000.json
0019_10/.../20260903_fullconf_cap75_cont0019/metrics/results_novelview_25000.json
```

六序列 novel-view 25000 指标：

| 序列 | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | 33.01568686167399 | 0.9782331516345342 | 21.13716055173427 |
| 0051_09 | 28.783620325724282 | 0.9720293283462524 | 30.7218116397659 |
| 0206_04 | 31.605688285827636 | 0.9710775057474772 | 32.52910814868907 |
| 0813_05 | 36.18921160697937 | 0.9873050153255463 | 17.918417408751946 |
| 0007_04 | 29.629452180862426 | 0.9590615123510361 | 43.69541169144213 |
| 0019_10 | 35.3702589670817 | 0.9812435547510783 | 20.87460512605806 |
| **六序列均值** | **32.43231970469157** | **0.9748250113593208** | **27.812752427740232** |

均值对比：

```text
original:      32.31045727729798, 0.9742256858282619, 28.405228959551703
part_moe_leg:  32.41623780992296, 0.9748385775420401, 27.595275557703445
l2_soft:       32.44707352585262, 0.974839673191309,  27.793325547180658
cap75:         32.43231970469157, 0.9748250113593208, 27.812752427740232
```

相对 `original` 的均值变化为：PSNR `+0.12186242739359503`，SSIM
`+0.0005993255310588474`，LPIPS*1000 `-0.5924765318114732`，三项均改善。
相对 `part_moe_leg`，PSNR `+0.016081894768608624`，但 SSIM
`-1.3566182719323047e-05`、LPIPS*1000 `+0.21747687003678587`；因此不能说三项
整体超过该模块。相对 `l2_soft`，PSNR `-0.014753821161058648`、SSIM
`-1.4661831988208954e-05`、LPIPS*1000 `+0.019426880559572385`，整体略差。

结论：cap75 修复确实比 `original` 稳定改善，但没有超过当前最强的 `l2_soft`。
它相对 `part_moe_leg` 只形成 PSNR 的小幅优势，尚不能证明组合模块在整体视觉质量上有效。

## 15. 2026-09-04 cap75 实验身份、归因与公平性说明

这次结果对应 `mapo_temporal_conditioned_part_full_confidence` 的 `cap75` 修复版，
不是新的独立方法，也不是 partial-sharing 版本。实际配置为完整 `l2 soft` 四时间分支，
再叠加 7 个跨时间共享的完整 Part predictor，并用 Part label confidence 在 temporal
输出和 Part 输出之间插值。

核心融合公式为：

```text
confidence_gate = 0.75 * part_warmup_gate
                  * normalized_part_confidence
                  * known_part_mask

output = temporal_output
         + confidence_gate * (part_output - temporal_output)
```

因此：
- unknown 或 confidence 不超过 0.5 的 Gaussian 保留 temporal 输出；
- 高置信度已知 Part Gaussian 才逐步引入 Part 输出；
- 即使 confidence 最高，Part 输出占比上限也是 0.75，不再像旧版那样最高接近 0.9。

这轮变化分为两类：

1. 代码/流程修复：训练 forward 正确进入组合路径、render 恢复组合 active 状态、修复
   subset 索引和 confidence 广播，并只计算当前有非零 temporal 权重的 branch。这些属于
   正确性和显存修复，不是为了刷指标的超参数搜索。
2. 明确调参：把 `temporal_conditioned_part_max_mix` 从旧设计最高约 0.9 限制为 0.75。
   这个数值是看到旧组合被 Part 输出主导后做的工程选择，属于调参；目前只验证了这一档，
   没有 0.5/0.6/0.75/0.9 的完整独立 sweep。

提升归因不能只写成“cap=0.75 带来”。相对 original 的收益同时来自：
- 4 个完整 temporal branch 的时间分段建模；
- 7 个 Part predictor 的额外部件建模容量；
- robust kNN=8、margin-distance confidence 和 unknown/低置信度回退；
- cap=0.75 对错误 Part 路由破坏的抑制。

旧 full-confidence 六序列均值为 `32.41579790910085 / 0.974808772901694 /
27.726357251716163`，cap75 为 `32.43231970469157 / 0.9748250113593208 /
27.812752427740232`。cap75 相对旧版只改善 PSNR 和 SSIM，LPIPS*1000 反而变差，
所以没有证据说 0.75 本身带来三项一致提升。

公平性分两层：
- 训练协议基本公平：六个序列、`iterations=25000`、`densify_until_iter=1800`、
  `seed=0`、原始 MLP width/depth、主要损失和 120-view novel-view 评价一致。
- 模型容量和模块变量不严格公平：original 约 645130 个 deformation 参数，l2 soft
  约 2580520；cap75 在完整 l2 soft 之外又加入约 `7 * 267786` 个 Part predictor，
  合计约 4455022，约为 original 的 6.905 倍、l2 soft 的 1.727 倍。此外 cap75 使用
  robust kNN=8/confidence，而旧 standalone `part_moe_leg` 对照没有完全相同的标签修复。

因此这次可以作为统一训练协议下的模块组合实验，并可确认相对 original 有效果；但不能
作为参数量匹配的严格归因实验，也不能证明提升来自 cap75、Part 路由或时间划分中的某一个。
而且 cap=0.75 是参考同一批序列结果选择的，只跑 seed=0，论文表述时要避免把它当作独立
测试集上的无偏结论。

## 16. 2026-09-04 正式命名为 part_time

用户将 `mapo_temporal_conditioned_part_full_confidence + max_mix=0.75` 正式命名为
`part_time`。运行脚本已新增 `part_time` 模式，后续新实验输出到：

```text
output/DNA-Rendering/<sequence>/part_time/<run_time>/
```

已有六序列结果不移动，仍保留在旧的
`mapo_temporal_conditioned_part_full_confidence/...cap75...` 目录，作为 `part_time`
首轮结果，避免破坏日志和指标路径。

只计算当前权重非零的 1-2 个 temporal branch 不会削弱四分支策略。10000 step 后四个
branch 都实际存在，分别负责 pose/frame id `0-24`、`25-49`、`50-74`、`75-99`。
对某一帧而言，非边界位置本来只有所属 branch 的权重为 1，另外三个权重为 0；边界附近
本来只有相邻两个 branch 权重非零。跳过零权重 branch 与“计算后乘 0”数学等价，也不会
产生梯度。训练采样到不同 pose id 时，四个 branch 会分别被使用和更新。

融合式中的：

```text
Part output - temporal output
```

表示 Part predictor 相对时间 predictor 给出的形变修正量。完整公式：

```text
output = temporal + gate * (part - temporal)
       = (1 - gate) * temporal + gate * part
```

它是普通线性插值，不是额外的负向形变。`gate=0` 时完全使用 temporal，`gate=0.75`
时等价于 25% temporal + 75% Part。

`cap` 是上限（ceiling/maximum cap）。`cap75` 表示 Part 融合权重最大为 0.75，而不是
所有 Gaussian 固定使用 75%。实际 gate 还会乘 warmup、置信度和 known mask；例如
threshold=0.5、confidence=0.75、warmup 已完成时，最终 gate 只有 `0.75*0.5=0.375`。

## 17. 2026-09-04 修复标签版 part_moe_leg 补充实验

为和 `part_time` 的标签处理公平对比，新增独立脚本模式：

```text
part_moe_leg_robust
```

该模式只启用 Part-MoE，不启用 MAPO temporal branch，配置保持：

```text
ITERATIONS=25000
DENSIFY_UNTIL_ITER=1800
SEED=0
part_moe_start_iter=10000
part_moe_warmup=1000
part_moe_global_keep=0.1
part_label_schema=part_moe_leg
num_parts=7
part_label_knn=8
part_label_robust=True
part_confidence_route=True
part_moe_conf_threshold=0.5
```

与旧 `part_moe_leg` 的差别只有标签和路由修复：
- 旧模式 `part_label_robust=False`，控制器会将 kNN 强制为 1，只使用最近 SMPL-X 顶点标签；
- 新模式使用 8 个 SMPL-X 顶点的距离加权投票，并用 top-1/top-2 vote margin 与距离共同计算
  confidence；
- 新模式把 confidence 乘到 Part 路由权重上，unknown 和低置信度 Gaussian 不使用 Part expert，
  保留 global/shared 输出；
- 两者都不加入 temporal branch，因此该实验用于单独测量“标签修复 + confidence route”的影响。

后续六序列结果目录为：

```text
output/DNA-Rendering/<sequence>/part_moe_leg_robust/<run_time>/
```

六序列正式结果（独立 render，iteration 25000；LPIPS 已乘 1000）：

```text
0007_04: PSNR 29.604445314407347, SSIM 0.9587279746929804, LPIPS*1000 43.766784885277346
0019_10: PSNR 35.366111278533936, SSIM 0.981381922463576,  LPIPS*1000 20.493410620838404
0044_11: PSNR 32.99337541262309,  SSIM 0.978214256465435,  LPIPS*1000 21.085442136973143
0051_09: PSNR 28.71457529067993,  SSIM 0.9716885983943939, LPIPS*1000 30.71185276688387
0206_04: PSNR 31.441924953460692, SSIM 0.9702938651045163, LPIPS*1000 32.98469982109964
0813_05: PSNR 36.220157591501874, SSIM 0.9873735765616098, LPIPS*1000 17.580131092108786

mean: PSNR 32.390098306867813, SSIM 0.97461336561375189, LPIPS*1000 27.770386887196864
```

结果路径分别为：

```text
0007_04/part_moe_leg_robust/20260904_part_moe_robust_0007/
0019_10/part_moe_leg_robust/20260904_part_moe_robust_rest/
0044_11/part_moe_leg_robust/20260904_part_moe_robust_parallel/
0051_09/part_moe_leg_robust/20260904_part_moe_robust_remaining/
0206_04/part_moe_leg_robust/20260904_part_moe_robust_remaining/
0813_05/part_moe_leg_robust/20260904_part_moe_robust_remaining/
```

和同为 `iterations=25000`、`densify_until_iter=1800`、`seed=0` 的旧 `part_moe_leg`
均值相比：

```text
part_moe_leg_robust - part_moe_leg:
    Delta PSNR  -0.026139503055148339
    Delta SSIM  -0.00022521192828813449
    Delta LPIPS +0.17511132949342093
```

因此，8-NN robust label + confidence route 在这次六序列结果中没有优于旧
`part_moe_leg`；它相对 original 的均值差为：

```text
Delta PSNR  +0.079641029569838054
Delta SSIM  +0.00038767978549003601
Delta LPIPS -0.63484207235483814
```

这说明 robust 标签修复版仍然能超过 original 的平均结果，但没有证明标签修复本身带来
收益。可能原因包括：confidence route 抑制了部分有效 Part expert、8-NN 投票把衣物点映射
到过于平滑的 body/leg 区域，以及 robust 版和旧版的路由行为不只是标签不同。下一步应
查看 confidence 分布和 known/unknown 比例，并补一个只替换标签、保持旧路由的消融，才能
把 kNN 标签修复与 confidence route 的影响分开归因。

## 2026-09-04 part_moe_leg_robust 下降原因复盘

本次“修复版”低于旧版，不能归因于标签修复单因素，因为实验同时改了三处：

1. 最近邻从 `1-NN` 改为距离加权 `8-NN` 投票；
2. 增加 `confidence = distance_conf * (0.5 + 0.5 * label_margin)`；
3. 增加 confidence route，把 Part 输出按 confidence 与 temporal/global 输出融合。

旧版的主要路由接近：

```text
output = 0.1 * global_output + 0.9 * part_output
```

robust 版实际变成：

```text
confidence_mix = clamp((confidence - 0.5) / 0.5, 0, 1)
effective_part_weight = 0.9 * confidence_mix * known_mask
output = temporal_output + effective_part_weight * (part_output - temporal_output)
```

所以 `confidence=0.6` 时 Part 只保留原来约 20% 的作用，`confidence=0.75` 时只保留约
50%，只有 confidence 接近 1 时才接近旧版的完整 Part 路由。实际有效 confidence 比例在
六个序列约为 `0.58~0.90`，其中 `0007_04` 只有约 `57.95%` 的 Gaussian 通过门控。

此外，`label_margin` 在六序列均值约 `0.97`，几乎不能区分投票是否不确定。因此当前
confidence 主要是在衡量 Gaussian 离 SMPL 表面的距离，而不是标签可靠性；衣物点离 SMPL
较远时会被误判为低置信度并退回 global/temporal 路径。

`8-NN` 也不是严格的部件邻域：它只使用欧氏距离，没有 SMPL 拓扑、法向一致性或部件边界
约束。肩、髋、膝、手脚连接处以及衣物点可能混入相邻部件标签，造成错误的 Part expert
选择。结果更像是“有效 Part 专家被削弱 + 部分标签被平滑错分”，因此整体下降。

严格结论：这次修复版没有证明 8-NN 标签修复无效，也没有证明 confidence route 有效；它
只说明当前三项联合实现低于旧版。下一步应拆成：

```text
A: 8-NN label + 旧 hard route
B: 1-NN label + confidence route
C: 8-NN label + confidence route
```

并与旧 `part_moe_leg` 对比。优先建议先取消 `confidence > 0.5` 的线性强门控，或只把
confidence 用作很小的融合修正；同时保留 unknown 点的原 temporal/global 输出。这样才
不会因为不可靠的 SMPL 距离先验直接覆盖已学到的 Part expert。

## 2026-09-04 part_moe_leg 修复版下降原因

已核对：修复版与旧版均使用 `ITERATIONS=25000`、`DENSIFY_UNTIL_ITER=1800`、`SEED=0`，
且没有开启 MAPO、point 或 DynOMo，因此下降主要来自 Part 路由改动，不是训练步数或显存
配置差异。

核心原因：修复版同时改变了标签和融合强度。旧版接近固定的强 Part 路由；修复版先要求
`known && confidence > 0.5`，再使用
`confidence_mix=(confidence-0.5)/0.5`，最终 Part 作用约为
`0.9 * confidence_mix`。因此 confidence 在 `0.6`、`0.75` 时，Part 只保留约 18%、45%
的原始强度。对于标签正确但离 SMPL 表面较远的衣物 Gaussian，这会错误地退回 temporal/global
输出，削弱原本有效的专家。

同时，8-NN 仅按欧氏距离投票，没有 SMPL 拓扑、法向或边界约束；肩、髋、膝、手脚连接处及
衣物/身体相邻点可能被平滑到邻近部件。这样会把部分 Gaussian 送进错误 expert。`label_margin`
在现有统计中接近 1，不能有效识别不确定标签，所以 confidence gate 没有真正修复错误标签，
反而进一步抑制了一批有效 Part 输出。

六序列均值相对旧 `part_moe_leg`：PSNR `-0.026139503055148339`，SSIM `-0.00022521192828813449`，
LPIPS*1000 `+0.17511132949342093`。因此这次下降应解释为“门控过强 + 邻域投票误平滑”的联合
副作用，而不是证明 robust label 思路本身无效。

## 2026-09-05 part_time_moe 独立消融

### 完整实验设置汇总

实验目标：在不加入 confidence route、robust label、额外正则或点操作的前提下，
把 standalone `l2 soft` 的时间分支和 standalone `part_moe_leg` 的空间部件专家
组合起来，验证“时间分组 + 人体部件分组”的直接组合效果。

训练和数据配置：

```text
数据集：DNA-Rendering
序列：0044_11、0051_09、0206_04、0813_05、0007_04、0019_10
训练步数：25000
densify_until_iter：1800
随机 seed：0
MLP depth/width：3 / 512
seq_len：8
seq_xyz_knn：8
time_step_num：3
max_time_step：3
minimal_time_step：1
图像数据：训练/评测时通过 IMAGE_DATA_DEVICE=cpu 运行，降低显存压力
```

损失和基础训练流程仍沿用当前仓库的原始训练流程：

```text
L1 loss weight：1.0
SSIM loss weight：0.01
LPIPS loss weight：0.01
不增加 AIAP、DynOMo、VGGT、motion temperature 或其他新正则项
```

高斯点设置：仍使用原始 SMPL-X 初始化和普通 Gaussian 优化流程；只在前 1800
步进行 densification，之后不改变点数量策略。该实验没有新增、复制或删除
Gaussian，也没有修改 canonical Gaussian 初始化。

时间分支配置：

```text
use_mapo_all_dynamic：true
mapo_max_partition_level：2
mapo_num_frames：100
level 0：训练开始到 4999 步，1 个 root temporal branch
level 1：5000 步激活，2 个 branch，每段 50 个 pose_id
level 2：10000 步激活，4 个 branch，每段 25 个 pose_id
mapo_partition_level1_iter：5000
mapo_partition_level2_iter：10000
mapo_partition_level3_iter：15000（本实验未使用 level 3）
mapo_soft_routing：true
mapo_soft_blend_width：4 帧
mapo_partial_sharing：false
mapo_shared_trunk：false
mapo_dynamic_score_enabled：false
```

时间 branch 是完整 MLP 的独立副本。新增 branch 在对应分层时从父 branch 的
参数快照继承，然后清除新 branch 的 optimizer state，继续独立训练。4 个 branch
并不是从随机参数开始。

在普通时间区间内只使用对应 branch；在边界前后 4 帧内，使用 smoothstep 对相邻
两个 branch 的输出进行软混合。当前 pose 通常只计算一个 branch，边界 pose 最多
计算两个 branch。

Part 专家配置：

```text
use_part_moe：true
part_label_schema：part_moe_leg
num_parts：7
part_moe_start_iter：10000
part_moe_warmup：1000
part_moe_global_keep：0.1
part_label_refresh_interval：5000
part_label_robust：false
part_confidence_route：false
```

虽然脚本默认传入 `part_label_knn=8`，但 `part_label_robust=false` 时控制器会强制
使用 `k=1`。因此本实验的标签是：每个 canonical Gaussian 找最近的一个
SMPL-X canonical 顶点，再继承该顶点的 `part_moe_leg` 标签；不是 8-NN 投票，也
不是置信度加权路由。

标签生成和刷新时间为：

```text
10000 步：保存当前 Gaussian，建立初始 part label，并激活 Part expert
15000 步：按 refresh_interval=5000 刷新标签
20000 步：再次刷新标签
```

距离超过 `part_max_smpl_dist=0.08` 或标签无效的 Gaussian 会保留为 label 0，
作为 unknown/global。标签文件、距离、调试 PLY 和元数据保存在每个实验目录的
`part_labels/iteration_<iter>/` 下。

Part expert 的实际结构不是 7 套完整原始 deformation MLP，而是 7 个
`TemporalConditionedFullPartExpert`。每个专家包含：

```text
Linear(512, 512) + ReLU
xyz head：512 -> 3
rotation head：512 -> 4
scaling head：512 -> 3
```

它们初始化时从 temporal 共享输出头复制预测头参数，body 初始化为近似恒等映射。
Part expert 接收 temporal branch 产生的 `temporal_hidden`，不是直接重新读取原始
Gaussian features。

两种路由的实际顺序和融合：

```text
原始 Gaussian features
    -> temporal branch 根据 pose_id 选择/软混合
    -> temporal_hidden
    -> 根据 Gaussian part_label 选择 7 个 Part expert 之一
    -> 得到 part_output
    -> temporal_output 与 part_output 融合
    -> d_xyz / d_rotation / d_scaling
```

对有效 part，Part expert 内部使用：

```text
part_output = 0.1 * global expert output + 0.9 * 对应 part expert output
```

unknown label 0 只使用 global Part expert。外层从 10000 步之后开始做 1000 步
warmup：

```text
10000 步及以前：part_moe_alpha=0，只有 temporal output
10001 到 11000 步：Part output 权重从 0 线性增加到 0.9
11000 步以后：part_gate=1，使用 Part output
```

本实验 `fusion_mode=replace`，所以不是把两个模块的最终输出简单相加，而是先对
每个时间 branch 得到 temporal output 和 Part output，再用 Part gate 在二者之间
插值：

```text
final = (1 - part_gate) * temporal_output
      + part_gate * part_output
```

当 pose 位于时间边界时，temporal output 和 part output 都先按相邻时间 branch
的权重 soft blend，再进行上面的 Part 融合。

评测配置：

```text
final_eval_only：true
训练中只保存/评测最终 25000 步结果，避免中间完整评测造成 DNA 显存压力
render.py：novel-view 评测
视角数：120
指标文件：metrics/results_novelview_25000.json
LPIPS 汇总时乘以 1000
```

输出目录格式：

```text
output/DNA-Rendering/<sequence>/part_time_moe/20260905_part_time_moe_plain_cpu/
```

本实验关闭的模块包括：`mapo_partial_sharing`、`mapo_shared_trunk`、动态分数、
运动温度、point 系列、DynOMo、VGGT、Tri/Token-Tri、robust 8-NN 标签、confidence
route、part score route、Part budget 和额外 consistency loss。

目标：只组合旧 `part_moe_leg` 与完整 `l2 soft`，验证组合是否超过任一单模块。

固定配置：

```text
experiment_name = part_time_moe
ITERATIONS = 25000
DENSIFY_UNTIL_ITER = 1800
SEED = 0
GPU = 1
mapo_max_partition_level = 2
mapo_partition_level1_iter = 5000
mapo_partition_level2_iter = 10000
mapo_soft_routing = true
mapo_soft_blend_width = 4
mapo_partial_sharing = false
part_moe_start_iter = 10000
part_moe_warmup = 1000
part_moe_global_keep = 0.1
part_label_schema = part_moe_leg
part_label_robust = false
part_confidence_route = false
temporal_conditioned_part_fusion_mode = replace
```

明确关闭：confidence route、robust 8-NN、part score route、point、DynOMo、VGGT、
motion temperature 和额外正则。非 robust 标签控制器会强制使用原始 `1-NN` 标签逻辑。

实现入口：`scripts/exps_dnarendering.sh part_time_moe`。输出目录：
`output/DNA-Rendering/<sequence>/part_time_moe/20260905_part_time_moe_plain/`。

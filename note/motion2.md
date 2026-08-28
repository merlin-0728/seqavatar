# SeqAvatar Point Notes

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

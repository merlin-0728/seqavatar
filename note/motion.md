# SeqAvatar Motion Notes

## 2026-07-08 state 消融命名统一

用户要求：

```text
停止当前 state 实验。
不要再同时使用两套名字，全部改成 state。
```

执行结果：

```text
1. 已停止 tmux:
   state_gpu2
   state_gpu3

2. 已确认相关 train.py 进程结束，GPU2/GPU3 已释放。

3. 已删除中断产生的 partial output:
   output/DNA-Rendering/0044_11/state/20260707_235031
   output/DNA-Rendering/0813_05/state/20260707_235031

4. logs/state 下日志保留，用于追溯中断原因和代码变更历史。
```

命名统一后的规则：

```text
实验名：state
脚本入口：bash scripts/exps_dnarendering.sh state
命令行开关：--use_state
内部 flag：use_state
参数：
    state_dim
    state_hidden_dim
    state_layers
    state_film
网络文件：
    nets/mlp_temporal_state.py
网络类：
    TemporalStateEncoder
条件名：
    state_conds
```

当前 state 方法含义：

```text
seq_pose_conds
    -> TemporalStateEncoder
    -> state feature
    -> FiLM gamma / beta

x_emb + pose_conds + seq_pose_conds + seq_xyz_conds
    -> state-modulated NonrigidDeformer hidden layers
    -> d_xyz, d_rotation, d_scaling
```

重要约束：

```text
默认 use_state=False，baseline 不受影响。
只有脚本追加 --use_state 时才启用 state 分支。
dataset_readers.py 暂时不改，第一版直接复用 seq_pose_conds 作为 state 输入。
```

## 2026-07-08 state 六序列重跑

用户要求：

```text
重新跑六个序列。
```

本次启动信息：

```text
RUN_TIME: 20260708_001225
日志目录: logs/state
输出目录: output/DNA-Rendering/<sequence>/state/20260708_001225/
脚本入口: bash scripts/exps_dnarendering.sh state
开关: --use_state
```

任务拆分：

```text
tmux state_gpu2 / GPU2:
    0044_11
    0051_09
    0206_04

tmux state_gpu3 / GPU3:
    0813_05
    0007_04
    0019_10
```

日志文件：

```text
logs/state/20260708_001225_DNA-Rendering_state_gpu2.log
logs/state/20260708_001225_DNA-Rendering_state_gpu3.log
```

实验配置确认：

```text
STATE_ENABLED: 1
FINAL_EVAL_ONLY: 1
TEST_ITERATIONS: 25000
SAVE_ITERATIONS: 25000
baseline 默认 use_state=False，不受本实验影响。
```

中途状态：

```text
0044_11 完成。
0051_09 完成。
0813_05 完成。
0007_04 完成。
0019_10 仍在 GPU3 训练。

0206_04 在 GPU2 首次训练到约 4100/25000 时出现 CUDA illegal memory access。
错误发生在 loss.backward()，不是参数缺失或 Python 逻辑错误。
已删除 0206_04 不完整 output：
    output/DNA-Rendering/0206_04/state/20260708_001225

已用同一 RUN_TIME 单独重启 0206_04：
    tmux state_gpu2_retry / GPU2
    RUN_TIME=20260708_001225
    SEQUENCES_OVERRIDE=0206_04
```

后续状态：

```text
0019_10 已在 GPU3 完成。

0206_04 在 GPU2 retry 中再次出现 CUDA illegal memory access，
失败位置仍在训练早期 backward 附近，tmux 已退出。

为排除 GPU2 或 CUDA context 问题，已删除 0206_04 retry 的不完整 output，
并改用 GPU3 单独重跑：
    tmux state_gpu3_retry / GPU3
    RUN_TIME=20260708_001225
    SEQUENCES_OVERRIDE=0206_04
```

最终状态：

```text
0206_04 在 GPU3 retry 中成功完成。
六个 DNA-Rendering 序列均完成 state 消融实验。
tmux 已全部退出。
```

最终评价口径：

```text
使用 render.py 阶段日志中的 novelview 指标：
    output/DNA-Rendering/<sequence>/state/20260708_001225/logs/render_<sequence>_state.log

不使用 metrics/results_novelview_25000.json，
因为该 json 对应训练内 evaluation 数值，和 render.py 最终数值存在轻微差异。
```

state 六序列结果：

| Sequence | PSNR | SSIM | LPIPS x1000 |
|---|---:|---:|---:|
| 0007_04 | 29.24228205680847 | 0.9572560116648674 | 46.17766332812607 |
| 0019_10 | 34.259186045328775 | 0.9774644588430722 | 25.161782838404175 |
| 0044_11 | 32.53612701098124 | 0.9758686065673828 | 23.281582545799512 |
| 0051_09 | 28.00005695025126 | 0.9683940375844637 | 34.243269908862814 |
| 0206_04 | 30.502982807159423 | 0.9656202296415964 | 37.32823890944322 |
| 0813_05 | 35.29887183507284 | 0.9844440807898839 | 21.13748899816225 |
| Average | 31.639917784267002 | 0.9715079041818778 | 31.22167108813301 |

运行异常记录：

```text
0206_04 在 GPU2 上两次出现 CUDA illegal memory access。
同一代码、同一 RUN_TIME、同一序列切到 GPU3 后完整跑完。
因此这次异常更像 GPU2/CUDA kernel 运行状态问题，而不是 state 代码必然错误。
```

## 2026-07-08 state 与 densify=1500 baseline 对比排查

用户指定的 baseline 日志：

```text
logs/20260701_162750_DNA-Rendering_orginal.log
    0044_11

logs/20260701_130518_DNA-Rendering_orginal.log
    0051_09
    0206_04
    0813_05
    0007_04
    0019_10
```

baseline 配置确认：

```text
DENSIFY_UNTIL_ITER: 1500
ITERATIONS: 25000
TEST_ITERATIONS: 3000 10000 25000
SAVE_ITERATIONS: 3000 10000 25000
seq_len: 8
seq_xyz_knn: 8
time_step_num: 3
max_time_step: 3
minimal_time_step: 1
```

state 配置确认：

```text
DENSIFY_UNTIL_ITER: 1500
ITERATIONS: 25000
FINAL_EVAL_ONLY: 1
TEST_ITERATIONS: 25000
SAVE_ITERATIONS: 25000
seq_len: 8
seq_xyz_knn: 8
time_step_num: 3
max_time_step: 3
minimal_time_step: 1
use_state: True
```

最终 render.py 指标对比：

| Sequence | Baseline PSNR | State PSNR | ΔPSNR | Baseline SSIM | State SSIM | ΔSSIM | Baseline LPIPS x1000 | State LPIPS x1000 | ΔLPIPS x1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.533587328592937 | 29.24228205680847 | -0.2913052717844664 | 0.9583350042502086 | 0.9572560116648674 | -0.0010789925853411653 | 45.38941358526548 | 46.17766332812607 | 0.7882497428605972 |
| 0019_10 | 35.220936361948645 | 34.259186045328775 | -0.9617503166198702 | 0.9806962579488754 | 0.9774644588430722 | -0.003231799105803179 | 21.26493356966724 | 25.161782838404175 | 3.8968492687369363 |
| 0044_11 | 32.97411826451619 | 32.53612701098124 | -0.4379912535349533 | 0.9779148245851199 | 0.9758686065673828 | -0.002046218017737078 | 21.396957943215966 | 23.281582545799512 | 1.884624602583547 |
| 0051_09 | 28.671355326970417 | 28.00005695025126 | -0.6712983767191574 | 0.9714479451378186 | 0.9683940375844637 | -0.0030539075533548843 | 31.09206970936308 | 34.243269908862814 | 3.151200199499729 |
| 0206_04 | 31.379924058914185 | 30.502982807159423 | -0.8769412517547615 | 0.9697981854279836 | 0.9656202296415964 | -0.004177955786387133 | 34.023474312076964 | 37.32823890944322 | 3.3047645973662556 |
| 0813_05 | 36.086734771728516 | 35.29887183507284 | -0.787862936655678 | 0.986892951031526 | 0.9844440807898839 | -0.002448870241642087 | 18.46366561173151 | 21.13748899816225 | 2.6738233864307404 |
| Average | 32.31110935211182 | 31.639917784267002 | -0.671191567844819 | 0.974180861396922 | 0.9715079041818778 | -0.0026729572150442 | 28.605085788553374 | 31.22167108813301 | 2.616585299579635 |

最终 Gaussian 点数对比：

| Sequence | Baseline #pts | State #pts | Δpts | Δ% |
|---|---:|---:|---:|---:|
| 0007_04 | 27199 | 24414 | -2785 | -10.239347034817456 |
| 0019_10 | 34400 | 28303 | -6097 | -17.723837209302324 |
| 0044_11 | 62790 | 55687 | -7103 | -11.312310877528269 |
| 0051_09 | 50516 | 45862 | -4654 | -9.212922638372001 |
| 0206_04 | 42792 | 38303 | -4489 | -10.490278556739579 |
| 0813_05 | 39653 | 33726 | -5927 | -14.947166670869796 |

排查结论：

```text
1. 不是 densify_until_iter 不一致导致。
   baseline 和 state 都是 densify_until_iter=1500。

2. 不是最终 render.py 与训练内 evaluation 口径导致。
   两组实验都存在很小的 train-eval / render.py 差异，0051_09 两边都有约 +0.09 PSNR render gap。

3. state 分支不是 baseline-preserving。
   use_state=True 时走新的 state_layers，而不是原始 self.mlp。
   state_layers 没有从 baseline self.mlp 拷贝权重。
   state_film_layer 的 gamma/beta 是随机初始化，不是 gamma=1、beta=0。

4. TemporalStateEncoder 使用 seq_pose_conds 生成 frame-level state，
   然后用同一组 FiLM gamma/beta 调制该帧所有 Gaussian。
   这个状态不是 point-wise / part-wise，容易压制局部细节。

5. state 影响了 densification 梯度轨迹。
   六个序列最终 Gaussian 数量全部减少 9%-18%，说明它不仅改了输出，
   还让点云增长和局部细节表达变弱。
```

当前判断：

```text
这次 state 负提升主要是方案/实现设计问题：
    新分支替换 baseline MLP；
    FiLM 非 identity 初始化；
    frame-level state 对所有 Gaussian 统一调制；
    导致优化和 densification 都偏离 baseline。

不像是日志解析错误、densify 参数不一致，或单纯评价脚本问题。
```

## 2026-07-08 state_warm 消融启动

目标：

```text
训练前期完全等价 baseline；
1500 iteration 后平滑打开 state 分支；
3000 iteration ramp 到完整 state branch；
先跑 0051_09 和 0007_04 做最小验证。
```

代码实现：

```text
新增开关:
    --use_state_warm
    --state_start_iter
    --state_ramp_iter

state_warm 逻辑:
    iteration < state_start_iter:
        h = self.mlp(features)

    iteration >= state_start_iter:
        h_base = self.mlp(features)
        h_state = identity-init state branch(features, state)
        alpha = clamp((iteration - state_start_iter) / state_ramp_iter, 0, 1)
        h = (1 - alpha) * h_base + alpha * h_state

state branch:
    Linear 层从 baseline self.mlp 拷贝初始化；
    FiLM 初始化为 gamma=1, beta=0；
    state 输入第一版复用 seq_pose_conds，不改 dataset_readers.py。
```

启动信息：

```text
RUN_TIME: 20260708_131628
脚本入口: bash scripts/exps_dnarendering.sh state_warm
日志目录: logs/state
输出目录: output/DNA-Rendering/<sequence>/state_warm/20260708_131628/
```

任务拆分：

```text
tmux state_warm_gpu2 / GPU2:
    0051_09

tmux state_warm_gpu3 / GPU3:
    0007_04
```

实验配置确认：

```text
STATE_WARM_ENABLED: 1
STATE_START_ITER: 1500
STATE_RAMP_ITER: 3000
DENSIFY_UNTIL_ITER: 1500
TEST_ITERATIONS: 25000
SAVE_ITERATIONS: 25000
```

实时状态更新（2026-07-08 13:21 CST）：

```text
tmux sessions:
    state_warm_gpu2: running 0051_09 on GPU2
    state_warm_gpu3: running 0007_04 on GPU3

当前进度:
    0051_09: 已越过 1500 iter，约 1980 / 25000
    0007_04: 已越过 1500 iter，约 2130 / 25000

当前判断:
    两个进程都在正常训练；
    未发现 Traceback / RuntimeError / CUDA OOM / illegal memory access；
    warm 分支已经开始接入，后续重点看最终 #pts 是否恢复到 densify=1500 baseline 附近。
```

## 2026-07-08 state_warm 最小验证结果

运行信息：

```text
RUN_TIME: 20260708_131628
实验: state_warm
序列: 0007_04, 0051_09
最终评价口径: render.py novelview #120, iteration 25000
```

最终指标：

| Sequence | Baseline PSNR | State PSNR | State warm PSNR | Baseline SSIM | State SSIM | State warm SSIM | Baseline LPIPS x1000 | State LPIPS x1000 | State warm LPIPS x1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.533587328592937 | 29.24228205680847 | 29.44352542559306 | 0.9583350042502086 | 0.9572560116648674 | 0.9577622433503469 | 45.38941358526548 | 46.17766332812607 | 45.668250089511275 |
| 0051_09 | 28.671355326970417 | 28.00005695025126 | 28.43164316813151 | 0.9714479451378186 | 0.9683940375844637 | 0.9703817427158355 | 31.09206970936308 | 34.243269908862814 | 31.73201344131182 |
| Average | 29.10247132778168 | 28.621169503529863 | 28.937584296862283 | 0.9648914746940136 | 0.9628250246246656 | 0.9640719930330912 | 38.24074164731428 | 40.21046661849444 | 38.70013176541155 |

state_warm 相比 baseline：

| Sequence | ΔPSNR | ΔSSIM | ΔLPIPS x1000 | Δpts |
|---|---:|---:|---:|---:|
| 0007_04 | -0.09006190299987793 | -0.0005727608998616907 | 0.2788365042457954 | 322 |
| 0051_09 | -0.2397121588389055 | -0.0010662024219830757 | 0.6399437319487404 | -80 |
| Average | -0.16488703091939172 | -0.0008194816609223832 | 0.45939011809726793 | 121.0 |

state_warm 相比旧 state：

| Sequence | ΔPSNR | ΔSSIM | ΔLPIPS x1000 | Δpts |
|---|---:|---:|---:|---:|
| 0007_04 | 0.20124336878458848 | 0.0005062316854794746 | -0.5094132386147976 | 3107 |
| 0051_09 | 0.43158621788025187 | 0.0019877051313718086 | -2.511256467550993 | 4574 |
| Average | 0.3164147933324202 | 0.0012469684084256416 | -1.5103348530828953 | 3840.5 |

最终点数：

| Sequence | Baseline #pts | State #pts | State warm #pts |
|---|---:|---:|---:|
| 0007_04 | 27199 | 24414 | 27521 |
| 0051_09 | 50516 | 45862 | 50436 |

结论：

```text
state_warm 解决了旧 state 的主要实现问题：
    最终 #pts 基本恢复到 baseline 附近；
    相比旧 state，PSNR / SSIM / LPIPS 都明显改善。

但 state_warm 仍没有超过 baseline：
    两个序列平均 PSNR 仍低 0.16488703091939172；
    SSIM 低 0.0008194816609223832；
    LPIPS x1000 高 0.45939011809726793。

因此当前结论不是“state 主线有效”，而是：
    warm / identity-init 能修复旧 state 的训练破坏；
    但当前 frame-level state FiLM 还没有证明能带来正向收益。
```

## 2026-07-09 state_warm 点数核查

点数来源：

```text
直接读取 point_cloud/iteration_25000/point_cloud.ply 的 header:
    element vertex <N>
```

核查结果：

| Sequence | Baseline #pts | State #pts | State warm #pts | State warm vs baseline |
|---|---:|---:|---:|---:|
| 0007_04 | 27199 | 24414 | 27521 | +322 |
| 0051_09 | 50516 | 45862 | 50436 | -80 |

判断：

```text
state_warm 的最终点数已经恢复到 baseline 附近：
    0007_04 比 baseline 多 322 个 Gaussian；
    0051_09 比 baseline 少 80 个 Gaussian。

因此现在指标仍低，不再主要是 densification / Gaussian 数量不足导致。
更可能的问题是 state 调制本身没有提供有效信息，或者 frame-level FiLM 对所有 Gaussian 的统一调制会轻微干扰非刚性形变预测。
```

## 2026-07-09 state_warm 保守策略实验启动

目标：

```text
验证当前 state_warm 负提升是否来自“最终完全切到 state 分支太强”。
只跑两个代表序列:
    0007_04
    0051_09
```

新增参数：

```text
--state_max_alpha

默认值 1.0，不影响 baseline / state / 原 state_warm。
state_warm_half 设置为 0.5，表示最多:
    h = 0.5 * h_base + 0.5 * h_state
```

实验设置：

| Method | state_start_iter | state_ramp_iter | state_max_alpha |
|---|---:|---:|---:|
| state_warm_late | 3000 | 5000 | 1.0 |
| state_warm_half | 1500 | 3000 | 0.5 |

启动信息：

```text
RUN_TIME: 20260709_014838

tmux state_warm_cons_gpu2:
    GPU2, 0051_09
    先跑 state_warm_late，再串行跑 state_warm_half

tmux state_warm_cons_gpu3:
    GPU3, 0007_04
    先跑 state_warm_late，再串行跑 state_warm_half

日志目录:
    logs/state

输出目录:
    output/DNA-Rendering/<sequence>/state_warm_late/20260709_014838/
    output/DNA-Rendering/<sequence>/state_warm_half/20260709_014838/
```

已确认：

```text
state_warm_late 训练命令包含:
    --state_start_iter 3000
    --state_ramp_iter 5000
    --state_max_alpha 1.0
```

## 2026-07-09 state_warm 保守策略结果

运行信息：

```text
RUN_TIME: 20260709_014838
最终评价口径: render.py novelview #120, iteration 25000
已完成:
    state_warm_late: 0007_04, 0051_09
    state_warm_half: 0007_04, 0051_09
```

逐序列结果：

| Method | Sequence | PSNR | SSIM | LPIPS x1000 | #pts |
|---|---|---:|---:|---:|---:|
| baseline | 0007_04 | 29.533587328592937 | 0.9583350042502086 | 45.38941358526548 | 27199 |
| state | 0007_04 | 29.24228205680847 | 0.9572560116648674 | 46.17766332812607 | 24414 |
| state_warm | 0007_04 | 29.44352542559306 | 0.9577622433503469 | 45.668250089511275 | 27521 |
| state_warm_late | 0007_04 | 29.4391961256663 | 0.957808281481266 | 45.61902492617567 | 27641 |
| state_warm_half | 0007_04 | 29.509914763768514 | 0.9583077356219292 | 45.128585336109 | 27334 |
| baseline | 0051_09 | 28.671355326970417 | 0.9714479451378186 | 31.09206970936308 | 50516 |
| state | 0051_09 | 28.00005695025126 | 0.9683940375844637 | 34.243269908862814 | 45862 |
| state_warm | 0051_09 | 28.43164316813151 | 0.9703817427158355 | 31.73201344131182 | 50436 |
| state_warm_late | 0051_09 | 28.408707920710246 | 0.970343432823817 | 32.259145316978294 | 50364 |
| state_warm_half | 0051_09 | 28.63980484008789 | 0.9714886670311292 | 31.14774018370857 | 51117 |

两序列平均：

| Method | PSNR | SSIM | LPIPS x1000 | #pts |
|---|---:|---:|---:|---:|
| baseline | 29.10247132778168 | 0.9648914746940136 | 38.24074164731428 | 38857.5 |
| state | 28.621169503529863 | 0.9628250246246656 | 40.21046661849444 | 35138.0 |
| state_warm | 28.937584296862283 | 0.9640719930330912 | 38.70013176541155 | 38978.5 |
| state_warm_late | 28.92395202318827 | 0.9640758571525414 | 38.93908512157698 | 39002.5 |
| state_warm_half | 29.074859801928202 | 0.9648982013265293 | 38.13816275990878 | 39225.5 |

相对 baseline：

| Method | ΔPSNR | ΔSSIM | ΔLPIPS x1000 | Δ#pts |
|---|---:|---:|---:|---:|
| state | -0.4813018242518119 | -0.0020664500693480248 | 1.9697249711801632 | -3719.5 |
| state_warm | -0.16488703091939172 | -0.0008194816609223832 | 0.45939011809726793 | 121.0 |
| state_warm_late | -0.17851930459340437 | -0.0008156175414721356 | 0.6983434742626997 | 145.0 |
| state_warm_half | -0.027611525853474816 | 0.0000067266325156079 | -0.10257888740549603 | 368.0 |

关键观察：

```text
1. state_warm_late 没有改善原 state_warm。
   推迟到 3000 iter、5000 iter ramp 后，平均 PSNR 仍比 baseline 低 0.17851930459340437。

2. state_warm_half 明显最好。
   相比原 state_warm:
       PSNR +0.1372755050659169
       SSIM +0.0008262082934379911
       LPIPS x1000 -0.561969005502764

3. state_warm_half 已经非常接近 baseline，并且平均 SSIM / LPIPS 略优于 baseline：
       PSNR 仍低 0.027611525853474816
       SSIM 高 0.0000067266325156079
       LPIPS x1000 低 0.10257888740549603

4. 点数不是问题。
   state_warm_half 两序列平均 #pts 比 baseline 多 368。
```

当前结论：

```text
完整切到 state 分支确实太激进。
state_warm_half 说明 state 信息不是完全无用，但它更适合作为弱 residual / 半强度调制，
不适合替换 baseline MLP 主路径。

下一步如果继续 state 线，应以 state_warm_half 或更弱的 residual FiLM 为基础，
不要再做 full switch。
```

## 2026-07-09 state_warm alpha 调参启动

目标：

```text
state_warm_half(alpha=0.5) 已经明显优于 alpha=1.0。
继续测试更弱的 state 辅助强度，寻找比 alpha=0.5 更稳的点。
```

实验设置：

| Method | state_start_iter | state_ramp_iter | state_max_alpha |
|---|---:|---:|---:|
| state_warm_a03 | 1500 | 3000 | 0.3 |
| state_warm_a02 | 1500 | 3000 | 0.2 |
| state_warm_a04 | 1500 | 3000 | 0.4 |

启动信息：

```text
RUN_TIME: 20260709_131629

优先顺序:
    a03 -> a02 -> a04

tmux state_warm_alpha_gpu2:
    GPU2, 0051_09
    依次跑 state_warm_a03, state_warm_a02, state_warm_a04

tmux state_warm_alpha_gpu3:
    GPU3, 0007_04
    依次跑 state_warm_a03, state_warm_a02, state_warm_a04

日志目录:
    logs/state

输出目录:
    output/DNA-Rendering/<sequence>/state_warm_a03/20260709_131629/
    output/DNA-Rendering/<sequence>/state_warm_a02/20260709_131629/
    output/DNA-Rendering/<sequence>/state_warm_a04/20260709_131629/
```

已确认：

```text
state_warm_a03 当前训练命令包含:
    --state_start_iter 1500
    --state_ramp_iter 3000
    --state_max_alpha 0.3
```

## 2026-07-09 state_warm alpha 调参状态

这轮 alpha 调参没有得到可用评价指标。

尝试运行：

```text
RUN_TIME: 20260709_131629

state_warm_a03:
    0051_09 on GPU2
    0007_04 on GPU3

计划串行继续:
    state_warm_a02
    state_warm_a04
```

实际结果：

```text
0051_09 / state_warm_a03 / GPU2:
    训练中途失败
    RuntimeError: CUDA error: unspecified launch failure

0007_04 / state_warm_a03 / GPU3:
    训练进度到 25000 iter
    但最终 evaluation / LPIPS 计算阶段失败
    RuntimeError: CUDA error: unknown error
    没有形成可用 novelview 指标和完整保存结果

state_warm_a02 / state_warm_a04:
    因 a03 失败，串行命令没有继续启动
```

机器状态：

```text
nvidia-smi 当前无法识别 GPU2:
    Unable to determine the device handle for GPU2: Unknown Error
```

判断：

```text
这轮失败更像 GPU2/驱动/CUDA 运行状态问题，不应该把 a03 当成方法负提升。
当前仍只能基于已经完成的 state_warm / state_warm_late / state_warm_half 做结论。
```

## 2026-07-10 state_warm_half 价值判断

为什么先做调参实验：

```text
旧 state 的失败原因不是单一的结构无效，而是 state 分支介入太强：
    1. use_state=True 直接替换 baseline MLP 主路径；
    2. 新 state_layers / FiLM 改变训练早期梯度；
    3. 最终 Gaussian 点数比 baseline 少 9%-18%；
    4. 非刚性变形和 densification 被同时扰动。

因此需要先做强度调参，把问题拆开：
    alpha=1.0: state 最终完全接管，测试 full state 是否可行；
    alpha=0.5: state 只作为辅助项，测试 weak modulation 是否可行；
    alpha=0.2/0.3/0.4: 进一步寻找更稳的辅助强度。
```

state_warm_half 的价值：

```text
state_warm_half 不是一个已经明显超过 baseline 的最终方法，
但它有实验价值。

原因：
    1. 它恢复了 Gaussian 点数，说明 warm + identity-init 解决了旧 state 的训练破坏问题；
    2. 它比 state_warm / state_warm_late / 旧 state 都好，说明 state 信息不能强替换主路径，但可以弱辅助；
    3. 两序列平均 SSIM 和 LPIPS 略优于 baseline，说明 motion state 对感知质量可能有局部收益；
    4. PSNR 只低 0.0276，已经接近实验噪声级别，值得继续小规模确认。

当前不能把 state_warm_half 当作主结果，
但可以作为 state 线继续优化的基准版本。
```

下一步判断标准：

```text
如果 alpha=0.2/0.3/0.4 能让 PSNR 接近或超过 baseline，同时保持 LPIPS 优势：
    state 线值得继续，定位为 weak residual / FiLM auxiliary branch。

如果更弱 alpha 仍不能稳定超过 baseline：
    不建议继续把 state 作为主线，只能作为分析性辅助实验。
```

state_warm_half 简要流程：

```text
1. baseline 主路径保持不变:
   features -> original non-rigid MLP -> h_base

2. 额外用 seq_pose_conds 编码 motion state:
   seq_pose_conds -> TemporalMotionStateEncoder -> h_t

3. 用 h_t 生成 FiLM 参数:
   h_t -> gamma / beta

4. state 分支用 identity-init 的 MLP 处理同一组 features:
   features -> state MLP + FiLM(gamma, beta) -> h_state

5. 训练前 1500 iter 完全等价 baseline:
   h = h_base

6. 1500 iter 后用 3000 iter 平滑打开 state:
   alpha 从 0 增加到 0.5

7. 最终只做半强度融合:
   h = (1 - alpha) * h_base + alpha * h_state
   max alpha = 0.5

8. 后面的输出头不变:
   h -> d_xyz, d_rotation, d_scaling
```

## 2026-07-10 state_warm alpha 调参完成

运行信息：

```text
RUN_TIME: 20260710_192726

state_warm_a02:
    GPU1
    state_max_alpha = 0.2
    sequences: 0007_04, 0051_09

state_warm_a03:
    GPU0
    state_max_alpha = 0.3
    sequences: 0007_04, 0051_09

state_warm_a04:
    GPU3
    state_max_alpha = 0.4
    sequences: 0007_04, 0051_09
```

最终 render 指标：

| Method | Sequence | PSNR | SSIM | LPIPS x1000 | #pts |
|---|---|---:|---:|---:|---:|
| state_warm_a02 | 0007_04 | 29.475177860260008 | 0.9582526445388794 | 44.86874416470528 | 27322 |
| state_warm_a02 | 0051_09 | 28.64856807390849 | 0.9714167733987172 | 31.141103074575465 | 50739 |
| state_warm_a03 | 0007_04 | 29.504997142155965 | 0.9582202404737472 | 44.96883531101048 | 27293 |
| state_warm_a03 | 0051_09 | 28.666919644673666 | 0.9713305761416753 | 31.359742232598368 | 50388 |
| state_warm_a04 | 0007_04 | 29.545680459340414 | 0.9586271822452544 | 44.053897711758815 | 27319 |
| state_warm_a04 | 0051_09 | 28.67199905713399 | 0.9714517717560133 | 30.91851979649315 | 50910 |

两序列平均：

| Method | PSNR | SSIM | LPIPS x1000 |
|---|---:|---:|---:|
| baseline | 29.10247132778168 | 0.9648914746940137 | 38.24074164731428 |
| state_warm_half | 29.074859801928202 | 0.9648982013265293 | 38.13816275990878 |
| state_warm_a02 | 29.06187296708425 | 0.9648347089687983 | 38.00492361964037 |
| state_warm_a03 | 29.085958393414813 | 0.9647754083077112 | 38.16428877180442 |
| state_warm_a04 | 29.108839758237202 | 0.9650394770006339 | 37.48620875412598 |

相对 baseline：

| Method | ΔPSNR | ΔSSIM | ΔLPIPS x1000 |
|---|---:|---:|---:|
| state_warm_a02 | -0.04059836069743028 | -0.00005676572521540191 | -0.2358180276739077 |
| state_warm_a03 | -0.016512934366865295 | -0.00011606638630246024 | -0.07645287550985813 |
| state_warm_a04 | 0.006368430455523821 | 0.00014800230662015412 | -0.7545328931882977 |

相对 state_warm_half：

| Method | ΔPSNR | ΔSSIM | ΔLPIPS x1000 |
|---|---:|---:|---:|
| state_warm_a02 | -0.012986834843953687 | -0.0000634923577309543 | -0.1332391402684081 |
| state_warm_a03 | 0.011098591486611298 | -0.00012279301881801263 | 0.026126011895641454 |
| state_warm_a04 | 0.033979956309000414 | 0.00014127567410460173 | -0.6519540057827982 |

结论：

```text
state_warm_a04 是目前 state 线最好的配置。

它相比 baseline:
    PSNR +0.006368430455523821
    SSIM +0.00014800230662015412
    LPIPS x1000 -0.7545328931882977

它相比 state_warm_half:
    PSNR +0.033979956309000414
    SSIM +0.00014127567410460173
    LPIPS x1000 -0.6519540057827982
```

判断：

```text
state 分支确实不能完整替换 baseline 主路径。
但 alpha=0.4 的 weak state modulation 在两个代表序列上已经同时改善 PSNR / SSIM / LPIPS。

如果继续 state 线，优先把 state_warm_a04 作为当前默认版本。
下一步应扩大到六序列验证，而不是继续在两个序列上细调 alpha。
```

## 2026-07-10 state_warm_a04 六序列验证

运行信息：

```text
RUN_TIME: 20260710_220456
Method: state_warm_a04
state_start_iter: 1500
state_ramp_iter: 3000
state_max_alpha: 0.4
final_eval_only: 1

GPU0: 0044_11
GPU1: 0206_04, 0813_05
GPU2: 0051_09
GPU3: 0007_04, 0019_10
```

调度说明：

```text
最初 GPU0 队列是 0044_11, 0051_09。
追加使用 GPU2 后，0051_09 改由 GPU2 单独跑。
GPU0 在完成 0044_11 后被 watchdog 停掉，避免继续重复训练 0051_09。

GPU0 停止前已经打印了 0051_09 的 sequence header，并把本地
output/DNA-Rendering/0051_09/state_warm_a04/20260710_220456/logs/train_0051_09_state_warm_a04.log
清成 0 字节。
0051_09 的有效最终指标以 GPU2 global log 为准：
logs/state/20260710_220456_DNA-Rendering_state_warm_a04_gpu2.log
```

最终 render 指标：

| Sequence | PSNR | SSIM | LPIPS x1000 | #pts |
|---|---:|---:|---:|---:|
| 0044_11 | 32.98019606272379 | 0.9779360016187032 | 21.452478552237152 | 61416 |
| 0051_09 | 28.68968474070231 | 0.9715397775173187 | 31.092260025131206 | 50323 |
| 0206_04 | 31.398172251383464 | 0.969858680665493 | 33.98444790703555 | 44053 |
| 0007_04 | 29.504483922322592 | 0.9582487439115842 | 45.16256918820242 | 27056 |
| 0813_05 | 36.0757737159729 | 0.9869193017482757 | 18.3957058393086 | 39841 |
| 0019_10 | 35.286135832468666 | 0.9809789876143138 | 20.932862628251316 | 34537 |

六序列平均：

| Method | PSNR | SSIM | LPIPS x1000 |
|---|---:|---:|---:|
| baseline | 32.31110935211182 | 0.974180861396922 | 28.605085788553374 |
| old state | 31.639917784267002 | 0.9715079041818778 | 31.22167108813301 |
| state_warm_a04 | 32.32240775426229 | 0.9742469155126147 | 28.503387356694372 |

相对 baseline：

| Method | ΔPSNR | ΔSSIM | ΔLPIPS x1000 |
|---|---:|---:|---:|
| state_warm_a04 | 0.011298402150465847 | 0.00006605411569271524 | -0.10169843185900262 |

相对 old state：

| Method | ΔPSNR | ΔSSIM | ΔLPIPS x1000 |
|---|---:|---:|---:|
| state_warm_a04 | 0.6824899699952844 | 0.0027390113307368402 | -2.718283731438639 |

结论：

```text
state_warm_a04 六序列验证成立，但提升幅度很小。

相比 baseline:
    PSNR +0.011298402150465847
    SSIM +0.00006605411569271524
    LPIPS x1000 -0.10169843185900262

相比 old state:
    PSNR +0.6824899699952844
    SSIM +0.0027390113307368402
    LPIPS x1000 -2.718283731438639

判断：
    full state 替换主路径明显有害；
    state_warm_a04 作为 weak modulation 能恢复到 baseline 并略微超过 baseline。
    这条线可以保留，但当前收益属于边际改善，后续优先看稳定性和更多数据集，而不是继续大幅加 state 强度。
```

## 2026-07-10 state_warm_a04 逐序列对比

说明：

```text
LPIPS x1000 越低越好。
ΔLPIPS x1000 为负数表示 state_warm_a04 更好。
baseline 使用 densify_until_iter=1500 的原始 orginal 结果：
    0044_11: logs/20260701_162750_DNA-Rendering_orginal.log
    其余五条: logs/20260701_130518_DNA-Rendering_orginal.log
old state 使用 20260708_001225 state 结果。
state_warm_a04 使用 20260710_220456 六序列结果。
```

state_warm_a04 相对 baseline：

| Sequence | ΔPSNR | ΔSSIM | ΔLPIPS x1000 | Δ#pts |
|---|---:|---:|---:|---:|
| 0044_11 | 0.006077798207599017 | 0.000021177033583286153 | 0.05552060902118683 | -1374 |
| 0051_09 | 0.01832941373189456 | 0.0000918323795000564 | 0.00019031576812622575 | -193 |
| 0206_04 | 0.018248192469279445 | 0.000060495237509394784 | -0.03902640504141175 | 1261 |
| 0007_04 | -0.02910340627034458 | -0.00008626033862435545 | -0.22684439706306136 | -143 |
| 0813_05 | -0.010961055755615234 | 0.000026350716749723446 | -0.06795977242290974 | 188 |
| 0019_10 | 0.06519947052002095 | 0.00028272966543840816 | -0.3320709414159211 | 137 |

state_warm_a04 相对 old state：

| Sequence | ΔPSNR | ΔSSIM | ΔLPIPS x1000 | Δ#pts |
|---|---:|---:|---:|---:|
| 0044_11 | 0.4440690517425523 | 0.0020673950513203643 | -1.8291039935623594 | 5729 |
| 0051_09 | 0.6896277904510519 | 0.0031457399328549407 | -3.151009883731607 | 4461 |
| 0206_04 | 0.8951894442240409 | 0.004238451023896528 | -3.34379100240767 | 5750 |
| 0007_04 | 0.26220186551412183 | 0.0009927322467168098 | -1.0150941399236544 | 2642 |
| 0813_05 | 0.7769018809000627 | 0.0024752209583918106 | -2.74178315885365 | 6115 |
| 0019_10 | 1.0269497871398912 | 0.0035145287712415874 | -4.228920210152857 | 6234 |

逐序列判断：

```text
相对 baseline:
    PSNR: 4 / 6 胜，0007_04 和 0813_05 略低。
    SSIM: 5 / 6 胜，只有 0007_04 略低。
    LPIPS: 4 / 6 胜，0044_11 和 0051_09 略高；0051_09 的差值只有 +0.000190 x1000，基本持平。

相对 old state:
    六个序列在 PSNR / SSIM / LPIPS 上全部改善。
    说明 state_warm_a04 的主要价值是修复 full state 替换主路径带来的退化；
    相比 baseline 是小幅、非一致但整体正向的提升。
```

## 2026-07-10 motion state 分支与 state MLP 分支说明

motion state 分支：

```text
输入:
    seq_pose_conds

作用:
    从一段 pose / motion 条件里提取当前帧的时序运动状态。
    它不直接预测 d_xyz / d_rotation / d_scaling，
    而是输出用于调制特征的 FiLM 参数。

流程:
    seq_pose_conds
        -> TemporalMotionStateEncoder
        -> motion state embedding
        -> gamma / beta

含义:
    motion state 分支回答的是“当前帧处在什么运动状态”。
    它提供全局/帧级的运动上下文，让后面的 state MLP 知道应该如何调整非刚性形变特征。
```

state MLP 分支：

```text
输入:
    与 baseline non-rigid MLP 相同的 Gaussian / pose / time features
    以及 motion state 分支产生的 FiLM gamma / beta

作用:
    生成一个被 motion state 调制过的中间形变特征 h_state。
    它不是单独替代输出头，而是产生一条辅助特征路径。

流程:
    features
        -> state MLP
        -> FiLM(gamma, beta)
        -> h_state

与 baseline 的关系:
    baseline 主路径:
        features -> original non-rigid MLP -> h_base

    state_warm_a04 最终融合:
        h = 0.6 * h_base + 0.4 * h_state

含义:
    state MLP 分支回答的是“在当前 motion state 下，这个 Gaussian 的非刚性形变特征应该怎样偏移”。
    它只作为 weak modulation 辅助主路径，不再完整替代 baseline。
```

简短总结：

```text
motion state 分支负责从时序 pose 条件中提取“帧级运动上下文”。
state MLP 分支负责把这个上下文注入到每个 Gaussian 的非刚性形变特征里。

前者产生调制信号，后者消费调制信号。
state_warm_a04 的关键是只用 0.4 alpha 弱融合，避免 full state 分支破坏 baseline 主路径。
```

## 2026-07-11 state_warm_a04 流程图

已保存简单流程图：

```text
note/state_warm_a04_flow.svg
note/state_warm_a04_flow_zh.svg
```

图中表达的主流程：

```text
features -> baseline non-rigid MLP -> h_base
seq_pose_conds -> TemporalMotionStateEncoder -> FiLM gamma/beta
features -> state MLP + FiLM -> h_state
h = 0.6 * h_base + 0.4 * h_state
h -> d_xyz / d_rotation / d_scaling
```

## 2026-07-11 warm 融合时间解释

容易误解的点：

```text
不是“训练 3000 步之后突然加入 0.4 权重的 state MLP 分支”。

state_warm_a04 的设置是：
    state_start_iter = 1500
    state_ramp_iter = 3000
    state_max_alpha = 0.4

也就是：
    iter < 1500:
        alpha = 0
        h = h_base

    1500 <= iter < 4500:
        alpha 从 0 线性增加到 0.4
        h = (1 - alpha) * h_base + alpha * h_state

    iter >= 4500:
        alpha = 0.4
        h = 0.6 * h_base + 0.4 * h_state
```

一句话：

```text
第 1500 步开始逐渐加入 state MLP 分支，
再经过 3000 步 warm ramp，
到第 4500 步以后固定为 0.4 权重。
```

## 2026-07-11 high-motion / boundary / high-error subset 评估

目的：

```text
全图平均指标提升很小，不代表 state_warm_a04 对局部没有帮助。
因此额外评估三个更贴近 motion state 假设的子集：
    1. high-motion frames
    2. boundary mask region
    3. high-error crop
```

评估脚本与输出：

```text
scripts/eval_state_subsets.py
note/state_subset_eval_20260711/subset_metrics.json
note/state_subset_eval_20260711/subset_metrics.md
```

比较对象：

```text
method:
    state_warm_a04 / 20260710_220456

baseline:
    orginal densify_until_iter=1500

baseline runs:
    0044_11 -> output/DNA-Rendering/0044_11/orginal/20260701_162750/
    0051_09 -> output/DNA-Rendering/0051_09/orginal/20260701_130518/
    0206_04 -> output/DNA-Rendering/0206_04/orginal/20260701_130518/
    0007_04 -> output/DNA-Rendering/0007_04/orginal/20260701_130518/
    0813_05 -> output/DNA-Rendering/0813_05/orginal/20260701_130518/
    0019_10 -> output/DNA-Rendering/0019_10/orginal/20260701_130518/
```

子集定义：

```text
high-motion frames:
    用 GT bkgd mask 的相邻帧 XOR / union 作为 motion score，
    每个序列取 top 25% frame。
    评估全图 L1 / PSNR / SSIM / LPIPS。

boundary mask region:
    用 GT mask 做 dilate - erode 得到边界带，
    boundary_width = 8。
    只评估 masked L1 / PSNR。

high-error crop:
    对每张图先用 baseline 与 GT 的误差图找最大误差 128x128 crop，
    然后在同一个 crop 上比较 baseline 与 state_warm_a04。
    评估 L1 / PSNR / SSIM / LPIPS。
```

整体结果，delta = state_warm_a04 - baseline：

```text
high_motion:
    L1      +0.000012    略差
    PSNR    +0.019877    略好
    SSIM    +0.000063    略好
    LPIPS   -0.037734 x1000    略好

boundary:
    L1      -0.000166    略好
    PSNR    +0.029871    略好

high_error_crop:
    L1      -0.000023    略好
    PSNR    +0.023158    略好
    SSIM    +0.001195    略好
    LPIPS   -1.267135 x1000    明显好于全图 LPIPS 幅度
```

逐序列观察：

```text
high_motion:
    6 个序列里 PSNR 有 4 个提升，SSIM 有 4 个提升，LPIPS 有 4 个提升，L1 有 3 个提升。
    说明快速动作帧上有弱正向趋势，但不是所有指标一致改善。

boundary:
    6 个序列里 5 个 boundary L1 / PSNR 改善。
    只有 0813_05 的边界区域略差。
    这是三个子集里最稳定支持 state_warm_a04 的结果。

high_error_crop:
    6 个序列里 L1 有 4 个改善，PSNR 有 5 个改善，SSIM 有 5 个改善，LPIPS 有 4 个改善。
    平均 LPIPS x1000 降低 1.267，说明 baseline 原本错误较大的局部 crop 上更容易看到收益。
```

结论：

```text
state_warm_a04 的全图平均收益很小，
但 boundary region 和 high-error crop 给出了更清晰的局部正向证据。

这支持一个更准确的说法：
    state_warm_a04 不是显著提升整体 novel-view rendering，
    但对衣服/人体边界以及 baseline 高误差局部有小幅修正作用。

high-motion frames 也有 PSNR / SSIM / LPIPS 的弱提升，
但 L1 没有同步改善，因此只能作为辅助证据，不能单独作为强结论。
```

## 2026-07-11 state_gate 消融实现与启动

动机：

```text
state_warm_a04 的固定 alpha 对所有 Gaussian 一视同仁：
    h = (1 - alpha) * h_base + alpha * h_state

subset 结果显示收益主要集中在 boundary / high-error crop。
因此下一版让每个 Gaussian 自己决定 state 分支强度：
    g_i = sigmoid(GateMLP(features_i, motion_state_t))
    h_i = h_base_i + alpha * g_i * (h_state_i - h_base_i)
```

已实现的新模式：

```text
experiment_name:
    state_gate

核心参数:
    --use_state_warm
    --use_state_gate
    --state_start_iter 1500
    --state_ramp_iter 3000
    --state_max_alpha 0.6
    --state_gate_hidden_dim 128
    --state_gate_bias -1.0
```

代码改动：

```text
nets/mlp_delta_non_rigid.py
    新增 point-wise state_gate:
        input  = concat(features_i, state_t)
        output = sigmoid(logit_i), shape [B, N, 1]

    当 use_state_gate=True:
        h = h_base + alpha * gate * (h_state - h_base)

    当 use_state_gate=False:
        保持原 state_warm:
        h = (1 - alpha) * h_base + alpha * h_state

arguments/__init__.py
    新增:
        use_state_gate
        state_gate_hidden_dim
        state_gate_bias

scene/gaussian_model.py
    将 gate 参数传给 NonrigidDeformer。

scripts/exps_dnarendering.sh
    新增 state_gate 模式。

scripts/eval_state_subsets.py
    支持 --state-experiment / --method-name，
    后续可直接评估 state_gate 的 high-motion / boundary / high-error subset。
```

初始化检查：

```text
用小张量 forward 通过。
gate 最后一层 weight 初始化为 0，bias = -1.0。
初始 gate mean = sigmoid(-1.0) = 0.268941。

最终有效 state 强度上限不是 0.6，
而是 alpha_max * gate。
训练开始后固定阶段的初始有效强度约为:
    0.6 * 0.269 = 0.161

这样比 state_warm_a04 的固定 0.4 更保守，
但 gate 可以学习把局部需要 state 的点抬高。
```

六序列训练已启动：

```text
RUN_TIME:
    20260711_032715

GPU2:
    0044_11 0051_09 0206_04

GPU3:
    0813_05 0007_04 0019_10

logs:
    logs/state/20260711_032715_DNA-Rendering_state_gate_gpu2.log
    logs/state/20260711_032715_DNA-Rendering_state_gate_gpu3.log

output:
    output/DNA-Rendering/<sequence>/state_gate/20260711_032715/
```

当前状态：

```text
两个训练进程已进入迭代。
GPU2/GPU3 均正常占用显存并运行。
后续完成后需要执行：
    1. 汇总 full novelview 指标
    2. 跑 subset eval:
       python scripts/eval_state_subsets.py \
           --state-experiment state_gate \
           --state-run 20260711_032715 \
           --method-name state_gate \
           --out-dir note/state_gate_subset_eval_20260711
    3. 对比 baseline / state_warm_a04 / state_gate
```

## 2026-07-11 实验结果阶段总结

当前结论按优先级排序：

```text
1. full state 分支失败。
   直接用 motion state/state MLP 替换主路径会明显破坏 baseline 优化路径。

2. state_warm_a04 是目前最稳的有效版本。
   它不替换 baseline 主路径，而是：
       h = 0.6 * h_base + 0.4 * h_state
   并且从 iter 1500 到 4500 线性 warm up。

3. state_warm_a04 的全图收益很小，但方向整体正向。
   它更像是“恢复 baseline + 小幅局部修正”，不是强全局提升。

4. subset 评估比全图指标更支持 state_warm_a04。
   boundary / high-error crop 的改善比全图平均更清楚。

5. state_gate 当前版本不成立。
   它只完成了 5/6 个序列，0206_04 在训练约 4100 iter 处 CUDA illegal memory access。
   已完成的 5 个序列平均指标也低于 baseline 和 state_warm_a04。
```

state_warm_a04 相对 baseline 的六序列结论：

```text
全图 novelview:
    PSNR / SSIM / LPIPS 都只有很小提升。
    PSNR 大约 +0.01 dB 量级。
    SSIM 大约 +0.00006 量级。
    LPIPS x1000 大约 -0.1 量级。

逐序列:
    PSNR: 4 / 6 胜。
    SSIM: 4 / 6 胜。
    LPIPS: 4 / 6 胜。

解释:
    这说明 state_warm_a04 没有显著改变全图平均质量，
    但它成功避免了 full state 的大幅退化。
```

state_warm_a04 的局部 subset 结果：

```text
high-motion frames:
    PSNR    +0.019877
    SSIM    +0.000063
    LPIPS   -0.037734 x1000
    L1      +0.000012

    结论:
        有弱正向趋势，但 L1 不一致，所以只能作为辅助证据。

boundary mask region:
    L1      -0.000166
    PSNR    +0.029871

    结论:
        6 个序列中 5 个改善，是最稳定的局部证据。

high-error crop:
    L1      -0.000023
    PSNR    +0.023158
    SSIM    +0.001195
    LPIPS   -1.267135 x1000

    结论:
        baseline 高误差局部更容易看到收益；
        LPIPS 局部改善明显大于全图平均。
```

state_gate 当前结果：

```text
run:
    state_gate / 20260711_032715

完成:
    0044_11
    0051_09
    0813_05
    0007_04
    0019_10

失败:
    0206_04

失败位置:
    约 iter 4100

错误:
    RuntimeError: CUDA error: an illegal memory access was encountered
```

state_gate 已完成 5 个序列上的全图平均，对齐同 5 个序列比较：

```text
baseline:
    PSNR  32.477631
    SSIM  0.974905
    LPIPS 0.027636

state_warm_a04:
    PSNR  32.487455
    SSIM  0.974970
    LPIPS 0.027528

state_gate:
    PSNR  32.448846
    SSIM  0.974888
    LPIPS 0.027769
```

state_gate 相对 baseline，5 序列平均：

```text
PSNR  -0.028785
SSIM  -0.000017
LPIPS +0.133435 x1000
```

state_gate 逐序列观察：

```text
0044_11:
    基本持平，PSNR +0.000355。

0051_09:
    退化，PSNR -0.070761。

0813_05:
    退化，PSNR -0.075668。

0007_04:
    PSNR -0.017439，但 SSIM/LPIPS 接近持平。

0019_10:
    略好，PSNR +0.019588。

0206_04:
    未完成，不能计入平均。
```

阶段性判断：

```text
state_warm_a04 可以作为当前主结果：
    全图小幅正向；
    局部 boundary / high-error crop 更有说服力；
    稳定完成六序列。

state_gate 当前不能作为正结果：
    训练稳定性不足；
    已完成序列平均指标低于 baseline；
    即使思想合理，当前实现/超参还没有证明有效。
```

下一步建议：

```text
优先不要继续扩大 state_gate 实验。
先定位 0206_04 CUDA illegal memory access。

如果继续做 gate，建议先做更保守版本：
    state_max_alpha = 0.4
    state_gate_bias = -1.5
    或者对 gate 输出加 clamp / detach 诊断

同时需要记录 gate mean / min / max，
否则无法判断 gate 是否在训练中异常放大。
```

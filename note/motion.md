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

## 2026-07-12 DNA / I3D state_warm_a04 脚本入口

本次只改启动脚本，不改训练代码和 state 分支实现。

DNA-Rendering 已有 state_warm_a04 消融入口，本次补充常用启动注释：

```bash
GPU_id=2 bash scripts/exps_dnarendering.sh state_warm_a04
SEQUENCES_OVERRIDE="0007_04 0019_10" GPU_id=2 bash scripts/exps_dnarendering.sh state_warm_a04
```

I3D-Human 新增同配置 state_warm_a04 消融入口：

```bash
GPU_id=3 bash scripts/exps_i3dhuman.sh state_warm_a04
SEQUENCES_OVERRIDE="ID1_1 ID1_2" GPU_id=3 bash scripts/exps_i3dhuman.sh state_warm_a04
```

I3D state_warm_a04 参数与 DNA 当前主实验保持一致：

```text
state_start_iter = 1500
state_ramp_iter  = 3000
state_max_alpha  = 0.4
final_eval_only  = 1
```

I3D 脚本行为：

```text
mode:
    state_warm_a04

训练参数:
    --use_state_warm
    --state_start_iter 1500
    --state_ramp_iter 3000
    --state_max_alpha 0.4

eval/save:
    只在最终 iter=15000 做 test/save，
    避免新增中间评估影响 I3D 现有实验成本。

日志:
    logs/state/<RUN_TIME>_I3D-Human_state_warm_a04_gpu<GPU_id>.log
```

对已有实验的影响：

```text
orginal / use_part_moe / part_moe_leg / part_moe_foot / part_moe_arm:
    保持原有启动方式和默认参数。

COMMON_TRAIN_ARGS:
    原本硬编码 test/save iterations，
    现在改成数组变量；
    非 state_warm_a04 模式下内容仍是 3000, part_moe_start_iter, iter。
```

补充：

```text
IDE 中显示的以下脚本当前不在仓库 scripts/ 目录：
    scripts/exps_dnarendering_part0.sh
    scripts/launch_dnarendering_part0_4gpu.sh

为了不影响其他实验代码，本次没有凭空新建这两个缺失脚本。
实际可用入口是：
    scripts/exps_dnarendering.sh
    scripts/exps_i3dhuman.sh
```

## 2026-07-12 消融实验入口约定

后续新增消融实验时，不使用下面这类 part0 / launch 临时脚本作为入口：

```text
scripts/exps_dnarendering_part0.sh
scripts/launch_dnarendering_part0_4gpu.sh
```

统一约定：

```text
1. 消融模式写进对应数据集主脚本的 MODE case：
       scripts/exps_dnarendering.sh
       scripts/exps_i3dhuman.sh
       scripts/exps_zjumocap.sh

2. 启动命令写在脚本顶部 Usage / 常用覆盖方式里。

3. 默认参数、日志路径、test/save iterations 都在同一个脚本内显式记录。

4. 不新增额外 launch 脚本，除非明确要求做批量调度封装。
```

## 2026-07-12 I3D / ZJU state_warm_a04 启动

本次目标：

```text
用 GPU 0, 1, 3 跑完 I3D-Human 和 ZJU-MoCap 上的 state_warm_a04，
完成后汇总评价指标。
```

入口约定：

```text
只使用数据集主脚本：
    scripts/exps_i3dhuman.sh
    scripts/exps_zjumocap.sh

不使用：
    scripts/exps_dnarendering_part0.sh
    scripts/launch_dnarendering_part0_4gpu.sh
```

本次统一运行时间戳：

```text
RUN_TIME = 20260712_222210
```

GPU 状态：

```text
GPU 0 / 1 / 3 空闲可用。
GPU 2 nvidia-smi 报 device handle error，本次不使用。
```

队列分配：

```text
GPU 0:
    I3D-Human: ID1_1 ID3_1
    ZJU-MoCap: CoreView_377 CoreView_392

GPU 1:
    I3D-Human: ID1_2
    ZJU-MoCap: CoreView_386 CoreView_393

GPU 3:
    I3D-Human: ID2_1
    ZJU-MoCap: CoreView_387 CoreView_394
```

ZJU state_warm_a04 已补到主脚本：

```text
state_start_iter = 1500
state_ramp_iter  = 3000
state_max_alpha  = 0.4
final_eval_only  = 1
```

注意：

```text
ZJU 默认总步数 iter=3000。
更正:
    代码里的 alpha 公式是:
        alpha = min(state_max_alpha, (iter - state_start_iter) / state_ramp_iter)

    所以沿用 1500 / 3000 / 0.4 时:
        iter 1500: alpha = 0
        iter 2100: alpha = 0.2
        iter 2700: alpha = 0.4
        iter 3000: alpha = 0.4

    之前“ZJU 终点 alpha 约 0.2”的判断不正确。
    ZJU 终点实际已经达到 max alpha 0.4。
```

启动结果：

```text
首次后台队列 PIDs:
    GPU0 queue: 469406
    GPU1 queue: 469512
    GPU3 queue: 469603

这些队列没有进入训练，未生成输出目录。
随后用前台短超时启动 I3D ID1_1 诊断，脚本能进入 train.py，
但 CUDA 初始化失败。
```

CUDA 诊断：

```text
nvidia-smi:
    GPU 0 / 1 / 3 可见且空闲。
    GPU 2 报错:
        Unable to determine the device handle for GPU2: 0000:B1:00.0: Unknown Error

PyTorch 自检:
    CUDA_VISIBLE_DEVICES=0 -> torch.cuda.is_available() = False, device_count = 0
    CUDA_VISIBLE_DEVICES=1 -> torch.cuda.is_available() = False, device_count = 0
    CUDA_VISIBLE_DEVICES=3 -> torch.cuda.is_available() = False, device_count = 0
    CUDA_VISIBLE_DEVICES=<GPU UUID> 也失败。

错误核心:
    CUDA initialization: CUDA unknown error
    Can't initialize NVML
    knn_cuda assert torch.cuda.is_available() is False
```

当前状态：

```text
I3D / ZJU state_warm_a04 脚本入口已准备好。
训练没有成功启动，因此还没有评价指标可以汇总。
需要先恢复系统 CUDA/NVML 状态，或者重置/隔离异常的 GPU2。
恢复后可以继续用 RUN_TIME=20260712_222210 或换新 RUN_TIME 复跑。
```

## 2026-07-12 GPU 0/1/3 状态复查

用户问题：

```text
除了卡2掉了，卡0，1，3也坏了吗
```

复查结果：

```text
nvidia-smi:
    GPU 0 / 1 / 3 仍能显示，温度/显存/进程信息正常。
    GPU 2 仍然报:
        Unable to determine the device handle for GPU2: 0000:B1:00.0: Unknown Error

PyTorch:
    CUDA_VISIBLE_DEVICES=0 -> available False, count 0
    CUDA_VISIBLE_DEVICES=1 -> available False, count 0
    CUDA_VISIBLE_DEVICES=3 -> available False, count 0
```

判断：

```text
不能直接说 GPU 0 / 1 / 3 硬件也坏了。
更准确说法是：
    GPU 0 / 1 / 3 在 nvidia-smi 层面可见；
    但当前 CUDA/NVML runtime 初始化失败，
    导致 PyTorch 无法使用任何单独指定的可用卡。

最可能是 GPU2 的底层异常污染了驱动/NVML 全局状态，
使 CUDA runtime 枚举设备失败。
```

## 2026-07-13 I3D / ZJU state_warm_a04 完成结果

完成情况：

```text
I3D-Human:
    run: 20260713_174844
    log: logs/state/20260713_174844_I3D-Human_state_warm_a04_gpu3.log
    GPU_id: 3
    sequences: ID1_1 ID1_2 ID2_1 ID3_1
    status: All sequences finished

ZJU-MoCap:
    run: 20260713_175243
    log: logs/state/20260713_175243_ZJU-MoCap_state_warm_a04_gpu2.log
    GPU_id: 2
    sequences: CoreView_377 CoreView_386 CoreView_387 CoreView_392 CoreView_393 CoreView_394
    status: All sequences finished
```

对比基线：

```text
I3D baseline:
    orginal / 20260616_212011

ZJU baseline:
    orginal / 20260618_233204_all

delta = state_warm_a04 - baseline
PSNR / SSIM 越高越好。
LPIPS 越低越好，所以 delta LPIPS x1000 为负数表示 state_warm_a04 更好。
```

I3D-Human novelview：

| Sequence | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS x1000 |
|---|---:|---:|---:|---:|---:|---:|
| ID1_1 | 32.044126 | 0.966966 | 0.025584 | +0.100544 | +0.000382 | -0.312622 |
| ID1_2 | 32.148249 | 0.966950 | 0.027135 | -0.096871 | -0.000185 | +0.094727 |
| ID2_1 | 31.607723 | 0.969412 | 0.028685 | +0.041105 | +0.000033 | -0.367033 |
| ID3_1 | 33.757207 | 0.966006 | 0.032919 | -0.017175 | -0.000021 | +0.081881 |
| Avg | 32.389326 | 0.967333 | 0.028581 | +0.006901 | +0.000053 | -0.125762 |

I3D-Human novelpose：

| Sequence | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS x1000 |
|---|---:|---:|---:|---:|---:|---:|
| ID1_1 | 30.072430 | 0.959931 | 0.031467 | +0.095783 | +0.000126 | -0.287849 |
| ID1_2 | 30.539378 | 0.959974 | 0.031341 | -0.043607 | +0.000031 | +0.483742 |
| ID2_1 | 28.375255 | 0.955890 | 0.039422 | +0.104563 | +0.000489 | -0.622556 |
| ID3_1 | 32.654811 | 0.959766 | 0.037225 | -0.058370 | -0.000475 | +0.483133 |
| Avg | 30.410469 | 0.958890 | 0.034864 | +0.024592 | +0.000042 | +0.014118 |

ZJU-MoCap test：

| Sequence | PSNR | SSIM | LPIPS | dPSNR | dSSIM | dLPIPS x1000 |
|---|---:|---:|---:|---:|---:|---:|
| CoreView_377 | 31.435721 | 0.973212 | 0.017981 | +0.010127 | +0.000098 | -0.288000 |
| CoreView_386 | 34.020936 | 0.969739 | 0.024785 | +0.000626 | +0.000129 | -0.258803 |
| CoreView_387 | 28.743490 | 0.955477 | 0.032468 | +0.040678 | +0.000101 | -0.192541 |
| CoreView_392 | 31.977012 | 0.964203 | 0.028458 | -0.005589 | -0.000230 | +0.001546 |
| CoreView_393 | 29.407416 | 0.954279 | 0.033703 | -0.018267 | -0.000175 | -0.013499 |
| CoreView_394 | 31.129175 | 0.957123 | 0.029885 | +0.030746 | +0.000205 | -0.405871 |
| Avg | 31.118959 | 0.962339 | 0.027880 | +0.009720 | +0.000021 | -0.192861 |

结论：

```text
I3D:
    novelview 平均小幅正向：
        PSNR +0.0069
        SSIM +0.000053
        LPIPS -0.126 x1000

    novelpose 平均 PSNR / SSIM 小幅正向：
        PSNR +0.0246
        SSIM +0.000042
        LPIPS +0.014 x1000

    逐序列不完全一致：
        ID1_1 / ID2_1 改善更明显；
        ID1_2 / ID3_1 有部分指标回退。

ZJU:
    六序列平均更稳定正向：
        PSNR +0.0097
        SSIM +0.000021
        LPIPS -0.193 x1000

    6 个序列里 4 个 PSNR 提升；
    6 个序列里 5 个 LPIPS 改善。

整体:
    state_warm_a04 在 I3D / ZJU 上没有带来显著大幅提升，
    但平均指标基本不掉，并有轻微正向趋势。
    这和 DNA 上的结论一致：
        weak state fusion 更像稳定的小幅调制，
        全图指标收益有限，
需要继续看 high-motion / boundary / high-error subset 才更可能体现价值。
```

## 2026-07-13 state_warm_a04 后续调参建议

问题：

```text
能不能通过调参再提高点
```

判断：

```text
可以尝试，但预期提升不会很大。
当前 I3D / ZJU 的全图平均已经基本贴近 baseline，
state_warm_a04 的收益量级是:
    PSNR: 约 +0.007 到 +0.025
    SSIM: 约 +0.00002 到 +0.00005
    LPIPS: 约 -0.13 到 -0.19 x1000，I3D novelpose 基本持平

这说明 state 分支不是明显欠调，而是全图指标里可提升空间很小。
调参目标应该是:
    1. 保持全图不掉；
    2. 减少 I3D 个别序列回退；
    3. 尽量扩大 LPIPS / 局部 subset 收益。
```

优先级 1：降低 I3D alpha，做更保守版本

```text
现象:
    I3D ID1_2 / ID3_1 有部分指标回退。

建议:
    state_warm_a03:
        state_start_iter = 1500
        state_ramp_iter  = 3000
        state_max_alpha  = 0.3

理由:
    对 I3D，a04 平均小幅正向但序列间不稳定。
    降到 0.3 可能牺牲一点正向序列收益，
    但更可能减少回退序列的负影响。

优先跑:
    I3D-Human 全 4 序列
```

优先级 2：ZJU 试更早 warm，但不提高 max alpha

```text
现象:
    ZJU 已经比较稳定正向。
    当前 alpha 在 2700 步达到 0.4，满 alpha 只覆盖最后约 300 步。

建议:
    state_warm_early_a04:
        state_start_iter = 1000
        state_ramp_iter  = 3000
        state_max_alpha  = 0.4

效果:
    iter 1000: alpha = 0
    iter 1600: alpha = 0.2
    iter 2200: alpha = 0.4
    iter 3000: alpha = 0.4

理由:
    ZJU 只有 3000 步，state 介入稍早一些，
    可能让后半段优化更充分利用 motion state。
    但 max alpha 不提高，避免破坏 baseline 主路径。
```

优先级 3：谨慎试 a05，不作为首选

```text
配置:
    state_warm_a05:
        state_start_iter = 1500
        state_ramp_iter  = 3000
        state_max_alpha  = 0.5

风险:
    DNA 早期实验显示强 state 容易破坏 baseline 路径。
    I3D 已有序列回退，a05 可能放大这种不稳定。

适合:
    只在 ZJU 或 high-motion subset 上验证，
    不建议直接作为全数据主实验。
```

不建议优先做：

```text
1. 继续加大到 alpha 0.6 / 0.8:
       风险大于收益。

2. 继续跑当前 state_gate:
       DNA 上已出现训练稳定性问题和平均指标下降。

3. 只看全图指标反复细调:
       当前收益太小，容易被随机性吞掉。
       应同步看 high-motion / boundary / high-error subset。
```

推荐下一组消融：

```text
I3D:
    state_warm_a03

ZJU:
    state_warm_early_a04

评价:
    1. 全图平均不能低于 baseline / state_warm_a04
    2. 单序列回退数量是否减少
    3. LPIPS 是否继续改善
    4. high-motion / boundary / high-error subset 是否扩大收益
```

## 2026-07-13 state 代码清理

用户要求：

```text
删除之前失败的 state 相关实验和代码，
只保留 state_warm_a04 的消融实验及其代码，
并把 state_warm_a04 相关实验名字改成 state。
```

执行范围：

```text
只清理代码入口和实验脚本。
没有删除历史 output / logs / metrics 结果目录。
历史 state_warm_a04 指标仍可用于对比和记录。
```

保留的 state 语义：

```text
新 state = 原 state_warm_a04

参数:
    state_start_iter = 1500
    state_ramp_iter  = 3000
    state_max_alpha  = 0.4

融合:
    h_base = baseline MLP(features)
    h_state = state-conditioned MLP(features, motion_state)
    h = (1 - alpha) * h_base + alpha * h_state

alpha:
    alpha = min(state_max_alpha, (iteration - state_start_iter) / state_ramp_iter)
```

删除/停用的旧 state 入口：

```text
旧 full-state:
    --use_state 的旧语义，即完全替换 baseline 主路径的 state MLP，已删除。

旧 warm 入口:
    --use_state_warm 已删除。

旧 gate 入口:
    --use_state_gate
    state_gate_hidden_dim
    state_gate_bias
    state_gate 分支已删除。

DNA 旧消融模式:
    state_warm
    state_warm_late
    state_warm_half
    state_warm_a02
    state_warm_a03
    state_warm_a04
    state_gate
```

当前脚本入口：

```bash
bash scripts/exps_dnarendering.sh state
bash scripts/exps_i3dhuman.sh state
bash scripts/exps_zjumocap.sh state
```

输出目录命名：

```text
之后新实验都写到:
    output/<Dataset>/<Sequence>/state/<RUN_TIME>/

日志写到:
    logs/state/<RUN_TIME>_<Dataset>_state_gpu<GPU_id>.log
```

代码清理点：

```text
arguments:
    保留 use_state / state_start_iter / state_ramp_iter / state_max_alpha。
    删除 use_state_warm / state_film / use_state_gate 等旧参数。

NonrigidDeformer:
    use_state=True 时直接走原 state_warm_a04 的弱融合路径。
    删除旧 full-state 替换路径和 gate 路径。

renderer / scene:
    只检查 use_state。
```

验证：

```text
bash -n:
    scripts/exps_dnarendering.sh
    scripts/exps_i3dhuman.sh
    scripts/exps_zjumocap.sh

py_compile:
    arguments/__init__.py
    scene/gaussian_model.py
    scene/__init__.py
    gaussian_renderer/__init__.py
    nets/mlp_delta_non_rigid.py
    scripts/eval_state_subsets.py

最小 forward:
    NonrigidDeformer(use_state=True) 输出 d_xyz / d_rotation / d_scaling shape 正常。
```

## 2026-07-13 part_state 消融启动

目标：

```text
新增同时使用 part_moe_leg 和 state 的消融实验:
    part_state

用 GPU 2 / 3 依次跑完三个数据集:
    DNA-Rendering
    I3D-Human
    ZJU-MoCap

完成后汇总评价指标。
```

part_state 定义：

```text
part_moe:
    enabled = 1
    part_label_schema = part_moe_leg
    num_parts = 7

state:
    enabled = 1
    state_start_iter = 1500
    state_ramp_iter  = 3000
    state_max_alpha  = 0.4

eval/save:
    final_eval_only = 1
```

脚本入口：

```bash
bash scripts/exps_dnarendering.sh part_state
bash scripts/exps_i3dhuman.sh part_state
bash scripts/exps_zjumocap.sh part_state
```

本次 RUN_TIME：

```text
20260713_220713
```

队列分配：

```text
GPU 2:
    DNA-Rendering: 0044_11 0051_09 0206_04
    I3D-Human: ID1_1 ID1_2
    ZJU-MoCap: CoreView_377 CoreView_386 CoreView_387

GPU 3:
    DNA-Rendering: 0813_05 0007_04 0019_10
    I3D-Human: ID2_1 ID3_1
    ZJU-MoCap: CoreView_392 CoreView_393 CoreView_394
```

验证：

```text
bash -n:
    scripts/exps_dnarendering.sh
    scripts/exps_i3dhuman.sh
    scripts/exps_zjumocap.sh
```

运行中状态：

```text
已完成:
    DNA-Rendering: 0044_11 0051_09 0813_05 0007_04 0019_10

GPU 2:
    0206_04 训练过程中报 CUDA error: unspecified launch failure。
    随后 nvidia-smi 报:
        Unable to determine the device handle for GPU2: Unknown Error
    判断为 GPU2 驱动侧掉卡/不可用，不再继续把剩余任务排到 GPU2。

恢复策略:
    先让 GPU3 完成原本队列:
        I3D-Human: ID2_1 ID3_1
        ZJU-MoCap: CoreView_392 CoreView_393 CoreView_394

    再用 GPU3 补跑 GPU2 未完成队列:
        DNA-Rendering: 0206_04
        I3D-Human: ID1_1 ID1_2
        ZJU-MoCap: CoreView_377 CoreView_386 CoreView_387

    继续沿用 RUN_TIME=20260713_220713，最终 metrics 仍汇总到同一组 part_state 目录。
```

当前阻塞：

```text
GPU3 原队列完成 I3D-Human/ID2_1 后，后续启动报:
    AssertionError: torch.cuda.is_available() is False

单独测试:
    CUDA_VISIBLE_DEVICES=3 torch.cuda.is_available() -> False
    CUDA_VISIBLE_DEVICES=1 torch.cuda.is_available() -> False
    torch.cuda.device_count() -> 0

nvidia-smi 仍能看到 GPU0/1/3，但 GPU2 报 Unknown Error；PyTorch/NVML 当前无法初始化 CUDA。
因此剩余 part_state 实验暂时无法在当前节点继续启动，需要恢复驱动/重启节点/换节点后补跑。
```

已完成指标：

| Dataset | Seq | Split | PSNR | SSIM | LPIPS |
|---|---:|---|---:|---:|---:|
| DNA-Rendering | 0007_04 | novelview | 29.5799 | 0.9588 | 0.0437 |
| DNA-Rendering | 0019_10 | novelview | 35.3723 | 0.9814 | 0.0207 |
| DNA-Rendering | 0044_11 | novelview | 33.0134 | 0.9782 | 0.0212 |
| DNA-Rendering | 0051_09 | novelview | 28.6450 | 0.9713 | 0.0309 |
| DNA-Rendering | 0813_05 | novelview | 36.1498 | 0.9872 | 0.0183 |
| I3D-Human | ID2_1 | novelview | 31.5835 | 0.9698 | 0.0284 |
| I3D-Human | ID2_1 | novelpose | 28.2920 | 0.9557 | 0.0397 |

已完成均值：

| Dataset | Split | N | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| DNA-Rendering | novelview | 5 | 32.5521 | 0.9754 | 0.0269 |
| I3D-Human | novelview | 1 | 31.5835 | 0.9698 | 0.0284 |
| I3D-Human | novelpose | 1 | 28.2920 | 0.9557 | 0.0397 |

待补跑：

```text
DNA-Rendering:
    0206_04

I3D-Human:
    ID1_1 ID1_2 ID3_1

ZJU-MoCap:
    CoreView_377 CoreView_386 CoreView_387 CoreView_392 CoreView_393 CoreView_394
```

已完成序列提升判断：

```text
参照选择:
    orginal:
        DNA 使用 20260701_130518；0044_11 使用 20260701_162750。
        I3D 使用 20260616_212011。

    state:
        DNA 使用 state_warm_a04/20260710_220456。
        I3D 使用 state_warm_a04/20260713_174844。

    part_moe_leg:
        DNA 使用最新可比 25000-step part_moe_leg 结果。
        I3D 使用 part_moe_leg/20260622_145118。
```

总体结论：

```text
相对 orginal:
    已完成 7 个评测项全部提升。
    平均变化:
        PSNR  +0.0586
        SSIM  +0.000432
        LPIPS -0.000630

相对 state:
    5/7 个评测项三项指标同时更好。
    平均变化:
        PSNR  +0.0308
        SSIM  +0.000312
        LPIPS -0.000411

相对 part_moe_leg:
    不稳定，不算明确提升。
    平均变化:
        PSNR  -0.0053
        SSIM  -0.000001
        LPIPS -0.000060
    只有 1/7 个评测项三项指标同时更好。
```

关键观察：

```text
DNA-Rendering:
    对 orginal / state 基本都有小幅收益。
    对 part_moe_leg:
        0007_04 / 0019_10 / 0044_11 / 0051_09 多数指标接近或略好，
        0813_05 明显不如 part_moe_leg。

I3D-Human ID2_1:
    对 orginal novelview/novelpose 都小幅提升。
    对 state 和 part_moe_leg:
        novelview LPIPS/SSIM 略好但 PSNR 略低；
        novelpose 三项都更差。

判断:
    part_state 已经说明“part_moe_leg + state”不会明显破坏 orginal，
    且比单独 state 通常更好。
    但目前还不能说它超过单独 part_moe_leg；
    下一步需要补齐剩余序列，并重点看 boundary/high-motion subset 是否有更稳定收益。
```

## 2026-07-15 part_state 补跑

用户要求：

```text
在卡3运行没运行完的序列。
```

恢复检查：

```text
nvidia-smi:
    GPU3 空闲，显存约 18 MiB。
    GPU2 已不再报 Unknown Error。

CUDA 自检:
    CUDA_VISIBLE_DEVICES=3 torch.cuda.is_available() -> True
    device_count -> 1
    CUDA tensor 分配正常。
```

补跑策略：

```text
沿用 RUN_TIME=20260713_220713。
只跑未完成序列，不重跑已完成指标。

GPU3 队列:
    DNA-Rendering: 0206_04
    I3D-Human: ID1_1 ID1_2 ID3_1
    ZJU-MoCap: CoreView_377 CoreView_386 CoreView_387 CoreView_392 CoreView_393 CoreView_394
```

补跑完成：

```text
完成时间:
    2026-07-15 20:04 CST

最终 metrics 文件数:
    20 / 20

补跑完成序列:
    DNA-Rendering: 0206_04
    I3D-Human: ID1_1 ID1_2 ID3_1
    ZJU-MoCap: CoreView_377 CoreView_386 CoreView_387 CoreView_392 CoreView_393 CoreView_394

日志:
    logs/state/20260713_220713_DNA-Rendering_part_state_gpu3.log
    logs/state/20260713_220713_I3D-Human_part_state_gpu3.log
    logs/state/20260713_220713_ZJU-MoCap_part_state_gpu3.log
```

完整 part_state 均值：

| Dataset | Split | N | PSNR | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| DNA-Rendering | novelview | 6 | 32.3903 | 0.9746 | 0.0280 |
| I3D-Human | novelview | 4 | 32.4071 | 0.9676 | 0.0284 |
| I3D-Human | novelpose | 4 | 30.4084 | 0.9589 | 0.0348 |
| ZJU-MoCap | test | 6 | 31.1636 | 0.9624 | 0.0281 |

## 2026-07-15 part_state vs 单独消融

参照：

```text
state:
    DNA-Rendering: state_warm_a04/20260710_220456
    I3D-Human: state_warm_a04/20260713_174844
    ZJU-MoCap: state_warm_a04/20260713_175243

part_moe_leg:
    DNA-Rendering: 最新可比 25000-step part_moe_leg
    I3D-Human: part_moe_leg/20260622_145118
    ZJU-MoCap: part_moe_leg/20260626_164558
```

相对单独 state：

| Dataset | Split | N | dPSNR | dSSIM | dLPIPS | 三指标全优 |
|---|---|---:|---:|---:|---:|---:|
| DNA-Rendering | novelview | 6 | +0.0847 | +0.000472 | -0.000673 | 6/6 |
| I3D-Human | novelview | 4 | +0.0178 | +0.000228 | -0.000171 | 2/4 |
| I3D-Human | novelpose | 4 | -0.0020 | +0.000010 | -0.000040 | 1/4 |
| ZJU-MoCap | test | 6 | +0.0446 | +0.000035 | +0.000181 | 2/6 |
| Overall | all | 20 | +0.0419 | +0.000200 | -0.000190 | 11/20 |

相对单独 part_moe_leg：

| Dataset | Split | N | dPSNR | dSSIM | dLPIPS | 三指标全优 |
|---|---|---:|---:|---:|---:|---:|
| DNA-Rendering | novelview | 6 | +0.0127 | +0.000069 | -0.000056 | 2/6 |
| I3D-Human | novelview | 4 | +0.0130 | +0.000117 | -0.000082 | 1/4 |
| I3D-Human | novelpose | 4 | -0.0460 | -0.000096 | +0.000192 | 0/4 |
| ZJU-MoCap | test | 6 | +0.0025 | -0.000045 | +0.000028 | 2/6 |
| Overall | all | 20 | -0.0020 | +0.000011 | +0.000014 | 5/20 |

判断：

```text
part_state 相比单独 state:
    整体有小幅提高，尤其 DNA 六条全优。
    I3D/ZJU 收益不稳定，但总体均值仍略好。

part_state 相比单独 part_moe_leg:
    不能认为有稳定提高。
    DNA / I3D novelview 接近持平略好；
    I3D novelpose 明显更差；
    ZJU 基本持平，PSNR 略好但 SSIM/LPIPS 略差。

结论:
    part_state 当前更像是“part_moe_leg 基础上加入 state 后，仍保持总体接近 part_moe_leg，并明显强于单独 state”。
    但它还没有证明组合优于单独 part_moe_leg。
```

## 2026-07-15 state 分支结构说明

state 实验新增的分支：

```text
不是新增一个直接预测 Gaussian 输出的完整 head。
新增的是 NonrigidDeformer 内部的一条 state-conditioned hidden feature branch:

    baseline branch:
        features -> baseline MLP -> h_base

    state branch:
        features -> copied MLP linear layers + FiLM(motion_state) -> h_state

    warm fusion:
        h = (1 - alpha) * h_base + alpha * h_state

    shared output heads:
        h -> gaussian_warp     -> d_xyz
        h -> gaussian_rotation -> d_rotation
        h -> gaussian_scaling  -> d_scaling
```

输入：

```text
1. 每个 Gaussian 的常规非刚性输入 features:
       x_emb
       pose feature
       sequence pose feature
       sequence xyz/KNN feature

   拼接后得到:
       features: [B, N, input_ch]

2. motion state 条件:
       state_conds

   当前代码里 renderer 传入:
       state_conds = cond_dict[pose_id].get("state_conds", seq_pose_conds)

   如果没有单独 state_conds，就回退使用 seq_pose_conds。
   在 NonrigidDeformer 中 reshape 为:
       state_seq: [B, L, C]

   C = 3 * (N_JOINT + 1) * time_step_num
       smpl:  N_JOINT=23
       smplx: N_JOINT=54
```

motion state 编码：

```text
TemporalStateEncoder:
    state_seq [B, L, C]
        -> 1D temporal conv / TCN
        -> 取最后一个时间步特征
        -> Linear MLP
        -> state embedding [B, state_dim]

默认:
    state_dim = 64
    state_hidden_dim = 128
    state_layers = 3
```

FiLM 调制：

```text
state_film_layer(state):
    [B, state_dim] -> [B, num_state_layers, 2, W]

每一层得到:
    gamma, beta

对 state branch 每层隐藏特征做:
    h_state = gamma * h_state + beta
    h_state = ReLU(h_state)

初始化:
    state branch 的 linear layer 从 baseline MLP 拷贝。
    state_film_layer weight/bias 初始化为 0。

所以训练初期:
    gamma = 1
    beta = 0
    h_state 接近 h_base
```

输出：

```text
state 分支本身的直接输出:
    h_state: [B, N, W]

融合后的输出:
    h: [B, N, W]

最终由共享 head 输出:
    d_xyz:      [B, N, 3]
    d_rotation: [B, N, 4]
    d_scaling:  [B, N, 3]
```

warm 融合：

```text
只有 iteration >= state_start_iter 才启用。

alpha = clamp((iteration - state_start_iter) / state_ramp_iter, 0, state_max_alpha)

当前 state / part_state 使用:
    state_start_iter = 1500
    state_ramp_iter  = 3000
    state_max_alpha  = 0.4

所以:
    1500 step 前: 不用 state branch
    1500 step 后: alpha 逐步增大
    达到上限后: h = 0.6 * h_base + 0.4 * h_state
```

直观理解：

```text
baseline MLP 负责稳定的空间/姿态非刚性表示。
state branch 用一段时间窗口里的 motion pose/state 编码，生成 FiLM 参数，
让隐藏特征根据当前运动状态轻微偏移。

它不是替换 baseline，而是最多以 0.4 权重混入 state-conditioned hidden feature。
```

state_warm_a04_flow_zh.svg 中两个“state”相关框的关系：

```text
图里的 “Motion state 分支” 和 “State MLP 分支” 不是同一个 MLP。
它们是两个模块，前者给后者提供调制条件。

1. Motion state 分支:
       对应代码里的 TemporalStateEncoder + state_film_layer。

       输入:
           state_conds / seq_pose_conds

       输出:
           state embedding
           FiLM 参数 gamma / beta

       作用:
           编码这一帧/这一段时间的运动状态。
           它不直接预测 d_xyz / d_rotation / d_scaling。

2. State MLP 分支:
       对应代码里的 self.state_layers。

       输入:
           和 baseline MLP 相同的 Gaussian features

       中间调制:
           每一层用 Motion state 分支给出的 gamma / beta 做 FiLM:
               h_state = gamma * h_state + beta

       输出:
           h_state hidden feature

3. 二者关系:
       Motion state 分支是条件生成器。
       State MLP 分支是被条件调制的特征分支。

       motion state -> gamma/beta -> 调制 State MLP -> h_state

4. 最终输出:
       h_state 不直接作为最终 Gaussian 变形。
       它先和 baseline 的 h_base 融合:
           h = (1 - alpha) * h_base + alpha * h_state

      再经过共享输出头得到:
          d_xyz / d_rotation / d_scaling
```

motion state 编码的具体实现：

```text
1. state_conds 当前通常就是 seq_pose_conds

renderer 里:
    state_conds = cond_dict[pose_id].get("state_conds", seq_pose_conds)

因为数据读取阶段没有额外写入 state_conds，
所以当前实验实际使用 seq_pose_conds 作为 motion state 输入。
```

seq_pose_conds 的来源：

```text
get_seq_pose_xyz_cond 会对当前帧和历史帧做差:

    cur_id    = pose_index - i * time_step
    former_id = pose_index - (i + 1) * time_step

    delta_pose_mat = cur_pose_mat @ inv(former_pose_mat)
    posedelta = matrix_to_axis_angle(delta_pose_mat)

也就是说，seq_pose_conds 不是单帧绝对 pose，
而是不同时间间隔上的关节旋转变化量。

它描述的是:
    这一帧相对过去几帧，身体各关节怎么动了。

这比单帧 pose 更接近“运动状态”，包含速度/方向/近期动态。
```

张量形状：

```text
seq_pose_conds 原始形状大致是:
    [B, time_step_num, seq_len, J+1, 3]

其中:
    time_step_num: 多种时间间隔
    seq_len: 每种时间间隔回看多少步
    J+1: SMPL/SMPL-X 关节数加 root
    3: axis-angle 旋转差

进入 state branch 前:
    B = state_conds.shape[0]
    L = state_conds.shape[1]
    state_seq = state_conds.reshape(B, L, -1)

所以变成:
    state_seq: [B, L, C]

C = 3 * (J + 1) * time_step_num

注意:
    这里 L 通常对应 seq_len / 时间序列长度。
    C 里包含所有关节、所有时间间隔的旋转差特征。
```

TemporalStateEncoder 怎么编码运动状态：

```text
输入:
    state_seq [B, L, C]

代码:
    x = state_seq.permute(0, 2, 1)

变成 Conv1d 需要的格式:
    x [B, C, L]

然后经过 3 层 temporal conv:
    kernel_size = 3
    dilation = 1, 2, 4

这些卷积沿 L 这个时间维滑动，
所以能看见相邻时间步以及更长间隔的运动变化模式。

直观上它学习:
    哪些关节最近变化大
    哪些方向在持续运动
    快动作/慢动作/停顿
    多时间尺度下的 pose delta pattern

最后:
    feat_t = feat[:, :, -1]

取最后一个时间位置的特征，
表示“到当前帧为止”的运动状态摘要。

再过 MLP:
    Linear -> ReLU -> Linear

得到:
    state embedding [B, state_dim]
```

从 state embedding 到 FiLM 参数：

```text
state_film_layer:
    [B, state_dim] -> [B, num_state_layers * 2 * W]

reshape:
    film = film.view(B, num_state_layers, 2, W)

对每个 state MLP layer 都得到:
    gamma: [B, W]
    beta:  [B, W]

应用到每个 Gaussian 的隐藏特征:
    gamma = 1 + gamma
    beta  = beta
    h_state = gamma * h_state + beta

因为 gamma/beta 是从帧级 motion state 来的，
同一帧内所有 Gaussian 共享这一组运动上下文；
但它们各自的 h_state 不同，所以调制后的结果仍然是点相关的。
```

为什么这叫“编码运动状态”：

```text
输入不是图像，也不是高误差区域标签，
而是一段时间窗口里的关节旋转差序列。

TemporalStateEncoder 用时间卷积把这些 pose delta 压缩成一个 state embedding。
这个 embedding 表示当前帧的整体运动状态：
    快/慢
    动作方向
    哪些关节变化明显
    多时间尺度的近期运动趋势

随后 FiLM 参数把这个状态注入到非刚性 MLP 的隐藏层，
让同样的 Gaussian features 在不同运动状态下产生不同的非刚性变形特征。
```

关于 state_conds 回退到 seq_pose_conds 是否有问题：

```text
结论:
    不是实现 bug，shape 和语义都能成立；
    但它不是一个额外独立的新 motion signal，而是对已有 seq_pose_conds 的另一种编码。

shape 上:
    get_seq_pose_xyz_cond 生成的 seq_pose_conds 形状约为:
        [B, seq_len, time_step_num, J+1, 3]

    state branch 中:
        B = state_conds.shape[0]
        L = state_conds.shape[1]
        state_seq = state_conds.reshape(B, L, -1)

    得到:
        [B, seq_len, 3 * (J+1) * time_step_num]

    这正好匹配:
        state_input_dim = 3 * (N_JOINT + 1) * time_step_num

所以回退到 seq_pose_conds 不会造成维度错误。

语义上:
    seq_pose_conds 是多时间间隔下的 pose delta，
    本身就是运动状态信息。
    用它作为 state_conds 是合理的。

局限:
    1. 它和原 SeqPoseEncoder 使用的是同一份输入，
       因此 state 分支的信息增量主要来自不同编码方式和 FiLM 调制，
       不是来自额外传感器/额外 motion label。

    2. 它是帧级/序列级 motion context，
       同一帧所有 Gaussian 共享同一个 state embedding。
       它不知道哪个 Gaussian 位于衣服边界或高误差区域。

    3. 当前 state 只包含 pose delta，
       没有显式使用局部 xyz velocity、part label、mask boundary 或 per-point error。

判断:
    作为 state_warm_a04 的保守实现，这一步可以接受；
    如果下一版想让 state 更强，应该考虑构造显式 state_conds，
    例如拼接 seq_pose_conds + 全局运动幅度 + per-part motion summary，
    或者进一步做 point-wise state/gate。
```

## 2026-07-15 GaussianAvatar 对显式 state_conds 的参考价值

问题：

```text
/media/image/mxz/human/GaussianAvatar 的论文和代码能不能用来显式构造 state_conds，
替换之前被迫使用的 seq_pose_conds？
```

结论：

```text
可以参考，而且方向是对的；
但不建议直接把 GaussianAvatar 的整套 pose encoder / UV decoder 搬进 SeqAvatar。

更合适的做法是借它的“显式 SMPL posed surface / position map”思想，
用 SMPL/SMPL-X 的逐帧显式几何运动来构造 state_conds。
这样 state_conds 不再只是 seq_pose_conds 的别名，而是有额外 motion signal。
```

GaussianAvatar 可借鉴的点：

```text
1. 它为每一帧生成 posed SMPL position map:
       scripts/gen_pose_map_our_smpl.py
       inp_posemap_<res>_<frame>.npz

2. 训练时读取 inp_pos_map:
       scene/dataset_mono.py

3. 用 pose_encoder 把 posed position map 编成 pose_featmap:
       model/avatar_model.py
       pose_featmap = self.pose_encoder(inp_posmap)

4. pose_featmap 和可学习 geom_featmap 一起进入 Gaussian decoder:
       model/network.py
       pix_feature = pose_featmap + geom_featmap

本质上，GaussianAvatar 不是只用关节 pose 向量，
而是把当前帧 SMPL 表面的显式空间状态作为 pose-dependent 条件。
```

SeqAvatar 当前接口是否支持替换：

```text
支持。

gaussian_renderer/__init__.py 里已经有：
    state_conds = pc.cond_dict[pose_id].get('state_conds', seq_pose_conds)

nets/mlp_delta_non_rigid.py 里也有：
    if state_conds is None:
        state_conds = seq_pose_conds

所以只要 dataset_readers.py 在 cond_dict 里写入 state_conds，
state 分支就会优先使用显式 state_conds。
```

需要注意的维度问题：

```text
当前 StateEncoder 默认输入维度是：
    3 * (N_JOINT + 1) * time_step_num

这正好对应 seq_pose_conds reshape 后的维度。

如果新的 state_conds 拼接了 xyz velocity、part summary、normal delta 等额外特征，
就必须同步修改 state_input_dim：
    1. 增加命令行参数，例如 --state_input_dim；
    2. 或根据 state_cond_mode 计算输入维度；
    3. 或保留同维度，把显式运动先投影回原维度。
```

推荐实现顺序：

```text
第一步，低风险显式 state_conds：
    state_conds = seq_pose_conds + 从 seq_xyz_conds 汇总出的显式运动统计

可加入：
    global xyz velocity mean / max
    per-part xyz velocity mean / max
    joint pose delta norm
    root / limb motion magnitude

这一步不需要生成 UV pose map，也不改变渲染主干。

第二步，GaussianAvatar-style surface state：
    用每帧 SMPL posed vertices 或 position map 计算：
        当前帧 posed surface
        前一帧 posed surface
        surface velocity
        normal / tangent change
        per-part motion summary

然后聚合成当前 StateEncoder 可吃的 frame-level state_conds。

第三步，更强但改动更大：
    把 surface motion 通过 nearest SMPL vertex / LBS weights 分配到每个 Gaussian，
    做 point-wise state_conds 或 point-wise gate。
    这比 frame-level state 更符合边界、袖口、裤腿、裙摆等局部收益目标。
```

判断：

```text
替换 seq_pose_conds 是可行的。

最稳的下一版不是直接接 GaussianAvatar 的 UNet，
而是在 SeqAvatar 的 dataset_readers.py 中显式写 state_conds：
    seq_pose_conds
    + seq_xyz_conds 的运动幅度统计
    + per-part motion summary

这样能验证“显式 motion state 是否比复用 seq_pose_conds 更有效”，
同时不影响 baseline、part_moe_leg 和现有 state 主干。
```

## 2026-07-15 显式 state_conds 计算方案

当前已有运动量：

```text
get_seq_pose_xyz_cond(pose_index, time_steps, interval, get_pose_xyz_func, ...)

对每个 time_step 和历史窗口 i：
    cur_id    = pose_index - i * time_step
    former_id = pose_index - (i + 1) * time_step

读取：
    cur_pose_mat, cur_obs_xyz
    former_pose_mat, former_obs_xyz

计算：
    delta_pose_mat = cur_pose_mat @ inverse(former_pose_mat)
    posedelta      = matrix_to_axis_angle(delta_pose_mat)
    xyz_delta      = cur_obs_xyz - former_obs_xyz
```

当前张量含义：

```text
seq_pose_conds:
    形状约为 [1, seq_len, time_step_num, J+1, 3]
    含义是多时间尺度的关节相对旋转。

seq_xyz_conds:
    形状约为 [V, seq_len, time_step_num, 3]
    含义是多时间尺度的 SMPL 顶点显式空间位移/速度。
```

推荐第一版显式 state_conds：

```text
目标：
    让 state_conds 不再等于 seq_pose_conds，
    而是包含 pose delta + 显式 surface motion summary。

对每个历史时间 l、每个 time_step s：
    pose_delta[l, s] = seq_pose_conds[0, l, s]          # [J+1, 3]
    xyz_delta[l, s]  = seq_xyz_conds[:, l, s]           # [V, 3]

计算 surface motion 统计：
    speed_v = ||xyz_delta||_2                           # [V]
    global_mean_speed = mean(speed_v)
    global_max_speed  = max(speed_v)
    global_p90_speed  = percentile(speed_v, 90)
    global_xyz_mean   = mean(xyz_delta, dim=V)          # [3]

如果有 part labels / vertex groups，再计算：
    part_mean_speed[p] = mean(speed_v[part == p])
    part_max_speed[p]  = max(speed_v[part == p])
    part_xyz_mean[p]   = mean(xyz_delta[part == p])     # [3]
```

拼接方式：

```text
state_feat[l, s] =
    flatten(pose_delta[l, s])
    + global_mean_speed
    + global_max_speed
    + global_p90_speed
    + global_xyz_mean
    + part motion summary

最后：
    state_conds = stack(state_feat over l and s)

建议保持接口形状：
    state_conds: [1, seq_len, time_step_num, C]

进入 NonrigidDeformer 后仍然：
    B = state_conds.shape[0]
    L = state_conds.shape[1]
    state_seq = state_conds.reshape(B, L, -1)
```

维度处理：

```text
如果 C 仍然等于 (J+1)*3，可以不改 StateEncoder 输入维度，
但这样显式 motion 必须先投影/压缩回原维度，不够直观。

更推荐：
    增加 state_cond_dim / state_input_dim 配置，
    或根据 state_cond_mode 自动计算：

    state_input_dim =
        time_step_num * C

其中：
    C = pose_dim + global_motion_dim + part_motion_dim

这样 state_conds 的新增信息是明确可控的。
```

为什么这样比直接回退 seq_pose_conds 更好：

```text
seq_pose_conds 只看关节相对旋转；
显式 state_conds 额外看 SMPL surface 真实空间位移。

这能表达：
    当前帧整体运动强不强
    哪些 part 动得更明显
    是否存在局部大幅运动
    衣服边界、袖口、裤腿、裙摆附近更可能需要 state 调制

这和 GaussianAvatar 的核心启发一致：
    不只使用抽象 pose 向量，
    还使用 posed surface / position map 代表当前帧几何状态。
```

## 2026-07-15 是否必须计算 part-level surface motion

问题：

```text
计算 state_conds 的时候，有没有必要按照 part 算 part-level surface motion？
```

判断：

```text
有价值，但不建议作为第一版显式 state_conds 的硬依赖。

第一版优先做：
    seq_pose_conds
    + global surface motion summary

然后再做 ablation：
    state_global
    state_global_part

用实验判断 part-level motion 是否真的带来增益。
```

理由：

```text
part-level surface motion 的优点：
    1. 比 global motion 更细。
       同一帧可能只是腿、袖口、裙摆在动，全局 mean 会把局部运动稀释掉。

    2. 和已有 part_moe_leg / part_state 方向一致。
       state 分支可以知道“哪个身体区域当前运动强”，
       不只是知道“这一帧整体运动强”。

    3. 对 boundary / high-motion subset 更可能有效。
       衣服边界、袖口、裤腿、裙摆通常和局部 part motion 更相关。

part-level surface motion 的风险：
    1. 需要可靠的 SMPL vertex segmentation。
       当前 part_moe 使用的是 Gaussian part label，
       state_conds 计算阶段需要的是 SMPL 顶点 part label，两者不是同一个对象。

    2. 会增加 state_conds 维度。
       维度变大后必须同步修改 state_input_dim，
       而且在小数据集上可能更容易过拟合。

    3. part 划分如果太粗或不稳定，可能引入噪声。
       例如衣服/裙摆并不严格等于 SMPL leg/body part。
```

推荐顺序：

```text
第一步 state_global：
    pose_delta
    + global_mean_speed
    + global_max_speed
    + global_p90_speed
    + global_xyz_mean

第二步 state_global_part：
    在 state_global 基础上加入 per-part:
        part_mean_speed
        part_max_speed
        part_xyz_mean

第三步 point-wise state/gate：
    如果 part-level 有收益，再考虑把 per-part 或 per-vertex motion
    通过 nearest SMPL vertex / LBS weights 分配到每个 Gaussian。
```

结论：

```text
不必一开始就按 part 算。

先用 global surface motion 证明“显式 state_conds 比复用 seq_pose_conds 有效”。
如果 global 只在少数序列有收益，或者 high-motion/boundary subset 仍不明显，
再加入 part-level surface motion。
```

## 2026-07-15 state_conds 消融实验

用户要求：

```text
新增消融实验 state_conds。
只在 DNA-Rendering 上跑。
用 GPU3。
最后汇总 novelview 全图评价指标，
并汇总 high-motion / boundary / high-error subset 结果。
```

实验目标：

```text
验证显式计算出来的 state_conds 是否比 state 实验中回退使用 seq_pose_conds 更有效。
```

本次 state_conds 计算方式：

```text
state_cond_mode = global_surface

对每个历史帧 l、每个 time_step s：
    pose_feats = flatten(seq_pose_conds[0, l, s])       # [3 * (J+1)]
    xyz_delta  = seq_xyz_conds[:, l, s]                 # [V, 3]
    speed      = ||xyz_delta||_2                        # [V]

额外拼接：
    global_mean_speed = mean(speed)
    global_max_speed  = max(speed)
    global_p90_speed  = quantile(speed, 0.9)
    global_xyz_mean   = mean(xyz_delta, dim=V)          # [3]

所以：
    state_conds = concat(
        pose_feats,
        global_mean_speed,
        global_max_speed,
        global_p90_speed,
        global_xyz_mean
    )
```

维度：

```text
SMPL-X:
    J+1 = 55
    pose_dim = 55 * 3 = 165
    global_surface_dim = 6
    per-time-step C = 171
    time_step_num = 3
    StateEncoder input_dim = 171 * 3 = 513

原 state:
    StateEncoder input_dim = 165 * 3 = 495
```

代码入口：

```text
scripts/exps_dnarendering.sh state_conds

该模式：
    experiment_name = state_conds
    --use_state
    --state_cond_mode global_surface
    --state_start_iter 1500
    --state_ramp_iter 3000
    --state_max_alpha 0.4
    final_eval_only = 1
```

实现约束：

```text
默认 state_cond_mode=pose。
因此 orginal / part_moe / part_moe_leg / state / part_state 不会生成新的 state_conds，
也不会改变旧 StateEncoder 输入维度。

只有 state_conds 模式才会在 dataset_readers.py 的 cond_dict 里写入显式 state_conds。
```

验证：

```text
py_compile 通过：
    arguments/__init__.py
    scene/__init__.py
    scene/dataset_readers.py
    scene/gaussian_model.py
    nets/mlp_delta_non_rigid.py

shape 自检：
    state_conds: [1, 8, 3, 171]
    state_seq reshape 后输入维度: 513

knn_cuda 导入检查：
    PATH=/media/image/mxz/.conda/envs/seqavatar/bin:$PATH
    import knn_cuda OK
```

运行结果：

```text
RUN_TIME = 20260715_231745
GPU = 3
入口 = GPU_id=3 bash scripts/exps_dnarendering.sh state_conds
日志 = logs/state/20260715_231745_DNA-Rendering_state_conds_gpu3.log
输出 = output/DNA-Rendering/<seq>/state_conds/20260715_231745/

六个序列均已完成训练和 render.py novelview 评价。
```

state_conds 全图 novelview 指标：

| Sequence | PSNR | SSIM | LPIPS x1000 |
|---|---:|---:|---:|
| 0007_04 | 29.558240 | 0.958422 | 45.119477 |
| 0019_10 | 35.258634 | 0.980866 | 21.278971 |
| 0044_11 | 32.931219 | 0.977931 | 21.489767 |
| 0051_09 | 28.631773 | 0.971384 | 31.043444 |
| 0206_04 | 31.429339 | 0.969940 | 33.865396 |
| 0813_05 | 36.095236 | 0.986987 | 18.354361 |
| Average | 32.317407 | 0.974255 | 28.525236 |

全图对比结论：

```text
对比 baseline 平均:
    baseline:    PSNR 32.311109, SSIM 0.974181, LPIPS x1000 28.605086
    state_conds: PSNR 32.317407, SSIM 0.974255, LPIPS x1000 28.525236

变化:
    ΔPSNR  +0.006298
    ΔSSIM  +0.000074
    ΔLPIPS -0.079850

全图平均基本持平，略好。
```

对比旧 state 平均：

```text
旧 state:
    PSNR 31.639918, SSIM 0.971508, LPIPS x1000 31.221671

state_conds:
    PSNR 32.317407, SSIM 0.974255, LPIPS x1000 28.525236

变化:
    ΔPSNR  +0.677489
    ΔSSIM  +0.002747
    ΔLPIPS -2.696435

显式 state_conds 明显修复了旧 state 全图指标下降的问题。
```

subset 评价输出：

```text
note/state_conds_subset_20260715_231745/subset_metrics.json
note/state_conds_subset_20260715_231745/subset_metrics.md
```

state_conds 相对 baseline 的 subset 结果：

| Subset | ΔL1 | ΔPSNR | ΔSSIM | ΔLPIPS x1000 |
|---|---:|---:|---:|---:|
| high_motion | -0.000014 | +0.047058 | +0.000195 | -0.090207 |
| boundary | -0.000011 | -0.000471 | NA | NA |
| high_error_crop | -0.000192 | +0.027593 | +0.001680 | -2.156282 |

对比旧 state_warm_a04 的 subset：

```text
high_motion:
    旧 state_warm_a04: ΔL1 +0.000012, ΔPSNR +0.019877, ΔSSIM +0.000063, ΔLPIPS -0.037734
    state_conds:       ΔL1 -0.000014, ΔPSNR +0.047058, ΔSSIM +0.000195, ΔLPIPS -0.090207
    结论: state_conds 更好。

boundary:
    旧 state_warm_a04: ΔL1 -0.000166, ΔPSNR +0.029871
    state_conds:       ΔL1 -0.000011, ΔPSNR -0.000471
    结论: state_conds 更弱，几乎没有 boundary 收益。

high_error_crop:
    旧 state_warm_a04: ΔL1 -0.000023, ΔPSNR +0.023158, ΔSSIM +0.001195, ΔLPIPS -1.267135
    state_conds:       ΔL1 -0.000192, ΔPSNR +0.027593, ΔSSIM +0.001680, ΔLPIPS -2.156282
    结论: state_conds 更好。
```

最终判断：

```text
显式计算 state_conds 是有效方向。

它让全图指标不再像旧 state 那样明显下降，
并且 high_motion / high_error_crop subset 比旧 state_warm_a04 更好。

但当前 global_surface 只用全局 surface motion summary，
boundary 区域收益反而弱于旧 state_warm_a04。
这说明只做全局 motion summary 还不够定位衣服边界/袖口/裤腿/裙摆。

下一步如果继续改 state_conds，优先加：
    1. part-level surface motion；
    2. 或 point-wise gate / per-Gaussian local motion。
```

## 2026-07-16 state_conds 正确比较口径修正

用户指出：

```text
state_conds 应该和 state_warm_a04 比较实验结果。
state_warm_a04 现在已经改名为 state。
```

修正：

```text
主要比较口径应为：
    state_conds / 20260715_231745
    vs
    state_warm_a04 / 20260710_220456

baseline 只作为参考，不作为判断 state_conds 是否有效的主要对象。
旧 full-state / 20260708_001225 也不是当前 state 主线。
```

全图平均对比，delta = state_conds - state_warm_a04：

| Method | PSNR | SSIM | LPIPS x1000 |
|---|---:|---:|---:|
| state_warm_a04 / current state | 32.322408 | 0.974247 | 28.503387 |
| state_conds | 32.317407 | 0.974255 | 28.525236 |
| delta | -0.005001 | +0.000008 | +0.021849 |

逐序列全图 delta，delta = state_conds - state_warm_a04：

| Sequence | ΔPSNR | ΔSSIM | ΔLPIPS x1000 |
|---|---:|---:|---:|
| 0007_04 | +0.053756 | +0.000173 | -0.043092 |
| 0019_10 | -0.027501 | -0.000113 | +0.346109 |
| 0044_11 | -0.048977 | -0.000005 | +0.037288 |
| 0051_09 | -0.057911 | -0.000156 | -0.048816 |
| 0206_04 | +0.031166 | +0.000081 | -0.119052 |
| 0813_05 | +0.019462 | +0.000068 | -0.041345 |

subset 直接对比，delta = state_conds - state_warm_a04：

| Subset | ΔL1 | ΔPSNR | ΔSSIM | ΔLPIPS x1000 |
|---|---:|---:|---:|---:|
| high_motion | -0.000026 | +0.027181 | +0.000132 | -0.052473 |
| boundary | +0.000154 | -0.030342 | NA | NA |
| high_error_crop | -0.000169 | +0.004435 | +0.000486 | -0.889147 |

修正结论：

```text
按当前主线 state_warm_a04/current state 对比：

1. 全图平均:
   state_conds 和 current state 基本打平；
   PSNR 略低，SSIM 几乎一样，LPIPS 略差。
   因此不能说 state_conds 全图优于 current state。

2. high_motion subset:
   state_conds 更好。

3. high_error_crop subset:
   state_conds 更好，尤其 LPIPS 改善更明显。

4. boundary subset:
   state_conds 更差。

所以 state_conds 的价值不是全图平均提升，
而是显式 surface motion 对 high-motion / high-error crop 更有帮助。
但它削弱了 current state 在 boundary 上的收益。

下一步若继续该方向，不应只加 global surface motion；
更应该加入 part-level surface motion 或 point-wise/local gate，
目标是保住 high-motion/high-error 收益，同时恢复 boundary 收益。
```

## 2026-07-16 h_state 含义解释

问题：

```text
h_state 是运动状态特征吗？
它是怎么一步步获得的？
```

结论：

```text
h_state 不是纯粹的“运动状态特征”。

更准确地说：
    state 是帧级运动状态 embedding。
    h_state 是每个 Gaussian 的隐藏变形特征，
    只不过这个隐藏特征被 state 产生的 FiLM 参数调制过。

所以 h_state 可以理解为：
    “带有当前运动状态影响的 Gaussian 非刚性变形隐藏特征”。
```

生成流程：

```text
1. 每个 Gaussian 先有自己的点特征:
       x_emb

2. 再拼上当前帧和历史运动相关条件:
       pose_feats      = PoseEncoder(pose_conds)
       seq_pose_feats  = SeqPoseEncoder(seq_pose_conds)
       seq_xyz_feats   = SeqXYZEncoder(seq_xyz_conds, x_emb)

3. 拼成每个 Gaussian 的输入特征:
       features = concat(x_emb, pose_feats, seq_pose_feats, seq_xyz_feats)

   features 是点级特征:
       每个 Gaussian 都不一样。

4. baseline 分支:
       h_base = self.mlp(features)

   h_base 是不额外使用 state FiLM 的隐藏特征。

5. state 条件:
       state_conds = cond_dict[pose_id].get("state_conds", seq_pose_conds)

   普通 state:
       state_conds 通常就是 seq_pose_conds

   state_conds 实验:
       state_conds = seq_pose_conds + global surface motion summary

6. reshape 成时间序列:
       state_seq = state_conds.reshape(B, L, -1)

   例如 DNA / SMPL-X:
       普通 state:      [1, 8, 495]
       state_conds:     [1, 8, 513]

7. TemporalStateEncoder 编码帧级运动状态:
       state = self.StateEncoder(state_seq)

   state 是帧级 embedding:
       shape [B, state_dim]

   它表达的是当前帧近期运动模式:
       关节怎么变
       动作快慢
       多时间尺度 motion pattern
       如果是 state_conds，还包含全局 surface motion 强度

8. state 生成 FiLM 参数:
       film = self.state_film_layer(state)
       film -> gamma / beta

   gamma / beta 是帧级参数:
       同一帧内所有 Gaussian 共享同一套 gamma / beta。

9. h_state 从 features 开始走 state_layers:
       h_state = features
       for layer:
           h_state = layer(h_state)
           h_state = gamma * h_state + beta
           h_state = ReLU(h_state)

   注意:
       h_state 的主体仍然来自每个 Gaussian 自己的 features。
       state 只是通过 gamma / beta 去调制这些点级隐藏特征。

10. 最后弱融合:
       h = (1 - alpha) * h_base + alpha * h_state

   当前 state_warm_a04/current state:
       alpha_max = 0.4

11. 输出非刚性变形:
       d_xyz      = gaussian_warp(h)
       d_rotation = gaussian_rotation(h)
       d_scaling  = gaussian_scaling(h)
```

直观理解：

```text
state:
    “这一帧整体处在什么运动状态？”

h_base:
    “不额外看 state 时，每个 Gaussian 应该怎么变形？”

h_state:
    “在当前运动状态影响下，每个 Gaussian 应该怎么变形？”

h:
    “保守融合 baseline 和 state 后的最终隐藏变形特征。”
```

重要限制：

```text
当前 state / state_conds 的 gamma 和 beta 是帧级共享的。
因此它知道“这一帧整体运动状态”，
但还不知道“哪个 Gaussian 是袖口/裤腿/边界/高误差区域”。

这也是为什么后续如果要增强 boundary 收益，
需要 part-level state 或 point-wise gate。
```

本次问答确认：

```text
用户问 h_state 是否就是运动状态特征。
回答口径：
    h_state 不是纯 motion state。
    motion state 先由 state_conds / seq_pose_conds 编码成帧级 state embedding，
    再生成 FiLM gamma / beta，
    最后调制每个 Gaussian 的隐藏变形特征，得到 h_state。
```

## 2026-07-16 state 引入步数消融回顾

用户问：

```text
前面的实验有没有试过不同步数开始引入 state mlp 分支的不同效果？
```

回顾结论：

```text
试过，但不是完整网格搜索。

做过的关键对比是:
    state_warm / state_warm_half / state_warm_a02/a03/a04:
        state_start_iter = 1500
        state_ramp_iter  = 3000

    state_warm_late:
        state_start_iter = 3000
        state_ramp_iter  = 5000
        state_max_alpha  = 1.0

state_warm_late 只在 DNA 的 0007_04 和 0051_09 两个代表序列上跑过。
结果显示“更晚开始 + 更长 ramp”没有改善原 state_warm。
两序列平均:
    state_warm:      PSNR 28.937584, SSIM 0.964072, LPIPSx1000 38.700132
    state_warm_late: PSNR 28.923952, SSIM 0.964076, LPIPSx1000 38.939085

所以已有证据说明:
    单纯把 state 分支更晚接入，不是主要改进方向。
    更关键的是限制 state 分支强度，也就是 state_max_alpha。

后续真正变好的配置是 state_warm_a04/current state:
    state_start_iter = 1500
    state_ramp_iter  = 3000
    state_max_alpha  = 0.4

它不是靠改开始步数提升，而是靠弱融合避免 state 分支破坏 baseline 主路径。
```

## 2026-07-16 state_conds subset 判断依据

用户问：

```text
如何判断 state_conds 在 high-motion / high-error 上更好？
```

判断方式：

```text
不是直接看 state_conds 相对 baseline 是否提升，
而是先分别计算:
    state_warm_a04/current state - baseline
    state_conds - baseline

然后比较这两个 delta。

如果 state_conds 的 delta 更好，才说 state_conds 在该 subset 上优于 current state。
指标方向:
    L1 越低越好，所以 delta 更负更好。
    PSNR 越高越好，所以 delta 更正更好。
    SSIM 越高越好，所以 delta 更正更好。
    LPIPS 越低越好，所以 delta 更负更好。
```

subset 构造方式：

```text
high_motion:
    对每个 frame，计算它和相邻帧前景 mask 的 XOR / Union 变化量；
    按 motion score 排序，取最高的 25% frame。

high_error_crop:
    先用 baseline render 和 GT 计算误差图；
    在每张图里找 baseline 误差最大的 128x128 crop；
    然后在同一个 crop 上比较 baseline / current state / state_conds。
```

直接对比结果，delta = state_conds - state_warm_a04/current state：

| Subset | ΔL1 | ΔPSNR | ΔSSIM | ΔLPIPS x1000 | 判断 |
|---|---:|---:|---:|---:|---|
| high_motion | -0.000026 | +0.027181 | +0.000132 | -0.052473 | 四个指标方向都更好 |
| high_error_crop | -0.000169 | +0.004435 | +0.000486 | -0.889147 | 四个指标方向都更好，LPIPS 更明显 |
| boundary | +0.000154 | -0.030342 | NA | NA | 更差 |

结论：

```text
state_conds 相比 current state，在 high_motion 和 high_error_crop 两个 subset 上更好；
但 boundary subset 更差，全图平均也基本打平，
所以不能说 state_conds 全面优于 current state。
```

## 2026-07-16 part_pamo Step 1：每个 part 加运动编码

用户要求：

```text
从现在开始每次对话都要读写 note/motion.md。
在 part_moe_leg 的基础上新建独立消融实验 part_pamo。
不影响其他实验和代码路径。
总日志保存到:
    /media/image/mxz/human/SeqAvatar/logs/pamo

第一步:
    对每个 SMPL part 预计算 part_pose_t / part_pose_{t-1} / part_pose_{t+1}
    part_velocity = pose_t - pose_{t-1}
    part_acc = pose_{t+1} - 2 pose_t + pose_{t-1}
    z_part = PartMotionEncoder(part_motion_feat)
    在 part_moe_leg 的每个 expert 输入里 concat 对应 part 的 z_part
```

实现口径：

```text
新增独立开关:
    --use_part_pamo
    --part_pamo_dim，默认 32

part_pamo 模式同时启用:
    --use_part_moe
    --part_label_schema part_moe_leg
    --num_parts 7

默认 use_part_pamo=False，所以原 baseline / part_moe / part_moe_leg /
part_moe_foot / part_moe_arm 不走该分支。
```

网络修改：

```text
nets/mlp_delta_non_rigid.py
    新增 PartMotionEncoder。
    part_motion_feat 维度为 15:
        mean(part_pose_t)
        mean(part_pose_{t-1})
        mean(part_pose_{t+1})
        mean(part_velocity)
        mean(part_acc)

    对每个 part 按 SMPL/SMPL-X joint group 做均值聚合，保证 SMPL 和 SMPL-X
    都能用固定维度输入。

    part_pamo 激活时，初始化 Part-MoE experts 会加宽 expert 第一层输入:
        old_input_dim -> old_input_dim + part_pamo_dim

    加宽时复制原权重，新增 z_part 列置零，避免 expert 初始化瞬间破坏
    part_moe_leg 的已有行为。

    forward_part_moe 中:
        global expert 对每个 Gaussian concat 它所属 label 的 z_part。
        routed expert_i 对 label==i 的 Gaussian concat z_part_i。
```

数据与渲染链路：

```text
scene/dataset_readers.py
    use_part_pamo=True 时，为 cond_dict[pose_id] 额外预计算:
        part_motion_conds: [1, num_parts, 15]

    pose_t / pose_{t-1} / pose_{t+1} 来自 get_pose_xyz_func 读出的 SMPL/SMPL-X
    pose matrix，再转 axis-angle。
    首尾帧如果 t-1 或 t+1 不存在，用当前 pose 兜底。

scene/__init__.py
    把 use_part_pamo / part_label_schema / num_parts 传给 dataset reader。

gaussian_renderer/__init__.py
    从 cond_dict 取 part_motion_conds，传入 NonrigidDeformer。

scene/gaussian_model.py
    保存 use_part_pamo / part_pamo_dim，并传给 NonrigidDeformer。
```

脚本修改：

```text
scripts/exps_zjumocap.sh part_pamo
scripts/exps_i3dhuman.sh part_pamo
scripts/exps_dnarendering.sh part_pamo

part_pamo 总日志:
    /media/image/mxz/human/SeqAvatar/logs/pamo

DNA-Rendering 的 part_pamo 沿用 part_moe_leg 的 final_eval_only=1。
```

已验证：

```text
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_dnarendering.sh

/media/image/mxz/.conda/envs/seqavatar/bin/python -m py_compile \
    nets/mlp_delta_non_rigid.py \
    scene/gaussian_model.py \
    gaussian_renderer/__init__.py \
    scene/__init__.py \
    scene/dataset_readers.py \
    arguments/__init__.py \
    train.py \
    render.py

额外用 CPU 小张量测试:
    NonrigidDeformer(use_part_moe=True, use_part_pamo=True, num_parts=7)
    init_part_moe_from_shared()
    forward(part_label, part_motion_conds)

输出维度保持:
    d_xyz: [1, N, 3]
    d_rotation: [1, N, 4]
    d_scaling: [1, N, 3]

直接加载 scene/dataset_readers.py 测试:
    get_part_motion_cond(..., part_label_schema="part_moe_leg", num_parts=7)
    输出 shape: [1, 7, 15]
```

## 2026-07-16 part_pamo Step 1 验证

本次只做验证，不保留验证脚本或生成文件。

验证结果：

```text
结论：part_pamo Step 1 当前实现是正确接通的。

已确认:
    1. part_pamo 是独立模式，默认 use_part_pamo=False。
    2. part_pamo 脚本模式基于 part_moe_leg:
        --use_part_moe
        --use_part_pamo
        --part_label_schema part_moe_leg
        --num_parts 7
    3. 三个脚本的 part_pamo 总日志目录均为:
        /media/image/mxz/human/SeqAvatar/logs/pamo
    4. dataset_readers 只在 use_part_pamo=True 时生成:
        part_motion_conds: [1, num_parts, 15]
    5. part_motion_conds 内容符合 Step 1:
        part_pose_t
        part_pose_{t-1}
        part_pose_{t+1}
        part_velocity = pose_t - pose_{t-1}
        part_acc = pose_{t+1} - 2 pose_t + pose_{t-1}
    6. NonrigidDeformer 在 part_pamo=True 时会:
        z_part = PartMotionEncoder(part_motion_conds)
        按 Gaussian 的 part label gather 对应 z_part
        concat 到 Part-MoE expert 输入
    7. expert 第一层只在 part_pamo 下加宽，普通 part_moe_leg 不加宽。
```

验证命令：

```text
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python 语法 compile 检查 8 个相关文件。

PYTHONDONTWRITEBYTECODE=1 python 最小运行时测试:
    NonrigidDeformer(use_part_moe=True, use_part_pamo=True, num_parts=7)
    forward 输出:
        d_xyz      [1, 5, 3]
        d_rotation [1, 5, 4]
        d_scaling  [1, 5, 3]
    get_part_motion_cond 输出:
        [1, 7, 15]

git diff --check
git status --short --untracked-files=all
```

## 2026-07-16 part_pamo Step 2：part-level rigid residual branch

用户要求：

```text
在 part_pamo 上继续第二步:
    每个 part 不只靠点级 MLP，而是先预测一个 part 级整体运动修正。

公式:
    R_p, t_p = PartRigidHead(z_part)
    d_part_rigid_i = R_p * (x_i - c_p) + c_p + t_p - x_i

约束:
    这个 branch 不重新做完整骨骼运动。
    只学习 SMPL/LBS 之外的 part-level residual motion。
    初始化必须 identity:
        R_p = I
        t_p = 0
    训练一开始等价于原来的 part_moe_leg。
```

实现口径：

```text
仍然只在 --use_part_pamo 时启用。
普通 baseline / part_moe / part_moe_leg / part_moe_foot / part_moe_arm 不走该分支。

c_p 采用当前 canonical Gaussian 坐标 query_xyz 按 part label 求均值：
    centers[p] = mean(query_xyz[label == p])

如果某个 part 当前没有 Gaussian，中心回退为全体 Gaussian 中心。
```

网络修改：

```text
nets/mlp_delta_non_rigid.py
    新增 PartRigidHead:
        input:  z_part
        output: rot_vec_p, t_p

    rot_vec_p 通过 axis-angle Rodrigues 转为 R_p。
    PartRigidHead 最后一层 weight/bias 全零初始化，所以初始:
        rot_vec_p = 0 -> R_p = I
        t_p = 0
        d_part_rigid_i = 0

    forward_part_moe 中流程变为:
        part_motion_conds -> PartMotionEncoder -> z_part
        z_part concat 到每个 Gaussian expert 输入
        PartRigidHead(z_part) 生成每个 part 的 rigid residual
        d_xyz = d_xyz_part_moe + part_weight * d_part_rigid

    使用 part_weight 缩放 rigid residual，使它跟 Part-MoE warmup 同步进入。
```

渲染链路：

```text
gaussian_renderer/__init__.py
    调用 non_rigid_deformer 时传入:
        query_xyz=means3D

means3D 是进入非刚性分支前的 canonical Gaussian 坐标。
rigid residual 仍然在 SeqAvatar 原有 SMPL/LBS coarse_deform_c2source 之前加到 d_xyz，
因此它是 SMPL/LBS 之外的 residual，不替代骨骼运动。
```

已验证：

```text
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python 语法 compile 检查 8 个相关文件。

PYTHONDONTWRITEBYTECODE=1 python 最小运行时测试:
    NonrigidDeformer(use_part_moe=True, use_part_pamo=True, num_parts=7)
    forward 输出:
        d_xyz      [1, 5, 3]
        d_rotation [1, 5, 4]
        d_scaling  [1, 5, 3]

    identity 初始化验证:
        apply_part_rigid_residual(zero_delta, ...)
        max_abs = 0.0

    普通 part_moe_leg 路径验证:
        use_part_pamo=False 时 expert 第一层输入维度不加宽。

git diff --check
git status --short --untracked-files=all
```

## 2026-07-16 part_pamo Step 2 进一步验证

用户要求验证：

```text
1. 单元测试 rigid residual 本身:
    已知 R_p / t_p
    x_new = x + d_part_rigid
    expected = R_p @ (x - c_p) + c_p + t_p
    max_abs < 1e-6

    还要检查:
        R=I, t=0 时 residual=0
        同一 part 内两点距离不变
        part 质心变为 c_p + t_p
        空 part 不产生 NaN
        part_weight=0 时不生效
        part_weight=1 时完整生效

2. 集成测试:
    use_part_pamo=False vs use_part_pamo=True + PartRigidHead zero init
    d_xyz 应接近 0 差异。

    手动给某个 part 非零 t_p:
        只看这个 part 的 Gaussian 是否整体平移
        其他 part 不动
```

本次执行方式：

```text
不保留测试脚本，不生成验证文件。
使用 PYTHONDONTWRITEBYTECODE=1 运行 inline Python。
```

单元测试结果：

```text
synthetic case:
    2 个 active part
    每个 part 5 个点
    另有空 part 用于 NaN 检查

手动设置:
    part 0: 绕 z 轴 +30 度，t=[0.1, 0.0, 0.0]
    part 1: 绕 z 轴 -30 度，t=[0.0, 0.2, 0.0]

结果:
    max_abs(x_new - expected): 0.0
    identity_max: 0.0
    empty_finite: True
    part_weight=0 residual max: 0.0
    part_weight=1 diff from full residual: 0.0

同一 part 内距离保持、质心等于 c_p + t_p 均通过 assert。
```

最小集成测试结果：

```text
构造两套 NonrigidDeformer:
    plain: use_part_pamo=False
    pamo:  use_part_pamo=True, PartRigidHead zero init

复制相同 shared MLP / head 权重后再 init_part_moe_from_shared。

zero init 对比:
    max |d_xyz_plain - d_xyz_pamo|      = 8.940696716308594e-08
    max |d_rotation_plain - d_rotation_pamo| = 0.0
    max |d_scaling_plain - d_scaling_pamo|  = 0.0

part_weight=0 对比:
    d_xyz / d_rotation / d_scaling 差异均为 0.0

手动只给 part 2 设置:
    t_2 = [0.25, -0.1, 0.05]

结果:
    part 2 额外位移误差 max: 7.078051567077637e-08
    其他 part 额外位移 max: 1.1920928955078125e-07

说明 label indexing、batch 维度、part mask 在该最小集成测试中没有错位。
```

SMPL/LBS 未被替代的验证：

```text
gaussian_renderer/__init__.py 顺序确认:
    line 82-93: 调 non_rigid_deformer，并传入 query_xyz=means3D
    line 98:    means3D = means3D + d_xyz
    line 100-102: 继续调用 coarse_deform_c2source(...)

因此 part rigid residual 只是在 coarse_deform_c2source 之前增加 residual d_xyz，
后续 SMPL/LBS coarse deformation 仍照常执行。
```

未执行项：

```text
没有跑完整 rasterizer 渲染图像差异和真实训练 loss 对比。
原因是完整图像验证需要加载真实数据、CUDA rasterizer 和模型流程，通常会产生渲染输出；
本次按“不保留验证过程代码和生成物”的口径，只做了不落盘的 deformer 级集成验证和 renderer 顺序验证。

基于 zero init 下 d_xyz/d_rotation/d_scaling 已接近完全一致，
初始渲染图像和 loss 理论上也应一致；若后续要做端到端图像验证，
建议单独指定一个序列、一个相机、一个 iteration，并把输出写到临时目录后删除。
```

验证后的清理状态：

```text
没有新增未跟踪验证文件。
没有保留临时验证代码。
```

## 2026-08-09 part_budget 有效重跑进度续接

用户要求：

```text
打开 part_budget 重新跑并给最后评价指标，用卡 3。
```

当前有效重跑：

```text
tmux session:
    part_budget_dna_rerun_20260809_010458

主日志:
    /media/image/mxz/human/SeqAvatar/logs/budget/20260809_010458_DNA-Rendering_part_budget.log

确认:
    cfg_args/use_part_budget=True
    命令包含 --use_part_budget
    日志出现 [PART_BUDGET] enabled=True
    PartBudget Stats 已持续打印，说明分支确实接入训练。
```

当前已完成 4/6 个 DNA 序列：

```text
0044_11:
    PSNR 33.005961147944134
    SSIM 0.9782079353928566
    LPIPS 0.02111889289226383

0051_09:
    PSNR 28.7546360651652
    SSIM 0.9720434183875719
    LPIPS 0.03035795715016623

0206_04:
    PSNR 31.473957554499307
    SSIM 0.9701999674240748
    LPIPS 0.033340699629237254

0813_05:
    PSNR 36.233865865071614
    SSIM 0.9873911579449971
    LPIPS 0.017785114779447515
```

当前状态：

```text
正在 GPU 3 上继续跑第 5 个序列 0007_04。
剩余序列:
    0019_10
```

## 2026-08-09 part_budget 有效重跑最终结果

运行信息：

```text
GPU:
    3

tmux:
    part_budget_dna_rerun_20260809_010458

主日志:
    /media/image/mxz/human/SeqAvatar/logs/budget/20260809_010458_DNA-Rendering_part_budget.log

结果:
    6/6 DNA 序列全部跑完。
    tmux 已退出。
    未检索到 Traceback / IndexError / RuntimeError / OOM。
```

part_budget 六序列结果：

```text
0044_11:
    PSNR 33.005961147944134
    SSIM 0.9782079353928566
    LPIPS 0.02111889289226383
    LPIPS*1000 21.11889289226383

0051_09:
    PSNR 28.7546360651652
    SSIM 0.9720434183875719
    LPIPS 0.03035795715016623
    LPIPS*1000 30.357957150166232

0206_04:
    PSNR 31.473957554499307
    SSIM 0.9701999674240748
    LPIPS 0.033340699629237254
    LPIPS*1000 33.34069962923726

0813_05:
    PSNR 36.233865865071614
    SSIM 0.9873911579449971
    LPIPS 0.017785114779447515
    LPIPS*1000 17.785114779447515

0007_04:
    PSNR 29.607305606206257
    SSIM 0.9588569129506747
    LPIPS 0.04381065218088528
    LPIPS*1000 43.81065218088528

0019_10:
    PSNR 35.40017274220784
    SSIM 0.981486864387989
    LPIPS 0.020604321577896673
    LPIPS*1000 20.604321577896673
```

part_budget 平均值：

```text
PSNR:
    32.412649830182396

SSIM:
    0.974697709414694

LPIPS:
    0.0278362730349828

LPIPS*1000:
    27.8362730349828
```

对比 DNA part_moe_leg 基线：

```text
baseline 日志:
    /media/image/mxz/human/SeqAvatar/logs/part/20260623_180431_DNA-Rendering_part_moe_leg.log

part_moe_leg 平均:
    PSNR 32.39075858328077
    SSIM 0.974622116320663
    LPIPS 0.027993953922608245
    LPIPS*1000 27.993953922608245

part_budget - part_moe_leg:
    PSNR +0.021891246901624584
    SSIM +0.00007559309403104564
    LPIPS*1000 -0.1576808876254457
```

逐序列变化：

```text
0044_11:
    PSNR +0.005390246709190194
    SSIM -0.000003036359945918221
    LPIPS*1000 +0.04112230769048181

0051_09:
    PSNR +0.055731598536173266
    SSIM +0.0003578260540961775
    LPIPS*1000 -0.4692763633405185

0206_04:
    PSNR -0.04500829378763882
    SSIM -0.00012703736623131956
    LPIPS*1000 -0.22252170989910808

0813_05:
    PSNR +0.031395753224693124
    SSIM +0.000051344434420230733
    LPIPS*1000 -0.12680551347633157

0007_04:
    PSNR +0.04018796284993442
    SSIM -0.000034307440122005595
    LPIPS*1000 -0.09175512629250315

0019_10:
    PSNR +0.043650213877356236
    SSIM +0.0002087692419687759
    LPIPS*1000 -0.07684892043471545
```

结论：

```text
part_budget 这次有效重跑相对 part_moe_leg 有正向但很小的平均收益。

优点:
    6 个序列里 5 个 PSNR 提升。
    5 个序列 LPIPS*1000 下降。
    平均 PSNR / SSIM / LPIPS 都优于 part_moe_leg。

不足:
    平均 PSNR 只提升约 +0.0219 dB。
    0044_11 的 LPIPS 略差。
    0206_04 的 PSNR 和 SSIM 下降。

判断:
    当前 part_budget 已经真正接入并能学习到非均匀 budget 分布，
    但收益仍属于弱提升，不是强创新结果。
```

## 2026-07-16 part_pamo Step 3：internal learnable rigidity

用户要求：

```text
PaMoSplat 还有 internal learnable rigidity。
迁移到 SeqAvatar 的 part_pamo:
    r_i = sigmoid(MLP_rigidity(x_emb_i, z_part_i))

含义:
    r_i 高:
        Gaussian 更服从 part-level rigid motion
    r_i 低:
        Gaussian 更多依赖 part-specific nonrigid MLP

最终输出:
    d_i = d_part_mlp_i + alpha * r_i * d_part_rigid_i
```

实现口径：

```text
仍然只在 --use_part_pamo 时启用。
脚本和数据读取不需要新增参数。

alpha 继续沿用已有 Part-MoE warmup 后得到的 part_weight。
因此 rigid residual 和 internal rigidity 一起随 Part-MoE 激活，不会在
part_moe_start_iter 之前介入。
```

网络修改：

```text
nets/mlp_delta_non_rigid.py
    新增 PartRigidityMLP:
        input:  concat(x_emb_i, z_part_i)
        output: sigmoid scalar r_i, shape [B, N, 1]

    z_part_i 来自 Step 1 的 PartMotionEncoder，并按 Gaussian 的 part label gather。

    Step 2 原公式:
        d_xyz = d_part_mlp + part_weight * d_part_rigid

    Step 3 后改为:
        d_xyz = d_part_mlp + part_weight * r_i * d_part_rigid

    r_i 是点级，因此同一 part 内不同 Gaussian 可以有不同刚性强度。
```

训练参数归属：

```text
PartRigidityMLP 是 NonrigidDeformer 的子模块。
training_setup 里 non_rigid_deformer.parameters() 已覆盖它。

Part-MoE 激活后 freeze_shared_after_part_moe 只冻结:
    self.mlp
    gaussian_warp
    gaussian_rotation
    gaussian_scaling

不会冻结 PartMotionEncoder / PartRigidHead / PartRigidityMLP。
```

已验证：

```text
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python 语法 compile 检查 8 个相关文件。

rigidity 数学测试:
    r_i = 0:
        residual max = 0.0
    r_i = 1:
        与完整 rigid residual diff = 0.0
    r_i = [0, 0.25, 0.5, 0.75, 1.0, 0.2], part_weight=0.4:
        与 0.4 * r_i * full_residual diff = 0.0
    part_weight=0:
        residual max = 0.0

最小集成测试:
    plain: use_part_pamo=False
    pamo:  use_part_pamo=True, PartRigidHead zero init

    zero init 对比:
        max |d_xyz_plain - d_xyz_pamo| = 2.9802322387695312e-08
        max |d_rotation_plain - d_rotation_pamo| = 0.0
        max |d_scaling_plain - d_scaling_pamo| = 0.0

    手动只给 part 2 设置平移，并设置该 part 三个点的 r_i:
        [0.0, 0.5, 1.0]

    结果:
        part 2 scaled shift diff = 5.21540641784668e-08
        other parts extra max = 2.9802322387695312e-08

说明 internal rigidity 确实按点缩放 part-level rigid residual，
且 label / part mask 没有明显错位。

git diff --check
git status --short --untracked-files=all
```

## 2026-07-16 part_pamo Step 3 真实训练验证

用户要求按 5 点方案验证：

```text
1. 验证 part_moe_leg / part_pamo 开关隔离。
2. 在真实 part_pamo 训练中打印 rigidity 统计。
3. 检查 PartRigidityMLP / PartMotionEncoder / PartRigidHead 梯度。
4. 检查 rigid residual 实际贡献大小。
5. 固定真实 batch 做 part_moe_leg vs part_pamo forward 对照。
```

本次执行方式：

```text
不保留验证脚本。
所有真实训练 / one-batch 验证输出都写到临时目录，命令结束后删除。

真实短训为节省时间使用:
    CoreView_377
    iterations=4
    part_moe_start_iter=1
    part_moe_warmup=1
    part_moe_global_keep=0
    part_pamo_log_interval=1
    part_pamo_dim=16
    non_rigid_mlp_width=256

这只改变验证规模，不改变 part_pamo 的代码路径。
正式脚本默认仍是 part_pamo_dim=32。
```

开关隔离验证：

```text
真实 train.py 启动 part_moe_leg:
    [PartPAMO] enabled=False; PartPAMO modules are not constructed.

真实 train.py 启动 part_pamo:
    [PartPAMO] enabled=True modules=PartMotionEncoder,PartRigidHead,PartRigidityMLP part_pamo_dim=16

part_pamo 激活 Part-MoE 后:
    [PartPAMO] Appending per-part motion code dim=16 to each expert input.

说明:
    part_moe_leg 不创建 PartMotionEncoder / PartRigidHead / PartRigidityMLP。
    part_pamo 才创建并调用这些模块。
```

真实训练 rigidity 统计：

```text
iter 2:
    r_mean=0.530715
    r_std =0.022822
    r_min =0.454957
    r_max =0.603784

iter 3:
    r_mean=0.531873
    r_std =0.022679
    r_min =0.457118
    r_max =0.603198

iter 4:
    r_mean=0.533615
    r_std =0.022814
    r_min =0.459061
    r_max =0.604675

结论:
    r_i 不是全 0、全 1，也不是固定 0.5。
    同一 part 内有非零 std。
    不同 part 的 r_mean 有差异。
```

per-part 统计示例：

```text
iter 4:
    part=1 count=2852 r_mean=0.531427 r_std=0.020166
    part=2 count=778  r_mean=0.544152 r_std=0.020081
    part=3 count=762  r_mean=0.526152 r_std=0.018682
    part=4 count=1168 r_mean=0.554316 r_std=0.014909
    part=5 count=665  r_mean=0.525984 r_std=0.020720
    part=6 count=665  r_mean=0.510491 r_std=0.019935
```

梯度验证：

```text
iter 2:
    rigid_norm=1.370981e-09
    PartMotionEncoder grad norm=4.092928e-12
    PartRigidHead    grad norm=3.011278e-01
    PartRigidityMLP  grad norm=1.214389e-10

iter 3:
    rigid_norm=6.791169e-03
    PartMotionEncoder grad norm=1.829209e-03
    PartRigidHead    grad norm=5.405230e-01
    PartRigidityMLP  grad norm=4.643150e-04

iter 4:
    rigid_norm=8.129032e-03
    PartMotionEncoder grad norm=1.743164e-03
    PartRigidHead    grad norm=3.482730e-01
    PartRigidityMLP  grad norm=6.127174e-04

解释:
    iter 2 时 PartRigidHead 是 zero init，d_part_rigid 仍接近 0，
    所以 PartRigidityMLP 梯度也接近 0，这符合公式
        d_i = d_part_mlp_i + alpha * r_i * d_part_rigid_i
    中 d_loss / d_r_i 依赖 d_part_rigid_i 的预期。

    iter 3 / iter 4 时 d_part_rigid 变非零后，
    PartMotionEncoder / PartRigidHead / PartRigidityMLP 都有非零梯度。
```

rigid residual 贡献验证：

```text
iter 2:
    mlp_norm=9.484535e-02
    rigid_norm=1.370981e-09
    rigid_contrib=7.299856e-10
    ratio=0.000000

iter 3:
    mlp_norm=8.582780e-02
    rigid_norm=6.791169e-03
    rigid_contrib=3.611372e-03
    ratio=0.042077

iter 4:
    mlp_norm=8.060258e-02
    rigid_norm=8.129032e-03
    rigid_contrib=4.337328e-03
    ratio=0.053811

结论:
    rigid contribution 没有长期为 0。
    短训后 ratio 在 0.04 到 0.054，处在较稳的 0.02 到 0.2 观察区间内。
    没有出现 rigid branch 一开始压过 part_moe_leg 的现象。
```

真实 one-batch forward 对照：

```text
使用真实 ZJU CoreView_377:
    初始 Gaussian 数: 6890
    part_motion_conds shape: (1, 7, 15)
    active_parts: [1, 2, 3, 4, 5, 6]
    target_part: 5

part_moe_leg vs part_pamo, part_weight=0:
    d_xyz      max diff = 1.862645149230957e-08
    d_rotation max diff = 1.862645149230957e-08
    d_scaling  max diff = 1.4901161193847656e-08

part_moe_leg vs part_pamo, part_weight=1, PartRigidHead zero init:
    d_xyz      max diff = 7.450580596923828e-09
    d_rotation max diff = 0.0
    d_scaling  max diff = 0.0

手动给 part 5 设置平移:
    t_5 = [0.12, -0.04, 0.02]
    r_i = 0.5
    part_weight = 1

结果:
    target part half-shift diff = 1.4901161193847656e-08
    other parts extra max       = 1.4901161193847656e-08

结论:
    真实数据链路中的 part_motion_conds / part_label / batch 维度没有错位。
    part_weight=0 时等价 part_moe_leg。
    手动非零 rigid residual 只作用到目标 part，其他 part 基本不动。
```

补充确认：

```text
gaussian_renderer/__init__.py 的执行顺序仍然是:
    1. non_rigid_deformer 输出 d_xyz
    2. means3D = means3D + d_xyz
    3. coarse_deform_c2source 继续执行 SMPL/LBS

因此 part_pamo 的 rigid residual 没有替代 SMPL/LBS，只是在 LBS 之前增加 residual。
```

验证命令：

```text
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile \
    nets/mlp_delta_non_rigid.py \
    scene/gaussian_model.py \
    gaussian_renderer/__init__.py \
    scene/__init__.py \
    scene/dataset_readers.py \
    arguments/__init__.py \
    train.py \
    render.py

git diff --check
git status --short --untracked-files=all
```

清理状态：

```text
没有保留验证脚本。
没有保留 output/_tmp_verify* 临时输出目录。
没有保留 logs/**/_tmp_verify* 临时日志目录。
```

## 2026-07-17 DNA part_pamo 运行失败定位

用户运行：

```text
bash /media/image/mxz/human/SeqAvatar/scripts/exps_dnarendering.sh part_pamo
```

失败位置：

```text
序列:
    DNA-Rendering/0206_04

训练进度:
    2570 / 25000

当时高斯点数:
    #pts=40145

报错:
    RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED
    when calling cublasSgemm(...)

栈位置:
    train.py line 233
    loss.backward()
```

判断：

```text
这次失败不是 part_pamo rigid / rigidity 分支直接触发。

原因:
    part_pamo 的 Part-MoE 激活点是:
        part_moe_start_iter=10000

    失败发生在 iter 2570，远早于 part labels 构建和 part experts 初始化。
    当时还没有进入 Part-MoE / part_pamo 的专家路由、PartRigidHead、
    PartRigidityMLP 路径。

更可能原因:
    CUDA/cuBLAS 反传阶段资源失败，通常和显存峰值、显存碎片、
    cuBLAS workspace 或 CUDA 状态有关。

DNA 脚本当前默认:
    SKIP_LOAD_TEST_CAMERAS=0
    IMAGE_DATA_DEVICE=cuda

因此训练阶段会把训练相机和 novelview 测试相机图像/ mask 放在 GPU，
再叠加 40145 个 Gaussian 的 rasterizer / LPIPS / non-rigid 反传，
容易在某个 batch 触发 cuBLAS 执行失败。
```

建议复跑方式：

```text
优先降低训练阶段显存占用:

SKIP_LOAD_TEST_CAMERAS=1 \
IMAGE_DATA_DEVICE=cpu \
SKIP_COMPLETED=1 \
GPU_id=3 \
bash scripts/exps_dnarendering.sh part_pamo

如果只想从失败后的剩余序列跑:

SEQUENCES_OVERRIDE="0206_04 0813_05 0007_04 0019_10" \
SKIP_LOAD_TEST_CAMERAS=1 \
IMAGE_DATA_DEVICE=cpu \
GPU_id=3 \
bash scripts/exps_dnarendering.sh part_pamo
```

补充：

```text
SKIP_LOAD_TEST_CAMERAS=1 只跳过 train.py 内部加载测试相机。
脚本训练结束后仍会单独调用 render.py 做最终 novelview 评估。

IMAGE_DATA_DEVICE=cpu 会让相机图像和 mask 常驻 CPU，
train.py 每次取 batch 时再 .cuda()，速度可能略慢，但显存更稳。
```

## 2026-07-17 DNA 0044 part_pamo 比 part_moe_leg 差的原因分析

用户观察：

```text
DNA-Rendering/0044_11 的 part_pamo 结果比 part_moe_leg 差。
```

指标对比：

```text
part_moe_leg:
    output/DNA-Rendering/0044_11/part_moe_leg/20260623_180431
    PSNR  = 33.000570344924924
    SSIM  = 0.9782106434305509
    LPIPS = 0.02107852928650876

part_pamo:
    output/DNA-Rendering/0044_11/part_pamo/20260716_214518
    PSNR  = 32.96493280728658
    SSIM  = 0.978027040263017
    LPIPS = 0.02122311471030116

差值 part_pamo - part_moe_leg:
    PSNR  = -0.035637537638343986
    SSIM  = -0.0001836031675338523
    LPIPS = +0.00014458542379239964
```

第一判断：

```text
差距存在，但幅度很小。
0.035 dB PSNR / 0.00018 SSIM / 0.000145 LPIPS 接近单次训练随机波动量级。
不能只凭这一组结果断言 part_pamo 结构一定显著劣化。
```

公平性问题：

```text
这两个 run 不是严格同轨迹对照。

part_moe_leg 在 iter 10000 构建 part label 时:
    Total gaussians: 61950

part_pamo 在 iter 10000 构建 part label 时:
    Total gaussians: 59119

差了 2831 个 Gaussian，约 4.6%。

原因可能包括:
    1. part_pamo 构建了额外模块，虽然 Part-MoE 激活前不参与 forward，
       但模块初始化会消耗 torch RNG。
    2. densify_and_split 中使用 torch.normal 采样新 Gaussian，
       RNG 状态不同会让增密轨迹不同。
    3. 因此 part_pamo 和旧 part_moe_leg 在 10000 步前已经不是完全相同模型状态。
```

part_pamo 自身诊断：

```text
DNA 0044 上 internal rigidity 几乎塌到 0。

iter 11000:
    r_mean=0.002066
    r_std =0.001410
    rigid_contrib=1.058419e-06
    ratio=0.000026

iter 15000:
    r_mean=0.000106
    r_std =0.000098
    rigid_contrib=4.494924e-08
    ratio=0.000001

iter 20000:
    r_mean=0.000019
    r_std =0.000026
    rigid_contrib=1.150937e-08
    ratio=0.000000

iter 25000:
    r_mean=0.000018
    r_std =0.000036
    rigid_contrib=2.293534e-08
    ratio=0.000001
```

结论：

```text
Step 2/3 的 part-level rigid residual 在 DNA 0044 上基本没有发挥作用。

模型学到的是:
    r_i -> 0

结果是:
    d_i = d_part_mlp_i + alpha * r_i * d_part_rigid_i
基本退化成:
    d_i ~= d_part_mlp_i

也就是说，rigid residual branch 被 internal rigidity gate 主动关掉了。
```

为什么会这样：

```text
1. DNA 当前评估是 novelview，不是强 novel-pose。
   SMPL/LBS + part_moe_leg 的 point-level nonrigid MLP 已经能解释大部分训练/测试需求。
   part-level rigid residual 对 novelview 帮助不明显。

2. PartRigidHead 是 part 级整体刚性修正，粒度比 point-level MLP 粗。
   对衣服边界、手部、脸部、腿脚等细节，整体刚性 residual 容易引入错误方向。
   优化器最简单的策略就是让 r_i 接近 0，把 rigid 分支关掉。

3. rigid 分支被关掉后，part_pamo 剩下的有效差异主要是 Step 1:
       expert 输入多了 z_part
   这个 z_part 是由 part-level SMPL pose/velocity/acc 编码来的。
   在 DNA novelview 上，这个运动编码可能提供的信息有限，反而增加了 expert 输入自由度，
   容易带来轻微过拟合或扰动。

4. 由于没有对 r_i 的分布做约束，sigmoid gate 可以无限接近 0。
   一旦 r_i 很小，rigid branch 对 loss 的贡献和梯度都很弱，
   后续基本很难重新变成有效分支。
```

后续建议：

```text
先不要直接用当前 part_pamo 作为最终版本。

建议做三个小消融定位:

1. part_pamo_step1_only:
    只保留 z_part concat expert，关闭 PartRigidHead / PartRigidityMLP。
    判断性能下降是不是来自运动编码本身。

2. part_pamo_rigid_no_gate 或 r_min 版本:
    例如 r_i = r_min + (1 - r_min) * sigmoid(...)
    r_min 可先试 0.05 或 0.1。
    防止 gate 直接塌到 0。

3. part_pamo_no_z_expert:
    不把 z_part concat 到 expert MLP，只用 z_part 预测 part rigid residual。
    判断是不是 z_part 注入 expert 后过拟合。

如果要做公平对比:
    需要在当前同一份代码、同一 seed 下重跑 part_moe_leg 和 part_pamo。
    更严格的做法是让 PartPAMO 模块延迟到 Part-MoE 激活时再初始化，
    或保存/恢复 RNG，避免 Part-MoE 激活前增密轨迹因为额外模块初始化而分叉。
```

## 2026-07-17 如何让 Step 2/3 的 part-level rigid residual 真正发挥作用

当前问题：

```text
DNA 0044 的日志说明:
    r_i 很快塌到接近 0
    rigid_contrib / mlp_contrib 接近 0

因此当前 part_pamo 实际上没有在使用 part-level rigid residual。

根因不是 alpha 不够大。
根因是:
    1. point-level part expert MLP 有足够能力解释所有残差；
    2. rigid branch 是较粗的 part-level 运动，一旦方向不够准，会被 loss 惩罚；
    3. r_i 没有下限或先验，优化器最容易把 sigmoid gate 推到 0；
    4. r_i 接近 0 后，rigid branch 的贡献和 rigidity MLP 梯度都变弱，很难恢复。
```

优先改法：

```text
第一优先级:
    给 internal rigidity 加下限和 warmup。

建议:
    r_raw = sigmoid(MLP_rigidity(x_emb_i, z_part_i))
    r_i = r_min + (1 - r_min) * r_raw

先试:
    r_min = 0.05 或 0.1

或者更稳:
    10000-12000 iter:
        r_min = 0.2
    12000 之后:
        r_min 线性降到 0.05

目的:
    防止 r_i 直接塌成 0；
    让 PartRigidHead 在训练早期至少有可见贡献和梯度。
```

第二优先级：

```text
做分阶段训练，避免 expert MLP 抢走所有残差。

建议:
    Part-MoE 激活后的前 500-1000 iter:
        1. 固定或下限约束 r_i，例如 r_i >= 0.2；
        2. PartRigidHead / PartMotionEncoder 用正常 lr；
        3. part experts 用较低 lr，或者短暂 freeze routed experts。

之后:
    放开 experts；
    允许 r_i 学习，但保留 r_min=0.05。

目的:
    先让 rigid branch 学到 part-level 低频趋势，
    再让 point-level MLP 负责细节。
```

第三优先级：

```text
显式把 d_part_mlp 约束成 detail residual，别让它同时学整体平移。

可做软约束:
    对每个 part 计算:
        mean_d_mlp_p = mean(d_part_mlp_i | label_i == p)

    加 loss:
        L_detail_zero_mean = sum_p ||mean_d_mlp_p||^2

目的:
    让 part-level 平移趋势更自然地交给 PartRigidHead 的 t_p；
    让 d_part_mlp_i 更多学习衣服边界、膝盖、脚踝等局部非刚性细节。

更强版本:
    直接从 d_part_mlp_i 中减掉 per-part mean，
    但这个可能更激进，建议先用 soft loss。
```

第四优先级：

```text
减少 z_part 对 expert MLP 的直接干扰。

当前 Step 1:
    expert input = concat(x_emb_i, pose_feat, seq_xyz_feat, z_part_i)

问题:
    z_part 可能被 expert MLP 用来拟合点级残差，
    rigid branch 仍然没有必要存在。

建议做 ablation:
    part_pamo_no_z_expert:
        不把 z_part concat 到 expert；
        z_part 只用于 PartRigidHead 和 PartRigidityMLP。

如果 no_z_expert 更好，说明 z_part 注入 expert 是主要扰动源。
```

建议的 pamo_v2 组合：

```text
part_pamo_v2:
    1. r_i 加 r_min 下限:
        r_min warmup 0.2 -> 0.05

    2. Part-MoE 激活后前 500-1000 iter:
        降低 routed expert lr 或 freeze routed experts，
        先训练 PartRigidHead。

    3. 加 detail zero-mean loss:
        L_detail_zero_mean = sum_p ||mean(d_part_mlp_i)||^2

    4. 做 no_z_expert 对照:
        z_part 只驱动 rigid/gate，不直接 concat 到 expert。
```

判断是否真正生效：

```text
训练日志里应该看到:
    r_mean 不再长期 < 0.001
    r_std 同一 part 内仍非零
    rigid_contrib / mlp_contrib 不再接近 0

较合理观察范围:
    r_mean:
        0.05 - 0.3

    ratio = mean_norm(alpha * r_i * d_part_rigid_i) / mean_norm(d_part_mlp_i):
        0.02 - 0.2

如果 ratio 仍然 < 0.001:
    rigid branch 还是没用上。

如果 ratio > 0.5:
    rigid branch 可能压过 part_moe_leg，需要减小 r_min / alpha / rigid lr。
```

公平对比补充：

```text
为了判断 pamo_v2 是否真的优于 part_moe_leg，
需要先修正公平性:
    让 PartPAMO 模块延迟到 Part-MoE 激活时初始化，
    或在初始化 PartPAMO 模块前后保存/恢复 RNG。

否则 part_pamo 和 part_moe_leg 在 10000 步前的 densification 轨迹会不同，
指标差异会混入随机增密因素。
```

## 2026-07-17 gate 的作用以及如何保留 rigid residual

用户问题：

```text
为什么要有 internal rigidity gate？
能不能改这个 gate，让 rigid residual 保留？
```

为什么需要 gate：

```text
PartRigidHead 预测的是 part 级整体刚性 residual:
    d_part_rigid_i

这个 residual 对同一 part 内所有 Gaussian 共享同一个 R_p / t_p。
它适合表达:
    大腿、小腿、躯干主体区域的低频整体趋势

但不一定适合:
    衣服边界
    关节附近
    脚踝/膝盖
    手部/脸部细节
    segmentation 边界附近的点

所以需要 point-wise gate:
    r_i 高: 更服从 part-level rigid
    r_i 低: 更多依赖 point-level nonrigid expert

没有 gate 的风险:
    rigid residual 会强行作用到整个 part，
    容易把软组织/边界/关节细节一起拖动，导致渲染变差。
```

当前 gate 的问题：

```text
当前公式:
    r_i = sigmoid(MLP_rigidity(...))
    d_i = d_part_mlp_i + alpha * r_i * d_part_rigid_i

问题:
    sigmoid 输出可以无限接近 0。
    DNA 0044 日志里 r_mean 最后约 1e-5 到 1e-4，
    等价于把 rigid residual branch 关掉。

因此当前 gate 太自由，缺少保留 rigid residual 的约束。
```

能不能改：

```text
可以，而且建议改。

不建议完全删除 gate。
更建议保留 gate 的 point-wise 选择能力，
但给 rigid residual 一个不可被关闭的基础通道。
```

推荐改法 1：floor gate

```text
r_raw = sigmoid(MLP_rigidity(x_emb_i, z_part_i))
r_i = r_min + (1 - r_min) * r_raw

d_i = d_part_mlp_i + alpha * r_i * d_part_rigid_i

推荐先试:
    r_min = 0.05
    r_min = 0.10

含义:
    每个 Gaussian 至少保留 5%-10% 的 part-level rigid residual。
    MLP 仍然能学习哪些点更 rigid，但不能把 rigid residual 完全关掉。
```

推荐改法 2：warmup fixed gate，再放开学习

```text
Part-MoE 激活后前 500-1000 iter:
    r_i = 0.2 或 0.3
    不使用 PartRigidityMLP 输出

之后:
    r_i = r_min + (1 - r_min) * sigmoid(MLP_rigidity(...))
    r_min = 0.05

含义:
    先让 PartRigidHead 学到有意义的 part-level residual，
    再让 gate 学哪些点应该减少刚性。
```

推荐改法 3：base rigid + learned residual gate

```text
r_raw = sigmoid(MLP_rigidity(...))
r_i = r_base + r_scale * r_raw

例如:
    r_base = 0.05
    r_scale = 0.45

则:
    r_i 范围是 [0.05, 0.50]

含义:
    rigid residual 永远保留一部分，
    但不会大到压过 point-level expert。
```

推荐改法 4：给 r_i 加均值约束

```text
L_r_mean = (mean(r_i) - r_target)^2

先试:
    r_target = 0.10 或 0.15
    loss weight = 1e-4 到 1e-3

含义:
    防止 r_i 全部塌到 0。
    但仍允许不同 Gaussian / 不同 part 有差异。
```

更符合部件先验的版本：

```text
不同 part 用不同 r_min:
    body:           0.05
    left_leg_foot:  0.10 或 0.15
    right_leg_foot: 0.10 或 0.15
    hands/face:     0.02 或 0.05
    unknown:        0.00 或 0.02

原因:
    腿部主体更应该受 part-level rigid motion 影响；
    脸、手、unknown 和边界点更需要 point-level nonrigid。
```

当前最推荐的实现路线：

```text
做一个独立消融:
    part_pamo_gate_floor

最小改动:
    新增参数:
        --part_pamo_rigidity_min
        默认 0.0，保证旧 part_pamo 行为不变。

    在 part_pamo_gate_floor 脚本模式中设:
        --part_pamo_rigidity_min 0.05

公式:
    r_raw = PartRigidityMLP(...)
    r_i = r_min + (1 - r_min) * r_raw

验证目标:
    r_mean 不再 < 0.001
    ratio 不再长期为 0
    目标范围:
        r_mean 0.05 - 0.3
        ratio  0.02 - 0.2
```

结论：

```text
gate 要保留，因为它负责区分刚性主体点和非刚性边界/关节点。
但当前 gate 不能无限自由地关掉 rigid branch。

最稳方案不是去掉 gate，
而是改成:
    有下限的 gate
    或先固定再学习的 gate
    或 part-dependent floor gate

这样可以保留 PaMoSplat 的 internal rigidity 思想，
同时确保 Step 2 的 part-level rigid residual 真正参与训练。
```

## 2026-07-20 part_pamo_gate_floor：降低 gate 关闭能力并用 GPU1 验证

用户要求：

```text
用卡1跑降低 gate 之后的 part_pamo 实验，并判断有没有改善。
```

实现口径：

```text
新增独立消融:
    part_pamo_gate_floor

默认旧 part_pamo 不变:
    --part_pamo_rigidity_min 0.0

新实验默认:
    --part_pamo_rigidity_min 0.05

公式:
    r_raw = sigmoid(MLP_rigidity(x_emb_i, z_part_i))
    r_i = r_min + (1 - r_min) * r_raw
    d_i = d_part_mlp_i + alpha * r_i * d_part_rigid_i
```

代码修改：

```text
arguments/__init__.py
    新增 part_pamo_rigidity_min，默认 0.0。

scene/gaussian_model.py
    读取并传给 NonrigidDeformer。

nets/mlp_delta_non_rigid.py
    NonrigidDeformer 保存 part_pamo_rigidity_min。
    compute_point_rigidity 中加 floor gate。
    启动日志打印 rigidity_min。

scripts/exps_dnarendering.sh
    新增模式:
        part_pamo_gate_floor

    基于 part_pamo / part_moe_leg:
        --use_part_moe
        --use_part_pamo
        --part_label_schema part_moe_leg
        --num_parts 7
        --part_pamo_rigidity_min 0.05

    日志仍保存到:
        /media/image/mxz/human/SeqAvatar/logs/pamo
```

启动前验证：

```text
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    arguments/__init__.py
    train.py
    render.py
    gaussian_renderer/__init__.py
    scene/__init__.py
    scene/dataset_readers.py

git diff --check

inline rigidity floor check:
    part_pamo_rigidity_min=0.0 / 0.05 均通过；
    r_min=0.05 时输出 rigidity 不低于 0.05。
```

运行计划：

```text
先跑 DNA-Rendering 0044_11，因为上一轮 part_pamo 和 part_moe_leg 对比、
gate collapse 诊断都来自该序列。

命令:
    SEQUENCES_OVERRIDE=0044_11
    GPU_id=1
    SKIP_LOAD_TEST_CAMERAS=1
    IMAGE_DATA_DEVICE=cpu
    bash scripts/exps_dnarendering.sh part_pamo_gate_floor
```

运行结果：

```text
已在 GPU1 跑完 DNA-Rendering 0044_11。

RUN_TIME:
    20260720_160355

输出目录:
    output/DNA-Rendering/0044_11/part_pamo_gate_floor/20260720_160355/

总日志:
    logs/pamo/20260720_160355_DNA-Rendering_part_pamo_gate_floor.log

最终 render.py novelview 指标:
    PSNR  32.962922636667884
    SSIM  0.9781010041634242
    LPIPS 0.021254854928702115
```

与已有结果对比：

```text
part_moe_leg:
    output/DNA-Rendering/0044_11/part_moe_leg/20260623_180431/
    PSNR  33.000570344924924
    SSIM  0.9782106434305509
    LPIPS 0.02107852928650876

原始 part_pamo:
    output/DNA-Rendering/0044_11/part_pamo/20260716_214518/
    PSNR  32.964932950337726
    SSIM  0.97802731047074
    LPIPS 0.021222742390818894

part_pamo_gate_floor - 原始 part_pamo:
    ΔPSNR  -0.002010313670
    ΔSSIM  +0.000073693693
    ΔLPIPS +0.000032112538

part_pamo_gate_floor - part_moe_leg:
    ΔPSNR  -0.037647708257
    ΔSSIM  -0.000109639267
    ΔLPIPS +0.000176325642
```

训练诊断：

```text
启动日志确认:
    [PartPAMO] enabled=True
    modules=PartMotionEncoder,PartRigidHead,PartRigidityMLP
    rigidity_min=0.05

iter 11000:
    r_mean=0.050730
    r_std=0.000685
    ratio=0.000810
    PartMotionEncoder / PartRigidHead / PartRigidityMLP 都有 grad

iter 23000:
    r_mean=0.050002
    r_std=0.000004
    ratio=0.005442

iter 25000:
    r_mean=0.050001
    r_std=0.000002
    r_min=0.050000
    r_max=0.050137
    mlp_norm=4.218914e-02
    rigid_norm=1.755154e-03
    rigid_contrib=7.898340e-05
    ratio=0.001872
```

Gaussian 点数：

```text
part_moe_leg:
    61950

原始 part_pamo:
    59119

part_pamo_gate_floor:
    58932
```

结论：

```text
降低 gate 关闭能力后，没有得到明确改善。

相对原始 part_pamo:
    SSIM 有极小提升，但 PSNR 和 LPIPS 都略差，整体不能算改善。

相对 part_moe_leg:
    PSNR / SSIM / LPIPS 全部更差。

原因:
    gate floor 技术上生效了，r_i 没有再掉到 0；
    但 r_i 几乎贴着 0.05 下限，std 接近 0，说明 internal rigidity 仍没有形成
    有意义的点级差异。

    rigid residual 的实际贡献仍太小:
        最终 ratio=0.001872
    远低于之前期望的 0.02 - 0.2。

判断:
    仅把 gate 下限设为 0.05 不足以让 Step 2/3 产生有效收益。

下一步更值得尝试:
    1. 提高 floor 到 0.10 或 0.20。
    2. 前若干千 iter 固定 r_i，再放开学习。
    3. 给 PartRigidHead 更强学习机会，例如降低 part MLP 对 rigid 趋势的抢占。
    4. 对 rigid contribution 加温和约束或目标范围监控，避免 ratio 长期接近 0。
```

## 2026-07-20 下一步如何检验 part_pamo 是否有用

用户问题：

```text
下一步建议怎么检验 pamo 有没有用。
```

建议不要直接继续盲跑全量，而是拆成三层验证：

```text
第一层：机制是否有用
    目标不是看最终 PSNR，而是确认 rigid residual 在真实训练中能产生有效贡献。

    建议做固定 gate 对照:
        part_pamo_r_fixed_0.2
        part_pamo_r_fixed_0.5
        part_pamo_r_fixed_1.0

    公式:
        d_i = d_part_mlp_i + alpha * r_fixed * d_part_rigid_i

    判断:
        如果 fixed r 明显优于当前 learnable gate，
        说明 PAMO rigid branch 有潜力，问题主要是 gate collapse。

        如果 r_fixed=0.5 / 1.0 仍然没提升甚至变差，
        说明 part-level rigid residual 本身对当前 SeqAvatar/DNA 设置帮助有限，
        或者中心 c_p / part label / motion feature 的定义还不够合适。
```

```text
第二层：分支贡献是否足够
    继续打印:
        r_mean / r_std / per_part r_mean / per_part r_std
        mean_norm(d_part_mlp)
        mean_norm(d_part_rigid)
        mean_norm(alpha * r_i * d_part_rigid)
        ratio = rigid_contrib / mlp_contrib

    当前 part_pamo_gate_floor 结果:
        final ratio = 0.001872

    这个量级太小，不足以影响最终渲染。

    更合理的检验目标:
        ratio 进入 0.02 - 0.2
        r_std 不长期接近 0
        per-part r_mean 有差异
```

```text
第三层：看 PAMO 应该发挥作用的子集
    不只看全帧平均 novel-view 指标。

    PaMo/PAMO 的预期收益主要在:
        快速摆腿
        膝盖/脚踝附近
        衣服边界随 part 整体运动的区域

    建议额外做:
        high-motion frames top 20% 评测
        low-motion frames bottom 20% 评测
        lower-body / leg part 区域局部指标

    判断:
        如果全局平均不提升，但 high-motion leg 子集提升，
        说明该点仍有论文价值，可以作为“运动剧烈区域更稳”的创新点。

        如果 high-motion 子集也不提升，
        基本说明当前 part_pamo 对该任务没有实质收益。
```

推荐最小实验矩阵：

```text
固定 0044_11，先不要跑六序列。

A. part_moe_leg
    已有:
        20260623_180431

B. part_pamo_step1_only
    只保留 z_part concat，不加 rigid residual / rigidity gate。
    用来判断 Step 1 的 part motion code 是否本身有收益。

C. part_pamo_r_fixed_0.2
    关闭 learnable gate，固定 r=0.2。

D. part_pamo_r_fixed_0.5
    关闭 learnable gate，固定 r=0.5。

E. part_pamo_r_fixed_1.0
    强制完整 rigid residual。
    这是 positive stress test，不一定最终采用。

F. part_pamo_gate_floor_0.2
    learnable gate + floor 0.2。
    如果 C/D 有用，再测这个。
```

优先判断规则：

```text
1. 如果 B 比 part_moe_leg 好:
       Step 1 的 motion encoding 是主要贡献。
       rigid/gate 应该弱化或重新设计。

2. 如果 B 不好，但 C/D 好:
       rigid residual 有用，当前 learnable gate 失败。
       下一步改 gate 策略。

3. 如果 C/D/E 都不好:
       part-level rigid residual 可能不适合当前设置，
       不建议继续在 Step 2/3 上投入太多。

4. 如果只 high-motion / leg 局部指标变好:
       该方法仍可保留，但论文表述要强调运动剧烈 part 的收益，
       不要只依赖全局平均指标。
```

## 2026-07-20 part_pamo 优化思路总结

用户要求：

```text
总结 part_pamo 的优化思路。
```

当前 part_pamo 的结构：

```text
基线:
    part_moe_leg

Step 1:
    给每个 SMPL part 加运动编码 z_part。
    z_part concat 到对应 part expert 输入。

Step 2:
    用 z_part 预测 part-level rigid residual:
        R_p, t_p = PartRigidHead(z_part)
        d_part_rigid_i = R_p * (x_i - c_p) + c_p + t_p - x_i

Step 3:
    学点级 internal rigidity:
        r_i = sigmoid(MLP_rigidity(x_emb_i, z_part_i))
        d_i = d_part_mlp_i + alpha * r_i * d_part_rigid_i
```

当前现象：

```text
代码链路已验证接通。
zero init 下与 part_moe_leg 基本等价。
PartMotionEncoder / PartRigidHead / PartRigidityMLP 都能拿到梯度。

但 0044_11 实验结果:
    原始 part_pamo 差于 part_moe_leg。
    part_pamo_gate_floor_0.05 也没有改善。

主要诊断:
    r_i 几乎贴着 0.05 下限。
    r_std 接近 0。
    rigid_contrib / mlp_contrib 最终只有 0.001872。

因此当前问题不是模块没接上，
而是 part-level rigid residual 在真实训练中的实际贡献太小，
无法影响最终渲染。
```

优化主线：

```text
主线 1：先判断 rigid residual 本身有没有用
    不要一开始就让 learnable gate 自由学习。
    先做 fixed gate:
        r=0.2
        r=0.5
        r=1.0

    如果 fixed gate 有提升:
        说明 Step 2 rigid branch 有潜力，问题是 Step 3 gate collapse。

    如果 fixed gate 也没有提升:
        说明当前 part-level rigid residual 本身可能不适合，
        需要重新考虑 c_p、part label、motion feature 或直接弱化 Step 2/3。

主线 2：把 Step 1 / Step 2 / Step 3 拆开
    part_pamo_step1_only:
        只用 z_part motion encoding，不加 rigid residual / rigidity gate。

    part_pamo_r_fixed:
        z_part + rigid residual + fixed r，不用 learnable gate。

    part_pamo_full:
        z_part + rigid residual + learnable gate。

    这样才能知道收益或退化来自哪一层。

主线 3：防止 gate 把 rigid branch 关掉
    当前 sigmoid gate 太自由。
    可选改法:
        floor gate:
            r_i = r_min + (1 - r_min) * sigmoid(...)

        fixed-to-learned warmup:
            前期固定 r_i=0.2/0.5，
            等 PartRigidHead 学出非零 residual 后再放开 gate。

        base rigid + learned residual gate:
            d_i = d_part_mlp_i + alpha * (r_base + r_learned) * d_part_rigid_i

主线 4：避免 part MLP 抢占 rigid 趋势
    part-specific MLP 表达能力强，可能直接吸收所有位移，
    导致 PartRigidHead 学不到稳定趋势。

    可尝试:
        降低 part MLP 学习率或容量。
        rigid branch 在 Part-MoE 激活后先训练若干 iter。
        对 rigid contribution ratio 加监控或轻量正则。

主线 5：用更匹配 PAMO 的评价方式
    不只看全局 novel-view 平均。
    PAMO 应主要改善:
        高运动帧
        腿部/脚踝/膝盖区域
        衣物边界随 part 整体运动的趋势

    需要补充:
        high-motion top 20% frames 指标
        low-motion bottom 20% frames 指标
        lower-body / leg 局部指标
```

推荐优先级：

```text
第一优先级:
    跑 part_pamo_step1_only。
    跑 part_pamo_r_fixed_0.2 / 0.5。

第二优先级:
    如果 fixed gate 有用，再做:
        part_pamo_gate_floor_0.2
        fixed-to-learned gate warmup

第三优先级:
    做 high-motion / lower-body 局部评测。

放弃条件:
    如果 step1_only、r_fixed_0.2、r_fixed_0.5、r_fixed_1.0 都不优于 part_moe_leg，
    且 high-motion leg 子集也没有收益，
    则不建议继续把 part_pamo 作为主创新点。
```

## 2026-07-20 part_pamo 完整机制验证启动

用户要求：

```text
用卡1和3跑完完整验证并汇报结果。
```

本次验证口径：

```text
固定 DNA-Rendering 0044_11。
使用 GPU1 和 GPU3 并行跑机制验证矩阵。

本次会重跑当前代码下的 part_moe_leg 作为公平基线，
然后跑:
    part_pamo_step1_only
    part_pamo_r_fixed_0.2
    part_pamo_r_fixed_0.5
    part_pamo_r_fixed_1.0
    part_pamo_gate_floor_0.2
```

新增独立开关：

```text
arguments/__init__.py:
    --part_pamo_step1_only
    --part_pamo_fixed_rigidity，默认 -1.0

nets/mlp_delta_non_rigid.py:
    part_pamo_step1_only=True:
        只创建/调用 PartMotionEncoder。
        不创建 PartRigidHead。
        不创建 PartRigidityMLP。
        只验证 Step 1 的 z_part motion encoding。

    part_pamo_fixed_rigidity >= 0:
        创建 PartMotionEncoder + PartRigidHead。
        不创建 PartRigidityMLP。
        r_i 固定为指定常数。

    默认:
        保持原 part_pamo 行为。
```

新增 DNA 脚本模式：

```text
scripts/exps_dnarendering.sh:
    part_pamo_step1_only
    part_pamo_r_fixed_0.2
    part_pamo_r_fixed_0.5
    part_pamo_r_fixed_1.0
    part_pamo_gate_floor_0.2
```

启动前验证：

```text
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    arguments/__init__.py
    train.py
    render.py
    gaussian_renderer/__init__.py
    scene/__init__.py
    scene/dataset_readers.py

inline CPU forward:
    step1_only:
        modules=PartMotionEncoder
        has_rigid=False
        has_gate=False

    fixed_0.2:
        modules=PartMotionEncoder,PartRigidHead
        has_rigid=True
        has_gate=False
        fixed=0.2

    learnable floor:
        modules=PartMotionEncoder,PartRigidHead,PartRigidityMLP
        has_rigid=True
        has_gate=True

    三种输出 shape 均保持:
        d_xyz [1, N, 3]
        d_rotation [1, N, 4]
        d_scaling [1, N, 3]

git diff --check
```

计划启动命令：

```text
GPU1:
    SEQUENCES_OVERRIDE=0044_11 GPU_id=1 SKIP_LOAD_TEST_CAMERAS=1 IMAGE_DATA_DEVICE=cpu \
    bash scripts/exps_dnarendering.sh part_moe_leg

    SEQUENCES_OVERRIDE=0044_11 GPU_id=1 SKIP_LOAD_TEST_CAMERAS=1 IMAGE_DATA_DEVICE=cpu \
    bash scripts/exps_dnarendering.sh part_pamo_r_fixed_0.5

    SEQUENCES_OVERRIDE=0044_11 GPU_id=1 SKIP_LOAD_TEST_CAMERAS=1 IMAGE_DATA_DEVICE=cpu \
    bash scripts/exps_dnarendering.sh part_pamo_gate_floor_0.2

GPU3:
    SEQUENCES_OVERRIDE=0044_11 GPU_id=3 SKIP_LOAD_TEST_CAMERAS=1 IMAGE_DATA_DEVICE=cpu \
    bash scripts/exps_dnarendering.sh part_pamo_step1_only

    SEQUENCES_OVERRIDE=0044_11 GPU_id=3 SKIP_LOAD_TEST_CAMERAS=1 IMAGE_DATA_DEVICE=cpu \
    bash scripts/exps_dnarendering.sh part_pamo_r_fixed_0.2

    SEQUENCES_OVERRIDE=0044_11 GPU_id=3 SKIP_LOAD_TEST_CAMERAS=1 IMAGE_DATA_DEVICE=cpu \
    bash scripts/exps_dnarendering.sh part_pamo_r_fixed_1.0
```

## 2026-07-20 part_pamo 完整机制验证停止

用户要求：

```text
停止运行。
```

执行状态：

```text
已检查当前进程:
    train.py
    render.py
    scripts/exps_dnarendering.sh

结果:
    没有发现仍在运行的 SeqAvatar 训练或渲染进程。
    因此没有额外 kill 进程。
    不删除已有输出。
```

停止前已完成的有效结果：

```text
当前代码 part_moe_leg:
    RUN_TIME 20260720_194556
    PSNR  33.019845469792685
    SSIM  0.9782838453849156
    LPIPS 0.021065409497047462
    #pts  62269

part_pamo_step1_only:
    RUN_TIME 20260720_194557
    PSNR  33.02426986694336
    SSIM  0.9782492324709892
    LPIPS 0.021121497095252077
    #pts  60691

part_pamo_r_fixed_0.2:
    RUN_TIME 20260720_203338
    PSNR  32.978520425160724
    SSIM  0.97816135485967
    LPIPS 0.02105262337718159
    #pts  61766
    final ratio 0.010664

part_pamo_r_fixed_0.5:
    RUN_TIME 20260720_203445
    PSNR  33.02952477137248
    SSIM  0.9781892021497091
    LPIPS 0.02123006567514191
    #pts  61987
    final ratio 0.012487
```

未完成项：

```text
part_pamo_gate_floor_0.2:
    训练已完成。
    RUN_TIME 20260720_212710
    #pts 59756
    final ratio 0.009767
    自动 render 失败:
        torch.cuda.is_available() is False

part_pamo_r_fixed_1.0:
    RUN_TIME 20260720_215556
    训练在 Part-MoE 激活后失败。
    错误:
        CUDA error: unknown error
```

环境异常：

```text
nvidia-smi -L 显示 GPU2 device handle unknown。
之后 CUDA_VISIBLE_DEVICES=0/1/3 的最小 torch.cuda 测试均返回:
    torch.cuda.is_available() = False
    device_count = 0

因此当前无法继续补 render 或重跑 r_fixed_1.0。
需要先恢复 GPU/NVML/CUDA 运行状态。
```

## 2026-07-20 part_pamo 已完成验证的阶段结论

用户问题：

```text
刚才跑完的能得到什么结论吗。
```

可用对照：

```text
当前代码 part_moe_leg:
    PSNR  33.019845469792685
    SSIM  0.9782838453849156
    LPIPS 0.021065409497047462

part_pamo_step1_only:
    PSNR  33.02426986694336
    SSIM  0.9782492324709892
    LPIPS 0.021121497095252077
    ΔPSNR  +0.004424397151
    ΔSSIM  -0.000034612914
    ΔLPIPS +0.000056087598

part_pamo_r_fixed_0.2:
    PSNR  32.978520425160724
    SSIM  0.97816135485967
    LPIPS 0.02105262337718159
    ΔPSNR  -0.041325044632
    ΔSSIM  -0.000122490525
    ΔLPIPS -0.000012786120
    final ratio 0.010664

part_pamo_r_fixed_0.5:
    PSNR  33.02952477137248
    SSIM  0.9781892021497091
    LPIPS 0.02123006567514191
    ΔPSNR  +0.009679301580
    ΔSSIM  -0.000094643235
    ΔLPIPS +0.000164656178
    final ratio 0.012487
```

阶段结论：

```text
1. Step 1 的 part motion encoding 有一点正向信号。
   step1_only 的 PSNR 比当前 part_moe_leg 高约 +0.0044。
   但 SSIM 和 LPIPS 略差，所以不能说稳定提升。

2. Step 2 的 rigid residual 不是完全没用。
   r_fixed_0.5 的 PSNR 比当前 part_moe_leg 高约 +0.0097，
   说明强制保留一定 rigid residual 后，至少 PSNR 有正向信号。

3. 但 rigid residual 贡献仍然偏小。
   r_fixed_0.2 final ratio=0.010664。
   r_fixed_0.5 final ratio=0.012487。
   仍低于原先希望的 0.02 - 0.2。

4. learnable gate / floor gate 仍有 collapse 倾向。
   gate_floor_0.2 训练完成时:
       r_mean=0.200001
       r_std=0.000001
       ratio=0.009767
   说明 gate 基本贴着 floor，没有学出点级 internal rigidity 差异。

5. 目前不能宣称完整 part_pamo 优于 part_moe_leg。
   已完成结果是:
       PSNR 有弱正向信号；
       SSIM/LPIPS 不稳定；
       ratio 仍偏小；
       r_i 没有形成有意义的空间差异。
```

推荐判断：

```text
part_pamo 目前适合作为继续优化候选，不适合作为已经成立的最终创新点。

更稳的下一步:
    保留 Step 1。
    暂时不要用完全 learnable gate 作为主结果。
    继续围绕 fixed/warmup gate 和 high-motion/leg 局部评测判断是否有真实收益。

如果后续补完:
    gate_floor_0.2 render
    r_fixed_1.0
    high-motion frames 指标

仍然只表现为 PSNR 微小提升、SSIM/LPIPS 不稳，
则 Step 2/3 不建议作为主创新点。
```

## 2026-07-20 part_pamo 补全未完成实验

用户要求：

```text
用卡0卡1卡3补全刚才没跑完的实验，最后告诉总的结果。
```

启动前状态：

```text
GPU/NVML 已恢复。
nvidia-smi -L 可识别 GPU0/1/2/3。
CUDA_VISIBLE_DEVICES=0/1/3 的最小 torch.cuda 测试均通过:
    torch.cuda.is_available() = True
    device_count = 1
```

补全计划：

```text
GPU0:
    补 render:
        part_pamo_gate_floor_0.2
        RUN_TIME 20260720_212710

GPU1:
    重跑完整:
        part_pamo_r_fixed_1.0

GPU3:
    保留为失败重试卡。
    不同时重复跑同一实验，避免输出混乱。
```

补全结果：

```text
已完成:
    GPU0:
        补完 part_pamo_gate_floor_0.2 的 final novelview render。
        RUN_TIME 20260720_212710

    GPU1:
        重跑并完成 part_pamo_r_fixed_1.0 的 train + final novelview render。
        RUN_TIME 20260720_234745

    GPU3:
        没有重复启动同名实验，避免产生两个 RUN_TIME 的同名结果导致判断混乱。
        本轮唯一未完成训练项是 r_fixed_1.0，已由 GPU1 成功补完。
```

0044_11 当前代码完整对照：

```text
part_moe_leg:
    RUN_TIME 20260720_194556
    PSNR  33.019845469792685
    SSIM  0.9782838453849156
    LPIPS 0.021065409497047462
    #pts  62269

part_pamo_step1_only:
    RUN_TIME 20260720_194557
    PSNR  33.02426986694336
    SSIM  0.9782492324709892
    LPIPS 0.021121497095252077
    #pts  60691
    ΔPSNR  +0.004424397151
    ΔSSIM  -0.000034612914
    ΔLPIPS +0.000056087598

part_pamo_r_fixed_0.2:
    RUN_TIME 20260720_203338
    PSNR  32.978520425160724
    SSIM  0.97816135485967
    LPIPS 0.02105262337718159
    #pts  61766
    final ratio 0.010664
    ΔPSNR  -0.041325044632
    ΔSSIM  -0.000122490525
    ΔLPIPS -0.000012786120

part_pamo_r_fixed_0.5:
    RUN_TIME 20260720_203445
    PSNR  33.02952477137248
    SSIM  0.9781892021497091
    LPIPS 0.02123006567514191
    #pts  61987
    final ratio 0.012487
    ΔPSNR  +0.009679301580
    ΔSSIM  -0.000094643235
    ΔLPIPS +0.000164656178

part_pamo_gate_floor_0.2:
    RUN_TIME 20260720_212710
    PSNR  32.98445512453715
    SSIM  0.9781745622555414
    LPIPS 0.021087642122680942
    #pts  59756
    final r_mean 0.200001
    final r_std  0.000001
    final ratio  0.009767
    ΔPSNR  -0.035390345256
    ΔSSIM  -0.000109283129
    ΔLPIPS +0.000022232626

part_pamo_r_fixed_1.0:
    RUN_TIME 20260720_234745
    PSNR  33.0063467502594
    SSIM  0.978175613284111
    LPIPS 0.02100635617195318
    #pts  62241
    final ratio 0.011873
    ΔPSNR  -0.013498719533
    ΔSSIM  -0.000108232101
    ΔLPIPS -0.000059053325
```

本轮补全后的结论：

```text
1. part_pamo 没有稳定超过 part_moe_leg。
   只有 step1_only 和 r_fixed_0.5 的 PSNR 略高:
       step1_only: +0.0044
       r_fixed_0.5: +0.0097
   但它们的 SSIM 都低于 part_moe_leg，LPIPS 也没有稳定变好。

2. Step 1 的 part motion encoding 是目前最稳的正向信号。
   它几乎不改动主结构，只让 PSNR 有轻微提升，但幅度太小，仍需要更多序列确认。

3. Step 2/3 的 rigid residual 确实接通并有梯度，但贡献偏小。
   fixed r=0.2 / 0.5 / 1.0 的 final ratio 都约为 0.01:
       0.010664 / 0.012487 / 0.011873
   即使强制 r=1，rigid contribution 也没有明显放大到预期的 0.02 - 0.2。

4. learnable gate / gate floor 仍然没有学出 internal rigidity。
   gate_floor_0.2 最终:
       r_mean=0.200001
       r_std=0.000001
   基本贴着 floor，说明点级 rigidity 没有形成空间差异。

5. r_fixed_1.0 不是更好选择。
   它 LPIPS 比 part_moe_leg 好一点:
       ΔLPIPS=-0.000059
   但 PSNR 和 SSIM 都下降:
       ΔPSNR=-0.0135
       ΔSSIM=-0.000108

总判断:
    part_pamo 当前只能作为候选方向，不能作为已经成立的主创新点。
    如果只看 0044_11，part_moe_leg 仍是更稳基线。
    part_pamo 里最值得保留继续试的是 Step 1；Step 2/3 需要重新设计约束或训练策略，
    否则 rigid residual 实际贡献太小，learnable rigidity 也容易塌到常数。
```

## 2026-07-21 part_pamo 当前方案是否值得继续优化

用户问题：

```text
总结当前方案是否值得优化。
```

判断：

```text
当前 part_pamo 不建议继续按原方案大投入优化。

值得保留:
    Step 1: part motion encoding

暂不建议作为主线继续投入:
    Step 2: part-level rigid residual branch
    Step 3: internal learnable rigidity gate
```

原因：

```text
1. 0044_11 上没有稳定超过 part_moe_leg。
   当前完整对照里只有:
       step1_only: PSNR +0.0044
       r_fixed_0.5: PSNR +0.0097
   但 SSIM/LPIPS 都不稳定，不能证明整体质量提升。

2. rigid residual 接通了，但贡献太小。
   fixed r=0.2 / 0.5 / 1.0 的 final ratio 都约 0.01。
   即使强制 r=1，rigid contribution 仍没有明显进入 0.02 - 0.2 的预期区间。

3. learnable rigidity 没学出点级差异。
   gate_floor_0.2 最终:
       r_mean=0.200001
       r_std=0.000001
   说明 gate 基本塌到常数/floor，没有体现 PaMoSplat internal rigidity 的核心价值。

4. 继续调 gate floor / fixed r 的收益不大。
   fixed r=1.0 没有带来更强收益，说明问题不是 gate 太小这么简单，
   而是当前 part rigid residual 本身没有形成有效可利用的残差运动。
```

建议：

```text
短期:
    只保留 Step 1 作为 part_pamo_step1_only 或 motion_part_moe 分支，继续多序列验证。
    不把 Step 2/3 作为主实验结果。

如果还想救 Step 2/3:
    不要继续只调 r/gate。
    应该改 rigid residual 的学习目标或约束，例如:
        只在腿/脚高运动 part 启用
        只在 high-motion frames 评估和训练增强
        给 rigid residual 显式正则/目标，避免被点级 MLP 吸收
        改为 part-local coordinate residual，而不是直接全 part rigid transform

决策:
    当前方案适合作为“探索过但不成立”的记录。
    真正值得继续优化的只有 Step 1。
```

## 2026-07-21 如何强化 part motion encoding

用户问题：

```text
part motion encoding 当前起作用很少，怎么强化它的作用。
```

当前弱的主要原因：

```text
1. z_part 只是 concat 到 expert 输入末尾。
   expert 是从原 shared MLP 复制来的，新增 z_part 输入列初始化为 0。
   这保证初始等价 part_moe_leg，但也让网络很容易继续沿用原路径，忽略 z_part。

2. part_motion_feat 信息太粗。
   当前每个 part 只有:
       mean(part_pose_t)
       mean(part_pose_{t-1})
       mean(part_pose_{t+1})
       mean(velocity)
       mean(acc)
   共 15 维。
   对腿/脚这种多关节 part，简单均值会抹掉膝盖、脚踝等局部差异。

3. 运动编码没有显式监督。
   它只通过最终 RGB/loss 间接学习，且 part-specific MLP 本身容量很大，
   很容易把运动相关残差直接吸收到点级 MLP，而不使用 z_part。

4. 运动信号没有被高运动帧强化。
   如果训练中多数帧运动不剧烈，z_part 的边际收益会被平均掉。
```

强化优先级：

```text
优先级 1：改 z_part 注入方式，不再只 concat。
    建议做 motion FiLM:
        gamma_p, beta_p = MotionFiLM(z_part)
        h = gamma_p * h + beta_p
    把 z_part 用来调制 expert 的 hidden feature。

    初始仍可保持等价:
        gamma 初始化为 1
        beta 初始化为 0

    这样 z_part 不是只作为输入列，而是直接调制 expert 中间表达，
    更难被网络忽略。

优先级 2：增强 part_motion_feat。
    不要只用 joint mean。
    对每个 part 计算:
        pose mean / std
        velocity mean / std / norm / max_norm
        acceleration mean / std / norm / max_norm
        part center velocity / acceleration，来自 SMPL posed xyz

    这样能保留:
        这个 part 整体是否在动
        part 内部关节是否不同步
        运动强度是否集中在膝盖、脚踝等局部

优先级 3：给 motion 分支更强优化权重。
    PartMotionEncoder 和 motion FiLM 单独 param group:
        lr = expert_lr * 2 或 *5
    同时对 z_part 相关新增层不要全零太久。
    可以保持输出等价初始化，但让 motion 分支的学习率更高。

优先级 4：训练时突出 high-motion frames。
    根据 part_velocity / part_acc 的 norm 给帧加权或重采样。
    对腿脚实验可以优先放大:
        left_leg_foot
        right_leg_foot
    的高运动帧。

优先级 5：给 z_part 一个辅助目标。
    例如让 z_part 预测每个 part 的平均非刚性残差或 residual norm:
        pred_part_residual = MLP_aux(z_part)
        target = stopgrad(mean(|d_xyz_part_mlp|) or mean residual per part)
    这个 loss 权重很小，只用来防止 z_part 完全无效。
```

推荐下一版消融：

```text
新增独立实验:
    part_pamo_motion_film

只基于 Step 1，不启用 Step 2/3。

保留:
    PartMotionEncoder
    part_motion_conds
    part_moe_leg schema

新增:
    1. richer part_motion_feat
    2. MotionFiLM(z_part) 调制 expert hidden feature
    3. PartMotionEncoder / MotionFiLM 更高 lr
    4. 每 1000 iter 打印:
        z_norm
        film_gamma_mean/std
        film_beta_norm
        PartMotionEncoder grad
        MotionFiLM grad

预期判断:
    如果 z_part 真起作用:
        film_beta_norm 不应长期接近 0
        gamma/std 应逐渐有变化
        high-motion frames 指标应比 part_moe_leg 更明显提升
```

不建议优先做：

```text
1. 单纯把 part_pamo_dim 从 32 加到 64/128。
   如果注入方式仍是 concat，网络仍可能忽略。

2. 继续调 Step 2/3 的 gate。
   当前问题不是 gate 数值，而是 rigid residual 本身贡献小。

3. 直接强制 z_part 大权重。
   这样容易破坏 part_moe_leg 的稳定初始化，导致指标下降。
```

## 2026-07-21 part_pamo motion encoding 强化判断实验

用户要求：

```text
part motion encoding 当前起作用很少，尝试几个判断方案。
用 GPU0 和 GPU1，在 DNA-Rendering 的 0044_11 和 0206_04 两个序列上跑，
最后总结结果。
```

新增独立实验模式：

```text
scripts/exps_dnarendering.sh part_pamo_motion_film
    基于 part_pamo_step1_only。
    保留原 Step1 concat z_part 到 expert 输入。
    额外增加 PartMotionFiLM(z_part_i)，identity init：
        gamma = 1
        beta = 0
    对每个 PartNonrigidExpert 的 hidden layers 做 FiLM。
    不启用 Step2/Step3 rigid residual。

scripts/exps_dnarendering.sh part_pamo_motion_film_rich
    基于 part_pamo_motion_film。
    part_motion_feat_mode=rich。
    PartMotionEncoder 输入从 15 维扩展到 28 维：
        原 15 维 mean features
        pose_t std
        velocity std
        acceleration std
        velocity norm mean/max
        acceleration norm mean/max
    PartMotionEncoder + PartMotionFiLM 使用单独 lr multiplier：
        --part_pamo_motion_lr_mult，默认 2.0
```

隔离性：

```text
默认:
    part_pamo_motion_film=False
    part_pamo_motion_feat_mode=mean
    part_pamo_motion_lr_mult=1.0

因此 part_moe_leg / part_pamo / part_pamo_step1_only /
part_pamo_r_fixed_* / part_pamo_gate_floor_* 的默认路径不变。
```

已做启动前验证：

```text
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    arguments/__init__.py
    nets/mlp_delta_non_rigid.py
    scene/dataset_readers.py
    scene/__init__.py
    scene/gaussian_model.py
    train.py
    render.py
    gaussian_renderer/__init__.py

CPU 小张量 forward:
    part_pamo_motion_film, mean feature:
        d_xyz      [1, 9, 3]
        d_rotation [1, 9, 4]
        d_scaling  [1, 9, 3]
        initial film_beta_abs=0.0

    part_pamo_motion_film_rich, rich feature:
        d_xyz      [1, 9, 3]
        d_rotation [1, 9, 4]
        d_scaling  [1, 9, 3]
        initial film_beta_abs=0.0

dataset reader 直接函数测试:
    mean -> part_motion_conds [1, 7, 15]
    rich -> part_motion_conds [1, 7, 28]
```

## 2026-07-21 part_pamo motion encoding 强化判断实验完整结果

用户要求：

```text
用 GPU0 和 GPU1 分别尝试 motion encoding 强化判断方案，
在 DNA-Rendering 的 0044_11 和 0206_04 两个序列上跑完并总结。
```

本次实际运行：

```text
GPU0:
    0044_11 / part_pamo_motion_film_rich
    RUN_TIME=20260721_0044_film_rich

GPU1:
    0206_04 / part_pamo_motion_film_rich
    RUN_TIME=20260721_0206_film_rich

同时纳入对比的已完成实验:
    0044_11 / part_moe_leg / 20260720_194556
    0044_11 / part_pamo_step1_only / 20260720_194557
    0044_11 / part_pamo_motion_film / 20260721_0044_film
    0206_04 / part_moe_leg / 20260630_170907
    0206_04 / part_pamo_motion_film / 20260721_0206_film
```

最终 novel-view 指标：

```text
0044_11:
    part_moe_leg:
        PSNR 33.0198454698
        SSIM 0.9782838454
        LPIPS 0.0210654095
        points 62269

    part_pamo_step1_only:
        PSNR 33.0242698669
        SSIM 0.9782492325
        LPIPS 0.0211214971
        points 60691
        vs part_moe_leg:
            PSNR +0.0044
            SSIM -0.000035
            LPIPS +0.000056

    part_pamo_motion_film:
        PSNR 32.9431526661
        SSIM 0.9780628939
        LPIPS 0.0211996398
        points 60129
        vs part_moe_leg:
            PSNR -0.0767
            SSIM -0.000221
            LPIPS +0.000134

    part_pamo_motion_film_rich:
        PSNR 32.9821028074
        SSIM 0.9781765704
        LPIPS 0.0210681518
        points 60118
        vs part_moe_leg:
            PSNR -0.0377
            SSIM -0.000107
            LPIPS +0.000003
        vs part_pamo_motion_film:
            PSNR +0.0390
            SSIM +0.000114
            LPIPS -0.000131

0206_04:
    part_moe_leg:
        PSNR 31.5415921052
        SSIM 0.9705195760
        LPIPS 0.0327569437
        points 58702

    part_pamo_motion_film:
        PSNR 31.4967257500
        SSIM 0.9700696304
        LPIPS 0.0338414388
        points 41029
        vs part_moe_leg:
            PSNR -0.0449
            SSIM -0.000450
            LPIPS +0.001084

    part_pamo_motion_film_rich:
        PSNR 31.4254674594
        SSIM 0.9702403083
        LPIPS 0.0333342852
        points 40908
        vs part_moe_leg:
            PSNR -0.1161
            SSIM -0.000279
            LPIPS +0.000577
        vs part_pamo_motion_film:
            PSNR -0.0713
            SSIM +0.000171
            LPIPS -0.000507
```

motion branch 诊断：

```text
0044_11 / part_pamo_motion_film / iter 25000:
    z_norm=2.247541e-01
    gamma_delta_abs=2.283984e-02
    beta_abs=1.063425e-03
    PartMotionEncoder grad nonzero
    PartMotionFiLM grad nonzero

0044_11 / part_pamo_motion_film_rich / iter 25000:
    z_norm=1.224082e-01
    gamma_delta_abs=4.634235e-02
    beta_abs=1.607426e-03
    PartMotionEncoder grad nonzero
    PartMotionFiLM grad nonzero

0206_04 / part_pamo_motion_film / iter 25000:
    z_norm=1.383678e-01
    gamma_delta_abs=1.979188e-02
    beta_abs=1.187511e-03
    PartMotionEncoder grad nonzero
    PartMotionFiLM grad nonzero

0206_04 / part_pamo_motion_film_rich / iter 25000:
    z_norm=3.840259e-03
    gamma_delta_abs=4.304950e-02
    beta_abs=1.771267e-03
    PartMotionEncoder grad nonzero
    PartMotionFiLM grad nonzero
```

结论：

```text
1. motion encoding / MotionFiLM 不是“没接上”:
    两个序列、mean/rich 两种设置下，PartMotionEncoder 和 PartMotionFiLM
    都有非零梯度，FiLM 的 gamma/beta 也明显偏离 identity。

2. 但当前强化方式没有带来稳定指标收益:
    0044_11:
        rich 比 mean-FiLM 好，但仍低于 part_moe_leg，
        只在 LPIPS 上几乎追平 part_moe_leg。
    0206_04:
        mean-FiLM 和 rich-FiLM 均低于 part_moe_leg，
        rich 的 LPIPS/SSIM 比 mean-FiLM 好一些，但 PSNR 更差。

3. rich feature 能强化 FiLM 调制幅度:
    gamma_delta_abs 从约 0.02 提升到约 0.043~0.046。
    说明更强 motion feature 确实会改变网络行为。
    但是这种改变目前不是正向稳定收益。

4. 0206 的点数差异较大，需要谨慎解释:
    part_moe_leg best run 有 58702 points；
    pamo motion film / rich 只有约 41000 points。
    指标下降可能同时来自 motion 注入和 densify/prune 轨迹改变。

5. 当前建议:
    part_pamo_motion_film 和 part_pamo_motion_film_rich 暂不作为主方案。
    它们证明 motion 分支可以被激活，但没有证明能提高最终重建质量。
    后续如果继续优化，优先做更保守的 gated/regularized FiLM，
    或只在 high-motion frames / high-motion parts 启用，而不是全帧全 part 调制。
```

清理状态：

```text
没有保留临时验证脚本。
没有发现仍在运行的 part_pamo_motion_film_rich train/render 进程。
```

## 2026-07-21 放弃当前 part_pamo，转向自适应动态区域建模

用户判断：

```text
当前 pamo 方法决定放弃。
动态区域 / 部件区域不能只靠统一 densify 和统一 MLP。
应该根据:
    1. 运动强弱
    2. 部件属性
    3. 局部刚性程度
决定哪里需要更强形变能力，哪里需要更稳定约束。
```

可行性判断：

```text
这个新思路是可行的，而且比当前 part_pamo 更贴合已有实验现象。

原因:
    1. part_moe_leg 已证明“部件专属形变能力”有价值。
    2. part_pamo 的失败说明:
        仅把 part motion code 注入网络，或者增加全局统一的 part-level residual，
        不足以稳定提升效果。
    3. 0044/0206 的 pamo motion film 结果显示:
        motion 分支确实能改变网络行为，但改变不一定正向。
    4. 0206 中 pamo 点数明显少于 part_moe_leg，提示 densify/prune 轨迹
        也可能是动态区域建模失败的重要因素。

因此下一步更合理的方向不是继续加 motion encoder，
而是做“motion-aware / part-aware / rigidity-aware 的容量和约束分配”。
```

建议的新方向命名：

```text
part_adaptive_dynamic
或
motion_aware_part_capacity
```

核心思想：

```text
对每个 Gaussian 或每个 part 估计一个 dynamic score:
    s_i = f(part_id, motion_strength, local_rigidity, residual/error)

用 s_i 控制:
    1. densify/prune 阈值
    2. non-rigid MLP 容量或 expert 路由
    3. 正则强度
    4. 是否更依赖稳定刚性约束

高动态 / 低刚性区域:
    更容易 densify
    更强 non-rigid capacity
    更弱刚性/平滑约束

低动态 / 高刚性区域:
    更保守 densify
    更强稳定约束
    避免过拟合和漂移
```

优先实现建议：

```text
第一步先不要改网络主体。
先做 motion-aware densify/prune 或 part-aware densify schedule。
原因:
    当前 part_pamo 的一个明显问题是点数轨迹被改变；
    如果动态区域没有足够 Gaussian，后面的 MLP 再强也补不回来。

第二步再做 capacity 分配:
    对高动态 part 使用更宽/更深 expert，或更高 expert LR。
    对低动态 part 保持 part_moe_leg 原设置。

第三步再加 rigidity-aware regularization:
    不是直接加 rigid residual，
    而是用局部刚性分数控制正则强弱。
```

## 2026-07-21 新方案：motion/part/rigidity adaptive capacity

用户要求：

```text
不要再想 PaMoSplat residual。
重新设计一个方案:
    根据运动强弱、部件属性、局部刚性程度，
    决定哪里需要更强形变能力，哪里需要更稳定约束。
```

方案命名：

```text
MPAC:
    Motion-Part Adaptive Capacity

或中文:
    运动-部件-刚性自适应容量分配
```

核心观点：

```text
不是给所有 Gaussian 同等 densify、同等 MLP、同等正则。
而是给每个 Gaussian / part 计算 capacity score:
    capacity_i = f(motion_i, part_prior_i, rigidity_i, error_i)

capacity_i 高:
    说明这个区域需要更强表达能力。

stability_i 高:
    说明这个区域应该更稳定，少自由形变。
```

输入信号：

```text
1. motion_strength:
    来自 SMPL part 的 velocity / acceleration。
    可以先按 part 计算，再 broadcast 到该 part 的 Gaussian。

2. part_prior:
    部件属性先验。
    torso/head:
        更稳定。
    upper/lower leg、arm:
        中等。
    foot/hand、关节边界、衣物边缘:
        更动态、更需要形变容量。

3. local_rigidity:
    不做 rigid residual。
    只估计这个 Gaussian 是否局部刚性。
    可由以下信号构成:
        skinning weight entropy
        到关节边界距离
        同 part 邻域 residual 方差
        同 part 邻域 motion 一致性

4. error_score:
    训练中的 EMA photometric residual / gradient norm。
    高运动但低误差:
        不一定加容量。
    高运动且高误差:
        优先加容量和 densify。
```

网络设计：

```text
以 part_moe_leg 为基线。
每个 part 不再只有一个 expert，而是分成两个容量等级:

    StableExpert_p:
        低容量、强约束。
        用于刚性强、运动弱、误差低区域。

    DynamicExpert_p:
        高容量、弱约束。
        用于运动强、局部刚性低、误差高区域。

由 CapacityRouter 输出:
    g_i = sigmoid(MLP_capacity(
        x_emb_i,
        part_onehot_i,
        motion_strength_i,
        rigidity_i,
        error_score_i
    ))

最终 deformation:
    d_i = (1 - g_i) * StableExpert_p(input_i)
        + g_i       * DynamicExpert_p(input_i)

这里 g_i 不是 PaMoSplat rigid gate。
它控制“使用稳定专家还是动态专家”，本质是自适应容量路由。
```

densify / prune 设计：

```text
给每个 Gaussian 一个 densify_priority:
    densify_priority_i =
        render_grad_i
        * (1 + a * motion_strength_i)
        * (1 + b * error_score_i)
        * (1 + c * (1 - rigidity_i))
        * part_weight_prior_i

高动态、低刚性、高误差区域:
    降低 densify threshold。
    更容易 split / clone。

高刚性、低误差区域:
    提高 densify threshold。
    避免无意义增点和漂移。

prune 也对应调整:
    动态高误差区域不要过早 prune。
    稳定低贡献区域可以正常 prune。
```

约束设计：

```text
不统一正则。
每个 Gaussian 的稳定约束权重:
    lambda_stable_i =
        lambda_base
        * rigidity_i
        * (1 - motion_strength_i)
        * (1 - error_score_i)

对高 rigidity 区域:
    加强 temporal smooth / neighbor consistency / deformation magnitude regularization。

对低 rigidity 高 motion 区域:
    降低这些约束，让 DynamicExpert 有空间拟合衣物边缘、关节附近、快速摆动。
```

为什么它比 part_pamo 更合理：

```text
part_pamo 的问题:
    给所有点同样注入 motion code。
    motion 分支确实学习了，但没有判断“哪里该用、用多少、是否需要更多点”。

MPAC 的重点:
    先判断区域类型，再分配建模资源。
    高动态区域获得:
        更多点
        更强 expert
        更弱稳定约束
    稳定区域获得:
        更少额外自由度
        更强约束
        更稳定训练
```

建议消融顺序：

```text
Ablation 1:
    part_moe_leg + motion/part/rigidity aware densify only
    不改网络。
    目标:
        先验证点数分布是否改善，尤其 0206 的动态区域点数是否不足。

Ablation 2:
    加 CapacityRouter + Stable/Dynamic dual experts。
    不加复杂正则。
    目标:
        验证动态区域是否需要更强 expert。

Ablation 3:
    加 rigidity-aware regularization。
    目标:
        验证稳定区域是否能减少漂移、减少过拟合。

Ablation 4:
    三者组合。
    目标:
        完整 MPAC。
```

预期验证指标：

```text
除了 PSNR/SSIM/LPIPS，还要看:
    每个 part 的 Gaussian 数量变化
    高 motion part 的 densify 次数
    g_i 的均值/方差/per-part 分布
    StableExpert 和 DynamicExpert 的使用比例
    deformation norm 在高/低 rigidity 区域的差异
    训练后低 motion 区域是否更稳定
```

## 2026-07-25 是否试过加 optical flow

用户问题：

```text
之前有没有试过加光流的做法
```

检查结果：

```text
没有发现之前真正做过 optical flow 相关实验。

repo / note 中搜索:
    光流
    optical
    flow
    RAFT / raft
    FlowNet / flownet

只有 note 里早期的:
    note/state_warm_a04_flow.svg
    note/state_warm_a04_flow_zh.svg

这里的 flow 是流程图 flow chart，不是 optical flow。
```

结论：

```text
之前的 state / part_pamo / motion_film 都是基于 pose、part、velocity、acceleration、
FiLM 或 expert routing 的运动建模，没有引入图像光流监督、光流一致性 loss、
RAFT 预估光流、或基于光流的 densify/动态区域判断。
```

## 2026-07-25 清理多余消融实验，只保留 original 与 part_moe_leg

用户要求：

```text
删除多余的消融实验代码及其开关。
只保留 original 和 part_moe_leg。
之后如果再提到 part_moe，默认就是 part_moe_leg 这个实验。
```

实现口径：

```text
保留:
    original / orginal baseline
    part_moe_leg

脚本别名:
    use_part_moe
    part_moe
    use_part_moe_leg
    part_moe_leg

全部统一映射到:
    experiment_name=part_moe_leg
    --use_part_moe
    --part_label_schema part_moe_leg
    --num_parts 7
```

已删除/停用：

```text
part_pamo 及所有变体:
    --use_part_pamo
    --part_pamo_dim
    --part_pamo_log_interval
    --part_pamo_rigidity_min
    --part_pamo_step1_only
    --part_pamo_fixed_rigidity
    --part_pamo_motion_film
    --part_pamo_motion_feat_mode
    --part_pamo_motion_lr_mult

网络分支:
    PartMotionEncoder
    PartMotionFiLM
    PartRigidHead
    PartRigidityMLP
    part-level rigid residual
    internal rigidity gate

其他消融 schema:
    part_moe_foot
    part_moe_arm
```

代码修改：

```text
arguments/__init__.py
    删除 part_pamo 参数开关。

nets/mlp_delta_non_rigid.py
    NonrigidDeformer 只保留 shared MLP 和 Part-MoE experts。
    PartNonrigidExpert 不再加宽输入，不再支持 FiLM。
    forward_part_moe 只按 part label 路由和 global/part warmup blend。

scene/dataset_readers.py
    cond_dict 不再生成 part_motion_conds。
    删除 part motion 预计算 helper。

scene/__init__.py
    不再向 dataset reader 传 part_pamo 参数。

gaussian_renderer/__init__.py
    non_rigid_deformer 调用不再传 part_motion_conds / query_xyz。

scene/gaussian_model.py
    删除 part_pamo 属性和特殊 motion LR 参数组。

train.py
    删除 part_pamo diagnostics / grad stats 日志。

part_label/common.py
    只保留 anatomy5 和 part_moe_leg schema。

scripts/exps_zjumocap.sh
scripts/exps_i3dhuman.sh
scripts/exps_dnarendering.sh
    只支持 orginal/original 和 use_part_moe/part_moe/part_moe_leg。
    part_moe 统一输出到 part_moe_leg 目录。
    DNA part_moe_leg 继续保留 final_eval_only=1。

freeview/freeview.py
freeview/render_circle.py
freeview/render_grid_video.py
    默认 experiment 从 part_moe_arm 改为 part_moe_leg。
```

已验证：

```text
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    arguments/__init__.py
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    gaussian_renderer/__init__.py
    scene/__init__.py
    scene/dataset_readers.py
    train.py
    render.py
    part_label/common.py
    freeview/freeview.py
    freeview/render_circle.py
    freeview/render_grid_video.py

最小 Part-MoE forward:
    NonrigidDeformer(use_part_moe=True, num_parts=7)
    init_part_moe_from_shared()
    forward(part_label, part_moe_alpha=0.5)

输出:
    d_xyz      [1, 11, 3]
    d_rotation [1, 11, 4]
    d_scaling  [1, 11, 3]

残留扫描:
    part_pamo / use_part_pamo / PartMotion / PartRigid / Rigidity /
    part_moe_foot / part_moe_arm / motion_film
    在可执行代码中无命中。

git diff --check:
    通过。
```

运行计划：

```text
DNA 六序列 tri 实验已启动。

GPU0:
    RUN_TIME=20260726_0020_tri_gpu0
    sequences:
        0044_11
        0051_09
        0206_04
    global log:
        logs/tri/20260726_0020_tri_gpu0_DNA-Rendering_tri.log

GPU1:
    RUN_TIME=20260726_0020_tri_gpu1
    sequences:
        0813_05
        0007_04
        0019_10
    global log:
        logs/tri/20260726_0020_tri_gpu1_DNA-Rendering_tri.log
```

注意：

```text
state 关键字仍会在 safe_state、checkpoint state_dict、optimizer state 中出现。
这些是正常运行状态，不是已删除的 state 消融实验代码。
```

## 2026-07-25 收紧脚本入口别名

用户追问：

```text
use_part_moe / part_moe / use_part_moe_leg / part_moe_leg 都统一映射到 part_moe_leg 是啥意思？
没删吗？
```

澄清与修正：

```text
之前保留的是脚本入口兼容别名，不是独立实验分支。
为避免歧义，本次继续收紧脚本 mode：
    只接受 orginal/original
    只接受 part_moe_leg

删除的脚本 mode 别名:
    use_part_moe
    part_moe
    use_part_moe_leg

保留的 Python 参数:
    --use_part_moe

原因:
    --use_part_moe 是启用 Part-MoE 逻辑的内部布尔开关。
    part_moe_leg 实验运行时仍必须把它传给 train.py/render.py。
    它不是一个独立消融实验名。
```

当前运行方式：

```text
baseline:
    bash scripts/exps_*.sh orginal
    或
    bash scripts/exps_*.sh original

Part-MoE 腿部分组:
    bash scripts/exps_*.sh part_moe_leg
```

复查结果：

```text
bash -n 三个实验脚本通过。
git diff --check 通过。

脚本 case 入口已经不再接受:
    use_part_moe
    part_moe
    use_part_moe_leg

脚本中仍出现 --use_part_moe 的位置只用于 PART_MOE_ARGS，
它是 train.py/render.py 启用 Part-MoE 的内部参数，不是实验 mode。
```

## 2026-07-25 删除 orginal 拼写错误入口

用户说明：

```text
orginal 是之前打错了，删掉，只保留 original。
```

修正：

```text
三个实验脚本默认 MODE 改为:
    original

baseline case 入口只接受:
    original

baseline 输出 experiment_name 改为:
    original

脚本支持模式现在严格为:
    original
    part_moe_leg
```

## 2026-07-25 新建 tri 独立消融实验开关

用户要求：

```text
新建一个独立的消融实验开关 tri。
相关实验日志保存在:
    /media/image/mxz/human/SeqAvatar/logs/tri

注意:
    代码和其他消融实验相互独立。
    必须在 part_moe_leg 的基础上进行。
    不是在 original 的基础上修改。
```

实现口径：

```text
新增 Python 开关:
    --use_tri

新增脚本 mode:
    tri

tri 模式强制继承 part_moe_leg 配置:
    --use_part_moe
    --use_tri
    --part_label_schema part_moe_leg
    --num_parts 7

输出实验目录:
    output/<dataset>/<sequence>/tri/<RUN_TIME>/

总日志目录:
    /media/image/mxz/human/SeqAvatar/logs/tri

Part-MoE 自动分层日志:
    /media/image/mxz/human/SeqAvatar/logs/tri/.auto_<RUN_TIME>_tri
```

代码修改：

```text
arguments/__init__.py
    新增 use_tri=False。

scene/gaussian_model.py
    读取 use_tri。
    如果 use_tri=True，则强制校验:
        use_part_moe=True
        part_label_schema=part_moe_leg
        num_parts=7

nets/mlp_delta_non_rigid.py
    NonrigidDeformer 新增 use_tri。
    新增 forward_tri() 独立 hook。
    当前 forward_tri() 等价调用 forward_part_moe()，
    后续 TRI 的网络级创新只改这个分支，不影响 original / part_moe_leg。

part_label/common.py
    use_tri=True 时 part 日志前缀使用 tri。
    未显式传 part_log_dir 时，默认日志目录为 logs/tri。

ablations/part_moe_controller.py
    part label meta 中记录 use_tri。

scripts/exps_zjumocap.sh
scripts/exps_i3dhuman.sh
scripts/exps_dnarendering.sh
    新增 tri mode。
    tri mode 使用 logs/tri。
    tri mode 仍走 part_moe_leg 的分组、启动 iteration、warmup、global_keep。
    DNA tri 沿用 part_moe_leg 的 final_eval_only=1。
```

当前状态：

```text
tri 目前是独立实验开关和独立网络 hook。
在具体 TRI 网络结构实现前，forward_tri 暂时等价于 part_moe_leg。
这样可以保证新实验入口、日志、cfg_args、label meta 已经隔离，
同时不提前改变 part_moe_leg 和 original 的行为。
```

已验证：

```text
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    arguments/__init__.py
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    part_label/common.py
    ablations/part_moe_controller.py
    train.py
    render.py

最小 TRI forward:
    NonrigidDeformer(use_part_moe=True, use_tri=True, num_parts=7)
    init_part_moe_from_shared()
    forward(part_label, part_moe_alpha=0.5)

输出:
    use_tri True
    d_xyz      [1, 11, 3]
    d_rotation [1, 11, 4]
    d_scaling  [1, 11, 3]

git diff --check:
    通过。
```

## 2026-07-26 tri：canonical Gaussian tri-plane feature

用户要求：

```text
在 DNA 数据集上进行 tri 实验。
在 canonical Gaussian 坐标处采样三平面特征:
    x_i -> sample tri-plane -> f_tri(x_i)

再 concat 到 non-rigid MLP 输入:
    input_i = concat(x_emb_i, pose_feat, seq_pose_feat, seq_xyz_feat, f_tri_i)

要求:
    在六个 DNA 序列上跑完并给出评价指标。
```

实现口径：

```text
tri 仍然只在 part_moe_leg 基础上启用:
    --use_part_moe
    --use_tri
    --part_label_schema part_moe_leg
    --num_parts 7

original 和 part_moe_leg 不采样 tri-plane，不加宽输入维度。
```

网络修改：

```text
nets/mlp_delta_non_rigid.py
    新增 TriPlaneFeature:
        learnable planes: [3, tri_plane_dim, tri_plane_res, tri_plane_res]
        默认:
            tri_plane_dim=32
            tri_plane_res=64
            tri_plane_extent=1.2

    对 canonical query_xyz 归一化到 [-1, 1]:
        xyz_norm = clamp(query_xyz / tri_plane_extent, -1, 1)

    分别采样:
        XY plane
        XZ plane
        YZ plane

    f_tri_i:
        三个 plane feature 平均，shape [B, N, tri_plane_dim]

    use_tri=True 时:
        feats = [x_emb, pose_feat, seq_pose_feat, seq_xyz_feat, f_tri]
        再进入 shared MLP / forward_tri / Part-MoE experts。

    Part-MoE experts 初始化时复制已经带 tri 输入的 shared MLP。
```

初始化与训练：

```text
tri-plane 参数零初始化。
MLP 第一层只在 use_tri=True 时增加 tri_plane_dim 输入。
新增 tri 输入列保留随机初始化，使 f_tri=0 时初始输出不受影响，
同时 tri-plane 可以通过这些随机输入列获得非零梯度。

tri-plane 参数属于 NonrigidDeformer 子模块，
因此被现有 non_rigid_deformer optimizer 参数组覆盖。
Part-MoE 启动后 freeze_shared_after_part_moe 只冻结 shared MLP/head，
不会冻结 TriPlaneFeature。
```

链路修改：

```text
gaussian_renderer/__init__.py
    调用 non_rigid_deformer 时传入:
        query_xyz=means3D

    means3D 是进入 non-rigid deformer 前的 canonical Gaussian 坐标。

arguments/__init__.py
    新增:
        --tri_plane_dim
        --tri_plane_res
        --tri_plane_extent

scene/gaussian_model.py
    保存并传入 tri_plane_* 参数。

scripts/exps_dnarendering.sh
scripts/exps_zjumocap.sh
scripts/exps_i3dhuman.sh
    tri mode 下显式传入 tri_plane_*。
```

已验证：

```text
bash -n 三个实验脚本通过。

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    arguments/__init__.py
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    gaussian_renderer/__init__.py
    part_label/common.py
    ablations/part_moe_controller.py
    train.py
    render.py

最小 TRI forward + grad:
    plain_in = 9
    tri_in   = 17   # 测试中 tri_plane_dim=8
    out:
        d_xyz      [1, 11, 3]
        d_rotation [1, 11, 4]
        d_scaling  [1, 11, 3]
    tri_grad_max = 4.0174793684855103e-04

git diff --check:
    通过。
```

## 2026-07-26 tri DNA 六序列完整运行结果

用户要求：

```text
在 DNA 数据集上进行 tri 实验。
在 canonical Gaussian 坐标处采样三平面特征:
    x_i -> sample tri-plane -> f_tri(x_i)

再 concat 到 non-rigid MLP 输入:
    input_i = concat(x_emb_i, pose_feat, seq_pose_feat, seq_xyz_feat, f_tri_i)

要求在六个 DNA 序列上跑完并给出评价指标。
```

运行命令：

```text
RUN_TIME=20260726_0020_tri_gpu0 GPU_id=0 SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" \
    bash scripts/exps_dnarendering.sh tri

RUN_TIME=20260726_0020_tri_gpu1 GPU_id=1 SEQUENCES_OVERRIDE="0813_05 0007_04 0019_10" \
    bash scripts/exps_dnarendering.sh tri
```

日志：

```text
logs/tri/20260726_0020_tri_gpu0_DNA-Rendering_tri.log
logs/tri/20260726_0020_tri_gpu1_DNA-Rendering_tri.log
```

运行确认：

```text
六个序列均完成 train.py + render.py。
两个总日志均显示:
    All sequences finished.

六个序列均在 10000 iteration 成功:
    Build labels
    Loaded part labels
    Initializing 7 experts from the shared non-rigid MLP
    Added 7 expert param groups

说明 tri 是跑在 part_moe_leg 之上，且 expert 初始化、路由和 render 加载
part labels 的链路均通过。
```

TRI 指标：

```text
结果文件，由 train.py 在 25000 eval 时写出:
    output/DNA-Rendering/<seq>/tri/<run_time>/metrics/results_novelview_25000.json

seq       PSNR       SSIM       LPIPS
0007_04   29.5785    0.9586     0.0444
0019_10   35.4580    0.9816     0.0206
0044_11   33.0066    0.9782     0.0211
0051_09   28.6086    0.9712     0.0308
0206_04   31.5952    0.9709     0.0331
0813_05   36.2102    0.9873     0.0178
avg       32.4095    0.9746     0.0280
```

render.py 总日志复算指标：

```text
seq       PSNR       SSIM       LPIPS
0007_04   29.5787    0.9587     0.0444
0019_10   35.4591    0.9816     0.0205
0044_11   33.0067    0.9782     0.0211
0051_09   28.7028    0.9719     0.0303
0206_04   31.5964    0.9709     0.0331
0813_05   36.2146    0.9873     0.0178
avg       32.4264    0.9747     0.0279
```

与已有 part_moe_leg 六序列完整基线 `20260623_180431` 对比
（使用 results_novelview_25000.json）：

```text
seq       dPSNR      dSSIM      dLPIPS
0044_11   +0.0061   -0.0001    -0.0000
0051_09   +0.0037   +0.0002    -0.0005
0206_04   +0.0779   +0.0006    -0.0005
0813_05   +0.0117   -0.0000    -0.0001
0007_04   +0.0120   -0.0002    +0.0005
0019_10   +0.1026   +0.0003    -0.0001
avg       +0.0357   +0.0001    -0.0001
```

结论：

```text
TRI 相比 part_moe_leg 的平均提升为:
    PSNR  +0.0357
    SSIM  +0.0001
    LPIPS -0.0001

方向整体为正，但幅度很小。
当前 tri-plane concat 到 non-rigid MLP 的方案可作为有效但弱提升的第二创新点初版；
如果继续优化，重点应放在增强 tri feature 的使用强度，而不是继续扩大普通 MLP。
```

## 2026-07-26 tri feature 强化方向

用户问题：

```text
怎么进一步强化 tri-plane feature 的使用强度。
是进行调参实验吗？跟什么参数有关？
```

当前实现里 tri-plane 的入口：

```text
query_xyz -> TriPlaneFeature -> f_tri
f_tri 只在 shared non-rigid MLP 的第一层被 concat 一次
然后再复制到 part_moe_leg experts。
```

最直接的可调参数：

```text
tri_plane_dim
    feature 维度，越大容量越强，但也更容易被其他特征淹没。

tri_plane_res
    平面分辨率，越高越细，但显存和优化难度更高。

tri_plane_extent
    归一化范围，太大时大量点落在 [-1,1] 中间窄区，太小会 clip。

tri_plane 初始化和学习率
    当前是零初始化，收敛更稳，但 tri 进入模型较慢。
```

更关键的不是单纯调参，而是“增强注入方式”：

```text
1. 给 f_tri 加独立投影或门控
   例如 tri_proj(f_tri) 后再进 MLP，或者 tri_gate * f_tri。

2. 把 tri 特征注入多层，而不是只在第一层 concat 一次
   这样 tri 不容易被前层压掉。

3. 给 tri 一个残差路径
   让 d_xyz / d_rotation / d_scaling 显式依赖 tri 分支输出。

4. 降低 tri 的稀释效应
   让 tri 和 pose / seq 特征先分别编码，再融合。

5. 用更贴近人体的三平面
   全局 tri-plane 可以换成 body-centric / part-centric tri-plane。
```

经验上优先顺序：

```text
先调 tri_plane_extent / tri_plane_dim / tri_plane_res
再加 tri_proj 或门控
再考虑多层注入或 part-centric tri-plane
```

## 2026-07-26 tri sweep 计划

目标：

```text
用 0/1/2 三张卡在六个 DNA 序列上做 tri-plane 调参，
尽量增强 tri-plane feature 的使用强度，并选出最好的配置。
```

候选配置：

```text
A:
    TRI_PLANE_DIM=64
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.2

B:
    TRI_PLANE_DIM=64
    TRI_PLANE_RES=96
    TRI_PLANE_EXTENT=1.0
```

说明：

```text
A 主要测“加大 tri 容量”。
B 主要测“加大 tri 容量 + 提高空间细节 + 稍微收紧 canonical 覆盖范围”。
```

## 2026-07-26 tri sweep A 结果

当前已完成配置 A:

```text
TRI_PLANE_DIM=64
TRI_PLANE_RES=64
TRI_PLANE_EXTENT=1.2
```

六序列最终结果来自:

```text
output/DNA-Rendering/<seq>/tri/20260726_triA_d64_r64_e12/metrics/results_novelview_25000.json
```

结果汇总:

| Sequence | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 0007_04 | 29.5739 | 0.9588 | 0.0442 |
| 0019_10 | 35.4125 | 0.9815 | 0.0208 |
| 0044_11 | 33.0145 | 0.9782 | 0.0213 |
| 0051_09 | 28.6164 | 0.9713 | 0.0310 |
| 0206_04 | 31.5893 | 0.9709 | 0.0329 |
| 0813_05 | 36.1516 | 0.9872 | 0.0181 |
| Average | 32.3930 | 0.9746 | 0.0280 |

对比默认 tri (`20260726_0020`)：

```text
Default tri average:
    PSNR 32.4095
    SSIM 0.9746
    LPIPS 0.0280

A 相比默认 tri:
    PSNR -0.0165
    SSIM +0.0000
    LPIPS +0.0001
```

阶段性结论:

```text
A 没有稳定超过默认 tri，只能算持平略弱。
如果继续做 tri-plane 调参，B 组仍然值得跑。
```

## 2026-07-26 tri sweep B 计划

准备启动配置 B:

```text
TRI_PLANE_DIM=64
TRI_PLANE_RES=96
TRI_PLANE_EXTENT=1.0
```

计划沿用同样的三卡拆分:

```text
GPU 0: 0044_11 0007_04
GPU 1: 0051_09 0206_04
GPU 2: 0813_05 0019_10
```

## 2026-07-26 tri sweep B2 结果

配置 B 原先使用后台 `nohup` 启动时没有真正进入训练，因此本次有效结果使用 B2：

```text
RUN_TIME=20260726_triB2_d64_r96_e10
TRI_PLANE_DIM=64
TRI_PLANE_RES=96
TRI_PLANE_EXTENT=1.0
```

六序列 `results_novelview_25000.json` 结果：

| Sequence | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 0007_04 | 29.5654 | 0.9590 | 0.0438 |
| 0019_10 | 35.3991 | 0.9814 | 0.0208 |
| 0044_11 | 32.9949 | 0.9783 | 0.0212 |
| 0051_09 | 28.6507 | 0.9713 | 0.0308 |
| 0206_04 | 31.5965 | 0.9709 | 0.0330 |
| 0813_05 | 36.1984 | 0.9872 | 0.0180 |
| Average | 32.4008 | 0.9747 | 0.0279 |

和默认 tri 对比：

```text
默认 tri average:
    PSNR  32.4095
    SSIM  0.9746
    LPIPS 0.0280

B2 相比默认 tri:
    PSNR  -0.0087
    SSIM  +0.0001
    LPIPS -0.0000
```

阶段性结论：

```text
B2 的 LPIPS 和 SSIM 略好，但 PSNR 没有超过默认 tri。
单纯把 dim 提到 64、res 提到 96 并收紧 extent 到 1.0，不是稳定更优配置。
```

## 2026-07-26 tri sweep C 计划

为了单独验证“增强 tri-plane 使用强度”里最直接的坐标覆盖因素，准备补 C 组：

```text
RUN_TIME=20260726_triC_d32_r64_e10
TRI_PLANE_DIM=32
TRI_PLANE_RES=64
TRI_PLANE_EXTENT=1.0
```

与默认 tri 相比，C 只把 `tri_plane_extent` 从 1.2 收紧到 1.0，不扩大 dim/res。
目的：减少过大 extent 对 canonical 坐标采样的稀释，观察是否比单纯加容量更有效。

## 2026-07-26 tri sweep C 结果

配置 C：

```text
RUN_TIME=20260726_triC_d32_r64_e10
TRI_PLANE_DIM=32
TRI_PLANE_RES=64
TRI_PLANE_EXTENT=1.0
```

六序列最终结果：

| Sequence | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 0007_04 | 29.5694 | 0.9587 | 0.0447 |
| 0019_10 | 35.4260 | 0.9816 | 0.0205 |
| 0044_11 | 33.0195 | 0.9782 | 0.0210 |
| 0051_09 | 28.6221 | 0.9712 | 0.0308 |
| 0206_04 | 31.6237 | 0.9710 | 0.0330 |
| 0813_05 | 36.2604 | 0.9874 | 0.0178 |
| Average | 32.4202 | 0.9747 | 0.0280 |

与默认 tri / A / B2 对比：

```text
default tri:
    PSNR  32.4095
    SSIM  0.974615
    LPIPS 0.027974

A_d64_r64_e12:
    PSNR  32.3930
    SSIM  0.974642
    LPIPS 0.028034

B2_d64_r96_e10:
    PSNR  32.4008
    SSIM  0.974675
    LPIPS 0.027933

C_d32_r64_e10:
    PSNR  32.4202
    SSIM  0.974673
    LPIPS 0.027966
```

结论：

```text
C 是当前四组里最好的。
它没有带来大幅提升，但在 PSNR 上超过默认 tri，LPIPS 也基本持平略优。
tri-plane 最有效的强化方向不是继续增大 dim/res，而是把 extent 收紧到更贴近人体 canonical 分布。
```

异常记录：

```text
0206_04 在 C 组第一次用 GPU1 跑到约 2480 iter 时触发 CUDA illegal memory access。
已删除不完整 output 后，改用 GPU3 重跑成功完成。
最终平均使用的是 GPU3 完整结果。
```

## 2026-07-26 tri 最佳全序列精确指标

用户要求：

```text
总结最好的一次全序列评价指标 psnr / ssim / lpips*1000，不要四舍五入。
```

最佳配置仍为：

```text
RUN_TIME=20260726_triC_d32_r64_e10
TRI_PLANE_DIM=32
TRI_PLANE_RES=64
TRI_PLANE_EXTENT=1.0
```

从六个 `results_novelview_25000.json` 读取的精确结果：

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0007_04 | 29.56935234069824 | 0.9587227498491605 | 44.70729020734628000 |
| 0019_10 | 35.42599445978801 | 0.9816343893607458 | 20.45246589307984000 |
| 0044_11 | 33.01947676340739 | 0.9781564354896546 | 21.018070510278146000 |
| 0051_09 | 28.622101895014445 | 0.9711858620246251 | 30.79613036631296000 |
| 0206_04 | 31.623733441034954 | 0.9709839204947154 | 32.99347322123746000 |
| 0813_05 | 36.260351737340294 | 0.9873521268367768 | 17.82754632489135000 |
| Average | 32.42016843954722216666666667 | 0.9746725806759463666666666667 | 27.96582942052433933333333333 |

## 2026-07-26 triC 参数与第一次 tri 对比

用户问题：

```text
triC_d32_r64_e10 这次具体参数设置是什么，和第一次 tri 相比进行了哪些修改。
```

对比依据：

```text
第一次 tri:
    output/DNA-Rendering/0044_11/tri/20260726_0020_tri_gpu0/cfg_args

triC:
    output/DNA-Rendering/0044_11/tri/20260726_triC_d32_r64_e10/cfg_args
```

共同设置：

```text
use_part_moe=True
use_tri=True
part_label_schema=part_moe_leg
num_parts=7

part_moe_start_iter=10000
part_moe_warmup=1000
part_moe_global_keep=0.1

iterations=25000
densify_until_iter=1500
final_eval_only=1
test_iterations=25000
save_iterations=25000

seq_len=8
seq_xyz_knn=8
time_step_num=3
max_time_step=3
minimal_time_step=1

non_rigid_mlp_depth=3
non_rigid_mlp_width=512

l1_loss_w=1.0
ssim_loss_w=0.01
lpips_loss_w=0.01
```

tri-plane 参数对比：

| Setting | 第一次 tri | triC_d32_r64_e10 |
|---|---:|---:|
| tri_plane_dim | 32 | 32 |
| tri_plane_res | 64 | 64 |
| tri_plane_extent | 1.2 | 1.0 |

结论：

```text
triC 没有改网络结构、Part-MoE 设置、MLP 宽深、训练轮数或 loss 权重。
它相对第一次 tri 的核心修改只有:
    TRI_PLANE_EXTENT: 1.2 -> 1.0

含义是把 canonical Gaussian 坐标采样三平面时的归一化范围收紧，
让人体附近的 Gaussian 在 tri-plane 上占据更大的有效坐标范围，
减少 tri feature 被过大 extent 稀释。
```

## 2026-07-26 part_moe 全部相关实验日志最佳结果查询

用户问题：

```text
在日志中查询关于 part_moe 的实验（不止 part_moe_leg）最好的一次结果是什么。
```

查询口径：

```text
优先使用最终 render.py 日志中的指标，而不是训练中间 evaluation。
DNA-Rendering 按完整六序列 novelview 平均 PSNR 作为主排序。
跳过 output 中 latest 重复目录。
同时检查 logs/part 中的 part_moe / part_moe_leg / part_moe_foot /
part_moe_arm / part_moe_pair / part_moe_leg_msti 等相关记录。
```

DNA-Rendering 完整六序列最佳：

```text
实验: part_moe_pair
RUN_TIME: 20260702_004310
日志:
    logs/part/20260702_004310_DNA-Rendering_part_moe_pair.log

Average:
    PSNR        32.412569573190474
    SSIM        0.974700798590978
    LPIPS*1000  27.84530530787177
```

逐序列最终 render.py 指标：

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0007_04 | 29.603131167093913 | 0.9588047777613004 | 44.05874328998228 |
| 0019_10 | 35.39810724258423 | 0.9813995202382405 | 20.683477298977472 |
| 0044_11 | 32.994045162200926 | 0.9782125651836395 | 21.130275043348473 |
| 0051_09 | 28.759435685475665 | 0.9719773272673289 | 30.416719110993046 |
| 0206_04 | 31.52202189763387 | 0.9704670558373133 | 32.92893118535479 |
| 0813_05 | 36.19867628415425 | 0.9873435452580451 | 17.85368591857453 |
| Average | 32.412569573190474 | 0.974700798590978 | 27.84530530787177 |

完整 DNA 六序列 top 对比：

| Rank | Experiment | RUN_TIME | PSNR | SSIM | LPIPS*1000 |
|---:|---|---|---:|---:|---:|
| 1 | part_moe_pair | 20260702_004310 | 32.412569573190474 | 0.974700798590978 | 27.84530530787177 |
| 2 | part_moe_arm | 20260624_174257 | 32.4106435696284 | 0.9747590608894825 | 27.902932606068337 |
| 3 | part_moe_arm | 20260623_231810 | 32.40436245600383 | 0.9747458954652152 | 27.841817202149997 |
| 4 | part_moe_leg | 20260623_180431 | 32.39075858328078 | 0.974622116320663 | 27.99395392260824 |

补充：

```text
如果按单个序列最高 PSNR 排序，最佳单条记录是:
    DNA-Rendering / 0813_05 / part_moe_arm / 20260624_174257
    PSNR        36.204474385579424
    SSIM        0.9873301754395167
    LPIPS*1000  17.771333324102066

但单序列结果不适合作为整体实验最佳结论。
整体比较应看完整六序列平均，因此 part_moe_pair/20260702_004310
是当前 part_moe 相关日志里的最佳完整 DNA 结果。
```

## 2026-07-26 ZJU / I3D part_moe 平均指标最佳查询

用户问题：

```text
在 zju 数据集和 i3d 数据集上看序列平均值的话，最好的分别是哪次 part_moe 实验？
```

查询口径：

```text
解析 logs/part 下所有 part_moe* 总日志。
每个模型路径取最后一次对应 split 的 evaluation 结果。
按同一 dataset / exp / run / split 的序列平均 PSNR 排序。
SSIM 和 LPIPS*1000 作为参考。
```

ZJU-MoCap：

```text
split: test
最佳实验:
    part_moe_leg
    RUN_TIME: 20260622_185140
    日志: logs/part/20260622_185140_ZJU-MoCap_part_moe_leg.log

Average over 6 sequences:
    PSNR        31.177024783422770
    SSIM        0.962445808135400
    LPIPS*1000  27.965317850610262
```

ZJU 前几名：

| Rank | Experiment | RUN_TIME | N | PSNR | SSIM | LPIPS*1000 |
|---:|---|---|---:|---:|---:|---:|
| 1 | part_moe_leg | 20260622_185140 | 6 | 31.177024783422770 | 0.962445808135400 | 27.965317850610262 |
| 2 | part_moe_arm | 20260626_145103 | 6 | 31.177020746985722 | 0.962446550360918 | 27.923035453375451 |
| 3 | part_moe_arm | 20260626_155504 | 6 | 31.171169836057487 | 0.962490478109493 | 27.848476823872581 |
| 4 | part_moe_leg | 20260626_164558 | 6 | 31.164491815006667 | 0.962489011329371 | 27.981840474027269 |

I3D-Human：

```text
I3D 日志同时有 novelview 和 novelpose。
如果按常用 novelview 序列平均 PSNR:
    最佳实验: part_moe_leg
    RUN_TIME: 20260622_145118
    日志: logs/part/20260622_145118_I3D-Human_part_moe_leg_4000分层.log

    Average over 6 sequences:
        PSNR        32.508371999251715
        SSIM        0.966997556634204
        LPIPS*1000  30.091111575937912

如果按 novelpose 序列平均 PSNR:
    最佳实验: part_moe_foot
    RUN_TIME: 20260623_113602
    日志: logs/part/20260623_113602_I3D-Human_part_moe_foot.log

    Average over 4 sequences:
        PSNR        30.458750609623916
        SSIM        0.958941118641874
        LPIPS*1000  34.590949769831944
```

I3D novelview 前几名：

| Rank | Experiment | RUN_TIME | N | PSNR | SSIM | LPIPS*1000 |
|---:|---|---|---:|---:|---:|---:|
| 1 | part_moe_leg | 20260622_145118 | 6 | 32.508371999251715 | 0.966997556634204 | 30.091111575937912 |
| 2 | part_moe_foot | 20260623_113602 | 4 | 32.391854174779013 | 0.967448833249617 | 28.372095181586779 |
| 3 | part_moe_pair | 20260702_004310 | 4 | 32.391090261741454 | 0.967414043202220 | 28.457868593944063 |
| 4 | part_moe | 20260617_002911 | 4 | 32.387198894649032 | 0.967479348432340 | 28.414017634324598 |

结论：

```text
按平均 PSNR:
    ZJU-MoCap 最好: part_moe_leg / 20260622_185140
    I3D-Human novelview 最好: part_moe_leg / 20260622_145118
    I3D-Human novelpose 最好: part_moe_foot / 20260623_113602
```

## 2026-07-26 DNA part_moe_leg 完整结果统计

用户问题：

```text
dna 上的 part_moe_leg 实验有几个完整结果，平均值分别多少。
```

查询口径：

```text
只统计 DNA-Rendering / part_moe_leg。
使用 output/DNA-Rendering/*/part_moe_leg/*/logs/render_*_part_moe_leg.log
中的最终 render.py novelview 指标。
完整结果定义为六个 DNA 序列都存在:
    0007_04, 0019_10, 0044_11, 0051_09, 0206_04, 0813_05
```

结论：

```text
DNA 上 part_moe_leg 只有 1 个完整六序列结果:
    RUN_TIME: 20260623_180431

Average:
    PSNR        32.390758583280778
    SSIM        0.974622116320663
    LPIPS*1000  27.993953922608245
```

完整结果逐序列：

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0007_04 | 29.567117643356323 | 0.958891220390797 | 43.902407307177782 |
| 0019_10 | 35.356522528330487 | 0.981278095146020 | 20.681170498331390 |
| 0044_11 | 33.000570901234944 | 0.978210971752802 | 21.077770584573347 |
| 0051_09 | 28.698904466629028 | 0.971685592333476 | 30.827233513506751 |
| 0206_04 | 31.518965848286946 | 0.970327004790306 | 33.563221339136362 |
| 0813_05 | 36.202470111846921 | 0.987339813510577 | 17.911920292923845 |
| Average | 32.390758583280778 | 0.974622116320663 | 27.993953922608245 |

未完整的 part_moe_leg run：

| RUN_TIME | N | Covered Sequences | Avg PSNR | Avg SSIM | Avg LPIPS*1000 |
|---|---:|---|---:|---:|---:|
| 20260622_145118 | 2 | 0007_04,0019_10 | 32.460019898414615 | 0.970102373758952 | 32.212788487474128 |
| 20260622_180520 | 2 | 0007_04,0019_10 | 32.559897081057230 | 0.970317085335652 | 32.068665492503598 |
| 20260622_225328 | 5 | 0007_04,0019_10,0051_09,0206_04,0813_05 | 32.172671368916824 | 0.973597814242045 | 29.665096487539511 |
| 20260630_170907 | 1 | 0206_04 | 31.541592105229697 | 0.970519575973352 | 32.756943736846246 |
| 20260720_194556 | 1 | 0044_11 | 33.019845469792685 | 0.978283845384916 | 21.065409497047462 |

## 2026-07-26 tri 调参命名规则统一

用户要求：

```text
triC_d32_r64_e10 这种命名太复杂。
调参过程不要取新的消融实验名称。
把参数写进脚本，在运行脚本的命令行里改参数即可。
```

执行结果：

```text
三个脚本保持固定消融入口:
    bash scripts/exps_dnarendering.sh tri
    bash scripts/exps_i3dhuman.sh tri
    bash scripts/exps_zjumocap.sh tri

实验名固定为:
    tri

输出路径固定为:
    output/<dataset>/<sequence>/tri/<RUN_TIME>/

总日志固定写入:
    logs/tri/<RUN_TIME>_<dataset>_tri.log

不再新建 triA / triB / triC / triC_d32_r64_e10 这类消融名称。
```

脚本默认 tri 参数：

```text
TRI_PLANE_DIM    默认 32
TRI_PLANE_RES    默认 64
TRI_PLANE_EXTENT 默认 1.0

其中 TRI_PLANE_EXTENT 从原始默认 1.2 改为 1.0，
对应之前 triC_d32_r64_e10 中表现最好的核心设置。
```

以后调参命令示例：

```bash
TRI_PLANE_DIM=32 TRI_PLANE_RES=64 TRI_PLANE_EXTENT=1.0 \
GPU_id=3 bash scripts/exps_dnarendering.sh tri

TRI_PLANE_DIM=64 TRI_PLANE_RES=96 TRI_PLANE_EXTENT=1.0 \
GPU_id=0 SEQUENCES_OVERRIDE="0044_11 0206_04" \
bash scripts/exps_dnarendering.sh tri
```

验证：

```text
bash -n scripts/exps_dnarendering.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_zjumocap.sh
git diff --check -- scripts/exps_dnarendering.sh scripts/exps_i3dhuman.sh scripts/exps_zjumocap.sh note/motion.md
```

## 2026-07-26 tri 作为第二模块创新点的总结

用户问题：

```text
总结现在的 tri 实验（tri_plane_extent: 1.2 -> 1.0 之后的）这个点
作为第二个模块的创新点是什么，怎么起作用的。
```

当前定位：

```text
第一个模块:
    part_moe_leg
    作用是按 SMPL part label 给不同人体部件分配专属 non-rigid expert，
    提升部件级形变建模能力。

第二个模块:
    tri
    作用是在 part_moe_leg 的基础上，为 canonical Gaussian 坐标提供可学习的
    tri-plane spatial feature，让每个部件 expert 不只依赖 Fourier 位置编码、
    pose / seq pose / seq xyz 条件，还能读取一个持久的、空间结构化的
    canonical geometry/motion memory。
```

核心公式：

```text
canonical Gaussian:
    x_i

tri-plane sampling:
    f_tri_i = TriPlaneFeature(x_i)

non-rigid expert input:
    input_i = concat(
        x_emb_i,
        pose_feat,
        seq_pose_feat,
        seq_xyz_feat,
        f_tri_i
    )

然后仍然走 part_moe_leg 的分部件专家:
    d_i = Expert_part(input_i)
```

实现机制：

```text
nets/mlp_delta_non_rigid.py
    TriPlaneFeature 维护三个 learnable planes:
        planes: [3, tri_plane_dim, tri_plane_res, tri_plane_res]

    对 canonical query_xyz 做归一化:
        xyz_norm = clamp(query_xyz / tri_plane_extent, -1, 1)

    分别采样:
        XY plane
        XZ plane
        YZ plane

    三个方向采样特征平均得到:
        f_tri_i: [B, N, tri_plane_dim]

    use_tri=True 时:
        MLP 第一层输入维度加宽 tri_plane_dim。
        forward 中把 f_tri_i concat 到 x_emb / pose / seq_pose / seq_xyz 后面。

gaussian_renderer/__init__.py
    调 non_rigid_deformer 时传入 query_xyz=means3D。
    means3D 是进入非刚性分支前的 canonical Gaussian 坐标。

scene/gaussian_model.py
    tri 只能和 use_part_moe 一起用。
    并强制:
        part_label_schema = part_moe_leg
        num_parts = 7
```

为什么不是小修小改：

```text
tri 不是加残差或改 loss。
它给 non-rigid deformation 增加了一套新的 learnable spatial field：
    canonical xyz -> tri-plane lookup -> part expert input

这相当于把原本“点坐标位置编码 + 时序/姿态条件”的 MLP，
扩展成“部件专家 + 显式可学习空间特征场”的网络结构。

它补的是 part_moe_leg 的短板:
    part_moe_leg 只知道每个 Gaussian 属于哪个部件，
    但同一部件内部哪里是稳定主体、哪里是边界、哪里更容易形变，
    主要还要靠 MLP 从 x_emb 中隐式拟合。

tri-plane 给同一部件内部提供更细粒度的空间索引，使 expert 可以按 canonical
位置学习不同区域的形变偏置。
```

tri_plane_extent 从 1.2 改到 1.0 的含义：

```text
归一化公式:
    xyz_norm = query_xyz / tri_plane_extent

extent 越大:
    人体 canonical 坐标被压得更靠近 tri-plane 中心，
    有效采样范围更小，空间分辨率更容易被稀释。

extent 从 1.2 收紧到 1.0:
    同样的 canonical Gaussian 分布在 tri-plane 上占据更大的有效区域，
    每个 plane cell 对人体区域的表达更细，
    f_tri_i 的空间区分度更强。

直观理解:
    不是增大网络容量，而是让已有 64x64 tri-plane 更集中服务人体附近区域，
    减少空白空间浪费。
```

作为论文/创新点的表达：

```text
Part-aware Tri-plane Conditioned Non-rigid Deformation

在 SMPL part-aware MoE deformation 的基础上，引入 canonical-space tri-plane
feature field。每个 Gaussian 根据其 canonical 坐标从三平面中查询可学习
空间特征，并将该特征注入对应部件 expert 的 non-rigid MLP 输入。
这样模型同时具备:
    1. part-level specialization
    2. point/region-level spatial memory
    3. pose/time conditioned deformation

该模块用于增强人体动态区域和部件内部复杂区域的形变表达，
尤其适合腿部、衣物边界、关节附近、手脚边界等同一 part 内部形变差异明显的位置。
```

当前默认脚本设置：

```text
TRI_PLANE_DIM    = 32
TRI_PLANE_RES    = 64
TRI_PLANE_EXTENT = 1.0
```

## 2026-07-26 tri 中“标准空间点确定，为什么还能采样”的解释

用户问题：

```text
标准空间的点不是确定的吗，为啥还能继续采样？
```

解释：

```text
这里的“采样”不是重新采样 Gaussian 点，也不是随机采样新点。
它指的是在确定的 canonical Gaussian 坐标 x_i 上，对一个可学习的 tri-plane
feature field 做 lookup / interpolation。

也就是:
    x_i 是查询位置，确定。
    tri-plane 是被查询的可学习特征场，不确定，会训练更新。
```

类比：

```text
固定 mesh 顶点有固定 UV 坐标，但仍然可以从 texture 上采样颜色。
UV 不变，不代表采样没意义，因为 texture 本身是可学习/可优化的。

tri 也是类似:
    canonical xyz 固定地决定查哪里；
    三平面参数决定查出来是什么 feature。
```

具体过程：

```text
给定 canonical Gaussian:
    x_i = (x, y, z)

归一化:
    xyz_norm = clamp(x_i / tri_plane_extent, -1, 1)

投影到三个 plane:
    XY: (x, y)
    XZ: (x, z)
    YZ: (y, z)

用 grid_sample 做 bilinear interpolation:
    f_xy = sample(plane_xy, x, y)
    f_xz = sample(plane_xz, x, z)
    f_yz = sample(plane_yz, y, z)

输出:
    f_tri_i = mean(f_xy, f_xz, f_yz)
```

为什么它还能学习：

```text
即使 x_i 在某一次 forward 中是确定的，loss 的梯度仍然会传到:
    rendered loss
    -> d_xyz / d_rotation / d_scaling
    -> part expert MLP
    -> f_tri_i
    -> grid_sample
    -> tri-plane 参数

所以训练过程中，x_i 查的位置可以固定，但这个位置上的 feature 会变。
```

补充：

```text
SeqAvatar / Gaussian Splatting 中 canonical Gaussian 坐标本身也不是绝对永久不变。
训练过程中 Gaussian center 会被优化，并且 densify/prune 会改变点集。
但 tri 的有效性不依赖点必须移动；即使固定点集，它仍然可以作为可学习空间表。
```

和 x_emb 的区别：

```text
x_emb 是 deterministic Fourier positional encoding:
    给定 x_i，x_emb_i 永远由固定函数算出来。

f_tri_i 是 learnable spatial feature:
    给定 x_i，查的位置固定，但查出来的内容由可训练 plane 参数决定。

因此 tri-plane 相当于给 part_moe_leg 增加一张 canonical-space memory table，
让同一 part 内的不同区域可以有不同的可学习形变偏置。
```

## 2026-07-26 tri-plane lookup 是否是常见做法

用户问题：

```text
利用三平面不是重新采样 Gaussian 点，而是在确定坐标处查询一个可学习特征场。
这种做法是三平面的常见做法吗？
```

结论：

```text
是常见做法。
三平面表示的标准用法就是:
    给定一个确定的 3D query coordinate
    -> 投影到三个 feature plane
    -> 用 bilinear interpolation 查询 feature
    -> 聚合 feature
    -> 送入小 MLP / decoder

也就是说，三平面里的 sample 通常指 feature lookup/interpolation，
不是重新生成点。
```

参考依据：

```text
EG3D:
    使用三张正交 feature planes + small implicit decoder。
    标准流程是查询 3D position，投影到三张 plane，
    bilinear interpolation 得到 feature，再聚合后解码。

Instant-NGP / hash grid:
    虽然不是 tri-plane，但属于同一类 coordinate -> trainable feature
    lookup encoding。
    给定输入坐标，查 trainable feature vectors，再插值并送入网络。
```

和当前 SeqAvatar tri 的关系：

```text
通用 tri-plane:
    query coordinate -> f_tri -> decoder -> color / density / SDF 等

当前 SeqAvatar tri:
    canonical Gaussian coordinate -> f_tri -> part expert MLP
        -> d_xyz / d_rotation / d_scaling

因此:
    “坐标处查询可学习特征场”是常规 tri-plane 操作。
    “把查到的 f_tri 注入 part_moe_leg 的非刚性形变 expert”是本实验的迁移和创新点。
```

需要避免的表述：

```text
不要说 tri 对 Gaussian 点重新采样。
更准确说法:
    在 canonical Gaussian 坐标处查询 tri-plane feature。
    或:
    sample / interpolate tri-plane feature at each canonical Gaussian position。
```

## 2026-07-26 当前 tri 与 part_moe 的适配程度澄清

用户判断：

```text
现在的做法并没有为 part_moe 做特别的适配，
只是学习了更多的空间特征，从而起到了优化各部件学习到的内容的作用，对吗？
```

结论：

```text
基本正确，但需要更精确地表述。

当前 tri 是 part_moe_leg 上的空间特征增强:
    它依赖 part_moe_leg 的分部件 expert 路由；
    但 tri-plane feature 本身是全局 canonical-space feature field，
    不是每个 part 一套独立 tri-plane，
    也没有 part-conditioned sampling / part gate / part-specific feature fusion。
```

当前代码里的 part_moe 适配点：

```text
1. tri 被强制运行在 part_moe_leg 上:
    --use_tri 必须和 --use_part_moe 同时使用。
    part_label_schema 必须是 part_moe_leg。
    num_parts 必须是 7。

2. f_tri_i 会 concat 到 features 中。
   这些 features 随后被 part_moe_leg 的 global expert 和 routed part experts 使用。

3. 因为每个 expert 都看到 f_tri_i，所以 tri-plane 可以帮助不同部件 expert
   学到更细的 spatial bias。
```

当前没有做的 part-specific 设计：

```text
1. 没有每个 part 独立一套 tri-plane:
    f_tri_i = TriPlane_p(x_i)

2. 没有用 part label 调制 tri feature:
    f_tri_i = FiLM(f_tri_i, part_id)

3. 没有为不同 part 设置不同的 tri feature dim / resolution / extent。

4. 没有 part-aware gate 控制某些 part 更依赖 tri，某些 part 更依赖原 MLP。
```

因此推荐表述：

```text
当前 tri 模块不是专门为每个 part 定制 tri-plane；
它是在 part_moe_leg 的 expert 输入中加入 canonical-space learnable spatial feature。
这个空间特征场为所有部件共享，但会被不同 part expert 分别解释和利用，
从而增强各部件内部区域的非刚性形变表达。
```

如果后续想更“part-aware”，可以升级为：

```text
Part-conditioned tri-plane:
    f_tri_i = TriPlane(x_i)
    e_part_i = Embedding(part_id)
    f_part_tri_i = MLP_fuse(f_tri_i, e_part_i)

或者 Part-specific tri-plane:
    f_tri_i = TriPlane_{part_i}(x_i)

但当前版本保持简单，主要贡献是:
    part_moe_leg + shared canonical tri-plane spatial memory。
```

## 2026-07-26 tri 效果有限后的增强方向

用户问题：

```text
现在这个 tri 相比只加第一个点的优化很有限。
已经调了参数，还想增加效果的话可以怎么改？
```

当前瓶颈判断：

```text
当前 tri 的结构是:
    shared TriPlaneFeature(x_i) -> concat 到 features -> part_moe_leg experts

它确实只增强了空间特征容量。
tri-plane 本身没有按 part 分开，也没有 part-conditioned gate / FiLM / adapter。
因此继续只调 tri_plane_dim / tri_plane_res / tri_plane_extent，收益会很快变小。
下一步应该强化 tri 和 part_moe_leg 路由之间的耦合。
```

优先建议 1：Part-conditioned tri feature fusion，低风险，最推荐先做

```text
给每个 part 一个 embedding:
    e_p = Embedding(part_label_i)

用 part embedding 调制共享 tri feature:
    gamma_i, beta_i = MLP_part_fuse(e_p)
    f_part_tri_i = gamma_i * f_tri_i + beta_i

expert 输入改为:
    input_i = concat(x_emb_i, pose_feat, seq_pose_feat, seq_xyz_feat, f_part_tri_i)
```

优点：

```text
1. 不需要每个 part 单独一套大 tri-plane，参数量可控。
2. tri-plane 仍共享空间记忆，但不同 part 对同一空间 feature 的解释不同。
3. 比单纯 concat 更明确地服务 part_moe_leg。
```

优先建议 2：tri contribution gate，让模型学会哪里该用 tri

```text
g_i = sigmoid(MLP_gate(x_emb_i, f_tri_i, e_part_i))
f_used_i = g_i * f_part_tri_i

或者对输出 residual 做 gate:
    d_i = d_base_i + g_i * d_tri_adapter_i
```

目标：

```text
让动态边界、膝盖、脚踝、衣物边缘等区域更依赖 tri；
让稳定躯干/主体区域少依赖 tri，避免噪声。
```

优先建议 3：Part-specific delta tri-plane，参数更大但更强

```text
共享主 tri-plane:
    f_shared = TriPlane_shared(x_i)

每个 part 一个小的 delta plane:
    f_delta = TriPlane_delta_part[p](x_i)

融合:
    f_part_tri = f_shared + lambda * f_delta
```

建议：

```text
不要一开始就每个 part 都用完整 32x64x64。
可先用小 delta:
    delta_dim = 8 或 16
    delta_res = 32 或 64

这样更像:
    全局 canonical spatial memory + part-specific correction。
```

优先建议 4：把 tri 从“只进第一层 concat”改成 hidden-layer FiLM / adapter

当前：

```text
features = concat(..., f_tri)
h = MLP(features)
```

更强版本：

```text
h = MLP_layer_1(base_features)
gamma, beta = TriAdapter(f_tri, e_part)
h = gamma * h + beta
h = MLP_layer_2(h)
```

原因：

```text
只在第一层 concat，网络可能弱化或忽略 f_tri。
FiLM / adapter 会让 tri feature 直接调制 expert hidden state，
使用强度更高。
```

优先建议 5：Motion-aware tri gate，而不是静态空间表

```text
motion_feat_i 可来自 seq_xyz_conds 或 part/point motion magnitude。

g_i = sigmoid(MLP(x_emb_i, f_tri_i, motion_feat_i, e_part_i))
f_used_i = g_i * f_tri_i
```

目标：

```text
让 tri 主要服务高运动/高非刚性区域。
否则静态 tri-plane 只会学几何空间 bias，对动态形变帮助有限。
```

不建议继续优先做的事情：

```text
1. 继续单纯增大 tri_plane_dim / tri_plane_res。
   这只是加容量，未解决和 part_moe 的弱耦合。

2. 直接给每个 part 完整大 tri-plane。
   参数多，容易过拟合，也容易让训练不稳定。

3. 再做类似 pamo rigid residual。
   用户已决定放弃 pamo residual 方向。
```

推荐下一步实验顺序：

```text
A. part-conditioned tri feature fusion:
    shared tri + part embedding FiLM

B. 在 A 基础上加 point-wise tri gate:
    g_i 控制 f_tri 使用强度

C. 如果 A/B 有提升，再试 part-specific delta tri-plane:
    shared tri + small per-part delta

D. 最后再考虑 motion-aware gate:
    让 tri 更集中服务动态区域
```

## 2026-07-26 tri 增强方案中文简化解释

用户要求：

```text
看不懂，用中文简单分析这几个改进方法。
```

总判断：

```text
现在 tri 的问题不是参数不够，而是它和 part_moe_leg 的关系太弱。
它只是给所有点多加了一份共享空间特征。
想继续提升，就要让这个空间特征更明确地按部件、按区域、按运动强弱发挥作用。
```

方案 1：给 tri 加“部件身份”

```text
现在:
    所有部件都查同一张 tri-plane，查出来的特征直接给 expert。

改法:
    给每个部件一个身份编码，比如左腿、右腿、躯干、手、脸。
    tri 特征先结合这个部件身份，再给对应 expert。

直观理解:
    同一个空间特征，左腿 expert 和躯干 expert 不应该完全一样地理解。
    加部件身份后，tri 会更贴合 part_moe_leg。

优点:
    改动小，参数少，风险低。

建议:
    最先做。
```

方案 2：加一个“是否使用 tri”的开关

```text
现在:
    每个 Gaussian 都固定拿 tri 特征。

改法:
    网络给每个 Gaussian 学一个 0 到 1 的权重。
    权重大，就多用 tri。
    权重小，就少用 tri。

直观理解:
    膝盖、脚踝、衣服边界可能需要 tri。
    躯干主体比较稳定，可能不需要太强 tri。

优点:
    防止 tri 在稳定区域引入噪声。

风险:
    如果 gate 学成全 0，tri 又会没作用。
```

方案 3：给每个部件一个“小修正三平面”

```text
现在:
    只有一套全局共享 tri-plane。

改法:
    保留共享 tri-plane。
    再给每个部件一个很小的 delta tri-plane，只学这个部件自己的补充信息。

直观理解:
    全局 tri-plane 负责共性。
    左腿/右腿/手/脸各自的小 tri-plane 负责部件特殊性。

优点:
    比方案 1 更强，更像真正的 part-aware tri。

风险:
    参数更多，可能过拟合。
    不建议一上来做很大的每部件三平面。
```

方案 4：不要只把 tri 放进第一层，改成中间层调制

```text
现在:
    tri 只是拼到 MLP 输入最前面。
    网络可能学着学着就把它忽略了。

改法:
    在 MLP 中间层用 tri 去调制 hidden feature。

直观理解:
    不是只在开头告诉网络一次 tri 信息，
    而是在网络计算过程中持续提醒它“这里的空间特征是什么”。

优点:
    tri 使用强度更高。

风险:
    改网络结构更多，训练稳定性要检查。
```

方案 5：让 tri 更关注运动强的区域

```text
现在:
    tri 是静态空间特征，只知道 canonical 空间位置。

改法:
    加入运动强弱信息。
    运动越强、非刚性越明显的位置，tri 权重越高。

直观理解:
    腿快速摆动、膝盖、脚踝、衣物边缘更需要额外表达。
    稳定区域不需要太多额外自由度。

优点:
    和“动态区域需要更强形变能力”这个新思路最一致。

风险:
    需要定义可靠的运动强弱特征，改动比方案 1/2 大。
```

推荐顺序：

```text
第一步:
    方案 1，给 tri 加部件身份。

第二步:
    方案 2，加每点 tri 使用权重。

第三步:
    如果前两步有效，再做方案 3，小的部件专属 delta tri-plane。

不建议现在继续只调:
    tri_plane_dim
    tri_plane_res
    tri_plane_extent

因为这只是扩大容量，不能解决 tri 和 part_moe_leg 结合不够紧的问题。
```

## 2026-07-26 tri_part：part-aware tri-plane feature fusion

用户要求：

```text
新增独立消融实验 tri_part。
按照“给 tri 加部件身份”的方案实现。
用 GPU 0 / 1 / 3 跑完 DNA-Rendering 六个序列并汇总评价指标。
```

实现口径：

```text
tri_part 是独立实验名，但网络基础仍然是 part_moe_leg。

脚本模式:
    bash scripts/exps_dnarendering.sh tri_part

自动启用:
    --use_part_moe
    --use_tri
    --use_tri_part
    --part_label_schema part_moe_leg
    --num_parts 7

日志目录:
    /media/image/mxz/human/SeqAvatar/logs/tri

默认 original / part_moe_leg / tri 不启用 use_tri_part。
```

网络修改：

```text
nets/mlp_delta_non_rigid.py
    新增 PartTriFeatureFiLM。

流程:
    canonical Gaussian 坐标 query_xyz
        -> shared TriPlaneFeature 采样 f_tri_i
        -> 按 part_label 取 part embedding
        -> FiLM 生成 gamma_i / beta_i
        -> f_part_tri_i = f_tri_i * (1 + gamma_i) + beta_i
        -> concat 到 non-rigid / part expert 输入

FiLM 最后一层 weight / bias 零初始化，所以训练开始时:
    gamma_i = 0
    beta_i = 0
    f_part_tri_i = f_tri_i

因此 tri_part 初始等价于普通 tri，不会一开始破坏已有 part_moe_leg + tri 行为。
```

脚本修改：

```text
scripts/exps_dnarendering.sh
scripts/exps_i3dhuman.sh
scripts/exps_zjumocap.sh

均新增 tri_part 模式。
DNA 默认 tri 参数沿用当前 tri 最优口径:
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
```

已验证：

```text
bash -n scripts/exps_dnarendering.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_zjumocap.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    arguments/__init__.py
    train.py
    render.py
    gaussian_renderer/__init__.py

CPU 最小前向测试:
    NonrigidDeformer(use_part_moe=True, use_tri=True, use_tri_part=True, num_parts=7)

结果:
    initial_identity_max = 0.0
    shared 输出:
        d_xyz      [1, 9, 3]
        d_rotation [1, 9, 4]
        d_scaling  [1, 9, 3]
    init_part_moe_from_shared(num_parts=7) 成功。
    part_enabled=True 输出维度保持一致。
```

本次 DNA 六序列运行计划：

```text
GPU0:
    0044_11
    0051_09

GPU1:
    0206_04
    0813_05

GPU3:
    0007_04
    0019_10

使用独立 RUN_TIME 后缀避免并发总日志互相交叉。
```

运行结果：

```text
首次启动:
    GPU0: RUN_TIME=20260726_215814_gpu0, sequences=0044_11 0051_09
    GPU1: RUN_TIME=20260726_215814_gpu1, sequences=0206_04 0813_05
    GPU3: RUN_TIME=20260726_215814_gpu3, sequences=0007_04 0019_10

GPU1/GPU3 完整完成:
    0206_04
    0813_05
    0007_04
    0019_10

GPU0 首次 0044_11 训练到 25000 后，在 train.py 最终 training_report
novel-view evaluation 中 OOM:
    torch.cuda.OutOfMemoryError in knn_cuda
    Tried to allocate 2.46 GiB

原因:
    训练内最终评估会同时保留训练显存状态并加载 test cameras / KNN。
    0044_11 点数较多，导致最终 evaluation OOM。
    这不是 tri_part 网络逻辑错误。

重跑 GPU0:
    RUN_TIME=20260726_224700_gpu0_retry
    sequences=0044_11 0051_09
    SKIP_LOAD_TEST_CAMERAS=1

处理口径:
    训练阶段跳过 train.py 内部 test camera evaluation，避免 OOM。
    最终 novel-view 指标仍使用 render.py 输出，不使用训练内中间指标。
```

最终 render.py novelview 指标：

| Sequence | RUN_TIME | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| 0044_11 | 20260726_224700_gpu0_retry | 33.01163388888041 | 0.9781492993235588 | 21.11751437963297 |
| 0051_09 | 20260726_224700_gpu0_retry | 28.716113980611166 | 0.9718945637345314 | 30.539619713090357 |
| 0206_04 | 20260726_215814_gpu1 | 31.522967306772866 | 0.9706513146559397 | 33.06893456416825 |
| 0813_05 | 20260726_215814_gpu1 | 36.2184441725413 | 0.987352258960406 | 17.88608469845106 |
| 0007_04 | 20260726_215814_gpu3 | 29.580058288574218 | 0.9589614530404409 | 43.73900143740077 |
| 0019_10 | 20260726_215814_gpu3 | 35.33606557846069 | 0.9813554614782333 | 20.5824658429871 |
| Average | - | 32.39754720264011 | 0.9747273918655184 | 27.822270105955084 |

对照当前已有结果：

```text
part_moe_leg / 20260623_180431:
    PSNR        32.390758583280778
    SSIM        0.974622116320663
    LPIPS*1000  27.993953922608245

tri / 20260726_triC_d32_r64_e10:
    PSNR        32.420168439547222
    SSIM        0.9746725806759464
    LPIPS*1000  27.96582942052434

tri_part - part_moe_leg:
    ΔPSNR        +0.006788619359333836
    ΔSSIM        +0.0001052755448553988
    ΔLPIPS*1000  -0.1716838166531609

tri_part - tri:
    ΔPSNR        -0.02262123690710638
    ΔSSIM        +0.000054811189571957186
    ΔLPIPS*1000  -0.14355931456925575
```

结论：

```text
tri_part 相比 part_moe_leg 略有提升，尤其 LPIPS 更好。
但相比当前普通 tri 最优设置，PSNR 下降 0.0226，SSIM / LPIPS 略好。

因此“part-conditioned tri FiLM”有一定作用，但不是明确压过普通 tri 的强提升版本。
它更像是在视觉感知指标上略改善，而不是显著提升重建峰值质量。
```

验证状态：

```text
git diff --check 通过。
六个 render.py novelview 指标均已生成。
tmux tri_part 训练/渲染进程均已退出。
GPU0/GPU3 已空闲；GPU1 当前还有非本次 tri_part tmux 的其他进程占用。
```

## 2026-07-27 I3D / ZJU 增加 tri 消融并运行

用户要求：

```text
按照 DNA 数据集上一样的修改规则，给 I3D-Human 和 ZJU-MoCap 增加 tri 消融实验。
保证三个数据集的 tri 创新一致且公平:
    tri_plane_extent = 1.0

用 GPU 0 / 1 / 3 在 I3D-Human 和 ZJU-MoCap 上跑完 tri，并汇总评价指标。
之后长实验使用 tmux 启动。
```

实现检查：

```text
scripts/exps_dnarendering.sh
scripts/exps_i3dhuman.sh
scripts/exps_zjumocap.sh

三个脚本的 tri 模式均一致:
    experiment_name=tri
    --use_part_moe
    --use_tri
    --part_label_schema part_moe_leg
    --num_parts 7

tri 参数默认一致:
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0

tri 仍然只在 part_moe_leg 基础上启用。
original / part_moe_leg 不采样 tri-plane，不加宽输入。
```

验证：

```text
bash -n scripts/exps_dnarendering.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_zjumocap.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    arguments/__init__.py
    train.py
    render.py
    gaussian_renderer/__init__.py

检查时 GPU 0 / 1 / 3 空闲。
```

运行计划：

```text
RUN 前缀:
    20260727_*

GPU0 tmux:
    ZJU:
        CoreView_377
        CoreView_386
    I3D:
        ID1_1
        ID1_2

GPU1 tmux:
    ZJU:
        CoreView_387
        CoreView_392
    I3D:
        ID2_1

GPU3 tmux:
    ZJU:
        CoreView_393
        CoreView_394
    I3D:
        ID3_1

每张卡串行运行该卡的 ZJU 分片和 I3D 分片，避免同卡多个训练进程竞争显存。
日志保存到:
    /media/image/mxz/human/SeqAvatar/logs/tri
```

运行结果：

```text
tmux:
    tri_i3d_zju_gpu0_122549
    tri_i3d_zju_gpu1_122549
    tri_i3d_zju_gpu3_122549

状态:
    三个 tmux 队列均已结束。
    ZJU-MoCap 六个序列均生成 iteration 3000 final metrics。
    I3D-Human 四个序列均生成 iteration 15000 final metrics。

指标来源:
    ZJU:
        output/ZJU-MoCap/<seq>/tri/<run>/metrics/results_test_3000.json
    I3D novelview:
        output/I3D-Human/<seq>/tri/<run>/metrics/results_novelview_15000.json
    I3D novelpose:
        output/I3D-Human/<seq>/tri/<run>/metrics/results_novelpose_15000.json
```

ZJU-MoCap test：

| Sequence | RUN_TIME | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| CoreView_377 | 20260727_122549_zju_gpu0 | 31.493320743433028 | 0.9732929329837908 | 18.117606290059083 |
| CoreView_386 | 20260727_122549_zju_gpu0 | 33.93984052388355 | 0.9693497542781059 | 24.947525858126504 |
| CoreView_387 | 20260727_122549_zju_gpu1 | 28.839587052663166 | 0.955779008341558 | 32.05587284996955 |
| CoreView_392 | 20260727_122549_zju_gpu1 | 32.11006375362999 | 0.9646155544730465 | 28.267431743532136 |
| CoreView_393 | 20260727_122549_zju_gpu3 | 29.472734139970513 | 0.9544327485659891 | 33.92314534126358 |
| CoreView_394 | 20260727_122549_zju_gpu3 | 31.160574717955157 | 0.9568838406015526 | 30.16416470795362 |
| Average | - | 31.169353488589234 | 0.9623923065406738 | 27.912624465150746 |

I3D-Human novelview：

| Sequence | RUN_TIME | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| ID1_1 | 20260727_122549_i3d_gpu0 | 32.11013536453247 | 0.9672574993222952 | 25.42038941755891 |
| ID1_2 | 20260727_122549_i3d_gpu0 | 32.135791446316624 | 0.9668524749817387 | 26.97141002262792 |
| ID2_1 | 20260727_122549_i3d_gpu1 | 31.634514747521816 | 0.9699061543513567 | 28.536427157142988 |
| ID3_1 | 20260727_122549_i3d_gpu3 | 33.82125525474548 | 0.9663376413285732 | 32.545816618949175 |
| Average | - | 32.42542420327909 | 0.967588442495991 | 28.36851080406975 |

I3D-Human novelpose：

| Sequence | RUN_TIME | PSNR | SSIM | LPIPS*1000 |
|---|---|---:|---:|---:|
| ID1_1 | 20260727_122549_i3d_gpu0 | 30.06187211672465 | 0.9597787355383237 | 31.6969720646739 |
| ID1_2 | 20260727_122549_i3d_gpu0 | 30.473948860168456 | 0.9596934636433919 | 31.01011705584824 |
| ID2_1 | 20260727_122549_i3d_gpu1 | 28.23384127700538 | 0.955693671839279 | 39.935521773275056 |
| ID3_1 | 20260727_122549_i3d_gpu3 | 32.66334866217847 | 0.9599625850623509 | 36.752575094688616 |
| Average | - | 30.35825272901924 | 0.9587821140208365 | 34.848796497121455 |

收尾检查：

```text
git diff --check 通过。
GPU 0 / 1 / 2 / 3 当前无本次实验残留训练进程。
I3D / ZJU 的 tri 配置与 DNA 保持一致:
    --use_part_moe
    --use_tri
    --part_label_schema part_moe_leg
    --num_parts 7
    tri_plane_dim=32
    tri_plane_res=64
    tri_plane_extent=1.0
```

## 2026-07-27 I3D / ZJU tri 对比单独 part_moe_leg

用户要求：

```text
比较这两个数据集上 tri 和单独 part_moe_leg 的结果。
```

对比口径：

```text
tri:
    ZJU:
        output/ZJU-MoCap/<seq>/tri/20260727_122549_zju_gpu*/metrics/results_test_3000.json
    I3D:
        output/I3D-Human/<seq>/tri/20260727_122549_i3d_gpu*/metrics/results_novelview_15000.json
        output/I3D-Human/<seq>/tri/20260727_122549_i3d_gpu*/metrics/results_novelpose_15000.json

part_moe_leg:
    ZJU 主对比使用历史最佳完整六序列:
        output/ZJU-MoCap/<seq>/part_moe_leg/20260622_185140/metrics/results_test_3000.json
    ZJU 同时检查另一套完整六序列:
        output/ZJU-MoCap/<seq>/part_moe_leg/20260626_164558/metrics/results_test_3000.json
    I3D 使用完整 part_moe_leg:
        output/I3D-Human/<seq>/part_moe_leg/20260622_145118/metrics/results_*_15000.json

注意:
    本次 tri 的 I3D 只跑了当前脚本中的 4 个序列:
        ID1_1, ID1_2, ID2_1, ID3_1
    因此主对比使用同 4 序列均值，不直接拿历史 6 序列均值作主结论。
```

ZJU-MoCap test，tri 对比 part_moe_leg / 20260622_185140：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| part_moe_leg | 31.17366016509712 | 0.9623772457124292 | 28.012560860967792 |
| tri | 31.169353488589234 | 0.9623923065406738 | 27.912624465150746 |
| tri - part_moe_leg | -0.00430667650788763 | +0.000015060828244539692 | -0.09993639581704628 |

ZJU-MoCap test，tri 对比 part_moe_leg / 20260626_164558：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| part_moe_leg | 31.161066850422543 | 0.962418893114703 | 28.03229686383637 |
| tri | 31.169353488589234 | 0.9623923065406738 | 27.912624465150746 |
| tri - part_moe_leg | +0.008286638166691063 | -0.000026586574029208165 | -0.11967239868562274 |

I3D-Human novelview，同 4 序列：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| part_moe_leg | 32.39407307725736 | 0.9674439822073264 | 28.4920528156119 |
| tri | 32.42542420327909 | 0.967588442495991 | 28.36851080406975 |
| tri - part_moe_leg | +0.03135112602173251 | +0.00014446028866454963 | -0.12354201154214906 |

I3D-Human novelpose，同 4 序列：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| part_moe_leg | 30.45449356313416 | 0.9589966609199754 | 34.63218343549833 |
| tri | 30.35825272901924 | 0.9587821140208365 | 34.848796497121455 |
| tri - part_moe_leg | -0.09624083411491924 | -0.00021454689913891212 | +0.21661306162312854 |

结论：

```text
ZJU:
    tri 和单独 part_moe_leg 基本持平。
    相对历史最佳 part_moe_leg，PSNR 略低 0.0043，SSIM 略高 0.000015，LPIPS*1000 好 0.0999。
    相对另一套 part_moe_leg，PSNR 略高 0.0083，SSIM 略低 0.000027，LPIPS*1000 好 0.1197。
    因此 ZJU 上 tri 没有形成明显 PSNR 增益，但感知指标 LPIPS 稳定略好。

I3D:
    novelview 上 tri 有小幅稳定收益:
        PSNR +0.0314
        SSIM +0.000144
        LPIPS*1000 -0.1235
    novelpose 上 tri 明显回退:
        PSNR -0.0962
        SSIM -0.000215
        LPIPS*1000 +0.2166

总判断:
    tri 作为 part_moe_leg 上的 canonical tri-plane spatial feature，
    在 I3D novelview 和 ZJU LPIPS 上有轻微正向，
    但对 I3D novelpose 泛化不利。
    它目前更像增强训练姿态附近/同分布视角的空间记忆，
    不是明确提升跨姿态泛化的第二创新点。
```

## 2026-07-27 tri_part 含义澄清

用户问题：

```text
现在 tri_part 的优化是不是给不同 part 分别学习对应部分的三平面特征，
但是效果下降了？
```

回答口径：

```text
不完全是。

当前 tri_part 不是“每个 part 单独学习一套三平面”。
它仍然使用 shared TriPlaneFeature:
    query_xyz -> shared tri-plane -> f_tri_i

然后根据 Gaussian 的 part_label 取 part embedding，
用 PartTriFeatureFiLM 对 f_tri_i 做部件条件调制:
    f_part_tri_i = f_tri_i * (1 + gamma_part_i) + beta_part_i

所以它的实际含义是:
    共享一个 canonical tri-plane 空间特征场，
    不同 part 用不同 FiLM 方式解释 / 调制这份 tri feature。

它比普通 tri 多的是 part-aware feature fusion，
不是 part-specific tri-plane bank。
```

效果判断：

```text
相对 part_moe_leg:
    tri_part 略好:
        PSNR +0.006788619359333836
        SSIM +0.0001052755448553988
        LPIPS*1000 -0.1716838166531609

相对当前普通 tri 最优:
    tri_part 的 PSNR 下降:
        PSNR -0.02262123690710638
    但 SSIM / LPIPS 略好:
        SSIM +0.000054811189571957186
        LPIPS*1000 -0.14355931456925575

因此如果以普通 tri 的 PSNR 为主指标，tri_part 确实下降了。
如果以 part_moe_leg 为基线，它没有下降，而是很小幅提升。
总体看 tri_part 没有证明比普通 tri 更强，只能说 part-aware FiLM 对感知指标有轻微帮助。
```

## 2026-07-27 tri 后续优化思路

用户问题：

```text
tri 还有什么优化的思路吗？
```

当前判断：

```text
普通 tri:
    在 part_moe_leg 的 expert 输入第一层 concat canonical tri-plane feature。
    作用更像给 canonical 空间加一张 learnable memory table。

tri_part:
    不是每个 part 一套独立三平面。
    是 shared tri-plane + part-conditioned FiLM 调制 tri feature。

已有结果说明:
    1. tri 相对 part_moe_leg 在 DNA / I3D novelview 有轻微提升。
    2. ZJU 基本持平，LPIPS 略好。
    3. I3D novelpose 回退，说明 tri 容易增强训练姿态附近的空间记忆，
       但不一定改善跨姿态泛化。
    4. tri_part 只在感知指标上略好，PSNR 不如普通 tri。

因此不建议继续只调:
    tri_plane_dim
    tri_plane_res
    tri_plane_extent
```

后续更有价值的结构方向：

```text
方案 1：学习 tri usage gate，优先级最高
    f_tri_raw = TriPlaneFeature(x_i)
    g_i = sigmoid(MLP_gate(x_emb_i, f_tri_raw, part_id, motion_strength_i))
    f_tri_eff = g_i * f_tri_raw

    再 concat:
        input_i = concat(x_emb_i, pose_feat, seq_pose_feat, seq_xyz_feat, f_tri_eff)

    目的:
        让静态/刚性区域少用 tri，避免过拟合 canonical 记忆；
        让边界、衣物、快速运动区域更多用 tri。

    初始化建议:
        gate bias 设成负值，让初始 g_i 很小，例如 0.05 或 0.1。
        这样训练一开始接近 part_moe_leg，不会破坏稳定基线。

方案 2：part-local tri-plane，而不是 global canonical tri-plane
    对每个 Gaussian 用 part 中心和 part 尺度归一化:
        x_local_i = (x_i - c_part) / s_part
    在 part-local 坐标采样 tri-plane:
        f_local_tri = TriPlaneFeatureLocal(x_local_i)

    目的:
        让三平面表达“部件内部位置”，例如大腿上/下、膝盖附近、脚踝附近，
        而不是只记住全局 canonical 坐标。
        这更贴合 part_moe_leg 的部件专家。

方案 3：shared tri + small part-specific delta tri
    不直接给每个 part 一套完整三平面，参数太多也容易过拟合。
    用低强度 residual:
        f_i = f_shared(x_i) + lambda * f_delta_part(x_i)

    其中 f_delta_part 可以低维、低分辨率、或低秩分解。
    lambda 初始化为 0 或很小。

    目的:
        比 tri_part 的 FiLM 更强，因为 part 确实有自己的局部空间记忆；
        但又比完全独立 part tri-plane 更稳。

方案 4：把 tri 从 input concat 改成 hidden adapter / FiLM
    当前 tri 只进第一层，网络可能忽略它，也可能把它当噪声。
    可以改成:
        h_l = h_l + beta * Adapter_l(f_tri_i)
    或:
        h_l = h_l * (1 + gamma_l(f_tri_i)) + beta_l(f_tri_i)

    identity 初始化:
        Adapter / gamma / beta 初始为 0。

    目的:
        让 tri 直接调制 part expert 的中间特征，比只 concat 到输入更强。

方案 5：给 tri 加 smooth / sparse 正则，保护 novelpose
    regularize:
        TV loss on tri planes
        L2 loss on tri feature norm
        gate sparsity loss

    目的:
        限制 tri 变成过强的 per-sequence memorization。
        这对 I3D novelpose 回退尤其关键。
```

推荐实验顺序：

```text
第一优先级:
    tri_gate
    原因:
        当前最大问题不是 tri 没容量，而是 tri 使用不受控。
        gate 能直接验证“哪里该用 tri，哪里不该用 tri”这个核心假设。

第二优先级:
    part-local tri
    原因:
        它比 global tri 更贴合 part_moe_leg，
        同时不会像完全独立 part tri-plane 那样大幅增加参数。

第三优先级:
    shared tri + small part-specific delta tri
    原因:
        这是更强版本，但需要正则和小初始化，否则可能继续伤 novelpose。

不建议优先做:
    更大 dim / 更高 resolution / 更复杂 tri_part FiLM。
    这些更可能增加记忆能力，而不是解决泛化问题。
```

## 2026-07-27 tri_gate 如何决定形变能力分配

用户问题：

```text
tri_gate 怎么根据运动强弱、部件属性和局部刚性程度，
决定哪里需要更强形变能力？
```

核心解释：

```text
tri_gate 不是硬编码“某个 part 一定强形变”。
它是给每个 Gaussian 学一个 soft gate:
    g_i in [0, 1]

当前 tri 是:
    f_tri_i = TriPlaneFeature(x_i)
    input_i = concat(x_emb_i, pose_feat, seq_pose_feat, seq_xyz_feat, f_tri_i)

tri_gate 后变为:
    f_tri_raw_i = TriPlaneFeature(x_i)
    g_i = sigmoid(MLP_gate(gate_input_i))
    f_tri_eff_i = g_i * f_tri_raw_i
    input_i = concat(x_emb_i, pose_feat, seq_pose_feat, seq_xyz_feat, f_tri_eff_i)

因此:
    g_i 高:
        这个 Gaussian 可以更多使用 tri-plane spatial memory，
        等价于给它更强的局部空间形变表达能力。
    g_i 低:
        tri feature 被压小，
        它更接近原来的 part_moe_leg，
        表达更稳定，不容易记忆训练姿态。
```

gate_input 建议包含：

```text
1. 运动强弱 motion_strength
    来自 SMPL part 的 pose velocity / acceleration，或 seq_xyz 预测的局部位移幅度。
    例如:
        m_part = norm(part_velocity) + k * norm(part_acc)

    作用:
        强运动 part 更可能需要额外形变能力。

2. 部件属性 part attribute
    使用 part embedding，或者固定属性 one-hot:
        torso/head: 更稳定
        upper/lower leg: 大运动
        arm/hand/foot: 大运动 + 边界复杂

    作用:
        同样运动强度下，不同 part 的形变需求不同。

3. 局部刚性 local rigidity
    可以是手工几何先验，也可以是可学习标量:
        r_i 高:
            Gaussian 在部件主体、远离关节/边界，更刚性
        r_i 低:
            Gaussian 在衣物边界、关节附近、脚踝/膝盖附近，更非刚性

    作用:
        高刚性点少用 tri，低刚性点多用 tri。

4. 点自身位置 / tri 原始特征
    x_emb_i 和 f_tri_raw_i。
    作用:
        让 gate 细化到同一 part 内部的不同位置。
```

一个更明确的形式：

```text
gate_input_i = concat(
    x_emb_i,
    f_tri_raw_i,
    part_embedding[label_i],
    motion_strength[label_i],
    local_rigidity_i
)

g_i = sigmoid(MLP_gate(gate_input_i))
f_tri_eff_i = g_i * f_tri_raw_i
```

直觉上的决策规则：

```text
强运动 + 低刚性 + 关节/边界附近:
    g_i 应该变高。
    例如膝盖附近、脚踝附近、衣服边界、快速摆动的腿部边缘。

弱运动 + 高刚性 + part 主体区域:
    g_i 应该变低。
    例如躯干主体、头部、腿部中间比较稳定的实体区域。

强运动但高刚性:
    g_i 不一定很高。
    例如大腿主体整体跟随 LBS 已经较好，未必需要强 tri memory。

弱运动但低刚性/边界复杂:
    g_i 可以中等。
    例如衣服轮廓处，即使 part 运动不大，也可能需要更细的局部修正。
```

训练和初始化建议：

```text
gate 初始 bias 设为负值:
    g_i 初始约 0.05 或 0.1

这样训练一开始接近 part_moe_leg，避免 tri 一开始过强。

可以打印:
    g_mean / g_std / g_min / g_max
    per_part g_mean
    high-motion vs low-motion g_mean
    rigid-core vs boundary g_mean

合理现象:
    high-motion / boundary 区域 gate 更高；
    torso/head/core 区域 gate 更低；
    gate 不是长期全 0，也不是全 1。
```

重要限制：

```text
如果只做 f_tri_eff = g_i * f_tri_i，
tri_gate 决定的是“是否使用额外 tri feature”，不是动态改变 MLP 层数或参数量。
它属于 soft capacity allocation。

如果要更强地体现“更强形变能力 / 更稳定约束”，可以进一步让 gate 同时控制:
    1. tri feature 强度
    2. 非刚性 delta 的幅度
    3. smooth / regularization loss 权重

例如:
    high gate:
        更强 tri feature，较弱 smooth 约束
    low gate:
        较弱 tri feature，较强稳定约束
```

## 2026-07-27 tri_gate 的 g_i 调节方式

用户问题：

```text
怎么调节 g_i？
```

建议不要只用裸公式：

```text
g_i = sigmoid(MLP_gate(...))
```

更稳的形式：

```text
g_raw_i = sigmoid((logit_i + b_part + b_prior_i) / tau)
g_i = alpha_tri(iter) * g_cap * g_raw_i
f_tri_eff_i = g_i * f_tri_raw_i
```

各项含义：

```text
1. gate bias 控制初始打开程度
    如果希望初始 g_i 约为 p:
        bias = log(p / (1 - p))

    常用:
        p=0.05 -> bias=-2.944
        p=0.10 -> bias=-2.197
        p=0.20 -> bias=-1.386
        p=0.50 -> bias=0

    推荐先用 p=0.05 或 0.10。
    这样初始接近 part_moe_leg。

2. alpha_tri(iter) 控制什么时候启用 tri_gate
    建议和 part_moe_start_iter 对齐，甚至稍晚一点:
        before start: alpha_tri = 0
        warmup:       alpha_tri 从 0 线性涨到 1

    目的:
        先让 part_moe_leg 学稳，再让 tri_gate 学哪里需要额外空间特征。

3. g_cap 控制最大强度
    早期可以设:
        g_cap = 0.3 或 0.5
    如果不够再放到:
        g_cap = 1.0

    目的:
        防止 tri 一开始压过 part_moe_leg。

4. tau 控制 sigmoid 尖锐程度
    g_raw = sigmoid(logit / tau)

    tau 大:
        gate 更平滑，不容易饱和。
    tau 小:
        gate 更接近 0/1，选择更硬。

    推荐:
        tau=2.0 起步，后面可退火到 1.0。

5. b_part 控制不同部件先验
    可以给每个 part 一个 learnable bias。
    初始化上:
        torso/head/core 偏低
        arm/leg/foot/boundary-prone part 略高

    但不要写死最终结果，只作为初始化先验。

6. b_prior_i 注入运动/刚性先验
    例如:
        b_prior_i = a * motion_strength_part - c * local_rigidity_i

    强运动会提高 gate；
    高刚性会降低 gate。

7. 正则控制 gate 不要失控
    sparse loss:
        L_gate_sparse = lambda_g * mean(g_i)

    target loss:
        L_gate_target = lambda_t * (mean(g_i) - target)^2

    建议 target 先试:
        0.10 ~ 0.25

    目的:
        避免所有点都开 gate，导致 tri 退化成强 canonical memory。
```

推荐第一版默认参数：

```text
gate_init = 0.10
gate_bias = -2.197
gate_cap = 0.5
gate_tau = 2.0
gate_warmup_start = part_moe_start_iter
gate_warmup_iters = 2000 for DNA/I3D, 500 for ZJU
gate_target = 0.15
gate_sparse_lambda = small, e.g. 1e-4 or 5e-5
```

判断调节是否合理：

```text
每隔固定 iter 打印:
    g_mean
    g_std
    g_min
    g_max
    per_part g_mean
    high-motion part g_mean
    low-motion part g_mean

合理现象:
    g_mean 不长期等于初始值 0.1
    g_std 不能接近 0
    high-motion / boundary part 的 g_mean 更高
    torso/head/core 的 g_mean 更低

异常:
    g_mean 接近 0:
        tri_gate 没打开，增大 gate_cap 或降低 sparse lambda。
    g_mean 接近 g_cap 且 g_std 很低:
        所有点都在用 tri，增加 sparse loss 或降低 gate_cap。
    novelview 提升但 novelpose 下降:
        gate 太像 canonical memory，降低 gate_cap / 增强正则 / 加 part-local 约束。
```

## 2026-07-27 当前 tri 代码实现总结

用户问题：

```text
结合代码详细总结现在 tri 是怎么做的。
```

脚本入口：

```text
scripts/exps_dnarendering.sh
scripts/exps_i3dhuman.sh
scripts/exps_zjumocap.sh

tri 模式都会设置:
    experiment_name=tri
    part_moe_enabled=1
    tri_enabled=1
    part_label_schema=part_moe_leg
    num_parts=7

因此当前 tri 不是 original 上加三平面，
而是强制建立在 part_moe_leg 上。

默认参数:
    tri_plane_dim=32
    tri_plane_res=64
    tri_plane_extent=1.0

其中 arguments/__init__.py 里的默认 extent 仍是 1.2，
但三个实验脚本运行 tri 时都会传入 1.0。
```

开关隔离：

```text
scene/gaussian_model.py:
    use_tri=True 时会检查:
        必须 use_part_moe=True
        part_label_schema 必须是 part_moe_leg
        num_parts 必须是 7

否则直接 raise ValueError。

所以 original / part_moe_leg 默认不会创建 tri-plane。
只有脚本传入 --use_tri 时才启用。
```

TriPlaneFeature 结构：

```text
nets/mlp_delta_non_rigid.py:
    TriPlaneFeature 内部参数:
        planes: [3, tri_plane_dim, tri_plane_res, tri_plane_res]

当前默认:
        [3, 32, 64, 64]

三张 plane 分别对应:
    xy plane
    xz plane
    yz plane

forward(query_xyz):
    1. query_xyz 如果是 [N, 3]，先扩成 [1, N, 3]
    2. coords = clamp(query_xyz / tri_plane_extent, -1, 1)
    3. 用 grid_sample 分别采样:
        xy_feat = plane_xy(coords[..., x,y])
        xz_feat = plane_xz(coords[..., x,z])
        yz_feat = plane_yz(coords[..., y,z])
    4. 输出:
        f_tri = (xy_feat + xz_feat + yz_feat) / 3

输出 shape:
    [B, N, tri_plane_dim]
```

采样坐标：

```text
gaussian_renderer/__init__.py:
    调 non_rigid_deformer 时传入:
        query_xyz=means3D

这里的 means3D 是进入非刚性分支前的 canonical Gaussian 坐标。

因此 tri 不是重新采样 Gaussian 点，
而是在已有 canonical Gaussian 坐标处查询一个可学习三平面特征场。
```

接入 NonrigidDeformer：

```text
nets/mlp_delta_non_rigid.py:
    use_tri=True 时:
        self.TriPlaneFeature = TriPlaneFeature(...)

    然后把 shared non-rigid MLP 第一层输入维度加宽:
        old_input_dim -> old_input_dim + tri_plane_dim

    _append_mlp_input_dim 会复制原第一层权重到前 old_input_dim 列。
    新增 tri 列保持 nn.Linear 默认初始化。

    但 TriPlaneFeature.planes 初始为全 0，
    因此初始 f_tri=0，第一次前向基本等价于不加 tri 的输入。
```

forward 数据流：

```text
feats 初始包含:
    x_emb

如果启用已有条件，还会 concat:
    pose_feats
    seq_pose_feats
    seq_xyz_feats

use_tri=True 后额外:
    tri_features = sample_tri_features(query_xyz)
    feats.append(tri_features)

最终:
    features = concat(x_emb, pose_feat, seq_pose_feat, seq_xyz_feat, f_tri)
```

和 Part-MoE 的关系：

```text
当前 tri 从训练一开始就进入 shared non-rigid MLP。
它不是等 part_moe_start_iter 之后才生效。

part_moe_start_iter 到达时:
    train.py 调 gaussians.init_part_moe_from_shared()

这会:
    1. 按当前 shared MLP 和输出头复制出 num_parts 个 PartNonrigidExpert
    2. 冻结原 shared MLP / gaussian_warp / gaussian_rotation / gaussian_scaling
    3. 把每个 expert 加入 optimizer

注意:
    expert 复制的是 MLP 和输出头。
    TriPlaneFeature 本身不复制到每个 expert。
    所有 expert 共享同一个 TriPlaneFeature 查询结果。
```

Part-MoE 路由：

```text
forward_tri 当前只是:
    return forward_part_moe(...)

也就是说 tri 没有单独路由逻辑。
它只是先把 f_tri concat 到 features，
然后继续走 part_moe_leg 的 expert 路由。

forward_part_moe:
    expert_0 是 global/unknown expert
    expert_1..num_parts-1 是 routed part experts

输出按 part_moe_alpha / part_moe_global_keep 混合:
    global_weight = 1 - part_weight
    part_weight = clamp(part_moe_alpha, 0, 1 - global_keep)

label==pid 的 Gaussian 使用:
    global_weight * global_output + part_weight * expert_pid_output
```

训练时 alpha：

```text
train.py:
    part_moe_alpha 在 part_moe_start_iter 前是 0
    之后按 part_moe_warmup 线性增加到 1 - part_moe_global_keep

脚本默认:
    DNA:
        part_moe_start_iter=10000
        part_moe_warmup=1000
        part_moe_global_keep=0.1
    I3D:
        part_moe_start_iter=4000
        part_moe_warmup=1000
        part_moe_global_keep=0.1
    ZJU:
        part_moe_start_iter=1000
        part_moe_warmup=500
        part_moe_global_keep=0.1
```

渲染 / evaluation：

```text
render.py:
    use_part_moe=True 时把 part_moe_alpha 设为:
        1 - part_moe_global_keep

    并从:
        model_path/part_labels/iteration_<part_moe_start_iter>/gaussian_part_label.npy
    加载 part label。

因此最终 render 使用完整 Part-MoE 混合状态。
```

tri_part 补充：

```text
tri_part = shared TriPlaneFeature + PartTriFeatureFiLM

流程:
    f_tri = shared TriPlaneFeature(query_xyz)
    part_emb = embedding(part_label)
    gamma, beta = Linear(part_emb)
    f_part_tri = f_tri * (1 + gamma) + beta

FiLM 最后一层零初始化:
    初始 gamma=0, beta=0
    所以 tri_part 初始等价普通 tri。

注意:
    tri_part 不是每个 part 单独一套 tri-plane。
```

当前 tri 的实质：

```text
它是在 part_moe_leg 的非刚性 MLP 输入中加入一个 canonical-space learnable spatial feature:

    x_i canonical coordinate
        -> query shared tri-plane
        -> f_tri_i
        -> concat 到 non-rigid features
        -> part_moe_leg expert routing
        -> d_xyz, d_rotation, d_scaling

它提供的是空间记忆 / 空间位置条件增强，
不是新的骨骼运动、不是重新采样 Gaussian，也不是每个 part 独立三平面。
```

当前局限：

```text
1. f_tri 只 concat 到第一层输入，没有 gate，也没有 hidden-layer adapter。
2. tri 从训练一开始就生效，不是等 part_moe_leg 稳定后再打开。
3. 所有 part 共享同一个 TriPlaneFeature。
4. 当前 tri 不显式使用运动强弱、局部刚性、边界属性。
5. 因此它容易变成 canonical spatial memory:
    novelview / LPIPS 可能略好，
    但 I3D novelpose 上会有泛化回退。
```

## 2026-07-27 tri_part 是否修改 shared tri-plane

用户问题：

```text
所有 part expert 共享同一个三平面特征。那 tri_part 是修改了这一点吗？
```

结论：

```text
没有完全修改这一点。

当前 tri_part 仍然共享同一个 TriPlaneFeature:
    self.TriPlaneFeature = TriPlaneFeature(...)

forward 里仍然先统一采样:
    tri_features = self.sample_tri_features(query_xyz, ...)

sample_tri_features 内部直接调用:
    self.TriPlaneFeature(query_xyz)

所以底层三平面参数 planes 仍然只有一套，
不是每个 part expert 一套，也不是每个 part 一个独立 tri-plane bank。
```

tri_part 改的是什么：

```text
tri_part 在 shared tri_features 后面增加了 PartTriFeatureFiLM:
    part_emb = part_embedding(part_label)
    gamma, beta = Linear(part_emb)
    f_part_tri = tri_features * (1 + gamma) + beta

也就是说:
    普通 tri:
        所有 part 直接使用同一份 f_tri

    tri_part:
        所有 part 先共享同一份 f_tri，
        再根据 part_label 对 f_tri 做不同的 gamma/beta 调制。

它改变的是“不同 part 如何解释 / 调制共享 tri feature”，
不是“不同 part 分别学习自己的 tri-plane”。
```

如果要真正修改 shared tri-plane：

```text
需要做新的结构，例如:
    1. 每个 part 一套独立 TriPlaneFeature
    2. shared tri + part-specific delta tri
    3. part-local coordinate tri-plane

其中最稳的是:
    shared tri + small part-specific delta tri

形式:
    f_i = f_shared(x_i) + lambda * f_delta_part(x_i)

lambda 初始设 0 或很小，避免一开始破坏普通 tri / part_moe_leg。
```

## 2026-07-27 新增 tri_gate 消融实验

用户要求：

```text
做消融实验 tri_gate。
给 tri 加 gate / adapter，避免它变成强 canonical 记忆。

不要把 f_tri 直接 concat 到第一层输入。
更稳形式:
    features = features_base + alpha * gate_i * Adapter(f_tri_i)

要求:
    Adapter 最后一层 zero init
    alpha 从 0 warm-up 到 0.2 / 0.4
    tri-plane 作为受控补充，而不是直接改写 non-rigid MLP 输入分布

改完跑 DNA-Rendering 六个序列并汇总评价指标。
```

实现口径：

```text
新增独立开关:
    --use_tri_gate
    --tri_gate_alpha，默认 0.2
    --tri_gate_init，默认 0.5
    --tri_gate_hidden_dim，默认 128

脚本新增模式:
    bash scripts/exps_dnarendering.sh tri_gate
    bash scripts/exps_i3dhuman.sh tri_gate
    bash scripts/exps_zjumocap.sh tri_gate

tri_gate 模式自动启用:
    --use_part_moe
    --use_tri
    --use_tri_gate
    --part_label_schema part_moe_leg
    --num_parts 7

日志仍保存到:
    /media/image/mxz/human/SeqAvatar/logs/tri

默认 original / part_moe_leg / tri / tri_part 不启用 use_tri_gate。
```

网络修改：

```text
nets/mlp_delta_non_rigid.py
    新增 TriGateAdapter:
        adapter:
            f_tri -> hidden -> base_feature_dim
        gate:
            concat(features_base, f_tri) -> hidden -> scalar gate_i

    adapter 最后一层 weight / bias 零初始化。
    gate 最后一层 weight 零初始化，bias 根据 tri_gate_init 初始化。

    use_tri_gate=True 时:
        仍创建 shared TriPlaneFeature。
        不调用 _append_mlp_input_dim。
        因此 shared MLP 第一层输入维度保持 part_moe_leg 原维度。

    forward:
        features_base = concat(x_emb, pose_feat, seq_pose_feat, seq_xyz_feat)
        f_tri = TriPlaneFeature(query_xyz)
        tri_res = gate_i * Adapter(f_tri)
        features = features_base + alpha * tri_res

    alpha:
        根据 part_moe_alpha / (1 - part_moe_global_keep) 归一化 warmup。
        最大值为 tri_gate_alpha，当前默认 0.2。
        part_moe_start_iter 前 alpha=0。
```

和普通 tri / tri_part 的区别：

```text
普通 tri:
    features = concat(features_base, f_tri)
    MLP 第一层输入维度加宽。

tri_part:
    features = concat(features_base, FiLM_part(f_tri))
    MLP 第一层输入维度加宽。

tri_gate:
    features = features_base + alpha * gate_i * Adapter(f_tri)
    MLP 第一层输入维度不变。
    tri 只能作为 residual 补充进入 base feature 空间。
```

已验证：

```text
bash -n:
    scripts/exps_dnarendering.sh
    scripts/exps_i3dhuman.sh
    scripts/exps_zjumocap.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    arguments/__init__.py
    train.py
    render.py
    gaussian_renderer/__init__.py

CPU 最小前向:
    plain part_moe_leg vs tri_gate

结果:
    shared first linear in dim plain:    6
    shared first linear in dim tri_gate: 6
    pre_part_alpha0 max diff: 0.0
    pre_part_alpha1 max diff: 0.0
    tri_gate_adapter_last_abs: 0.0, 0.0
    part_enabled max diff: 0.0
    output shapes:
        d_xyz      [1, 8, 3]
        d_rotation [1, 8, 4]
        d_scaling  [1, 8, 3]

git diff --check 通过。
```

DNA 六序列运行计划：

```text
GPU0:
    0044_11
    0051_09

GPU1:
    0206_04
    0813_05

GPU3:
    0007_04
    0019_10

默认参数:
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    TRI_GATE_ALPHA=0.2
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
```

启动状态：

```text
使用 tmux 启动。
为避免 DNA train.py 内部最终 test-camera evaluation OOM，统一设置:
    SKIP_LOAD_TEST_CAMERAS=1

最终评价仍由脚本后续 render.py 生成 results_novelview_25000.json。

RUN_TIME:
    GPU0: 20260727_171828_tri_gate_gpu0
    GPU1: 20260727_171828_tri_gate_gpu1
    GPU3: 20260727_171828_tri_gate_gpu3

tmux:
    tri_gate_dna_gpu0_20260727_171828
        0044_11
        0051_09
    tri_gate_dna_gpu1_20260727_171828
        0206_04
        0813_05
    tri_gate_dna_gpu3_20260727_171828
        0007_04
        0019_10
```

运行结果：

```text
DNA-Rendering tri_gate 六序列已跑完。

参数:
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    TRI_GATE_ALPHA=0.2
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128

指标来自 render.py novelview evaluation:
    [ITER 25000] Evaluating novelview #120

0007_04:
    PSNR       29.596143039067584
    SSIM       0.9586856300632158
    LPIPS*1000 44.537590257823466000

0019_10:
    PSNR       35.393041642506915
    SSIM       0.9814122041066488
    LPIPS*1000 20.737392936522762000

0044_11:
    PSNR       32.999435583750405
    SSIM       0.9781217371424039
    LPIPS*1000 21.279939799569547000

0051_09:
    PSNR       28.69997552235921
    SSIM       0.9717189763983091
    LPIPS*1000 30.450001576294503000

0206_04:
    PSNR       31.47780532836914
    SSIM       0.9701835026343664
    LPIPS*1000 33.83658217887084000

0813_05:
    PSNR       36.187635850906375
    SSIM       0.987221019466718
    LPIPS*1000 18.041820048044124000

六序列平均:
    PSNR       32.3923394944932715
    SSIM       0.9745571783019436666666666666666666666667
    LPIPS*1000 28.147221132854207000
```

对比已有 DNA 平均结果：

```text
part_moe_leg / 20260623_180431:
    PSNR       32.390758583280778
    SSIM       0.974622116320663
    LPIPS*1000 27.993953922608245

tri / 20260726_triC_d32_r64_e10:
    PSNR       32.420168439547222
    SSIM       0.9746725806759464
    LPIPS*1000 27.96582942052434

tri_part:
    PSNR       32.39754720264011
    SSIM       0.9747273918655184
    LPIPS*1000 27.822270105955084

tri_gate - part_moe_leg:
    PSNR       +0.0015809112124935
    SSIM       -0.0000649380187193333333333333333333333333
    LPIPS*1000 +0.153267210245962000

tri_gate - tri:
    PSNR       -0.0278289450539505
    SSIM       -0.0001154023740027333333333333333333333333
    LPIPS*1000 +0.181391712329867000

tri_gate - tri_part:
    PSNR       -0.0052077081468385
    SSIM       -0.0001702135635747333333333333333333333333
    LPIPS*1000 +0.324951026899123000
```

结论：

```text
tri_gate 可以正常训练和评估，开关/日志/渲染链路接通。
但当前默认设置 alpha=0.2、gate_init=0.5 下，DNA 平均结果没有超过已有 tri / tri_part。
相对 part_moe_leg 只有 PSNR 极小提升，SSIM 和 LPIPS*1000 变差。

因此 tri_gate 当前版本更像是稳定控制 tri-plane 影响的结构验证，
还不能作为比 tri / tri_part 更强的最终方案。
```

验证：

```text
git diff --check 通过。
tmux 训练会话已全部结束。
```

## 2026-07-27 tri_gate 未超过 tri 的原因分析

用户问题：

```text
为什么 tri_gate 没有超过 tri？
```

结论：

```text
当前 tri_gate 没超过 tri，主要不是因为实现没接通，而是因为这个版本把 tri-plane
的作用压得太弱、太晚，并且 residual 加法位置不够理想。

tri 的方式:
    features = concat(features_base, f_tri)
    MLP 第一层输入维度加宽。
    tri-plane 从训练一开始就直接参与 shared MLP 学习。
    Part-MoE 在 10000 iter 激活时，expert 是从已经学过 tri 的 shared MLP 拷贝来的。

tri_gate 的方式:
    tri_res = gate * Adapter(f_tri)
    features = features_base + alpha * tri_res
    MLP 第一层输入维度不变。
    alpha 依赖 part_moe_alpha，part_moe_start_iter 前基本为 0。
    默认 alpha 最大只有 0.2。
    Adapter 最后一层 zero init，所以初始 tri_res=0。

因此 tri_gate 的实际效果是:
    1. 前 10000 iter 几乎不使用 tri-plane。
    2. 10000 iter 后才开始把 tri-plane 作为很小 residual 加进去。
    3. zero-init adapter 让学习更稳，但也让早期梯度路径更弱。
    4. ordinary tri 已经通过直接 concat 获得了更强、更早的空间特征表达。
```

代码依据：

```text
nets/mlp_delta_non_rigid.py
    line 226-228:
        普通 tri 会 _append_mlp_input_dim，把 tri_plane_dim 直接拼进 MLP 输入。

    line 362-370:
        tri_gate 的 alpha = tri_gate_alpha * normalized(part_moe_alpha)
        默认 tri_gate_alpha=0.2。

    line 397-410:
        tri_gate 走 features = apply_tri_gate_adapter(...)
        普通 tri 走 features = torch.cat([features, tri_features], dim=-1)

scripts/exps_dnarendering.sh
    line 116-123:
        part_moe_start_iter=10000
        part_moe_warmup=1000
        TRI_GATE_ALPHA 默认 0.2
```

指标现象：

```text
tri_gate - tri:
    平均 PSNR       -0.0278289450539505
    平均 SSIM       -0.0001154023740027333333333333333333333333
    平均 LPIPS*1000 +0.181391712329867000

逐序列看:
    0007_04:
        PSNR 略升，但 SSIM / LPIPS 变差。
    0206_04:
        PSNR 略升，但 SSIM / LPIPS 变差。
    0813_05:
        LPIPS 略好，但 PSNR / SSIM 变差。
    0019_10 / 0044_11 / 0051_09:
        整体不如 tri。

说明 tri_gate 不是完全没用，但它的平均收益被弱化，且没有形成稳定提升。
```

进一步判断：

```text
tri_gate 的目标是避免 tri-plane 强记忆，所以刻意做了:
    zero init
    warmup
    small alpha
    residual adapter
    gate 控制

这些都增强了稳定性，但也削弱了容量。
当前 DNA 上 ordinary tri 的直接 concat 并没有出现明显过拟合到劣化的情况，
所以更强表达的 tri 反而更好。
```

如果继续优化 tri_gate，更合理方向：

```text
1. 不要让 tri_gate 等到 part_moe_start_iter 后才生效。
   让 alpha 从训练早期 warmup 到一个小值，例如 3000-6000 iter 到 0.2。

2. 把 residual 加法位置从原始 input feature 改到 MLP hidden feature。
   当前 features_base 是 x_emb / pose_feat / seq_feat 的拼接，不是一个干净的可加隐空间。
   更合理:
       h0 = first_layer(features_base)
       h0 = h0 + alpha * gate * Adapter(f_tri)

3. 提高 alpha 上限做小网格:
       0.2 / 0.4 / 0.6
   当前 0.2 可能太保守。

4. 给 adapter 加很小非零初始化或只 zero-init 输出 bias。
   否则初期 tri-plane 和 adapter 前层梯度进入较慢。

5. gate 不要一开始就完全自由学习，可以按 part 或运动强度给先验。
   例如腿脚/手部更大 gate，躯干/脸部更小 gate。
```

## 2026-07-27 tri_gate 是否能靠调参超过 tri

用户问题：

```text
是不是可以通过调参让 tri_gate 更稳直到超过 tri？
```

判断：

```text
有机会，但不能理解成“继续让 tri_gate 更稳就能超过 tri”。

当前 tri_gate 的问题不是不稳，而是过于保守：
    alpha 最大只有 0.2
    alpha 还绑定 part_moe_alpha，10000 iter 之前基本不生效
    Adapter 最后一层 zero init
    tri-plane 只作为 additive residual，而不是直接进入 MLP

所以继续降低 gate / 降低 alpha / 更强 zero-init，只会更稳但更没用。
如果目标是超过 tri，调参方向应该是“受控地增强 tri_gate 的使用强度”，不是继续压低它。
```

优先尝试：

```text
不改代码的参数实验:
    TRI_GATE_ALPHA=0.4, TRI_GATE_INIT=0.5, TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_ALPHA=0.6, TRI_GATE_INIT=0.5, TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_ALPHA=0.4, TRI_GATE_INIT=0.7, TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_ALPHA=0.4, TRI_GATE_INIT=0.5, TRI_GATE_HIDDEN_DIM=256

预期:
    alpha=0.4 可能是最值得先试的。
    alpha=0.6 如果 PSNR 上升但 LPIPS/SSIM 变差，说明 tri 贡献过强。
    gate_init=0.7 可以让 gate 初始更开放，但也可能更像普通 tri。
    hidden=256 增加 adapter 容量，可能提升但也更容易记忆。
```

更可能有效的代码级优化：

```text
1. tri_gate 不要绑定 part_moe_alpha。
   新增独立:
       tri_gate_start_iter
       tri_gate_warmup
   例如:
       start=3000
       warmup=3000
       alpha_max=0.2 或 0.4

   这样 shared MLP 在 Part-MoE 激活前已经见过 tri_gate 特征。

2. 把 adapter residual 加到 hidden feature，而不是 input feature。
   当前加法发生在:
       features_base = concat(x_emb, pose_feat, seq_feat)
   这个空间不是统一语义隐空间。

   更合理:
       h = first_layer(features_base)
       h = h + alpha * gate * Adapter(f_tri)

3. gate 做 part / motion prior，而不是完全自由 gate。
   对运动强的部件给更高上限，对稳定部件给更低上限。
```

推荐实验策略：

```text
第一阶段只用 0044_11 和 0206_04 快速筛参数:
    先跑 alpha=0.4
    再跑 alpha=0.6
    如果 alpha=0.4 两个序列都优于当前 tri_gate，再扩到六序列。

判断标准:
    如果 PSNR 上升但 LPIPS*1000 明显变差，不算成功。
    至少要平均 PSNR 接近或超过 tri，同时 SSIM / LPIPS 不退。

当前 tri_gate 距 tri 很近:
    PSNR 只差 0.0278289450539505
但 LPIPS*1000 差 0.181391712329867，
所以单纯提高 alpha 可能补 PSNR，但要小心 LPIPS 继续变差。
```

## 2026-07-28 tri_gate 调参和独立 warmup

用户要求：

```text
不改变和基线实验对比公平性的基础上调整 tri_gate。
可以调参和添加 warmup。
只需要在 DNA 数据集上做实验。
目标是最后结果都超过 tri 实验。
用 GPU 0,1,2，开 tmux 窗口跑。
```

实现修改：

```text
新增 tri_gate 独立 warmup 参数:
    --tri_gate_start_iter
    --tri_gate_warmup

默认:
    tri_gate_start_iter = 3000
    tri_gate_warmup = 3000

训练时:
    train.py 每轮计算:
        tri_gate_alpha_scale = warmup(iteration, tri_gate_start_iter, tri_gate_warmup)

渲染时:
    render.py 设置:
        tri_gate_alpha_scale = 1.0

gaussian_renderer/__init__.py:
    把 pc.tri_gate_alpha_scale 传给 NonrigidDeformer。

nets/mlp_delta_non_rigid.py:
    tri_gate alpha 从:
        tri_gate_alpha * normalized(part_moe_alpha)
    改为:
        tri_gate_alpha * tri_gate_alpha_scale

因此 tri_gate 不再等 part_moe_start_iter=10000 后才生效，
shared MLP 在 Part-MoE 拷贝 expert 前已经能学习 gated tri feature。
```

公平性口径：

```text
不改:
    DNA 数据集
    六个序列
    25000 iterations
    part_moe_leg 基础
    part_label_schema=part_moe_leg
    num_parts=7
    tri_plane_extent=1.0
    tri_plane_dim/res 的默认候选仍从 tri 已有最佳设置开始

只改 tri_gate 自身参数:
    TRI_GATE_ALPHA
    TRI_GATE_INIT
    TRI_GATE_HIDDEN_DIM
    TRI_GATE_START_ITER
    TRI_GATE_WARMUP
```

已验证：

```text
bash -n scripts/exps_dnarendering.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    gaussian_renderer/__init__.py
    train.py
    render.py
    arguments/__init__.py

CPU 小张量测试:
    plain MLP input dim: 6
    ordinary tri MLP input dim: 38
    tri_gate MLP input dim: 6
    tri_gate forward 输出:
        d_xyz/d_rotation/d_scaling shape 正常

train.py warmup 函数测试:
    start=3000, warmup=3000
    iter 3000 -> 0.0
    iter 4500 -> 0.5
    iter 6000 -> 1.0
    iter 9000 -> 1.0
```

注意：

```text
当前 shell 的 PATH 没有 conda env bin，直接导入 train.py 时 knn_cuda 会找不到 ninja。
tmux 启动命令需要显式:
    PATH=/media/image/mxz/.conda/envs/seqavatar/bin:$PATH
```

第一轮完整六序列：

```text
实验标识:
    tri_gate_a04_s3w3

参数:
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    TRI_GATE_ALPHA=0.4
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=3000
    TRI_GATE_WARMUP=3000
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_a04_s3w3_gpu0
        GPU0: 0044_11, 0051_09
    tri_gate_a04_s3w3_gpu1
        GPU1: 0206_04, 0813_05
    tri_gate_a04_s3w3_gpu2
        GPU2: 0007_04, 0019_10

启动状态:
    参数已写入 logs/tri/20260728_tri_gate_a04_s3w3_gpu*_DNA-Rendering_tri_gate.log
    tmux 会话正常启动。
```

第一轮运行中状态：

```text
tri_gate_a04_s3w3:
    GPU0 0044_11 正常训练。
    GPU2 0007_04 正常训练。
    GPU1 0206_04 在 14040 iter 左右失败:
        RuntimeError: CUDA error: an illegal memory access was encountered

失败位置:
    train.py loss.backward()

判断:
    这不是 render 阶段失败。
    alpha=0.4 可能对 0206_04 偏激进，也可能触发了 CUDA rasterizer / backward 的不稳定。
    因此 tri_gate_a04_s3w3 不能作为最终完整方案。

保持 GPU0/GPU2 继续跑完，用于参考逐序列趋势。
```

GPU1 retry：

```text
实验标识:
    tri_gate_a03_s3w4

参数:
    TRI_GATE_ALPHA=0.3
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=3000
    TRI_GATE_WARMUP=4000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_a03_s3w4_gpu1
        GPU1: 0206_04, 0813_05

说明:
    这组先在 GPU1 上补测 0206/0813。
    最终只有同一组参数完整跑完六序列才算有效最终结果。
```

alpha=0.4 第一轮淘汰：

```text
tri_gate_a04_s3w3 已停止继续占卡。

原因:
    1. 0206_04 在 14040 iter 左右 CUDA illegal memory access。
    2. 0044_11 已出指标，但低于 tri:
        tri:
            PSNR       33.01947204271952
            SSIM       0.9781568845113119
            LPIPS*1000 21.017010944585007
        tri_gate_a04_s3w3:
            PSNR       32.983991940816246
            SSIM       0.9780832613507906
            LPIPS*1000 21.273130232778686

参考:
    0007_04 的 alpha=0.4 指标略超过 tri:
        PSNR       29.571448135375977
        SSIM       0.9587977970639865
        LPIPS*1000 44.22216749129196

结论:
    alpha=0.4 对部分序列有收益，但稳定性和 0044_11 表现不满足“超过 tri”目标。
```

完整候选组：

```text
实验标识:
    tri_gate_a03_s3w4

参数:
    TRI_GATE_ALPHA=0.3
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=3000
    TRI_GATE_WARMUP=4000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_a03_s3w4_gpu0
        GPU0: 0044_11, 0051_09
    tri_gate_a03_s3w4_gpu1
        GPU1: 0206_04, 0813_05
    tri_gate_a03_s3w4_gpu2
        GPU2: 0007_04, 0019_10

状态:
    三张卡已启动同一组参数，作为下一组完整六序列候选。
```

运行中淘汰记录：

```text
tri_gate_a03_s3w4 已停止继续占卡。

原因:
    该组已经有两个关键序列低于 tri，不能作为最终“超过 tri”的方案。

已完成指标:
    0044_11:
        tri:
            PSNR       33.01947204271952
            SSIM       0.9781568845113119
            LPIPS*1000 21.017010944585007
        tri_gate_a03_s3w4:
            PSNR       32.99650395711263
            SSIM       0.9781053364276886
            LPIPS*1000 21.21883356012404

    0206_04:
        tri:
            PSNR       31.623818985621135
            SSIM       0.9710065623124441
            LPIPS*1000 32.972589057559766
        tri_gate_a03_s3w4:
            PSNR       31.545159530639648
            SSIM       0.9709060991803805
            LPIPS*1000 33.106669411063194

    0007_04:
        tri:
            PSNR       29.57000511487325
            SSIM       0.9587606683373451
            LPIPS*1000 44.59049558887879
        tri_gate_a03_s3w4:
            PSNR       29.58457115491231
            SSIM       0.9587902670105298
            LPIPS*1000 44.31699852769574

判断:
    alpha=0.3, start=3000, warmup=4000 对 0007_04 有收益，
    但对 0044_11 / 0206_04 仍偏负。
    下一轮需要更保守地降低最终 alpha 或延后/拉长 warmup，减少 gate 对 tri 特征的早期扰动。
```

下一轮修改方向：

```text
当前 additive tri_gate 的问题:
    f_tri -> adapter 后直接加到 base features 上。
    这会改变原始 x/pose/seq feature 的输入语义，
    且 adapter 最后一层 zero init 会让 tri-plane 早期梯度链路偏弱。

新增只属于 tri_gate 的可选模式:
    --tri_gate_mode additive   # 保留现有形式
    --tri_gate_mode concat     # 新形式

concat 形式:
    gate_i = sigmoid(MLP_gate(features_base, f_tri_i))
    features = concat(features_base, alpha * warmup * gate_i * f_tri_i)

公平性:
    不改 original / part_moe_leg / tri / tri_part。
    tri_gate 仍基于 part_moe_leg。
    tri_plane_dim/res/extent 仍从当前 tri 最佳公平设置 dim=32,res=64,extent=1.0 开始。

目的:
    保留 direct tri 的 MLP 输入容量，
    但用 gate + warmup 控制 tri 特征强度，避免直接 concat 的强记忆问题。
```

代码与最小验证：

```text
新增参数:
    --tri_gate_mode

取值:
    additive:
        保留原 tri_gate:
            features = features_base + alpha * warmup * gate_i * Adapter(f_tri_i)

    concat:
        新增 gated concat:
            features = concat(features_base, alpha * warmup * gate_i * f_tri_i)

接线文件:
    arguments/__init__.py
    scene/gaussian_model.py
    nets/mlp_delta_non_rigid.py
    scripts/exps_dnarendering.sh

脚本环境变量:
    TRI_GATE_MODE=additive|concat

验证:
    bash -n scripts/exps_dnarendering.sh

    PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
        nets/mlp_delta_non_rigid.py
        scene/gaussian_model.py
        gaussian_renderer/__init__.py
        train.py
        render.py
        arguments/__init__.py

    CPU 小张量 forward:
        plain first_in            6
        tri first_in              38
        tri_gate_additive first_in 6
        tri_gate_concat first_in   38

说明:
    tri_gate_concat 保留 direct tri 的输入容量。
    additive 旧路径保留。
    original / part_moe_leg / tri / tri_part 不受该模式影响。
```

screening 组 1：

```text
目的:
    用三个敏感序列先判断 gated concat 是否有希望超过 tri。

参数:
    TRI_GATE_MODE=concat
    TRI_GATE_ALPHA=1.0
    TRI_GATE_INIT=0.9
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=0
    TRI_GATE_WARMUP=3000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_screen1_gpu0
        GPU0: 0044_11
    tri_gate_screen1_gpu1
        GPU1: 0206_04
    tri_gate_screen1_gpu2
        GPU2: 0007_04

说明:
    RUN_TIME 只作为运行标识，实验模式仍是 tri_gate。
    参数写在命令行环境变量和日志开头，不新建 triC 之类消融名称。
```

screening 组 1 结果：

```text
参数:
    TRI_GATE_MODE=concat
    TRI_GATE_ALPHA=1.0
    TRI_GATE_INIT=0.9
    TRI_GATE_START_ITER=0
    TRI_GATE_WARMUP=3000

0044_11:
    tri:
        PSNR       33.01947204271952
        SSIM       0.9781568845113119
        LPIPS*1000 21.017010944585007
    tri_gate:
        PSNR       32.96001152992248
        SSIM       0.9781957934300105
        LPIPS*1000 21.057943495300908

0206_04:
    tri:
        PSNR       31.623818985621135
        SSIM       0.9710065623124441
        LPIPS*1000 32.972589057559766
    tri_gate:
        PSNR       31.534537426630656
        SSIM       0.9705853089690208
        LPIPS*1000 33.093597662324704

0007_04:
    tri:
        PSNR       29.57000511487325
        SSIM       0.9587606683373451
        LPIPS*1000 44.59049558887879
    tri_gate:
        PSNR       29.578313477834065
        SSIM       0.9588532492518425
        LPIPS*1000 44.22089303843677

结论:
    gated concat 保留容量后，0007_04 三项超过 tri。
    但 0044_11 / 0206_04 仍不满足要求。
    这组不会进入完整六序列。

下一组:
    增强 tri_gate_concat 的有效强度:
        TRI_GATE_ALPHA=1.1
        TRI_GATE_INIT=0.95
        TRI_GATE_WARMUP=1000
    目标是更接近 direct tri 的输入强度，同时保留可学习 gate。
```

screening 组 2：

```text
参数:
    TRI_GATE_MODE=concat
    TRI_GATE_ALPHA=1.1
    TRI_GATE_INIT=0.95
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=0
    TRI_GATE_WARMUP=1000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_screen2_gpu0
        GPU0: 0044_11
    tri_gate_screen2_gpu1
        GPU1: 0206_04
    tri_gate_screen2_gpu2
        GPU2: 0007_04
```

screening 组 2 结果：

```text
参数:
    TRI_GATE_MODE=concat
    TRI_GATE_ALPHA=1.1
    TRI_GATE_INIT=0.95
    TRI_GATE_START_ITER=0
    TRI_GATE_WARMUP=1000

0044_11:
    tri:
        PSNR       33.01947204271952
        SSIM       0.9781568845113119
        LPIPS*1000 21.017010944585007
    tri_gate:
        PSNR       32.97856990496317
        SSIM       0.9781970143318176
        LPIPS*1000 21.016903702790536

0206_04:
    tri:
        PSNR       31.623818985621135
        SSIM       0.9710065623124441
        LPIPS*1000 32.972589057559766
    tri_gate:
        PSNR       31.59208936691284
        SSIM       0.971009287238121
        LPIPS*1000 32.53928067473074

0007_04:
    tri:
        PSNR       29.57000511487325
        SSIM       0.9587606683373451
        LPIPS*1000 44.59049558887879
    tri_gate:
        PSNR       29.568023363749187
        SSIM       0.958826502164205
        LPIPS*1000 44.10010984477898

结论:
    相比 screen1，0206_04 明显改善，0044_11 LPIPS 基本追平 tri，
    但 PSNR 仍低于 tri。
    0007_04 在更强初始 gate 下 PSNR 反而略低。

下一组:
    不继续单纯提高 gate_init。
    改为:
        TRI_GATE_ALPHA=1.5
        TRI_GATE_INIT=0.6
        TRI_GATE_WARMUP=1000
    让有效强度初始接近 screen1，但给 gate 留出更大的可学习上限。
```

screening 组 3：

```text
参数:
    TRI_GATE_MODE=concat
    TRI_GATE_ALPHA=1.5
    TRI_GATE_INIT=0.6
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=0
    TRI_GATE_WARMUP=1000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_screen3_gpu0
        GPU0: 0044_11
    tri_gate_screen3_gpu1
        GPU1: 0206_04
    tri_gate_screen3_gpu2
        GPU2: 0007_04
```

screening 组 3 结果：

```text
参数:
    TRI_GATE_MODE=concat
    TRI_GATE_ALPHA=1.5
    TRI_GATE_INIT=0.6
    TRI_GATE_START_ITER=0
    TRI_GATE_WARMUP=1000

0044_11:
    tri:
        PSNR       33.01947204271952
        SSIM       0.9781568845113119
        LPIPS*1000 21.017010944585007
    tri_gate:
        PSNR       32.98333891232809
        SSIM       0.9781879449884097
        LPIPS*1000 21.006147206450503

0206_04:
    tri:
        PSNR       31.623818985621135
        SSIM       0.9710065623124441
        LPIPS*1000 32.972589057559766
    tri_gate:
        PSNR       31.53624178568522
        SSIM       0.9707346618175506
        LPIPS*1000 33.04816630358497

0007_04:
    tri:
        PSNR       29.57000511487325
        SSIM       0.9587606683373451
        LPIPS*1000 44.59049558887879
    tri_gate:
        PSNR       29.623930740356446
        SSIM       0.9589480131864547
        LPIPS*1000 44.43178423680365

结论:
    高 alpha 上限 + 低 gate init 对 0007_04 有明显收益，
    但 0206_04 退化，0044_11 仍未超过 tri。
    concat gate 作为 tri 的唯一入口不够稳。

下一步代码方向:
    新增 tri_gate_mode=scale。
    形式:
        gate_i = sigmoid(MLP_gate(features_base, f_tri_i))
        f_tri'_i = (1 + alpha * warmup * (gate_i - gate_init)) * f_tri_i
        features = concat(features_base, f_tri'_i)

    初始 gate_i = gate_init，所以初始严格等价 direct tri。
    这样公平性更好:
        不改变 direct tri 的初始训练路径，
        只让 gate 学 per-point 的小幅增强/抑制。
```

tri_gate_mode=scale 实现与验证：

```text
新增可选模式:
    --tri_gate_mode scale

实现:
    gate_i = sigmoid(MLP_gate(features_base, f_tri_i))
    f_tri'_i = (1 + alpha * warmup * (gate_i - gate_init)) * f_tri_i
    features = concat(features_base, f_tri'_i)

性质:
    gate 最后一层 weight=0，bias=logit(gate_init)。
    初始 gate_i=gate_init，因此 f_tri'_i=f_tri_i。
    也就是说 scale 模式初始严格等价 direct tri，
    后续只学习点级 tri 特征缩放。

验证:
    bash -n scripts/exps_dnarendering.sh

    PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
        nets/mlp_delta_non_rigid.py
        scene/gaussian_model.py
        gaussian_renderer/__init__.py
        train.py
        render.py
        arguments/__init__.py

    CPU 小张量 forward:
        tri_gate_scale first_in = 38
        输出:
            d_xyz      [1, 5, 3]
            d_rotation [1, 5, 4]
            d_scaling  [1, 5, 3]
```

screening 组 4：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.5
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=3000
    TRI_GATE_WARMUP=5000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

目的:
    保留 direct tri 的完整路径；
    3000 iter 后再逐步学习点级缩放，避免早期训练被 gate 扰动。

tmux:
    tri_gate_screen4_gpu0
        GPU0: 0044_11
    tri_gate_screen4_gpu1
        GPU1: 0206_04
    tri_gate_screen4_gpu2
        GPU2: 0007_04
```

screening 组 4 中止说明：

```text
运行中发现公平性问题:
    tri_gate 的 gate 模块在 MLP/head 初始化前创建，
    会额外消耗随机数。
    因此即使 scale 模式公式初始等价 direct tri，
    实际 MLP/head 初始权重也不一定与 tri 对齐。

修复:
    gate 模块延后到 MLP/head 初始化之后创建。

验证:
    同一 random seed 下，对比 tri 与 tri_gate_mode=scale:
        mlp               max diff 0.0
        gaussian_warp     max diff 0.0
        gaussian_rotation max diff 0.0
        gaussian_scaling  max diff 0.0

screen4 是修复前启动的，且 0206_04 已低于 tri:
    0206_04:
        PSNR       31.502585474650065
        SSIM       0.9704470753669738
        LPIPS*1000 33.3999853891631

因此停止 screen4 剩余 0044_11，重新启动修复后的公平版 screen5。
```

screening 组 5：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.5
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=3000
    TRI_GATE_WARMUP=5000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

与 screen4 的区别:
    使用修复后的 gate 初始化顺序。
    tri_gate_scale 的 MLP/head 初始权重与 direct tri 对齐。

tmux:
    tri_gate_screen5_gpu0
        GPU0: 0044_11
    tri_gate_screen5_gpu1
        GPU1: 0206_04
    tri_gate_screen5_gpu2
        GPU2: 0007_04
```

screening 组 5 结果：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.5
    TRI_GATE_INIT=0.5
    TRI_GATE_START_ITER=3000
    TRI_GATE_WARMUP=5000

0044_11:
    tri:
        PSNR       33.01947204271952
        SSIM       0.9781568845113119
        LPIPS*1000 21.017010944585007
    tri_gate:
        PSNR       33.056373023986815
        SSIM       0.9782408942778905
        LPIPS*1000 21.020517467210688

0206_04:
    tri:
        PSNR       31.623818985621135
        SSIM       0.9710065623124441
        LPIPS*1000 32.972589057559766
    tri_gate:
        PSNR       31.566159280141193
        SSIM       0.9707474038004875
        LPIPS*1000 33.18445282056928

0007_04:
    tri:
        PSNR       29.57000511487325
        SSIM       0.9587606683373451
        LPIPS*1000 44.59049558887879
    tri_gate:
        PSNR       29.56226814587911
        SSIM       0.9588541895151138
        LPIPS*1000 44.36838128603995

结论:
    初始化公平后，0044_11 有收益。
    0007_04 的感知指标有收益但 PSNR 低一点。
    0206_04 仍明显低于 tri。

下一组:
    将 gate 缩放推迟到 Part-MoE 启动后:
        TRI_GATE_START_ITER=10000
        TRI_GATE_WARMUP=5000
        TRI_GATE_ALPHA=0.3
    前 10000 iter 完全等价 tri，避免早期共享 MLP 被 gate 调制影响。
```

screening 组 6：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.3
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=10000
    TRI_GATE_WARMUP=5000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_screen6_gpu0
        GPU0: 0044_11
    tri_gate_screen6_gpu1
        GPU1: 0206_04
    tri_gate_screen6_gpu2
        GPU2: 0007_04
```

screening 组 6 结果：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.3
    TRI_GATE_INIT=0.5
    TRI_GATE_START_ITER=10000
    TRI_GATE_WARMUP=5000

0044_11:
    tri:
        PSNR       33.01947204271952
        SSIM       0.9781568845113119
        LPIPS*1000 21.017010944585007
    tri_gate:
        PSNR       33.03347080548604
        SSIM       0.9782720178365707
        LPIPS*1000 20.98686106813451

0206_04:
    tri:
        PSNR       31.623818985621135
        SSIM       0.9710065623124441
        LPIPS*1000 32.972589057559766
    tri_gate:
        PSNR       31.523118098576862
        SSIM       0.9707509453097979
        LPIPS*1000 32.657264002288386

0007_04:
    tri:
        PSNR       29.57000511487325
        SSIM       0.9587606683373451
        LPIPS*1000 44.59049558887879
    tri_gate:
        PSNR       29.59822532335917
        SSIM       0.9590465133388837
        LPIPS*1000 43.87906308596333

结论:
    0044_11 和 0007_04 三项均超过 tri。
    0206_04 的 LPIPS 超过 tri，但 PSNR/SSIM 仍低。
后续先集中筛 0206_04。
```

screening 组 7：0206_04 专项 concat 强度扫描

```text
共同参数:
    TRI_GATE_MODE=concat
    TRI_GATE_INIT=0.95
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=0
    TRI_GATE_WARMUP=1000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_screen7a_gpu0
        GPU0: 0206_04
        TRI_GATE_ALPHA=1.2

    tri_gate_screen7b_gpu1
        GPU1: 0206_04
        TRI_GATE_ALPHA=1.3

    tri_gate_screen7c_gpu2
        GPU2: 0206_04
        TRI_GATE_ALPHA=1.4
```

screening 组 7 结果：

```text
0206_04 tri:
    PSNR       31.623818985621135
    SSIM       0.9710065623124441
    LPIPS*1000 32.972589057559766

TRI_GATE_ALPHA=1.2:
    PSNR       31.47824543317159
    SSIM       0.9702153558532397
    LPIPS*1000 33.80788190600772

TRI_GATE_ALPHA=1.3:
    19080 iter 左右失败:
        RuntimeError: CUDA error: an illegal memory access was encountered

TRI_GATE_ALPHA=1.4:
    PSNR       31.52886069615682
    SSIM       0.9709127172827721
    LPIPS*1000 32.82117587514222

结论:
    concat 强度扫描没有让 0206_04 超过 tri。
    alpha=1.4 的 LPIPS 超过 tri，但 PSNR/SSIM 仍低。

下一步:
    使用目前最稳的 screen6 参数补跑剩余 0051_09 / 0813_05 / 0019_10。
    结合 screen6 已有 0044_11 / 0206_04 / 0007_04 计算六序列平均。
```

screen6 参数补跑剩余三序列：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.3
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=10000
    TRI_GATE_WARMUP=5000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_screen6_rest_gpu0
        GPU0: 0051_09
    tri_gate_screen6_rest_gpu1
        GPU1: 0813_05
    tri_gate_screen6_rest_gpu2
        GPU2: 0019_10
```

screen6 六序列汇总：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.3
    TRI_GATE_INIT=0.5
    TRI_GATE_START_ITER=10000
    TRI_GATE_WARMUP=5000

逐序列:
    0007_04:
        PSNR       29.59822532335917
        SSIM       0.9590465133388837
        LPIPS*1000 43.87906308596333

    0019_10:
        PSNR       35.43037147521973
        SSIM       0.9815082629521688
        LPIPS*1000 20.48678658902645

    0044_11:
        PSNR       33.03347080548604
        SSIM       0.9782720178365707
        LPIPS*1000 20.98686106813451

    0051_09:
        PSNR       28.75835018157959
        SSIM       0.9720531940460205
        LPIPS*1000 30.28237490604321

    0206_04:
        PSNR       31.523118098576862
        SSIM       0.9707509453097979
        LPIPS*1000 32.657264002288386

    0813_05:
        PSNR       36.232907358805335
        SSIM       0.9873329182465871
        LPIPS*1000 17.892275812725227

六序列平均:
    screen6:
        PSNR       32.429407207171124
        SSIM       0.9748273086216713
        LPIPS*1000 27.697437577363516

    tri:
        PSNR       32.43667180538178
        SSIM       0.9748073815471597
        LPIPS*1000 27.8485285402793

    diff(screen6 - tri):
        PSNR       -0.007264598210654209
        SSIM        0.0000199270745118163
        LPIPS*1000 -0.15109096291578203

结论:
    screen6 的 SSIM / LPIPS 平均超过 tri。
    PSNR 平均仍低 0.007264598210654209。

下一步:
    补跑 screen5 参数在 0051_09 / 0813_05 / 0019_10 上的结果。
    screen5 在 0044_11 上 PSNR 增益更大，有机会把平均 PSNR 补过 tri。
```

screen5 参数补跑剩余三序列：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.5
    TRI_GATE_INIT=0.5
    TRI_GATE_HIDDEN_DIM=128
    TRI_GATE_START_ITER=3000
    TRI_GATE_WARMUP=5000
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
    SKIP_LOAD_TEST_CAMERAS=1

tmux:
    tri_gate_screen5_rest_gpu0
        GPU0: 0051_09
    tri_gate_screen5_rest_gpu1
        GPU1: 0813_05
    tri_gate_screen5_rest_gpu2
        GPU2: 0019_10
```

screen5 六序列汇总：

```text
参数:
    TRI_GATE_MODE=scale
    TRI_GATE_ALPHA=0.5
    TRI_GATE_INIT=0.5
    TRI_GATE_START_ITER=3000
    TRI_GATE_WARMUP=5000

逐序列:
    0007_04:
        PSNR       29.56226814587911
        SSIM       0.9588541895151138
        LPIPS*1000 44.36838128603995

    0019_10:
        PSNR       35.49039780298869
        SSIM       0.9817660937706629
        LPIPS*1000 20.440836576744914

    0044_11:
        PSNR       33.056373023986815
        SSIM       0.9782408942778905
        LPIPS*1000 21.020517467210688

    0051_09:
        PSNR       28.692925961812335
        SSIM       0.9719656507174174
        LPIPS*1000 30.153782626924414

    0206_04:
        PSNR       31.566159280141193
        SSIM       0.9707474038004875
        LPIPS*1000 33.18445282056928

    0813_05:
        PSNR       36.21320161819458
        SSIM       0.9872532958785692
        LPIPS*1000 17.97534309638043

六序列平均:
    screen5:
        PSNR       32.43022097216712
        SSIM       0.9748045879933569
        LPIPS*1000 27.85721897897828

    tri:
        PSNR       32.43667180538178
        SSIM       0.9748073815471597
        LPIPS*1000 27.8485285402793

    diff(screen5 - tri):
        PSNR       -0.006450833214654968
        SSIM       -0.0000027935538027268336
        LPIPS*1000  0.008690438698977824

最终阶段结论:
    本轮没有找到“PSNR / SSIM / LPIPS*1000 全部超过 tri”的 tri_gate 参数。

    最接近的是 screen6:
        PSNR       32.429407207171124
        SSIM       0.9748273086216713
        LPIPS*1000 27.697437577363516

    screen6 相比 tri:
        PSNR       -0.007264598210654209
        SSIM        0.0000199270745118163
        LPIPS*1000 -0.15109096291578203

    也就是说:
        screen6 的 SSIM 和 LPIPS 超过 tri，
        但 PSNR 仍略低。

主要瓶颈:
    0206_04 在所有公平 tri_gate 调参里都没有稳定超过 tri。
    0206_04 最接近的一次是 screen2:
        TRI_GATE_MODE=concat
        TRI_GATE_ALPHA=1.1
        TRI_GATE_INIT=0.95
        TRI_GATE_START_ITER=0
        TRI_GATE_WARMUP=1000

        PSNR       31.59208936691284
        SSIM       0.971009287238121
        LPIPS*1000 32.53928067473074

    但 PSNR 仍低于 tri 的 31.623818985621135。

判断:
    在不改变数据、训练轮数、评价方式、不针对具体序列手工设置参数的公平前提下，
    当前 tri_gate 方向继续调 alpha / gate_init / start / warmup 的边际收益很小。
```

## 2026-07-28 tri 当前优化机制与不足总结

用户判断：

```text
目前 tri_gate 可以放弃。
需要重新总结当前 tri 是怎么优化原网络的，以及还有什么不足。
```

当前 tri 的实现机制：

```text
tri 不是重新采样 Gaussian 点，也不是替代 SMPL/LBS。
它是在每个 canonical Gaussian 坐标 query_xyz=means3D 处查询一个可学习三平面特征场:
    x_i -> sample tri-plane -> f_tri(x_i)

三平面由三个可学习参数平面组成:
    XY / XZ / YZ

采样方式:
    query_xyz / tri_plane_extent 后 clamp 到 [-1, 1]
    分别对三个平面 grid_sample
    最后取三平面特征平均

当前最佳公平设置:
    TRI_PLANE_DIM=32
    TRI_PLANE_RES=64
    TRI_PLANE_EXTENT=1.0
```

tri 与 part_moe_leg 的关系：

```text
tri 明确建立在 part_moe_leg 上。
代码中要求:
    --use_part_moe
    --part_label_schema part_moe_leg
    --num_parts 7

因此 tri 的优化对象不是 original，而是在 part_moe_leg 已经按部件分 expert 的基础上，
给每个 Gaussian 额外提供一个 canonical 空间可学习局部特征。
```

tri 优化原网络的方式：

```text
part_moe_leg 的 expert 输入原本主要来自:
    x_emb_i
    pose_feat
    seq_pose_feat
    seq_xyz_feat

tri 后变成:
    concat(x_emb_i, pose_feat, seq_pose_feat, seq_xyz_feat, f_tri_i)

也就是说，tri 给每个 Gaussian 增加了一个“标准空间位置相关的可学习局部记忆”。
Part-MoE expert 再根据各自 part label 使用这个增强后的输入预测:
    d_xyz
    d_rotation
    d_scaling
```

为什么 tri 能带来提升：

```text
1. 原来的 positional embedding 是固定编码，只提供坐标频率信息；
   tri-plane 是可学习特征场，可以记住标准空间中哪些区域更容易出现细节误差。

2. part_moe_leg 已经把不同身体部件分给不同 expert；
   tri 进一步给这些 expert 提供更细的局部空间条件，使 expert 不只知道“属于哪个 part”，
   还知道“在这个 part 的哪个 canonical 区域”。

3. 对 DNA 这类局部外观和衣物细节明显的数据，canonical 空间上的局部特征可以补足
   shared MLP / part expert 对空间细节表达不足的问题。
```

当前 tri 的主要不足：

```text
1. tri 只是空间特征增强，不是真正的运动自适应模块。
   它不显式使用运动强弱、速度、加速度、局部刚性或时序变化。
   因此它更像是增强 canonical 空间表达能力，而不是根据动态难度分配形变能力。

2. 所有 part expert 默认共享同一组三平面特征。
   虽然不同 part 的 expert 会学习不同映射，但 f_tri 本身没有天然的部件语义。
   tri_part 尝试过 part-conditioned FiLM，但实验结果下降，说明简单地给 tri 加 part 条件不一定更稳。

3. tri 直接 concat 到第一层，容易改变 non-rigid MLP / expert 的输入分布。
   这种方式表达力强，但控制性弱，可能让网络偏向记住 canonical 空间细节，
   对跨动作动态规律的建模有限。

4. tri_plane 是静态 canonical 特征场。
   同一个 Gaussian 坐标在不同帧查到的 f_tri 基本相同，动态差异仍主要依赖 pose / seq 条件和 expert。
   如果某个序列误差来自快速运动、遮挡边界或局部非刚性变化，tri 只能间接帮助。

5. 增益已经接近饱和。
   DNA 上调过 dim / res / extent 后，extent=1.0 的 tri 是当前较稳结果；
   后续 tri_gate 大量调参没有稳定超过 tri，说明简单 gate/adapter 的边际收益很小。

6. 指标收益不均衡。
   tri_gate 最好一次只做到 SSIM / LPIPS 略优于 tri，但 PSNR 仍低；
   0206_04 是主要瓶颈序列，说明当前 tri 类方法对某些复杂动态场景的提升不足。
```

对 tri_gate 的结论：

```text
tri_gate 的出发点是避免 tri 直接 concat 后变成强记忆，通过 gate/adapter 控制 tri 特征强度。
但 DNA 六序列公平调参后，没有找到 PSNR / SSIM / LPIPS*1000 全部超过 tri 的设置。

当前最接近的 screen6:
    PSNR       32.429407207171124
    SSIM       0.9748273086216713
    LPIPS*1000 27.697437577363516

tri:
    PSNR       32.43667180538178
    SSIM       0.9748073815471597
    LPIPS*1000 27.8485285402793

screen6 相比 tri:
    PSNR       -0.007264598210654209
    SSIM        0.0000199270745118163
    LPIPS*1000 -0.15109096291578203

判断:
    tri_gate 可以暂时放弃。
    后续如果继续做第二创新点，不应继续围绕 gate_init / alpha / warmup 小范围调参，
    而应该转向更明确的 motion-aware / part-aware / rigidity-aware 形变能力分配。
```

阶段性判断：

```text
tri 可以作为第二模块的基础版本:
    在 part_moe_leg 基础上引入 canonical 空间可学习三平面特征，
    提升各部件 expert 对局部空间细节的表达能力。

但它的创新强度主要体现在“空间特征场增强”，不是“动态区域自适应”。
如果论文中要把第二点讲得更强，需要进一步设计:
    根据运动强弱、部件属性、局部刚性程度，动态决定哪里增强形变能力、哪里保持稳定约束。

当前可保留结论:
    tri 有稳定价值；
    tri_gate 暂不值得继续优化；
    下一阶段应避免只做 gate 调参。
```

## 2026-07-28 tri_part 失败后 part-aware 方向判断

用户问题：

```text
之前尝试 tri_part 不成功，part-aware 还有改进价值吗？
```

判断：

```text
part-aware 方向仍然有改进价值，但 tri_part 当前这种做法不值得继续沿用。

原因是:
    part_moe_leg 已经把人体按 part 分 expert。
    如果 tri_part 只是再对同一个 tri feature 做 part-conditioned FiLM，
    很容易和 part_moe_leg 的 expert 分工重复。

也就是说，tri_part 失败不能说明 part-aware 没价值；
它更说明“简单把 part embedding/FiLM 加到 tri feature 上”不是有效形式。
```

tri_part 不成功的可能原因：

```text
1. 与 part_moe_leg 功能重复。
   part_moe_leg 已经让不同 part 用不同 expert 学非刚性形变。
   tri_part 再给 tri 特征加 part FiLM，新增信息有限，反而增加优化难度。

2. part label 是粗粒度的。
   当前 7 个 part 适合做 expert 路由，但不一定适合直接调制每个点的空间特征。
   同一个 part 内还有主体、边界、关节附近、衣物区域等差异，单一 part 条件太粗。

3. tri feature 本身是 canonical 空间静态记忆。
   加 part FiLM 后仍然没有显式引入运动强弱、局部刚性或时序变化。
   因此它没有解决 tri 的核心不足，只是换了一种融合方式。

4. FiLM 会直接改变 tri feature 分布。
   对已经调稳的 tri 来说，额外调制可能破坏原本有效的空间特征，而不一定提供可靠收益。
```

更有价值的 part-aware 改进方向：

```text
1. part-aware capacity allocation
   不再只是给 feature 加 part 条件，而是让不同 part 拥有不同形变容量。
   例如腿/手臂/脚等高动态 part 使用更大的隐层、更多 tri 通道或更强非刚性分支；
   躯干等稳定 part 使用较小容量和更强正则。

2. part-aware regularization
   不同 part 使用不同约束强度。
   高刚性区域加强平滑/稳定约束；
   关节、衣物边界、高运动区域放松约束。
   这比单纯增加网络容量更符合“哪里该动、哪里该稳”的目标。

3. part + motion aware gating
   gate 不只依赖 f_tri 或 base feature，而要显式依赖 part 的运动强弱:
       velocity / acceleration / joint angular change / seq_xyz motion magnitude
   让高动态 part 或高动态帧自动获得更强 tri/nonrigid 能力。

4. intra-part aware design
   只用 part label 不够，应继续区分同一 part 内部区域。
   例如根据到关节边界的距离、局部 LBS weight entropy、局部 motion magnitude，
   判断 Gaussian 是刚性主体、关节过渡区，还是衣物/边界区域。
```

阶段建议：

```text
如果继续优化第二创新点，不建议继续做 tri_part 这种“part-conditioned tri feature”。

更值得尝试的是:
    在 tri 的基础上加入 part-aware 的容量/正则分配，
    并让这个分配依赖运动强弱和局部刚性，而不是只依赖 part label。

    一句话:
    part-aware 还有价值；
    但价值不在“给 tri 加 part FiLM”，而在“按部件和局部动态属性分配形变能力与稳定约束”。
```

## 2026-08-08 tri_part 为什么会失败

```text
如果你说的是早先那个 tri_part = shared tri + part-conditioned FiLM 版本，
它失败的核心不是“part-aware 完全没用”，而是“这个改法给 tri 增加的有效信息太少”。

主要原因：
    1. 它没有给每个 part 单独一套 tri-plane。
       底层仍然是 shared TriPlaneFeature，
       所以 part 之间共享同一张 canonical 空间记忆表。

    2. PartFiLM 只是在 shared tri 上做缩放/平移，
       改的是“怎么解释 tri feature”，
       不是“tri feature 本身学什么”。
       这很容易和 part_moe_leg 已有的 expert 分工重复。

    3. part label 太粗。
       7 个 part 够做路由，但不够表达同一 part 内部的主体/边界/关节/衣物差异。

    4. tri 本身还是静态 canonical feature，
       不看速度、加速度、局部刚性。
       所以 tri_part 只是给静态记忆再做一次部件调制，
       没有补上真正的动态难点。

    5. 这个结构里 residual 和 gate 是乘在一起的，
       早期 tri_part 还是接近普通 tri，
       part-specific 信号进入得慢，优化会更难。

所以它不是“把部件信息加上去就一定更强”；
更准确地说，是“共享空间表 + 粗部件 FiLM”这个形式不够锋利，
容易把问题变复杂，但没有带来足够新的表达能力。
```

## 2026-07-28 tri_part 改造：part/motion-aware residual tri feature

用户要求：

```text
继续修改 tri_part。
在 DNA 六个序列上开 tmux 跑实验。
尝试“按部件和局部动态属性分配形变能力与稳定约束”的思路，
目标是直到超过 tri 指标。
GPU0 有三百多 MB 显存占用，OOM 时优先换卡。
```

本次实现：

```text
废弃旧 tri_part 的简单 part-conditioned FiLM。

新的 tri_part 仍建立在 tri + part_moe_leg 上，普通 tri / part_moe_leg 不走该分支。

保留 shared TriPlaneFeature:
    f_tri_shared(x_i)

新增 part-specific residual tri planes:
    f_part_delta_p(x_i)

用 part label、seq_xyz motion strength、part confidence 预测逐点 gate:
    gate_i = sigmoid(part_prior_p
                     + motion_gain * motion_norm_i
                     + boundary_gain * (1 - part_conf_i)
                     + MLP(f_tri_i, part_emb_p, motion_norm_i, boundary_i))

输出给 expert 的 tri 特征:
    f_tri_part_i = f_tri_shared_i + alpha * gate_i * f_part_delta_p(x_i)

其中:
    高动态点 motion_norm 高 -> gate 更大 -> 更强形变特征容量
    低 part_conf / 边界点 -> gate 更大 -> 边界/不确定区域更自由
    高置信稳定点 -> gate 较小，并通过 reg 更稳定
```

初始化与隔离：

```text
part_delta_planes 全零初始化。
因此训练开始时:
    f_tri_part_i == f_tri_shared_i
    tri_part 与 tri 完全等价

最小 CPU forward 验证:
    普通 tri 和新 tri_part 输出 max diff = 0.0

新增参数只在 --use_tri_part 下创建/调用。
--use_tri 仍保持原来的 shared tri 直接 concat。
```

新增参数：

```text
--tri_part_alpha
--tri_part_motion_gain
--tri_part_boundary_gain
--tri_part_hidden_dim
--tri_part_reg_w

DNA 脚本默认:
    TRI_PART_ALPHA=1.0
    TRI_PART_MOTION_GAIN=0.5
    TRI_PART_BOUNDARY_GAIN=0.5
    TRI_PART_HIDDEN_DIM=64
    TRI_PART_REG_W=0.0001
```

训练正则：

```text
只在 use_tri_part=True 时启用。

对新增 residual tri feature 加轻量稳定约束:
    reg_weight = (1 - sigmoid(motion_norm_i)) * part_conf_i
    reg = mean(reg_weight * ||alpha * gate_i * f_part_delta_p(x_i)||^2)

含义:
    稳定且高置信的 Gaussian 不鼓励过多新增特征扰动；
    高动态或边界/低置信 Gaussian 允许更强局部表达。
```

已验证：

```text
bash -n scripts/exps_dnarendering.sh
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh

PYTHONDONTWRITEBYTECODE=1 py_compile:
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    gaussian_renderer/__init__.py
    render.py
    train.py
    arguments/__init__.py

最小 forward:
    NonrigidDeformer(use_tri=True, use_tri_part=True, use_part_moe=True, num_parts=7)
    输出 shape:
        d_xyz      [1, 12, 3]
        d_rotation [1, 12, 4]
        d_scaling  [1, 12, 3]
    与普通 tri 输出 max diff:
        0.0
```

## 2026-08-06 tri_part_pa_v1 六序列完成确认

状态：

```text
已经跑完。
tmux 进程已退出。
六个序列的 train.log 都有 Training complete。
六个序列的 render.log 都有 25000 iter 的 novelview evaluation。
```

本轮六序列结果：

```text
0007_04:
    PSNR  29.654573583602904
    SSIM  0.9590438589453697
    LPIPS 0.044234784319996834

0019_10:
    PSNR  35.433257484436034
    SSIM  0.9814779336253802
    LPIPS 0.02056680005043745

0044_11:
    PSNR  32.98671916325887
    SSIM  0.9782769590616226
    LPIPS 0.020906405655356744

0051_09:
    PSNR  28.72750129699707
    SSIM  0.9719237764676412
    LPIPS 0.030388579556408026

0206_04:
    PSNR  31.6219766775767
    SSIM  0.9712262744704883
    LPIPS 0.032623827721302706

0813_05:
    PSNR  36.14729544321696
    SSIM  0.9871677324175835
    LPIPS 0.018049360268438855
```

六序列平均：

```text
PSNR       32.42855394151476
SSIM       0.9748527558313476
LPIPS*1000 27.794959595323437
```

和 tri 对比：

```text
PSNR       -0.008117863867020406
SSIM        0.0000453742841878912
LPIPS*1000 -0.053568944955862574
```

结论：

```text
这轮 tri_part 跑完了。
平均上还没有超过 tri 的 PSNR，
但 SSIM / LPIPS*1000 有小幅改善。
```

## 2026-08-06 tri 消融实验代码总结

只讨论 tri 本身，不包含后续 tri_part / tri_gate 优化。

```text
tri 的改动很单纯:
    在 part_moe_leg 的 non-rigid MLP 输入里，
    额外拼接一个 canonical 空间的可学习三平面特征 f_tri。

它没有改 SMPL/LBS，没有改损失函数，没有改 part expert 路由，
只是在部件专家前面加了一个可学习空间特征场。
```

代码层改动：

```text
1. nets/mlp_delta_non_rigid.py
    新增 TriPlaneFeature:
        3 个可学习平面参数:
            XY / XZ / YZ
        参数初始化为 0。
        对 query_xyz / tri_plane_extent 做归一化后，
        用 grid_sample 分别采样三个平面并平均，得到 f_tri。

    在 NonrigidDeformer.forward 中:
        features = concat(x_emb, pose_feat, seq_pose_feat, seq_xyz_feat)
        if use_tri:
            tri_features = sample_tri_features(query_xyz=means3D)
            features = concat(features, tri_features)

    也就是:
        tri 不是单独输出一个分支，
        而是直接作为非刚性 MLP / Part-MoE expert 的额外输入。

2. scene/gaussian_model.py
    tri 只允许建立在 part_moe_leg 上:
        --use_part_moe
        --part_label_schema part_moe_leg
        --num_parts 7
    并把 tri_plane_dim / tri_plane_res / tri_plane_extent 传给 NonrigidDeformer。

3. gaussian_renderer/__init__.py
    tri 的查询坐标来自 means3D，也就是进入非刚性分支前的 canonical Gaussian 坐标。
    之后仍然会继续走 coarse_deform_c2source，
    所以 tri 只是 residual feature augmentation，不替代 SMPL/LBS。

4. scripts/exps_dnarendering.sh
    tri 模式 = part_moe_leg + use_tri
    日志写到 logs/tri
    DNA 上固定 final_eval_only=1
    tri_plane 用的公平配置是:
        TRI_PLANE_DIM=32
        TRI_PLANE_RES=64
        TRI_PLANE_EXTENT=1.0
```

tri 的优化逻辑：

```text
原来的 part_moe_leg 主要依赖:
    x_emb_i + pose / seq 条件

tri 之后多了:
    f_tri_i = canonical 空间局部特征

这等于给每个 Gaussian 加了一个“标准空间记忆”。
同一个位置在不同帧上查到的 tri 特征基本一致，
所以 expert 可以更容易记住:
    哪些 canonical 区域更容易出现细节误差、边界误差或局部非刚性。

tri 的作用是增强部件专家对局部空间细节的表达，
不是增加运动建模分支。
```

当前 tri 的边界：

```text
它本质上还是静态空间特征场。
所以 tri 更像“空间记忆增强”，
而不是“根据运动强弱决定哪里更该动”的模块。
```

## 2026-08-06 tri 如何学到更强的运动表达

```text
tri 本身不直接建模“运动”，
它是把 canonical 空间的可学习局部特征 f_tri 注入到非刚性 MLP / Part-MoE expert 里，
让网络更容易学到更复杂的位移、旋转、缩放残差。

关键链路:
    query_xyz=means3D
    -> TriPlaneFeature.sample
    -> f_tri
    -> concat 到 x_emb / pose_feat / seq_pose_feat / seq_xyz_feat
    -> 非刚性 MLP / Part expert
    -> d_xyz / d_rotation / d_scaling
```

代码里的核心点：

```text
1) TriPlaneFeature.forward
    对同一个 canonical 坐标 query_xyz，
    分别在 XY / XZ / YZ 三个平面采样，再平均得到 f_tri。
    这让 tri 变成一个连续的 3D 空间特征场，而不是离散表格。

2) NonrigidDeformer.__init__
    use_tri=True 时创建 TriPlaneFeature，
    并把 shared MLP 第一层输入维度加宽 tri_plane_dim。
    这表示 tri 不是旁路分支，
    而是直接参与运动残差的特征建模。

3) NonrigidDeformer.forward
    先组装基础条件:
        x_emb + pose_feat + seq_pose_feat + seq_xyz_feat
    再采样 tri_features，
    最后 concat 进 features。
    之后输出的 d_xyz / d_rotation / d_scaling 都建立在这个更强的条件上。

4) Part-MoE 路径
    当 part_moe_active 后，forward_tri 会把带 tri 的 features 送进每个 part expert。
    也就是说:
        同一份 tri 特征会被不同 part expert 按各自职责重新解释，
        从而学到 part-specific 的运动残差。

5) renderer 里用的是 canonical 的 means3D
    所以 tri 查到的是“标准空间位置”的特征，
    同一个位置跨帧一致，梯度会在这个 canonical 空间里累积。
    这就是它能变成“空间记忆”，进而帮助运动表达的原因。
```

一句话：

```text
tri 通过给非刚性网络增加 canonical 空间的局部记忆，
让 part expert 在同样的 pose / seq 条件下，
更容易区分边界、局部非刚性、细小位移和部件内部差异，
因此表现成“更强的运动表达”。
```

## 2026-08-06 canonical 局部特征场是什么

```text
canonical 局部特征场就是一个函数:
    f_tri = F(query_xyz)

这里的 query_xyz 不是当前帧变形后的点，
而是 canonical 空间里的坐标，也就是标准姿态下的坐标。

在代码里它对应:
    TriPlaneFeature.forward(query_xyz)
    -> 在 XY / XZ / YZ 三个平面采样
    -> 得到一个 feature 向量 f_tri

所以它不是“每个点一个固定编号”，
而是“空间中任意位置都能查询到一个特征”。
这个特征在同一个 canonical 位置上，跨帧基本一致，
因此像一个可学习的空间记忆。
```

## 2026-08-06 它是不是最开始的标准空间特征

```text
不是“最开始那一版固定特征”。

更准确地说：
    1. 坐标系是 canonical 的，固定不变。
    2. 特征值 f_tri 是训练中学出来的，会不断更新。
    3. 训练初期 tri plane 参数是 0 初始化，
       所以一开始几乎不提供额外信息。

所以它学到的不是“某个初始时刻的标准空间特征”，
而是“在 canonical 空间这个坐标系下，哪些位置应该对应什么特征”。
```

## 2026-08-07 相关论文

```text
这种“在坐标处查询可学习特征场，再送入小 decoder / MLP”的做法是常见路线。

接近的代表：
    EG3D: 三平面特征 + 小型隐式 decoder
    K-Planes: 将三平面推广到空间/时间等高维场
    TriHuman: 面向人体的 motion-conditioned tri-plane
    TE-NeRF: 把 triplane 特征和 SMPL 关联起来做人像渲染

所以 SeqAvatar 里的 tri 不是孤立设计，
而是把通用 tri-plane feature field 迁移到人体 part expert 上。
```

## 2026-08-07 当前 tri 和相关论文的异同

```text
当前 tri 的定位:
    不是完整三平面 NeRF / GAN / 人体生成表示。
    而是在 SeqAvatar + part_moe_leg 的 non-rigid residual 分支里，
    给每个 canonical Gaussian 坐标查询一个可学习三平面特征 f_tri，
    再 concat 到 MLP / part expert 输入。

共同点:
    都使用 XY / XZ / YZ 平面来编码 3D 空间。
    都是在 3D 坐标处采样 feature，而不是重新采样点。
    都把采到的 feature 送给后续网络解码。

和 EG3D:
    相同:
        三个正交平面 + 小 decoder/MLP。
    不同:
        EG3D 的 tri-plane 是生成器 backbone 从 latent 生成的场，
        主要解码 density/color 做 3D-aware image synthesis。
        当前 tri 是单序列优化得到的可学习参数，
        输出不是 density/color，而是帮助预测 Gaussian 的 d_xyz/d_rotation/d_scaling。

和 K-Planes:
    相同:
        都是显式平面特征场，坐标查询后再解码。
    不同:
        K-Planes 推广到 d 维，动态场会有空间/时间等多组平面，
        目标是 radiance field 重建。
        当前 tri 只有 3D canonical 空间三平面，
        没有时间平面，时间/运动仍靠 SeqAvatar 原来的 pose/seq 条件。

和 TriHuman:
    相同:
        都是人体场景里的 tri-plane，
        都利用 canonical / undeformed 空间查询特征。
    不同:
        TriHuman 会根据 skeletal motion 生成 motion-conditioned tri-plane，
        并用于 density/color 的人体 NeRF 渲染。
        当前 tri 是静态 canonical tri-plane，
        不随 pose 生成，只作为 non-rigid deformation expert 的输入增强。

和 TE-NeRF:
    相同:
        都把 triplane 和人体/SMPL 先验结合。
    不同:
        TE-NeRF 更偏 SMPL 对齐的 NeRF density/artifact reduction。
        当前 tri 不把特征绑定到 SMPL 顶点，也不预测密度，
        而是在 Gaussian 坐标上采样，服务于 SeqAvatar 的残差形变预测。

一句话:
    这些论文把 tri-plane 当作主要的 3D/4D 表示或渲染表示；
    当前 tri 把 tri-plane 当作 part_moe_leg 的局部空间条件增强。
```

## 2026-08-07 tri 到底学的是哪个空间的特征

```text
当前 tri 学的是:
    Gaussian 当前 canonical 坐标空间里的可学习特征。

不是:
    当前相机空间 / posed frame 空间 / SMPL source frame 空间。

代码顺序:
    gaussian_renderer/__init__.py
        means3D = pc.get_xyz[None]
        query_xyz=means3D 传给 non_rigid_deformer

    nets/mlp_delta_non_rigid.py
        TriPlaneFeature(query_xyz)
        在 query_xyz 的 XY / XZ / YZ 三个投影位置采样 f_tri

    之后:
        features = concat(..., f_tri)
        non-rigid MLP 输出 d_xyz / d_rotation / d_scaling
        means3D = means3D + d_xyz
        再 coarse_deform_c2source 到当前帧 posed space

所以 tri 的坐标发生在 SMPL/LBS 之前。
它描述的是 canonical Gaussian 空间中不同位置的局部特征，
用于帮助预测从 canonical 到当前帧前的非刚性 residual。

注意:
    pc.get_xyz 是可训练的 canonical Gaussian 坐标，
    densify / optimize 后会更新。
    因此 tri 不是初始点云空间的固定特征，
    而是当前优化中的 canonical Gaussian 空间特征。
```

## 2026-08-07 canonical Gaussian 空间是不是第 0 步初始化点

```text
不是完全等同于第 0 步初始化高斯点。

第 0 步初始化点云只是 canonical Gaussian 空间的起点:
    create_from_pcd(...)
        self._xyz = nn.Parameter(fused_point_cloud)

训练中 self._xyz 会继续变化:
    training_setup(...)
        optimizer 里有 xyz 参数组
    optimizer.step()
        会更新 self._xyz
    densify / clone / split / prune
        会新增、复制、拆分、删除 Gaussian

所以 tri 查询的 query_xyz=pc.get_xyz[None] 是“当前迭代的 canonical Gaussian 坐标”，
不是固定的 iteration 0 坐标。

可以这样理解:
    坐标系: canonical 空间，始终不变。
    点的位置: 从初始化点云开始，但训练中会被优化和 densify 改变。
    tri 特征: 在当前这些 canonical 坐标处查询并学习。
```

## 2026-08-08 canonical 位置是不是当前帧已经形变后的位置

```text
不是。

这里的 canonical 位置指的是:
    形变前的模板/标准姿态空间里的 Gaussian 坐标。

当前帧已经形变后的坐标要在后面经过:
    non-rigid residual d_xyz
    -> coarse_deform_c2source / SMPL-LBS

之后才得到。

所以 tri 查询时用的 query_xyz=pc.get_xyz，
对应的是“形变前的 canonical Gaussian 位置”，
不是“当前帧已经形变完成的位置”。
```

## 2026-08-08 canonical 坐标值会不会随着迭代变化

```text
会变化。

要区分两件事：
    1. 坐标系不变:
        还是 canonical / 形变前的空间。
    2. 坐标值会变:
        self._xyz 是可训练参数，
        optimizer.step() 会更新它，
        densify / clone / split / prune 也会改变点的位置和数量。

所以 tri 每次查到的不是固定的第 0 步坐标，
而是“当前迭代里，canonical 空间下这些 Gaussian 的位置”。
```

## 2026-08-08 为什么 tri 带来的提升偏小

```text
当前 tri 的提升偏小，主要因为它是“补充空间记忆”，不是“改主运动机制”。

几个直接原因：
    1. part_moe_leg + pose / seq 条件本来就已经能解释大部分运动，
       tri 只是在局部空间细节上补一点信息。

    2. tri 是静态 canonical 空间特征场，
       不显式看速度、加速度、刚性强弱，
       所以对真正的动态难点帮助有限。

    3. tri 只在 non-rigid MLP 第一层 concat 一次，
       很容易被后面的隐藏层“消化”掉，
       没有形成强制使用的路径。

    4. tri 是所有 part 共享的一张空间表，
       没有天然的部件语义，
       对某些 part 只能带来很弱的偏置。

    5. Gaussian 本身已经有 per-point learnable xyz / scaling / rotation / opacity，
       tri 再加一张共享空间记忆，增量自然有限。

    6. 后面还有 SMPL/LBS coarse deformation，
       tri 的改进是加在 residual 上，最终效果会被后续变形链路部分稀释。
```

## 2026-08-08 如果想减弱 LBS/SMPL 对 tri 的稀释

```text
当前 tri 只在 coarse_deform_c2source 之前提供 canonical residual。
如果想让 tri 更不容易被后续骨骼链路稀释，优先做两种改法：

1. 在 SMPL/LBS 之后再加一个 posed-space tri residual
    x_can -> tri + non-rigid -> LBS -> x_posed
    然后再用 tri / part / motion 预测一个很小的 d_post。

    final = x_posed + alpha_post * d_post

    这样 tri 的信息直接作用在最终输出上，不会再被后续骨骼变换吸收。

2. 用 tri 去调 LBS 本身，而不是只修 residual
    让 tri 预测 delta skinning weights / delta pose correction / delta joint transform。

    这样 tri 参与的是“怎么做骨骼变形”，不是“骨骼变完以后再补一点”。

实践上更稳的顺序：
    先做 post-LBS residual branch，零初始化，保持初始与原模型一致。
    如果它有效，再考虑 tri-conditioned delta LBS weights。
```

## 2026-08-08 tri 和部件结合时哪个更好实现

```text
更好实现的是:
    post-LBS residual + part-aware tri fusion

原因很直接：
    1. 现有代码已经把 part_label / part_conf 传进 non_rigid_deformer。
    2. tri 也已经在 non-rigid 分支里可用。
    3. 只要在 coarse_deform_c2source 之后再加一个小 residual 分支，
       就能让 tri 和 part 一起作用，不需要重写 LBS 链路。

不建议先做的：
    tri-conditioned delta LBS weights

因为这要动到:
    coarse_deform_c2source
    skinning weights 归一化
    pose transform 组合逻辑
    与原 SMPL/LBS 的兼容性

实现难度和调试风险都高很多。

所以如果目标是“tri + 部件 + 更少被 LBS 稀释”，
最稳的是：
tri + part_moe_leg + post-LBS part residual
```

## 2026-08-08 新方案：part-aware deformation budget router

```text
完全抛弃旧 tri_part:
    shared tri + part-conditioned FiLM
这个分支不再继续。

新思路不是调 tri feature，
而是让部件信息直接决定“哪里该给更多形变预算、哪里该更稳”。

核心模块：
    1. PartStatsEncoder
        对每个 part 聚合当前 Gaussian 的 motion / boundary / confidence 统计，
        得到 part token z_part。

    2. PartBudgetHead
        输入:
            x_emb_i
            z_part_i
            motion_i
            boundary_i
        输出:
            b_i in [0,1]
        b_i 表示这个点应该更偏向稳定约束还是更偏向强形变能力。

    3. PartAdapter
        在 non-rigid MLP 的 hidden state 上加一个瓶颈 residual adapter。
        不是只改输入，而是改中间表征。

    4. Budget-mixing
        h_i = h_shared_i + b_i * Adapter(h_shared_i, z_part_i)
        d_i = Head(h_i)

    5. Light regularization
        让高 motion / 边界点更容易拿到大 budget，
        稳定主体区域更容易拿到小 budget。

为什么比 tri_part 更值得做：
    它不要求 shared tri-plane 去承载 part 语义，
    也不只是给 tri feature 做 FiLM。
    它直接把 part info 变成“形变能力分配器”，
    更贴近你要的:
        哪些地方需要更强形变能力
        哪些地方需要更稳定约束

每步可验：
    1. PartStatsEncoder 只做聚合，不改输出，先查 masked mean 对不对。
    2. PartBudgetHead 只打印 b_i，不接入主干，先看分布是否合理。
    3. PartAdapter 零初始化，先验证 b_i=0 时输出和 part_moe_leg 一致。
    4. 打开 budget-mixing 后，看高 motion / 边界点是否真的拿到更大 b_i。
```

## 2026-08-08 这版 budget router 会不会太弱

```text
如果 PartBudgetHead 最后只输出一个标量 b_i，
那它是能用部件信息的，但强度仍然偏弱。

原因：
    1. 标量 budget 只能控制“多一点/少一点”，
       不能表达“这个 part 该偏刚性、那个 part 该偏非刚性、边界该怎么过渡”。

    2. 如果 part 信息只进一次 PartStatsEncoder，
       网络很容易把它当成软条件，而不是必须使用的结构约束。

    3. part_moe_leg 本身已经有 part routing，
       所以单纯再加一个标量 gate，新增语义仍然有限。

更强的做法是：
    1. budget 不是 1 个标量，而是 2-3 维向量:
        rigid_budget / nonrigid_budget / boundary_budget

    2. part info 不只调一个 Adapter，
       而是同时调 hidden state、输出 head、以及 rigid/nonrigid 混合系数。

    3. 每个 part 共享主干，但有小型 part-specific low-rank adapter，
       这样部件信息会真正改动表征，而不是只改一个缩放系数。

结论：
    这版思路比 tri_part 强很多，
    但如果只停在“标量 budget”，仍然偏软；
    要真正利用部件信息，至少要把 part 信息用于
    “容量分配 + 刚柔混合 + 小型 part-specific adapter” 三处。
```

## 2026-08-08 分阶段可验证版本：part-aware deformation router v2

```text
目标:
    不再做 tri_part 的 part-conditioned FiLM。
    改成把部件信息用于“刚性 / 非刚性 / 边界”三路形变预算分配，
    并且每一步都能单独验对错。

Stage 0: PartStats only
    对每个 part 聚合当前 Gaussian 的统计量:
        motion_mean / motion_std
        boundary_mean / boundary_std
        conf_mean
        count
    输出 part token z_part。
    这一阶段不改变任何网络输出，只做日志和单元测试。

Stage 1: PartRouter only
    输入:
        x_emb_i
        z_part_i
        motion_i
        boundary_i
    输出三维 budget:
        g_i = [g_rigid, g_nonrigid, g_boundary]
    先只打印 g_i，不接入主干。
    验证:
        高 motion / 边界点的 g_nonrigid 应更高；
        稳定主体的 g_rigid 应更高；
        空 part 不出 NaN。

Stage 2: Hidden adapter, zero-init
    h_shared_i = BaseMLP(features_i)
    h_i = h_shared_i + g_nonrigid * Adapter(h_shared_i, z_part_i)
    Adapter 最后一层零初始化。
    当 g_nonrigid=0 或 Adapter 输出为 0 时，
    输出必须和 part_moe_leg 完全一致。

Stage 3: Branch split
    用 g_i 控制三路输出混合:
        d_i = d_base_i
            + g_nonrigid * d_extra_nonrigid_i
            + g_rigid    * d_extra_rigid_i
    这里的 extra 分支都由同一 shared backbone 派生，
    但使用不同小 head。
    验证:
        手动固定某个 part 的 g 值，
        对应 part 的 rigid / nonrigid 分支响应要按预期变化。

Stage 4: Post-LBS residual
    在 coarse_deform_c2source 之后再加一个很小的 part residual:
        x_final = x_lbs + g_boundary * d_post_i
    这样部件信息不会完全被 LBS 吞掉。
    验证:
        d_post 零初始化时和原模型一致；
        只打开某个 part 的 d_post 时，只有该 part 的最终位置变化。

为什么这版比旧 tri_part 更强:
    1. 部件信息不再只调 tri feature，而是直接分配形变预算。
    2. 不是一个标量 gate，而是 rigid / nonrigid / boundary 三维信号。
    3. 既改中间表征，也改最终输出，还能在 LBS 后补一刀。
    4. 每个阶段都能独立验证，不容易一上来就把链路写坏。

建议实现顺序:
    1. Stage 0 + Stage 1 只做观测。
    2. Stage 2 保持零初始化，先过数值等价测试。
    3. Stage 3 再打开 branch split。
    4. Stage 4 最后补，专门对抗 LBS 稀释。
```

## 2026-08-08 part_budget Step 0：开关和日志层接通

本步只做接线，不改变任何网络输出。

已完成：

```text
1. 新增独立开关:
    --use_part_budget
    --part_budget_alpha
    --part_budget_start_iter
    --part_budget_warmup
    --part_budget_hidden_dim
    --part_budget_token_dim

2. 三个脚本都加入 part_budget 模式:
    scripts/exps_dnarendering.sh
    scripts/exps_zjumocap.sh
    scripts/exps_i3dhuman.sh

3. part_budget 独立日志目录:
    /media/image/mxz/human/SeqAvatar/logs/budget

4. part_label/common.py 已把 use_part_budget 映射成独立日志模式 budget，
   不再混进 part / tri 日志命名。
```

验证：

```text
bash -n scripts/exps_dnarendering.sh
bash -n scripts/exps_zjumocap.sh
bash -n scripts/exps_i3dhuman.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile
    arguments/__init__.py
    part_label/common.py
```

## 2026-08-08 part_budget Step 1：网络层接通并验证 no-op

已完成：

```text
1. NonrigidDeformer 增加了 part_budget 三个子模块:
    PartStatsEncoder
    PartBudgetRouter
    PartBudgetAdapter

2. part_budget 只挂在 part_moe_leg 路径上:
    默认不影响 original / tri / tri_part / tri_gate

3. 训练与评估都接入了 part_budget_alpha_scale:
    train.py 每步按 start_iter / warmup 计算
    render.py 固定为 1.0

4. gaussian_renderer 会把 part_budget_alpha_scale 传入 non-rigid 分支。
```

验证结果：

```text
1. bash -n scripts/exps_dnarendering.sh
   bash -n scripts/exps_zjumocap.sh
   bash -n scripts/exps_i3dhuman.sh
   通过。

2. py_compile 通过:
    nets/mlp_delta_non_rigid.py
    scene/gaussian_model.py
    gaussian_renderer/__init__.py
    train.py
    render.py
    part_label/common.py
    arguments/__init__.py

3. CPU 小张量对照测试:
    part_budget=False vs part_budget=True + alpha_scale=0
    输出完全一致，max diff = 0.0

4. part_budget=True + alpha_scale=1 时:
    budget_mean = [0.33333334, 0.33333334, 0.33333334]
    budget_std = [0.0, 0.0, 0.0]
    feature_delta_norm = 0.0
    说明当前初始化确实是干净的 no-op。
```

## 2026-08-08 part_budget DNA 运行记录 0：第一次启动 OOM

第一次在 DNA 上跑 `part_budget` 时，训练阶段在加载训练相机时触发 CUDA OOM。

现象：

```text
torch.cuda.OutOfMemoryError
发生在 scene/cameras.py 里把 original_image 放到 cuda 时
```

原因判断：

```text
DNA 的训练相机加载在 cuda 上太吃显存，
即使跳过测试相机，训练阶段仍然会把训练相机 image 放到 GPU。
```

处理：

```text
part_budget 的 DNA 启动脚本改成默认:
    IMAGE_DATA_DEVICE=cpu
    SKIP_LOAD_TEST_CAMERAS=1
这样训练和最终 render 分开走，先把主训练跑起来。
```

## 2026-08-08 part_budget DNA 运行中状态

当前运行：

```text
PID: 1674574
序列: 0044_11
模式: part_budget
GPU: 3
启动时间: 2026-08-08 19:09:03 CST
```

实时状态：

```text
训练已正常启动并持续推进，没有再次触发 OOM。
当前进度大约在 20980 / 25000 iter，约 84%。
日志中 loss / ssim / lpips 都在正常波动，没有异常中断。
```

## 2026-08-08 part_budget 序列完成情况

```text
已完成序列:
    0044_11

当前进行中:
    0051_09

0044_11 最终结果:
    PSNR 33.001909764607746
    SSIM 0.9782760689655939
    LPIPS 0.021070281501548986
```

## 2026-08-09 part_budget DNA 全部完成

```text
已完成序列:
    0044_11
    0051_09
    0206_04
    0813_05
    0007_04
    0019_10

各序列最终指标:
    0044_11: PSNR 33.001909764607746  SSIM 0.9782760689655939  LPIPS 0.021070281501548986
    0051_09: PSNR 28.76159183184306   SSIM 0.9719492380817731  LPIPS 0.030244823452085255
    0206_04: PSNR 31.56024735768636   SSIM 0.9704658562938372  LPIPS 0.033273935597389934
    0813_05: PSNR 36.2131846110026    SSIM 0.9873562256495158  LPIPS 0.017797200203252334
    0007_04: PSNR 29.614092858632404  SSIM 0.9587977856397628  LPIPS 0.044322934669132036
    0019_10: PSNR 35.4025904973348    SSIM 0.9814229314525922  LPIPS 0.020552238690045972

序列平均值:
    PSNR 32.4256028201845
    SSIM 0.974711351013846
    LPIPS 0.02787690235224242

全局日志:
    /media/image/mxz/human/SeqAvatar/logs/budget/20260808_190903_DNA-Rendering_part_budget.log
```

## 2026-08-09 part_budget 与 part_moe_leg 对比

对照基线：

```text
part_moe_leg 完整 DNA 六序列日志:
    /media/image/mxz/human/SeqAvatar/logs/part/20260623_180431_DNA-Rendering_part_moe_leg.log
```

平均指标对比：

```text
part_moe_leg:
    PSNR       32.39075858328077
    SSIM       0.974622116320663
    LPIPS*1000 27.993953922608245

part_budget:
    PSNR       32.4256028201845
    SSIM       0.974711351013846
    LPIPS*1000 27.87690235224242

part_budget - part_moe_leg:
    PSNR       +0.034844236903730064
    SSIM       +0.00008923469318300459
    LPIPS*1000 -0.11705157036582398
```

逐序列变化：

```text
0044_11: PSNR +0.0013388633728013133  SSIM +0.00006509721279146508  LPIPS*1000 -0.007489083024361104
0051_09: PSNR +0.06268736521403184    SSIM +0.00026364574829740306  LPIPS*1000 -0.5824100614214949
0206_04: PSNR +0.04128150939941477    SSIM +0.0001388515035311011   LPIPS*1000 -0.2892857417464284
0813_05: PSNR +0.010714499155682233   SSIM +0.000016412138938881604 LPIPS*1000 -0.11472008967151198
0007_04: PSNR +0.04697521527608117    SSIM -0.00009343475103384957  LPIPS*1000 +0.42052736195425516
0019_10: PSNR +0.046067969004312204   SSIM +0.00014483630657191604  LPIPS*1000 -0.12893180828541656
```

判断：

```text
part_budget 相比 part_moe_leg 有一致方向的小幅正收益，
但平均 PSNR 只提升 0.0348 dB，SSIM 只提升 0.000089，LPIPS*1000 只降低 0.117。
这不能算明显提高，更像轻微增益或随机波动边界内的收益。

0007_04 虽然 PSNR 上升，但 SSIM 和 LPIPS 变差，说明提升不完全稳。
```

## 2026-08-09 part_budget 有效性复查

复查原因：

```text
用户追问收益弱是因为没有发挥作用，还是参数设置保守。
```

关键发现：

```text
这轮 20260808_190903 的 DNA 六序列结果不能作为有效 part_budget 结果。

原因:
    六个输出目录的 cfg_args 都显示:
        use_part_budget=False

    训练日志中也没有:
        [PART_BUDGET]
        [PartBudget Status]
        [PartBudget Stats]

    之前运行时 ps 看到的真实 train.py 命令也没有:
        --use_part_budget
        --part_budget_alpha
        --part_budget_start_iter
        --part_budget_warmup
        --part_budget_hidden_dim
        --part_budget_token_dim

因此本轮虽然日志总标题写的是 part_budget，
但实际训练路径是 part_moe_leg，而不是 part_budget。
```

修正判断：

```text
当前这组 +0.0348 dB / +0.000089 SSIM / -0.117 LPIPS*1000
不能解释为 part_budget 的真实收益。

它更可能只是 part_moe_leg 重新跑一次造成的随机波动。

所以现在不能判断 part_budget 是“没发挥作用”还是“参数太保守”；
因为这轮有效分支根本没有打开。
```

## 2026-08-09 part_budget 重新启动

修复内容：

```text
已把 --use_part_budget 的注入位置从 tri_enabled 分支中移出，
现在 part_budget 单独启动时也会真正传入训练命令。
```

当前重跑：

```text
GPU: 3
启动方式: tmux part_budget_dna_fix_20260809
模式: bash scripts/exps_dnarendering.sh part_budget
额外环境:
    IMAGE_DATA_DEVICE=cpu
    SKIP_LOAD_TEST_CAMERAS=1
    KEEP_SPLIT_PART_LOGS=1
```

## 2026-08-09 part_budget 有效重跑前修复

用户要求：

```text
打开 part_budget 重新跑并给最后评价指标，用卡 3。
```

接续状态：

```text
上一轮有效重跑已经真正传入 --use_part_budget，
日志中出现:
    [PART_BUDGET] enabled=True

但在 0044_11 约 10000/25000 iter，即 Part-MoE/part_budget 激活后崩溃。
```

崩溃原因：

```text
nets/mlp_delta_non_rigid.py:
    PartStatsEncoder._normalize_inputs

训练中 pc.get_part_conf 返回的是一维 [N]。
原代码只处理了 [B, N] / [B, N, 1]，
没有把 [N] 扩成 [B, N, 1]。

因此 boundary_score 仍是一维，执行:
    boundary_score.mean(dim=1, keepdim=True)
触发:
    IndexError: Dimension out of range
```

修复内容：

```text
nets/mlp_delta_non_rigid.py
    PartStatsEncoder._normalize_inputs:
        motion_strength / part_conf 均支持 [N] 输入，
        统一扩成 [B, N, 1]。

    NonrigidDeformer._normalize_budget_motion
    NonrigidDeformer._normalize_budget_conf:
        同样补齐 [N] 输入处理，避免后续 router/gate 处再遇到同类问题。
```

已验证：

```text
bash -n scripts/exps_dnarendering.sh
bash -n scripts/exps_i3dhuman.sh
bash -n scripts/exps_zjumocap.sh

PYTHONDONTWRITEBYTECODE=1 python -m py_compile:
    nets/mlp_delta_non_rigid.py
    train.py
    gaussian_renderer/__init__.py
    scene/gaussian_model.py
    arguments/__init__.py

CPU 小张量测试:
    part_label: [N]
    part_conf:  [N]
    motion:     [N]

    PartStatsEncoder 输出:
        part_token [1, 7, 32]
        part_stats [1, 7, 7]

    NonrigidDeformer(use_part_budget=True).apply_part_budget 输出:
        features [1, 12, 63]

结论:
    一维 part_conf / motion_strength 已能正确进入 part_budget 分支。
```

## 2026-08-09 part_budget 弱收益原因分析

代码状态：

```text
part_budget 运行是健康的。
    1. 已跑完 DNA 六序列。
    2. 没有新报错，也没有 OOM。
    3. Budget 分布从均匀变成了轻微偏置，但 entropy 仍然接近 1.09。
```

关键日志：

```text
11000:
    mean = [0.3333333432674408, 0.3333333432674408, 0.3333333432674408]
    entropy = 1.098612

12000:
    mean = [0.3379809856414795, 0.3076769709587097, 0.3543420732021332]
    feature_delta_norm = 0.108138

25000:
    mean = [0.3271176218986511, 0.283312052488327, 0.3895703852176666]
    std = [0.0010996104683727026, 0.005012987181544304, 0.00608061021193862]
    entropy = 1.090006
    feature_delta_norm = 0.435912
```

为什么提升小：

```text
1. 这是 feature reweighting，不是结构性改网络。
   它只改 non-rigid 输入，不改 part expert 结构、densify、LBS 或 part 分配。

2. router 还偏保守。
   entropy 一直很高，说明三路预算没有明显分工。

3. motion signal 被 batch 内标准化压平了。
   `_normalize_budget_motion()` 做了 log1p + z-score。
   日志里 motion_mean 长期接近 0，这是这个设计的直接结果。

4. part_conf 是静态先验。
   它来自一次性 Gaussian-part confidence，不是随帧变化的动态信号。

5. 后续 part experts 仍可吸收预算变化。
   所以 budget 的净贡献会被稀释。
```

可调空间：

```text
优先参数:
    part_budget_start_iter 更早
    part_budget_warmup 更短
    part_budget_alpha 更大
    part_budget_hidden_dim / token_dim 更大

更关键的改法:
    1. 保留原始 motion 通道，不只喂 z-score 后的 motion。
    2. 给 part_conf 加动态分量。
    3. 让 budget 更直接影响 part experts，而不是只改输入特征。
```

## 2026-08-09 代码迁移建议

用户问题：

```text
想把当前项目迁移到另一台服务器。
另一台服务器已有数据集，也有添加 part_budget 之前的代码。
询问是否适合用 git。
```

当前仓库状态：

```text
当前分支:
    motion

远程:
    origin  https://github.com/merlin-0728/seqavatar.git
    3dhgsseq https://github.com/merlin-0728/3dhgsseq.git

当前改动:
    17 个 tracked 文件修改。
    没有未跟踪文件。

判断:
    适合用 git 迁移。
    优先推荐 commit + push/pull。
    如果不想推远程，使用 git bundle 或 git diff patch。
```

推荐迁移方式：

```text
方案 A:
    在当前服务器 commit 到独立分支并 push origin。
    新服务器 fetch/pull 该分支。

方案 B:
    如果不方便推 GitHub，用 git bundle。
    bundle 可以保留 commit 和分支信息，比裸 patch 稳。

方案 C:
    只做一次性迁移可用 git diff --binary 生成 patch。
    但 patch 不保留 commit 历史，且目标代码版本差异大时更容易冲突。
```

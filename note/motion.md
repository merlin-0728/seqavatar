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

# SeqAvatar 第二创新点 Motion 笔记

## 2026-06-29 当前目标

本对话用于决定以 SeqAvatar 为基线实验的第二个创新点。第二创新点初步确定和运动建模有关，计划通过独立消融模块添加。

必须遵守：

- 不影响已有 baseline 实验和流程。
- 不影响第一个创新点 Part-MoE 相关实验和流程。
- 隔离三类代码路径：原始 baseline、Part-MoE、第二创新点 Motion/MSTI。
- 每次后续对话都要先参考本文件，再根据新结论更新本文件，避免重复犯同样错误。

## 已有第一创新点

第一创新点是代码中和 Part-MoE 相关的内容，包括：

- `use_part_moe`
- `part_moe_start_iter`
- `part_moe_warmup`
- `part_moe_global_keep`
- `part_label_schema`
- `part_grouping_mode`
- `part_max_smpl_dist`

后续 Motion/MSTI 改动不能改变 Part-MoE 默认行为，也不能让 Part-MoE 实验必须依赖 Motion/MSTI 参数。

## 第二创新点候选方向

当前候选方向围绕 SeqAvatar 的 motion condition：

- 原始 SeqAvatar 已有粗骨骼 motion `delta_pose / delta P`。
- 原始 SeqAvatar 已有细粒度 SMPL vertex motion `delta xyz / V`。
- 第二创新点希望改善大时间跨度 motion condition 下的非刚性变形预测。

初步模块名：

- `MSTI`
- `Motion-aware SMPL Temporal Interpolation`
- 消融版本可叫 `MSTI`

## 当前最小消融设定

先做不判断运动强度的消融：

- 不计算 motion score。
- 不设置 high-motion threshold。
- 对每个真实历史跨度 `t-s -> t` 都插入一个虚拟 SMPL 中点。
- 虚拟中点只用于构造额外 motion condition。
- 不新增 RGB 图片。
- 不新增 mask。
- 不新增 camera。
- 不输出中间帧。
- 监督仍然只使用真实帧 `t` 的图像。

原始 condition：

```text
full_s: t-s -> t
```

MSTI-all condition：

```text
full_s: t-s -> t
sub1_s: t-s -> mid_virtual
sub2_s: mid_virtual -> t
```

例如 I3D 当前 3 个尺度 `[42, 33, 24]`：

```text
baseline channel: 3
MSTI-all channel: 3 * 3 = 9
```

ZJU 当前常见 2 个尺度 `[6, 3]`：

```text
baseline channel: 2
MSTI-all channel: 2 * 3 = 6
```

DNA 当前 3 个尺度 `[3, 2, 1]`：

```text
baseline channel: 3
MSTI-all channel: 3 * 3 = 9
```

## 重要实现约束

不要直接把脚本里的 `time_step_num` 从 3 改成 9。

原因：

- 当前 `time_step_num` 同时控制 `generate_time_steps()` 生成多少个原始采样尺度。
- 当前 `time_step_num` 也控制 `SeqPoseEncoder / SeqXYZEncoder` 的输入 channel 数。
- 如果直接改成 9，会先生成 9 个原始尺度，再插虚拟中点，可能变成 27 个 condition channel。

正确设计应拆开：

```text
base_time_step_num: 原始采样尺度数量
cond_time_step_num: 输入 encoder 的 condition channel 数量
```

默认关闭 MSTI：

```text
cond_time_step_num = base_time_step_num
```

开启 MSTI-all：

```text
cond_time_step_num = base_time_step_num * 3
```

## 当前代码落点

最自然落点是：

- `scene/dataset_readers.py`
- 函数：`get_seq_pose_xyz_cond`

原因：

- 当前 `seq_pose_conds` 和 `seq_xyz_conds` 都在这个函数里预计算。
- renderer 只通过 `pose_id` 从 `pc.cond_dict` 读取 condition。
- 只要不创建新的 `CameraInfo`，虚拟中间 SMPL 不会成为输出帧。

相关路径：

- `scene/dataset_readers.py`
- `scene/__init__.py`
- `gaussian_renderer/__init__.py`
- `nets/mlp_delta_non_rigid.py`
- `arguments/__init__.py`
- `scripts/exps_i3dhuman.sh`
- `scripts/exps_zjumocap.sh`
- `scripts/exps_dnarendering.sh`

## 虚拟中点方案

Pose 中点：

- 当前项目已有 `axis_angle_to_matrix` 和 `matrix_to_axis_angle`。
- 可以用 SO(3) log/exp 思路实现半步旋转。
- 不需要新增依赖。

XYZ 中点有两种：

1. 低风险消融版：

```text
mid_xyz = 0.5 * former_xyz + 0.5 * cur_xyz
```

优点：接入简单，改动小。  
缺点：`sub1/sub2` 的 xyz velocity 可能和 full velocity 高度重复，创新解释较弱。

2. 严谨论文版：

```text
mid_pose -> SMPL forward -> mid_xyz
```

优点：几何上更合理。  
缺点：不同数据集 reader 需要暴露 SMPL/SMPLX model、shape、translation 等信息，改动更大。

当前建议先做低风险消融版，再决定是否升级为 SMPL forward 版。

## 已确认风险点

1. `seq_xyz_conds` 归一化疑似已有问题

当前 `dataset_readers.py` 中 `seq_xyz_conds` 的归一化使用循环结束后的 `time_step`，不是每个 channel 自己的时间间隔。MSTI 加入 `full/sub1/sub2` 后，每个 channel 的 `dt` 不同，这个问题会更明显。

后续实现 MSTI 时，新分支必须按每个 delta 的实际 `dt` 归一化。

2. 虚拟中点不应改变输出

MSTI 只改输入 motion condition，不改：

- `pose_id`
- `CameraInfo`
- RGB image
- mask
- camera
- render target
- loss supervision

3. 需要配置开关

必须默认关闭：

```text
use_msti_virtual_mid = False
```

开启时才进入新逻辑。

可选参数：

```text
use_msti_virtual_mid
msti_virtual_mid_alpha = 0.5
msti_mid_xyz_mode = linear / smpl_forward
```

## 运动强度版本留作下一阶段

后续 Motion-aware MSTI 可以基于 SeqAvatar 已有 `delta P` 定义运动强度：

```text
m_pose(t, s) = mean_k ||delta_p_k(t, s)||_2 / s
```

其中 `delta_p_k` 是第 k 个关节的 axis-angle 相对旋转向量，范数表示旋转角度。

但当前第一阶段消融不使用 motion score，不做 high-motion 判断。

## 2026-06-29 MSTI-lite / full 方案评审

用户提供了一个分阶段实现方案：

- 先做 `MSTI-lite`：只对最大时间跨度加 `sub1/sub2`。
- 再做 `MSTI-full`：对所有时间跨度都加 `sub1/sub2`。
- 暂时不改 renderer、loss、监督图像。

总体可行，但需要修正以下点：

1. 不能直接把脚本里的 `time_step_num` 从 `3` 改成 `5` 或 `9`

原因仍然是 `time_step_num` 当前同时控制：

- 原始采样尺度数量；
- encoder 输入 condition channel 数量。

如果 I3D 直接 `time_step_num=5`，`generate_time_steps()` 会从 `[24, ..., 42]` 中生成 5 个原始尺度，而不是保留 `[42, 33, 24]` 后添加 2 个子通道。

正确做法：

```text
base_time_step_num = 原始尺度数量
cond_time_step_num = encoder 输入 condition channel 数量
```

例如 I3D：

```text
baseline: base=3, cond=3
MSTI-lite: base=3, cond=5
MSTI-full: base=3, cond=9
```

2. 提供方案里用的是 `mid_id` 真实中间帧，不是纯虚拟中点

如果使用：

```text
mid_id = former_id + (cur_id - former_id) // 2
```

并读取 `P_mid / X_mid`，这属于 real-mid temporal subdivision。它不是 SLERP 虚拟中点。

这可以作为低风险版本，而且比线性虚拟 `mid_xyz` 更有真实运动信息；但论文表述要区分：

```text
Real-Mid MSTI: 使用真实中间 SMPL 帧
Virtual-Mid MSTI: 使用 SLERP / 插值生成虚拟中间 SMPL
```

3. `get_seq_pose_xyz_cond()` 里要对每个 `i` 都处理

当前函数不是只构造 `t-s -> t`，而是对每个序列位置 `i` 构造：

```text
cur_id    = t - i*s
former_id = t - (i+1)*s
```

因此 MSTI-lite/full 的逻辑必须嵌入当前双层循环中，而不是只处理当前帧一次。

4. `dt` 必须按每个 channel 单独计算

不能继续使用循环结束后的 `time_step` 做统一除法。

每个 channel 应该有自己的时间间隔：

```text
dt_full = cur_id - former_id
dt1 = mid_id - former_id
dt2 = cur_id - mid_id
```

边界处 `former_id == cur_id` 时需要避免除 0，可用 `max(dt, 1)`，此时 delta 通常为 0。

5. “只加 pose 不加 xyz”不是当前结构下的最小改动

当前 `SeqPoseEncoder` 和 `SeqXYZEncoder` 都使用同一个 `time_step_num`。

如果只把 `seq_pose_conds` 从 3 改成 5，而 `seq_xyz_conds` 保持 3，同时 encoder 仍共用同一个 `time_step_num`，会导致 Linear 输入维度不匹配。

可选做法：

- 要么 pose 和 xyz 一起扩展到相同 channel 数；
- 要么新增 `seq_pose_time_step_num` 和 `seq_xyz_time_step_num`，改动更大；
- 要么给未扩展的 xyz channel 做 zero padding 到同样 channel 数。

6. 不建议直接把 `seq_pose_conds / seq_xyz_conds` 改成 half

`cond_dict` 会被搬到 GPU，half 可以省显存，但当前网络 Linear 默认 float32。直接 half 可能产生 dtype mismatch。除非系统性启用 AMP 或统一 dtype，否则不要作为第一版方案。

7. lite 推荐按“最大原始尺度”判断，而不是硬编码 `s == 42`

I3D 最大尺度是 42，ZJU 可能是 6，DNA 可能是 3。为了跨数据集复用，应该用：

```text
time_step == max(time_steps.keys())
```

或者通过参数指定 `msti_lite_target_step`。

当前建议：

```text
第一版做 MSTI-lite real-mid，pose+xyz 一起扩展，base_time_step_num 不变，cond_time_step_num 单独设置。
```

## 2026-06-29 剩余实现注意点

修正版方案总体可行，但实现前还要明确以下边界：

1. 保持旧参数兼容

为了不破坏已有脚本和旧实验，建议保留现有 `time_step_num` 作为原始尺度数量，即等价于：

```text
base_time_step_num = time_step_num
```

然后新增单独的 encoder channel 参数，例如：

```text
motion_cond_time_step_num
```

或内部自动计算：

```text
none: motion_cond_time_step_num = time_step_num
lite: motion_cond_time_step_num = time_step_num + 2
full: motion_cond_time_step_num = time_step_num * 3
```

不要要求所有旧脚本立刻改名为 `base_time_step_num`，否则 baseline / Part-MoE 脚本容易被破坏。

2. 训练和渲染配置必须完全一致

`render.py` 会读取训练目录里的 `cfg_args`，但命令行参数仍可能覆盖。MSTI 相关参数必须保证 train/render 一致：

```text
use_msti
msti_mode
msti_mid_type
time_step_num/base_time_step_num
motion_cond_time_step_num
```

否则 checkpoint 中的 `SeqPoseEncoder / SeqXYZEncoder` 输入维度和 render 时构造的 condition shape 会不一致。

3. checkpoint 不能跨 MSTI 模式混用

baseline checkpoint、Part-MoE checkpoint、MSTI checkpoint 的 non-rigid deformer Linear 输入维度不同。不要用 baseline checkpoint 直接加载 MSTI 模型，反过来也不行，除非专门写兼容加载逻辑。

4. delta cache key 要区分数据来源和 motion 类型

当前 `delta_pose_xyz_cache` 使用类似 `cur-former` 的 key。实现 MSTI 后：

- full delta 可以继续复用原 key；
- real-mid sub delta 可以用真实 pair key，如 `mid-former`、`cur-mid`；
- virtual-mid delta 不是已有真实 pair，必须用单独 key，避免和 real pair 混淆。

另外 I3D train/novelview/novelpose 共用 `cond_dict` 和 `delta_pose_xyz_cache`，数值 pose id 可能跨 split 重复。后续如果发现 novelpose cond 被 train cond 复用，需要把 split 信息加入 cache/cond key。

5. early frames 的边界行为要固定

当前逻辑会 clamp 到 0：

```text
former_id = max(..., 0)
```

序列开头会出现 `former_id == cur_id` 或 `mid_id == former_id`。这时 delta 应为 0，`dt` 用 `max(dt, 1)` 避免除 0。实现后要检查这些边界不会产生 NaN。

6. real-mid 的 odd interval 会不对称

例如 `s=33` 时：

```text
dt1 = 16
dt2 = 17
```

这是可以接受的，但必须按各自 `dt` 归一化。不要假设 `dt1 == dt2`。

7. lite 的 channel 顺序必须固定并记录

建议：

```text
[full_max, sub1_max, sub2_max, full_other1, full_other2, ...]
```

full 模式建议：

```text
[full_s, sub1_s, sub2_s] for each s in original order
```

顺序一旦确定，train/render/eval 必须一致。

8. Part-MoE 与 MSTI 组合时只共享最终 features

Part-MoE 路由发生在 non-rigid deformer 拼接 feature 之后。MSTI 只改变 `seq_pose_feats / seq_xyz_feats` 的输入，不应该改变 part label 生成、part expert 初始化、densify 约束等逻辑。

9. pose delta 的定义要和原代码完全一致

当前原代码使用：

```text
delta_pose_mat = cur_pose_mat @ inverse(former_pose_mat)
```

MSTI 的 `full/sub1/sub2` 必须沿用同一方向。不要在新分支里改成 `former relative to cur`，否则 encoder 输入语义会变。

10. global translation / root 处理要一致

如果原始 `seq_pose_conds` 只包含 pose rotation delta，不额外包含 global translation，MSTI 也不要在 pose condition 里新增 translation。`seq_xyz_conds` 已经隐含 observation/world space 的 vertex motion，保持原始表达即可。

11. real-mid 的 `mid_id` 必须在同一个 split / 同一个视频序列内

不能跨 sequence、subject 或 split 取中间帧。尤其 I3D / ZJU / DNA 中 pose id 可能跨 split 重复，后续如果发现冲突，需要在 `cond_dict` / cache key 中加入 sequence 或 split 信息。

12. Virtual-Mid 版本需要重新 forward SMPL

如果后续做 Virtual-Mid，不能只插值 pose 参数就结束。需要：

```text
SLERP pose -> P_mid -> 使用一致的 beta/transl 策略 -> SMPL forward -> X_mid
```

否则 `delta_pose_sub` 和 `delta_xyz_sub` 没有可靠几何对应。Real-Mid 第一版可以避开这个问题。

13. condition channel 数必须 assert

dataset 输出和 encoder 输入前都建议检查：

```text
seq_pose_conds channel_dim == motion_cond_time_step_num
seq_xyz_conds channel_dim == motion_cond_time_step_num
```

这样能快速发现 train/render channel 不一致、lite/full 顺序不一致、或配置错误。

14. 配置应自动推导，避免矛盾手填

不要允许无提示的矛盾配置，例如：

```text
msti_mode = lite
motion_cond_time_step_num = 9
```

建议内部推导：

```text
none: motion_cond_time_step_num = time_step_num
lite: motion_cond_time_step_num = time_step_num + 2
full: motion_cond_time_step_num = time_step_num * 3
```

如果保留手填，则必须加 assert。

15. checkpoint 加载要有明确报错

如果用 baseline checkpoint 加载 MSTI 模型，或者反过来加载，应该明确报：

```text
motion_cond_time_step_num mismatch
```

不要静默跳过或 partial load，除非专门实现 ablation 初始化逻辑。

16. cache 的版本号要加 MSTI 信息

cache key 或 cache metadata 最好包含：

```text
msti_mode
msti_mid_type
dt_norm
pose_delta_version
```

未来比较 `real-mid`、`virtual-mid`、`no-dt-norm`、`with-dt-norm` 时，避免读到旧 cache。

17. full/sub channel 的数值尺度要检查

实现后要打印统计：

```text
mean/std/max of dP_full/sub1/sub2
mean/std/max of dX_full/sub1/sub2
```

尤其确认 pose delta 是否按原始设计不除 dt，xyz delta 是否按每个 channel 的 dt 归一化，避免 sub channel 数值异常。

18. Lite 的最大尺度选择要稳定

不要依赖 dict 遍历顺序来判断最大尺度。显式计算：

```text
max_step = max(time_steps)
```

输出顺序固定为：

```text
[full_max, sub1_max, sub2_max, full_other1, full_other2, ...]
```

其中 `other` 也要按原始顺序固定。

19. 确认 encoder 是否假设 time_step 顺序

当前 `SeqPoseEncoder / SeqXYZEncoder` 主要是 flatten + MLP，但仍然对 channel 顺序敏感。后续若加入 positional embedding、for-loop 或 attention，必须继续保证 train/render 的 channel 顺序完全一致。

20. 实验公平性

MSTI 从 3 channel 变为 5/9 channel 后，encoder 参数量和计算量可能增加。对比 baseline 时建议报告：

```text
参数量
显存
训练时间
FPS
```

如果需要更严格公平性，可加 `Baseline-wide` 对照：保持原始 3 channel，但把 encoder hidden dim 调到接近 MSTI 参数量。

## 当前最终落地建议

第一版实现目标固定为：

```text
Real-Mid MSTI-lite
time_step_num 保持原始含义，例如 I3D 仍为 3
motion_cond_time_step_num 自动推导为 time_step_num + 2
pose/xyz 同时扩展
per-channel dt 归一化
renderer/loss 不动
默认关闭，不影响 baseline 和 Part-MoE
```

## 2026-06-29 DNA-Rendering MSTI-lite 实现记录

本次开始在 DNA-Rendering 六个序列上加入 MSTI 消融实验，先实现并运行低风险版本：

```text
Real-Mid MSTI-lite
time_step_num = 3
minimal_time_step = 1
max_time_step = 3
seq_len = 8
motion_cond_time_step_num = 5
part_moe_enabled = 0
```

实现边界：

- baseline 默认路径仍然是 `msti_mode=none`，`motion_cond_time_step_num=time_step_num`。
- Part-MoE 参数和路径保持独立，不自动开启 MSTI。
- MSTI 只改 `seq_pose_conds / seq_xyz_conds`，不新增中间 RGB、mask、camera、CameraInfo，也不改变 render/loss 输出目标。
- DNA 脚本新增 `msti` / `msti_lite` / `real_mid_msti_lite` 模式，超参数写在 `scripts/exps_dnarendering.sh`。
- MSTI-lite 输出顺序固定为 `[full_max, sub1_max, sub2_max, full_other1, full_other2]`。
- MSTI 分支的 xyz delta 使用每个 pair 自己的实际 `dt` 归一化；baseline 路径不改，以免影响旧实验。
- 当前只支持 `msti_mid_type=real`，Virtual-Mid 仍留作后续版本。

已验证：

```text
py_compile: arguments / scene / gaussian_model / mlp_delta_non_rigid 通过
bash -n scripts/exps_dnarendering.sh 通过
git diff --check 通过
DNA-like shape test:
  baseline pose/xyz = (1,8,3,55,3) / (12,8,3,3)
  MSTI-lite pose/xyz = (1,8,5,55,3) / (12,8,5,3)
```

正式 DNA-Rendering 六序列运行：

```text
启动脚本:
  scripts/exps_dnarendering.sh msti

GPU 2:
  sequences = 0044_11 0051_09 0206_04
  RUN_TIME = 20260629_181414_msti_lite_gpu2
  global log = logs/20260629_181414_msti_lite_gpu2_DNA-Rendering_msti_lite.log
  first train pid = 73908

GPU 3:
  sequences = 0813_05 0007_04 0019_10
  RUN_TIME = 20260629_181414_msti_lite_gpu3
  global log = logs/20260629_181414_msti_lite_gpu3_DNA-Rendering_msti_lite.log
  first train pid = 73913
```

启动后状态：

```text
0044_11 已进入训练迭代，约 1200+ / 25000 iter
0813_05 已进入训练迭代，约 1200+ / 25000 iter
两条训练显存约 22GB，未出现 MSTI shape/channel 报错
```

## 2026-06-29 DNA-Rendering MSTI-lite 六序列结果

主结果取各序列落盘文件：

```text
output/DNA-Rendering/<sequence>/msti_lite/20260629_181414_msti_lite_gpu*/metrics/results_novelview_25000.json
```

六序列 novelview 指标如下，平均值为序列级简单算术平均，不做 Test 张数加权：

| Sequence | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| 0044_11 | 32.9590 | 0.977824 | 0.021593 |
| 0051_09 | 28.5970 | 0.970736 | 0.031475 |
| 0206_04 | 31.3579 | 0.969788 | 0.034011 |
| 0813_05 | 36.0313 | 0.986773 | 0.018758 |
| 0007_04 | 29.4669 | 0.958055 | 0.045600 |
| 0019_10 | 35.2625 | 0.980950 | 0.021226 |
| Mean | 32.2791 | 0.974021 | 0.028777 |

同一结果按 `LPIPS * 1000` 展示：

| Sequence | PSNR | SSIM | LPIPS x1000 |
|---|---:|---:|---:|
| 0044_11 | 32.9590 | 0.977824 | 21.5928 |
| 0051_09 | 28.5970 | 0.970736 | 31.4754 |
| 0206_04 | 31.3579 | 0.969788 | 34.0112 |
| 0813_05 | 36.0313 | 0.986773 | 18.7585 |
| 0007_04 | 29.4669 | 0.958055 | 45.6002 |
| 0019_10 | 35.2625 | 0.980950 | 21.2255 |
| Mean | 32.2791 | 0.974021 | 28.7773 |

补充说明：

- `render.py` 重新渲染后的 log 指标与 JSON 基本一致。
- render-log 序列级平均为 `PSNR=32.2956, SSIM=0.974152, LPIPS=0.028677`。
- 后续论文表格建议优先固定使用 JSON 主结果，避免 train final eval 和 render re-eval 混用。

## 2026-06-29 DNA-Rendering MSTI-lite 运行命令

单卡跑完整六序列 MSTI-lite 实验：

```bash
cd /media/image/mxz/human/SeqAvatar
GPU_id=2 bash scripts/exps_dnarendering.sh msti
```

等价绝对路径写法：

```bash
GPU_id=2 bash /media/image/mxz/human/SeqAvatar/scripts/exps_dnarendering.sh msti
```

注意：

- 如果不指定 `GPU_id`，脚本默认使用 `GPU_id=2`。
- 如果不指定 `SEQUENCES_OVERRIDE`，脚本默认依次跑六个 DNA 序列：`0044_11 0051_09 0206_04 0813_05 0007_04 0019_10`。
- `msti` 模式对应 `Real-Mid MSTI-lite`，脚本内固定 `time_step_num=3`、`motion_cond_time_step_num=5`、`msti_mode=lite`、`msti_mid_type=real`。
- 单卡会六序列串行跑；如果要和本次实验一样双卡拆分，需要开两个终端分别运行：

```bash
SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash /media/image/mxz/human/SeqAvatar/scripts/exps_dnarendering.sh msti
SEQUENCES_OVERRIDE="0813_05 0007_04 0019_10" GPU_id=3 bash /media/image/mxz/human/SeqAvatar/scripts/exps_dnarendering.sh msti
```

## 2026-06-29 MSTI 日志规则更新

按新的约定，DNA-Rendering 脚本中涉及 MSTI 的日志不再写到普通 `logs/`，而是写到：

```text
/media/image/mxz/human/SeqAvatar/logs/msti
```

日志命名方式参考 Part-MoE：

```text
<RUN_TIME>_DNA-Rendering_msti.log
```

例如：

```text
logs/msti/20260629_181414_DNA-Rendering_msti.log
```

MSTI 模式下，同一次脚本运行只保留这个全局日志，不再额外生成：

```text
output/.../logs/train_<sequence>_msti_lite.log
output/.../logs/render_<sequence>_msti_lite.log
```

单卡完整六序列运行仍然是：

```bash
GPU_id=2 bash /media/image/mxz/human/SeqAvatar/scripts/exps_dnarendering.sh msti
```

如果双卡拆分但希望两个进程写入同一个 MSTI 日志，需要手动指定相同的 `RUN_TIME`：

```bash
RUN_TIME=20260629_181414 SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash /media/image/mxz/human/SeqAvatar/scripts/exps_dnarendering.sh msti
RUN_TIME=20260629_181414 SEQUENCES_OVERRIDE="0813_05 0007_04 0019_10" GPU_id=3 bash /media/image/mxz/human/SeqAvatar/scripts/exps_dnarendering.sh msti
```

这样两个命令都会追加到：

```text
logs/msti/20260629_181414_DNA-Rendering_msti.log
```

## 2026-06-29 MSTI 正确性检查现状

当前已有的检查分为两类：

1. 日志中能直接看到的配置输出：

```text
USE_MSTI: 1
MSTI_MODE: lite
MSTI_MID_TYPE: real
TIME_STEP_NUM(base): 3
MOTION_COND_TIME_STEP_NUM: 5
```

2. 代码中已有但不会主动打印“通过”的断言/报错：

```text
arguments.resolve_motion_condition_args:
  msti_mode 合法性
  msti_mid_type 合法性
  motion_cond_time_step_num 是否等于 time_step_num + 2

dataset_readers.get_seq_pose_xyz_cond_msti:
  MSTI mode 合法性
  seq_len 是否一致
  输出 channel 数是否等于 expected_channels
  seq_pose_conds / seq_xyz_conds channel 维度是否正确

SeqPoseEncoder / SeqXYZEncoder:
  encoder 实际收到的 motion channel 是否等于模型构造时的 motion_cond_time_step_num
```

这些检查不会在日志里打印成功信息；如果不满足，会直接 `ValueError` / `RuntimeError` 停止训练或渲染。因此六个序列完整训练和 render 完成，说明这些基本 shape/config 检查已经通过。

额外做过的直接证据：

```text
六个 cfg_args 均为:
  time_step_num = 3
  motion_cond_time_step_num = 5
  use_msti = True
  msti_mode = lite
  msti_mid_type = real

六个 checkpoint 的 non_rigid_deformer 均为:
  SeqPoseEncoder.mlp1.0.weight = (16, 825)
  SeqXYZEncoder.vel_encoder.0.weight = (64, 120)
```

其中：

```text
825 = 55 * 3 * 5
120 = 3 * seq_xyz_knn(8) * 5
```

如果还是 baseline 3 channel，DNA/SMPL-X 下对应维度应为：

```text
SeqPoseEncoder.mlp1.0.weight = (16, 495)
SeqXYZEncoder.vel_encoder.0.weight = (64, 72)
```

因此当前 checkpoint 已能证明模型实际使用了 5 个 motion condition channel。

当前还没有主动输出的诊断：

```text
MSTI full/sub channel 的 pose/xyz mean/std/max
每个 time_step 的输出 channel 顺序打印
每个 cur/former/mid pair 的抽样打印
参数量 / 显存 / 训练时间 / FPS 的统一汇总表
```

后续如果要更强地证明 MSTI “确实有效且不是偶然”，建议补一个只在调试开关开启时打印一次的诊断：

```text
SEQAVATAR_MSTI_DEBUG=1
```

打印内容：

```text
time_steps 原始顺序
MSTI-lite channel 顺序
pose/xyz condition shape
full/sub1/sub2 的 mean/std/max
若干 cur/former/mid id 样例
```

## 2026-06-30 MSTI 日志 20260629_205644 指标记录

本次查看日志：

```text
logs/msti/20260629_205644_DNA-Rendering_msti.log
```

该日志不是完整六序列结果：

- `0051_09` 完成训练和 render 复评。
- `0206_04` 开始训练后在约 `5740 / 25000` iter 处中断。
- 报错为 `RuntimeError: CUDA error: CUBLAS_STATUS_EXECUTION_FAILED`。
- 因此该日志不能计算六序列平均，只能记录已完成的 `0051_09` 指标。

`0051_09` 主结果取落盘 JSON：

```text
output/DNA-Rendering/0051_09/msti_lite/20260629_205644/metrics/results_novelview_25000.json
```

指标：

| Source | Sequence | PSNR | SSIM | LPIPS x1000 |
|---|---|---:|---:|---:|
| train final / JSON | 0051_09 | 28.5781 | 0.970808 | 31.5001 |
| render re-eval log | 0051_09 | 28.6719 | 0.971524 | 30.9937 |

后续论文表格或汇总建议继续以 JSON 主结果为准，避免和 render re-eval 指标混用。

## 2026-06-30 DNA-Rendering 单序列运行命令

`scripts/exps_dnarendering.sh` 支持用 `SEQUENCES_OVERRIDE` 只跑指定序列。

MSTI 单序列示例：

```bash
cd /media/image/mxz/human/SeqAvatar
SEQUENCES_OVERRIDE="0051_09" GPU_id=2 bash scripts/exps_dnarendering.sh msti
```

如果要指定固定日志/输出时间戳，加入 `RUN_TIME`：

```bash
RUN_TIME=20260630_005109 SEQUENCES_OVERRIDE="0051_09" GPU_id=2 bash scripts/exps_dnarendering.sh msti
```

同理，baseline 或 Part-MoE 只需要把最后的模式换成 `orginal`、`use_part_moe`、`part_moe_leg`、`part_moe_foot`、`part_moe_arm`。

## 2026-06-30 当前已完成 MSTI 序列指标汇总

本次扫描路径：

```text
output/DNA-Rendering/*/msti_lite/*/metrics/results_novelview_25000.json
```

共找到 11 个已完成 MSTI JSON，六个 DNA-Rendering 序列均已有完成结果。若同一序列有多次完成结果，下面按当前最新 `RUN_TIME` 选取：

| Sequence | RUN_TIME | PSNR | SSIM | LPIPS x1000 |
|---|---|---:|---:|---:|
| 0007_04 | 20260630_143504 | 29.5715 | 0.958566 | 44.5145 |
| 0019_10 | 20260630_143525 | 35.2645 | 0.981049 | 20.9985 |
| 0044_11 | 20260629_181414_msti_lite_gpu2 | 32.9590 | 0.977824 | 21.5928 |
| 0051_09 | 20260629_205644 | 28.5781 | 0.970808 | 31.5001 |
| 0206_04 | 20260630_121121 | 31.4159 | 0.970262 | 33.5490 |
| 0813_05 | 20260630_121155 | 35.9521 | 0.986651 | 18.6504 |
| Mean | latest-per-sequence | 32.2902 | 0.974193 | 28.4676 |

注意：

- 上表是“每个序列最新完成结果”的混合汇总，不是同一个 `RUN_TIME` 的完整六序列实验。
- 如果论文表格要求严格同一次完整运行，仍建议使用 `20260629_181414_msti_lite_gpu2/gpu3` 那组六序列完整结果。
- `20260629_205733_DNA-Rendering_msti.log` 中 `0044_11` 出现 OOM，没有可用完成指标。
- `20260629_205644_DNA-Rendering_msti.log` 中 `0051_09` 完成，但后续 `0206_04` 报 CUBLAS 中断；`0206_04` 的最新完成指标来自 `20260630_121121`。

## 2026-06-30 最新 densify_until_iter=1800 MSTI 指标口径

当前 `scripts/exps_dnarendering.sh` 中 MSTI 训练参数为：

```text
densify_until_iter = 1800
iter = 25000
time_step_num = 3
motion_cond_time_step_num = 5
msti_mode = lite
msti_mid_type = real
```

按“最新一次 1800 重跑且已经落盘 `results_novelview_25000.json`”统计，当前完成 5 个序列：

| Sequence | RUN_TIME | PSNR | SSIM | LPIPS x1000 |
|---|---|---:|---:|---:|
| 0051_09 | 20260629_205644 | 28.5781 | 0.970808 | 31.5001 |
| 0206_04 | 20260630_121121 | 31.4159 | 0.970262 | 33.5490 |
| 0813_05 | 20260630_121155 | 35.9521 | 0.986651 | 18.6504 |
| 0007_04 | 20260630_143504 | 29.5715 | 0.958566 | 44.5145 |
| 0019_10 | 20260630_143525 | 35.2645 | 0.981049 | 20.9985 |
| Mean completed-5 | - | 32.1564 | 0.973467 | 29.8425 |

`0044_11` 的最新 1800 重跑：

```text
RUN_TIME = 20260629_205733
log = logs/msti/20260629_205733_DNA-Rendering_msti.log
status = CUDA out of memory
```

因此 `0044_11` 暂时没有最新 1800 完成指标。若为了临时做六序列表补上上一轮已完成结果，则使用：

```text
0044_11 / 20260629_181414_msti_lite_gpu2:
PSNR = 32.9590
SSIM = 0.977824
LPIPS x1000 = 21.5928
```

补上该旧完成结果后的六序列混合平均为：

```text
PSNR = 32.2902
SSIM = 0.974193
LPIPS x1000 = 28.4676
```

注意：这个六序列平均是“5 个最新 1800 完成结果 + 1 个 0044_11 旧完成结果”的混合口径，不应写成严格同一次完整实验平均。

## 2026-06-30 Part-MoE-Leg + MSTI 组合实验判断

当前代码层面 `train.py / render.py / GaussianModel / NonrigidDeformer` 可以同时接收：

```text
--use_msti
--msti_mode lite
--msti_mid_type real
--motion_cond_time_step_num 5
--use_part_moe
--part_label_schema part_moe_leg
--num_parts 7
```

因为：

- MSTI 改的是 `seq_pose_conds / seq_xyz_conds` 的 motion condition channel 数。
- Part-MoE 改的是 non-rigid deformer 后半段的专家路由。
- 两者在 `NonrigidDeformer.forward()` 中的交汇点是最终拼接后的 `features`，原则上可以组合。

但当前 `scripts/exps_dnarendering.sh` 的 `MODE` 是互斥的：

```text
msti -> use_msti=1, part_moe_enabled=0
part_moe_leg -> use_msti=0, part_moe_enabled=1
```

因此不能直接用：

```bash
bash scripts/exps_dnarendering.sh part_moe_leg msti
```

这种方式开启两个消融；脚本只会读取第一个模式参数，且日志/输出目录会混乱。

推荐新增独立组合模式：

```text
part_moe_leg_msti
```

建议设定：

```text
experiment_name = part_moe_leg_msti
part_moe_enabled = 1
use_msti = 1
msti_mode = lite
msti_mid_type = real
part_label_schema = part_moe_leg
num_parts = 7
final_eval_only = 1
time_step_num = 3
motion_cond_time_step_num = 5
densify_until_iter = 1800 或更保守值
```

注意事项：

- 组合实验必须单独输出到 `output/DNA-Rendering/<seq>/part_moe_leg_msti/<RUN_TIME>/`，不要复用 `msti_lite` 或 `part_moe_leg`。
- checkpoint 不能和单独 MSTI 或单独 Part-MoE-Leg 混用，因为 encoder channel 和 expert 结构不同。
- DNA 上 `part_moe_leg` 本身显存紧，MSTI 又会增加 motion encoder 输入维度，建议先单序列 smoke test，再跑六序列。
- 如果 OOM，优先考虑降低 `densify_until_iter`、开启 `SKIP_LOAD_TEST_CAMERAS=1` 或单卡单序列运行。

## 2026-06-30 Part-MoE-Leg + MSTI DNA 五序列消融启动记录

按用户要求新增独立 DNA 脚本模式：

```text
MODE = part_moe_leg_msti
experiment_name = part_moe_leg_msti
part_moe_enabled = 1
part_label_schema = part_moe_leg
num_parts = 7
use_msti = 1
msti_mode = lite
msti_mid_type = real
time_step_num = 3
motion_cond_time_step_num = 5
densify_until_iter = 1800
final_eval_only = 1
```

隔离原则：

- 不改变 `msti` / `msti_lite` 的输出目录和参数。
- 不改变 `part_moe_leg` 的输出目录和参数。
- 组合实验单独输出到：

```text
output/DNA-Rendering/<sequence>/part_moe_leg_msti/<RUN_TIME>/
logs/part/<RUN_TIME>_DNA-Rendering_part_moe_leg_msti.log
```

本次运行排除 `0044_11`，只跑五个序列：

```text
0051_09 0206_04 0813_05 0007_04 0019_10
```

启动命令模板：

```bash
SEQUENCES_OVERRIDE="0051_09 0206_04 0813_05 0007_04 0019_10" \
GPU_id=2 \
bash scripts/exps_dnarendering.sh part_moe_leg_msti
```

实际已启动运行：

```text
RUN_TIME = 20260630_160756
GPU_id = 2
PID = 654398
global log = logs/part/20260630_160756_DNA-Rendering_part_moe_leg_msti.log
launch log = /tmp/seqavatar_20260630_160756_part_moe_leg_msti.launch.log
```

日志头部已确认：

```text
Mode = part_moe_leg_msti
Experiment = part_moe_leg_msti
PART_LABEL_SCHEMA = part_moe_leg
NUM_PARTS = 7
USE_MSTI = 1
MSTI_MODE = lite
MSTI_MID_TYPE = real
MOTION_COND_TIME_STEP_NUM = 5
DENSIFY_UNTIL_ITER = 1800
Sequences = 0051_09 0206_04 0813_05 0007_04 0019_10
```

启动后 `0051_09` 已进入训练迭代，说明组合模式的参数解析、MSTI condition 构造、Part-MoE 初始化和数据读取阶段没有立即失败。

## 2026-06-30 单独 MSTI 消融代码改动总结

单独 MSTI 消融涉及的核心代码文件：

```text
arguments/__init__.py
scene/__init__.py
scene/dataset_readers.py
scene/gaussian_model.py
nets/mlp_delta_non_rigid.py
scripts/exps_dnarendering.sh
```

改动目标：

- 默认不启用 MSTI，不影响 baseline。
- 不影响 Part-MoE 单独实验。
- 只在显式 `--use_msti --msti_mode lite` 时扩展 motion condition。
- 不新增中间 RGB、mask、camera、CameraInfo。
- 中间帧只用于构造额外 SMPL motion condition，不作为监督输出。

文件级说明：

1. `arguments/__init__.py`

新增参数：

```text
motion_cond_time_step_num
use_msti
msti_mode
msti_mid_type
```

新增 `resolve_motion_condition_args()`，用于自动推导和校验：

```text
msti_mode=none -> motion_cond_time_step_num = time_step_num
msti_mode=lite -> motion_cond_time_step_num = time_step_num + 2
msti_mode=full -> motion_cond_time_step_num = time_step_num * 3
```

如果手填的 `motion_cond_time_step_num` 和模式不匹配，会直接报错，避免 train/render channel 不一致。

2. `scene/__init__.py`

把 MSTI 相关参数打包成 `motion_cond_options`，传给各 dataset reader：

```text
use_msti
msti_mode
msti_mid_type
motion_cond_time_step_num
```

这样 baseline / MSTI 只在 dataset reader 构造 condition 时分流，renderer 和 loss 不需要改。

3. `scene/dataset_readers.py`

保留原始 `get_seq_pose_xyz_cond()` baseline 路径；当 `use_msti=True` 且 `msti_mode!=none` 时，进入新增的：

```text
get_seq_pose_xyz_cond_msti()
```

当前实现为 `Real-Mid MSTI-lite/full`：

- `lite`：只对最大原始时间跨度插入 `sub1/sub2`，DNA 中 `time_step_num=3` 时 channel 从 3 变成 5。
- `full`：对每个原始时间跨度都插入 `sub1/sub2`，channel 从 `N` 变成 `3N`。
- 当前只支持 `msti_mid_type=real`。

MSTI-lite 的 channel 顺序固定为：

```text
[full_max, sub1_max, sub2_max, full_other1, full_other2, ...]
```

delta 定义保持和原代码一致：

```text
cur_pose_mat @ inverse(former_pose_mat)
```

xyz delta 在 MSTI 分支中按每个 pair 的实际 `dt` 单独归一化，避免 full/sub 通道使用同一个旧 `time_step` 归一化。

4. `scene/gaussian_model.py`

读取 MSTI 参数，并用：

```text
motion_cond_time_step_num
```

而不是原来的 `time_step_num` 去构造 `NonrigidDeformer`。这样 `time_step_num` 继续表示原始采样尺度数量，encoder 输入 channel 数由单独参数控制。

5. `nets/mlp_delta_non_rigid.py`

让 `SeqPoseEncoder / SeqXYZEncoder` 按传入的 `time_step_num` 初始化输入维度。这里传入的已经是 `motion_cond_time_step_num`。

实际 DNA/SMPL-X 下：

```text
baseline 3 channel:
  SeqPoseEncoder.mlp1.0.weight = (16, 495)
  SeqXYZEncoder.vel_encoder.0.weight = (64, 72)

MSTI-lite 5 channel:
  SeqPoseEncoder.mlp1.0.weight = (16, 825)
  SeqXYZEncoder.vel_encoder.0.weight = (64, 120)
```

这些维度也作为验证 MSTI 真的进入模型的直接证据。

6. `scripts/exps_dnarendering.sh`

新增 `msti / msti_lite / real_mid_msti_lite` 模式：

```text
experiment_name = msti_lite
use_msti = 1
msti_mode = lite
msti_mid_type = real
time_step_num = 3
motion_cond_time_step_num = 5
densify_until_iter = 1800
final_eval_only = 1
```

MSTI 日志单独写到：

```text
logs/msti/<RUN_TIME>_DNA-Rendering_msti.log
```

输出目录为：

```text
output/DNA-Rendering/<sequence>/msti_lite/<RUN_TIME>/
```

运行命令：

```bash
GPU_id=2 bash scripts/exps_dnarendering.sh msti
```

单序列运行：

```bash
SEQUENCES_OVERRIDE="0051_09" GPU_id=2 bash scripts/exps_dnarendering.sh msti
```

补充：后续加入的 `part_moe_leg_msti` 是组合实验入口，不属于单独 MSTI 消融本身；单独 MSTI 的输出目录仍是 `msti_lite`，不会和组合实验混用。

## 2026-06-30 none / lite / full 与 motion_cond_time_step_num 含义

`time_step_num` 和 `motion_cond_time_step_num` 是两个不同概念：

```text
time_step_num:
  原始 SeqAvatar 要取多少个历史时间尺度。

motion_cond_time_step_num:
  最终送进 SeqPoseEncoder / SeqXYZEncoder 的 motion condition channel 数。
```

以 DNA 当前设置为例：

```text
time_step_num = 3
原始时间尺度约为 [3, 2, 1]
```

三种模式含义：

1. `none`

```text
motion_cond_time_step_num = time_step_num
```

不做 MSTI，完全是 baseline motion condition：

```text
[full_3, full_2, full_1]
```

DNA 中 channel 数为 `3`。

2. `lite`

```text
motion_cond_time_step_num = time_step_num + 2
```

只对最大时间跨度加中间帧子运动，把最大跨度：

```text
full_3: t-3 -> t
```

拆出两个额外子通道：

```text
sub1_3: t-3 -> mid
sub2_3: mid -> t
```

最终顺序：

```text
[full_3, sub1_3, sub2_3, full_2, full_1]
```

DNA 中 channel 数为 `5`。这是当前主实验采用的低风险版本。

3. `full`

```text
motion_cond_time_step_num = time_step_num * 3
```

对每个原始时间尺度都加 `sub1/sub2`：

```text
[full_3, sub1_3, sub2_3,
 full_2, sub1_2, sub2_2,
 full_1, sub1_1, sub2_1]
```

DNA 中 channel 数为 `9`。信息最多，但显存和参数量更大，而且短跨度如 `1` 的 real-mid 可能退化，当前没有作为主实验。

作用总结：

- `msti_mode` 决定是否给原始 motion condition 加中间运动子通道。
- `motion_cond_time_step_num` 决定模型 encoder 的输入维度。
- 这不会增加训练图片、测试图片、camera 或监督帧。
- 如果 train/render 的 `motion_cond_time_step_num` 不一致，会造成 checkpoint 和 condition shape 不匹配，所以代码中做了自动推导和 assert。

## 2026-06-30 full_3 / full_2 / full_1 的来源

`full_3 / full_2 / full_1` 不是三张图片，也不是三个输出帧，而是三个“历史时间跨度”的 motion condition。

DNA 脚本当前设置：

```text
minimal_time_step = 1
max_time_step = 3
time_step_num = 3
seq_len = 8
```

`scene/__init__.py` 调用：

```python
generate_time_steps(args.minimal_time_step, args.max_time_step, 1, args.time_step_num, args.seq_len)
```

`generate_time_steps()` 逻辑是：

```text
all_steps = [1, 2, 3]
选 time_step_num=3 个尺度
反转顺序 -> [3, 2, 1]
返回 {3: 8, 2: 8, 1: 8}
```

所以对当前真实帧 `t`，baseline 会构造三个尺度的相对运动：

```text
full_3: t-3 -> t
full_2: t-2 -> t
full_1: t-1 -> t
```

更准确地说，代码里还会沿时间序列长度 `seq_len=8` 继续往前取，例如对 `full_3`：

```text
i=0: t-3  -> t
i=1: t-6  -> t-3
i=2: t-9  -> t-6
...
i=7: t-24 -> t-21
```

每个 pair 都会算：

```text
pose delta: cur_pose @ inverse(former_pose)
xyz delta:  cur_xyz - former_xyz
```

因此 `[full_3, full_2, full_1]` 的意思是：对同一个训练目标帧 `t`，提供三个不同时间间隔的历史运动条件，让 non-rigid deformer 同时看到长、中、短三种运动跨度。

## 2026-06-30 part_moe_leg_msti 命令使用注意

`scripts/exps_dnarendering.sh` 只读取第一个位置参数：

```bash
MODE=${1:-orginal}
```

因此下面这种命令是错误的：

```bash
SEQUENCES_OVERRIDE="0206_04 0813_05" GPU_id=2 bash scripts/exps_dnarendering.sh part_moe_leg part_moe_leg_msti
```

它只会运行：

```text
MODE = part_moe_leg
USE_MSTI = 0
MOTION_COND_TIME_STEP_NUM = 3
```

第二个 `part_moe_leg_msti` 会被脚本忽略。

正确运行组合实验应写成：

```bash
SEQUENCES_OVERRIDE="0206_04 0813_05" GPU_id=2 bash scripts/exps_dnarendering.sh part_moe_leg_msti
```

判断是否真的跑组合实验，看日志头部必须同时满足：

```text
Mode: part_moe_leg_msti
Experiment: part_moe_leg_msti
USE_MSTI: 1
MOTION_COND_TIME_STEP_NUM: 5
PART_LABEL_SCHEMA: part_moe_leg
NUM_PARTS: 7
```

## 2026-06-30 tmux 启动 part_moe_leg_msti 五序列

按用户要求，在 GPU 2 和 GPU 3 上用 tmux 启动 `part_moe_leg_msti`，排除 `0044_11`。

运行组：

```text
RUN_GROUP = 20260630_181505_part_moe_leg_msti
```

GPU 分配：

```text
GPU 2:
  session = seqavatar_pmlm_gpu2_20260630_181505_part_moe_leg_msti
  sequences = 0051_09 0206_04 0813_05
  wrapper log = logs/part/20260630_181505_part_moe_leg_msti_gpu2.wrapper.log

GPU 3:
  session = seqavatar_pmlm_gpu3_20260630_181505_part_moe_leg_msti
  sequences = 0007_04 0019_10
  wrapper log = logs/part/20260630_181505_part_moe_leg_msti_gpu3.wrapper.log
```

每个序列单独调用一次：

```bash
SEQUENCES_OVERRIDE="$SEQ" RUN_TIME="$RUN_TIME" GPU_id=<2/3> \
bash scripts/exps_dnarendering.sh part_moe_leg_msti
```

这样某个序列如果 OOM 或其它非零退出，外层 tmux 循环会记录 `FAIL` 并继续下一个序列。

已确认首批两个序列日志头：

```text
0051_09 / GPU 2:
  Mode = part_moe_leg_msti
  USE_MSTI = 1
  MOTION_COND_TIME_STEP_NUM = 5
  DENSIFY_UNTIL_ITER = 1800

0007_04 / GPU 3:
  Mode = part_moe_leg_msti
  USE_MSTI = 1
  MOTION_COND_TIME_STEP_NUM = 5
  DENSIFY_UNTIL_ITER = 1800
```

监控命令：

```bash
tmux attach -t seqavatar_pmlm_gpu2_20260630_181505_part_moe_leg_msti
tmux attach -t seqavatar_pmlm_gpu3_20260630_181505_part_moe_leg_msti

tail -f logs/part/20260630_181505_part_moe_leg_msti_gpu2.wrapper.log
tail -f logs/part/20260630_181505_part_moe_leg_msti_gpu3.wrapper.log
```

## 2026-06-30 part_moe_leg_msti 五序列结果汇总

运行组：

```text
RUN_GROUP = 20260630_181505_part_moe_leg_msti
```

执行状态：

```text
0051_09: OOM，wrapper 记录 FAIL，未产生 results_novelview_25000.json
0206_04: completed
0813_05: completed
0007_04: completed
0019_10: completed
```

`0051_09` 报错：

```text
torch.cuda.OutOfMemoryError: CUDA out of memory
```

主结果取各序列落盘 JSON：

```text
output/DNA-Rendering/<sequence>/part_moe_leg_msti/20260630_181505_part_moe_leg_msti_<sequence>_gpu*/metrics/results_novelview_25000.json
```

完成序列指标：

| Sequence | RUN_TIME | PSNR | SSIM | LPIPS x1000 |
|---|---|---:|---:|---:|
| 0007_04 | 20260630_181505_part_moe_leg_msti_0007_04_gpu3 | 29.5823 | 0.958989 | 43.5209 |
| 0019_10 | 20260630_181505_part_moe_leg_msti_0019_10_gpu3 | 35.4667 | 0.981665 | 20.3041 |
| 0206_04 | 20260630_181505_part_moe_leg_msti_0206_04_gpu2 | 31.4194 | 0.969954 | 33.5044 |
| 0813_05 | 20260630_181505_part_moe_leg_msti_0813_05_gpu2 | 36.1462 | 0.987179 | 18.0954 |
| Mean completed-4 | - | 33.1536 | 0.974447 | 28.8562 |

render re-eval 日志指标作为参考：

| Sequence | PSNR | SSIM | LPIPS x1000 |
|---|---:|---:|---:|
| 0007_04 | 29.5824 | 0.959001 | 43.4950 |
| 0019_10 | 35.4678 | 0.981685 | 20.2839 |
| 0206_04 | 31.4216 | 0.970044 | 33.3545 |
| 0813_05 | 36.1502 | 0.987191 | 18.0774 |
| Mean completed-4 | 33.1555 | 0.974480 | 28.8027 |

后续汇总表建议继续使用 JSON 主结果，避免混用 train final eval 和 render re-eval。

## 2026-06-30 full_3 拆成 sub1_3 / sub2_3 的具体方式

在 `MSTI-lite` 中，只展开最大时间跨度：

```text
max_step = max(time_steps.keys())
```

DNA 当前 `time_steps = {3: 8, 2: 8, 1: 8}`，所以只展开 `time_step=3`。

代码逻辑：

```python
cur_id = t
former_id = t - 3
mid_id = former_id + (cur_id - former_id) // 2
```

因此如果当前目标帧是 `t=100`：

```text
cur_id = 100
former_id = 97
mid_id = 97 + (100 - 97) // 2 = 98
```

原始 baseline 的最大跨度是：

```text
full_3: frame 97 -> frame 100
```

MSTI-lite 额外加入两个子运动：

```text
sub1_3: frame 97 -> frame 98
sub2_3: frame 98 -> frame 100
```

同时仍保留原始 `full_3`，所以最大跨度部分变成：

```text
[full_3, sub1_3, sub2_3]
```

再加上没展开的短跨度：

```text
[full_3, sub1_3, sub2_3, full_2, full_1]
```

注意：因为当前 real-mid 使用整数帧 id：

```text
mid_id = former_id + (cur_id - former_id) // 2
```

当跨度是 3 时，拆分不是严格的 `1.5 + 1.5`，而是：

```text
97 -> 98  = 1 帧
98 -> 100 = 2 帧
```

这是当前低风险 `Real-Mid MSTI-lite` 的实现选择。`xyz_delta` 在 MSTI 分支里会分别按每个 pair 的实际 dt 归一化：

```text
full_3: dt = 3
sub1_3: dt = 1
sub2_3: dt = 2
```

pose delta 方向三者一致，都是：

```text
cur_pose @ inverse(former_pose)
```

## 2026-06-30 DNA 上 Real-Mid MSTI-lite 的通道重复问题

以目标帧 `t=100`、DNA 当前时间尺度 `[3, 2, 1]` 为例，baseline 不是输入单帧特征，而是输入三个相对运动 pair：

```text
full_3: 97 -> 100
full_2: 98 -> 100
full_1: 99 -> 100
```

当前 `Real-Mid MSTI-lite` 对最大跨度 `full_3` 做整数 real-mid：

```text
mid = 97 + (100 - 97) // 2 = 98
```

因此 MSTI-lite 变成：

```text
full_3: 97 -> 100
sub1_3: 97 -> 98
sub2_3: 98 -> 100
full_2: 98 -> 100
full_1: 99 -> 100
```

结论：

- `sub1_3: 97 -> 98` 是新增的局部运动信息。
- `sub2_3: 98 -> 100` 和 `full_2: 98 -> 100` 是同一个 pair，确实重复。
- 所以在 DNA 的 `[3, 2, 1]` 短跨度设置下，MSTI-lite 的新增信息并不充分，存在“一个新子段 + 一个重复子段”的问题。

这不等于完全没有新东西，但说明当前 DNA 版 Real-Mid MSTI-lite 的方法解释比较弱。后续更干净的方案：

```text
1. Virtual-mid: 构造 98.5 的虚拟 SMPL 中间状态，再拆成 97 -> 98.5 和 98.5 -> 100。
2. 使用更大的 max_time_step，例如 4 或 6，让 real-mid 不直接和已有 full_2/full_1 重复。
3. 做去重版 MSTI-lite，只保留不重复的新增 pair，但这样 channel 数和解释需要重新设计。
```

## 2026-06-30 full_3 与 seq_len=8 的关系

`full_3` 有两个层面的含义，容易混淆：

```text
1. channel 名称:
   full_3 表示“时间跨度为 3 的完整运动通道”。

2. 该 channel 内部的历史序列:
   因为 seq_len=8，所以 full_3 这个通道里不是只有一个 pair，而是 8 个连续历史 pair。
```

以目标帧 `t=100` 为例：

```text
full_3 在 i=0 时是 97 -> 100
```

这就是前面说的：

```text
full_3: frame 97 -> frame 100
```

但代码还有一层循环：

```python
for i in range(seq_len):
    cur_id = t - i * time_step
    former_id = t - (i + 1) * time_step
```

当 `time_step=3`、`seq_len=8` 时，`full_3` 通道完整包含：

```text
i=0:  97 -> 100
i=1:  94 -> 97
i=2:  91 -> 94
i=3:  88 -> 91
i=4:  85 -> 88
i=5:  82 -> 85
i=6:  79 -> 82
i=7:  76 -> 79
```

所以更准确的说法是：

```text
full_3 是一个时间跨度为 3 的 motion channel；
对目标帧 100，它的第一个历史片段是 97 -> 100；
由于 seq_len=8，它还包含更早的 7 段同跨度历史运动。
```

同理：

```text
full_2:
  i=0: 98 -> 100
  i=1: 96 -> 98
  ...

full_1:
  i=0: 99 -> 100
  i=1: 98 -> 99
  ...
```

## 2026-06-30 sub1_3 / sub2_3 也有 seq_len=8 段

是的，`sub1_3` 和 `sub2_3` 也是 motion condition channel。它们和 `full_3/full_2/full_1` 一样，都沿 `seq_len=8` 维度保存 8 段历史运动。

区别是：

```text
full_3:
  每个 i 直接取跨度 3 的 pair。

sub1_3 / sub2_3:
  每个 i 先取同一个跨度 3 的 full pair，再把这个 pair 按 real-mid 拆成两个子 pair。
```

代码位置：

```python
for i in range(seq_len):
    cur_id = t - i * time_step
    former_id = t - (i + 1) * time_step
    mid_id = former_id + (cur_id - former_id) // 2
```

以 `t=100`、`time_step=3`、`seq_len=8` 为例：

```text
i=0:
  full_3:  97 -> 100
  sub1_3:  97 -> 98
  sub2_3:  98 -> 100

i=1:
  full_3:  94 -> 97
  sub1_3:  94 -> 95
  sub2_3:  95 -> 97

i=2:
  full_3:  91 -> 94
  sub1_3:  91 -> 92
  sub2_3:  92 -> 94

...

i=7:
  full_3:  76 -> 79
  sub1_3:  76 -> 77
  sub2_3:  77 -> 79
```

因此 MSTI-lite 的 5 个 channel，每个 channel 都有 `seq_len=8` 段：

```text
channel 0: full_3,  8 段
channel 1: sub1_3,  8 段
channel 2: sub2_3,  8 段
channel 3: full_2,  8 段
channel 4: full_1,  8 段
```

## 2026-06-30 sub1_3 不是预测第 98 帧

以 `sub1_3: 97 -> 98` 为例，它不是网络“预测到 98 帧”。

当前数据集中第 97、98、100 帧的 SMPL 参数本来就是已知的。MSTI 在 dataset reader 阶段直接读取这些已知 SMPL 状态，预先计算：

```text
97 -> 98 的 pose delta / xyz delta
```

然后把这个 delta 作为第 100 帧训练/渲染时的额外 motion condition 输入。

网络的最终目标没有变：

```text
仍然渲染 / 监督第 100 帧
仍然预测当前目标帧对应的 Gaussian 非刚性形变 d_xyz / d_rotation / d_scaling
不会输出第 98 帧图像
不会新增第 98 帧监督
```

因此，MSTI 的本质是：

```text
不改变输出目标；
只改变输入条件；
把原来 3 个 motion channel 扩展成 5 个 motion channel。
```

代码层面的网络改动是输入维度变了：

```text
baseline:
  motion_cond_time_step_num = 3

MSTI-lite:
  motion_cond_time_step_num = 5
```

`SeqPoseEncoder / SeqXYZEncoder` 会接收更多 motion condition channel，但后面的渲染目标和 loss 仍然是当前真实帧。

## 2026-06-30 full_3 / full_2 / full_1 如何一起学习

`[full_3, full_2, full_1]` 不是求平均。

在 `SeqPoseEncoder` 中，输入形状可理解为：

```text
B x seq_len x motion_channels x joints x 3
```

baseline 时：

```text
motion_channels = 3
channels = [full_3, full_2, full_1]
```

代码做法：

```python
x = self.mlp1(x.view(bs, T, -1))
x = self.mlp2(x.view(bs, -1))
```

含义：

```text
1. 对每个历史位置 i，把 full_3/full_2/full_1 以及所有 joints 的 delta 拼成一个长向量。
2. 用 Linear + ReLU 学习这个长向量的组合关系。
3. 再把 seq_len=8 个历史位置的特征拼起来，用第二个 Linear + ReLU 融合成最终 seq_pose_feats。
```

所以它是：

```text
concat / flatten -> MLP 学习融合
```

不是：

```text
mean(full_3, full_2, full_1)
```

`SeqXYZEncoder` 也类似，不是求平均。它对每个高斯附近的 KNN vertex motion，把：

```text
full_3/full_2/full_1 的 xyz delta
```

拼成一个长向量后送入 `vel_encoder`，再和高斯位置编码拼接，最后通过 MLP 融合 `seq_len=8` 的历史。

因此不同通道的重要性是由网络 Linear 权重学出来的；如果某个通道有用，训练会给它更有效的权重。如果通道重复或无用，也可能被学成较小影响。

## 2026-06-30 AMC 替代 MSTI 的方案讨论

用户提出将第二创新点从当前 `MSTI` 调整为：

```text
AMC = Autoregressive Motion Context
```

动机：

- 当前 DNA 的 `Real-Mid MSTI-lite` 只加 `sub1_max/sub2_max`。
- 当最大跨度为 3 时，`sub2_3: 98 -> 100` 和 `full_2: 98 -> 100` 重复。
- 因此 MSTI 在 DNA 上创新解释偏弱。

用户希望以目标帧 `t=100` 为例，从 baseline 的：

```text
full_3: 97 -> 100
full_2: 98 -> 100
full_1: 99 -> 100
```

增加历史内部的因果运动：

```text
97 -> 98
97 -> 99
98 -> 99
```

直觉解释：

```text
98 根据 97 学到
99 根据 97 和 98 学到
100 根据 97、98、99 学到
```

可行性判断：

```text
可行，但要区分“加 pairwise delta 通道”和“真正自回归 encoder”。
```

### 1. 低风险版本：AMC-pair / causal pairwise context

构造局部时间窗口：

```text
p0 = 97
p1 = 98
p2 = 99
p3 = 100
```

计算所有因果 pair：

```text
stage 1:
  e01 = 97 -> 98

stage 2:
  e02 = 97 -> 99
  e12 = 98 -> 99

stage 3 / target:
  e03 = 97 -> 100
  e13 = 98 -> 100
  e23 = 99 -> 100
```

最终 motion channels 可以设为：

```text
[e01, e02, e12, e03, e13, e23]
```

DNA 中 channel 数从 baseline 的 3 变成：

```text
motion_cond_time_step_num = 6
```

优点：

- 实现风险相对低。
- 不增加 RGB、mask、camera、监督帧。
- 仍然只改变 motion condition。
- 比 MSTI-lite 更少出现 `sub2_3 == full_2` 这种显式重复。

问题：

- 如果仍然用当前 `SeqPoseEncoder / SeqXYZEncoder` 的 flatten + MLP，它不会强制执行“98 -> 99 -> 100”的自回归顺序。
- 所以这种版本更准确叫：

```text
Causal Pairwise Motion Context
```

而不是严格的 autoregressive model。

### 2. 更强版本：AMC causal encoder

为了让 `Autoregressive Motion Context` 名字更站得住，建议不是简单 concat 6 个 pair，而是做阶段式因果聚合：

```text
h1 = f1(e01)
h2 = f2(h1, e02, e12)
h3 = f3(h2, e03, e13, e23)
```

其中：

```text
h1 表示 98 从 97 得到的 motion state
h2 表示 99 从 97、98 得到的 motion state
h3 表示 100 从 97、98、99 得到的 motion context
```

最后将 `h3` 作为目标帧的 sequential motion feature 输入 non-rigid deformer。

这才更符合：

```text
98 is conditioned on 97
99 is conditioned on 97 and 98
100 is conditioned on 97, 98, and 99
```

代价：

- 需要新增或改造 `SeqPoseEncoder / SeqXYZEncoder`。
- 不只是 channel 数变化，而是 encoder 结构变化。
- 实验风险和调参成本高于 AMC-pair。

### 3. seq_len=8 下怎么推广

对每个历史窗口重复这个 causal construction。以 `max_step=3` 为 anchor：

```text
i=0: [97, 98, 99, 100]
i=1: [94, 95, 96, 97]
i=2: [91, 92, 93, 94]
...
```

每个窗口都构造 6 个因果 pair：

```text
p0 -> p1
p0 -> p2
p1 -> p2
p0 -> p3
p1 -> p3
p2 -> p3
```

这样 AMC 的输入仍然有 `seq_len=8` 个历史位置，只是每个位置的 motion channels 从 3 变成 6。

### 4. 需要注意的事实

当前 baseline 的 `full_1` 通道在 `seq_len` 维度里已经包含一些短期 pair：

```text
full_1, i=0: 99 -> 100
full_1, i=1: 98 -> 99
full_1, i=2: 97 -> 98
```

因此，`97 -> 98` 和 `98 -> 99` 并不是完全从无到有的新信息；它们在 baseline 中已经以更早历史位置存在。

AMC 的真正价值应解释为：

```text
把这些历史内部运动重新组织到同一个局部 causal window 中，
并显式建模 p0/p1/p2/p3 的因果关系。
```

如果只是简单增加通道，创新强度仍然有限；如果配合 causal encoder，解释会更强。

### 5. 推荐路线

建议后续如果替换 MSTI，分两步：

```text
Step 1:
  AMC-pair，6 channel，先验证是否比 MSTI-lite 有提升。

Step 2:
  AMC-causal encoder，阶段式 h1/h2/h3 聚合，作为正式创新版本。
```

第一版实验命名可以用：

```text
AMC-pair
```

正式版本命名：

```text
AMC = Autoregressive Motion Context
```

如果论文表述要稳妥，应避免把简单 concat 版本说成严格自回归。

## 2026-06-30 AMC-pair 与 AMC causal encoder 区别

核心区别：

```text
AMC-pair:
  只改变输入 motion channels。
  把更多 pairwise delta 拼起来，仍然交给原来的 MLP 自己学。

AMC causal encoder:
  改变 motion encoder 结构。
  显式按 97 -> 98 -> 99 -> 100 的因果阶段递推。
```

以局部窗口 `[97, 98, 99, 100]` 为例。

### AMC-pair

构造 6 个 pair：

```text
e01 = 97 -> 98
e02 = 97 -> 99
e12 = 98 -> 99
e03 = 97 -> 100
e13 = 98 -> 100
e23 = 99 -> 100
```

然后直接作为 6 个 channel 输入：

```text
[e01, e02, e12, e03, e13, e23]
```

encoder 做的事仍然接近当前 SeqAvatar：

```text
concat / flatten -> MLP
```

特点：

- 实现简单，主要改 dataset reader 和 `motion_cond_time_step_num=6`。
- 不强制先学 98、再学 99、再学 100。
- MLP 可以自己学通道关系，但因果顺序不是结构约束。
- 更适合作为低风险消融：`AMC-pair`。

### AMC causal encoder

仍然使用这些 pair，但不是一次性拼接，而是按阶段聚合：

```text
h1 = f1(e01)
h2 = f2(h1, e02, e12)
h3 = f3(h2, e03, e13, e23)
```

含义：

```text
h1: 98 基于 97 的运动状态
h2: 99 基于 97、98 的运动状态
h3: 100 基于 97、98、99 的最终 motion context
```

最后只把 `h3` 或 `[h1,h2,h3]` 融合结果送给 non-rigid deformer。

特点：

- 需要新建或重写 `SeqPoseEncoder / SeqXYZEncoder`。
- 结构上显式体现自回归/因果阶段。
- 论文表述更强，更配得上 `Autoregressive Motion Context`。
- 实现和调参风险更高。

### 简表

| 项目 | AMC-pair | AMC causal encoder |
|---|---|---|
| 改 dataset condition | 是 | 是 |
| 改 encoder 结构 | 基本不用 | 需要 |
| 输入 channel | 6 个 pair | 6 个 pair 或阶段输入 |
| 融合方式 | flatten + MLP | h1 -> h2 -> h3 阶段递推 |
| 是否严格自回归 | 否 | 是，更接近 |
| 实现风险 | 低 | 中高 |
| 论文创新强度 | 中等 | 更强 |

建议：

```text
先做 AMC-pair 验证方向；
如果有效，再做 AMC causal encoder 作为正式版本。
```

### 命名规范：`amc_pair` vs 正式 `amc`

后续实验命名建议固定为：

```text
amc_pair:
  第一阶段低风险消融。
  只扩展 motion condition channel。
  不改 SeqPoseEncoder / SeqXYZEncoder 的基本融合方式。
  本质是 autoregressive-inspired pairwise context。

amc:
  正式 Autoregressive Motion Context。
  不只增加 pairwise delta，还要改变 motion encoder 的融合结构。
  需要显式按历史状态递推或因果聚合，例如 h1 -> h2 -> h3。
```

核心判断：

```text
amc_pair 证明“历史内部 pairwise motion 是否有用”；
amc 证明“结构化自回归 motion context 是否有用”。
```

论文中如果只实现 `amc_pair`，建议不要把方法主张写成严格自回归；可以写成：

```text
autoregressive-inspired pairwise motion context
```

只有实现 causal encoder 后，才更适合把正式方法称为：

```text
Autoregressive Motion Context, AMC
```

## 2026-06-30 AMC 三数据集适配分析

当前 AMC 方案不能简单把 DNA 的 `[97, 98, 99, 100]` 连续窗口解释直接套到 I3D / ZJU。原因是当前 `generate_time_steps()` 会按脚本里的 `minimal_time_step / max_time_step / time_step_num` 生成原始历史尺度，不同数据集的尺度语义不同。

当前脚本默认设置：

```text
DNA-Rendering:
  seq_len = 8
  time_step_num = 3
  time_steps = [3, 2, 1]

I3D-Human:
  seq_len = 8
  time_step_num = 3
  time_steps = [42, 33, 24]

ZJU-MoCap:
  seq_len = 3
  time_step_num = 2
  time_steps = [6, 3]
```

### 1. AMC-pair 的统一定义

给定原始历史点：

```text
h_i = t - s_i
```

baseline 原始 motion channel 是：

```text
h_i -> t
```

AMC-pair 额外加入历史点之间的 pairwise motion：

```text
h_i -> h_j, where h_i is older than h_j
```

如果原始历史点数量是 `N = time_step_num`，则：

```text
motion_cond_time_step_num = N + C(N, 2)
```

例如 `N=3` 时为 6 个 channel：

```text
[h1->t, h2->t, h3->t, h1->h2, h1->h3, h2->h3]
```

注意：这只是 autoregressive-inspired pairwise context。若 encoder 仍然是当前 flatten + MLP，它不会严格执行 `h1 -> h2 -> h3 -> t` 的递推。

### 1.1 `amc_pair` 和 `amc_scale` 的命名区别

这两个名字不是两个完全不同的代码模块，更准确地说是同一类第一版 AMC 的两个侧重点：

```text
amc_pair:
  强调新增的 condition 类型。
  即在原始 h_i -> t 之外，加入 h_i -> h_j 的历史内部 pairwise motion。

amc_scale:
  强调跨数据集适配策略。
  即不强行统一成连续 [t-3,t-2,t-1,t]，而是复用各数据集原始 time_steps。
```

因此第一版实现时，实验名建议用：

```text
amc_pair
```

论文/方法描述里可以写成：

```text
scale-adaptive AMC-pair
```

含义是：

```text
用 pairwise motion context，
但历史点来自每个数据集自己的原始 motion scales。
```

避免误解：

```text
amc_pair 不是 causal encoder；
amc_scale 也不是单独的新网络，只是 time_steps 选择策略。
```

### 2. DNA-Rendering 适配

DNA 当前 `[3,2,1]` 最适合 AMC-pair，因为它自然对应短期连续窗口：

```text
t-3, t-2, t-1, t
```

以 `t=100` 为例：

```text
baseline:
  97->100, 98->100, 99->100

AMC-pair:
  97->100, 98->100, 99->100,
  97->98, 97->99, 98->99
```

这版可以作为 AMC 第一阶段最低风险实验：

```text
motion_cond_mode = amc_pair
time_step_num = 3
motion_cond_time_step_num = 6
renderer/loss 不动
```

但要记住一个解释风险：baseline 的 `full_1` 在 `seq_len` 维度里已经包含一部分相邻短期运动，例如 `99->100 / 98->99 / 97->98`。AMC-pair 的价值不是凭空新增这些 delta，而是把同一个局部窗口内的历史内部运动显式放到同一个 condition slot 中。

### 3. I3D-Human 适配

I3D 当前 `[42,33,24]` 不是连续短期窗口。以 `t=100` 为例，历史点是：

```text
58, 67, 76, 100
```

AMC-pair 会变成：

```text
58->100, 67->100, 76->100,
58->67, 58->76, 67->76
```

这表达的是“多尺度历史状态之间的运动趋势”，不是 DNA 那种 `97->98->99->100` 短期自回归。论文里如果三数据集统一叫 AMC，需要把 I3D 表述为：

```text
multi-scale autoregressive motion context
```

而不是逐帧短期 autoregression。

如果想让 I3D 也做短期连续 AMC，就必须新增独立于原始 `time_steps` 的局部窗口参数，例如：

```text
amc_local_window = 3
amc_local_step = 1 或数据集帧间隔
```

这会改变 baseline 的 motion sampling 语义，风险更高，不建议作为第一版。

### 4. ZJU-MoCap 适配

ZJU 当前只有两个历史尺度 `[6,3]`，因此统一 AMC-pair 只有：

```text
t-6->t, t-3->t, t-6->t-3
```

通道数为：

```text
2 + C(2,2) = 3
```

这比 DNA/I3D 的 6 channel 弱很多。不要为了强行对齐 6 channel 直接把 ZJU 的 `time_step_num` 从 2 改成 3；那会改变原始采样尺度，baseline 公平性变差。

ZJU 第一版建议保持：

```text
time_step_num = 2
motion_cond_time_step_num = 3
```

论文和实验表述中说明：AMC 根据各数据集原始历史尺度自适应扩展，ZJU 因原始尺度为 2，所以 pairwise channel 较少。

### 5. 两种可选路线

低风险统一路线：

```text
AMC-scale / AMC-pair:
  使用每个数据集原始 time_steps。
  DNA: [3,2,1] -> 6 channels
  I3D: [42,33,24] -> 6 channels
  ZJU: [6,3] -> 3 channels
```

优点是公平、隔离、实现风险低；缺点是三数据集语义不完全一致，I3D 不是短期连续自回归，ZJU 增量较弱。

更强但风险更高的路线：

```text
AMC-local:
  每个数据集额外定义局部连续窗口。
  尽量都构造 [t-3, t-2, t-1, t] 或等价局部点。
```

优点是 AMC 语义统一；缺点是会改变原始 SeqAvatar 的 motion scale 设计，并引入更多边界帧、缺帧、split 内查找和公平性问题。

### 6. 当前建议

如果接下来替换 MSTI，建议不要一开始做三数据集统一局部窗口，而是：

```text
第一阶段:
  实现 AMC-pair / AMC-scale。
  复用现有 time_steps。
  自动推导 motion_cond_time_step_num = N + C(N,2)。
  pose/xyz 同时扩展。
  renderer/loss/Part-MoE 不动。

第二阶段:
  如果 AMC-pair 有提升，再实现 AMC causal encoder。
  用结构化 h1 -> h2 -> h3 聚合支撑 Autoregressive Motion Context 的正式命名。
```

实验命名建议：

```text
amc_pair:
  低风险 pairwise channel 版本。

amc:
  后续 causal encoder 正式版本。
```

不要把第一版简单 concat 的 `amc_pair` 过度表述成严格自回归。

## 2026-06-30 AMC-pair 方案补充问题

用户提供的 AMC-pair checklist 基本覆盖了实现隔离、`time_step_num`、train/render 一致、pose/xyz 同步扩展、channel 顺序、per-pair dt、cache、日志和输出目录等主要风险。除此之外，还需要特别注意以下补充问题。

### 1. `seq_len` 下 baseline 并不存在统一 anchor

这是当前最容易被忽略的问题。

原始 `get_seq_pose_xyz_cond()` 对每个 `time_step=s` 和 `seq_i=i` 使用：

```text
cur_id    = t - i * s
former_id = t - (i + 1) * s
```

因此以 DNA `time_steps=[3,2,1]`、`t=100` 为例：

```text
i=0:
  s=3: 97->100
  s=2: 98->100
  s=1: 99->100

i=1:
  s=3: 94->97
  s=2: 96->98
  s=1: 98->99
```

只有 `i=0` 时三个尺度共享同一个 target `t=100`。到 `i=1` 后，三个尺度的 `cur_id` 分别是 97、98、99，并不是同一个 anchor。

所以 AMC-pair 不能简单写成：

```text
for seq_i:
  anchor_id = 当前 slot 的帧
  hist_ids = anchor_id - [3,2,1]
```

除非明确接受它改变 baseline 的 `seq_len` 时间网格。

### 2. 必须先决定两种语义之一

方案 A：保持 baseline full channel 完全不变。

```text
优点:
  可以说 AMC-pair 是在 baseline condition 上额外加 pairwise channel。

问题:
  i>0 时不同 time_step 没有共同 anchor，history-history pair 的定义不自然。
  需要明确 pairwise 是围绕哪个 target/window 构造。
```

方案 B：改成局部 causal window。

例如以最大跨度为 anchor stride：

```text
i=0: [97,98,99,100]
i=1: [94,95,96,97]
i=2: [91,92,93,94]
```

每个 slot 都构造：

```text
[h1->anchor, h2->anchor, h3->anchor, h1->h2, h1->h3, h2->h3]
```

优点是 AMC 语义最清楚；缺点是这不再严格保留 baseline 的 `full_2/full_1` 序列，因为 baseline 的 `i=1` 中 `full_2/full_1` 是 `96->98 / 98->99`，不是 `95->97 / 96->97`。

结论：实现前必须选定方案，并在实验命名和论文表述中说明。若声称“只新增 pairwise motion，不改变原始 full channel”，就必须做单元测试确认 AMC 输出的前 `N` 个 channel 与 baseline 完全一致。

### 3. 三数据集脚本需要统一接入方式

当前 DNA 脚本已经有 MSTI 相关的 `motion_cond_time_step_num` 推导，但 I3D / ZJU 脚本还没有同等的 motion-condition 分支。AMC-pair 如果要跑三数据集，需要三个脚本都显式支持：

```text
amc_pair mode
use_amc_pair
motion_cond_time_step_num 自动推导
logs/amc_pair
output/<dataset>/<seq>/amc_pair/<RUN_TIME>
```

不要只在 DNA 中实现，否则后续三数据集实验会出现配置不一致。

### 4. `resolve_motion_condition_args()` 需要处理互斥关系

当前参数解析主要围绕 MSTI：

```text
use_msti
msti_mode
motion_cond_time_step_num
```

加入 AMC-pair 后不能让这些开关互相叠加成未定义状态。第一版建议显式禁止：

```text
use_msti == 1 and use_amc_pair == 1
```

除非后续专门做 `msti_amc_pair` 组合实验。

更稳的长期形式是统一成：

```text
motion_cond_mode = none / msti_lite / msti_full / amc_pair
```

但为了不影响已有 MSTI 实验，第一版也可以保留旧参数，同时加严格 assert。

### 5. I3D 的 pairwise dt 尺度会很特殊

I3D 当前 `[42,33,24]` 下，history-history pair 的间隔是：

```text
t-42 -> t-33: dt=9
t-42 -> t-24: dt=18
t-33 -> t-24: dt=9
```

而 history-current 是：

```text
dt=42,33,24
```

如果 xyz 按 dt 归一化，I3D 新增 pairwise channel 的数值分布可能明显不同于原始 full channel。需要打印统计：

```text
mean/std/max of pose delta per channel
mean/std/max of xyz delta per channel
```

否则可能出现提升或下降来自数值尺度，而不是 AMC 结构。

### 6. 边界退化比例需要统计

序列开头 clamp 后可能出现大量重复 id：

```text
from_id == to_id
```

尤其 I3D 最大跨度 42、ZJU 最大跨度 6 时，早期帧会有更多退化 pair。实现后建议输出：

```text
degenerate_pair_count / total_pair_count
```

如果比例较高，需要确认这些帧是否进入训练、是否影响训练早期分布。

### 7. cache key 建议独立命名并记录版本

不要复用 baseline 的：

```text
cur-former
```

也不要复用 MSTI 的：

```text
msti_real_dt_v1:cur-former
```

AMC-pair 建议使用：

```text
amc_pair_dt_v1:to_id-from_id
```

如果后续改变 anchor 策略、dt 归一化或 channel 顺序，需要升级版本号，避免旧 cache 混入。

### 8. 公平性需要区分“新增信息”和“重排信息”

DNA 中 `97->98`、`98->99` 这类短期运动并非完全新信息，因为 baseline 的 `full_1` 在 `seq_len` 维度中已经包含它们。AMC-pair 更准确的创新点是：

```text
把历史内部运动重排到同一个局部窗口的 condition channel 中，
让 encoder 在同一个 seq slot 内看到 history-history relation。
```

因此对比时最好加一个轻量消融：

```text
amc_adjacent:
  只加相邻历史 pair，例如 h1->h2, h2->h3

amc_pair:
  加全部 pair，例如 h1->h2, h1->h3, h2->h3
```

这样可以回答 `h1->h3` 是否只是冗余通道。

## 2026-06-30 DNA AMC-pair 方案 A 实现记录

本次按用户要求先选择方案 A：

```text
AMC-pair / baseline_full
```

核心含义：

```text
前 N 个 full channel 保持 baseline 的原始 time_step / seq_len 时间网格不变；
后面追加由这些 full channel 的 former endpoints 构造的 history-history pairwise channel。
```

以 DNA `time_steps=[3,2,1]`、`t=100` 为例，`i=0`：

```text
baseline full:
  97->100, 98->100, 99->100

AMC-pair additional:
  97->98, 97->99, 98->99
```

对 `i>0`，不改 baseline full channel；新增 pairwise channel 使用该 `seq_i` 下各 full channel 的 `former_id`。例如：

```text
i=1 baseline full:
  94->97, 96->98, 98->99

i=1 AMC-pair additional:
  94->96, 94->98, 96->98
```

这保留了方案 A 的“full channel 不改”，但也意味着 AMC-pair 第一版仍然不是严格的局部 causal window。

### 修改文件

```text
arguments/__init__.py
  新增 use_amc_pair / amc_pair_mode。
  默认关闭。
  与 MSTI 互斥。
  自动推导:
    motion_cond_time_step_num = N + N * (N - 1) // 2

scene/__init__.py
  将 use_amc_pair / amc_pair_mode 传入 motion_cond_options。

scene/dataset_readers.py
  新增 get_seq_pose_xyz_cond_amc_pair()。
  baseline / MSTI 分支保持独立。
  AMC cache key 使用:
    amc_pair_scheme_a_raw_v1:cur-former

scripts/exps_dnarendering.sh
  新增 amc_pair 模式。
  日志目录:
    logs/AMC
  DNA AMC 默认六序列:
    0044_11 0051_09 0206_04 0813_05 0007_04 0019_10
  AMC 模式下 train/render 失败会记录 warning 并跳过当前序列继续下一个。
```

### 验证

已通过：

```text
/media/image/mxz/.conda/envs/seqavatar/bin/python -m py_compile arguments/__init__.py scene/__init__.py scene/dataset_readers.py
bash -n scripts/exps_dnarendering.sh
git diff --check
```

DNA-like dummy shape test：

```text
time_steps = {3: 8, 2: 8, 1: 8}
baseline pose/xyz = (1, 8, 3, 24, 3) / (5, 8, 3, 3)
AMC-pair pose/xyz = (1, 8, 6, 24, 3) / (5, 8, 6, 3)
front_channels_match_baseline = True
```

参数解析验证：

```text
use_amc_pair=True, time_step_num=3 -> motion_cond_time_step_num=6
use_msti=True and use_amc_pair=True -> ValueError
```

### DNA 六序列启动记录

使用 tmux 在 GPU 2 / GPU 3 启动：

```text
GPU 2:
  session = seqavatar_amc_pair_gpu2_20260630_213132
  sequences = 0044_11 0051_09 0206_04
  log = logs/AMC/20260630_213132_DNA-Rendering_amc_pair.log

GPU 3:
  session = seqavatar_amc_pair_gpu3_20260630_213134
  sequences = 0813_05 0007_04 0019_10
  log = logs/AMC/20260630_213134_DNA-Rendering_amc_pair.log
```

启动日志已确认：

```text
Mode = amc_pair
USE_AMC_PAIR = 1
AMC_PAIR_MODE = baseline_full
TIME_STEP_NUM(base) = 3
MOTION_COND_TIME_STEP_NUM = 6
DENSIFY_UNTIL_ITER = 1800
FINAL_EVAL_ONLY = 1
```

启动后状态：

```text
GPU 2:
  0044_11 已进入训练进度。

GPU 3:
  0813_05 在 Loading Training Cameras 阶段 CUDA OOM。
  脚本按 AMC-pair 规则跳过该序列，继续运行 0007_04。
```

## 2026-06-30 DNA AMC-pair OOM 诊断

本次 `amc_pair` 首轮 DNA 六序列并不是全部同一种失败。

GPU 3 日志：

```text
0813_05 / 0007_04 / 0019_10 都在 Loading Training Cameras 阶段 OOM。
```

原因不是 AMC condition shape 错，而是物理 GPU 3 上已有其它任务占用显存。日志中虽然写 `GPU 0`，但这是因为 `CUDA_VISIBLE_DEVICES=3` 后，物理 GPU 3 在进程内部被重编号为 local GPU 0。

典型日志：

```text
Tried to allocate 20.00 MiB
only 7-18 MiB free
Process 1458222 has 13.21 GiB memory in use
```

也就是说，相机图片还没完全搬到 CUDA，就已经没有足够显存。

GPU 2 日志：

```text
0044_11 / 0051_09 进入训练后 OOM。
```

原因是 AMC-pair 把 motion condition channel 从 3 扩到 6，`SeqPoseEncoder / SeqXYZEncoder` 输入和 `cond_dict` 显存都会增加；再加上 DNA 图片默认加载到 CUDA，0044/0051 训练峰值已经接近 21GB，后续还需要一次 2.6-3.0GB 的大块分配，因此 OOM。

典型日志：

```text
process has 20.78-21.01 GiB memory in use
Tried to allocate 2.62-2.96 GiB
only 2.48-2.72 GiB free
```

当前建议：

```text
1. GPU 3 的失败序列不是代码问题，等 GPU 3 空出来后单独重跑。
2. GPU 2 的 0044/0051 需要降显存重跑，优先尝试 IMAGE_DATA_DEVICE=cpu 或 SKIP_LOAD_TEST_CAMERAS=1。
3. 如果仍 OOM，再考虑降低 seq_xyz_knn 或增加显存更空的卡。
```

## 2026-06-30 DNA AMC-pair densify_until_iter=1500 重跑

用户要求将 `amc_pair` 的 `densify_until_iter` 改为 1500 后重跑 DNA 六序列。

脚本修改：

```text
scripts/exps_dnarendering.sh
  原全局 densify_until_iter=1800 保持不变。
  仅在 use_amc_pair=1 时默认:
    densify_until_iter=${AMC_DENSIFY_UNTIL_ITER:-1500}
```

这样只影响 `amc_pair`，不改变 baseline / MSTI / Part-MoE / Part-MoE+MSTI 的默认 densify 设置。

已验证：

```text
bash -n scripts/exps_dnarendering.sh
```

由于 GPU 3 仍被其它 `src/trainer.py` 占用约 13.5GB，继续使用 GPU 3 会重复 Loading Cameras 阶段 OOM。本次改用空闲的 GPU 1 和 GPU 2 重跑六序列：

```text
GPU 2:
  session = seqavatar_amc_pair_d1500_gpu2_20260630_222432
  sequences = 0044_11 0051_09 0206_04
  log = logs/AMC/20260630_222432_DNA-Rendering_amc_pair.log

GPU 1:
  session = seqavatar_amc_pair_d1500_gpu1_20260630_222434
  sequences = 0813_05 0007_04 0019_10
  log = logs/AMC/20260630_222434_DNA-Rendering_amc_pair.log
```

启动日志已确认：

```text
DENSIFY_UNTIL_ITER = 1500
MOTION_COND_TIME_STEP_NUM = 6
USE_AMC_PAIR = 1
AMC_PAIR_MODE = baseline_full
```

启动后状态：

```text
GPU 2:
  0044_11 已进入训练。

GPU 1:
  0813_05 已进入训练。
```

## 2026-07-01 DNA AMC-pair densify_until_iter=1500 指标

本次汇总使用最终 render/eval 阶段的测试行：

```text
[ITER 25000] Evaluating novelview #120: PSNR ... SSIM ... LPIPS ...
```

对应日志：

```text
logs/AMC/20260630_222432_DNA-Rendering_amc_pair.log
logs/AMC/20260630_222434_DNA-Rendering_amc_pair.log
```

六个序列均完成。

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | 32.9405 | 0.977879 | 21.3269 |
| 0051_09 | 28.6317 | 0.971326 | 31.1716 |
| 0206_04 | 31.3774 | 0.969677 | 34.1438 |
| 0813_05 | 36.0377 | 0.986744 | 18.8930 |
| 0007_04 | 29.4856 | 0.958052 | 45.2343 |
| 0019_10 | 35.2100 | 0.980821 | 21.3176 |
| Mean | 32.2805 | 0.974083 | 28.6812 |

## 2026-07-01 DNA 脚本序列选择约定

用户明确不需要按实验模式自动选择不同序列。后续 DNA 脚本统一使用一套默认序列；如果某次实验要跑其它序列，手动改这一处或使用 `SEQUENCES_OVERRIDE`。

当前 `scripts/exps_dnarendering.sh` 逻辑：

```text
if SEQUENCES_OVERRIDE is set:
  使用 SEQUENCES_OVERRIDE
else:
  使用统一默认列表
```

已删除此前 `amc_pair` 专属的六序列分支，避免出现“某个实验自动对应某些序列”的隐式行为。

更正确认：

```text
2026-07-01 用户发现脚本中仍残留 amc_pair 专属序列分支。
已再次检查并删除该分支。
当前实际文件中只保留:
  if SEQUENCES_OVERRIDE is set -> 使用覆盖序列
  else -> 使用统一默认序列
```

同时确认 `densify_until_iter=1500` 只在 `use_amc_pair=1` 时生效，默认其它 DNA 实验仍为 1800。

## 2026-07-01 AMC-pair 代码改动总结

AMC-pair 目标：

```text
在不改 renderer / loss / CameraInfo / Part-MoE / MSTI 默认路径的前提下，
给 SeqAvatar 的 motion condition 增加 history-history pairwise motion。
```

第一版采用方案 A `baseline_full`：

```text
前 N 个 full channel 完全保持 baseline 的原始 time_step / seq_len 时间网格；
后面追加由这些 full channel 的 former endpoints 构造的 pairwise channel。
```

以 DNA `time_steps=[3,2,1]`、`t=100`、`i=0` 为例：

```text
baseline full:
  97->100, 98->100, 99->100

AMC-pair:
  97->100, 98->100, 99->100,
  97->98, 97->99, 98->99
```

### 1. `arguments/__init__.py`

改动：

```text
新增参数:
  use_amc_pair
  amc_pair_mode = baseline_full

新增规则:
  use_amc_pair 和 use_msti 互斥
  amc_pair 自动推导 motion_cond_time_step_num
```

推导公式：

```text
N_cond = N + C(N, 2)
```

DNA 中：

```text
time_step_num = 3
motion_cond_time_step_num = 3 + 3 = 6
```

目的：

```text
不用手填 channel 数，避免 train/render 维度不一致；
默认关闭，隔离 baseline / MSTI / Part-MoE；
禁止 AMC-pair 和 MSTI 同时打开，避免未定义的混合通道语义。
```

### 2. `scene/__init__.py`

改动：

```text
把 use_amc_pair / amc_pair_mode 放进 motion_cond_options，
传给 dataset reader。
```

目的：

```text
Scene 构建 cond_dict 时能选择 AMC-pair 分支；
train 和 render 都走同一套参数，保持 condition 构造一致。
```

### 3. `scene/dataset_readers.py`

改动：

```text
在 get_seq_pose_xyz_cond() 中新增 AMC-pair 分支；
新增 get_seq_pose_xyz_cond_amc_pair()；
baseline 分支和 MSTI 分支保持独立。
```

AMC-pair 分支做的事：

```text
1. 先按 baseline 原逻辑构造前 N 个 full channel；
2. 记录每个 full channel 的 former_id；
3. 对 former_id 两两组合，追加 history-history pairwise delta；
4. pose 和 xyz 同步扩展；
5. assert 输出 channel 数等于 N + C(N,2)。
```

cache key：

```text
amc_pair_scheme_a_raw_v1:cur-former
```

目的：

```text
让 encoder 在同一个 seq slot 中看到历史帧之间的运动关系；
同时保证前 N 个 channel 与 baseline 保持一致，便于公平消融。
```

注意：

```text
当前 AMC-pair 仍然使用原 SeqPoseEncoder / SeqXYZEncoder 的 flatten + MLP。
它不是正式 causal encoder，也不是严格自回归结构。
```

### 4. `scripts/exps_dnarendering.sh`

改动：

```text
新增运行模式:
  amc_pair

新增日志目录:
  logs/AMC

新增命令行参数:
  --use_amc_pair
  --amc_pair_mode baseline_full

新增自动 channel:
  motion_cond_time_step_num = time_step_num + time_step_num * (time_step_num - 1) / 2

仅 AMC-pair 默认:
  densify_until_iter = ${AMC_DENSIFY_UNTIL_ITER:-1500}
```

当前序列选择：

```text
不按实验模式自动切序列。
如果 SEQUENCES_OVERRIDE 存在，使用覆盖序列；
否则使用统一默认序列。
```

目的：

```text
让 AMC-pair 可以用同一个 DNA 脚本启动；
日志与 baseline / MSTI / Part-MoE 隔离；
降低 AMC-pair 显存峰值；
避免不同实验隐式绑定不同序列。
```

### 5. 没有改的部分

AMC-pair 没有改：

```text
renderer
loss
CameraInfo
RGB / mask / camera 数据
Part-MoE 路由和专家逻辑
MSTI 现有分支
SeqPoseEncoder / SeqXYZEncoder 结构
```

因此它是一个 condition-level 消融：

```text
只改变输入给 motion encoder 的条件通道；
不改变监督目标和渲染流程。
```

## 2026-07-01 DNA-Rendering baseline rerun metrics

日志：

```text
/media/image/mxz/human/SeqAvatar/logs/20260701_005912_DNA-Rendering_orginal.log
```

说明：

```text
该日志完整结束，包含 5 个默认序列：
0051_09, 0206_04, 0813_05, 0007_04, 0019_10
未包含 0044_11。
日志头部记录 DENSIFY_UNTIL_ITER: 1800。
下面采用 render.py 最终 novelview 评价行，而不是训练结束时的中间评价行。
```

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0051_09 | 28.6849 | 0.971489 | 31.1649 |
| 0206_04 | 31.3086 | 0.969506 | 33.9268 |
| 0813_05 | 36.0529 | 0.986883 | 18.4098 |
| 0007_04 | 29.5087 | 0.958475 | 44.2631 |
| 0019_10 | 35.1483 | 0.980624 | 21.5088 |
| Mean | 32.1407 | 0.973395 | 29.8547 |

## 2026-07-01 DNA-Rendering baseline rerun metrics, densify 1500

日志：

```text
/media/image/mxz/human/SeqAvatar/logs/20260701_130518_DNA-Rendering_orginal.log
```

说明：

```text
该日志完整结束，包含 5 个默认序列：
0051_09, 0206_04, 0813_05, 0007_04, 0019_10
未包含 0044_11。
日志头部记录 DENSIFY_UNTIL_ITER: 1500。
下面采用 render.py 最终 novelview 评价行，而不是训练结束时的中间评价行。
```

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0051_09 | 28.6714 | 0.971448 | 31.0921 |
| 0206_04 | 31.3799 | 0.969798 | 34.0235 |
| 0813_05 | 36.0867 | 0.986893 | 18.4637 |
| 0007_04 | 29.5336 | 0.958335 | 45.3894 |
| 0019_10 | 35.2209 | 0.980696 | 21.2649 |
| Mean | 32.1785 | 0.973434 | 30.0467 |

### densify 1500 vs 1800 baseline observation

对比：

```text
1500 - 1800 mean:
PSNR  +0.0378
SSIM  +0.000039
LPIPS*1000 +0.1920
```

解释：

```text
1500 在 PSNR / SSIM 上略高，但 LPIPS*1000 略差，因此不能简单说 1500 全面更好。
差值很小，可能包含随机初始化、训练采样、CUDA 非确定性带来的波动。
```

代码行为：

```text
train.py 中只有 iteration < densify_until_iter 时才继续统计并执行 densify_and_prune。
densification_interval=100，densify_from_iter=400。
因此 1500 相比 1800 少了约 3 轮 densify/prune 机会。
```

最终日志中的 #pts：

| Sequence | d1800 #pts | d1500 #pts |
|---|---:|---:|
| 0051_09 | 67545 | 50516 |
| 0206_04 | 58631 | 42792 |
| 0813_05 | 50717 | 39653 |
| 0007_04 | 35497 | 27199 |
| 0019_10 | 43261 | 34400 |

判断：

```text
1500 的 Gaussian 数量更少，模型容量更小，可能减少对训练视角/训练 mask 边界/局部噪声的过拟合；
但这也可能损失细节，所以 LPIPS 没有同步变好。
如果论文对比 AMC-pair 使用 densify 1500，baseline 也应该使用 densify 1500 才公平。
```

## 2026-07-01 DNA-Rendering baseline 0044_11 metrics, densify 1500

日志：

```text
/media/image/mxz/human/SeqAvatar/logs/20260701_162750_DNA-Rendering_orginal.log
```

说明：

```text
该日志完整结束，只包含序列 0044_11。
日志头部记录 DENSIFY_UNTIL_ITER: 1500。
下面采用 render.py 最终 novelview 评价行，而不是训练结束时的中间评价行。
```

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | 32.9741 | 0.977915 | 21.3970 |

## 2026-07-01 AMC-pair vs baseline, densify 1500 analysis

公平对比设置：

```text
baseline: original, densify_until_iter=1500
AMC-pair: amc_pair, densify_until_iter=1500
DNA-Rendering six sequences
```

逐序列差值，AMC-pair - baseline：

| Sequence | dPSNR | dSSIM | dLPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | -0.0336 | -0.000036 | -0.0701 |
| 0051_09 | -0.0397 | -0.000122 | +0.0795 |
| 0206_04 | -0.0025 | -0.000121 | +0.1203 |
| 0813_05 | -0.0490 | -0.000149 | +0.4293 |
| 0007_04 | -0.0480 | -0.000283 | -0.1551 |
| 0019_10 | -0.0109 | +0.000125 | +0.0527 |
| Mean | -0.0306 | -0.000098 | +0.0761 |

判断：

```text
AMC-pair 当前没有带来稳定收益。
差值很小，不能证明运动上下文方向失败，但说明当前“追加历史-历史 pair channel + 原 flatten MLP encoder”的形式不够有效。
```

可能原因：

```text
1. 额外 pair channel 与原 full channel 信息高度冗余。
2. pair channel 的语义不是 former -> current，而是 history -> history，和 baseline channel 混在同一个 encoder 输入维度里可能造成语义不一致。
3. SeqPoseEncoder / SeqXYZEncoder 仍是 flatten + MLP，没有真正的 causal / autoregressive 结构，不能强制 97 -> 98 -> 99 -> 100 的递进建模。
4. DNA 原始 time_steps=[3,2,1] 已经很密，full_1/full_2/full_3 可能足够覆盖局部运动。
5. 新增 channel 增加输入维度和噪声，模型可能学会忽略，或者轻微过拟合。
```

后续优化优先级：

```text
1. 不要继续只堆 pair channel，优先做带归纳偏置的版本。
2. 加 learnable gate / attention，让模型自动决定 pair channel 权重，并观察 gate 是否接近 0。
3. 将 full motion 和 pair motion 分两个 encoder，再 late fusion，避免语义混在一个 flatten 输入里。
4. 尝试 adjacent chain：97->98, 98->99, 99->100，而不是所有 pair。
5. 尝试二阶运动/加速度：比较相邻 motion velocity 的变化，比 pair delta 更直接表达运动趋势。
6. 最终如果要称 AMC，建议做 causal encoder：h97 -> h98 -> h99 -> h100，而不是 AMC-pair。
```

## 2026-07-01 AMC-causal implementation on DNA-Rendering

目标：

```text
实现正式 AMC causal encoder；
只对高运动关节 / 高运动区间启用 causal residual；
低运动区域保持 baseline motion encoder 输出；
先只在 DNA-Rendering 脚本中暴露消融模式；
不影响 baseline / MSTI / AMC-pair / Part-MoE。
```

实验名：

```text
amc_causal
```

日志目录：

```text
/media/image/mxz/human/SeqAvatar/logs/AMC
```

核心设计：

```text
motion_cond_time_step_num 仍为 time_step_num。
DNA 中 time_step_num=3，因此 AMC-causal 输入仍是 3 个 baseline channel:
  [full_3, full_2, full_1]

AMC-causal 不再像 AMC-pair 一样把 channel 扩成 6。
它使用 baseline 已有的 full_1 短步运动在 seq_len 维度上的相邻片段：
  97->98, 98->99, 99->100

做 old-to-current causal GRU：
  h98 = f(97->98)
  h99 = f(h98, 98->99)
  h100 = f(h99, 99->100)
```

高运动启用策略：

```text
SeqPoseEncoder:
  对最近 amc_causal_window=3 个 full_1 短步 pose delta 计算每个 joint 的运动幅度；
  用 mean + alpha * std 得到自适应高运动阈值；
  只把高运动 joint 的 causal residual 加到 baseline pose feature 上。

SeqXYZEncoder:
  对每个 Gaussian 附近 KNN 顶点的最近 full_1 短步 xyz delta 计算运动幅度；
  用同样的自适应 gate；
  只把高运动 point/interval 的 causal residual 加到 baseline xyz feature 上。

输出形式：
  final_feature = baseline_feature + motion_gate * causal_delta
```

默认超参数：

```text
amc_causal_mode = gated_residual
amc_causal_window = 3
amc_motion_gate_alpha = 1.0
amc_motion_gate_temp = 0.5
motion_cond_time_step_num = 3
densify_until_iter = 1500
```

已修改文件：

```text
arguments/__init__.py
scene/__init__.py
scene/dataset_readers.py
scene/gaussian_model.py
nets/mlp_delta_non_rigid.py
scripts/exps_dnarendering.sh
```

隔离规则：

```text
use_msti / use_amc_pair / use_amc_causal 三者互斥。
baseline 不开 use_amc_causal 时不会创建 causal residual 参数。
MSTI 和 AMC-pair 仍走原来的分支。
Part-MoE 路由不变，AMC-causal 只影响 non-rigid deformer 前面的 motion feature 编码。
```

验证：

```text
bash -n scripts/exps_dnarendering.sh 通过。
py_compile: arguments / scene / dataset_readers / gaussian_model / mlp_delta_non_rigid 通过。
dummy shape test:
  baseline cond steps = 3
  causal cond steps = 3
  SeqPoseEncoder baseline/amc output = (1, 32)
  SeqXYZEncoder baseline/amc output = (1, 4, 96)
  NonrigidDeformer output = [(1, 4, 3), (1, 4, 4), (1, 4, 3)]
```

启动实验：

```text
时间: 20260701_180553
日志: /media/image/mxz/human/SeqAvatar/logs/AMC/20260701_180553_DNA-Rendering_amc_causal.log

GPU 1:
  0044_11, 0051_09, 0206_04

GPU 2:
  0813_05, 0007_04, 0019_10
```

备注：

```text
两个 tmux 在同一秒启动，因此共享同一个 RUN_TIME 和同一个全局日志文件。
输出目录按序列隔离：
  output/DNA-Rendering/<sequence>/amc_causal/20260701_180553/

启动后已确认：
  USE_AMC_CAUSAL=1
  MOTION_COND_TIME_STEP_NUM=3
  DENSIFY_UNTIL_ITER=1500
  两个首个序列均进入 Training 进度，无初始 OOM。
```

### AMC-causal metrics, 20260701_180553

日志：

```text
/media/image/mxz/human/SeqAvatar/logs/AMC/20260701_180553_DNA-Rendering_amc_causal.log
```

说明：

```text
六个 DNA-Rendering 序列全部完成。
两个 tmux 进程共享同一个日志文件，因此日志中训练进度有交错。
下面采用每个 Finished sequence 前最近一条 render.py 最终 novelview 评价行。
```

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | 32.9390 | 0.977826 | 21.5267 |
| 0051_09 | 28.6215 | 0.971177 | 31.1707 |
| 0206_04 | 31.3885 | 0.970309 | 33.2033 |
| 0813_05 | 36.0316 | 0.986819 | 18.6497 |
| 0007_04 | 29.5400 | 0.958471 | 45.2457 |
| 0019_10 | 35.2663 | 0.980813 | 21.3089 |
| Mean | 32.2978 | 0.974236 | 28.5175 |

与同 densify_until_iter=1500 baseline 的差值，AMC-causal - baseline：

| Sequence | dPSNR | dSSIM | dLPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | -0.0351 | -0.000089 | +0.1297 |
| 0051_09 | -0.0499 | -0.000271 | +0.0786 |
| 0206_04 | +0.0086 | +0.000510 | -0.8201 |
| 0813_05 | -0.0551 | -0.000074 | +0.1860 |
| 0007_04 | +0.0064 | +0.000136 | -0.1437 |
| 0019_10 | +0.0454 | +0.000117 | +0.0440 |
| Mean | -0.0133 | +0.000055 | -0.0876 |

初步判断：

```text
AMC-causal 相比 AMC-pair 更接近有效：mean SSIM 和 LPIPS*1000 略优于 baseline，PSNR 略低。
提升幅度仍很小，不能算稳定明显收益。
下一步应检查 gate 是否真的只在高运动区域激活；如果 gate 过小或过大，需要调 amc_motion_gate_alpha / temp。
```

### AMC-causal result analysis and optimization plan

当前结果：

```text
AMC-causal - baseline, densify 1500:
PSNR        -0.0133
SSIM        +0.000055
LPIPS*1000  -0.0876
```

判断：

```text
当前 AMC-causal 不是失败，但收益太小。
它说明 causal residual 比 AMC-pair 更有希望，但当前 gate / residual / causal 信息量都偏保守。
```

主要原因：

```text
1. 差值本身接近训练随机波动。
   PSNR -0.0133、SSIM +0.000055 都非常小，不足以证明稳定提升。

2. 当前 causal 分支只用 full_1 的最近 3 段短步运动。
   DNA baseline 已经有 [full_3, full_2, full_1]，并且 seq_len=8。
   因此 97->98, 98->99, 99->100 并不是全新数据，只是换了一种编码方式。

3. residual 是安全保守设计。
   amc_out 采用 zero init，初始完全等于 baseline。
   这能避免伤 baseline，但也会让 causal 分支学习较慢、影响幅度较小。

4. pose gate 最后被平均成一个全局标量。
   代码中 joint_gate 先按 joint 计算，但最后 motion_gate = mean(joint_gate)。
   如果只有手臂/腿部少数关节高运动，平均后 residual 会被稀释。

5. 当前高运动判断是 mean + alpha * std。
   alpha=1.0 可能过严，导致 gate 激活很少；
   也可能在不同序列上阈值不稳定。

6. 训练目标仍是全帧平均 loss。
   如果高运动帧/高运动区域占比不大，causal 分支获得的监督信号会被低运动区域冲淡。

7. 没有 gate 统计输出。
   目前无法判断是 gate 没开、residual 太小，还是 causal 特征本身没有贡献。
```

优先优化：

```text
第一步先加诊断，不急着继续换结构：
  gate mean / max / active ratio
  causal_delta norm
  baseline_feature norm
  residual / baseline norm
  pose gate 与 xyz gate 分开统计

如果 gate active ratio 很低：
  扫 amc_motion_gate_alpha = 0.0 / 0.5 / 1.0
  扫 amc_motion_gate_temp = 0.25 / 0.5 / 1.0

如果 residual / baseline norm 很低：
  把 amc_out 从 zero init 改为 small init；
  或者增加 learnable residual scale，并初始化为小正数。

如果 pose 分支伤害、xyz 分支有效：
  做 amc_causal_xyz_only / amc_causal_pose_only 消融。

如果 full_1 causal 信息冗余：
  加入 acceleration / second-order motion:
    a1 = v98_99 - v97_98
    a2 = v99_100 - v98_99
  让 AMC 提供 baseline 不直接具备的运动趋势信息。

如果全局平均掩盖高运动收益：
  增加 high-motion subset 评价；
  或训练时对高运动帧/高运动区域加采样权重。

如果要更强正式 AMC：
  做 part-aware causal gate。
  用 SMPL joint/vertex part mask 让 arms/legs/hand 等局部高运动 residual 只影响对应高斯，而不是 pose feature 全局平均。
```

下一轮最小风险实验建议：

```text
1. 先不改核心结构，只加 gate/residual 统计输出。
2. 跑一个短 debug 或单序列 0044_11 / 0206_04，确认 gate 是否有效。
3. 再跑 alpha/temp 小网格：
   alpha=0.5,temp=0.5
   alpha=0.0,temp=0.5
   alpha=0.5,temp=1.0
4. 同时做 xyz_only / pose_only，确定主要收益来自哪一路。
```

## 2026-07-01 AMC 是否继续推进的阶段性判断

结论：

```text
AMC 方向可以继续，但不建议现在作为强第二创新点主打。
当前更适合作为候选分支继续验证，优先做高运动/局部区域评估，而不是继续直接改模型结构。
```

原因：

```text
AMC-pair 三项平均指标均弱于 baseline，说明简单追加 history-history pair channel 基本无效。
AMC-causal 比 AMC-pair 更合理，SSIM / LPIPS*1000 略优于同 densify=1500 baseline，但 PSNR 略低，收益太小。
当前结果只能说明 structured temporal modeling 有弱正信号，不足以支撑强主贡献。
```

下一步决策树：

```text
1. 先做 high-motion subset 评价。
   如果 top motion frames 上 LPIPS/SSIM 明显提升，则 AMC-causal 继续保留。
   如果 high-motion subset 也没有提升，则不建议继续主推 AMC。

2. 再做局部区域评价。
   优先看 arms / legs / silhouette boundary。
   如果局部动态区域明显改善，可以把 AMC 定位为 dynamic-region motion module。

3. 再做 Part-MoE + AMC-causal 组合。
   如果组合稳定提升，AMC 可作为 Part-MoE 的时间增强补充。
   如果组合仍弱，则放弃 AMC 作为第二创新点主线。
```

隔离要求：

```text
当前阶段不改 baseline / MSTI / Part-MoE / AMC 已有训练流程。
优先新增独立 evaluation 脚本读取已有输出结果，计算 high-motion subset 和局部区域指标。
如果后续必须改训练代码，必须通过新开关和新 experiment_name 隔离：
  amc_causal_debug
  amc_causal_xyz_only
  amc_causal_pose_only
  amc_accel
  part_moe_leg_amc_causal

所有新开关默认关闭。
不能改变 original / msti / amc_pair / amc_causal / part_moe_leg 的默认行为。
```

## 2026-07-01 AMC 继续/放弃评估脚本

按当前决策树新增只读评估脚本：

```text
scripts/evaluate_amc_decision.py
```

脚本用途：

```text
比较 baseline 与 AMC-causal 的:
  1. overall 指标
  2. top-motion pose subset 指标
  3. low-motion subset 指标
  4. motion score 与逐图指标变化的相关性
  5. 可选 silhouette-boundary PSNR

根据阈值输出:
  CONTINUE_AS_CANDIDATE
  HOLD_AND_DIAGNOSE
  STOP_AS_MAIN
```

隔离原则：

```text
脚本只读 output/、DNA-Rendering/model/ 和可选 render/mask 图片。
不改 train.py / render.py / dataset reader / model / checkpoint。
不启动训练，不启动渲染，不改变 original / MSTI / AMC / Part-MoE 默认流程。
```

默认运行命令：

```bash
cd /media/image/mxz/human/SeqAvatar
/media/image/mxz/.conda/envs/seqavatar/bin/python scripts/evaluate_amc_decision.py
```

当前默认比较口径：

```text
baseline exp = orginal
baseline run:
  0044_11 -> 20260701_162750
  其它五个序列 -> 20260701_130518

candidate exp = amc_causal
candidate run = 20260701_180553
iteration = 25000
split = novelview
top_motion_ratio = 0.2
motion_step = 5
motion_score = pose
```

重要口径说明：

```text
high-motion subset 必须依赖逐视角指标。
当前 render.py 最终复评日志只打印总指标，不保存 per-view JSON。
因此该脚本默认使用 metrics/results_novelview_25000.json 和
metrics/per_viewnovelview_25000.json，也就是 training eval 落盘口径。
不要把脚本输出误写成 render-log 总指标口径。
```

本次默认六序列评估已生成：

```text
logs/AMC/amc_decision_20260701_204852.md
logs/AMC/amc_decision_20260701_204852.json
```

报告主结论：

```text
Decision = STOP_AS_MAIN
```

关键聚合结果，AMC-causal - baseline：

```text
overall:
  dPSNR = -0.0134
  dSSIM = +0.000055
  dLPIPS*1000 = -0.0745

high-motion subset:
  dPSNR = +0.0046
  dSSIM = +0.000070
  dLPIPS*1000 = -0.1903

low-motion subset:
  dPSNR = -0.0109
  dSSIM = +0.000163
  dLPIPS*1000 = -0.2566

boundary_all:
  dPSNR = +0.0039

boundary_high_motion:
  dPSNR = +0.0532
```

判断：

```text
high-motion subset 上没有达到预设的有效提升阈值:
  LPIPS*1000 改善阈值 = -0.5
  SSIM 改善阈值 = +0.0002
  boundary PSNR 改善阈值 = +0.10 dB

因此当前 AMC-causal 不建议继续作为强第二创新点主线推进。
如果继续保留，只应作为候选支线，后续需要更强局部/part-aware gate、
acceleration/second-order motion 或 Part-MoE 组合实验提供新证据。
```

## 2026-07-02 TDP 方案可行性评审

用户提出 TDP：

```text
Temporal Difference Pyramid
时间差分金字塔
```

核心判断：

```text
方案技术上可行。
它比 AMC-pair 更像一个独立 motion condition 消融，因为它不是简单增加 history-history pair，
而是显式加入 displacement / velocity / acceleration 三阶时间差分。
```

需要先固定一个关键表述：

```text
SeqAvatar baseline 不是 TDP。
baseline 是多尺度历史到当前帧的位移型 condition：
  [t-3 -> t, t-2 -> t, t-1 -> t]

TDP 是局部时间窗口上的多阶差分：
  displacement + adjacent velocity + acceleration
```

以 DNA `t=100`、`time_steps=[3,2,1]` 为例，推荐第一版 TDP-local：

```text
P0 = P97
P1 = P98
P2 = P99
P3 = P100

d  = P97 -> P100
v1 = P97 -> P98
v2 = P98 -> P99
v3 = P99 -> P100
a1 = v2 - v1
a2 = v3 - v2

channels = [d, v1, v2, v3, a1, a2]
motion_cond_time_step_num = 6
```

重要注意：

```text
这个 6-channel TDP-local 不是在 baseline [97->100, 98->100, 99->100] 后面追加。
它保留 long displacement 97->100，
但会把 baseline 的 98->100 和 99->100 替换成 adjacent velocity / acceleration 信息。
```

如果想保留 baseline 三个 full channel 再追加 TDP 信息，则应另开模式，例如：

```text
tdp_keep_base:
  [97->100, 98->100, 99->100, v1, v2, v3, a1, a2]
  DNA channel = 8
```

但第一版建议先做低风险版本，同时最终至少补 `tdp_keep_base`：

```text
tdp_local:
  DNA channel = 2 * time_step_num = 6
  ZJU channel = 2 * time_step_num = 4

tdp_keep_base:
  DNA channel = time_step_num + 2 * time_step_num - 1 = 8
  ZJU channel = time_step_num + 2 * time_step_num - 1 = 5
```

跨数据集适配：

```text
给定原始 offsets 按从远到近排列:
  offsets = [s_max, ..., s_min, 0]

每个 seq slot 使用明确 local anchor:
  anchor_i = pose_index - i * s_max

构造局部点:
  p_j = anchor - offsets[j]

displacement:
  p_0 -> p_last

velocity:
  p_j -> p_{j+1}

acceleration:
  v_{j+1} - v_j
```

DNA：

```text
anchor_i = t - i * 3

i=0: [t-3,t-2,t-1,t]
i=1: [t-6,t-5,t-4,t-3]
i=2: [t-9,t-8,t-7,t-6]

channels = 1 + 3 + 2 = 6
```

ZJU：

```text
[6,3,0] -> [t-6,t-3,t]
channels = 1 + 2 + 1 = 4
```

I3D 需要谨慎：

```text
[42,33,24,0] -> [t-42,t-33,t-24,t]
intervals = 9, 9, 24
```

I3D 的速度间隔不均匀，`v3 - v2` 会混合不同 dt 的速度。若后续跑 I3D，必须明确：

```text
velocity 是否按 dt 归一化；
acceleration 是否按相邻 velocity 的时间间隔再归一化；
是否只先在 DNA/ZJU 做 TDP-local。
```

代码落点和现有基础：

```text
arguments/__init__.py:
  已有 resolve_motion_condition_args()
  已经能自动推导 motion_cond_time_step_num
  需要新增 use_tdp / tdp_mode，并和 use_msti / use_amc_pair / use_amc_causal 互斥。

scene/__init__.py:
  已经把 motion_cond_options 传给 dataset reader。
  需要把 use_tdp / tdp_mode 加入 motion_cond_options。

scene/dataset_readers.py:
  get_seq_pose_xyz_cond() 已经有 baseline / MSTI / AMC-pair 分支。
  新增 get_seq_pose_xyz_cond_tdp() 最自然。

scene/gaussian_model.py:
  已经用 motion_cond_time_step_num 构造 NonrigidDeformer。
  TDP 不需要额外改 deformer 构造逻辑。

nets/mlp_delta_non_rigid.py:
  SeqPoseEncoder / SeqXYZEncoder 已经按传入 time_step_num 建 Linear 输入维度。
  TDP 只要 condition channel 数正确，就能复用 flatten + MLP。

scripts/exps_dnarendering.sh:
  新增 tdp 模式，输出目录建议:
    output/DNA-Rendering/<seq>/tdp_local/<RUN_TIME>/
  日志目录建议:
    logs/TDP/<RUN_TIME>_DNA-Rendering_tdp_local.log
```

实现时需要固定的定义：

```text
channel order:
  tdp_local:
    [d, v1, v2, ..., vN, a1, a2, ..., a_{N-1}]

  tdp_keep_base:
    [full_smax, ..., full_smin, v1, v2, ..., vN, a1, a2, ..., a_{N-1}]

DNA tdp_local:
  [97->100, 97->98, 98->99, 99->100, a1, a2]

DNA tdp_keep_base:
  [97->100, 98->100, 99->100, 97->98, 98->99, 99->100, a1, a2]

ZJU tdp_local:
  [t-6->t, t-6->t-3, t-3->t, a1]
```

pose delta 方向必须沿用 baseline：

```python
delta_pose_mat = cur_pose_mat @ inverse(former_pose_mat)
```

xyz velocity 建议按每个 pair 的真实 dt 归一化：

```python
v_xyz = (X_to - X_from) / dt
a_xyz = v_xyz_next - v_xyz_prev
```

pose velocity / acceleration 有一个理论细节：

```text
pose delta 是 SO(3) relative rotation 的 axis-angle log vector。
直接 a_pose = v_pose_next - v_pose_prev 是 Lie algebra 上的近似二阶差分。
短间隔 DNA 上可以作为消融第一版，但论文里要说清楚是 log-rotation finite difference approximation。
```

建议第一版：

```text
use_tdp = False 默认关闭
和 MSTI / AMC / Part-MoE 默认路径隔离
先只接 DNA 脚本 smoke test

tdp_mode = local:
  motion_cond_time_step_num 自动推导为 2 * time_step_num

tdp_mode = keep_base:
  motion_cond_time_step_num 自动推导为 time_step_num + 2 * time_step_num - 1
```

建议同时加调试输出：

```text
SEQAVATAR_TDP_DEBUG=1
打印:
  channel order
  sample frame ids
  d/v/a pose mean/std/max
  d/v/a xyz mean/std/max
  degenerate pair count
```

主要风险：

```text
1. TDP-local 6 channel 会改变 baseline 信息组成，不是纯追加。
2. 若只跑 TDP-local，提升可能来自速度/加速度，也可能来自删掉 full_2/full_1 后 condition 更干净。
3. TDP-local 的 seq_len anchor 会从 baseline 的 multi-scale grid 变成 local window grid，必须说明这是新的 temporal condition 构造方式。
4. pose acceleration 是近似定义，不能过度包装成严格 SO(3) 加速度。
5. I3D 原始 time_steps 不等间隔，必须单独处理 dt。
6. channel 从 3 到 6/8，参数量增加，需要报告参数量/显存/训练时间。
7. 现有 encoder 仍是 flatten + MLP，TDP 只改变 condition，不保证网络一定利用 acceleration。
```

阶段性结论：

```text
TDP 值得作为 AMC 之后的新 motion 候选继续做。
它的创新解释比 AMC-pair 更清楚，改动复杂度低于 AMC-causal。
但最终实验至少需要:
  baseline
  tdp_local
  tdp_keep_base

这样才能区分:
  TDP 作为新 condition 是否有效；
  在保留 baseline full channel 时，速度/加速度是否仍然带来增益。
```

## 2026-07-02 TDP 推荐实验路线更新

当前更稳的路线：

```text
Step 1: DNA smoke test tdp_local
Step 2: DNA full run tdp_local
Step 3: DNA full run tdp_keep_base
Step 4: 对比 baseline / AMC-causal / TDP
Step 5: 如果 TDP 有效，再做 Part-MoE + TDP
```

公平性记录必须包含：

```text
baseline params
tdp_local params
tdp_keep_base params
显存峰值
训练时间
FPS 或 render 时间
```

如果 TDP 提升明显，后续可加：

```text
baseline_wide
```

即 baseline 保持 3 channel，但调大 encoder hidden dim 或非刚性网络宽度，使参数量接近 TDP，回答“是不是只是参数更多”的质疑。

xyz 归一化注意：

```text
不要顺手修 baseline 的 seq_xyz_conds 除 time_step bug。
TDP 分支内部可以按每个 pair 的 dt 归一化。
如果要修 baseline，应单独开:
  baseline_fixed_xyz
  tdp_on_fixed_xyz
否则 TDP 收益会混入 bug fix。
```

## 2026-07-03 TDP-keep-base 实现和 DNA 六序列结果

本次按用户指定方案实现并运行：

```text
TDP-keep-base
experiment_name = tdp
time_step_num = 3
motion_cond_time_step_num = 8
densify_until_iter = 1500
use_tdp = 1
tdp_mode = keep_base
use_msti = 0
use_amc_pair = 0
use_amc_causal = 0
use_part_moe = 0
```

核心 channel 定义固定为：

```text
[full_3, full_2, full_1, v1, v2, v3, a1, a2]
```

以 DNA `t=100` 为例：

```text
full_3 = 97 -> 100
full_2 = 98 -> 100
full_1 = 99 -> 100
v1     = 97 -> 98
v2     = 98 -> 99
v3     = 99 -> 100
a1     = v2 - v1
a2     = v3 - v2
```

实现边界：

```text
1. baseline / MSTI / AMC / Part-MoE 默认路径不变。
2. TDP 与 MSTI / AMC-pair / AMC-causal 互斥。
3. motion_cond_time_step_num 自动推导:
     tdp_local: 2 * time_step_num
     tdp_keep_base: time_step_num + 2 * time_step_num - 1
   DNA keep_base 因此为 8。
4. TDP 分支的 xyz delta 按每个 pair 的真实 dt 归一化。
5. baseline 分支未修正已有 xyz 归一化问题，避免改变旧实验。
6. renderer / loss / CameraInfo / RGB / mask / camera 均未改。
```

修改文件：

```text
arguments/__init__.py
  新增 use_tdp / tdp_mode。
  新增 TDP channel 自动推导和互斥检查。

scene/__init__.py
  将 use_tdp / tdp_mode 传入 motion_cond_options。

scene/dataset_readers.py
  新增 get_seq_pose_xyz_cond_tdp()。
  baseline / MSTI / AMC 分支保持独立。

scripts/exps_dnarendering.sh
  新增 tdp / tdp_keep_base / tdp_keepbase 模式。
  日志目录:
    /media/image/mxz/human/SeqAvatar/logs/TDP
  全局日志:
    <RUN_TIME>_DNA-Rendering_tdp.log
```

验证：

```text
py_compile:
  arguments/__init__.py
  scene/__init__.py
  scene/dataset_readers.py
  scene/gaussian_model.py
  nets/mlp_delta_non_rigid.py

bash -n scripts/exps_dnarendering.sh 通过
git diff --check 通过

DNA-like shape test:
  baseline pose/xyz = (1, 8, 3, 24, 3) / (7, 8, 3, 3)
  TDP pose/xyz      = (1, 8, 8, 24, 3) / (7, 8, 8, 3)

参数解析:
  use_tdp=True, tdp_mode=keep_base, time_step_num=3
  -> motion_cond_time_step_num=8
```

运行记录：

```text
RUN_TIME = 20260702_215942
global log = logs/TDP/20260702_215942_DNA-Rendering_tdp.log

GPU 2:
  sequences = 0044_11 0051_09 0206_04

GPU 1:
  sequences = 0813_05 0007_04 0019_10
```

启动命令等价为：

```bash
RUN_TIME=20260702_215942 SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 bash scripts/exps_dnarendering.sh tdp
RUN_TIME=20260702_215942 SEQUENCES_OVERRIDE="0813_05 0007_04 0019_10" GPU_id=1 bash scripts/exps_dnarendering.sh tdp
```

日志已确认：

```text
USE_TDP: 1
TDP_MODE: keep_base
TIME_STEP_NUM(base): 3
MOTION_COND_TIME_STEP_NUM: 8
DENSIFY_UNTIL_ITER: 1500
FINAL_EVAL_ONLY: 1
```

### TDP JSON 主结果

主结果取：

```text
output/DNA-Rendering/<sequence>/tdp/20260702_215942/metrics/results_novelview_25000.json
```

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | 32.9598 | 0.977815 | 21.5627 |
| 0051_09 | 28.5351 | 0.970581 | 31.8608 |
| 0206_04 | 31.3351 | 0.969414 | 34.3639 |
| 0813_05 | 36.0758 | 0.986793 | 18.8062 |
| 0007_04 | 29.4485 | 0.957898 | 45.9910 |
| 0019_10 | 35.3015 | 0.980876 | 21.5235 |
| Mean | 32.2760 | 0.973896 | 29.0180 |

### TDP render 复评结果

下面采用 `render.py` 最终 novelview 评价行：

```text
[ITER 25000] Evaluating novelview #120: PSNR ... SSIM ... LPIPS ...
```

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | 32.9598 | 0.977815 | 21.5626 |
| 0051_09 | 28.6275 | 0.971297 | 31.3700 |
| 0206_04 | 31.3385 | 0.969456 | 34.3120 |
| 0813_05 | 36.0801 | 0.986805 | 18.7885 |
| 0007_04 | 29.4487 | 0.957920 | 45.9556 |
| 0019_10 | 35.3025 | 0.980895 | 21.4998 |
| Mean | 32.2929 | 0.974031 | 28.9147 |

### TDP vs baseline, densify 1500

公平对比使用同 `densify_until_iter=1500` 的 baseline render 复评结果。

TDP - baseline：

| Sequence | dPSNR | dSSIM | dLPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | -0.0143 | -0.000100 | +0.1656 |
| 0051_09 | -0.0439 | -0.000151 | +0.2779 |
| 0206_04 | -0.0414 | -0.000342 | +0.2885 |
| 0813_05 | -0.0066 | -0.000088 | +0.3248 |
| 0007_04 | -0.0849 | -0.000415 | +0.5662 |
| 0019_10 | +0.0816 | +0.000199 | +0.2349 |
| Mean | -0.0182 | -0.000149 | +0.3096 |

阶段性判断：

```text
TDP-keep-base 当前六序列平均没有优于 baseline。
PSNR / SSIM 平均略低，LPIPS*1000 平均更高。

这说明“保留 baseline full motion 后追加 v/a channel”的第一版没有带来稳定收益。
目前不建议把 TDP-keep-base 直接作为第二创新点主线。
```

可能原因：

```text
1. DNA 的 [3,2,1] 已经很密，baseline full_3/full_2/full_1 已包含足够近邻运动信息。
2. 额外 velocity / acceleration channel 与 full channel 冗余，flatten MLP 可能不能有效区分阶数语义。
3. pose acceleration 只是 log-rotation 空间的近似二阶差分，可能带来噪声。
4. xyz acceleration 对非刚性区域可能有用，但全局平均指标被静态/低运动区域稀释。
```

后续若继续探索 TDP：

```text
1. 先用 evaluate_amc_decision.py 类似逻辑做 high-motion subset / boundary subset 评价。
2. 如果 high-motion subset 也没有提升，不建议继续主推 TDP。
3. 若要继续改模型，优先做分支式 encoder:
     full motion encoder
     velocity encoder
     acceleration encoder
     late fusion / gate
   不要继续简单把所有 channel flatten 混在一起。
4. 可补 tdp_local，但要明确它不是纯追加 baseline，而是重新定义 motion condition。
```

## 2026-07-03 TDP 效果下降原因和无效判断标准

当前不能说 TDP 理论方向被彻底否定，更准确的结论是：

```text
当前 TDP-keep-base 实现，在 DNA / 当前 flatten+MLP encoder / densify=1500 设置下，
没有带来可复现的全局指标收益，因此不适合作为第二创新点主线。
```

### 为什么可能下降

1. DNA baseline 的 `[3,2,1]` 已经很密

baseline 已有：

```text
full_3 = t-3 -> t
full_2 = t-2 -> t
full_1 = t-1 -> t
```

对 xyz 这类向量运动，如果 full delta 都按 dt 归一化，则：

```text
full_1 = v3
full_2 = (v2 + v3) / 2
full_3 = (v1 + v2 + v3) / 3
```

因此速度可以由 full motion 近似线性组合得到：

```text
v3 = full_1
v2 = 2 * full_2 - full_1
v1 = 3 * full_3 - 2 * full_2
```

这说明 TDP 追加的 `v1/v2/v3/a1/a2` 在 DNA 短时间跨度上信息增量可能很小，更多是冗余通道。

2. 当前 encoder 不知道这些 channel 的阶数语义

当前 `SeqPoseEncoder / SeqXYZEncoder` 仍然是 flatten + MLP。

它看到的是 8 个平铺通道：

```text
[full_3, full_2, full_1, v1, v2, v3, a1, a2]
```

但没有显式结构告诉网络：

```text
前三个是 displacement
中间三个是 velocity
最后两个是 acceleration / trend
```

所以“加入速度/加速度”不等于模型真的按时间差分金字塔使用它们。额外通道也可能只是增加输入噪声和优化难度。

3. pose acceleration 是近似二阶差分，可能带噪声

当前：

```text
a_pose = v_pose_next - v_pose_prev
```

其中 `v_pose` 是 relative rotation 的 axis-angle/log vector。

这只能称为：

```text
log-rotation space approximate second-order difference
```

不是严格物理角加速度。对旋转较复杂的关节，它可能引入不稳定的高频噪声。

4. TDP 的 seq_len 组织方式改变了 baseline 的时间网格

baseline 对每个 time_step 各自推进：

```text
step=3: t-3 -> t, t-6 -> t-3, ...
step=2: t-2 -> t, t-4 -> t-2, ...
step=1: t-1 -> t, t-2 -> t-1, ...
```

TDP 使用局部 anchor：

```text
anchor_i = t - i * 3
i=0: [t-3,t-2,t-1,t]
i=1: [t-6,t-5,t-4,t-3]
```

因此 TDP-keep-base 只在 `i=0` 的当前窗口完整保留 baseline 的 `[full_3, full_2, full_1]`。
在 `i>0` 时，它不是原 baseline 的多尺度时间网格，而是局部 causal window 网格。

这会改变 encoder 过去已经适配的 `seq_len/channel` 语义，可能抵消 TDP channel 的潜在收益。

5. TDP 分支的 xyz 归一化和 baseline 不完全一致

为了不影响旧实验，baseline 分支未修已有 xyz dt 归一化问题。
TDP 分支内部则按每个 pair 的真实 dt 归一化。

因此当前对比不是单纯：

```text
baseline + v/a channel
```

还包含：

```text
TDP 分支 xyz 数值尺度变化
seq_len 时间网格变化
channel 数增加
```

这意味着不能把下降只归因于 acceleration 无效，但可以判断当前整体实现没有收益。

### 怎么判断当前方法没有效果

当前判断依据不是看某一个序列，而是看公平设置下的整体趋势：

```text
same dataset: DNA six sequences
same split: novelview
same iteration: 25000
same densify_until_iter: 1500
same renderer / loss / CameraInfo
```

TDP render 复评相对 baseline：

```text
Mean dPSNR        = -0.0182
Mean dSSIM        = -0.000149
Mean dLPIPS*1000  = +0.3096
```

逐序列趋势：

```text
PSNR: 5/6 序列下降，只有 0019_10 上升。
SSIM: 5/6 序列下降，只有 0019_10 上升。
LPIPS: 6/6 序列变差。
```

因此当前不是“有轻微不稳定收益”，而是：

```text
全局平均三项都不优；
感知指标 LPIPS 在所有序列上都更差；
唯一 PSNR/SSIM 上升的 0019_10，LPIPS 仍更差。
```

判定标准：

```text
1. 如果全局平均三项至少两项没有提升，不能作为主方法。
2. 如果 LPIPS 在所有序列变差，不能声称视觉质量改善。
3. 如果收益只出现在单个序列，且其它序列多数下降，只能认为是偶然或序列特异。
4. 如果方法主张改善运动区域，还必须看 high-motion subset / boundary subset；
   若这些子集也无提升，则应停止作为 motion 主线。
```

当前结论：

```text
TDP-keep-base 当前结果足以判定:
  不适合作为第二创新点主线。

但还不能严格判定:
  所有 TDP 形式都无效。

如果继续，只应作为支线做:
  high-motion subset / boundary subset 评价；
  或分支式 full/velocity/acceleration encoder + gate。
```

## 2026-07-03 TDP 消融实验代码改动说明

TDP 消融实验的目标是：

```text
在不改 renderer / loss / CameraInfo / RGB / mask / Part-MoE / MSTI / AMC 默认路径的前提下，
只改变输入 non-rigid motion encoder 的 motion condition channel。
```

本次 TDP-keep-base 的 condition 定义：

```text
baseline:
  [full_3, full_2, full_1]

TDP-keep-base:
  [full_3, full_2, full_1, v1, v2, v3, a1, a2]
```

### 1. `arguments/__init__.py`

新增参数：

```text
use_tdp = False
tdp_mode = keep_base
```

目的：

```text
1. 默认关闭 TDP，保证 baseline / MSTI / AMC / Part-MoE 旧实验不受影响。
2. 通过命令行 `--use_tdp --tdp_mode keep_base` 显式进入 TDP 分支。
3. 统一在 resolve_motion_condition_args() 中推导 encoder 输入 channel 数，避免脚本手填错误。
```

新增互斥规则：

```text
use_msti / use_amc_pair / use_amc_causal / use_tdp 最多只能开一个。
```

目的：

```text
避免多个 motion 消融分支同时改变 condition 语义，导致 train/render 维度和解释混乱。
```

新增 channel 推导：

```text
tdp_mode = local:
  motion_cond_time_step_num = 2 * time_step_num

tdp_mode = keep_base:
  motion_cond_time_step_num = time_step_num + 2 * time_step_num - 1
```

DNA 中：

```text
time_step_num = 3
TDP-keep-base channel = 3 + 2 * 3 - 1 = 8
```

目的：

```text
保留 time_step_num 的原始含义:
  仍然用于生成原始尺度 [3,2,1]

单独用 motion_cond_time_step_num 表示 encoder 真正接收的 condition channel 数:
  baseline = 3
  TDP-keep-base = 8
```

### 2. `scene/__init__.py`

在 `motion_cond_options` 中新增：

```text
use_tdp
tdp_mode
motion_cond_time_step_num
```

目的：

```text
Scene 构建 dataset / cond_dict 时，把 train/render 的 TDP 配置传给 dataset reader。
这样训练和渲染阶段会用同一套 condition 构造逻辑。
```

### 3. `scene/dataset_readers.py`

在入口 `get_seq_pose_xyz_cond()` 中新增 TDP 分支：

```text
if use_tdp:
  return get_seq_pose_xyz_cond_tdp(...)
```

目的：

```text
把 TDP 和 baseline / MSTI / AMC-pair 分开。
不开 use_tdp 时，原始 baseline condition 构造逻辑完全不变。
```

新增函数：

```text
get_seq_pose_xyz_cond_tdp()
```

核心逻辑：

```text
1. 按原始 time_steps 从大到小排序:
     DNA: [3,2,1]

2. 每个 seq slot 使用局部 anchor:
     anchor_i = pose_index - i * max_step

3. 对 DNA 构造局部时间点:
     [anchor-3, anchor-2, anchor-1, anchor]

4. keep_base 模式先构造 full channels:
     anchor-3 -> anchor
     anchor-2 -> anchor
     anchor-1 -> anchor

5. 再构造 adjacent velocity:
     anchor-3 -> anchor-2
     anchor-2 -> anchor-1
     anchor-1 -> anchor

6. 最后构造 acceleration / trend:
     a1 = v2 - v1
     a2 = v3 - v2
```

pose delta 方向保持和 baseline 一致：

```text
delta_pose = cur_pose @ inv(former_pose)
```

xyz delta 在 TDP 分支中按每个 pair 自己的 dt 归一化：

```text
xyz_delta = (cur_xyz - former_xyz) / dt
```

cache key：

```text
tdp_<tdp_mode>_dt_v1:<cur_id>-<former_id>
```

目的：

```text
1. 避免 TDP 的 dt-normalized delta 与 baseline 旧 cache key 混用。
2. 避免 keep_base / local 不同模式之间复用错误缓存。
```

shape assert：

```text
seq_pose_conds.shape[2] == expected_channels
seq_xyz_conds.shape[2] == expected_channels
```

目的：

```text
快速暴露 train/render 配置不一致或 channel 构造错误。
```

### 4. `scripts/exps_dnarendering.sh`

新增运行模式：

```bash
bash scripts/exps_dnarendering.sh tdp
```

等价别名：

```text
tdp_keep_base
tdp_keepbase
```

TDP 模式配置：

```text
experiment_name = tdp
use_tdp = 1
tdp_mode = keep_base
part_moe_enabled = 0
final_eval_only = 1
time_step_num = 3
motion_cond_time_step_num = 8
densify_until_iter = 1500
```

日志目录：

```text
/media/image/mxz/human/SeqAvatar/logs/TDP
```

日志命名：

```text
<RUN_TIME>_DNA-Rendering_tdp.log
```

目的：

```text
1. TDP 输出和 baseline / MSTI / AMC / Part-MoE 日志隔离。
2. train.py 和 render.py 都带上相同的:
     --use_tdp
     --tdp_mode keep_base
     --motion_cond_time_step_num 8
   确保 checkpoint 维度和 render 构造的 condition 维度一致。
3. TDP 模式不生成普通 output/.../logs/train/render 分日志，只保留全局 TDP 日志，和 MSTI/AMC 规则一致。
```

### 5. 没有改的部分

TDP 没有改：

```text
renderer
loss
CameraInfo
RGB image / mask / camera
Gaussian densify 逻辑
Part-MoE 专家 / 路由 / label
MSTI 分支
AMC-pair / AMC-causal 分支
SeqPoseEncoder / SeqXYZEncoder 网络结构
NonrigidDeformer 主体结构
```

`gaussian_model.py` 和 `nets/mlp_delta_non_rigid.py` 已经支持通过 `motion_cond_time_step_num` 改 encoder 输入维度，所以 TDP 只复用了这套机制：

```text
baseline checkpoint:
  encoder input channels = 3

TDP checkpoint:
  encoder input channels = 8
```

因此 baseline checkpoint 和 TDP checkpoint 不能直接混用。

### 6. 代码改动的实验目的总结

本次代码改动服务于一个单一消融问题：

```text
在完整保留 SeqAvatar baseline 多尺度 full motion 的情况下，
额外加入 adjacent velocity 和 approximate acceleration / trend，
能不能改善 non-rigid deformation 的 motion condition？
```

代码层面的隔离设计保证：

```text
如果 TDP 有收益，收益主要来自 condition channel 内容变化；
不是来自 renderer/loss/监督图像/相机/Part-MoE/MSTI/AMC 的变化。
```

## 2026-07-03 TDP semantic branch encoder 方案评审

用户提出下一版 TDP 不再把：

```text
[full_3, full_2, full_1, v1, v2, v3, a1, a2]
```

直接 flatten 到同一个 MLP，而是按语义拆成：

```text
full: [full_3, full_2, full_1]
vel:  [v1, v2, v3]
acc:  [a1, a2]
```

分别编码后用残差门控融合：

```text
f_motion = f_full + g_v * f_vel + g_a * f_acc
```

结论：

```text
方案可行，而且比上一版 TDP-keep-base 更合理。
它直接针对上一版失败原因:
  full / velocity / acceleration 语义混在 flatten MLP 中，网络没有结构性归纳偏置。
```

但这已经不是单纯 condition-channel 消融，而是：

```text
TDP condition + semantic motion encoder
```

因此实验名建议不要继续叫普通 `tdp`，而应单独命名，例如：

```text
tdp_semantic
tdp_branch_gate
tdp_semantic_gate
```

### 对当前代码的适配判断

当前 `SeqPoseEncoder` 输入实际是：

```text
x: [B, seq_len, motion_channels, J, 3]
```

DNA 使用 SMPL-X 时，`J` 不是 24，而是：

```text
N_JOINT[smplx] + 1 = 55
```

所以 DNA TDP pose condition shape 是：

```text
[1, 8, 8, 55, 3]
```

而不是：

```text
[B, L, 8, 24, 3]
```

因此实现时不能硬编码 `24`，必须沿用当前代码：

```text
self.input_dim = 3 * (N_JOINT[smpl_type] + 1)
```

Pose 分支维度应是：

```text
full input = self.input_dim * 3
vel input  = self.input_dim * 3
acc input  = self.input_dim * 2
```

当前 `SeqXYZEncoder` 输入实际是：

```text
x: [B, N, seq_len, KNN, motion_channels, 3]
```

因此 xyz 分支要沿 channel 维 `x.shape[4]` 拆：

```text
full = x[:, :, :, :, 0:3, :]
vel  = x[:, :, :, :, 3:6, :]
acc  = x[:, :, :, :, 6:8, :]
```

xyz 分支输入维度应是：

```text
full input = 3 * seq_xyz_knn * 3
vel input  = 3 * seq_xyz_knn * 3
acc input  = 2 * seq_xyz_knn * 3
```

### 推荐实现方式

不要替换原 `SeqPoseEncoder / SeqXYZEncoder` 默认逻辑。

建议新增开关：

```text
use_tdp_semantic_encoder = False
tdp_semantic_mode = gated_residual
tdp_gate_init_bias = -4.0
```

默认关闭，只有同时满足：

```text
use_tdp = True
tdp_mode = keep_base
use_tdp_semantic_encoder = True
motion_cond_time_step_num = 8
```

才使用 semantic branch encoder。

这样隔离关系是：

```text
baseline:
  原 encoder, 3 channel

tdp:
  原 encoder, 8 channel flatten

tdp_semantic:
  semantic branch encoder, 8 channel
```

这能清楚回答：

```text
1. 只加 TDP channel 有没有用？
2. 给 TDP 加语义分支归纳偏置有没有用？
```

### Pose encoder 推荐结构

对 pose：

```text
full -> full_encoder -> f_full
vel  -> vel_encoder  -> f_vel
acc  -> acc_encoder  -> f_acc

gate_v = sigmoid(gate_v(f_full))
gate_a = sigmoid(gate_a(f_full))

f_step = f_full + gate_v * f_vel + gate_a * f_acc
f_pose = mlp2(flatten(seq_len, f_step))
```

其中 gate 推荐输出：

```text
[B, seq_len, 1]
```

初始 bias：

```text
gate_v final Linear bias = -4.0
gate_a final Linear bias = -4.0
```

这样：

```text
sigmoid(-4) ~= 0.018
```

模型初始接近：

```text
f_step ~= f_full
```

注意：

```text
f_full 并不完全等于旧 baseline encoder。
它是只看 TDP local anchor 下 full_3/full_2/full_1 的分支。
因此它是“接近 full-only 分支”，不是严格复现 baseline。
```

如果希望更接近 baseline，还要把 TDP 的 seq_len 时间网格问题一并处理。

### XYZ encoder 推荐结构

对 xyz 不建议只照搬 pose 伪代码，因为当前 `SeqXYZEncoder` 还融合了 positional embedding：

```text
pos_feat = pos_emb_proj(x_emb)
```

推荐：

```text
full -> full_vel_encoder -> f_full
vel  -> vel_vel_encoder  -> f_vel
acc  -> acc_vel_encoder  -> f_acc

gate_v = sigmoid(gate_v(concat(f_full, pos_feat)))
gate_a = sigmoid(gate_a(concat(f_full, pos_feat)))

f_motion = f_full + gate_v * f_vel + gate_a * f_acc
h = concat(f_motion, pos_feat)
mlp1 / mlp2 保持原结构
```

gate 形状建议：

```text
[B, N, seq_len, 1]
```

这样每个 Gaussian / 每个时间 slot 都能决定是否使用 velocity / acceleration。

### 需要修改的文件

如果实现该方案，预计修改：

```text
arguments/__init__.py
  新增 use_tdp_semantic_encoder / tdp_semantic_mode / tdp_gate_init_bias
  限制只能和 use_tdp + keep_base 同时使用

scene/gaussian_model.py
  将 use_tdp_semantic_encoder / tdp_gate_init_bias 传入 NonrigidDeformer

nets/mlp_delta_non_rigid.py
  SeqPoseEncoder 增加 TDP semantic 分支逻辑
  SeqXYZEncoder 增加 TDP semantic 分支逻辑

scripts/exps_dnarendering.sh
  新增 tdp_semantic 或 tdp_branch_gate 模式
  日志目录仍可用 logs/TDP
  experiment_name 建议用 tdp_semantic
```

`scene/dataset_readers.py` 原则上不需要改：

```text
沿用当前 TDP-keep-base 已生成的 8 channel condition。
```

### 风险点

1. 参数量增加

相比当前 flatten TDP，三分支 encoder 会增加参数。
如果有效，仍需记录：

```text
参数量
显存
训练时间
```

必要时做：

```text
baseline_wide
```

2. `f_full` 不严格等价 baseline

因为 TDP condition 的 seq_len local anchor 与 baseline 时间网格不同。
即使 gate 初始化接近 0，模型也不是完全等于 baseline。

3. acceleration 可能仍然是噪声

gate 能降低伤害，但如果 acc 分支一直接近 0，说明 acceleration 对当前设置贡献不大。

建议日志诊断：

```text
gate_v mean/std/max
gate_a mean/std/max
gate_v active ratio
gate_a active ratio
```

### 实验优先级

该方案比继续跑普通 `tdp_local` 更值得尝试。

推荐路线：

```text
1. 先实现 tdp_semantic，默认关闭。
2. 单序列 smoke test，确认:
     pose / xyz shape 正确
     gate 初值约 0.018
     loss 不 NaN
3. 跑 DNA 六序列 tdp_semantic。
4. 对比:
     baseline densify=1500
     tdp_keep_base flatten
     tdp_semantic
5. 如果 tdp_semantic 仍不提升，再停止 TDP 主线。
```

阶段性判断：

```text
可行，且是 TDP 当前最合理的下一步。
但它是 encoder 结构消融，不再只是 condition channel 消融。
必须独立命名、独立日志、默认关闭，并和现有 tdp flatten 结果分开解释。
```

## 2026-07-03 TDP-semantic 实现与 DNA 六序列结果

用户要求在不影响其它实验路径的条件下实现 `tdp_semantic` 消融：

```text
TDP keep-base condition:
  [full_3, full_2, full_1, v1, v2, v3, a1, a2]

tdp_semantic encoder:
  full -> FullEncoder
  vel  -> VelEncoder
  acc  -> AccEncoder
  fused = f_full + gate_v * f_vel + gate_a * f_acc
```

实现隔离：

```text
baseline / Part-MoE / MSTI / AMC / 普通 TDP flatten 默认行为不变。
tdp_semantic 只有同时开启:
  use_tdp = 1
  tdp_mode = keep_base
  use_tdp_semantic_encoder = 1
才进入新 encoder 分支。
```

主要代码改动：

```text
arguments/__init__.py
  新增 use_tdp_semantic_encoder / tdp_semantic_mode / tdp_gate_init_bias。
  限制 semantic encoder 只能与 use_tdp + keep_base 组合。

scene/gaussian_model.py
  将 TDP semantic 参数传入 NonrigidDeformer。

nets/mlp_delta_non_rigid.py
  SeqPoseEncoder 新增 full/vel/acc 三分支 gated residual 编码。
  SeqXYZEncoder 新增 full/vel/acc 三分支 gated residual 编码，并保留 pos_feat 融合。
  gate 最后一层 bias 初始化为 -4.0，初始 sigmoid 约 0.018。
  默认未开启 use_tdp_semantic_encoder 时仍走原 flatten encoder。

scripts/exps_dnarendering.sh
  新增 tdp_semantic 模式。
  默认:
    experiment_name = tdp_semantic
    use_tdp = 1
    tdp_mode = keep_base
    motion_cond_time_step_num = 8
    densify_until_iter = 1500
    final_eval_only = 1
  新增可选 GLOBAL_LOG_SUFFIX / LOG_SUFFIX，用于双卡并行时拆分主日志。
  默认不设置后缀时，旧日志命名保持不变。
```

验证：

```text
bash -n scripts/exps_dnarendering.sh 通过。
git diff --check 通过。
前置 shape test 已确认:
  SeqPoseEncoder semantic input [B,8,8,55,3] -> [B,32]
  SeqXYZEncoder semantic input [B,N,8,8,8,3] -> [B,N,128]
  gate 初值约 0.016-0.018
```

运行记录：

```text
第一次尝试:
  RUN_TIME = 20260703_002549
  两卡共用一个主日志，并且 GPU2 在加载阶段 OOM。
  已终止，不作为结果。

第二次尝试:
  RUN_TIME = 20260703_003245
  使用 IMAGE_DATA_DEVICE=cpu 后可进入训练。
  但用户要求双卡主日志拆分，因此早期终止，不作为结果。

正式运行:
  RUN_TIME = 20260703_003540
  IMAGE_DATA_DEVICE = cpu

GPU2:
  sequences = 0044_11 0051_09 0206_04
  log = logs/TDP/20260703_003540_DNA-Rendering_tdp_semantic_gpu2.log

GPU1:
  sequences = 0813_05 0007_04 0019_10
  log = logs/TDP/20260703_003540_DNA-Rendering_tdp_semantic_gpu1.log
```

启动命令：

```bash
RUN_TIME=20260703_003540 GLOBAL_LOG_SUFFIX=gpu2 SEQUENCES_OVERRIDE="0044_11 0051_09 0206_04" GPU_id=2 IMAGE_DATA_DEVICE=cpu bash scripts/exps_dnarendering.sh tdp_semantic
RUN_TIME=20260703_003540 GLOBAL_LOG_SUFFIX=gpu1 SEQUENCES_OVERRIDE="0813_05 0007_04 0019_10" GPU_id=1 IMAGE_DATA_DEVICE=cpu bash scripts/exps_dnarendering.sh tdp_semantic
```

六序列均完成，无 `Traceback / RuntimeError / CUDA out of memory / NaN`。

主结果采用 render.py 最终 novelview 行：

```text
[ITER 25000] Evaluating novelview #120: PSNR ... SSIM ... LPIPS ...
```

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | 32.9554 | 0.977727 | 21.6095 |
| 0051_09 | 28.6507 | 0.971516 | 30.8890 |
| 0206_04 | 31.3627 | 0.970109 | 33.5955 |
| 0813_05 | 36.0005 | 0.986649 | 18.8276 |
| 0007_04 | 29.5074 | 0.958098 | 45.5770 |
| 0019_10 | 35.2780 | 0.980853 | 21.1745 |
| Mean | 32.2925 | 0.974159 | 28.6122 |

与 DNA baseline densify_until_iter=1500 的差值，TDP-semantic - baseline：

| Sequence | dPSNR | dSSIM | dLPIPS*1000 |
|---|---:|---:|---:|
| 0044_11 | -0.0187 | -0.000188 | +0.2125 |
| 0051_09 | -0.0207 | +0.000068 | -0.2031 |
| 0206_04 | -0.0172 | +0.000311 | -0.4280 |
| 0813_05 | -0.0862 | -0.000244 | +0.3639 |
| 0007_04 | -0.0262 | -0.000237 | +0.1876 |
| 0019_10 | +0.0571 | +0.000157 | -0.0904 |
| Mean | -0.0186 | -0.000022 | +0.0071 |

与普通 TDP flatten 均值对比：

```text
TDP flatten mean:
  PSNR 32.2929, SSIM 0.974031, LPIPS*1000 28.9147

TDP-semantic - TDP flatten:
  dPSNR        -0.0004
  dSSIM        +0.000128
  dLPIPS*1000  -0.3025
```

阶段性判断：

```text
tdp_semantic 相比 baseline:
  PSNR 略低，SSIM 基本持平，LPIPS*1000 基本持平。

tdp_semantic 相比普通 TDP flatten:
  PSNR 几乎相同，SSIM 略高，LPIPS*1000 明显更好。

结论:
  semantic branch/gate 能缓解普通 TDP flatten 的感知指标下降，
  但相对 baseline 仍没有形成稳定、明确的整体提升。
  TDP 主线还不能直接作为强第二创新点，需要进一步看 high-motion / boundary subset 或 gate 统计。
```

### TDP-semantic OOM 处理说明

本次为了避免 `tdp_semantic` 加载阶段 OOM，实际没有降低模型/训练核心参数：

```text
没有降低:
  iterations = 25000
  seq_len = 8
  seq_xyz_knn = 8
  time_step_num = 3
  motion_cond_time_step_num = 8
  non_rigid_mlp_depth = 3
  non_rigid_mlp_width = 512
  densify_until_iter = 1500
  图像分辨率 / train-test split / renderer / loss

没有开启:
  SKIP_LOAD_TEST_CAMERAS = 1
```

真正用于降显存的是运行时环境变量：

```text
IMAGE_DATA_DEVICE=cpu
```

脚本会把它传成：

```text
SEQAVATAR_IMAGE_DATA_DEVICE=cpu
```

代码效果：

```text
scene/cameras.py:
  original_image / mask 常驻在 CPU，而不是构造 Camera 时直接常驻 GPU。

train.py:
  每次训练只把当前 viewpoint 的 original_image / mask 搬到 GPU。

render.py:
  评价时同样在使用时再搬到 GPU。
```

这会降低显存峰值，代价是 CPU->GPU 拷贝导致训练/评价略慢。它不改变输入图像数值、模型结构、loss、采样帧、Gaussian densify 策略或最终 render 逻辑，因此理论上不应系统性改变指标。

`final_eval_only=1` 也不是降低模型参数。它只把中间 `3000 / part_moe_start_iter` 的 eval/save 去掉，保留最终 `25000` eval/save：

```text
test_iterations = [25000]
save_iterations = [25000]
```

它不参与 optimizer update，不改变训练 loss。主要作用是减少中间全测试集评估带来的显存和时间压力。

因此本次指标下降不应归因于 `IMAGE_DATA_DEVICE=cpu` 或日志拆分，而应归因于 `tdp_semantic` 本身的运动编码/encoder 改动和训练随机波动。

### TDP-semantic 指标下降原因判断

相对 baseline 的均值差很小：

```text
dPSNR        -0.0186
dSSIM        -0.000022
dLPIPS*1000  +0.0071
```

这更像“没有明确收益”，不是大幅退化。

可能原因：

```text
1. tdp_semantic 不是严格 baseline 等价初始化。
   gate 初始接近 0 只抑制 vel/acc residual，
   但 f_full 是新的 full 分支 encoder，不是原 baseline flatten encoder。

2. TDP 的 seq_len local anchor 与 baseline 原时间网格不同。
   即使保留 full_3/full_2/full_1，每个 seq slot 的组织方式也不是完全相同。

3. velocity / acceleration 对 DNA 可能冗余。
   DNA 的 [3,2,1] 已经很密，full_1/full_2/full_3 已经包含大部分短期运动信息。

4. acceleration 是二阶差分，可能放大 SMPL pose / vertex motion 的噪声。

5. 三分支 encoder 增加参数和优化难度。
   gate 能减少伤害，但不能保证一定优于 baseline。

6. 六序列逐项并非全部下降。
   0051_09 / 0206_04 / 0019_10 的 LPIPS 更好，
   但 0044_11 / 0813_05 / 0007_04 变差，平均后没有稳定收益。
```

相比普通 TDP flatten：

```text
TDP-semantic - TDP flatten:
  dPSNR        -0.0004
  dSSIM        +0.000128
  dLPIPS*1000  -0.3025
```

说明 semantic branch/gate 对普通 TDP flatten 的 LPIPS 下降有缓解作用，但仍不足以超过 baseline。

## 2026-07-03 I3D / ZJU TDP-semantic 脚本与运行记录

目标：

```text
在 I3D-Human 和 ZJU-MoCap 上复用 DNA 的 TDP-semantic 消融；
不改变 baseline / Part-MoE / MSTI / AMC / 普通 TDP flatten 默认路径；
日志统一写到 logs/TDP；
GPU1 / GPU2 并行时使用分主日志，避免两张卡日志混在一起。
```

脚本改动：

```text
scripts/exps_i3dhuman.sh
scripts/exps_zjumocap.sh
```

新增模式：

```text
tdp_semantic
tdp_branch_gate
tdp_semantic_gate
```

tdp_semantic 模式固定：

```text
use_tdp = 1
tdp_mode = keep_base
use_tdp_semantic_encoder = 1
tdp_semantic_mode = gated_residual
tdp_gate_init_bias = -4.0
final_eval_only = 1
use_part_moe = 0
```

channel 自动推导：

```text
motion_cond_time_step_num = time_step_num + 2 * time_step_num - 1

I3D:
  time_step_num = 3
  time_steps = [42,33,24]
  motion_cond_time_step_num = 8

ZJU:
  time_step_num = 2
  time_steps = [6,3]
  motion_cond_time_step_num = 5
```

日志规则：

```text
TDP_LOG_DIR = /media/image/mxz/human/SeqAvatar/logs/TDP
GLOBAL_LOG_SUFFIX = gpu1 / gpu2

I3D:
  logs/TDP/<RUN_TIME>_I3D-Human_tdp_semantic_gpu1.log
  logs/TDP/<RUN_TIME>_I3D-Human_tdp_semantic_gpu2.log

ZJU:
  logs/TDP/<RUN_TIME>_ZJU-MoCap_tdp_semantic_gpu1.log
  logs/TDP/<RUN_TIME>_ZJU-MoCap_tdp_semantic_gpu2.log
```

显存控制：

```text
IMAGE_DATA_DEVICE=cpu
```

脚本会转成：

```text
SEQAVATAR_IMAGE_DATA_DEVICE=cpu
```

该设置只影响 image/mask 常驻设备，降低显存峰值；不改变模型结构、loss、renderer、训练帧、测试帧或最终指标计算语义。

已验证：

```text
bash -n scripts/exps_i3dhuman.sh 通过
bash -n scripts/exps_zjumocap.sh 通过
bash -n scripts/exps_dnarendering.sh 通过
git diff --check 通过

shape smoke test 通过:
  I3D base=3, cond=8
  ZJU base=2, cond=5
```

### I3D-Human TDP-semantic 启动记录

启动时间：

```text
RUN_TIME = 20260703_143023
```

GPU1：

```bash
RUN_TIME=20260703_143023 \
GLOBAL_LOG_SUFFIX=gpu1 \
SEQUENCES_OVERRIDE="ID1_1 ID1_2" \
GPU_id=1 \
IMAGE_DATA_DEVICE=cpu \
bash scripts/exps_i3dhuman.sh tdp_semantic
```

日志：

```text
logs/TDP/20260703_143023_I3D-Human_tdp_semantic_gpu1.log
```

GPU2：

```bash
RUN_TIME=20260703_143023 \
GLOBAL_LOG_SUFFIX=gpu2 \
SEQUENCES_OVERRIDE="ID2_1 ID3_1" \
GPU_id=2 \
IMAGE_DATA_DEVICE=cpu \
bash scripts/exps_i3dhuman.sh tdp_semantic
```

日志：

```text
logs/TDP/20260703_143023_I3D-Human_tdp_semantic_gpu2.log
```

启动后确认：

```text
USE_TDP = 1
TDP_MODE = keep_base
USE_TDP_SEMANTIC_ENCODER = 1
TIME_STEP_NUM(base) = 3
MOTION_COND_TIME_STEP_NUM = 8
FINAL_EVAL_ONLY = 1
IMAGE_DATA_DEVICE = cpu
```

当前状态：

```text
两个首序列 ID1_1 / ID2_1 已进入训练迭代；
GPU1 / GPU2 显存约 22GB；
未出现 OOM、shape mismatch 或 TDP 配置报错。
```

### I3D-Human TDP-semantic 指标, 20260703_143023

结果来源：

```text
output/I3D-Human/<sequence>/tdp_semantic/20260703_143023/metrics/results_novelview_15000.json
output/I3D-Human/<sequence>/tdp_semantic/20260703_143023/metrics/results_novelpose_15000.json
```

说明：

```text
四个 I3D-Human 序列均已完成。
下面采用 metrics/results JSON 主结果。
平均值为序列级简单算术平均，不按测试图片数量加权。
```

Novelview:

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| ID1_1 | 31.9781 | 0.966546 | 25.8807 |
| ID1_2 | 32.0687 | 0.966366 | 27.3639 |
| ID2_1 | 31.5075 | 0.969134 | 29.4292 |
| ID3_1 | 33.6861 | 0.965457 | 33.6492 |
| Mean | 32.3101 | 0.966876 | 29.0808 |

Novelpose:

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| ID1_1 | 29.9252 | 0.959751 | 31.8501 |
| ID1_2 | 30.5655 | 0.959789 | 31.2806 |
| ID2_1 | 27.8554 | 0.953647 | 41.9104 |
| ID3_1 | 32.6332 | 0.959644 | 37.6441 |
| Mean | 30.2448 | 0.958208 | 35.6713 |

### ZJU-MoCap TDP-semantic 队列记录

由于 I3D 已占用 GPU1/GPU2，ZJU 采用同卡排队启动，等待对应 I3D 序列组结束后自动运行。

队列等待方式：

```text
等待对应 I3D tmux session 退出后再启动 ZJU，
避免在 I3D 同卡两个序列切换的短间隔误判 GPU 空闲。
```

队列时间：

```text
RUN_TIME = 20260703_143851
```

GPU1 队列：

```bash
RUN_TIME=20260703_143851 \
GLOBAL_LOG_SUFFIX=gpu1 \
SEQUENCES_OVERRIDE="CoreView_377 CoreView_386 CoreView_387" \
GPU_id=1 \
IMAGE_DATA_DEVICE=cpu \
bash scripts/exps_zjumocap.sh tdp_semantic
```

预期日志：

```text
logs/TDP/20260703_143851_ZJU-MoCap_tdp_semantic_gpu1.log
```

GPU2 队列：

```bash
RUN_TIME=20260703_143851 \
GLOBAL_LOG_SUFFIX=gpu2 \
SEQUENCES_OVERRIDE="CoreView_392 CoreView_393 CoreView_394" \
GPU_id=2 \
IMAGE_DATA_DEVICE=cpu \
bash scripts/exps_zjumocap.sh tdp_semantic
```

预期日志：

```text
logs/TDP/20260703_143851_ZJU-MoCap_tdp_semantic_gpu2.log
```

tmux 队列：

```text
seqavatar_zju_tdp_semantic_gpu1_20260703_143851
seqavatar_zju_tdp_semantic_gpu2_20260703_143851
```

队列 launch 日志：

```text
/tmp/seqavatar_20260703_143851_zju_tdp_semantic_gpu1.launch.log
/tmp/seqavatar_20260703_143851_zju_tdp_semantic_gpu2.launch.log
```

### ZJU-MoCap TDP-semantic 指标, 20260703_143851

结果来源：

```text
output/ZJU-MoCap/<sequence>/tdp_semantic/20260703_143851/metrics/results_test_3000.json
```

说明：

```text
六个 ZJU-MoCap 序列均已完成。
下面采用 metrics/results JSON 主结果。
平均值为序列级简单算术平均，不按测试图片数量加权。
render.py 最终日志行与 JSON 只有极小差异；论文表格建议固定使用 JSON 主结果。
```

Test:

| Sequence | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| CoreView_377 | 31.5128 | 0.973515 | 18.0263 |
| CoreView_386 | 33.9771 | 0.969577 | 24.7938 |
| CoreView_387 | 28.8232 | 0.955757 | 32.2784 |
| CoreView_392 | 31.9010 | 0.964121 | 28.4618 |
| CoreView_393 | 29.4774 | 0.954615 | 33.5957 |
| CoreView_394 | 31.0709 | 0.956886 | 30.4775 |
| Mean | 31.1271 | 0.962412 | 27.9389 |

## 2026-07-03 三数据集 TDP-semantic 对比基线阶段判断

用户观察：

```text
DNA-Rendering / I3D-Human / ZJU-MoCap 上，
TDP-semantic 相比 SeqAvatar baseline 都几乎没有变化。
```

当前可对齐的均值差：

```text
DNA-Rendering, novelview, 25000 iter, densify_until_iter=1500:
  TDP-semantic - baseline
  dPSNR        = -0.0186
  dSSIM        = -0.000022
  dLPIPS*1000  = +0.0071

ZJU-MoCap, test, 3000 iter:
  使用 render-log 最终行对齐:
    baseline mean      = PSNR 31.1122, SSIM 0.962380, LPIPS*1000 28.0268
    TDP-semantic mean  = PSNR 31.1300, SSIM 0.962474, LPIPS*1000 27.8958
    delta              = PSNR +0.0178, SSIM +0.000095, LPIPS*1000 -0.1310
  使用 JSON 主结果:
    TDP-semantic mean  = PSNR 31.1271, SSIM 0.962412, LPIPS*1000 27.9389
    与 baseline 差值同样属于极小波动。

I3D-Human, 15000 iter:
  原始脚本 baseline 设置为 15000。
  baseline 结果来源:
    output/I3D-Human/<sequence>/orginal/20260616_212011/metrics/results_*_15000.json
  TDP-semantic 结果来源:
    output/I3D-Human/<sequence>/tdp_semantic/20260703_143023/metrics/results_*_15000.json

  novelview:
    baseline mean      = PSNR 32.3824, SSIM 0.967281, LPIPS*1000 28.7066
    TDP-semantic mean  = PSNR 32.3101, SSIM 0.966876, LPIPS*1000 29.0808
    delta              = PSNR -0.0723, SSIM -0.000405, LPIPS*1000 +0.3742

  novelpose:
    baseline mean      = PSNR 30.3859, SSIM 0.958848, LPIPS*1000 34.8498
    TDP-semantic mean  = PSNR 30.2448, SSIM 0.958208, LPIPS*1000 35.6713
    delta              = PSNR -0.1411, SSIM -0.000640, LPIPS*1000 +0.8215
```

阶段性判断：

```text
TDP-semantic 当前没有形成稳定有效收益。
DNA 与 ZJU 差值接近随机波动量级。
I3D 按原始 15000 baseline 对齐后，TDP-semantic 明确略低于 baseline，尤其 novelpose LPIPS 变差更明显。
```

主要原因分析：

```text
1. TDP keep-base 的新通道与 baseline full motion 高度冗余。
   DNA 中:
     full_3 = 97->100
     full_2 = 98->100
     full_1 = 99->100
     v1 = 97->98
     v2 = 98->99
     v3 = 99->100
   对 xyz 这种线性位移来说，v/full 之间几乎可由线性组合互相恢复；
   acceleration 也是这些 velocity 的二阶组合。
   因此 TDP 更多是重参数化，而不是提供真正新信息。

2. ZJU 的 time_step_num=2，TDP 金字塔太浅。
   ZJU keep-base 只有:
     [full_6, full_3, v1, v2, a1]
   其中 full_3 与 v2 语义接近，full_6 又接近 v1/v2 的平均。
   新增 trend 信息更少。

3. baseline 的 flatten MLP 已经能学习 full channel 的差分组合。
   即使没有显式 v/a，MLP 也可能从 [full_s] 中近似推断短期变化。

4. semantic gate 初始接近 0，会保护 baseline，但也容易让 vel/acc 被忽略。
   如果没有额外监督或强运动采样，gate 可能学不到明显作用。
   当前还没有 gate/residual 统计，无法确认它是没开、开了但无效，还是信号本身冗余。

5. 当前 TDP-semantic 不是严格 baseline 等价初始化。
   gate 接近 0 只抑制 vel/acc residual；
   full 分支本身是新的 FullEncoder，不是原 baseline encoder 权重/结构的直接复用。
   因此即使 vel/acc 没贡献，也可能只有接近 baseline，而不是完全等于 baseline。

6. acceleration 可能放大噪声。
   pose acceleration 是 log/axis-angle 空间中的近似二阶差分，不是严格角加速度；
   xyz acceleration 会放大 SMPL 顶点估计误差和局部抖动。

7. 全图平均指标会稀释动态区域收益。
   高运动肢体、衣服边界、silhouette 只占图像小区域。
   如果 TDP 只在这些区域有轻微收益，PSNR/SSIM/LPIPS 全图平均很难显著变化。
   但 DNA high-motion / boundary subset 也没有显示明确强收益，因此当前证据仍偏弱。

8. I3D 的时间间隔不均匀。
   I3D time_steps=[42,33,24]，局部间隔为 9,9,24。
   当前第一版 acceleration 仍是 velocity difference，未严格除以二阶时间间隔；
   这可能让 I3D 的趋势通道尺度和语义不干净。
```

建议改进优先级：

```text
P0: 先加诊断，不急着继续大规模跑。
  记录:
    gate_v mean/max/active ratio
    gate_a mean/max/active ratio
    ||f_vel|| / ||f_full||
    ||f_acc|| / ||f_full||
    residual / full feature norm
  如果 gate 长期接近 0，说明模型基本忽略 TDP。
  如果 gate 很大但指标无提升，说明 v/a 信号本身冗余或噪声大。

P1: 改成严格 baseline-residual 结构。
  不要让 full 分支重新学 baseline。
  建议:
    f_base = 原始 baseline SeqPoseEncoder/SeqXYZEncoder(full channels)
    f_tdp  = gate_v * VelEncoder(v) + gate_a * AccEncoder(a)
    output = f_base + zero_init_projection(f_tdp)
  这样初始模型严格等价或更接近 baseline，
  TDP 只作为残差增益，能更干净判断新增趋势信息是否有用。

P2: 做 baseline checkpoint adapter 实验。
  从 baseline checkpoint 初始化；
  冻结或半冻结 baseline encoder / non-rigid 主干；
  只训练 vel/acc adapter 和 gate 若干轮；
  如果 adapter 学不到收益，说明 TDP 作为附加信息价值有限。

P3: 做 part-aware / point-aware gate，而不是全局 gate。
  TDP 的潜在收益应该集中在手臂、腿、衣服边界等局部高运动区域。
  建议:
    pose gate: joint/part-level gate
    xyz gate: Gaussian/part-level gate
  或与 Part-MoE 结合，让动态 part 专家接收 TDP residual。

P4: 改趋势定义，减少冗余。
  当前 v/a 大多可由 full channel 线性组合恢复。
  更值得试:
    nonlinear motion residual:
      e_t = current - linear_extrapolate(previous states)
    normalized acceleration:
      a = (v_next - v_prev) / dt_between_velocity_centers
    direction-change / curvature feature:
      angle(v_prev, v_next), ||v_next - v_prev||
  尤其 I3D 必须处理不等时间间隔。

P5: 加 high-motion / local-region 训练或评价。
  评价:
    top motion frames
    silhouette boundary
    arms / legs / loose-cloth-like areas
  训练:
    high-motion frame sampling weight
    boundary/dynamic-region loss weight
  如果局部评价仍没有收益，应停止把 TDP 作为主线。

P6: 公平性补实验。
  I3D 当前已经有 baseline 15000 vs TDP-semantic 15000 的对齐结果。
  若要确认不是 seed 波动，需要至少一组 seed 复跑。
  25000 的 I3D 日志属于非原始脚本口径的额外重跑，不应再作为默认 baseline 口径。
```

当前决策：

```text
TDP-semantic 不建议作为第二创新点主线直接继续堆实验。
下一步最小成本是加 gate/residual 诊断 + baseline-residual adapter 版本。
如果诊断显示 gate 不用或高运动区域无收益，应转向 Part-aware temporal residual / dynamic-region objective，
而不是继续增加 TDP channel 或更复杂 encoder。
```

## 2026-07-03 I3D TDP-semantic iteration 二次纠正

用户提供三个数据集最初脚本，确认原始 baseline 设置为：

```text
DNA:
  iter = 25000

I3D:
  iter = 15000

ZJU:
  iter = 3000
```

因此此前根据 `logs/20260621_183404_I3D-Human_orginal_25000.log` 判断 I3D baseline 为 25000 是错误口径。
该 25000 log 是额外重跑，不是最初脚本 baseline 默认设置。

脚本已改回：

```text
scripts/exps_i3dhuman.sh:
  iter=${ITER:-15000}
```

影响：

```text
当前 I3D TDP-semantic run:
  logs/TDP/20260703_143023_I3D-Human_tdp_semantic_gpu*.log
  iteration = 15000

与原始 I3D baseline 15000 设置是对齐的。
不需要因为 iteration 问题重跑 I3D TDP-semantic。
```

验证：

```text
bash -n scripts/exps_i3dhuman.sh 通过
git diff --check -- scripts/exps_i3dhuman.sh 通过
```

## 2026-07-03 TDP-semantic 代码总结与后续改进方向

TDP-semantic 本次改动目标：

```text
在不影响 baseline / Part-MoE / MSTI / AMC / 普通 TDP flatten 的前提下，
把 TDP-keep-base 的 8 个 motion condition channel 按语义分开编码：
  full: [full_3, full_2, full_1]
  vel:  [v1, v2, v3]
  acc:  [a1, a2]
再用 gated residual 融合。
```

代码改动范围：

```text
arguments/__init__.py
  新增:
    use_tdp_semantic_encoder
    tdp_semantic_mode
    tdp_gate_init_bias
  约束:
    use_tdp_semantic_encoder 只能和 use_tdp=True、tdp_mode=keep_base 同时使用。
    use_tdp 与 MSTI / AMC-pair / AMC-causal 互斥。
  自动推导:
    tdp_keep_base cond channels = time_step_num + 2 * time_step_num - 1。

scene/dataset_readers.py
  TDP condition 构造仍由 get_seq_pose_xyz_cond_tdp() 完成。
  tdp_keep_base channel 顺序固定为:
    [full channels, velocity channels, acceleration channels]
  不开 use_tdp 时仍走原 baseline condition 构造。

scene/gaussian_model.py
  将:
    use_tdp_semantic_encoder
    tdp_semantic_mode
    tdp_base_time_step_num
    tdp_gate_init_bias
  传给 NonrigidDeformer。

nets/mlp_delta_non_rigid.py
  SeqPoseEncoder / SeqXYZEncoder 新增 TDP semantic 分支。
  开启后:
    full -> full_encoder
    vel  -> vel_encoder
    acc  -> acc_encoder
    fused = f_full + gate_v * f_vel + gate_a * f_acc
  gate 最后一层 bias 初始化为 tdp_gate_init_bias=-4.0，
  初始 sigmoid 约 0.018，尽量让模型从接近 full 分支开始。
  未开启 use_tdp_semantic_encoder 时仍使用原 flatten encoder。

scripts/exps_dnarendering.sh
scripts/exps_i3dhuman.sh
scripts/exps_zjumocap.sh
  新增 tdp_semantic 模式。
  日志写入 logs/TDP。
  支持 GLOBAL_LOG_SUFFIX，把两张 GPU 的主日志拆开。
  train/render 都传入一致的 motion_cond_time_step_num 和 TDP semantic 参数。
  I3D 默认 iter 保持原始脚本 15000。
  ZJU 默认 iter 保持 3000。
  DNA 默认 iter 保持 25000。
```

三数据集结果后的判断：

```text
TDP-semantic 目前没有形成稳定收益。
DNA 与 ZJU 只有极小波动级别差异。
I3D 在 15000 iter 对齐原始 baseline 后略低于 baseline。
因此不能把当前 TDP-semantic 作为第二创新点主线。
```

没有优化的核心原因：

```text
1. TDP keep-base 的 v/a 与 baseline full channels 信息高度冗余。
   对 xyz 位移，v/full/acc 很多可以线性组合互相恢复。

2. baseline flatten MLP 本身已经可能从 full channels 中学到近似差分。
   显式添加 v/a 不一定提供新信息。

3. semantic gate 只抑制 vel/acc，但 full 分支不是原 baseline encoder。
   所以当前结构不是严格 baseline 等价初始化。

4. gate bias=-4.0 保护 baseline，但也可能让 vel/acc 长期被忽略。
   当前还没有 gate/residual 统计，无法判断模型是否真的使用了 TDP。

5. acceleration 通道可能放大 SMPL pose / xyz 抖动。
   pose acceleration 只是 log/axis-angle 空间近似二阶差分，不是严格物理角加速度。

6. I3D 的时间间隔 [42,33,24] 对应局部间隔 9,9,24。
   当前 acceleration 对不等时间间隔的二阶归一化还不够严谨。

7. 如果收益只集中在四肢、衣服边界、silhouette 等局部动态区域，
   全图 PSNR/SSIM/LPIPS 很容易把收益稀释。
```

后续最小风险改进路线：

```text
P0: 先加诊断，不直接继续大跑。
  记录 gate_v / gate_a 的 mean、max、active ratio。
  记录 ||f_vel||/||f_full||、||f_acc||/||f_full||、residual/full norm。

P1: 改成真正 baseline-residual adapter。
  f_base = 原 baseline encoder(full channels)
  f_tdp  = gate_v * VelEncoder(v) + gate_a * AccEncoder(a)
  out    = f_base + zero_init_projection(f_tdp)
  这样更接近严格 baseline 初始化，只验证 TDP residual 是否有用。

P2: 从 baseline checkpoint 初始化 adapter。
  冻结或半冻结 baseline 主干，只训练 vel/acc adapter 和 gate。
  如果 adapter 仍没有收益，说明 TDP 附加信息价值有限。

P3: 做 part-aware / point-aware gate。
  TDP residual 不应全局作用，优先作用到高运动 part / Gaussian / boundary 区域。
  可和 Part-MoE 组合，让动态 part 专家接收 temporal residual。

P4: 换更少冗余的趋势定义。
  例如:
    current - linear_extrapolate(history)
    normalized acceleration with unequal dt
    direction-change / curvature feature

P5: 若 high-motion / boundary / part-local 评价仍无收益，应停止 TDP 主线，
  转向 dynamic-region objective 或 Part-aware temporal residual。
```

### TDP 改进项分别解决什么问题

这些改进不是同时全部做，而是按诊断顺序逐步排除问题。

```text
1. gate / residual 诊断
   作用:
     判断 TDP-semantic 里的 vel/acc 分支到底有没有被模型使用。
   能回答:
     gate 接近 0 -> 模型基本忽略 TDP，继续大跑意义小。
     gate 很大但指标不升 -> vel/acc 信号可能冗余或有噪声。
     residual norm 很小 -> TDP 分支影响不到 non-rigid deformation。
     residual norm 很大但指标差 -> TDP 分支在干扰 baseline。

2. baseline-residual adapter
   作用:
     让模型初始严格接近 baseline，只把 TDP 作为增量残差加入。
   当前问题:
     TDP-semantic 的 full_encoder 不是原 baseline encoder，
     所以即使 gate 抑制 vel/acc，也不是严格 baseline 等价。
   能回答:
     如果 adapter 有提升，说明 TDP 趋势信息确实有增量价值。
     如果 adapter 仍无提升，说明 v/a 对当前任务帮助有限。

3. 从 baseline checkpoint 初始化，只训练 adapter + gate
   作用:
     排除随机初始化和整体重新训练带来的干扰。
   能回答:
     在一个已经训练好的 baseline 上，单独给 vel/acc 一个小残差分支，
     是否还能进一步改善动态区域。
   好处:
     实验更便宜，解释更干净。
     不会因为主干重新训练波动掩盖 TDP 的真实作用。

4. part-aware / point-aware gate
   作用:
     让 TDP residual 只影响真正需要运动趋势的局部区域。
   当前问题:
     全局 gate 会把四肢/边界的小范围运动信号平均掉，
     也可能把 acceleration 噪声加到静态躯干区域。
   能回答:
     TDP 是否只对高运动 part / Gaussian / silhouette boundary 有局部收益。
   适合与 Part-MoE 结合:
     动态 part 专家接收 temporal residual，
     静态 part 继续保持 baseline 表达。

5. 替换趋势定义
   作用:
     减少 raw velocity / raw acceleration 与 baseline full channel 的冗余。
   当前问题:
     v/a 很多可以由 full channels 线性组合得到，
     acceleration 还可能放大 SMPL 抖动。
   更值得试的特征:
     current - linear_extrapolate(history)
       表示当前帧偏离匀速预测多少，更像非线性运动残差。
     normalized acceleration with unequal dt
       对 I3D 这种 [42,33,24] 不等间隔更合理。
     direction-change / curvature
       直接描述转向、急停、反向运动，比 raw v/a 更不冗余。
```

推荐执行顺序：

```text
先做 P0 gate/residual 诊断。

如果 gate≈0:
  不要继续大跑，先尝试 adapter 或调 gate 初始化。

如果 gate 有效但全图指标不升:
  看 high-motion / boundary / part-local 指标。

如果局部有收益:
  做 part-aware / point-aware gate 或和 Part-MoE 结合。

如果局部也无收益:
  放弃 raw TDP，改用 nonlinear residual / direction-change 特征。

如果 adapter + 新趋势定义仍无收益:
  停止把 TDP 作为第二创新点主线。
```

## 2026-07-03 TDP-adapter 实现与 ZJU 诊断结果

用户要求按上面的改进继续验证 TDP，最后判断是否还要继续。

本次没有把所有想法混在一个实验里，而是先做最小可解释版本：

```text
tdp_adapter

目标:
  让 baseline full 分支保持 baseline encoder 结构；
  TDP 的 velocity / acceleration 只作为 zero-init residual adapter；
  同时输出 gate / residual 诊断。

未放入本实验:
  baseline checkpoint adapter
  part-aware / point-aware gate
  nonlinear extrapolation trend
原因:
  避免一次改太多，无法解释指标变化来源。
```

新增代码：

```text
arguments/__init__.py
  tdp_semantic_mode 新增 baseline_residual。
  新增:
    tdp_debug_stats
    tdp_debug_interval

nets/mlp_delta_non_rigid.py
  NonrigidDeformer 新增 pop_tdp_stats()。
  SeqPoseEncoder / SeqXYZEncoder 新增 tdp_semantic_mode=baseline_residual:
    base/full path = 原 baseline encoder 结构，只吃 full channels。
    residual path = gate_v * VelEncoder(v) + gate_a * AccEncoder(a)。
    residual projection zero-init，初始 residual 输出为 0。
  记录:
    gate_v mean/max/active ratio
    gate_a mean/max/active ratio
    ||f_vel||/||f_full||
    ||f_acc||/||f_full||
    residual/full feature norm

train.py
  新增 maybe_log_tdp_stats()。
  开启 tdp_debug_stats 时按 interval 打印 [TDP Stats]。

scripts/exps_dnarendering.sh
scripts/exps_i3dhuman.sh
scripts/exps_zjumocap.sh
  新增 tdp_adapter / tdp_residual / tdp_baseline_residual 模式。
  tdp_adapter:
    use_tdp=1
    tdp_mode=keep_base
    use_tdp_semantic_encoder=1
    tdp_semantic_mode=baseline_residual
    tdp_debug_stats=1
```

验证：

```text
bash -n scripts/exps_dnarendering.sh 通过
bash -n scripts/exps_i3dhuman.sh 通过
bash -n scripts/exps_zjumocap.sh 通过
py_compile arguments / gaussian_model / mlp_delta_non_rigid / train 通过
dummy shape test 通过:
  pose_out = (1, 32)
  xyz_out  = (1, 4, 128)

ZJU smoke:
  CoreView_377, ITERATIONS=2, tdp_adapter
  train/render 均通过
  初始 residual_feat_base_feat_norm_ratio = 0
  说明 zero-init adapter 生效
```

### ZJU-MoCap tdp_adapter 六序列实验

设置：

```text
RUN_TIME = 20260703_180739
logs:
  logs/TDP/20260703_180739_ZJU-MoCap_tdp_adapter_gpu1.log
  logs/TDP/20260703_180739_ZJU-MoCap_tdp_adapter_gpu2.log

GPU1:
  CoreView_377, CoreView_386, CoreView_387
GPU2:
  CoreView_392, CoreView_393, CoreView_394

iter = 3000
densify_until_iter = 1200
time_step_num = 2
motion_cond_time_step_num = 5
tdp_semantic_mode = baseline_residual
```

下面使用 render.py 最终 test 行，与 baseline `20260618_233204_ZJU-MoCap_orginal_3000.log` 口径一致。

| Sequence | Base PSNR | Adapter PSNR | dPSNR | Base SSIM | Adapter SSIM | dSSIM | Base LPIPS*1000 | Adapter LPIPS*1000 | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CoreView_377 | 31.4262 | 31.4697 | +0.0434 | 0.973127 | 0.973527 | +0.000400 | 18.2643 | 17.9429 | -0.3214 |
| CoreView_386 | 34.0227 | 33.9260 | -0.0966 | 0.969627 | 0.969310 | -0.000317 | 25.0206 | 24.9220 | -0.0986 |
| CoreView_387 | 28.7058 | 28.8055 | +0.0997 | 0.955544 | 0.955720 | +0.000176 | 32.4304 | 32.2320 | -0.1984 |
| CoreView_392 | 31.9875 | 31.9840 | -0.0035 | 0.964477 | 0.964325 | -0.000152 | 28.4383 | 28.4842 | +0.0460 |
| CoreView_393 | 29.4300 | 29.4486 | +0.0186 | 0.954530 | 0.954590 | +0.000061 | 33.6920 | 33.5379 | -0.1542 |
| CoreView_394 | 31.1008 | 31.0688 | -0.0320 | 0.956973 | 0.956922 | -0.000051 | 30.3150 | 30.7222 | +0.4072 |
| Mean | 31.1122 | 31.1171 | +0.0049 | 0.962380 | 0.962399 | +0.000019 | 28.0268 | 27.9735 | -0.0532 |

诊断统计，取每个序列 ITER 3000 的 `[TDP Stats]` 平均：

```text
pose/gate_v_mean                = 0.024088
pose/gate_a_mean                = 0.009739
pose/gate_v_active_0p1          = 0.004278
pose/gate_a_active_0p1          = 0.000000
xyz/gate_v_mean                 = 0.089826
xyz/gate_a_mean                 = 0.007278
xyz/gate_v_active_0p1           = 0.083333
xyz/gate_a_active_0p1           = 0.000000
pose/residual_feat_base_ratio   = 0.071598
xyz/residual_feat_base_ratio    = 0.800741
```

解释：

```text
1. acceleration gate 基本完全没开。
   pose gate_a_active_0p1 = 0
   xyz gate_a_active_0p1 = 0
   说明 raw acceleration / trend 在当前定义下没有被模型采用。

2. pose velocity 也基本没开。
   pose gate_v_active_0p1 平均只有 0.004278。
   pose residual 对最终特征影响很小。

3. 主要被使用的是 xyz velocity residual。
   xyz residual/base feature norm 不小，但最终指标只提升:
     PSNR +0.0049
     SSIM +0.000019
     LPIPS*1000 -0.0532
   这个幅度仍属于随机波动级别，不能作为有效创新点证据。
```

阶段结论：

```text
不建议继续把 TDP 作为第二创新点主线。
```

原因：

```text
1. 普通 TDP flatten 在 DNA 上没有超过 baseline。
2. TDP-semantic 在 DNA / I3D / ZJU 三个数据集都没有稳定收益。
3. 本次更严格的 baseline-residual adapter 在 ZJU 上也只有极小波动收益。
4. 诊断显示 acceleration 基本没被用，pose trend 也基本没被用。
5. 真正被用到的是 xyz velocity residual，但它带来的收益极小。
```

后续建议：

```text
停止继续大规模 TDP 实验。

如果还想保留 motion 方向，只保留一个很小的支线：
  dynamic-region / part-aware temporal residual

也就是:
  不再做全局 TDP channel；
  不再强调 acceleration pyramid；
  只把局部 velocity residual 给高运动 part / boundary / Gaussian。

更优先的方向:
  回到 Part-MoE 主线，
  或做 Part-MoE + 局部动态 residual，
  而不是继续扩展 TDP。
```

## 2026-07-03 STMS 网络级优化候选

用户根据论文 4.2 Spatio-temporal Multi-scale Sampling 总结了 STMS 的不足，并要求提出网络级优化，不能只是加残差或调权重。

当前判断：

```text
不要再把重点放在 raw TDP / 直接追加 velocity-acceleration channel。
TDP 系列实验已经说明：
  只改变 condition channel 或加语义分支，收益不稳定；
  acceleration 基本不被使用；
  全局指标和 high-motion / boundary 证据不足。

新的 STMS 优化应直接改变 motion encoder 的结构归纳偏置：
  显式区分尺度；
  动态选择时间尺度；
  动态选择空间邻域；
  建模粗骨架 motion 与细顶点 motion 的交互；
  对不同 body part / Gaussian 使用不同 motion context。
```

### 候选 1：Scale-Token Temporal Transformer

目标：

```text
解决 STMS 多尺度 motion 只是 concat 的问题。
```

核心做法：

```text
把每个时间尺度当作一个 token，而不是 flatten channel：
  full_1 token
  full_2 token
  full_3 token

每个 token 加 scale embedding / dt embedding，
再用 cross-scale self-attention 建模短期、中期、长期尺度之间的关系。
```

可用于：

```text
SeqPoseEncoder:
  skeleton scale tokens -> transformer -> f_deltaP

SeqXYZEncoder:
  per-Gaussian local vertex scale tokens -> lightweight transformer / attention mixer -> f_V
```

优点：

```text
1. 明确保留尺度语义，不再让 MLP 从 flatten 向量中猜。
2. 不需要新增 RGB / mask / camera。
3. 比 TDP 更贴近 STMS 原始问题：多尺度采样后的融合方式太弱。
```

风险：

```text
1. 计算量比 MLP 大，XYZ per-Gaussian attention 需要做轻量化。
2. 仍然依赖原始固定尺度，不能完全解决尺度自适应问题。
```

### 候选 2：Motion-Adaptive Scale Router

目标：

```text
解决固定时间尺度不能按动作状态自适应的问题。
```

核心做法：

```text
为 short / mid / long scale 建立不同 temporal experts：
  E_short
  E_mid
  E_long

根据当前 pose motion、xyz local motion、body part 或 Gaussian feature 生成 routing logits，
对每个 part / Gaussian 动态选择 top-k scale experts。
```

与普通加权不同：

```text
不是把 3 个尺度简单乘权重求和；
而是不同尺度进入不同专家网络，
每个专家学习不同时间窗口下的 non-rigid motion pattern。
```

适合 SeqAvatar 的解释：

```text
快速挥手 / 踢腿:
  动态 limbs 路由到 short-scale expert。

躯干 / 慢动作:
  路由到 long-scale expert。

突然停止 / 转向:
  可以同时激活 short + mid expert。
```

优点：

```text
1. 网络级结构变化明显，论文解释强。
2. 能和 Part-MoE 主线自然结合。
3. 比全局 TDP 更符合“局部动态区域才需要特殊 motion modeling”的实验结论。
```

风险：

```text
1. 需要控制路由稳定性，避免所有点都塌缩到同一个 expert。
2. 需要记录 routing 分布，作为可解释性证据。
```

### 候选 3：Dynamic Spatio-Temporal Motion Graph

目标：

```text
解决空间 KNN 固定、欧氏最近不一定 motion 相关的问题。
```

核心做法：

```text
构建动态图：
  节点: SMPL joints / SMPL vertices / Gaussians
  边:
    1. kinematic skeleton edges
    2. canonical KNN edges
    3. motion-similarity dynamic edges
    4. optional part edges

用 graph attention / message passing 在时空图上传播 motion feature。
```

对当前固定 KNN 的替代：

```text
先保留 K 个 canonical nearest vertices 作为 candidate pool，
再用 motion query 选择真正相关的邻居：
  Gaussian query = [x, pose feature, local velocity]
  SMPL vertex key = [template position, part label, motion feature]
  attention 得到动态邻域聚合。
```

优点：

```text
1. 直接针对 STMS 空间采样不足。
2. 更适合衣服、边界、关节附近区域。
3. 可以输出邻域 attention map，解释哪些 SMPL 顶点影响某个 Gaussian。
```

风险：

```text
1. 工程改动比 temporal encoder 大。
2. per-Gaussian graph attention 显存压力大，需要 candidate pool 和低维 attention。
```

### 候选 4：Coarse-Fine Motion Cross-Attention

目标：

```text
解决粗骨架 motion f_deltaP 与细顶点 motion f_V 只是隐式拼接融合的问题。
```

核心做法：

```text
把 skeleton joints 作为 coarse tokens；
把 Gaussian-local KNN vertices 或 Gaussian motion features 作为 fine tokens；
做双向 cross-attention：

  fine queries attend to skeleton tokens:
    每个 Gaussian 知道应该关注哪些 joints / body parts。

  skeleton queries attend to fine tokens:
    骨架 motion feature 获得局部非刚性反馈。
```

最终 non-rigid MLP 输入：

```text
不是简单 [x, pose, f_deltaP, f_V]，
而是 cross-attended motion feature:
  f_motion_i = CrossAttn(fine_i, skeleton_tokens, vertex_tokens)
```

优点：

```text
1. 直接解决 STMS 粗细 motion 融合简单的问题。
2. 对不同 Gaussian 自适应选择 coarse / fine 信息源。
3. 比单独强化 pose 或 xyz encoder 更有结构创新。
```

风险：

```text
1. 需要设计 token 数量，避免 vertex token 太多。
2. 需要确认 cross-attention 不引入明显 OOM。
```

### 候选 5：Causal Motion State Encoder

目标：

```text
解决 STMS 只看 t-s -> t 位移，缺少连续运动过程建模的问题。
```

核心做法：

```text
不再手工堆 raw velocity / acceleration channel，
而是让网络按时间顺序编码 motion state：

  P_{t-3}, P_{t-2}, P_{t-1}, P_t
  X_{t-3}, X_{t-2}, X_{t-1}, X_t

通过 TCN / GRU / SSM / lightweight causal transformer 得到 motion state。
```

和 TDP 的区别：

```text
TDP:
  手工构造 [v, a]，再送 MLP。

Causal Motion State Encoder:
  网络直接看有序状态序列，
  自己学习速度、转向、急停、非线性趋势。
```

优点：

```text
1. 避免 raw acceleration 放大 SMPL 抖动。
2. 比 AMC-causal 更正式，可以统一 pose/xyz 的状态建模。
3. 对 fast motion path change 的解释更自然。
```

风险：

```text
1. 对 I3D/ZJU 不等间隔采样要加入 dt embedding。
2. 如果只在全局 pose 上做，局部收益可能仍被全图平均淹没。
```

### 候选 6：Part-Aware Scale-Adaptive Motion Field

目标：

```text
把 STMS 改成 body-part aware 的动态 motion field，
同时承接第一创新点 Part-MoE。
```

核心做法：

```text
每个 body part 拥有独立的 temporal-scale encoder / scale router：
  torso expert
  arm expert
  leg expert
  head expert
  hand/foot optional expert

Gaussian 根据 SMPL part label / learned part assignment 选择对应 motion field。
动态 part 使用更短期、更强的 temporal encoder；
稳定 part 使用长期尺度或轻量 encoder。
```

与 Part-MoE 的关系：

```text
Part-MoE 主要解决 non-rigid deformation 的空间部位专家化。
Part-Aware STMS 则解决每个部位使用什么 motion context。

二者可以组合成：
  part-specific temporal motion encoder
  +
  part-specific deformation expert
```

优点：

```text
1. 最符合当前实验结论：全局 TDP 没收益，局部动态区域才可能有收益。
2. 与已有 Part-MoE 代码和论文叙事兼容。
3. 可解释性强：不同 part 的 scale routing / motion attention 可以可视化。
```

风险：

```text
1. 需要谨慎隔离，不能让 Part-MoE 默认依赖 Motion 模块。
2. 需要 part-local / high-motion subset 评价，否则全图指标可能看不出收益。
```

### 推荐优先级

```text
P1: Part-Aware Scale-Adaptive Motion Field
    最适合当前项目，因为它能承接 Part-MoE，并避免继续做全局 TDP。

P2: Scale-Token Temporal Transformer
    最容易从现有 SeqPoseEncoder / SeqXYZEncoder 改起，直接解决 concat 弱点。

P3: Coarse-Fine Motion Cross-Attention
    结构创新较强，针对 f_deltaP / f_V 融合不足。

P4: Dynamic Spatio-Temporal Motion Graph
    论文价值高，但工程和显存风险较大。

P5: Causal Motion State Encoder
    比 TDP 更合理地建模连续运动，但需要重新定义跨数据集时间窗口。
```

当前最建议的第二创新点候选：

```text
Part-aware Scale-Adaptive STMS

一句话定义：
  让每个 Gaussian / body part 根据自身运动状态，
  在短期、中期、长期 motion experts 中动态选择时空上下文，
  并通过 coarse-fine cross-attention 融合骨架和局部 vertex motion。
```

## 2026-07-03 Part-MoE 对 STMS 的影响确认

问题：

```text
在已加入的 Part-MoE 消融实验中，STMS 是否受到影响？
```

代码结论：

```text
普通 Part-MoE 消融不会改变 STMS 的 motion condition 构造，也不会改变 STMS encoder 输入通道数。
```

依据：

```text
1. STMS condition 构造仍在 scene/dataset_readers.py:get_seq_pose_xyz_cond()。
   普通 Part-MoE 没有作为 motion_cond_options 传入该函数。
   只有 use_msti / use_amc_pair / use_amc_causal / use_tdp 会切换 motion condition 分支。

2. scene/__init__.py 只把 use_msti / use_amc / use_tdp / motion_cond_time_step_num
   传给 dataset reader。
   use_part_moe 不参与 time_steps 或 seq_pose_conds / seq_xyz_conds 生成。

3. scripts 中普通 part_moe / part_moe_leg / part_moe_foot / part_moe_arm:
   use_msti=0
   use_amc_pair=0
   use_amc_causal=0
   use_tdp=0
   因此 motion_cond_time_step_num = time_step_num。

4. NonrigidDeformer.forward() 中执行顺序是:
   SeqPoseEncoder(seq_pose_conds)
   SeqXYZEncoder(seq_xyz_conds, x_emb)
   concat features
   然后才根据 use_part_moe / part_label 路由到 part experts。

5. 因此 Part-MoE 改的是 STMS feature 之后的 non-rigid deformation head：
   baseline shared MLP/head
   -> part-specific expert MLP/head
   而不是 STMS 的采样、delta 计算、KNN、encoder channel 或 condition 内容。
```

需要区分的例外：

```text
part_moe_leg_msti 是组合实验。
它显式打开 use_msti=1，因此会改变 STMS motion condition channel。
这个影响来自 MSTI，不是普通 Part-MoE 本身。
```

表述建议：

```text
普通 Part-MoE 可以写成:
  在保持 SeqAvatar 原 STMS motion condition 不变的情况下，
  将 non-rigid deformation head 改为 part-specific mixture-of-experts。

不要写成:
  Part-MoE 改进了 STMS motion sampling。
```

## 2026-07-03 Gaussian 按 part label / learned assignment 选择 motion field 的具体方案

目标：

```text
把 STMS 从“所有 Gaussian 共用同一套 motion encoder / 同一组尺度融合”
改成“不同 body part / Gaussian 使用不同 motion field”。
```

这不是普通 Part-MoE 的重复：

```text
Part-MoE:
  STMS feature 已经算完；
  再把后面的 non-rigid deformation head 按 part 分专家。

Part-aware motion field:
  在 STMS motion encoder 阶段就按 part / Gaussian 选择 motion context；
  解决每个部位应该看什么时间尺度、什么骨架关节、什么局部 vertex motion。
```

### 版本 A：SMPL hard label 路由

第一版建议先做 hard label，因为当前代码已经有标签生成链路：

```text
part_moe_start_iter 时:
  canonical Gaussian -> nearest canonical SMPL vertex
  SMPL vertex dominant LBS joint -> part label
  保存 gaussian_part_label.npy
```

新增一个独立开关：

```text
use_part_motion_field = False
part_motion_assignment = smpl_hard
part_motion_start_iter = part_moe_start_iter
part_motion_warmup = 1000
```

不要强制依赖 `use_part_moe`，但可以复用同一份 `gaussian_part_label.npy`。

结构：

```text
seq_pose_conds: [B, L, S, J, 3]
seq_xyz_conds:  [B, N, L, K, S, 3]
part_label:     [N]

对每个 part p 建 motion encoder:
  E_pose_p
  E_xyz_p
  optional scale router R_p

Gaussian i:
  p_i = part_label[i]
  f_pose_i = E_pose_{p_i}(seq_pose_conds)
  f_xyz_i  = E_xyz_{p_i}(seq_xyz_conds_i, x_i)
  f_motion_i = fuse_p(f_pose_i, f_xyz_i)
```

为了省显存，不建议第一版真的复制完整大 encoder：

```text
更稳实现:
  shared scale token projection
  + part-specific small temporal adapters / experts
  + part-specific scale router
```

例如：

```text
shared_pose_tokens = PoseScaleProjector(seq_pose_conds)
shared_xyz_tokens_i = XYZScaleProjector(seq_xyz_conds_i)

for part p:
  f_i_p = PartMotionExpert_p(shared_pose_tokens, shared_xyz_tokens_i, x_i)

hard route:
  f_i = f_i_{part_label[i]}
```

### 版本 B：learned soft assignment

hard label 的问题：

```text
衣服、边界、高斯漂移区域不一定严格属于最近 SMPL 顶点；
关节附近可能需要同时参考上下游 part；
nearest SMPL label 可能过硬。
```

learned assignment 做法：

```text
a_i = softmax(AssignNet([x_emb_i, local_xyz_motion_i, optional_smpl_part_onehot_i]))
a_i: [num_parts]

f_i = sum_p a_{i,p} * f_i_p
```

训练约束：

```text
用 SMPL hard label 做初始化监督:
  CE(a_i, smpl_part_label_i)

边界 / unknown / 低置信度区域可以减小 CE 权重，
允许 soft blend。

可加 entropy / balance 正则，避免所有 Gaussian 塌缩到一个 part。
```

阶段建议：

```text
先做 smpl_hard。
如果 hard label 有局部收益，再做 learned_soft。
不要第一版直接 learned assignment，否则难判断收益来自 motion field 还是 assignment 学习。
```

### part 内部的 scale-adaptive motion field

每个 part 不应只选择一个普通 encoder，而应选择不同时间尺度专家：

```text
E_{p,short}
E_{p,mid}
E_{p,long}
```

router：

```text
r_{i,s} = R_p([x_i, f_xyz_i, pose_motion_score_p, scale_dt_s])
f_i = sum_s r_{i,s} * E_{p,s}(motion_token_{i,s})
```

解释：

```text
手臂 / 腿:
  更常用 short-scale 或 high-frequency expert。

躯干:
  更常用 mid/long-scale expert。

关节边界:
  可以混合相邻 part 或多个 scale expert。
```

### 为什么可能有效

核心原因：

```text
STMS 当前是全局统一 motion encoder。
同一个 skeleton motion feature 会扩展给所有 Gaussian。
但不同部位的非刚性形变规律完全不同。
```

具体收益来源：

```text
1. 减少全局 motion 混淆
   手部快速运动不应该强迫躯干 Gaussian 使用同样的短期高频 motion。
   躯干稳定信息也不应该冲淡手臂/腿的动态信息。

2. 时间尺度更合理
   limbs 需要短期尺度；
   torso/head 更需要稳定长期上下文；
   关节和轮廓边界需要多尺度混合。

3. 空间邻域更语义化
   固定 KNN 只看欧氏距离，关节附近容易取到运动语义不一致的顶点。
   part label / soft assignment 给 KNN motion 加了 body semantic prior。

4. 更符合已有实验结果
   全局 TDP / AMC 的收益很弱；
   说明“给所有 Gaussian 加同一类 temporal 信息”不够有效。
   如果 motion 增益存在，更可能集中在高运动 part / boundary / limb 区域。

5. 可以和 Part-MoE 形成互补
   Part-aware motion field 负责“每个 part 看什么 motion context”；
   Part-MoE 负责“每个 part 如何预测 deformation”。
```

### 最小消融路线

```text
Step 1:
  part_motion_hard
  只用 SMPL hard label 路由 motion field；
  不开 Part-MoE deformation experts。
  目的: 单独验证 part-aware STMS 是否有效。

Step 2:
  part_motion_hard + part_moe
  目的: 验证 part-aware motion encoder 和 part-specific deformation head 是否互补。

Step 3:
  part_motion_soft
  用 learned soft assignment 替代 hard label。
  目的: 处理衣服、边界、关节附近 hard label 不准确的问题。
```

评价不能只看全图平均：

```text
必须同时看:
  high-motion subset
  part-local metrics
  silhouette / boundary metrics
  routing distribution per part
  scale router distribution per part
```

## 2026-07-03 STMS / Part-MoE / Part-Motion 框架图

已生成三张会议论文风格的 SVG 矢量框架图：

```text
note/figures/baseline_stms_pipeline.svg
  Baseline SeqAvatar STMS:
  multi-scale skeleton motion + local vertex KNN motion
  -> shared SeqPoseEncoder / SeqXYZEncoder
  -> shared non-rigid MLP。

note/figures/part_moe_pipeline.svg
  SeqAvatar + Part-MoE:
  STMS motion condition 和 motion encoder 保持不变；
  只在 STMS feature 之后按 Gaussian part label 路由到 part-specific deformation experts。

note/figures/part_moe_part_motion_pipeline.svg
  SeqAvatar + Part-MoE + Part-aware Motion Field:
  Gaussian part assignment 同时控制 motion-context encoder 和 deformation expert；
  动态 part 可选择 short-scale motion，稳定 part 可选择 long-scale motion。
```

校验：

```text
python3 XML parse 通过。
当前环境没有 rsvg-convert / inkscape / magick / convert，因此暂未导出 PNG。
SVG 可直接用浏览器打开，也可后续在论文工具链中转 PDF/PNG。
```

补充：

```text
2026-07-03 已将三张 SVG 改成中英文双语版本。
标题、模块名、关键说明和底部解释均包含中文和英文。
再次用 python3 XML parse 校验通过。
当前环境没有 cairosvg / rsvg-convert / inkscape / magick / convert，因此仍只保留 SVG。
```

再次补充：

```text
2026-07-03 已把三张 SVG 中的代码式数学写法改为真正 SVG 上下标。
例如:
  x_i -> x 下标 i
  P_t -> P 下标 t
  ΔP^{s1} -> ΔP 上标 s_1
  f_ΔP -> f 下标 ΔP
  R^{32} -> R 上标 32
  E0 / Ep -> E 下标 0 / p

python3 XML parse 通过；
rg 检查未再发现 x_i / P_t / E_V / f_ / ^{} 等代码式上下标残留。
```

## 2026-07-03 Part motion experts 输出如何接回统一 non-rigid MLP

问题：

```text
如果根据 SMPL part label 选择对应 motion MLP，
多个 motion experts 不合并成一个 MLP，
但后面的 non-rigid MLP 只接收一个统一 motion embedding，
这两者如何兼容？
```

结论：

```text
不合并 experts 的参数。
只在输出 tensor 层面按 Gaussian part label 做 gather / routing。
后面的 non-rigid MLP 仍然只接收一个统一维度的 f_motion_i。
```

形状设计：

```text
原始 SeqAvatar:
  f_deltaP: [B, Dp] -> expand -> [B, N, Dp]
  f_V:      [B, N, Dv]
  features_i = [x_i, P_t, f_deltaP, f_V_i]

Part-aware motion field:
  E_pose_p(seq_pose_conds) -> f_deltaP_p: [B, P, Dp]
  E_xyz_p(seq_xyz_conds_i) -> f_V_p:      [B, N, P, Dv]

  part_label_i -> p_i

  f_deltaP_i = gather(f_deltaP_p, p_i) -> [B, N, Dp]
  f_V_i      = gather(f_V_p, p_i)      -> [B, N, Dv]

  features_i = [x_i, P_t, f_deltaP_i, f_V_i]
```

因此后面的接口仍然是：

```text
δx_i, δs_i, δr_i = E_nonrigid(x_i, P_t, f_deltaP_i, f_V_i)
```

只是 `f_deltaP_i / f_V_i` 的来源从 shared encoder 变成：

```text
Gaussian i 的 part label 选择出来的 part-specific motion encoder 输出。
```

实现上有两种等价方式：

```text
方式 A: 先算所有 part 的 embedding，再 gather。
  优点: 代码直观。
  缺点: 多算了一些未被使用的 part embedding。

方式 B: 像当前 Part-MoE deformation head 一样，按 part label 分组 index_select，
        每组只跑对应 expert，再 index_copy 回统一输出 tensor。
  优点: 更省计算和显存。
  缺点: 代码稍复杂。
```

重要表述：

```text
这里“统一 motion embedding”指的是输出张量接口统一，
不是把多个 expert 的参数合并成一个网络。

也就是说:
  多个 expert 参数保留；
  每个 Gaussian 只拿一个对应 expert 的输出；
  non-rigid MLP 看到的仍是一个 f_motion_i。
```

如果和 Part-MoE deformation head 组合：

```text
part label 同时用于:
  1. 选择 part-specific motion field，得到 f_motion_i；
  2. 选择 part-specific deformation expert，预测 δx_i / δs_i / δr_i。

两个模块可以共享 part label，但不能混成一个模块解释。
```

## 2026-07-03 Part motion field 的分专家主轴

问题：

```text
Part-aware STMS 里 motion MLP 到底应该按 body part 分，
还是按长短时间序列 / temporal scale 分？
```

结论：

```text
第一版消融建议按 part 分 MLP。
长短序列 / scale 不作为第一版的专家主轴，而是作为每个 part motion MLP 的输入。
```

原因：

```text
1. 当前已有 Part-MoE 标签链路，按 part 分 motion MLP 最容易复用。
2. 已有实验表明全局 TDP / AMC / TDP-semantic 没有稳定收益，
   说明单纯沿时间维度加专家或加通道不够可靠。
3. STMS 的真正问题之一是所有 Gaussian 共享同一套 motion encoder。
   按 part 分 MLP 可以直接解决不同身体部位运动规律不同的问题。
4. 如果第一版同时按 part 和 scale 分成 E_{p,s}，
   参数量和解释复杂度都会上升，不利于判断收益来源。
```

第一版定义：

```text
part_motion_hard

输入仍是 baseline STMS 的多尺度 condition:
  [full_s1, full_s2, full_s3]

但 motion encoder 按 part label 选择:
  M_body([full_s1, full_s2, full_s3])
  M_arm([full_s1, full_s2, full_s3])
  M_leg([full_s1, full_s2, full_s3])
  ...

输出统一维度:
  f_motion_i = M_{part_i}(multi_scale_motion_i)
```

也就是说：

```text
按 part 分专家；
每个 part 专家内部仍然看完整的长/中/短多尺度序列。
```

后续增强版本：

```text
part_scale_router

先按 part 选择 motion field，
再在 part 内部学习 short / mid / long scale 的动态选择。

形式:
  f_i = M_{part_i}( R_{part_i}([full_short, full_mid, full_long]) )

或者 factorized:
  shared scale encoders E_short / E_mid / E_long
  + part-specific router R_p
  + part-specific adapter A_p
```

不建议第一版直接做：

```text
E_{part,scale}
```

原因：

```text
专家数量 = num_parts * num_scales。
例如 6 parts * 3 scales = 18 个 motion experts，
参数量、显存、训练稳定性和消融解释都会变差。
```

最终论文表述可以是：

```text
We first introduce part-specific motion encoders to model heterogeneous motion patterns
across body regions. Temporal scales are retained as multi-scale inputs within each
part-specific encoder. A scale-adaptive router can be further introduced inside each
part branch to dynamically select short-, mid-, and long-term contexts.
```

## 2026-07-03 放弃 Part-aware motion field，转向 D-IF inspired Δx distribution

用户决定放弃 Part-aware motion field，转而参考：

```text
paper:
  note/D-IF.pdf

code:
  /media/image/mxz/human/D-IF_release
```

D-IF 关键机制：

```text
不是直接预测一个确定 occupancy value，
而是预测每个 query point 的 Gaussian distribution:
  μ(p), σ(p)

训练时:
  z ~ N(μ, σ)
  rectifier MLP 输入 [原特征, z, μ, σ]
  输出 refined occupancy。

代码对应:
  D-IF_release/lib/net/MLP_DIF.py
    line 82: mu_0, sigma_0 = torch.split(y, 1, dim=1)
    line 83: sigma_0 = F.softplus(sigma_0)
    line 85-86: Normal(mu_0, sigma_0).rsample()
    line 88-91: concat [feature, z/mu, mu, sigma]

  D-IF_release/lib/net/HGPIFuNet.py
    line 407-411: 用 target_sigma 和 KL loss 约束预测分布。
```

需要澄清：

```text
D-IF 的“表面点”不是直接从 Gaussian distribution 里挑一个点当表面点。
它仍然预测 occupancy / smooth occupancy field，
最后通过 iso-surface / Marching Cubes 得到表面。

SeqAvatar 是 3D Gaussian deformation，不是 occupancy field。
所以不能机械照搬“选 surface point”。
更合理的迁移是:
  把 deterministic Δx 改为 uncertainty-aware displacement distribution。
```

### 当前 SeqAvatar 对应位置

当前 non-rigid deformer 是确定性输出：

```text
nets/mlp_delta_non_rigid.py

h = self.mlp(features)
d_xyz      = self.gaussian_warp(h)
d_scaling  = self.gaussian_scaling(h)
d_rotation = self.gaussian_rotation(h)
```

也就是：

```text
Δx_i = MLP(features_i)
```

### 推荐第一版：Uncertainty-aware Δx head

只改 `Δx`，先不动 rotation / scaling：

```text
h = shared_nonrigid_mlp(features_i)

μ_i       = W_mu(h)          # [B, N, 3]
logσ_i    = W_sigma(h)       # [B, N, 1] or [B, N, 3]
σ_i       = softplus(logσ_i) + eps

训练:
  ε ~ N(0, I)
  z_i = μ_i + σ_i * ε

测试:
  z_i = μ_i

rectifier:
  r_i = R_delta([h_i, z_i, μ_i, σ_i])
  Δx_i = z_i + r_i
```

第一版建议：

```text
σ_i 先用 scalar isotropic std: [B, N, 1]。
不要第一版就用 [B, N, 3] diagonal std，避免不稳定。

rectifier 最后一层 zero-init。
这样初始时:
  Δx_i ≈ μ_i
```

### 为什么对 SeqAvatar 可能有用

SeqAvatar 的 `Δx` 存在天然不确定性：

```text
1. 多视角监督不是每个 Gaussian 都有直接 3D 位移 GT。
2. cloth / hair / silhouette boundary 的非刚性偏移本身多解。
3. 快速运动时，同一个 motion condition 到最终可见表面的映射不唯一。
4. 原始 deterministic Δx MLP 容易把所有不确定区域压成一个平均偏移，
   可能导致边界拖影、局部过平滑或错位。
```

D-IF-inspired 分布头可以让网络表达：

```text
这个 Gaussian 应该往哪里动，以及这个判断有多确定。
```

低不确定区域：

```text
σ 小，Δx 接近 μ，行为接近 baseline。
```

高不确定区域：

```text
σ 大，rectifier 可以利用 [z, μ, σ] 学到更稳健的修正。
```

### 训练难点：SeqAvatar 没有 Δx ground truth

D-IF 有 occupancy / smooth occupancy GT，可以直接约束：

```text
μ -> O_gt
σ -> target_sigma(distance to surface)
```

SeqAvatar 没有 `Δx_gt`，只有最终 render loss，因此不能直接写：

```text
KL(N(μ_Δx, σ_Δx), N(Δx_gt, σ_target))
```

需要使用 proxy uncertainty regularization。

可选 proxy：

```text
1. motion magnitude:
   高 motion Gaussian / 高 motion frame -> target σ 稍大。

2. SMPL distance:
   canonical Gaussian 离最近 SMPL vertex 越远，越可能是衣服/头发/非贴体区域，
   target σ 可稍大。

3. silhouette / boundary contribution:
   位于轮廓或高渲染残差区域的 Gaussian 允许更大 σ。

4. opacity / visibility stability:
   长期稳定可见的 Gaussian σ 小；
   可见性不稳定区域 σ 大。
```

第一版最稳：

```text
target_sigma_i = σ_min + (σ_max - σ_min) * normalize(
    a * motion_score_i + b * smpl_dist_i
)

L_sigma = SmoothL1(log σ_i, stopgrad(log target_sigma_i))
L_var   = mean(σ_i^2)  # 防止 σ 爆炸
```

同时保留原 render loss：

```text
L = L_render + λ_sigma L_sigma + λ_var L_var
```

### 更稳的训练路线

推荐从 baseline checkpoint 初始化：

```text
1. μ head 初始化为原 gaussian_warp。
2. σ head bias 初始化为很小值，例如 σ ≈ 1e-4 或 1e-3。
3. rectifier zero-init。
4. 前几千步只训练 σ head + rectifier，或给 μ head 小学习率。
```

这样可以保证初始行为接近 baseline：

```text
Δx ≈ μ ≈ baseline Δx。
```

否则随机采样 `z = μ + σε` 很容易直接伤 PSNR / SSIM。

### 关于“选择特定点作为表面点”

如果一定要引入“从多个候选位移中选择”的思想，可以做第二阶段：

```text
K 个候选:
  Δx_i^k = μ_i + σ_i ε_k

用 soft selection:
  w_k = softmax(-E_k / τ)
  Δx_i = Σ_k w_k Δx_i^k
```

但难点是 `E_k` 怎么定义：

```text
如果每个 candidate 都完整 render 一次，代价太高。
如果只用 SMPL distance / temporal smoothness 做 E_k，又可能和真实图像误差脱节。
```

所以第一版不建议做 K-sample selection。
第一版只做：

```text
single rsample during training + μ during test + rectifier。
```

### 推荐实验命名

```text
dif_delta_x
uncertain_delta_x
delta_x_distribution
```

最小消融：

```text
baseline
dif_delta_x_mean      # 只预测 μ/σ，但 forward 用 μ，不采样，用 σ 作为 rectifier 输入
dif_delta_x_sample    # train 用 rsample，test 用 μ
dif_delta_x_rectifier # sample + rectifier
```

阶段判断：

```text
如果 dif_delta_x_mean 都不提升:
  说明 μ/σ 描述本身没有带来增益，采样版风险更高。

如果 mean 稳定、sample 下降:
  说明 stochastic training 噪声伤害渲染，保留 deterministic uncertainty descriptor。

如果 sample + rectifier 提升 high-motion/boundary:
  该方向可以作为第二创新点继续推进。
```

### 从不确定分布中得到最终 Δx 的规则

问题：

```text
把确定性 Δx MLP 改成输出 μ 和 σ 后，
最终用于 Gaussian deformation 的 Δx 应该怎么从分布里选？
```

结论：

```text
第一版不要做 hard selection。
训练时用 reparameterized sample；
测试时用均值 / MAP，也就是 μ。
```

原因：

```text
对 Gaussian distribution N(μ, σ²)，概率最大点就是 μ。
如果没有额外 render-level candidate score，
硬从多个 sample 里选一个并不可靠，还可能造成帧间抖动。
```

推荐 forward：

```text
h_i = shared_nonrigid_mlp(features_i)

μ_i = W_mu(h_i)                         # [B, N, 3]
σ_i = softplus(W_sigma(h_i)) + eps       # [B, N, 1] or [B, N, 3]

if training and use_delta_x_sampling:
    ε_i ~ N(0, I)
    z_i = μ_i + σ_i * ε_i                # rsample, 可反传
else:
    z_i = μ_i                            # MAP / mean

if use_delta_x_rectifier:
    r_i = R_delta([h_i, z_i, μ_i, σ_i])
    Δx_i = z_i + r_i
else:
    Δx_i = z_i
```

测试时：

```text
Δx_i = μ_i
```

如果有 rectifier：

```text
Δx_i = μ_i + R_delta([h_i, μ_i, μ_i, σ_i])
```

这样测试是确定性的，不会因为随机采样导致同一帧重复渲染结果不同。

可选但不建议第一版做的 hard/soft candidate selection：

```text
采 K 个候选:
  Δx_i^k = μ_i + σ_i ε_i^k

定义 proxy energy:
  E_i^k =
    λ_smpl * surface_prior(x_i + Δx_i^k)
  + λ_temp * ||Δx_i^k - Δx_{i,prev}||
  + λ_mag  * ||Δx_i^k||

soft selection:
  w_i^k = softmax(-E_i^k / τ)
  Δx_i = Σ_k w_i^k Δx_i^k
```

但风险很大：

```text
1. 真正可靠的 E_i^k 应该来自 render loss；
   但每个 candidate 都 render 一次代价太高。
2. 只用 SMPL distance / temporal smoothness 做 E，
   可能选择到“几何上平滑但图像上错误”的 Δx。
3. hard argmin 不可导，softmin 又会退化成加权平均。
4. 随机候选选择容易造成 temporal flicker。
```

因此推荐消融顺序：

```text
1. dif_delta_x_mean:
   不采样，Δx = μ 或 μ + R([h, μ, μ, σ])。
   σ 作为 uncertainty descriptor 参与 rectifier 和 regularization。

2. dif_delta_x_sample:
   train 用 z = μ + σε，test 用 μ。
   验证 stochastic training 是否带来鲁棒性。

3. dif_delta_x_candidate:
   只有前两步有收益时，再考虑 K candidate soft selection。
```

一句话：

```text
SeqAvatar 中不是“从分布里选一个表面点”，而是:
  训练时从 Δx distribution 采样来学习不确定性；
  测试时使用最可能的 Δx，即 μ；
  rectifier 可用 σ 作为不确定性上下文修正 μ。
```

## 2026-07-03 DIF-Δx 实现与启动记录

用户要求新增两个消融：

```text
dif_delta_x_mean
  不采样，Δx = μ + R([h, μ, μ, σ])

dif_delta_x_sample
  train:  Δx = z + R([h, z, μ, σ]), z = μ + σε
  render: Δx = μ + R([h, μ, μ, σ])
```

实现范围：

```text
只改 non-rigid deformation 的 Δx head。
不改:
  STMS condition
  SeqPoseEncoder / SeqXYZEncoder
  Δs / Δr heads
  renderer
  loss
  CameraInfo / RGB / mask
  baseline / MSTI / AMC / TDP / Part-MoE 默认路径
```

新增参数：

```text
arguments/__init__.py
  use_dif_delta_x = False
  dif_delta_x_mode = mean / sample
  dif_delta_x_sigma_init = -7.0
  dif_delta_x_eps = 1e-6

use_dif_delta_x 与 use_part_moe 第一版互斥，
避免 Part-MoE expert head 和 DIF Δx head 混成未定义组合。
```

核心代码：

```text
nets/mlp_delta_non_rigid.py

baseline:
  h = self.mlp(features)
  d_xyz = self.gaussian_warp(h)

DIF-Δx:
  μ = self.gaussian_warp(h)
  σ = softplus(self.gaussian_warp_sigma(h)) + eps

  mean mode:
    z = μ

  sample mode:
    if training:
      z = μ + σ * randn_like(μ)
    else:
      z = μ

  d_xyz = z + zero_init_rectifier([h, z, μ, σ])
```

初始化：

```text
gaussian_warp 仍作为 μ head。
gaussian_warp_sigma:
  weight = 0
  bias = -7.0
  softplus(-7) ≈ 0.00091

rectifier 最后一层 zero-init。
因此 mean mode 初始接近 baseline Δx。
```

DNA 脚本：

```text
scripts/exps_dnarendering.sh
  dif_delta_x_mean
  dif_delta_x_sample

日志目录:
  /media/image/mxz/human/SeqAvatar/logs/dif
```

验证：

```text
bash -n scripts/exps_dnarendering.sh 通过
py_compile arguments / gaussian_model / mlp_delta_non_rigid 通过
dummy forward:
  baseline / dif_mean / dif_sample 输出 shape 均为:
    d_xyz      (1, 5, 3)
    d_rotation (1, 5, 4)
    d_scaling  (1, 5, 3)
```

### DNA 六序列启动

设置：

```text
RUN_TIME = 20260703_202436
iter = 25000
densify_until_iter = 1500
time_step_num = 3
motion_cond_time_step_num = 3
final_eval_only = 1

sequences:
  0044_11, 0051_09, 0206_04, 0813_05, 0007_04, 0019_10
```

启动：

```text
GPU 1:
  dif_delta_x_mean
  log = logs/dif/20260703_202436_DNA-Rendering_dif_delta_x_mean.log

GPU 2:
  dif_delta_x_sample
  log = logs/dif/20260703_202436_DNA-Rendering_dif_delta_x_sample.log
```

启动后确认：

```text
两个模式均打印:
  USE_DIF_DELTA_X = 1
  DIF_DELTA_X_MODE = mean / sample
  DIF_DELTA_X_SIGMA_INIT = -7.0
  DIF_DELTA_X_EPS = 1e-6

两个模式首个序列 0044_11 均进入 Training。
```

### DNA 六序列 tmux 重启

原因：

```text
普通 exec 会被会话中断影响，改为 tmux detached 方式重跑。
第一次重启 20260703_222605 因脚本当前默认 SEQUENCES 只有 0044_11，已立即停止，不作为正式结果。
```

正式重启：

```text
RUN_TIME = 20260703_222657
SEQUENCES_OVERRIDE = 0044_11 0051_09 0206_04 0813_05 0007_04 0019_10

GPU 1:
  tmux = seqavatar_dif_mean_20260703_222657
  mode = dif_delta_x_mean
  log  = logs/dif/20260703_222657_DNA-Rendering_dif_delta_x_mean.log

GPU 2:
  tmux = seqavatar_dif_sample_20260703_222657
  mode = dif_delta_x_sample
  log  = logs/dif/20260703_222657_DNA-Rendering_dif_delta_x_sample.log
```

启动后确认：

```text
两个模式均使用 DNA 六序列。
两个模式首个序列 0044_11 均进入 Training。
train 参数确认：
  use_dif_delta_x = True
  dif_delta_x_mode = mean / sample
  motion_cond_time_step_num = 3
  densify_until_iter = 1500
  iterations = 25000
```

### DIF-Delta-X 三个框图

已生成三张方法框图：

```text
note/figures/dif_delta_x_mean_pipeline.svg
note/figures/dif_delta_x_sample_pipeline.svg
note/figures/dif_delta_x_rectifier_pipeline.svg
```

三张图的语义区分：

```text
dif_delta_x_mean:
  预测 μ / σ。
  不采样，选择 z = μ。
  σ 作为 uncertainty descriptor 输入 rectifier。
  Δx = μ + R([h, μ, μ, σ])。

dif_delta_x_sample:
  预测 μ / σ。
  训练使用 reparameterized sample:
    z = μ + σ ⊙ ε
  测试 / render 使用 z = μ。
  Δx = z，不额外使用 rectifier。

dif_delta_x_rectifier:
  预测 μ / σ。
  训练 sample，测试 / render 使用 μ。
  再用 D-IF 风格 rectifier 修正：
    Δx = z + R([h, z, μ, σ])
  这是最接近 D-IF 原始思想的版本。
```

## 2026-07-04 重新定义 DIF 消融：Delta-X Gaussian Peak

用户澄清真正想要的 DIF 实验：

```text
原始 SeqAvatar:
  h_i -> Δx_i

新 DIF:
  h_i -> (μ_i, σ_i)
  用高斯分布建模每个 Gaussian 点自己的 Δx 分布:
    Δx_i ~ N(μ_i, σ_i)
  取分布最高点作为最终运动偏移:
    Δx_i = argmax_x N(x | μ_i, σ_i)
```

关键结论：

```text
单峰 Gaussian 的最高点就是均值 μ_i。
因此当前阶段真正 forward 应为:
  Δx_i = μ_i

暂时不考虑方差作用:
  不 sampling
  不 rectifier
  不让 σ 进入 render / loss
  不做 σ regularization
```

因此旧实验定义废弃：

```text
dif_delta_x_mean:
  μ + rectifier([h, μ, μ, σ])

dif_delta_x_sample:
  train sample + rectifier
```

旧代码路径已删除 / 替换为单一新开关：

```text
use_dif
dif_sigma_init = -7.0
dif_eps = 1e-6
```

新代码语义：

```python
mu = gaussian_warp(h)
sigma = softplus(gaussian_warp_sigma(h)) + eps
d_xyz = mu
```

注意：

```text
由于 σ 暂时不参与 loss / render，σ head 目前不会学习到有效不确定性。
这个消融只能验证“把 Δx head 参数化为 Gaussian 并取 peak=μ”是否会影响结果。
如果要真正证明每个点的不确定性有意义，下一步必须引入:
  sampling
  uncertainty regularization
  error / boundary / motion-aware σ supervision
  或 rectifier 使用 σ
```

验证：

```text
bash -n scripts/exps_dnarendering.sh 通过
py_compile arguments / gaussian_model / mlp_delta_non_rigid 通过
dummy forward:
  baseline 输出 shape = [(1,4,3), (1,4,4), (1,4,3)], 无 sigma head
  dif      输出 shape = [(1,4,3), (1,4,4), (1,4,3)], 有 sigma head
```

### 新 DIF DNA 启动

设置：

```text
RUN_TIME = 20260704_001710
experiment_name = dif
iter = 25000
densify_until_iter = 1500
time_step_num = 3
motion_cond_time_step_num = 3
final_eval_only = 1
```

并行启动：

```text
GPU 1:
  tmux = seqavatar_dif_gpu1_20260704_001710
  sequences = 0044_11, 0051_09, 0206_04
  log = logs/dif/20260704_001710_DNA-Rendering_dif_gpu1.log

GPU 2:
  tmux = seqavatar_dif_gpu2_20260704_001710
  sequences = 0813_05, 0007_04, 0019_10
  log = logs/dif/20260704_001710_DNA-Rendering_dif_gpu2.log
```

启动后确认：

```text
两个主日志均打印:
  Mode = dif
  Experiment = dif
  USE_DIF = 1
  DIF_SIGMA_INIT = -7.0
  DIF_EPS = 1e-6

命令行参数确认:
  --use_dif --dif_sigma_init -7.0 --dif_eps 1e-6

两个首个序列均进入 Training 进度条。
```

### 新 DIF DNA 结果

运行状态：

```text
RUN_TIME = 20260704_001710
GPU1 log = logs/dif/20260704_001710_DNA-Rendering_dif_gpu1.log
GPU2 log = logs/dif/20260704_001710_DNA-Rendering_dif_gpu2.log

0206_04 首次在 GPU1 训练到约 4100 iter 时出现一次 CUDA/cuBLAS 执行失败：
  CUBLAS_STATUS_EXECUTION_FAILED

随后单独重跑 0206_04 成功：
  RUN_TIME = 20260704_0206_retry
  log = logs/dif/20260704_0206_retry_DNA-Rendering_dif_gpu1_retry.log
```

最终使用 render 阶段不带 `L1` 的 `Evaluating novelview #120` 指标。

与 DNA densify=1500 baseline 对比：

| Sequence | Base PSNR | DIF PSNR | dPSNR | Base SSIM | DIF SSIM | dSSIM | Base LPIPS*1000 | DIF LPIPS*1000 | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0044_11 | 32.9741 | 32.9414 | -0.0327 | 0.977914 | 0.977834 | -0.000080 | 21.3979 | 21.5731 | 0.1752 |
| 0051_09 | 28.5780 | 28.5824 | 0.0044 | 0.970733 | 0.971281 | 0.000548 | 31.5899 | 31.4612 | -0.1287 |
| 0206_04 | 31.3778 | 31.3604 | -0.0174 | 0.969745 | 0.969381 | -0.000364 | 34.1104 | 34.6370 | 0.5266 |
| 0813_05 | 36.0828 | 35.9868 | -0.0960 | 0.986881 | 0.986573 | -0.000308 | 18.4874 | 19.0358 | 0.5484 |
| 0007_04 | 29.5333 | 29.4813 | -0.0520 | 0.958320 | 0.958168 | -0.000152 | 45.4151 | 45.6882 | 0.2731 |
| 0019_10 | 35.2200 | 35.3034 | 0.0834 | 0.980679 | 0.980988 | 0.000309 | 21.2880 | 21.2776 | -0.0104 |

平均：

```text
dPSNR        = -0.0184
dSSIM        = -0.000008
dLPIPS*1000  = +0.2307
```

判断：

```text
当前 peak-only DIF 不建议继续作为第二创新点主线。

原因：
1. Δx = argmax Gaussian = μ，实际输出仍等价于 deterministic mean head。
2. σ 暂时不进入 loss / render / rectifier，也没有 regularization，因此不会学习到有意义 uncertainty。
3. 平均 PSNR / SSIM / LPIPS 均没有稳定提升，且 4/6 序列 LPIPS 变差。
4. 0206_04 首次出现一次 CUDA/cuBLAS 失败，虽然 retry 成功，但当前实现没有带来足够收益来抵消额外复杂度。

如果后续还想沿 D-IF 思路继续，必须让 σ 真正参与优化：
  sampling + 测试用 μ
  σ regularization
  error / boundary / motion-aware σ supervision
  或使用 σ 的 rectifier / uncertainty-aware residual

但“只输出 Gaussian 参数并取最高点 μ”的版本到此停止。
```

## 2026-07-04 DIF 方差引入方案

关键判断：

```text
如果仍使用单峰 Gaussian 并取最高点:
  Δx = argmax_x N(x | μ, σ) = μ

则 σ 不影响 Δx，也不影响 render / loss。
因此 σ 不会被有效优化。
```

要让方差有意义，必须至少满足一个条件：

```text
1. σ 参与 Δx forward:
   Δx = μ + σ * something

2. σ 参与 loss:
   例如 uncertainty NLL / regularization / pseudo supervision

3. 使用 mixture Gaussian:
   多个 μ_k / σ_k / π_k 时，mode selection 才可能受 σ 影响。
```

最小可做版本：

```text
dif_sigma_sample:
  train:
    ε ~ N(0, I)
    Δx = μ + σ * ε
  test/render:
    Δx = μ

需要:
  sigma clamp
  sigma warmup
  sigma regularization
  记录 sigma mean / max / active ratio
```

更推荐版本：

```text
dif_sigma_rectifier:
  z = μ + σ * ε        # train
  z = μ                # test/render
  Δx = z + R([h, z, μ, logσ])

或更稳定的 deterministic 版本:
  Δx = μ + σ * tanh(R([h, μ, logσ]))

优点:
  σ 直接控制 residual 幅度；
  R 可以 zero-init，初始接近 baseline；
  σ 不再是无效旁路。
```

更强但复杂版本：

```text
dif_mixture_mode:
  输出 K 个分布:
    {π_k, μ_k, σ_k}_{k=1..K}
  选择峰值最高的 component:
    k* = argmax_k π_k / prod(σ_k)
    Δx = μ_{k*}

这样 mode selection 会受 σ 影响。
但 argmax 不平滑，训练更复杂，第一阶段不建议直接做。
```

当前建议：

```text
下一步若继续 DIF，不做 peak-only。
优先做 dif_sigma_rectifier，而不是直接 sample-only。
```

### dif_sigma_rectifier 前三序列实验

用户要求先在 DNA 前三个序列测试：

```text
sequences = 0044_11, 0051_09, 0206_04
experiment_name = dif_sigma_rectifier
```

实现：

```text
mu = gaussian_warp(h)
sigma = softplus(gaussian_warp_sigma(h)) + eps
r = rectifier([h, mu, log(sigma)])
d_xyz = mu + sigma * tanh(r)
```

设计细节：

```text
dif_mode = sigma_rectifier
sigma 是每个 Gaussian 点一个标量，broadcast 到 xyz 三维 residual。
rectifier 最后一层 zero-init，初始 d_xyz = mu。
dif_sigma_init = -4.0，初始 sigma mean 约 0.018。
```

验证：

```text
bash -n scripts/exps_dnarendering.sh 通过
py_compile arguments / gaussian_model / mlp_delta_non_rigid 通过
dummy forward:
  output shapes = [(5,3), (5,4), (5,3)]
  max_initial_delta_from_mu = 0.0
  sigma_mean = 0.01815
```

启动：

```text
RUN_TIME = 20260704_042454
tmux = seqavatar_dif_sigma_rectifier_gpu1_20260704_042454
GPU = 1
log = logs/dif/20260704_042454_DNA-Rendering_dif_sigma_rectifier_gpu1.log
```

用户随后要求 GPU1/GPU2 同时跑前三个序列，调整为：

```text
GPU1:
  sequence = 0044_11
  tmux = seqavatar_dif_sigma_rectifier_gpu1_20260704_042454
  log = logs/dif/20260704_042454_DNA-Rendering_dif_sigma_rectifier_gpu1.log

GPU2:
  sequences = 0051_09, 0206_04
  tmux = seqavatar_dif_sigma_rectifier_gpu2_20260704_042454
  log = logs/dif/20260704_042454_DNA-Rendering_dif_sigma_rectifier_gpu2.log
```

注意：

```text
GPU1 原脚本最初包含 0044_11 / 0051_09 / 0206_04。
已增加外部监控：
  当 GPU1 日志出现 Finished sequence: 0044_11 后，
  自动停止 GPU1 tmux，避免重复跑 0051_09 / 0206_04。
```

运行中调整：

```text
GPU1 的 0044_11 已完成后被自动停止。
GPU2 的 0051_09 已完成。
GPU2 跑 0206_04 到约 4680 iter 时出现:
  CUBLAS_STATUS_INTERNAL_ERROR

0206_04 改为单独在 GPU1 retry:
  RUN_TIME = 20260704_0206_dsr_retry
  tmux = seqavatar_dif_sigma_rectifier_0206_gpu1_20260704_0206_dsr_retry
  log = logs/dif/20260704_0206_dsr_retry_DNA-Rendering_dif_sigma_rectifier_gpu1_retry.log
```

结果：

```text
最终使用 render 阶段不带 L1 的 Evaluating novelview #120 指标。
```

| Sequence | Base PSNR | Rect PSNR | dPSNR | Base SSIM | Rect SSIM | dSSIM | Base LPIPS*1000 | Rect LPIPS*1000 | dLPIPS*1000 | dPSNR vs peak | dLPIPS*1000 vs peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0044_11 | 32.9741 | 32.9575 | -0.0166 | 0.977914 | 0.977818 | -0.000096 | 21.3979 | 21.5402 | 0.1423 | 0.0161 | -0.0330 |
| 0051_09 | 28.5780 | 28.6249 | 0.0469 | 0.970733 | 0.971135 | 0.000402 | 31.5899 | 31.2958 | -0.2941 | 0.0425 | -0.1654 |
| 0206_04 | 31.3778 | 31.3292 | -0.0486 | 0.969745 | 0.969387 | -0.000358 | 34.1104 | 34.5895 | 0.4791 | -0.0312 | -0.0476 |

平均：

```text
vs baseline:
  dPSNR        = -0.0061
  dSSIM        = -0.000017
  dLPIPS*1000  = +0.1091

vs peak-only DIF:
  dPSNR        = +0.0091
  dSSIM        = -0.000052
  dLPIPS*1000  = -0.0820
```

阶段判断：

```text
sigma-rectifier 比 peak-only DIF 略好，说明让 σ 控制 residual 至少不是完全无效。
但前三序列平均仍没有超过 baseline，尤其 0206_04 退化明显。

当前不建议直接扩大到完整 DNA / I3D / ZJU。
如果继续，需要先加诊断:
  sigma mean / max / min
  residual norm
  residual / mu norm
  sigma 是否在 high-motion / boundary 区域更大

如果诊断显示 residual 很小:
  sigma_init=-3 或 residual scale 放大。

如果 residual 过大伤 PSNR:
  加 sigma clamp / residual clamp / warmup。

如果 sigma 没有空间结构:
  需要 error-aware 或 boundary-aware sigma supervision。
```

## 2026-07-04 DIF peak-only 是否可以改 loss

当前问题：

```text
DIF peak-only:
  Δx = argmax N(Δx | μ, σ) = μ

如果 σ 不参与 forward / loss:
  最终几何仍是 deterministic Δx = μ
  σ 不会学到有意义 uncertainty
```

因此如果“暂时不考虑方差，让 δ=均值”，可以改 loss，但实验含义要写清楚：

```text
1. 不输出 / 不使用 σ:
   所谓 distribution 退化为固定方差 Gaussian。
   约束分布本质等价于约束 μ，也就是约束 Δx peak。

2. 输出 σ 但 forward 仍用 μ:
   可以通过 KL / sigma regularization 约束分布形状。
   但如果 σ 只被 regularizer 约束，不和 image residual / geometry residual 绑定，
   σ 学到的多半是人为先验，不是真正 uncertainty。
```

可做的 loss 版本：

```text
dif_mean_peak_reg:
  Δx = μ
  L = L_render + λ_mu * robust(||μ||)
  目的：防止 Δx 过大，约束 Gaussian 分布峰值不要乱漂。
  缺点：这不是严格 uncertainty modeling，容易和已有 AIAP 几何约束重复。

dif_mean_kl_prior:
  Δx = μ
  q(Δx) = N(μ, σ)
  p(Δx) = N(0, σ0)
  L = L_render + λ_kl * KL(q || p)
  目的：同时约束 μ 和 σ，防止分布无限扩散或塌缩。
  缺点：如果没有 delta_x GT / residual supervision，σ 仍主要学习先验。

dif_mean_surface_reg:
  Δx = μ
  L = L_render + λ_surface * point-to-plane / nearest-SMPL-surface regularization
  目的：让分布峰值 μ 后的 Gaussian 位置不要远离人体表面。
  缺点：衣服、头发、松散区域可能被过度拉回 SMPL，权重要很小。

dif_mean_render_uncertainty:
  Δx = μ
  输出 σ，并让 σ 解释 image / geometry residual
  例如把 residual 大的位置允许更大 σ，小 residual 约束更小 σ。
  这是更合理的 distribution loss，但实现需要 per-Gaussian residual / visibility attribution，
  比 peak_reg / KL_prior 复杂。
```

代码落点建议：

```text
train.py:
  当前已有:
    image L1 / mask / SSIM / LPIPS
    full_aiap_loss(scene.gaussians.get_xyz, render_pkg["deformed_means3D"], ...)

  如果只约束 δ=μ:
    delta = render_pkg["deformed_means3D"] - scene.gaussians.get_xyz[None]
    loss += λ_mu * robust(delta)

nets/mlp_delta_non_rigid.py:
  如果要 KL / sigma loss:
    forward_delta_x 中保存 last_dif_mu / last_dif_sigma
    train.py 读取 gaussians.non_rigid_deformer.last_dif_mu / last_dif_sigma 加 loss
```

当前建议：

```text
可以做一个小消融 dif_mean_peak_reg 或 dif_mean_kl_prior。
但不要宣称它已经建模有效 uncertainty。
如果最终仍是 δ=μ，最干净的说法是:
  distribution-peak regularization / probabilistic displacement prior
而不是 D-IF 式真正从不确定分布中选择点。
```

## 2026-07-04 DIF 方差继续推进的具体做法

核心结论：

```text
单峰 Gaussian 如果仍取最高点:
  δ = argmax N(δ | μ, σ) = μ

则 σ 不影响最终 Δx。
因此真正“引入方差”必须让 σ 至少进入 forward 或进入有效 loss。
```

当前最推荐继续版本：

```text
dif_sigma_rectifier_v2
```

形式：

```python
mu = gaussian_warp(h)
raw_sigma = gaussian_warp_sigma(h)
sigma = softplus(raw_sigma) + eps
sigma = clamp(sigma, sigma_min, sigma_max)

r = rectifier([h, mu, log(sigma)])
delta_x = mu + beta * sigma * tanh(r)
```

原因：

```text
1. σ 直接控制 residual 最大幅度，因此不是无效旁路。
2. rectifier zero-init 后，初始 delta_x = mu，不会一开始破坏 baseline。
3. 不引入随机采样噪声，比 sample-only 稳定。
4. σ 可以解释为每个 Gaussian 点允许的非刚性修正空间。
```

建议配套：

```text
sigma shape:
  第一版用 [N,1] scalar sigma，broadcast 到 xyz。
  比 [N,3] 更稳，参数更少，也更容易解释。

sigma init:
  dif_sigma_init = -4.0 或 -3.5
  初始 sigma 约 0.018 到 0.030。

residual scale:
  beta = 0.5 / 1.0 可做小网格。

clamp:
  sigma_min = 1e-4
  sigma_max = 0.05 或 0.10

warmup:
  前 1000-3000 iter 让 beta 从 0 线性升到目标值。
```

loss 建议：

```text
不要只对 sigma 加强 regularization。
如果 sigma 只被 regularizer 约束，它学到的是人为先验，不是真正 uncertainty。

可以加很轻的 sigma prior，防止无限放大:
  L_sigma_prior = mean((log(sigma) - log(sigma0))^2)

也可以加 residual ratio 约束:
  residual = beta * sigma * tanh(r)
  L_residual = mean(|residual|) 或 smooth_l1(residual)

权重要小，避免把非刚性形变压没。
```

必须加诊断：

```text
每 1000 iter 记录:
  sigma_mean / sigma_min / sigma_max
  residual_norm
  mu_norm
  residual_norm / mu_norm
  active_ratio = mean(|residual| > threshold)

如果 sigma_mean 一直接近初始值且 residual_norm 很小:
  说明 σ 没有被用起来。

如果 sigma 快速顶到 sigma_max 且 PSNR 掉:
  说明 residual 过强，需要减小 beta 或 sigma_max。
```

不推荐优先做：

```text
dif_sigma_sample:
  train: delta_x = mu + sigma * eps
  test: delta_x = mu

原因:
  会引入随机噪声；
  单样本 render 梯度方差大；
  对当前 SeqAvatar 这种逐点高斯渲染训练不够稳。
```

更强但后做：

```text
dif_mixture:
  输出 K 个 Gaussian component:
    {pi_k, mu_k, sigma_k}
  用 log peak = log pi_k - sum(log sigma_k) 选择 component。

这个版本 σ 会影响 mode selection，
但训练和解释都更复杂，应在 sigma_rectifier_v2 有正信号后再做。
```

### dif_sigma_rectifier_v2 四序列实验

实现：

```text
mode = dif_sigma_rectifier_v2
experiment_name = dif_sigma_rectifier_v2

mu = gaussian_warp(h)
sigma = softplus(gaussian_warp_sigma(h)) + eps
sigma = clamp(sigma, sigma_min, sigma_max)
r = rectifier([h, mu, log(sigma)])
delta_x = mu + beta * sigma * tanh(r)
```

新增隔离参数：

```text
dif_sigma_min = 1e-4
dif_sigma_max = 0.05
dif_residual_beta = 1.0
dif_residual_warmup = 3000
dif_sigma_prior = 0.02
dif_sigma_prior_w = 1e-4
dif_debug_interval = 1000
```

代码落点：

```text
arguments/__init__.py:
  新增 v2 参数和 dif_mode = sigma_rectifier_v2。

scene/gaussian_model.py:
  透传 v2 参数到 NonrigidDeformer。

nets/mlp_delta_non_rigid.py:
  sigma clamp。
  beta warmup。
  delta_x = mu + beta * sigma * tanh(rectifier)。
  pop_dif_stats 输出 sigma / residual 诊断。

train.py:
  每轮训练前 set_dif_iteration(iteration)。
  可选 dif_sigma_prior_loss。
  每 dif_debug_interval 打印 DIF Stats。

scripts/exps_dnarendering.sh:
  新增 dif_sigma_rectifier_v2 模式。
  日志仍保存到 logs/dif。
```

验证：

```text
bash -n scripts/exps_dnarendering.sh 通过。
py_compile arguments / gaussian_model / mlp_delta_non_rigid / train 通过。
dummy forward:
  output shapes = [(1,5,3), (1,5,4), (1,5,3)]
  初始 residual_norm = 0。
  sigma_mean ≈ 0.029751。
```

运行：

```text
RUN_TIME = 20260704_170652

GPU1:
  sequences = 0044_11, 0051_09
  log = logs/dif/20260704_170652_DNA-Rendering_dif_sigma_rectifier_v2_gpu1.log

GPU2:
  sequences = 0206_04, 0813_05
  log = logs/dif/20260704_170652_DNA-Rendering_dif_sigma_rectifier_v2_gpu2.log
```

四序列 render 指标（novelview, iteration=25000，对比 DNA densify=1500 baseline）：

| Sequence | PSNR | dPSNR | SSIM | dSSIM | LPIPS*1000 | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| 0044_11 | 32.9507 | -0.0234 | 0.977865 | -0.000049 | 21.5199 | +0.1220 |
| 0051_09 | 28.6206 | +0.0426 | 0.971323 | +0.000590 | 31.0691 | -0.5208 |
| 0206_04 | 31.3884 | +0.0106 | 0.969837 | +0.000092 | 34.3448 | +0.2344 |
| 0813_05 | 36.0767 | -0.0061 | 0.986871 | -0.000010 | 18.5154 | +0.0280 |

平均：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| baseline | 32.2532 | 0.976318 | 26.3964 |
| dif_peak | 32.2177 | 0.976267 | 26.6768 |
| dif_sigma_rectifier_v2 | 32.2591 | 0.976474 | 26.3623 |
| v2 - baseline | +0.0059 | +0.000155 | -0.0341 |
| v2 - peak | +0.0414 | +0.000206 | -0.3145 |

和前三序列旧 sigma_rectifier 对比：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| baseline | 30.9766 | 0.972797 | 29.0327 |
| dif_sigma_rectifier | 30.9705 | 0.972780 | 29.1418 |
| dif_sigma_rectifier_v2 | 30.9866 | 0.973008 | 28.9779 |
| v2 - old rectifier | +0.0161 | +0.000228 | -0.1639 |

诊断：

```text
DIF Stats 有正常输出。
sigma_mean 大多在 0.019-0.021。
sigma_max 大多在 0.021-0.022，未顶到 sigma_max=0.05。
residual_norm / mu_norm 大约 0.03-0.06。
active_ratio = 1.0。
说明 σ 已经参与 forward，但 residual 幅度较温和。
```

阶段判断：

```text
dif_sigma_rectifier_v2 比 peak-only DIF 和旧 sigma_rectifier 更好，
说明“让 σ 控制 residual”比只输出 μ/σ 后取 μ 更合理。

但相对 baseline 的平均提升仍很小：
  dPSNR = +0.0059
  dSSIM = +0.000155
  dLPIPS*1000 = -0.0341

不能直接作为强第二创新点主线。
如果继续 DIF，建议只作为候选分支，并优先看 high-motion / boundary subset。
如果 high-motion 或 boundary 没有明显收益，则停止 DIF 主线。
```

## 2026-07-04 Token 消融实验设计与实现记录

目标：

```text
根据 D-IF / token 化形变思想，新增 Part-aware Motion Token SeqAvatar 消融。
本轮只在 DNA-Rendering 上先跑 4 个序列，不影响 baseline / Part-MoE / MSTI / AMC / TDP / DIF 默认路径。
日志目录：
  logs/token
```

实现边界：

```text
主实现放在 NonrigidDeformer。
不改 renderer/rasterizer 主体。
不改已有 Part-MoE 专家流程。
不改已有 TDP / DIF 开关行为。
所有 token 开关默认关闭。
token 与 MSTI / AMC / TDP / DIF / Part-MoE 互斥。
```

新增消融模式：

```text
token_fix_stms:
  只修 SeqAvatar 原 STMS 中 seq_xyz_conds 多尺度归一化问题。
  原始 baseline 默认仍保持旧逻辑。

token_acc:
  token_fix_stms + 额外 seq_acc_conds。
  acceleration = 当前尺度速度 - 上一段同尺度速度。

token_part:
  token_fix_stms + SMPL-LBS part embedding。
  不依赖 Part-MoE 标签文件。

token_codebook:
  token_fix_stms + motion-deformation soft token codebook。
  token 输入来自 seq_xyz_feats。

token_full:
  token_fix_stms + acc + part + codebook。
```

代码落点：

```text
arguments/__init__.py:
  use_motion_token
  motion_token_mode
  motion_token_num
  motion_token_dim
  motion_token_part_dim
  motion_token_acc_dim

scene/dataset_readers.py:
  仅 token_fix_stms 模式按每个 time_step 自己归一化 seq_xyz_conds。
  仅 token_acc / token_full 生成 seq_acc_conds。

gaussian_renderer/__init__.py:
  仅 token_acc / token_full gather seq_acc_conds。
  仅 token_part / token_full 用 KNN SMPL 顶点 LBS weight 得到 part_label。

scene/gaussian_model.py:
  透传 token 参数到 NonrigidDeformer。

nets/mlp_delta_non_rigid.py:
  MotionTokenEncoder soft codebook。
  SeqAccEncoder。
  SMPL part embedding。

scripts/exps_dnarendering.sh:
  token_fix_stms / token_acc / token_part / token_codebook / token_full。
  token 日志保存到 logs/token。
```

验证：

```text
bash -n scripts/exps_dnarendering.sh 通过。
py_compile arguments / scene / renderer / mlp_delta_non_rigid 通过。
dummy forward:
  token_fix_stms / token_acc / token_part / token_codebook / token_full
  output shapes 都为 [(B,N,3), (B,N,4), (B,N,3)]。
```

运行：

```text
RUN_TIME = 20260704_062146
GPU1:
  0044_11, 0051_09
  logs/token/*_gpu1.log
GPU2:
  0206_04, 0813_05
  logs/token/*_gpu2.log
```

四序列 render 指标均值（novelview, iteration=25000）：

| Mode | PSNR | dPSNR | SSIM | dSSIM | LPIPS*1000 | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| token_fix_stms | 32.2289 | -0.0242 | 0.976370 | +0.000052 | 26.4343 | +0.0379 |
| token_acc | 32.2658 | +0.0127 | 0.976509 | +0.000191 | 26.2656 | -0.1308 |
| token_part | 32.2423 | -0.0109 | 0.976324 | +0.000006 | 26.4279 | +0.0315 |
| token_codebook | 32.2337 | -0.0195 | 0.976253 | -0.000065 | 26.5647 | +0.1683 |
| token_full | 32.2415 | -0.0116 | 0.976283 | -0.000035 | 26.3553 | -0.0411 |

阶段结论：

```text
token_acc 是本轮最好的 token 系消融：
  PSNR / SSIM / LPIPS 三项平均都优于四序列 baseline。

token_full 不如 token_acc：
  说明 part embedding + codebook 与 acc 叠加后没有继续带来收益，反而有干扰。

token_codebook 单独变差：
  当前 soft codebook token 没有证明有效。

token_part 单独收益不稳定：
  0051_09 提升明显，但 0206_04 / 0813_05 下降。

建议：
  继续 token_acc 方向；
  暂时不要把 token_codebook 作为主创新；
  如果继续 codebook，需要先做 token usage entropy / top-k usage / part-conditioned token 诊断，而不是直接大跑 full。
```

## 2026-07-04 token_acc DNA 六序列补跑结果

补跑：

```text
RUN_TIME = 20260704_144804
GPU1:
  0007_04
  logs/token/20260704_144804_DNA-Rendering_token_acc_gpu1_extra.log
GPU2:
  0019_10
  logs/token/20260704_144804_DNA-Rendering_token_acc_gpu2_extra.log
```

六序列 token_acc render 指标（novelview, iteration=25000，对比 DNA baseline）：

| Sequence | PSNR | dPSNR | SSIM | dSSIM | LPIPS*1000 | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| 0044_11 | 32.9857 | +0.0116 | 0.977889 | -0.000025 | 21.4960 | +0.0981 |
| 0051_09 | 28.6115 | +0.0335 | 0.971266 | +0.000533 | 31.1322 | -0.4577 |
| 0206_04 | 31.3695 | -0.0083 | 0.969906 | +0.000161 | 33.9640 | -0.1464 |
| 0813_05 | 36.0967 | +0.0139 | 0.986977 | +0.000096 | 18.4700 | -0.0174 |
| 0007_04 | 29.5245 | -0.0088 | 0.958187 | -0.000133 | 45.7227 | +0.3076 |
| 0019_10 | 35.2177 | -0.0023 | 0.980864 | +0.000185 | 21.2928 | +0.0048 |

六序列均值：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| baseline | 32.2943 | 0.974045 | 28.7148 |
| token_acc | 32.3009 | 0.974181 | 28.6796 |
| delta | +0.0066 | +0.000136 | -0.0352 |

结论：

```text
token_acc 六序列平均仍是弱正向：
  PSNR / SSIM / LPIPS 三项均略优于 baseline。

但提升幅度很小：
  6 个序列中 PSNR 只有 3 个提升；
  LPIPS 只有 3 个明显或轻微提升；
  0007_04 上 PSNR / SSIM / LPIPS 都变差。

当前不能把 token_acc 作为强主创新直接推进。
如果继续，需要优先看 high-motion subset / boundary subset 是否有更清晰收益。
```

## 2026-07-04 Token usage 诊断结论

重要更正：

```text
当前 token_acc 并没有真正的 softmax codebook token。

token_acc 实现是:
  token_fix_stms + SeqAccEncoder(seq_acc_conds)
  然后把 acc feature concat 到 NonrigidDeformer MLP。

它没有调用 MotionTokenEncoder。
因此 token_acc 下不存在:
  token softmax
  token_entropy
  token_max_prob
  token_usage
```

真正有 softmax token 的模式只有：

```text
token_codebook:
  seq_xyz_feats -> MotionTokenEncoder -> softmax(codebook)

token_full:
  seq_xyz_feats + acc_feats + part_feats -> MotionTokenEncoder -> softmax(codebook)
```

已新增隔离诊断开关：

```text
arguments/__init__.py:
  motion_token_debug_stats
  motion_token_debug_interval

nets/mlp_delta_non_rigid.py:
  MotionTokenEncoder 缓存 last_weights。
  pop_motion_token_stats 输出:
    entropy_mean
    entropy_norm
    entropy_min / entropy_max
    max_prob_mean
    max_prob_min / max_prob_max
    usage_nonzero
    usage_max
    usage_entropy
    usage_entropy_norm
    codebook_active
  同时输出完整 top-1 usage 向量。

train.py:
  打印:
    [MotionToken Stats][ITER ...]
    [MotionToken Usage][ITER ...]

scripts/exps_dnarendering.sh:
  MOTION_TOKEN_DEBUG_STATS=1
  MOTION_TOKEN_DEBUG_INTERVAL=1000
```

验证：

```text
token_acc dummy forward:
  codebook_active = 0
  usage = []

token_codebook dummy forward:
  codebook_active = 1
  entropy_norm ≈ 0.999
  max_prob_mean ≈ 0.142 for 8 tokens
  usage_nonzero = 1

这个 dummy 例子说明：
  softmax 权重几乎均匀，但 top-1 全落到一个 token；
  属于“权重平均 + top1 collapse”的坏情况。
```

判断：

```text
之前 token_acc 的小幅提升不能解释为“离散 motion token 被有效使用”。
更准确解释是:
  acceleration condition 作为额外连续特征带来弱增益；
  或者只是增加了一点 encoder / MLP 容量。

如果要证明 token/codebook 方向有效，必须重新检查 token_codebook 或 token_full:
  MOTION_TOKEN_DEBUG_STATS=1 bash scripts/exps_dnarendering.sh token_codebook
  MOTION_TOKEN_DEBUG_STATS=1 bash scripts/exps_dnarendering.sh token_full

重点看：
  entropy_norm 接近 1 且 max_prob 低 -> token 没有形成明确模式；
  usage_nonzero 很小 / usage_max 很高 -> codebook collapse；
  两者任一出现，都不能把平均指标小涨解释成真正离散运动模式。
```

## 2026-07-04 token_codebook debug 真实两序列诊断

目的：

```text
先判断 soft codebook token 有没有被用起来。
不先追完整 PSNR/SSIM/LPIPS。
```

运行：

```text
0051_09:
  GPU1
  logs/token/20260704_200324_DNA-Rendering_token_codebook_gpu1_debug.log

0007_04:
  GPU2
  logs/token/20260704_200324_DNA-Rendering_token_codebook_gpu2_debug.log
```

为保证 token stats 实时落盘，`train.py` 的 MotionToken debug print 增加了 `flush=True`。
这只影响日志刷新，不影响训练结构和数值计算。

结果：

| Sequence | Iter | entropy_norm | max_prob_mean | usage_nonzero | usage_max | usage_entropy_norm |
|---|---:|---:|---:|---:|---:|---:|
| 0051_09 | 1000 | 0.999299 | 0.035909 | 1 | 1.000000 | 0.000000 |
| 0051_09 | 2000 | 0.999403 | 0.035444 | 3 | 0.616482 | 0.219617 |
| 0051_09 | 3000 | 0.998992 | 0.035917 | 6 | 0.456069 | 0.319081 |
| 0051_09 | 4000 | 0.999541 | 0.034399 | 1 | 1.000000 | 0.000000 |
| 0051_09 | 5000 | 0.999513 | 0.034705 | 2 | 0.859564 | 0.117076 |
| 0007_04 | 1000 | 0.998816 | 0.037452 | 2 | 0.999930 | 0.000214 |
| 0007_04 | 2000 | 0.998896 | 0.035823 | 4 | 0.791115 | 0.188783 |
| 0007_04 | 3000 | 0.999159 | 0.036542 | 2 | 0.973738 | 0.035057 |
| 0007_04 | 4000 | 0.999399 | 0.034604 | 5 | 0.697306 | 0.263857 |
| 0007_04 | 5000 | 0.999329 | 0.034749 | 4 | 0.808900 | 0.186108 |
| 0007_04 | 6000 | 0.999414 | 0.034536 | 5 | 0.822561 | 0.163670 |

解释：

```text
32 tokens 均匀 softmax 的 max_prob 约为 1 / 32 = 0.03125。
当前 max_prob_mean 只有 0.034-0.037，entropy_norm 长期约 0.999。

这说明 softmax 权重几乎是均匀分布，没有形成明确 token 选择。
usage_nonzero 偶尔变多，但主要来自接近均匀权重下的 argmax 微小差异。
真实 top-1 usage 仍高度集中到少数 token。
```

决策：

```text
token_codebook usage 不健康。
本轮没有跑满 25000，也没有做最终指标评估。
原因是当前目标是判定 codebook 是否值得继续；诊断已经满足停止条件。

不建议继续把离散 token/codebook 作为主线。
如果继续 token，应该先改 token 机制或加 usage/entropy 约束；
否则平均指标的小涨更可能来自额外 MLP 容量，而不是离散运动模式。
```

## 2026-07-04 当前 DIF 消融超参数含义和调参依据

当前代码里 DIF 只改 `NonrigidDeformer.forward_delta_x`：

```text
mu = gaussian_warp(h)
sigma = softplus(gaussian_warp_sigma(h)) + eps
```

三种模式：

```text
peak:
  d_xyz = mu
  sigma 只被记录，不影响最终 delta_x。

sigma_rectifier:
  r = rectifier([h, mu, log(sigma)])
  d_xyz = mu + sigma * tanh(r)

sigma_rectifier_v2:
  sigma = clamp(sigma, sigma_min, sigma_max)
  beta  = linear warmup to dif_residual_beta
  d_xyz = mu + beta * sigma * tanh(r)
```

当前关键参数：

| 参数 | 当前常用值 | 含义 |
|---|---:|---|
| `dif_mode` | `peak` / `sigma_rectifier` / `sigma_rectifier_v2` | 决定 sigma 是否只记录、是否参与 residual、是否 clamp/warmup |
| `dif_sigma_init` | peak: -7.0, rectifier: -4.0, v2: -3.5 | 初始化 sigma head bias；越大 residual 允许空间越大 |
| `dif_eps` | 1e-6 | 防止 sigma/log(sigma) 数值为 0 |
| `dif_sigma_min` | 1e-4 | v2 下 sigma 下界 |
| `dif_sigma_max` | 0.05 | v2 下 sigma 上界，也是 residual 大小硬上限之一 |
| `dif_residual_beta` | 1.0 | v2 residual 总强度系数 |
| `dif_residual_warmup` | 3000 | v2 前 3000 iter 线性打开 residual |
| `dif_sigma_prior` | 0.02 | sigma prior 目标值 |
| `dif_sigma_prior_w` | 默认 0.0，v2 实验用过 1e-4 | 约束 log(sigma) 接近 log(prior) 的权重 |
| `dif_debug_interval` | 1000 | 每隔多少 iter 打印 sigma/residual 统计 |

sigma 初值近似：

```text
softplus(-7.0) ≈ 0.0009
softplus(-4.0) ≈ 0.018
softplus(-3.5) ≈ 0.030
```

调参优先级：

```text
1. 先固定使用 sigma_rectifier_v2。
2. 看 DIF Stats:
   sigma_mean
   sigma_max
   residual_mu_ratio
   active_ratio
3. residual 太弱:
   提高 dif_sigma_init，如 -4.0 -> -3.5；
   或提高 dif_sigma_max，如 0.05 -> 0.08。
4. residual 太强或指标掉:
   降低 dif_sigma_init，如 -3.5 -> -4.5；
   降低 dif_sigma_max；
   拉长 dif_residual_warmup，如 3000 -> 5000/8000。
5. sigma 不稳定或频繁顶到 sigma_max:
   加强 dif_sigma_prior_w，如 1e-4 -> 3e-4；
   或降低 sigma_max。
6. 如果 sigma_mean 被 prior 钉死但指标无收益:
   降低 dif_sigma_prior_w；
   否则 sigma 只是人为先验，不是有效 uncertainty。
```

当前 v2 诊断现象：

```text
sigma_mean 大多在 0.019-0.021 附近。
residual_mu_ratio 大多约 0.03-0.06。
说明 residual 是小修正，不是主导形变。
这个量级比较稳，但全图收益有限；后续应优先看 high-motion / boundary subset。
```

## 2026-07-04 DIF sigma_rectifier_v2 DNA 六序列结果

设置：

```text
method = dif_sigma_rectifier_v2
dataset = DNA-Rendering
iteration = 25000
densify_until_iter = 1500
logs = logs/dif

0044_11,0051_09,0206_04,0813_05:
  run = 20260704_170652

0007_04,0019_10:
  run = 20260704_203101
```

结果来自各序列：

```text
metrics/results_novelview_25000.json
```

| Sequence | PSNR | SSIM | LPIPS*1000 | dPSNR vs base | dSSIM | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| 0044_11 | 32.9507 | 0.977864 | 21.5263 | -0.0234 | -0.000050 | +0.1284 |
| 0051_09 | 28.5271 | 0.970610 | 31.5712 | -0.0509 | -0.000123 | -0.0187 |
| 0206_04 | 31.3877 | 0.969811 | 34.3681 | +0.0099 | +0.000066 | +0.2577 |
| 0813_05 | 36.0722 | 0.986859 | 18.5396 | -0.0106 | -0.000022 | +0.0522 |
| 0007_04 | 29.4909 | 0.958110 | 45.1296 | -0.0424 | -0.000210 | -0.2855 |
| 0019_10 | 35.1574 | 0.980605 | 21.4971 | -0.0626 | -0.000074 | +0.2091 |

平均：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| baseline | 32.2943 | 0.974045 | 28.7148 |
| dif_sigma_rectifier_v2 | 32.2643 | 0.973976 | 28.7720 |
| delta | -0.0300 | -0.000069 | +0.0572 |

结论：

```text
dif_sigma_rectifier_v2 在六个 DNA 序列全图 novelview 平均指标上没有超过 baseline。
PSNR/SSIM 小幅下降，LPIPS*1000 小幅变差。

它对 0206_04 的 PSNR/SSIM 有轻微正向，
对 0007_04 的 LPIPS 有正向，
但收益不稳定，不能作为当前主线结论。

如果继续 DIF，优先做 high-motion / boundary subset。
如果局部动态区域也没有稳定提升，建议停止 DIF 主线。
```

## 2026-07-04 DIF uncertainty-loss DNA 六序列结果

设置：

```text
method = dif_uncert_loss
dataset = DNA-Rendering
iteration = 25000
densify_until_iter = 1500
lambda_unc = 0.01
s_clamp = [-6, 3]
Delta x = mu
sigma only enters image residual uncertainty loss
logs = logs/dif
run = 20260704_215303
```

实现要点：

```text
1. MLP 输出 mu/sigma，但非刚性位移仍使用 Delta x = mu。
2. 将每个 Gaussian 的 log variance 渲染成 uncertainty map。
3. 使用图像残差异方差 loss:
   L_unc = r^2 * exp(-s) + s
4. 原 RGB/L1/SSIM/LPIPS loss 不变，总 loss:
   L = L_rgb + lambda_unc * L_unc
5. uncertainty render 中 geometry 和主 RGB residual detach，
   避免 uncertainty loss 直接改主几何，只监督 sigma。
```

运行日志：

```text
logs/dif/20260704_215303_DNA-Rendering_dif_uncert_loss_gpu1.log
logs/dif/20260704_215303_DNA-Rendering_dif_uncert_loss_gpu2.log
logs/dif/20260704_215303_DNA-Rendering_dif_uncert_loss_gpu1_retry.log
```

说明：

```text
0206_04 首次在 uncertainty RGB render backward 时触发 CUDA illegal memory access。
定位到 logvar_color 使用 expand 产生非连续 tensor，已改为 contiguous。
该修改不改变数学含义，只修复 CUDA rasterizer 输入内存布局。
0206_04 retry 后完整跑完。
```

| Sequence | PSNR | SSIM | LPIPS*1000 | dPSNR vs base | dSSIM | dLPIPS*1000 |
|---|---:|---:|---:|---:|---:|---:|
| 0007_04 | 29.5019 | 0.958139 | 45.4081 | -0.0314 | -0.000181 | -0.0070 |
| 0019_10 | 35.2759 | 0.980920 | 21.2660 | +0.0559 | +0.000241 | -0.0220 |
| 0044_11 | 32.9431 | 0.977740 | 21.7349 | -0.0310 | -0.000174 | +0.3370 |
| 0051_09 | 28.5181 | 0.970487 | 31.9282 | -0.0599 | -0.000246 | +0.3383 |
| 0206_04 | 31.2591 | 0.968655 | 35.1748 | -0.1187 | -0.001090 | +1.0644 |
| 0813_05 | 36.0480 | 0.986802 | 18.6788 | -0.0348 | -0.000079 | +0.1914 |

平均：

| Method | PSNR | SSIM | LPIPS*1000 |
|---|---:|---:|---:|
| baseline | 32.2943 | 0.974045 | 28.7148 |
| dif_uncert_loss | 32.2577 | 0.973791 | 29.0318 |
| delta | -0.0366 | -0.000255 | +0.3170 |

诊断：

```text
DIF Stats 显示 sigma_mean/sigma_min/sigma_max 长期完全相同:
sigma = 0.048588

这说明当前 sigma 没有学出 point-wise uncertainty 分布，
基本退化成全局常量 uncertainty。
```

结论：

```text
dif_uncert_loss 在 DNA 六序列全图 novelview 平均指标上没有超过 baseline。
只有 0019_10 有轻微正向，其余序列多数下降，0206_04 下降最明显。

当前结果更像是:
sigma 没有获得有效的 spatial / point-wise 区分能力，
uncertainty loss 只提供了弱的全局正则扰动。

不建议把这个版本作为 DIF 主线。
如果继续 DIF，需要先让 sigma 真正空间化，例如增加 sigma map 可视化、
记录 s_map mean/std，或者改成可学习 per-Gaussian uncertainty residual。
```

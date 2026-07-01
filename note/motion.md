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

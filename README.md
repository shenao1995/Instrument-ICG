# Instrument-ICG

独立仓库：用 **ICG / ICG+ / Mb-ICG** 做内镜双器械多刚体 6D 位姿跟踪。

旧项目 `Instrument-pose-opt` 只作参考，本仓库不修改、不覆盖其中任何文件。已验证正确的几何约定从那里复制而来，包括：

- 器械 CAD / 相机坐标系（OpenCV，`+x` 右、`+y` 下、`+z` 前）
- `camera_intrinsics.json` 读取与主点重定心
- 左右内镜基线平移
- `shaft.obj` / `wrist.obj` / 夹爪 mesh 与 `simulated_dual_arm_v1` 对齐
- 腕部 6D + 关节 `alpha, theta_left, theta_right` 的正向运动学
- 双器械语义通道顺序
- nvdiffrast clip 坐标（像素中心投影 + 行翻转）

## 方法

| `--method` | 含义 |
| --- | --- |
| `icg` | 每支器械一个区域；Robust Region LM + Tikhonov；只优化腕部 6DoF，关节冻结 |
| `icgplus` | 杆身 / 腕 / 夹爪多区域（ICG+）；Region-only LM |
| `mbicg` | 多区域 + 运动学雅可比，每支器械 9DoF（腕部 6D + 3 关节） |

观测：

- `--observation mask`：用仿真语义 mask 作为区域后验（验证几何 / FK）
- `--observation histogram`：经典 ICG 颜色直方图

每帧：外层渲染一次并建立 correspondence，冻结 CAD part-local 点、normal、observed 和 confidence；内层重新计算候选位姿的 FK、投影、完整 Jacobian，使用 Huber + LM，并仅接受固定对应点 robust cost 严格下降的步长。

本阶段只验证 Region 优化。`--texture` 旧开关保留，但明确报错，不混入 ORB residual；没有新增 Depth、Tip、RGB histogram 改造或 recovery。现有 histogram 后验代码不变。相机主点重新置中、renderer 和 `simulated_dual_arm_v1` 均保持原样。mask observation 强制 `scale=1`，`--scales` 只影响 histogram。

## 数据

默认路径：

- 序列：`/data/data1/shena/data/simulated_data/data30`
- 网格：`/data/data1/shena/data/simulated_instrument`

## 环境

```bash
conda env create -f environment.yml
conda activate instrument-icg
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install setuptools wheel ninja
python -m pip install git+https://github.com/NVlabs/nvdiffrast.git@253ac4fcea7de5f396371124af597e6cc957bfae --no-build-isolation
```

也可复用已安装 `instrument-pose-opt` 环境中的 torch / nvdiffrast，只需能 `import icg`（在本仓库根目录运行）。

## 单帧运行与验收（PowerShell）

本机已安装的解释器可直接运行，无需在 PowerShell 中先初始化 Conda：

```powershell
Set-Location E:\Work\pythonWorkplace\Instrument-ICG
$python = 'D:\anaconda\envs\instrument-icg\python.exe'
$common = @(
  'track_icg.py',
  '--data-root', 'data/simulated_data/data30',
  '--instrument', 'data/simulated_instrument',
  '--runs', 'run_1789376009_gpu1_env0',
  '--method', 'mbicg', '--observation', 'mask',
  '--start-frame', '0', '--limit', '1', '--seed', '2026',
  '--save-history'
)

# TEST 1: GT，仅评估，不优化。
& $python @common --init gt --corr-iterations 0 --output runs/lm-validation/test1-gt-zero

# TEST 2: GT，4 次 correspondence × 2 次固定对应点 LM update。
& $python @common --init gt --corr-iterations 4 --update-iterations 2 --output runs/lm-validation/test2-gt-lm

# TEST 3: 1 deg / 1 mm / 1 deg，固定随机种子。
& $python @common --init perturb-gt --init-rotation-deg 1 --init-translation-mm 1 --init-joints-deg 1 --corr-iterations 4 --update-iterations 2 --output runs/lm-validation/test3-small-lm
```

这些是噪声标准差参数，实际初始误差以输出的 `initial_pose_error` 为准。
TEST 3 的左右器械平移、旋转、关节 MAE 都应低于各自 initial 值；如果不满足，停止，不运行 TEST 4。

```powershell
# 只有 TEST 3 上述六项误差均改善后，才执行下面的单帧 TEST 4。
& $python @common --init perturb-gt --init-rotation-deg 2 --init-translation-mm 2 --init-joints-deg 2 --corr-iterations 4 --update-iterations 2 --output runs/lm-validation/test4-medium-lm
```

`--limit 1 --start-frame 0` 固定为第 0 帧。保留的 loader 在 `--limit > 1` 时均匀抽帧，不代表连续的前几帧。本轮不运行完整数据集。

输出目录包含 `config.json`、`frames.jsonl`、`summary.json`、`summary.csv`、`overlays/`；`--save-history` 额外保存 `history/*.json`。
`frames.jsonl` 与 `summary.json` 保存初始/最终 Dice、平移误差、旋转误差、关节误差；这些 GT 指标只用于 evaluation。`optimized` 表示至少接受过一次 update，零优化基线仍正常输出评价指标。

## 新优化参数

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--corr-iterations` / `--iterations` | 4 | 同一 argparse 参数的两个名称；0 仅评估初值 |
| `--update-iterations` | 2 | 同批 correspondence 的内层更新次数 |
| `--lm-lambda` | 0.01 | 每帧初始 LM 阻尼，帧内延续 |
| `--lm-lambda-min` / `--lm-lambda-max` | 1e-8 / 1e8 | 阻尼上下限 |
| `--lm-max-retries` | 5 | 首次 solve 之后最多重试 5 次，总计最多 6 个 candidate |
| `--huber-delta-px` | 2 | Huber 阈值，单位像素 |
| `--max-rotation-step-deg` | 3 | 每支器械旋转 3-vector 的范数上限 |
| `--max-translation-step-mm` | 2 | 每支器械平移 3-vector 的范数上限 |
| `--max-joint-step-deg` | 3 | 每个关节分别限幅 |
| `--early-stop-cost-rel` | 1e-4 | accepted step 的相对 cost 降幅阈值 |
| `--early-stop-translation-mm` | 0.05 | 小步长平移阈值 |
| `--early-stop-rotation-deg` | 0.02 | 小步长旋转阈值 |
| `--early-stop-joint-deg` | 0.02 | 所有关节最大变化阈值 |
| `--early-stop-patience` | 2 | 连续达到小步长或小降幅的 accepted update 次数 |
| `--min-correspondences` | 30 | 少于该数量不更新 |
| `--tikhonov-rotation` / `--tikhonov-translation` / `--tikhonov-joint` | 100 / 1000 / 50 | 保留原来的增量正则化系数，和 LM 阻尼分开命名 |
| `--region-sigma-px` | 25 15 10 | 保留原 Region likelihood 方差缩放；按外层取值，最后一个重复 |

固定一批 correspondence 后：

```text
r(q) = n.T [project(FK(local_xyz, q)) - observed]
J(q) = n.T J_projection(X(q)) J_kinematic(local_xyz, q)
confidence_weight = confidence / sigma_px^2
IRLS_weight = confidence_weight * min(1, huber_delta_px / abs(r))
H = sum(IRLS_weight * J.T J)
g = sum(IRLS_weight * J.T r)
H_lm = H + lm_lambda * diag(max(diag(H), 1e-12)) + Tikhonov
E(q) = sum(confidence_weight * Huber(r(q)))
```

sigma 在同一批 correspondence 内固定，Huber 阈值仍以原始像素计量。保留 sigma 很重要：去掉旧方差缩放而继续使用原 Tikhonov 数值，会显著削弱增量正则化。

Tikhonov 只惩罚本次增量，没有 GT 或初始 pose 作为 regularization target。acceptance 仅比较 `E(candidate) < E(current)`：接受后阻尼乘 0.5，拒绝后乘 10 并重试；全部失败则保留原 pose，停止该帧。投影无效（例如点到相机后方）时拒绝整个 candidate，不通过删除点降低 cost。左右器械以及 rotation / translation / 各关节独立限幅。

小步长条件要求两支器械同时满足三个物理单位阈值；它与相对 cost 降幅条件为 OR 关系。连续满足 `--early-stop-patience` 后停止该帧。新的对应点集合有新的目标函数，不能把不同外层的 cost 直接视作同一个单调序列，也不能从 cost 下降推断所有 GT pose error 必然下降。

## History

每个外层条目保存：`corr_iteration`、`scale`、`residual_sigma_px`、总数及六个 `region_id` 的 correspondence 数量、`region_residual_before`、`updates` 和可选的 `stop_reason`。

每个 update 保存：`update_iteration`、`cost_before/after`、`raw_cost_before/after`、`accepted`、`lm_lambda_before/after`、`lm_retries`、每次候选的 `attempts`、左右器械的实际旋转/平移/三个关节步长，以及 `residual_before/after`。关节步长包含关节边界裁剪的实际效果；拒绝后的 retained step 为 0。拒绝的候选 cost 在 `attempts` 中，`cost_after` 始终表示保留位姿的 cost。

残差统计包括未使用 Huber 的 `raw_cost`、`robust_cost`，以及绝对残差的 mean / median / p90 / max（像素）。两种 cost 均含 confidence / sigma² 权重。history 不再用混合 rad/metre 的 `delta_norm`。

## Jacobian 与优化器检查

在已激活环境的项目根目录运行：

```powershell
python -m icg.check_jacobians
python -m icg.check_region_jacobians
python -m unittest discover -s tests -v
```

或直接使用本机解释器：

```powershell
& $python -m icg.check_jacobians
& $python -m icg.check_region_jacobians
& $python -m unittest discover -s tests -v
```

原 point Jacobian 检查保留。新增 Region 检查覆盖两支器械的四个部件、全部 18 DoF、不同候选姿态、非零 shaft offset、local/camera 往返，并使用独立 FK 的中心差分；任意误差超出 `atol=2e-5, rtol=2e-6` 就失败。CPU 优化器测试覆盖 Huber、cost/gradient 一致性、拒绝回滚、重试、独立限幅、固定对应点、early stop 和禁止读取 GT。

## 本机单帧验收状态（2026-09-17）

帧 0000，seed=2026，默认参数。TEST 1 的 Dice 为 0.973850；两个 Jacobian 检查及 12 项 CPU 测试通过。
TEST 2 的 Dice 为 0.972247，8/8 accepted inner updates 的 robust cost 严格下降，但尚不能据此宣布跨 correspondence 的 pose drift 已解决。
TEST 3 的 Dice 从 0.687424 提升到 0.964829，但部分平移、旋转及关节误差增大，**TEST 3 未通过**；未执行 TEST 4，也未运行大规模 dataset。

| TEST 3 指标 | left initial → final | right initial → final |
| --- | --- | --- |
| 平移误差 mm | 0.661206 → 2.038203 | 1.527710 → 2.248275 |
| 旋转误差 deg | 1.953845 → 1.402211 | 1.220771 → 1.953244 |
| 关节 MAE deg | 0.411816 → 1.082460 | 1.067971 → 1.251399 |

这一版完成了固定对应点上的 Robust LM 接受机制，单帧结果尚不支持“小扰动的所有三维 pose error 单调改善”的结论。保留失败结果，不通过改动主点、几何、contour search 或读取 GT 来掩盖验收失败。

单帧诊断另外比较了更高初始 LM 阻尼（1.0）与 contour/projection anchor 对齐；二者均未通过全部 pose-error 标准，后者还使 GT Dice 降至 0.958414。因此未据此更改默认阻尼、observed 坐标、相机或搜索逻辑；这些诊断不属于 TEST 4。

## 状态向量

两支器械共 18 维增量：每支 `θr(3) + θt(3) + α + θ_left + θ_right`。增量定义在腕部模型坐标系，与 ICG 的 axis-angle 变分一致；Mb-ICG 用刚体雅可比把它投到各 link。

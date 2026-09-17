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
| `icg` | 每支器械一个区域；Newton + Tikhonov；只优化腕部 6DoF，关节冻结 |
| `icgplus` | 杆身 / 腕 / 夹爪多区域（ICG+）；可选 `--texture` ORB |
| `mbicg` | 多区域 + 运动学雅可比，每支器械 9DoF（腕部 6D + 3 关节） |

观测：

- `--observation mask`：用仿真语义 mask 作为区域后验（验证几何 / FK）
- `--observation histogram`：经典 ICG 颜色直方图

每帧：渲染当前位姿轮廓 → 沿法向搜索对应线 → 组装梯度 / Hessian → 正则 Newton 更新。

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

## 运行

```bash
# Mb-ICG，第一帧扰动 GT，检查 2 帧
python track_icg.py --runs run_1789376009_gpu1_env0 --limit 2 --method mbicg

# ICG / ICG+
python track_icg.py --runs run_1789376009_gpu1_env0 --limit 5 --method icg
python track_icg.py --runs run_1789376009_gpu1_env0 --limit 5 --method icgplus --texture

# 第一帧扰动，之后用上一帧结果
python track_icg.py --runs run_1789376009_gpu1_env0 --track --method mbicg --limit 50
```

输出写到 `runs/<method>/<样本>/`：`summary.json`、`frames.jsonl`、`overlays/`。

## 状态向量

两支器械共 18 维增量：每支 `θr(3) + θt(3) + α + θ_left + θ_right`。增量定义在腕部模型坐标系，与 ICG 的 axis-angle 变分一致；Mb-ICG 用刚体雅可比把它投到各 link。

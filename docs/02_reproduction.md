# 02｜50 GB 方案：下载、验证与单场景闭环

## 1. 空间预算

本次 Windows 实测如下。GiB 采用 $2^{30}$ 字节；供应商常写的 GB 采用 $10^9$ 字节，两者不要混用。

| 组件 | 下载/本地占用 | 解压后 | 是否提交 Git |
|---|---:|---:|---|
| nuPlan mini DB | 7.96 GiB | 13.37 GiB | 否 |
| nuPlan maps | 0.90 GiB | 1.33 GiB | 否 |
| PLUTO 权重 | 0.048 GiB | 0.048 GiB | 否 |
| Conda 环境 | — | 6.60 GiB | 否 |
| 上游源码与紧凑结果 | — | 0.24 GiB | 仅原创小文件 |
| **保留压缩包时峰值** |  | **约 30.6 GiB** |  |

50 GB 方案留下约 19 GB 余量，用于断点续传的临时文件、Hydra 日志、视频以及文件系统波动。本次没有下载 nuPlan camera/lidar sensor blobs；PLUTO 规划复现只需 DB 和地图。完整 mini sensor 包会让 50 GB 方案失效。

机器要求：Python 3.9、NVIDIA 驱动可支持 CUDA 11.8、Conda、Git。实测设备为 RTX 4090 D；模型只有约 424 万参数，显存不是主要瓶颈，地图与数据库空间才是。

## 2. 为什么不把上游代码、数据和权重塞进仓库

固定的 PLUTO 提交没有 `LICENSE` 文件，不能据此推定有再分发权。nuPlan 数据也有自己的使用条款。故本仓库采用三条规则：

1. `_deps/` 由脚本从官方仓库克隆并固定提交；
2. `data/`、`downloads/`、`checkpoints/` 永不提交；
3. 只发布本仓库原创脚本、兼容层、教学和紧凑的结果图/JSON。

使用数据前仍应自行阅读并遵守 [nuPlan 条款与下载说明](https://www.nuscenes.org/nuplan)。脚本中的下载动作不替代你对条款的判断。

## 3. 环境与源码

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
```

脚本首先检查当前盘至少有 50 GB 可用空间，然后固定：

```text
PLUTO       b9964b649c660f1f4a971d614c66f5992e24c18a
nuPlan v1.2 ce3c323af01c0d7ec5672f7832ef53f9c679aab0
```

环境定义在 `environment.yml`。关键版本是 Python 3.9、PyTorch 2.0.1 + CUDA 11.8、torchvision 0.15.2、PyTorch Lightning 2.0.1、torchmetrics 0.10.2、NumPy 1.23.4。

### Windows 的两个兼容点

NATTEN 0.14.6 没有 Windows wheel，且新版 NATTEN 的 state dict 与旧权重不兼容。本仓库的 `compat/natten` 精确保留 `qkv/rpb/proj` 参数名、相对偏置形状和旧窗口边界索引，用普通 PyTorch 完成短序列前向与反向。

nuPlan v1.2 的地图模块在 Windows 也会无条件 `import fcntl`。`compat/fcntl.py` 只为“完整地图已经下载、sequential 单进程、只读”的本教程提供导入兼容；若你要并行下载/改写地图，不可使用这个 no-op 锁。

## 4. 数据和权重

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_artifacts.ps1
```

脚本做四件事：

1. 用 `curl -C -` 从 nuPlan 官方 CDN 断点续传 maps 和 mini DB；
2. 对压缩包计算 SHA-256，与本次已验证值比较；
3. 解压并建立 nuPlan v1.2 期望的 `nuplan-v1.1/splits/mini` 目录联接；
4. 从 RIFT 项目公开的 PLUTO 权重目录只下载原始 `pluto_1M_aux_cil.ckpt`。

PLUTO README 的 OneDrive 地址在 2026-07-18 实测会因 HKUST 账户迁移跳转到组织登录，不能匿名直链下载。RIFT 的公开目录明确列出 Pluto 与微调变体；本仓库只取名为 `pluto_1M_aux_cil.ckpt` 的原始文件，并做结构验证，而不是只相信文件名。

| 文件 | 字节 | SHA-256 |
|---|---:|---|
| `nuplan-maps-v1.0.zip` | 971,557,640 | `d0310009...702ffc6` |
| `nuplan-v1.1_mini.zip` | 8,550,100,030 | `a3fe40af...8d987b` |
| `pluto_1M_aux_cil.ckpt` | 51,431,136 | `ce60d7c8...2943e1` |

完整值和来源见 `docs/artifact_manifest.json`。这些哈希是本地观察值，不是发布方签名；它们用于重复本次实验和发现传输损坏。

## 5. 三层验证

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\self_test.ps1
```

第一层是数值兼容性：验证 NATTEN 边界窗口、GPU 前向、反向梯度和有限值。

第二层是检查点结构：从上游 planner YAML 实例化模型，去除 Lightning 的 `model.` 前缀后执行 `strict=True`。本次结果：

```json
{
  "state_tensor_count": 438,
  "parameter_count": 4240589,
  "missing_keys": [],
  "unexpected_keys": []
}
```

第三层是数据完整性：64 个 nuPlan mini `.db` 和 4 个 `map.gpkg` 全部以只读 SQLite 连接运行 `PRAGMA quick_check(1)`，结果均为 `ok`。这比“文件存在”强得多，也能发现解压中断造成的页损坏。

## 6. 运行一个闭环场景

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_mini.ps1 -Render
```

固定随机种子为 0，`mini_demo_scenario` 限制总场景数为 1，worker 为 sequential，challenge 为 `closed_loop_nonreactive_agents`。每个规划周期执行：

```text
DB 当前帧 → 构建历史/地图/参考线 → PLUTO GPU 前向
→ 候选剪枝 → 参与者预测参与 TTC → 规则评分 → 轨迹执行
→ 环境推进 0.1 s → 下一周期
```

结果目录必须包含 challenge 字符串。例如：

```text
demo/outputs/closed_loop_nonreactive_agents/run-001/
```

nuPlan v1.2 的聚合器通过路径字符串过滤 challenge。如果强行把 `output_dir` 设成不含 `closed_loop_nonreactive_agents` 的目录，17 个单项指标仍会生成，但聚合器会报告 “No metric files found”，而 PLUTO 末尾的 `max()` 又会因空列表报错。这不是推理失败，是输出目录破坏了聚合约定。

## 7. 本次结果怎样读

场景 token 为 `9e30155b8bb55fd9`，类型 `changing_lane_to_left`，15 秒、10 Hz，共渲染 149 帧。闭环总分为

$$
S=0.9497837499.
$$

二值安全/合规项全部为 1：无责任碰撞、TTC、可行驶区域、方向、舒适、进度门槛和限速。连续进度项为

$$
P_{expert}=0.8393079998.
$$

nuPlan weighted-average 聚合先计算非乘法指标的加权平均，再乘以碰撞、可行驶性等硬门控项。因此所有硬门控通过时，连续进度不足仍会让总分低于 1；任一关键乘法指标为 0，则总分可能被直接压到 0。

平均规划周期 196.7 ms、中位数 176.7 ms。模拟频率是 10 Hz，但本教程没有宣称实时：Windows 纯 PyTorch NATTEN、渲染、地图查询和规则评估都包含在计时中。`PlutoPlanner` 源码的 `feature_building_runtimes` 计时位置实际上在构建前立即结束，接近 $5\times10^{-7}$ 秒，不能作为真实特征耗时；应看 `inference_runtimes` 或自行加 profiler。

## 8. 和论文数字的边界

论文的 Val14/Test14 是成百上千个场景的统计，且官方模型使用约 1M 训练样本。这里一个 mini 场景的 0.9498 只能证明：数据读取、模型预测、后处理、闭环状态推进、指标和渲染形成了完整链路。它不能证明论文 93.57 的总体结论被复现，也不能用于模型排名。

若要做更可信的小规模实验，下一步不是挑一个高分场景，而是固定一组 token，至少覆盖跟车、换道、路口、停车、弱势交通参与者等类型，报告均值、分位数、失败数和随机种子。

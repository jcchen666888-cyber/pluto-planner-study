# 03｜训练教学：本仓库不完整训练，但把训练链路讲完整

官方 README 明确标注训练部分 “not fully tested”。本仓库的本机任务只下载预训练权重并推理，没有启动完整训练；下面的目的，是让你能读懂训练代码、设计 sanity run，并知道扩展到 1M 数据时发生什么。

## 1. 数据流分成缓存与训练两阶段

直接从 SQLite、地图和场景拓扑实时构建 1M 个样本会让 GPU 等 CPU。PLUTO 先运行 `py_func=cache`，把历史参与者、地图折线、参考线、目标轨迹、有效掩码和 cost map 预计算到 cache；训练时设置 `cache.use_cache_without_dataset=true`，主要读取缓存张量。

最小 sanity cache（上游语义，50 个场景）为：

```powershell
$env:NUPLAN_DATA_ROOT = "C:\path\to\data"
$env:NUPLAN_MAPS_ROOT = "$env:NUPLAN_DATA_ROOT\maps"
$env:PYTHONPATH = ".\compat;.\_deps\pluto-upstream;.\_deps\nuplan-devkit"

conda run -n pluto-study python .\_deps\pluto-upstream\run_training.py `
  py_func=cache +training=train_pluto `
  scenario_builder=nuplan_mini `
  cache.cache_path=.\cache\sanity `
  cache.cleanup_cache=true `
  scenario_filter=training_scenarios_tiny `
  worker=sequential
```

这一步会写缓存；它不同于本仓库已经完成的推理。完整 1M cache 需要 trainval DB，不是 50 GB mini 方案能覆盖的内容。

## 2. 一个 batch 里有什么

主要张量可抽象为：

```text
agent.position/heading/velocity/valid_mask/target
map.polygon_center/valid_mask/point features
static_objects
reference_line.position/valid_mask/future_projection
cost_maps
data_n_valid_mask（启用 CIL 时）
```

设原始 batch size 为 $B$。启用 CIL 后，每个样本生成 origin、positive、negative 三份，编码器实际看到 $3B$ 个场景；只有 origin 和 positive 的前 $2B$ 份进入轨迹/预测监督，三份一起进入对比损失。因此显存不会只增加一点，场景编码部分近似扩大三倍。

## 3. 目标模式匹配

每条参考线 $r$ 和纵向模式 $m$ 对应一条候选。真值终点先投影到参考线：投影结果可写成 $(s_r,d_r)$，分别为沿线距离和横向距离。

$$
r^*=\arg\min_r\left(d_r+10^6\mathbb I[r\text{ is padding}]\right),
$$

$$
m^*=\operatorname{clip}\left(\left\lfloor s_{r^*}/(R/N_L)\right\rfloor,0,N_L-1\right).
$$

源码根据有效未来点数选择对应的终点索引，再构造 one-hot 分类目标。匹配是离散、不可微的，但它只负责分配监督；被选轨迹上的 smooth L1、辅助损失和分类交叉熵仍可正常反传。

一个常见误解是“让所有候选都拟合真值”。这样会直接消灭多模态。正确做法是 winner-takes-target：匹配到的模式回归真值，所有有效模式通过分类分数竞争。

## 4. 实际总损失

公开训练器的实际实现为

$$
\mathcal L=\mathcal L_{ego-reg}+\mathcal L_{ego-cls}+\mathcal L_{prediction}
+\mathcal L_{contrast}+\mathcal L_{collision}+\mathcal L_{ref-free}.
$$

各项在不启用对应功能时返回 0。论文写了抽象权重 $w_i$，但公开代码没有默认的每项可配权重。由于各损失量纲不同，直接加入自定义权重前应先画每项的 epoch 曲线和梯度范数，不能只凭总 loss 调参。

### 回归中的 mask 归一化

$$
\mathcal L_{reg}=\frac{\sum_{b,t}m_{b,t}\sum_d\operatorname{smoothL1}(\hat y-y)}{\sum_{b,t}m_{b,t}}.
$$

分母是有效时间点数，不是 batch size。这使不同轨迹长度的样本按有效点数贡献，但也意味着短轨迹和长轨迹的单样本权重不同。若 `valid_mask.sum()==0`，预测损失可能除零；数据构建和自测应保证监督目标至少有一个有效点。

### 航向向量正则

可选的 `regulate_yaw` 约束

$$
\mathcal L_{yaw}=\left|\sqrt{c^2+s^2}-1\right|,
$$

使 $(c,s)$ 接近单位圆。否则推理虽可用 `atan2(s,c)` 得角度，但接近 $(0,0)$ 时梯度和角度都不稳定。

## 5. 优化器与学习率

默认训练器使用 AdamW，并把参数分两组：Linear/Conv/MHA 等权重使用 weight decay；bias、LayerNorm、Embedding 等不衰减。默认：

```text
epochs=25
warmup_epochs=3
lr=1e-3
min_lr=1e-6
weight_decay=1e-4
gradient_clip_val=5.0（L2 norm）
```

warmup 阶段第 $e$ 个 epoch 的学习率为

$$
\eta_e=\eta_{max}\frac{e+1}{E_w},\quad e<E_w.
$$

随后余弦退火：

$$
\eta_e=\eta_{min}+\frac12(\eta_{max}-\eta_{min})
\left[1+\cos\left(\pi\frac{e-E_w}{E-E_w}\right)\right].
$$

warmup 缓解随机初始化 Transformer 早期的大梯度；末期余弦退火让参数在较小步长下收敛。

## 6. 先做什么 sanity training

如果未来允许做极小训练，推荐的验收顺序是：

1. `fast_dev_run=true`：1 个训练 batch + 1 个验证 batch，只验证 wiring；
2. 固定 1 个 batch 过拟合：检查 loss 能否持续下降，验证梯度、target 和 mask；
3. 50 场景、1 epoch：检查 cache、checkpoint、恢复训练和验证指标；
4. 多 GPU 前先在单 GPU 上保存和严格重载 checkpoint；
5. 最后才扩大数据、batch 和 worker。

最小命令示意：

```powershell
conda run -n pluto-study python .\_deps\pluto-upstream\run_training.py `
  py_func=train +training=train_pluto `
  scenario_builder=nuplan_mini `
  cache.cache_path=.\cache\sanity cache.use_cache_without_dataset=true `
  worker=sequential `
  data_loader.params.batch_size=1 data_loader.params.num_workers=0 `
  lightning.trainer.params.accelerator=gpu `
  lightning.trainer.params.devices=1 `
  lightning.trainer.params.strategy=auto `
  lightning.trainer.params.fast_dev_run=true
```

这是教学命令，未在本任务执行，也不会产生可用模型。Windows 多进程 DataLoader 的 spawn 行为与 Linux 不同，sanity 阶段先用 `num_workers=0`。

## 7. CIL 和完整 checkpoint 必须显式对齐配置

上游 `config/model/pluto_model.yaml` 默认：

```yaml
use_hidden_proj: false
cat_x: false
ref_free_traj: false
```

但下载的 `pluto_1M_aux_cil.ckpt` 与 planner YAML 对应：

```yaml
use_hidden_proj: true
cat_x: true
ref_free_traj: true
```

CIL 还要求：

```text
model.use_hidden_proj=true
+custom_trainer.use_contrast_loss=true
```

因此，“官方默认 training YAML + 官方 CIL checkpoint”并非天然严格兼容。训练与继续训练必须保存完整 Hydra config，并至少显式覆盖：

```text
model.use_hidden_proj=true
model.cat_x=true
model.ref_free_traj=true
+custom_trainer.use_contrast_loss=true
```

少一个 flag 都可能表现为 missing/unexpected keys，或者更隐蔽地实例化了错误结构。推理应使用 `config/planner/pluto_planner.yaml`，本仓库的严格检查正是从该文件实例化模型。

## 8. 多 GPU 的注意点

原配置为 `devices=-1`、`sync_batchnorm=true`、`strategy=ddp_find_unused_parameters_false`。这适合 Linux 多卡目标，却不适合直接拿来做 Windows 单卡 sanity。还要注意：

- CIL 的 batch 是三倍编码负载；所谓 batch size 32 是每 GPU 的原样本数还是增强后样本数，必须在显存测量中说明；
- `find_unused_parameters_false` 假设每个启用的 head 都参与 loss，若打开 head 却没接监督，DDP 可能挂起或报错；
- 全局 batch 改变时，是否线性缩放学习率不是定理。先保持 $10^{-3}$ 做短跑并观察梯度；
- SyncBatchNorm 对本模型价值有限且会引入通信，单 GPU 要关闭；
- checkpoint 中优化器、scheduler 和 global step 只有在“恢复训练”时使用；纯微调可只加载 `state_dict` 并新建优化器。

## 9. 怎样判断训练真的学到了

只看 `val_loss` 不够。建议同时记录：

- planning minADE / minFDE / miss rate；
- prediction ADE / FDE；
- 分类熵和有效模式使用率，发现 mode collapse；
- 每个损失项与主要模块梯度范数；
- 无参考线样本比例与 ref-free 误差；
- 按场景类型分桶的闭环碰撞、TTC、进度与舒适；
- 固定 token 的可视化回归测试。

离线 ADE 变好不保证闭环变好：闭环存在状态分布漂移、tracker 可执行性和规则交互。PLUTO 的 CIL、SDE、前向模拟和后处理正是为弥补这条鸿沟，但最终仍要在未参与训练的闭环场景上验证。

## 10. 从本教程到完整训练还缺什么

50 GB 方案只含 nuPlan mini，远少于官方约 1M 样本训练所需的 trainval DB、缓存空间和训练时间。完整训练前应单独重新做容量计划，至少考虑：原始 trainval、feature cache、多个 checkpoint、W&B/TensorBoard、临时 cache 和失败重跑。不要在本教程剩余空间上直接启动 `training_scenarios_1M`。

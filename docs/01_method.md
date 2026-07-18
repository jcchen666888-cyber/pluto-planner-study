# 01｜PLUTO 方法：从场景张量到可执行轨迹

本文的主线不是“背模块名”，而是回答三个问题：网络到底输出什么；为什么候选轨迹不会轻易塌缩为同一种行为；学习分数为什么还需要规则后处理。

## 1. 问题定义

令自车为 $A_0$，动态交通参与者为 $A_{1:N_A}$，静态障碍物为 $O_{1:N_S}$，高精地图为 $M$，信号灯等上下文为 $C$。历史和未来长度分别为 $T_H,T_F$。PLUTO 网络 $f_\phi$ 同时产生：

$$
(\mathbf T_0,\boldsymbol\pi_0),\ \mathbf P_{1:N_A}
=f_\phi(\mathcal A,\mathcal O,M,C),
$$

其中

$$
\mathbf T_0=\{\tau_i\}_{i=1}^{N_T},\quad
\tau_i=\mathbf y_{0,i}^{1:T_F},\quad
\boldsymbol\pi_0=\{\pi_i\}_{i=1}^{N_T}.
$$

$\mathbf P$ 是其他交通参与者的未来预测。真正执行的轨迹不是简单的 $\arg\max_i\pi_i$，而是

$$
(\tau^*,\pi^*)=\arg\max_{(\tau_i,\pi_i)}
\mathcal S(\tau_i,\pi_i,\mathbf P,\mathcal O,M,C).
$$

这已经揭示 PLUTO 的结构：神经网络负责生成和学习偏好，显式后处理负责动力学可执行性和安全筛选。

本仓库使用的权重配置为 $T_H=21$ 帧（当前帧加 2 秒历史）、$T_F=80$ 帧（未来 8 秒），采样周期 $\Delta t=0.1$ 秒，隐藏维度 $D=128$，纵向查询数 $N_L=12$。

## 2. 输入表示

### 2.1 自车坐标系

所有局部几何量都应先变换到当前自车坐标系。若世界坐标点为 $p^w$，自车位置和航向为 $p_e^w,\theta_e$，则

$$
p^e=R(-\theta_e)(p^w-p_e^w),\qquad
R(\theta)=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}.
$$

平移消除绝对位置，旋转消除全局朝向，使相同局部交通结构在不同城市位置具有相近表示。代价是丢失全局几何信息，因此编码器后面还要补 Fourier 位置嵌入。

### 2.2 动态参与者历史

论文把单帧状态写成

$$
s_i^t=(p_i^t,\theta_i^t,v_i^t,b_i^t,\mathbb I_i^t),
$$

其中 $b=(l,w)$，$\mathbb I$ 表示该帧是否有效。相邻差分为

$$
\hat s_i^t=(p_i^t-p_i^{t-1},\ \theta_i^t-\theta_i^{t-1},\
v_i^t-v_i^{t-1},\ b_i^t,\mathbb I_i^t).
$$

差分的直觉是把“在地图哪个绝对位置”换成“如何运动”。工程实现的 `history_channel=9` 与论文文字中的 8 维记号并不完全一致；复现时应以特征构建器实际张量为准，不能只按论文维数重写网络。

历史编码采用 1D 邻域注意力与 FPN。对时间位置 $t$、头 $h$ 和局部窗口 $\mathcal N(t)$：

$$
q_t=W_Qx_t,\quad k_j=W_Kx_j,\quad v_j=W_Vx_j,
$$

$$
a_{tj}^{(h)}=
\frac{\exp\left(q_t^{(h)\top}k_j^{(h)}/\sqrt{d_h}+r_{t-j}^{(h)}\right)}
{\sum_{u\in\mathcal N(t)}\exp\left(q_t^{(h)\top}k_u^{(h)}/\sqrt{d_h}+r_{t-u}^{(h)}\right)},
$$

$$
z_t^{(h)}=\sum_{j\in\mathcal N(t)}a_{tj}^{(h)}v_j^{(h)}.
$$

相对位置偏置 $r_{t-j}$ 正是旧版 NATTEN 检查点里的 `.attn.rpb` 参数。窗口注意力复杂度约为 $O(T_Hkd)$，而完整时间自注意力为 $O(T_H^2d)$。这里 $T_H$ 很短，纯 PyTorch 兼容实现也能满足教学推理。

### 2.3 自车状态与 SDE

模仿学习很容易仅凭速度、加速度外推，而忽略红灯或前车。PLUTO 只编码当前自车状态，并使用 state dropout encoder（SDE）。若状态通道掩码 $m_j\sim\operatorname{Bernoulli}(1-p)$，训练时可写为

$$
\tilde s_j=\frac{m_j}{1-p}s_j.
$$

本权重的 dropout 比例是 $p=0.75$。高比例不是普通正则化的随意选择，而是在主动破坏“照抄运动学状态”的捷径，迫使场景交互特征参与决策。

### 2.4 地图折线与静态障碍

一条车道折线第 $i$ 个点使用

$$
f_i=(p_i-p_0,\ p_i-p_{i-1},\ p_i-p_i^{left},\ p_i-p_i^{right})\in\mathbb R^8.
$$

前两项表达沿线几何，后两项表达左右边界和可行驶宽度。PointNet 风格编码可概括为

$$
e_{poly}=\max_i\operatorname{MLP}(f_i),
$$

其中逐点共享 MLP 保证点数变化可处理，池化得到固定维度。静态锥桶、路障等用位置、航向和尺寸经过两层 MLP 编码。

## 3. 场景 Transformer

把自车、动态体、静态体和地图 token 拼接：

$$
E_0=\operatorname{concat}(E_{AV},E_A,E_O,E_P)+PE(p,\theta)+E_{attr}.
$$

$PE$ 是 Fourier 位置编码。可将一维标量 $u$ 的部分编码理解为

$$
\gamma(u)=[\sin(2\pi b_1u),\cos(2\pi b_1u),\ldots,
\sin(2\pi b_Ku),\cos(2\pi b_Ku)],
$$

再经 MLP 投影到 $D$ 维。多频率使网络能同时表示缓慢变化的道路拓扑和细粒度相对位置。

第 $l$ 个 pre-norm Transformer 块为

$$
\hat E=\operatorname{LN}(E_{l-1}),
$$

$$
E'_l=E_{l-1}+\operatorname{MHA}(\hat E,\hat E,\hat E),
$$

$$
E_l=E'_l+\operatorname{FFN}(\operatorname{LN}(E'_l)).
$$

单头注意力是

$$
\operatorname{Attn}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V,
$$

无效参与者和 padding 折线在 $M$ 中置为负无穷。这里的掩码 dtype 若不是 `bool`，PyTorch 2.0 Windows 会给性能警告；数值不变，但会多一次转换。

## 4. 半锚定的多模态解码

### 4.1 为什么把行为分成横向和纵向

一个驾驶行为可以近似拆成：走哪条几何通道（横向）以及沿通道走多快/多远（纵向）。PLUTO 用路线附近的参考线 $Q_{lat}\in\mathbb R^{N_R\times D}$ 做横向查询，用 $N_L$ 个可学习向量 $Q_{lon}\in\mathbb R^{N_L\times D}$ 表示纵向模式：

$$
Q_0[r,l]=W\,[Q_{lat}[r];Q_{lon}[l]]+b.
$$

因此候选数为 $N_T=N_RN_L$。参考线是“半锚”：给出横向几何先验；纵向 query 保持数据驱动。

### 4.2 因子化注意力的复杂度推导

若把 $N_RN_L$ 个 query 拉平做完整注意力，注意力矩阵边长为 $N_RN_L$，配对数为

$$
(N_RN_L)^2=N_R^2N_L^2.
$$

PLUTO 先固定纵向索引，对每个 $l$ 在 $N_R$ 条参考线间做横向注意力，成本 $N_LN_R^2$；再固定参考线，对每个 $r$ 在 $N_L$ 个纵向模式间做注意力，成本 $N_RN_L^2$。总计

$$
O(N_LN_R^2+N_RN_L^2),
$$

相对完整注意力节省因子约为

$$
\frac{N_R^2N_L^2}{N_LN_R^2+N_RN_L^2}
=\frac{N_RN_L}{N_R+N_L}.
$$

随后 query 对场景 token 做 cross-attention：

$$
Q_l=\operatorname{CrossAttn}(\hat Q_{l-1},E_{enc},E_{enc}).
$$

两个 MLP 分别解码轨迹和置信度：

$$
\mathbf T_0=\operatorname{MLP}_{traj}(Q_{dec}),\qquad
\boldsymbol\pi_0=\operatorname{MLP}_{score}(Q_{dec}).
$$

每个点实际回归 $(x,y,\cos\theta,\sin\theta)$；用单位方向向量比直接回归跨越 $-\pi/\pi$ 的角度连续。

## 5. 教师强制与模仿损失

训练时不能把所有模式都拉向同一条真值，否则会模式坍缩。PLUTO 先把真值终点投影到各参考线，选横向距离最小的参考线 $r^*$。再把沿线距离 $s^*$ 分桶：

$$
l^*=\operatorname{clip}\left(\left\lfloor\frac{s^*}{\Delta s}\right\rfloor,0,N_L-1\right),
\qquad \Delta s=\frac{R}{N_L}.
$$

源码中 $R=120$ m、$N_L=12$，所以每个纵向模式约负责 10 m 的区间。目标模式为 $(r^*,l^*)$，只有它做主回归：

$$
\mathcal L_{reg}=\frac{1}{\sum_tm_t}
\sum_t m_t\sum_d\operatorname{smoothL1}(\hat y_{r^*,l^*,t,d}-y^{gt}_{t,d}).
$$

其中

$$
\operatorname{smoothL1}(e)=
\begin{cases}\frac12e^2,&|e|<1\\|e|-\frac12,&|e|\ge1.\end{cases}
$$

小误差区是二次函数，提供平滑梯度；大误差区是线性函数，降低异常标注的影响。分类目标为 one-hot：

$$
\mathcal L_{cls}=-\log
\frac{\exp\pi_{r^*,l^*}}{\sum_{r,l}\exp\pi_{r,l}}.
$$

无参考线的场景由 reference-free head 直接回归一条轨迹，损失同样是 masked smooth L1。其他参与者的预测损失是

$$
\mathcal L_p=\frac{\sum_{a,t}m_{a,t}\,\operatorname{smoothL1}(\hat P_{a,t}-P^{gt}_{a,t})}
{\sum_{a,t}m_{a,t}}.
$$

## 6. 可微 ESDF 辅助约束

### 6.1 用圆近似车身

将车身用 $N_c=3$ 个圆覆盖。轨迹点为 $(p_t,\theta_t)$，第 $i$ 个圆心相对后轴偏移为 $l_i$：

$$
c_{t,i}=p_t+l_i[\cos\theta_t,\sin\theta_t]^\top.
$$

源码根据车宽 $w$、车长分段 $\Delta l=L/N_c$ 取圆半径

$$
R_c=\frac12\sqrt{w^2+\Delta l^2}-r_{map},
$$

其中地图分辨率 $r_{map}=0.2$ m 的减项用于离散栅格补偿。

### 6.2 双线性查询为什么可微

将圆心投影到 ESDF 网格的连续坐标 $(u,v)$。设 $u_0=\lfloor u\rfloor,v_0=\lfloor v\rfloor$，$a=u-u_0,b=v-v_0$，则距离为

$$
d(u,v)=(1-a)(1-b)D_{00}+a(1-b)D_{10}+(1-a)bD_{01}+abD_{11}.
$$

在每个网格单元内部，关于 $u$ 的导数是

$$
\frac{\partial d}{\partial u}
=(1-b)(D_{10}-D_{00})+b(D_{11}-D_{01}),
$$

关于 $v$ 同理。因此梯度可以沿

$$
\mathcal L_{aux}\rightarrow d\rightarrow c_{t,i}
\rightarrow(p_t,\theta_t)\rightarrow\phi
$$

反传到网络，不需要把每个候选轨迹渲染成整张图。

论文形式为

$$
\mathcal L_{aux}=\frac1{T_F}\sum_{t=1}^{T_F}\sum_{i=1}^{N_c}
\max(0,R_c+\epsilon-d_{t,i}).
$$

源码实现仅对网格有效范围内且 $R_c-d>0$ 的圆计损失，并按违规圆数归一化。这是论文公式和公开代码之间一个应明确记录的实现差异。

## 7. 对比模仿学习（CIL）

对每个原样本 $x$ 构造正样本 $x^+$ 与改变因果结构的负样本 $x^-$，共享编码器后得到归一化表示 $z,z^+,z^-$。相似度为余弦相似度

$$
s^+=z^\top z^+,\qquad s^-=z^\top z^-.
$$

两类 InfoNCE 损失为

$$
\mathcal L_c=-\log\frac{e^{s^+/\sigma}}{e^{s^+/\sigma}+e^{s^-/\sigma}}.
$$

把分子分母同除 $e^{s^+/\sigma}$：

$$
\mathcal L_c=\log\left(1+e^{(s^--s^+)/\sigma}\right)
=\operatorname{softplus}\left(\frac{s^--s^+}{\sigma}\right).
$$

于是

$$
\frac{\partial\mathcal L_c}{\partial s^+}
=-\frac1\sigma\operatorname{sigmoid}\left(\frac{s^--s^+}{\sigma}\right),
$$

$$
\frac{\partial\mathcal L_c}{\partial s^-}
=\frac1\sigma\operatorname{sigmoid}\left(\frac{s^--s^+}{\sigma}\right).
$$

结论非常直观：困难三元组（负样本比正样本更像原样本）获得更大梯度。源码温度 $\sigma=0.1$。

正增强包括轻微状态扰动、非交互参与者删除；负增强包括删除/插入前车、删除交互参与者、反转信号灯。负样本的原轨迹可能已无效，所以只参与 $\mathcal L_c$，不参与模仿监督。

论文把总目标概括为

$$
\mathcal L=w_1\mathcal L_i+w_2\mathcal L_p+w_3\mathcal L_{aux}+w_4\mathcal L_c.
$$

公开训练器实际直接等权相加：

$$
\mathcal L=\mathcal L_{reg}+\mathcal L_{cls}+\mathcal L_p+
\mathcal L_c+\mathcal L_{collision}+\mathcal L_{ref-free}.
$$

因此若自行实现 $w_i$，那是实验改动，不应称为原仓库默认行为。

## 8. 从候选到执行：动力学与混合评分

先按网络置信度保留 Top-$K$，论文和配置默认 $K=20$。每个候选不是直接拿来打分，而是交给 LQR tracker 和车辆模型前向模拟。简化的运动学自行车离散式为

$$
x_{t+1}=x_t+v_t\cos\psi_t\Delta t,
$$

$$
y_{t+1}=y_t+v_t\sin\psi_t\Delta t,
$$

$$
\psi_{t+1}=\psi_t+\frac{v_t}{L}\tan\delta_t\Delta t,
\qquad v_{t+1}=v_t+a_t\Delta t.
$$

LQR 通过最小化

$$
J=\sum_t(e_t^\top Qe_t+u_t^\top Ru_t)+e_T^\top Q_Fe_T
$$

求反馈 $u_t=-K_te_t$，使车辆跟踪候选。这样规则评估针对“控制器真正能走出的 rollout”，而不是网络理想曲线。

规则评分检查碰撞、TTC、可行驶区域、方向、进度、舒适、限速等。最后

$$
\pi_i^{final}=\pi_i^{rule}+\alpha\pi_i^{learn},\qquad \alpha=0.3,
$$

$$
i^*=\arg\max_i\pi_i^{final}.
$$

注意后处理只选择，不优化或修改原候选。紧急制动模块是例外：检测到紧急碰撞条件时可替换为制动轨迹。

## 9. 论文消融如何支持这些设计

论文在独立的 14 类、每类 20 场景子集上报告：基础模型 87.04；加入 SDE 为 89.64；加入辅助损失为 90.03；加入无参考线 head 为 90.69；加入 CIL 为 91.66；再加入后处理为 93.57。它说明提升来自一条组合链，而非单个“大模型”模块。

纵向 query 数 $N_L=12$ 得 91.66，继续加到 24 反而降到 87.90：模式多并不自动等于多样性好，冗余 query 会增加匹配和优化难度。用学习预测做 TTC 评估为 93.57，常速度预测为 92.82。这些都是论文范围内的多场景结果，与本仓库单场景 0.9498 分数口径不同。

## 10. 一句话心智模型

PLUTO 不是“图像进、方向盘出”的感知驾驶大模型，而是一个以真值目标、对象轨迹和 HD map 为输入的闭环规划器：向量化场景编码提供交互理解，参考线 × 纵向 query 提供多模态候选，CIL 抑制错误因果捷径，显式动力学和规则评分把学习输出约束为更可执行的决定。

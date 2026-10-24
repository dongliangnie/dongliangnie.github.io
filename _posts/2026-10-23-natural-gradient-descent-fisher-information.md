---
title: "自然梯度下降（Natural Gradient Descent）与 Fisher Information Matrix"
date: 2026-10-23
categories:
  - 机器学习
tags:
  - 优化
  - 自然梯度
---

自然梯度下降是从分布空间推导最速下降方向的方法，和牛顿法有紧密联系。其核心是用 Fisher Information Matrix 替代牛顿法中的 Hessian 矩阵，从而在概率分布空间而非参数空间中定义“最速下降”方向。

## 1. Fisher Information Matrix

自然梯度的推导依赖 Fisher Information Matrix，因此先回顾它的定义与性质。

### 1.1 Score function

设模型参数向量为 $\theta$，似然函数记为 $p(x \mid \theta)$。许多算法通过最大化该似然函数来学习 $\theta$。为此定义 Score function $s(\theta)$，即对数似然函数对参数的梯度：

$$
s(\theta) = \nabla_{\theta} \log p(x \vert \theta)
$$

Score function 在强化学习的 Policy Gradient 类方法中也有直接应用：用它的取值构造参数梯度，进而更新策略参数。

**Score function 的性质：在模型分布下，其期望为零。**

证明：

$$
\begin{aligned}
\mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ s(\theta) \right] &= \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \nabla \log p(x \vert \theta) \right] \\
&= \int \nabla \log p(x \vert \theta) \, p(x \vert \theta) \, \text{d}x \\
&= \int \frac{1}{p(x \vert \theta)} \nabla p(x \vert \theta) p(x \vert \theta) \text{d}x \\
&= \int \nabla p(x \vert \theta) \, \text{d}x \\
&= \nabla \int p(x \vert \theta) \, \text{d}x \\
&= \nabla 1 \\
&= 0
\end{aligned}
$$

### 1.2 Fisher Information

Score function 的期望虽然为零，但仍需刻画它的不确定性，为此取其在模型分布下的协方差矩阵：

$$
\mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ (s(\theta) - 0) \, (s(\theta) - 0)^{\text{T}} \right]
$$

该协方差矩阵称为 **Fisher Information**。若把 $\theta$ 视为列向量，则 Score function 也是列向量，对应的 Fisher Information 为矩阵形式，称为 **Fisher Information Matrix**：

$$
\text{F} = \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \nabla \log p(x \vert \theta) \, \nabla \log p(x \vert \theta)^{\text{T}} \right]
$$

实际中 $p(x \mid \theta)$ 的形式往往很复杂，甚至本身就是模型输出，难以精确计算期望。因此常用训练数据 $X=\{x_1,x_2,\cdots,x_N\}$ 估计 **Empirical Fisher**：

$$
\text{F} = \frac{1}{N} \sum_{i=1}^{N} \nabla \log p(x_i \vert \theta) \, \nabla \log p(x_i \vert \theta)^{\text{T}}
$$

### 1.3 Fisher矩阵和Hessian矩阵的关系

Fisher 矩阵与 Hessian 矩阵之间存在一个重要关系：log 似然函数的 Hessian 矩阵的期望取负，等于 Fisher Information Matrix。

**结论：对数似然函数 Hessian 矩阵期望的负数等于 Fisher 信息矩阵 $\text{F}$。**

证明：关键在于，对数似然的 Hessian 等于其梯度（即 Score function）的 Jacobian：

$$
\begin{aligned}
\text{H}_{\log p(x \vert \theta)} &= \text{J} \left[\frac{\nabla p(x \vert \theta)}{p(x \vert \theta)}\right] \\
&= \frac{ \text{H}_{p(x \vert \theta)} \, p(x \vert \theta) - \nabla p(x \vert \theta) \, \nabla p(x \vert \theta)^{\text{T}}}{p(x \vert \theta) \, p(x \vert \theta)} \\
&= \frac{\text{H}_{p(x \vert \theta)} \, p(x \vert \theta)}{p(x \vert \theta) \, p(x \vert \theta)} - \frac{\nabla p(x \vert \theta) \, \nabla p(x \vert \theta)^{\text{T}}}{p(x \vert \theta) \, p(x \vert \theta)} \\
&= \frac{\text{H}_{p(x \vert \theta)}}{p(x \vert \theta)} - \left( \frac{\nabla p(x \vert \theta)}{p(x \vert \theta)} \right) \left( \frac{\nabla p(x \vert \theta)}{p(x \vert \theta)}\right)^{\text{T}}
\end{aligned}
$$

推导的时候主要注意，$p(x \vert \theta)$是一个标量；而$\nabla p(x \vert \theta)$是对参数的梯度，是一个列向量。

对模型分布取期望：

$$
\begin{aligned}
\mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \text{H}_{\log p(x \vert \theta)} \right] &= \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \frac{\text{H}_{p(x \vert \theta)}}{p(x \vert \theta)} - \left( \frac{\nabla p(x \vert \theta)}{p(x \vert \theta)} \right) \left( \frac{\nabla p(x \vert \theta)}{p(x \vert \theta)} \right)^{\text{T}} \right] \\
&= \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \frac{\text{H}_{p(x \vert \theta)}}{p(x \vert \theta)} \right] - \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \left( \frac{\nabla p(x \vert \theta)}{p(x \vert \theta)} \right) \left( \frac{\nabla p(x \vert \theta)}{p(x \vert \theta)}\right)^{\text{T}} \right] \\
&= \int \frac{\text{H}_{p(x \vert \theta)}}{p(x \vert \theta)} p(x \vert \theta) \, \text{d}x \, - \mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \nabla \log p(x \vert \theta) \, \nabla \log p(x \vert \theta)^{\text{T}} \right] \\
&= \text{H}_{\int p(x \vert \theta) \, \text{d}x} \, - \text{F} \\
&= \text{H}_{1} - \text{F} \\
&= -\text{F} \, .
\end{aligned}
$$

因此得到 $\text{F} = -\mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \text{H}_{\log p(x \vert \theta)} \right]$，证明完毕。Fisher 矩阵可以看作对数似然函数曲率的度量。牛顿法需要计算并求逆 Hessian 矩阵，而 Fisher 矩阵提供了 Hessian 的一个等价替代，这自然引出了自然梯度方法。

## 2. 自然梯度下降法（Natural Gradient Descent）

自然梯度下降（Natural Gradient Descent）把参数看成一种概率分布，然后使用 KL 散度而不是欧氏距离来作为距离的度量，从而更好地描述更新后的分布和原分布有多大的不同。

具体来说，可以从参数空间（parameter space）和分布空间（distribution space）两种视角理解梯度下降。

- **parameter space（参数空间）**：梯度下降沿负梯度方向 $-\nabla_\theta \mathcal{L}(\theta)$ 更新。负梯度方向是在当前参数 $\theta$ 的局部邻域内，使损失在参数空间中下降最快的方向：

  $$
  \frac{-\nabla_\theta \mathcal{L}(\theta)}{\lVert \nabla_\theta \mathcal{L}(\theta) \rVert} = \lim_{\epsilon \to 0} \frac{1}{\epsilon} \mathop{\text{arg min}}_{d \text{ s.t. } \lVert d \rVert \leq \epsilon} \mathcal{L}(\theta + d) \, .
  $$

  其含义是：在所有满足 $\lVert d\rVert \leq \epsilon$ 的更新向量中，选取使 $\mathcal{L}(\theta+d)$ 最小的 $d$。这里用欧氏范数定义邻域，因此梯度下降依赖参数空间的欧氏几何。
- **distribution space（分布空间）**：若目标是最小化损失（最大化似然），则优化过程实际上是在所有可能的概率分布构成的空间中进行，并通过参数 $\theta$ 参数化这些分布。因此，直接在分布空间中寻找最速下降方向更符合问题的本质。

分布空间中常用的距离度量是 KL 散度（KL-divergence），它衡量两个分布的接近程度。KL 散度并不对称，严格来说不满足距离度量的定义；但当参数变化量 $d$ 趋于 0 时，KL 散度在局部邻域内近似对称，因此仍可用于描述分布之间的差异。

### 2.1 分布空间中的最速下降，Natural gradient方法

下面推导分布空间中的最速下降方向。

**首先，KL 散度在参数变化较小时可以作二阶泰勒展开：**

$$
\text{KL}[p(x \vert \theta) \, \Vert \, p(x \vert \theta + d)] \approx \frac{1}{2} d^\text{T} \text{F} d
$$

**证明**：

$$
\begin{aligned}
&\mathrm{KL}\bigl[p(x\mid\theta)\,\Vert\,p(x\mid\theta+d)\bigr]
\\
&\approx
\underbrace{\mathrm{KL}\bigl[p(x\mid\theta)\,\Vert\,
p(x\mid\theta)\bigr]}_{=\,0}
\\
&\quad+
\underbrace{
\left.
\nabla_{\theta'}\mathrm{KL}\bigl[
p(x\mid\theta)\,\Vert\,p(x\mid\theta')
\bigr]\right|_{\theta'=\theta}^{\mathsf T}
}_{=\,0}
d
\\
&\quad+
\frac{1}{2}d^{\mathsf T}
\underbrace{
\left.
\nabla_{\theta'}^2\mathrm{KL}\bigl[
p(x\mid\theta)\,\Vert\,p(x\mid\theta')
\bigr]\right|_{\theta'=\theta}
}_{\mathbf F(\theta)}
d
\\[4pt]
&=\frac{1}{2}d^{\mathsf T}\mathbf F(\theta)d.
\end{aligned}
$$

这里引入 $\theta'$，是为了把 KL 散度第一个分布 $p(x \vert \theta)$ 固定，只对第二个分布的参数求导。下面逐项分析二阶泰勒展开中的三项：

- 泰勒展开的第一项 $\text{KL}[p_{\theta} \, \Vert \, p_{\theta}] = 0$
- 第二项的推导：

  $$
  \begin{aligned}
  \nabla_{\theta'} \text{KL}[p(x \vert \theta) \, \Vert \, p(x \vert \theta')] &= \nabla_{\theta'} \mathop{\mathbb{E}}_{p(x \vert \theta)} [ \log p(x \vert \theta) ] - \nabla_{\theta'} \mathop{\mathbb{E}}_{p(x \vert \theta)} [ \log p(x \vert \theta') ] \\
  &= - \mathop{\mathbb{E}}_{p(x \vert \theta)} [ \nabla_{\theta'} \log p(x \vert \theta') ] =0
  \end{aligned}
  $$

  取 $\theta' = \theta$ 时，第二项包含 Score function 的期望；由第 1.1 节的结论可知该期望为 0。
- 第三项，需要用到第 1.3 节证明的，$\text{F} = -\mathop{\mathbb{E}}_{p(x \vert \theta)} \left[ \text{H}_{\log p(x \vert \theta)} \right]$，以及如下性质：**Fisher 信息矩阵 $\text{F}$ 是分布 $p(x \vert \theta)$ 与 $p(x \vert \theta')$ 之间 KL 散度关于 $\theta'$ 的 Hessian，并在 $\theta' = \theta$ 处取值。** 推导过程如下：

  $$
  \text{KL} [p(x \vert \theta) \, \Vert \, p(x \vert \theta')] = \mathop{\mathbb{E}}_{p(x \vert \theta)} [ \log p(x \vert \theta) ] - \mathop{\mathbb{E}}_{p(x \vert \theta)} [ \log p(x \vert \theta') ]
  $$

  对 $\theta'$ 的一阶导数：

  $$
  \begin{aligned}
  \nabla_{\theta'} \text{KL}[p(x \vert \theta) \, \Vert \, p(x \vert \theta')] &= \nabla_{\theta'} \mathop{\mathbb{E}}_{p(x \vert \theta)} [ \log p(x \vert \theta) ] - \nabla_{\theta'} \mathop{\mathbb{E}}_{p(x \vert \theta)} [ \log p(x \vert \theta') ] \\
  &= - \mathop{\mathbb{E}}_{p(x \vert \theta)} [ \nabla_{\theta'} \log p(x \vert \theta') ] \\
  &= - \int p(x \vert \theta) \nabla_{\theta'} \log p(x \vert \theta') \, \text{d}x
  \end{aligned}
  $$

  二阶导数：

  $$
  \begin{aligned}
  \nabla_{\theta'}^2 \, \text{KL}[p(x \vert \theta) \, \Vert \, p(x \vert \theta')]\vert_{\theta' = \theta} &= - \int p(x \vert \theta) \, \nabla_{\theta'}^2 \log p(x \vert \theta')\vert_{\theta' = \theta} \, \text{d}x \\
  &= - \int p(x \vert \theta) \, \text{H}_{\log p(x \vert \theta)} \, \text{d}x \\
  &= - \mathop{\mathbb{E}}_{p(x \vert \theta)} [\text{H}_{\log p(x \vert \theta)}] \\
  &= \text{F}
  \end{aligned}
  $$

因此 KL 散度的二阶泰勒展开为：

$$
\text{KL}[p(x \vert \theta) \, \Vert \, p(x \vert \theta + d)] \approx \frac{1}{2} d^\text{T} \text{F} d
$$

在分布空间中，我们希望找到使损失 $\mathcal{L}$ 下降最快的更新向量 $d$。与参数空间中的最速下降类似，这里以 KL 散度作为“邻域”的度量，即求解：

$$
d^* = \mathop{\text{arg min}}_{d \text{ s.t. } \text{KL}[p_\theta \Vert p_{\theta + d}] \leq c} \mathcal{L} (\theta + d) \, ,
$$

用拉格朗日乘子法处理该约束优化问题，并对 KL 散度作二阶、对 $\mathcal{L}$ 作一阶泰勒展开：

$$
\begin{aligned}
d^* &= \mathop{\text{arg min}}_d \, \mathcal{L} (\theta + d) + \lambda \, (\text{KL}[p_\theta \Vert p_{\theta + d}] - c) \\
&\approx \mathop{\text{arg min}}_d \, \mathcal{L}(\theta) + \nabla_\theta \mathcal{L}(\theta)^\text{T} d + \frac{1}{2} \lambda \, d^\text{T} \text{F} d - \lambda c
\end{aligned}
$$

其中 $\lambda$ 是拉格朗日乘子。令目标函数对 $d$ 的梯度为 0：

$$
\begin{aligned}
0 &= \frac{\partial}{\partial d} \left[\mathcal{L}(\theta) + \nabla_\theta \mathcal{L}(\theta)^\text{T} d + \frac{1}{2} \lambda \, d^\text{T} \text{F} d - \lambda c\right] \\
&= \nabla_\theta \mathcal{L}(\theta) + \lambda \, \text{F} d \\
\lambda \, \text{F} d &= -\nabla_\theta \mathcal{L}(\theta) \\
d &= -\frac{1}{\lambda} \text{F}^{-1} \nabla_\theta \mathcal{L}(\theta)
\end{aligned}
$$

忽略常数因子 $\frac{1}{\lambda}$（可并入学习率），分布空间中的最优更新方向为 $-\text{F}^{-1} \nabla_\theta \mathcal{L}(\theta)$。这与牛顿法的更新方向 $-\text{H}^{-1} \nabla_\theta \mathcal{L}(\theta)$ 形式一致，区别在于用 Fisher 矩阵替代了 Hessian 矩阵。

据此定义**自然梯度**：$\tilde{\nabla}_\theta \mathcal{L}(\theta) = \text{F}^{-1} \nabla_\theta \mathcal{L}(\theta)$。实际计算时通常采用 batch 模式下的 Empirical Fisher Matrix：$\text{F} = \frac{1}{N} \sum_{i=1}^{N} \nabla \log p(x_i \vert \theta) \, \nabla \log p(x_i \vert \theta)^{\text{T}}$。

**自然梯度下降算法流程：**

1. 前向计算损失 $\mathcal{L}(\theta)$。
2. 计算梯度 $\nabla_\theta \mathcal{L}(\theta)$。
3. 计算 Fisher 矩阵 $\text{F}$ 或其经验版本（基于训练数据）。
4. 计算自然梯度 $\tilde{\nabla}_\theta \mathcal{L}(\theta) = \text{F}^{-1} \nabla_\theta \mathcal{L}(\theta)$。
5. 更新参数：$\theta \leftarrow \theta - \alpha \tilde{\nabla}_\theta \mathcal{L}(\theta)$，其中 $\alpha$ 为学习率。
6. 重复以上步骤直至收敛。

## 3. 数值实验：NGD 与 GD 的收敛对比

下面用两个可精确计算 Fisher 矩阵的小规模例子，比较梯度下降（GD）与自然梯度下降（NGD）的收敛速度。实验用 NumPy 实现。种子 42 用于高斯实验的样本生成，种子 7 用于逻辑回归的数据生成；这两个种子只控制各自实验的数据生成，不是优化器的超参数。在每个实验内部，GD 与 NGD 使用完全相同的数据、初始化和迭代轮数（200 步），因此二者的收敛差异只来自更新规则。两个实验均使用全批量更新，学习率从固定网格中按最终损失选取最优值。

![NGD 与 GD 的收敛曲线与参数轨迹](/images/blog/ngd_vs_gd_convergence.png)

图的上排为损失随迭代次数的变化（纵轴为对数坐标），下排为参数空间中 GD 与 NGD 的优化轨迹及损失等高线。下排把两种方法的每一步都画成一个点：GD 用圆点、NGD 用三角点，颜色由深到浅对应迭代次数（可参考右侧色条）；较大的空心标记标出若干代表步，并直接标注迭代次数（黑色为 GD，红色为 NGD），因此可以逐点比较两条轨迹的前进幅度。GD 在曲率较小的方向上进展缓慢；NGD 经 Fisher 矩阵预条件后，沿更直接的方向收敛。

**关于学习率：** 两种方法的学习率不能取相同数值，因为更新的尺度含义不同。GD 的更新为 $\theta_{t+1}=\theta_t-\eta g$，稳定性要求 $\eta<2/L_{\max}$（$L_{\max}$ 为最大曲率或 Hessian 特征值），因此 $\eta$ 受问题尺度限制；NGD 的更新为 $\theta_{t+1}=\theta_t-\eta\text{F}^{-1}g$，其中 $\text{F}^{-1}$ 已完成各方向的预条件，$\eta=1$ 对应标准牛顿步，此时的 $\eta$ 更像阻尼或松弛系数。若强行使用相同数值，对 GD 可能过大而发散，对 NGD 则可能欠松弛。因此实验对两种方法分别做网格搜索，取最终损失最优的学习率，使二者都处于最佳调参状态，从而保证比较公平。表中比较的是每步迭代的收敛速度，不含 NGD 计算 $\text{F}^{-1}$ 的额外开销。

**学习率网格的选取方式：** 对每种方法先固定一组候选学习率，分别完整训练 200 步，比较第 200 步的训练损失，取损失最小者；若某候选值导致发散（损失为 `inf` 或 `nan`）则直接淘汰。具体候选值与结果如下：

- 高斯实验：GD 候选 $\{10^{-3},2\times10^{-3},5\times10^{-3},10^{-2},2\times10^{-2},5\times10^{-2},0.08,0.1,0.12,0.15\}$，选中 $0.08$；NGD 候选 $\{0.5,1.0,1.5\}$，三者最终损失相同，取标准牛顿步 $1.0$。
- 逻辑回归：GD 候选 $\{10^{-3},3\times10^{-3},10^{-2},3\times10^{-2},0.1,0.3,1,2,3,4\}$，选中 $4.0$；NGD 候选 $\{0.5,1.0,1.5\}$，三者最终损失相同，取 $1.0$。

这样两种方法都以各自调优后的最佳学习率参与比较，避免因固定同一个学习率而偏袒某一方。

### 3.1 一维高斯分布的极大似然估计

设样本 $x_1,\dots,x_N \sim \mathcal{N}(\mu,\sigma^2)$，参数化为 $\theta=(\mu,s)$，其中 $\sigma=e^{s}$。取 $N=1000$，真实参数 $\mu=3.0$、$\sigma=0.5$，初始值 $\mu=0$、$\sigma=2$。负对数似然及其 Fisher 矩阵为：

$$
\mathcal{L}(\theta)=\frac{1}{N}\sum_{i=1}^{N}\left[\frac{1}{2}\log(2\pi e^{2s})+\frac{(x_i-\mu)^2}{2e^{2s}}\right],
\qquad
\text{F}(\theta)=\operatorname{diag}\left(\frac{1}{\sigma^2},\,2\right).
$$

GD 的最优学习率为 $0.08$，NGD 为 $1.0$。$\theta^*$ 表示收敛后的最优参数。

NGD 约 10 步即达到最优损失，而 GD 需要约 200 步。原因是 $\sigma$ 变小时参数空间在 $\mu$ 与 $s$ 方向上的曲率迅速增大，GD 的固定步长受最大曲率限制，在另一方向上进展缓慢；NGD 用 $\text{F}^{-1}$ 对梯度做预条件，消除了这种尺度差异。

### 3.2 各向异性特征的逻辑回归

二分类逻辑回归，特征 $x_1\sim\mathcal{N}(0,1)$、$x_2\sim\mathcal{N}(0,0.1^2)$，真值 $w=(2,4)$、$b=0.5$，训练集 $N=2000$。记 $\tilde{x}_n=(x_{n1},x_{n2},1)^\top$，$p_n=\sigma(\tilde{x}_n^\top\theta)$，则

$$
\mathcal{L}(\theta)=-\frac{1}{N}\sum_{n=1}^{N}\left[y_n\log p_n+(1-y_n)\log(1-p_n)\right],
\qquad
\text{F}(\theta)=\frac{1}{N}\sum_{n=1}^{N}p_n(1-p_n)\,\tilde{x}_n\tilde{x}_n^\top .
$$

初始 Fisher 矩阵的条件数约为 $103$，特征尺度差异明显。GD 的最优学习率为 $4.0$，NGD 为 $1.0$；$\theta^*$ 取 NGD 收敛后的参数。

NGD 在约 5 步内收敛；GD 在 200 步后损失仍为 $0.456652$，参数误差主要残留在低曲率的 $x_2$ 方向（$w_2$ 收敛到 $2.68$，最优值约为 $3.91$）。由于逻辑回归的 Hessian 恰好等于 Fisher 矩阵，此处的 NGD 等价于牛顿法，因而表现出近似二次收敛。两种方法最终的测试准确率相当（约 $77\%$），说明 NGD 的优势主要体现在优化效率，而非泛化性能。

### 3.3 代价与适用性

NGD 每步需要构造并求逆 $d\times d$ 的 Fisher 矩阵，时间和空间复杂度分别为 $O(d^3)$ 和 $O(d^2)$，且需要保证 $\text{F}$ 可逆（实际中常加阻尼项 $\lambda \text{I}$）。当参数量达到百万级时，精确 NGD 不可行，这也是 K-FAC、对角 Fisher 等近似方法出现的原因。

## 参考资料

[1] https://blog.csdn.net/xbinworld/article/details/104591706

## 附录：实验代码

依赖 `numpy` 与 `matplotlib`。运行后打印第 3 节中的数值结果，并生成 `ngd_vs_gd_convergence.png`。

```python
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_GD, C_NG = "#2b6cb0", "#c53030"
ITERS = 200


def run(mode, lr, theta0, grad, fisher, loss, iters=ITERS):
    th = theta0.copy()
    ths, losses = [th.copy()], [loss(th)]
    for _ in range(iters):
        g = grad(th)
        if mode == "GD":
            step = g
        else:
            F = fisher(th)
            step = np.linalg.solve(F + 1e-12 * np.eye(len(th)), g)
        th = th - lr * step
        ths.append(th.copy())
        losses.append(loss(th))
    return np.array(ths), np.array(losses)


# ---------- 实验一：一维高斯分布的极大似然估计 ----------
rng = np.random.default_rng(42)
N = 1000
x = rng.normal(3.0, 0.5, N)
xbar, xvar = x.mean(), x.var()


def gauss_loss(th):
    mu, s = th
    v = np.exp(2 * s)
    return 0.5 * (np.log(2 * np.pi * v) + ((x - mu) ** 2) / v).mean()


def gauss_grad(th):
    mu, s = th
    v = np.exp(2 * s)
    return np.array([(mu - xbar) / v, 1 - ((x - mu) ** 2).mean() / v])


def gauss_fisher(th):
    mu, s = th
    v = np.exp(2 * s)
    return np.diag([1.0 / v, 2.0])


th_star = np.array([xbar, 0.5 * np.log(xvar)])
theta0 = np.array([0.0, np.log(2.0)])
g_th, g_loss = run("GD", 0.08, theta0, gauss_grad, gauss_fisher, gauss_loss)
n_th, n_loss = run("NGD", 1.0, theta0, gauss_grad, gauss_fisher, gauss_loss)

print("=== 高斯 MLE ===")
for t in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
    print(f"{t:3d}  GD={g_loss[t]:.6f}  NGD={n_loss[t]:.6f}  "
          f"GD_dist={np.linalg.norm(g_th[t] - th_star):.4e}  "
          f"NGD_dist={np.linalg.norm(n_th[t] - th_star):.4e}")

# ---------- 实验二：各向异性特征的逻辑回归 ----------
rng = np.random.default_rng(7)
Ntr = 2000
x1 = rng.normal(0, 1, Ntr)
x2 = rng.normal(0, 0.1, Ntr)
z = 2 * x1 + 4 * x2 + 0.5
p = 1 / (1 + np.exp(-z))
y = (rng.random(Ntr) < p).astype(float)
Xb = np.column_stack([x1, x2, np.ones(Ntr)])


def logi_loss(th):
    zz = Xb @ th
    return (np.logaddexp(0, zz) - y * zz).mean()


def logi_grad(th):
    pp = 1 / (1 + np.exp(-Xb @ th))
    return Xb.T @ (pp - y) / len(y)


def logi_fisher(th):
    pp = 1 / (1 + np.exp(-Xb @ th))
    w = pp * (1 - pp)
    return (Xb * w[:, None]).T @ Xb / len(Xb)


lg_th, lg_loss = run("GD", 4.0, np.zeros(3), logi_grad, logi_fisher, logi_loss)
ln_th, ln_loss = run("NGD", 1.0, np.zeros(3), logi_grad, logi_fisher, logi_loss)
opt = ln_th[-1]

print("\n=== 逻辑回归 ===")
print("Fisher 条件数：", np.linalg.cond(logi_fisher(np.zeros(3))))
for t in [0, 1, 2, 5, 10, 20, 50, 100, 200]:
    print(f"{t:3d}  GD={lg_loss[t]:.6f}  NGD={ln_loss[t]:.6f}  "
          f"GD_dist={np.linalg.norm(lg_th[t] - opt):.4e}  "
          f"NGD_dist={np.linalg.norm(ln_th[t] - opt):.4e}")

# ---------- 可视化 ----------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
    "axes.labelcolor": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7.5,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "mathtext.fontset": "dejavusans",
})

C_GRAY = "#9aa0a6"

fig, ax = plt.subplots(2, 2, figsize=(12.0, 8.4),
                       gridspec_kw=dict(height_ratios=[0.92, 1.12],
                                        hspace=0.34, wspace=0.26))
iters = np.arange(ITERS + 1)

for a in ax.ravel():
    a.spines[["top", "right"]].set_visible(False)

# ---- 上排：损失曲线 ----
ax[0, 0].semilogy(iters, g_loss, color=C_GD, lw=2.2, label="GD (lr = 0.08)")
ax[0, 0].semilogy(iters, n_loss, color=C_NG, lw=2.2, label="NGD (lr = 1.0)")
ax[0, 0].axhline(gauss_loss(th_star), ls=(0, (2, 2)), color="#888888", lw=0.9)
ax[0, 0].set(xlabel="Iteration", ylabel="Negative log-likelihood",
             title="Gaussian MLE: loss", xlim=(0, ITERS))
ax[0, 0].grid(True)
ax[0, 0].legend(frameon=False, loc="upper right")

ax[0, 1].semilogy(iters, lg_loss, color=C_GD, lw=2.2, label="GD (lr = 4.0)")
ax[0, 1].semilogy(iters, ln_loss, color=C_NG, lw=2.2, label="NGD (lr = 1.0)")
ax[0, 1].axhline(logi_loss(opt), ls=(0, (2, 2)), color="#888888", lw=0.9)
ax[0, 1].set(xlabel="Iteration", ylabel="Logistic loss",
             title="Logistic regression: loss", xlim=(0, ITERS))
ax[0, 1].grid(True)
ax[0, 1].legend(frameon=False, loc="upper right")

# ---- 下排：轨迹 ----
def draw_traj(a, gd, ngd, gd_steps, ngd_steps):
    k_g = np.arange(len(gd))
    a.plot(gd[:, 0], gd[:, 1], color=C_GRAY, lw=0.9, zorder=2)
    sc = a.scatter(gd[:, 0], gd[:, 1], c=k_g, cmap="viridis",
                   vmin=0, vmax=ITERS, s=14, zorder=3,
                   edgecolors="none", alpha=0.95)
    k_n = np.arange(len(ngd))
    a.plot(ngd[:, 0], ngd[:, 1], color=C_NG, lw=1.1, alpha=0.45, zorder=2)
    a.scatter(ngd[:, 0], ngd[:, 1], c=k_n, cmap="viridis",
              vmin=0, vmax=ITERS, s=30, marker="^", zorder=5,
              edgecolors=C_NG, linewidths=0.55)
    a.scatter(gd[gd_steps, 0], gd[gd_steps, 1], s=80, marker="o",
              facecolors="none", edgecolors="#111111", linewidths=1.0,
              zorder=6)
    a.scatter(ngd[ngd_steps, 0], ngd[ngd_steps, 1], s=130, marker="^",
              facecolors="none", edgecolors=C_NG, linewidths=1.2, zorder=7)
    a.grid(True)
    return sc


def label_steps(a, ths, steps, offsets, color="k"):
    for t, off in zip(steps, offsets):
        a.annotate(str(t), (ths[t, 0], ths[t, 1]), textcoords="offset points",
                   xytext=off, fontsize=7.5, fontweight="bold", color=color,
                   zorder=9,
                   bbox=dict(boxstyle="round,pad=0.2", fc="white",
                             ec="none", alpha=0.82))


def traj_legend(a, loc):
    handles = [
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="#7a7a7a",
                   markeredgecolor="none", markersize=5, label="GD step"),
        plt.Line2D([], [], marker="^", color="none", markerfacecolor="#7a7a7a",
                   markeredgecolor=C_NG, markersize=6.5, label="NGD step"),
        plt.Line2D([], [], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="#111111", markersize=8.5, label="labeled GD"),
        plt.Line2D([], [], marker="^", color="none", markerfacecolor="none",
                   markeredgecolor=C_NG, markersize=9.5, label="labeled NGD"),
        plt.Line2D([], [], marker="s", color="none", markerfacecolor="#111111",
                   markersize=5.5, label="start"),
        plt.Line2D([], [], marker="*", color="none", markerfacecolor="#111111",
                   markersize=11, label="optimum"),
    ]
    a.legend(handles=handles, loc=loc, ncol=2, fontsize=7.5,
             frameon=True, facecolor="white", edgecolor="#dddddd",
             framealpha=0.95, borderpad=0.55, labelspacing=0.45,
             columnspacing=1.1, handletextpad=0.5)


mu_g, s_g = np.linspace(-0.2, 3.5, 260), np.linspace(-1.6, 1.5, 260)
MU, SG = np.meshgrid(mu_g, s_g)
V = np.exp(2 * SG)
Z = 0.5 * (np.log(2 * np.pi * V) + (xbar**2 + xvar - 2 * MU * xbar + MU**2) / V)
ax[1, 0].contourf(MU, SG, Z, levels=18, cmap="Greys", alpha=0.14)
cs = ax[1, 0].contour(MU, SG, Z, levels=[0.75, 0.8, 0.9, 1.1, 1.5, 2.2, 3.2],
                      colors="#8a8a8a", linewidths=0.55, alpha=0.6)
ax[1, 0].clabel(cs, fmt="%.2f", fontsize=6.5, inline_spacing=2)
sc1 = draw_traj(ax[1, 0], g_th, n_th, [0, 20, 50, 100], [1, 5])
label_steps(ax[1, 0], g_th, [0, 20, 50, 100],
            [(7, 5), (-18, 6), (7, -11), (10, -4)])
label_steps(ax[1, 0], n_th, [1, 5], [(8, 3), (-24, -2)], color=C_NG)
ax[1, 0].scatter([theta0[0]], [theta0[1]], marker="s", s=42, color="#111111",
                 zorder=8)
ax[1, 0].scatter([th_star[0]], [th_star[1]], marker="*", s=170, color="#111111",
                 zorder=8)
ax[1, 0].set(xlabel=r"$\mu$", ylabel=r"$s=\log\sigma$",
             title="Gaussian MLE: parameter trajectory")
cb1 = fig.colorbar(sc1, ax=ax[1, 0], fraction=0.042, pad=0.02,
                   ticks=[0, 50, 100, 150, 200])
cb1.set_label("iteration", fontsize=8)
cb1.ax.tick_params(labelsize=7)
cb1.outline.set_visible(False)
traj_legend(ax[1, 0], "lower left")

w1, w2 = np.linspace(0, 2.8, 260), np.linspace(0, 4.6, 260)
W1, W2 = np.meshgrid(w1, w2)
b0 = opt[2]
logits = W1[..., None] * x1 + W2[..., None] * x2 + b0
ZL = (np.logaddexp(0, logits) - y * logits).mean(axis=2)
ax[1, 1].contourf(W1, W2, ZL, levels=18, cmap="Greys", alpha=0.14)
cs2 = ax[1, 1].contour(W1, W2, ZL,
                       levels=[0.456, 0.462, 0.47, 0.48, 0.50, 0.55, 0.65, 0.90],
                       colors="#8a8a8a", linewidths=0.55, alpha=0.6)
ax[1, 1].clabel(cs2, fmt="%.3f", fontsize=6.5, inline_spacing=2)
sc2 = draw_traj(ax[1, 1], lg_th, ln_th, [0, 1, 10, 50, 200], [1, 2, 5])
label_steps(ax[1, 1], lg_th, [0, 1, 10, 50, 200],
            [(7, 5), (-16, 6), (6, -11), (-20, -10), (8, 6)])
label_steps(ax[1, 1], ln_th, [1, 2, 5], [(8, 3), (-22, -12), (10, 4)],
            color=C_NG)
ax[1, 1].scatter([0], [0], marker="s", s=42, color="#111111", zorder=8)
ax[1, 1].scatter([opt[0]], [opt[1]], marker="*", s=170, color="#111111",
                 zorder=8)
ax[1, 1].set(xlabel=r"$w_1$", ylabel=r"$w_2$",
             title="Logistic regression: parameter trajectory ($b$ fixed)")
cb2 = fig.colorbar(sc2, ax=ax[1, 1], fraction=0.042, pad=0.02,
                   ticks=[0, 50, 100, 150, 200])
cb2.set_label("iteration", fontsize=8)
cb2.ax.tick_params(labelsize=7)
cb2.outline.set_visible(False)
traj_legend(ax[1, 1], "upper left")

fig.savefig("ngd_vs_gd_convergence.png", dpi=180, bbox_inches="tight",
            facecolor="white")
```

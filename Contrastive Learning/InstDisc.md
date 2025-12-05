# Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination

> 《Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination》
> 参考：李沐论文精度系列之[《对比学习论文综述》](https://www.bilibili.com/video/BV19S4y1M7hm/?vd_source=21011151235423b801d3f3ae98b91e94)、[精度笔记](https://www.bilibili.com/read/cv14700928?spm_id_from=333.999.0.0)
> 参考：[《对比学习一 |Instance Discrimination》](https://zhuanlan.zhihu.com/p/457986773)、[《Instance Discrimination论文阅读笔记》](https://blog.csdn.net/Nin7a/article/details/103020861?ops_request_misc=%7B%22request%5Fid%22%3A%22166720909816782427479081%22%2C%22scm%22%3A%2220140713.130102334..%22%7D&request_id=166720909816782427479081&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-4-103020861-null-null.142^v62^control,201^v3^control_1,213^v1^control&utm_term=instance discrimination&spm=1018.2226.3001.4187)、
>   这篇文章提出了个体判别任务（代理任务）以及`memory bank` ，非常经典，后人给它的方法起名为InstDisc。

## 研究动机

>  		在有监督学习的分类模型中，如果给一张豹子图片进行分类，会发现排前几名的都是跟这张图很像的图片，而排名靠后的那些往往是跟豹子一点关系都没有的类别。
>		
>  	  作者研究发现，让这些图片聚集在一起的原因并不是因为它们有相似的语义标签，而是因为这些照片里的物体都很相似。最后作者由此提出了个体判别任务：把每一个instance（实例，这里就是指每一张图）都看成是一个类别，目标是学一种特征，把每张图片都区分开来。

<img src="https://gitee.com/dongliangnie/typora-image-bed-2/raw/master/202512041406311.png" alt="在这里插入图片描述" style="zoom: 67%;" />

## 算法

### 模型结构

> 	将图片经过CNN网络编码后得到的图片特征，使用对比学习的方式将其在特征空间中尽可能的区分开来（因为每张图都是自己的类）。
>
>   既然是对比学习，就需要正负样本。InstDisc中正样本就是就是这个图片本身（可能经过一些数据增强），负样本就是数据集里所有其它的图片，这些负样本都存储在 memory bank里。对于ImageNet有128万张图片，那么memory bank就要存储128万行，所以最后每张图都用128维特征表示（维度太高存储不了）

![在这里插入图片描述](https://gitee.com/dongliangnie/typora-image-bed-2/raw/master/202512041406710.png)

### 前向过程

>+ ![image-20251203235900123](https://gitee.com/dongliangnie/typora-image-bed-2/raw/master/202512041406692.png)
>
>+ 论文的softmax不设置参数w。而是和Word2vec一样把特征当作参数，并创建一个叫做memory bank的堆进行存储所有单词的128维特征，每次通过loss更新。这样训练和测试通过存储的memory bank同使用一个度量空间。
>
>+ 论文取batch_size=256，则每个batch有256个正样本，然后从 memory bank 里随机地抽取4096个负样本。根据正负样本计算对比学习目标函数NCELoss。然后根据loss更新backbone和memory bank（把 minibatch里的数据样本所对应的那些特征，在 memory bank 里更换掉，这样无论是训练还是测试就都来自于一个度量空间了）。
>
>+ 测试时，使用KNN进行分类,我们获得了训练好的模型后，对于一张图片提取他的特征，将他和memorybank中所有的存储图片特征计算相似度，然后采用k近邻算法，返回最相似的k张图片。最后根据相似度权重投票，得到其类别c。

## 训练细节

  本文的一些超参数设定，比如backbone选择ResNet50，batch_size=256，负样本采样数为4096，特征维度dim=128，epoch=200，初始lr=0.03，计算NCELoss时τ=0.07；这些超参数在在MoCo 中也是沿用的，没有进行更改。
NCE Loss

### Noise-Contrastive Estimation

> ####  **Parametric Classifier（参数化分类器）**
>
> 在传统的参数化 Softmax 中，对于图像 $x$ 及其特征：
> $$
> v = f_\theta(x)
> $$
> 被识别为第 $i$ 类的概率为：
> $$
> P(i|v)=\frac{\exp(w_i^{T}v)}{\sum_{j=1}^n \exp(w_j^{T}v)}
> $$
> 其中：
>
> - $v$ 是卷积网络输出特征
> - $i$ 是预测类别（实例级）
> - $w_i$ 是第 $i$ 类的权重参数
> - 这是一个需要学习参数 $w$ 的分类器
>
> ####  **Non-Parametric Softmax（非参数化 Softmax 分类器）**
>
> 作者认为权重 $w$ 限制了实例之间的对比，因此提出无参版本：
>
> 将参数 $w_i$ 替换成 feature memory bank 中存储的特征向量 $v_i$：
> $$
> P(i|v)=\frac{\exp(v_i^{T}v / \tau)}{\sum_{j=1}^n \exp(v_j^{T}v / \tau)}
> $$
> 其中：
>
> - $\tau$ 是温度系数，控制分布的锐度
> - Memory Bank $V$ 用于存储所有 $v_j$
> - 在训练中，每次将旧的 $v_i$ 替换为新的 $f_i$
>
> #### **Noise-Contrastive Estimation（NCE）**
>
> Softmax 分母包含全部 $n$ 个样本，开销太大：
> $$
>  Z_i = \sum_{j=1}^n \exp(v^{T}f_i/\tau)
> $$
> 
>
> NCE 通过“噪声对比”将 **多分类问题 → 一组二分类问题**：
>
> - 真实样本来自数据分布 $P_d$
> - 噪声样本来自 $P_n$
> - 每个真实样本配 $m$ 个噪声样本（noise-to-data ratio）
>
> 设噪声分布为均匀分布：$ P_n(i)=\frac{1}{n}$
>
> 则样本 $i$ 属于真实数据的后验概率为：
> $$
> h(i,v)=P(D=1|i,v)=\frac{P(i|v)}{P(i|v)+mP_n(i)}
> $$
> 由于 $P_n(i)=1/n$：$ mP_n(i)=\frac{m}{n}$ 
>
> #### NCE Loss（训练目标函数）
>
> $$
> J_{NCE}(\theta)=
> 
> - E_{P_d}[\log h(i,v)]
> - m \cdot E_{P_n}[\log (1-h(i,v'))]
> $$
>
> 其中：
>
> - $v = f_i$ 是真实样本的特征
> - $v'$ 从噪声分布随机采样
> - 两者都来自 Memory Bank
>
> #### **Softmax 分母的近似（蒙特卡洛近似）**
>
> 由于 $Z_i$ 无法直接计算：
> $$
> Z_i=\sum_{j=1}^n \exp(v_j^Tf_i/\tau)
> $$
> 作者近似：
> $$
> Z \simeq Z_i \simeq  nE_j  \left[  \exp(v_j^{T} f_i / \tau)  \right] \simeq \frac{n}{m}
>  \sum_{k=1}^m
>  \exp(v_{j_k}^{T} f_i / \tau)
> $$
> 其中：
>
> - 随机采 $m$ 个噪声样本 $j_k$
> - 用它们估计整个 $n$ 项求和
>
> NCE reduces the computational complexity from O(n) to O(1) per sample. With such drastic reduction, our experiments still yield competitive performance.

### Proximal Regularization

> #### ✅ **为何需要 Proximal Regularization（近端正则）？**
>
> 在 Instance Discrimination（比如 CPC/InstDisc）中：
>
> - 每个 “类” 就是一个实例
> - 每个类只有 **1 个样本**
>
> 因此：
>
> 👉 一个样本（一个类）在整个 epoch 中 **只被访问一次**
>  👉 Memory Bank 里的特征更新非常稀疏
>  👉 导致训练 **非常不稳定**
>
> 为了解决这个问题，论文在损失函数中加入了一个 **平滑约束项（惩罚项）**：
>
> #### ✅ **加入的正则项（Proximal Regularization）**
>
> $$
> -\log h(i, \mathbf{v}_{i}^{(t-1)}) + \lambda \left| \mathbf{v}_{i}^{(t)} - \mathbf{v}_{i}^{(t-1)} \right|_2^2
> $$
>
> 其中：
>
> - $\mathbf{v}_{i}^{(t)} = f_\theta(x_i)$第 t次迭代中由 backbone 提取的当前特征
> - $\mathbf{v}_{i}^{(t-1)}$Memory Bank 中存储的上一轮特征
>
> 第一项：$ -\log h(i,\mathbf{v}_{i}^{(t-1)})$ 是正常的 NCE 损失。
>
> 第二项：$\lambda |\mathbf{v}_{i}^{(t)} - \mathbf{v}_{i}^{(t-1)}|_2^2$
>  是平滑项，鼓励：
>
> > **新的特征不要和旧特征差太多，让训练更平稳。**
>
> #### 损失函数
>
> $$
> J_{\text{NCE}}(\theta) =
>  -\mathbb{E}_{P_d} \Big[
>  \log h(i, v^{(t-1)}_i )- \lambda | v^{(t)}_i - v^{(t-1)}_i |_2^2
>    \Big]
> - m \cdot \mathbb{E}_{P_n} \Big[
>    \log(1 - h(i, v^{\prime (t-1)}))
>    \Big]
> $$
>
> 
>
> #### ✅ **直观理解**
>
> Memory Bank 的更新方式变成了 **动量式更新**：
>
> - 当前特征不能突然变化太大
> - 新特征逐渐“靠近”旧特征
> - Backbone 输出与 Memory Bank 中存储的历史特征逐渐一致
>
> 这类似于 MoCo 的思想：**用动量更新来保持特征队列的稳定性。**
>
> 最终：
>
> - 特征不会剧烈抖动
> - 收敛更快、更稳定
> - Memory Bank 表示与 Backbone 表示逐步对齐

<img src="https://gitee.com/dongliangnie/typora-image-bed-2/raw/master/202512041427167.png" alt="image-20251204142739023" style="zoom:67%;" />

## 实验结果分析

> ​	进行了 4 组实验来评估方法。第一组是在 CIFAR-10 上，用于将非参数 Softmax 与参数 SoftMax 进行比较。第二组是在 ImageNet 上将我们的方法与其他无监督学习方法进行比较。最后两组实验研究了两种不同的任务：半监督学习和目标检测，以展示我们学习的特征表示的泛化能力。

### 非参数 Softmax 与参数 SoftMax 比较

<img src="https://gitee.com/dongliangnie/typora-image-bed-2/raw/master/202512041446892.png" alt="image-20251204144615642" style="zoom:67%;" />

## 参考

[zhirongw/lemniscate.pytorch: Unsupervised Feature Learning via Non-parametric Instance Discrimination](https://github.com/zhirongw/lemniscate.pytorch)

[李沐论文精读系列三：MoCo、对比学习综述（MoCov1/v2/v3、SimCLR v1/v2、DINO等）_moco论文-CSDN博客](https://blog.csdn.net/qq_56591814/article/details/127564330)
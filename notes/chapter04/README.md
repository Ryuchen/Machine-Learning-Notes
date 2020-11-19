## 第四章 决策树

---

### 4.1 基本流程

#### 基本概念

**决策树是基于树结构来进行决策的**

决策树的目的是为了产生一棵泛化能力强，即处理未见示例能力强的决策树，其基本流程遵循简单且直观的“分而治之”（divide-and-conquer）策略

#### 西瓜例子

![西瓜例子](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/49c16f1fdba957c4be22b5258d3a89a8.webp)

#### 构建流程

- 决策过程的最终结论对应了我们所希望的判定结果
- 决策过程中提出的每个判定问题都是对某个属性的“测试”
- 每个测试的结果或是导出最终结论，或是导出进一步的判定问题，其考虑范围是上次决策结果的下定范围之内

#### 树的组成

- 一个根结点
- 若干个内部结点
- 若干个叶结点

叶子节点对应于决策结果，其他每个节点则对应于一个属性测试

> 每个结点包含的样本集合根据属性测试的结果被划分到子结点中，而根结点包含了样本全集
> 从根结点到每个叶结点的路径对应了一个判定测试序列

#### 算法示例

![算法示例](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/c0d22d97464b4bb1fadf8dbeb97053fd.webp)

**决策树的生成是一个递归过程**

三种会导致递归返回的情形

- 当前结点包含的样本全属于同一类别，无需划分
- 当前属性集为空，或是所有样本在所有属性上取值相同，无法划分
  - 把当前结点标记为叶结点，并将其类别设定为该结点所含样本最多的类别
  - 利用当前结点的后验分布
- 当前结点包含的样本集合为空，不能划分
  - 把当前结点标记为叶结点，但将其类别设定为父结点所含样本做多的类别
  - 把父结点的样本分布作为当前结点的先验分布

**决策树各类算法的核心就就在于：如何解决“从A中选择最优划分属性”？**

### 4.2 划分选择

> 核心问题：如何解决“从A中选择最优划分属性”？

> 划分目的：决策树分支结点所包含的样本尽可能属于同一类别，即结点的“纯度”（purity）越来越高

#### 信息熵（information entropy）

在决策树种，信息熵通常是用来度量样本集合纯度最常用的一种指标

$$\text{Ent}(D)=-\sum_{k=1}^{|\mathcal{Y}|}p_k\log_2p_k$$

- $\text{Ent}(D)$ 的值越小，则 $D$ 的纯度越高
- $\text{Ent}(D)$ 的最小值为0，最大值为 $\log_2|\mathcal{Y}|$

> PS: 更加具体的内容参见前一个上一篇博文（决策树 铺垫）中的内容

#### 西瓜例子

![](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/a9237f1110f818d79dfd8afc3d0935fd.webp)

接下来就是围绕这个数据集进行展开

#### 常见方法

参数设置

- 假定离散属性 $a$ 有 $V$ 个可能的取值 $\{a^1,a^2,...,a^V\}$
- 其中第 $v$ 个分支结点包含了 $D$ 中所有在属性 $a$ 上取值为 $a^v$ 的样本，记为 $D^v$
- 根据信息熵公式计算出 $D^v$ 的信息熵
- 再根据不同的分支结点所包含的样本数不同，给分支结点赋予不同的权重 $\frac{|D^v|}{|D|}$

##### 信息增益（information gain）

**关键公式:**

$$\text{Gain}(D,a)=\text{Ent}(D)-\sum_{v=1}^V\frac{|D^v|}{|D|}\text{Ent}(D^v)$$

- 信息增益越大，则意味着使用属性a来进行划分所获得的“纯度提升”越大
- 经典的ID3决策树算法就是以信息增益来为准则进行属性选择的

**西瓜例子:**

![西瓜例子](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/fa9933db77500123689811b28fca321d.webp)

**计算过程:**

根结点的信息熵为：

$$\text{Ent}(D)=-\sum_{k=1}^2p_k\log_2p_k=-(\frac{8}{17}\log_2\frac{8}{17}+\frac{9}{17}\log_2\frac{9}{17})=0.998$$

例如以“色泽”作为划分依据，则可以建立三个分支其对应的信息熵为

$$\text{Ent}(D^1)=-(\frac{3}{6}\log_2\frac{3}{6}+\frac{3}{6}\log_2\frac{3}{6})=1.000$$

$$\text{Ent}(D^2)=-(\frac{4}{6}\log_2\frac{4}{6}+\frac{2}{6}\log_2\frac{2}{6})=0.918$$

$$\text{Ent}(D^3)=-(\frac{1}{5}\log_2\frac{1}{5}+\frac{4}{5}\log_2\frac{4}{5})=0.722$$

则“色泽”对应的信息增益为

$$
\begin{align}
\text{Gain}(D,色泽) &= \text{Ent}(D) - \sum_{v=1}^3\frac{|D^v|}{|D|}\text{Ent}(D^v) \\\\
&= 0.998 - (\frac{6}{17} \times {1.000} + \frac{6}{17} \times {0.918} + \frac{5}{17} \times {0.722}) \\\\
&= 0.109
\end{align}
$$

其他属性对应的信息增益

$\text{Gain}(D,根蒂) = 0.143$

$\text{Gain}(D,敲声) = 0.141$

$\text{Gain}(D,纹理) = 0.381$

$\text{Gain}(D,脐部) = 0.289$

$\text{Gain}(D,触感) = 0.006$

> ID*3 算法就是使用的信息增益算法

**算法特点:**

信息增益准则对可取值数目较多的属性有所偏好，因此其在选择分类属性的时候倾向混乱程度更大的属性

##### 增益率（gain ratio）

**关键公式:**

$$
\begin{align}
& \text{Gain_ratio}(D,a)=\frac{\text{Gain}(D,a)}{\text{IV}(a)} \\\\
& \text{IV}(a)=-\sum_{v=1}^V\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}
\end{align}
$$

其中 $\text{IV}(a)=-\sum_{v=1}^V\frac{|D^v|}{|D|}\log_2\frac{|D^v|}{|D|}$ 称为属性 $a$ 的固有值(intrinsic value)

> 属性a的可能取值数目越多（即V越大），则IV（a）得值通常会越大
> 著名的C4.5决策算法则是使用“增益率”来选择最优划分属性

**西瓜例子:**

略

**算法特点**

增益率准则对可取数值数目较少的属性有所偏好，因此其在选择分类属性的时候倾向混乱程度更小的属性

##### 基尼指数（Gini index）

**关键公式:**

$$
\begin{align}
\text{Gini_index}(D,a) & = \sum_{v=1}^V\frac{|D^v|}{|D|}\text{Gini(D^v)} \\\\
\text{Gini}(D) & = \sum_{k=1}^{|\mathcal{Y}|}\sum_{k'\neq k}p_kp_{k'} \\\\
& = 1 - \sum_{k=1}^{|\mathcal{Y}|}p_{k}^2
\end{align}
$$

直观来说基尼指数反省了从数据集D中随机抽取两个样本，期类别标记不一致的概率

基尼指数越小，数据集D的纯度越高

在候选属性集合A中，选择那个使得划分后基尼指数最小的属性作为最优划分属性，即

$$
a_* = \underset{a \in A}{\arg\min} \text{Gini_index}(D,a)
$$

> CART决策树使用基尼指数来选择划分属性 

**西瓜例子:**

略

**算法特点**

基尼系数是一种衡量信息不确定性的方法，与信息熵计算出来的结果差距很小，基本可以忽略，但是基尼系数要计算快得多，因为没有对数

在特征选取中，会优先选择基尼指数最小的属性作为优先划分属性

### 4.3 剪枝处理

在决策树算法中对付“过拟合”的主要手段

**如何理解决策树过拟合？**

决策树分支结点过多，把训练集本生学得太好了，以致于把训练集自身的一些特点当作是所有数据集都具有的一般性质而导致过拟合

#### 基本策略

##### 预剪枝（prepruning）

预剪枝是指在决策树生成过程中，对每个结点在划分前先进行估计，若当前结点的划分不能带来决策树泛化性能提升，则停止划分并将当前结点标记为叶结点

##### 后减枝（postpruning）

后剪枝则是先从训练集生成一颗完整的决策树，然后自底向上地对非叶子结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点

#### 西瓜例子

数据集

![](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/fc4aa5db8195f09a8fefaa08deb3d73e.webp)

生成树

![](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/b0aa4ac4f8479cd39002351bdbd16e1f.webp)

##### 预剪枝

假设以 “脐部” 属性来对训练集进行划分

> 不进行划分与使用脐部进行划分验证集精度前后变化

![](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/336203e2853ed8c2fc7acca122ac581d.webp)

**特点**

- 预剪枝使得决策树的分支都没有展开，这不仅降低了过拟合的风险，还显著减少了决策树的训练时间开销和测试时间开销
- 有些分支的当前划分虽然不能提升泛化性能、甚至可能导致泛化性能暂时下降，但在其基础上进行的后续划分却有可能导致性能的显著提高

> PS: 使用的是贪心算法，因此会有欠拟合的风险

##### 后减枝

假设以 “脐部” 属性来对训练集进行划分

> 自下而上观察子树替换为叶结点后的验证集精度

![](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/23196a0255d3be76f1af31e9bd46efc9.webp)

**特点**

- 一般情况下，后剪枝决策树的欠拟合风险很小，繁华性能往往优于预剪枝决策树
- 后剪枝过程是在生成完全决策树之后进行的，并且要自底向上地对树中的所有非叶结点进行逐一考察，因此其训练时间开销比未剪枝决策树和预剪枝决策树都要大得多

---

> PS：后剪枝决策树通常比预剪枝决策树保留了更多的分支

---

### 4.4 连续与缺失值

#### 连续值处理

**二分法（bi-partition）**

概念

- 给定样本集 $D$ 合连续属性 $a$，假定 $a$ 在 $D$ 上出现了 $n$ 个不同的取值，将这些值从小到大进行排序，记为 $\{a^1,a^2,...,a^n\}$
- 基于划分点 $t$ 可将 $D$ 分为子集 $D_{t^-}$ 和 $D_{t^+}$ ，其中 $D_{t^-}$ 中包含那些在属性 $a$ 上取值不大于 $t$ 的样本，而 $D_{t^+}$ 则包含那些在属性 $a$ 上取值大于 $t$ 的样本

步骤

- 对相邻的属性取值 $a^i$ 和 $a^{i+1}$ 来说，$t$ 在区间 $[a^i,a^{i+1})$ 中取任意值所产生的划分结果相同

属性在实际情况中取值是连续的，但是样本集中也是呈现离散的装态，这里的取值不影响划分是指样本集中的离散状态

- 对连续属性 $a$，可考察包含 $n-1$ 个元素的候选划分点集合

$$T_a=\{\frac{a^i+a^{i+1}}{2}|1 \le i \le n-1\}$$

即将每个区间的中位点作为候选划分点

- 通过离散属性值一样里考察这些划分点，选择最优划分点进行样本集合的划分

$$
\begin{align}
\text {Gain} (D,a) &= \underset{t\in T_a}{\max } \text {Gain} (D,a,t) \\\\
&= \underset{t\in T_a}{\max} \text{Ent}(D) -  \sum_{\lambda \in \{ -,+ \}}
\frac{|D_t^{\lambda}|}{|D|}\text{Ent}(D_t^{\lambda})
\end{align}
$$

> 与离散属性不同，若当前结点划分属性为连续属性，该属性还可作为其后代结点的划分属性

#### 缺失值处理

##### 问题描述

现实任务中常会遇到不完整的样本，即样本的某些属性值缺失

（1）如何在属性值缺失的情况下进行划分属性选择？

（2）给定划分属性，若样本在该属性上的值缺失，如何对样本进行划分？

**权重法**

符号定义

- 令 $\tilde{D}$ 表示 $D$ 在属性 $a$ 上没有缺失值得样本子集
- 令 $\tilde{D^v}$ 表示 $\tilde{D}$ 中在属性 $a$ 上取值为 $a^v$ 的样本子集
- 令 $\tilde{D_k}$ 表示 $\tilde{D}$ 中属于第 $k$ 类的样本子集

为每个样本 $\boldsymbol{x}$ 赋予一个权重 $w_{\boldsymbol{x}}$

- 无缺失值的比例

$$\rho=\frac{\sum_{\boldsymbol{x} \in \tilde{D}}w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in D}w_{\boldsymbol{x}}}$$

- 无缺失样本中第 $k$ 类所占的比例

$$\tilde{p_k}=\frac{\sum_{\boldsymbol{x} \in \tilde{D_k}}w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in D}w_{\boldsymbol{x}}}, (1 \le k \le |\mathcal{Y}|)$$

- 无缺失样本中属性 $a$ 上取值的样本所占的比例

$$\tilde{r_v}=\frac{\sum_{\boldsymbol{x} \in \tilde{D^v}}w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in D}w_{\boldsymbol{x}}}, (1 \le v \le V)$$

公式

- 解决问题（1）

$$
\begin{align}
\text{Gain}(D,a) & = \rho \times \text{Gain}(\tilde{D}, a) \\\\
& = \rho \times \left ( \text{Ent}(\tilde{D} - \sum_{v=1}^V\tilde{r_v} \text{Ent}(\tilde{D^v})) \right )
\end{align}
$$

- 让同一个样本以不同的概率划分到不同的子结点中去：解决的问题（2）

若样本 $\boldsymbol{x}$ 在划分属性 $a$ 上的取值未知，则将 $\boldsymbol{x}$ 同时划入所有子结点，且样本权值在于属性值 $a^v$ 对应的子结点中调整为 $\tilde{r_v}·w_{\boldsymbol{x}}$

### 4.5 多变量决策树

#### 单变量决策树（univariate decision tree）

特点：轴平行（axis-parallel）：即它的分类边界由若干个与坐标轴平行的分段组成

![](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/e78b2687d3f15147410e1c58d78dc584.webp)

#### 多变量决策树（multivariate decision tree）

特点：非叶结点不再是仅对某个属性进行测试，而是对属性的线性组合进行测试

每一个非叶结点是一个形如 $\sum_{i=1}^dw_ia_i=t$ 的线性分类器，其中 $w_i$ 是属性 $a_i$ 的权重

![](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/11/16/047fa2288461f524df9b3ade35797aa0.webp)
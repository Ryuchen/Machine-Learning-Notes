## 第三章 线性模型

---

## 3.1 基本形式

### 核心思想

给定d个属性描述的示例 $\boldsymbol{x}=(x_1;x_2;...;x_d)$ ，其中 $x_i$ 是 $\boldsymbol{x}$ 在第 $i$ 个属性上的取值，<u>试图学得一个通过属性的线性组合来进行预测的函数</u>

### 公式

**一般形式**

$$f(\boldsymbol{x})=w_1x_1+w_2x_2+...+w_dx_d+b$$

**向量形式**

$$f(\boldsymbol{x})=\boldsymbol{w}^T\boldsymbol{x}+b$$

> $\boldsymbol{w}$ 和 $b$ 学得后，模型就得以确认

---

**$\boldsymbol{w}$ 直观表达了各属性在预测中的<u>重要性</u>**

---

### 优点

形式简单

易于建模

有很好的可解释性（comprehensibility）

> 注：非线性模型（nonlinear model）可在线性模型的基础上通过引入 <u>层级结构</u> 和 <u>高维映射</u> 而得

## 3.2 线性回归

### 核心思想

 线性回归（linear regression）试图学得一个线性模型以尽可能准确地预测实值输出标记

### 常见线性回归

#### 一元线性回归

输入的属性数量只有一个，即 

<div>$$D=\left\{(x_i,y_i)\right\}_{i=1}^m$$</div>

其中 $y_i\in \mathbb{R}$

> $w$ 和 $b$ 都只有一个来确定其预测标记

---

**目标函数**

$f(x_i) = wx_i+b$ ，使得 $f(x_i) \simeq y_i$

---

**问题求解**

> *如何确定 $w$ 和 $b$ ？*

求解方程

<div>
$$
\begin{align}
(w^{*},b^{*}) 
&= \underset{(w,b)}{\arg \min}\sum_{i=1}^{m} \left( f(x_i)-y_i\right)^2 \\
&= \underset{(w,b)}{\arg \min}\sum_{i=1}^{m}(y_i-wx_i-b )^2 
\end{align}
$$
</div>

求解方法：

> **最小二乘“参数估计”（parameter estimation）**

> 求解 $w$ 和 $b$ 使 $E_(w,b)=\sum_{i=1}^m(y_i-wx_i-b)^2$ 最小化的过程

求解过程：

- 对 $w$ 求偏导
		
<div>$$\frac{\partial{E_{(w,b)}}}{\partial{w}}=2 \left( w\sum_{i=1}^mx_i^2-\sum_{i=1}^m(y_i-b)x_i \right)$$</div>
		
- 对 $b$ 求偏导
	
<div>$$\frac{\partial{E_{(w,b)}}}{\partial{b}}=2 \left( mb-\sum_{i=1}^m(y_i-wx_i) \right)$$</div>

- 令两式均为0
	
    得到 $w$ 和 $b$ 的最优解的闭式(closed-form)解

结论：

$$w = \frac{\sum_{i=1}^m y_i(x_i-\overline{x})}{\sum_{i=1}^mx_i^2-\frac{1}{m}(\sum_{i=1}^m x_i)^2}$$
$$b = \frac{1}{m}\sum_{i=1}^m(y_i-wx_i)$$

> 其中 $\overline{x}=\frac{1}{m}\sum_{i=1}^m x_i$

#### 多元线性回归

输入的属性值有多个

<div>$$D=\left\{(\boldsymbol{x}\_1,y_1), (\boldsymbol{x}\_2,y_2), ..., (\boldsymbol{x}\_m,y_m)\right\}$$</div>

其中

<div>$$\boldsymbol{x}_i=(x_{i1};x_{i2};...;x_{id})，y_i \in \mathbb{R}$$</div>

> $w$ 有多个才能确定其预测标记

---

**目标函数**

$f(\boldsymbol{x}_i)=\boldsymbol{w}^T\boldsymbol{x}_i + b$，使得 $f(\boldsymbol{x}_i) \simeq y_i$

---

**问题求解**

> 如何确定 $\boldsymbol{w}$ 和 $b$ ？

设 $\hat{\boldsymbol{w}}=(\boldsymbol{w};b)$，相应的，把数据集 $D$ 表示为一个 $m \times (d+1)$ 大小的矩阵 $\mathbf{X}$

<div>
$$
\mathbf{X} = 
\begin{pmatrix}
	x_{11} & x_{12} & ··· & x_{1d} & 1 \\\\
	x_{21} & x_{22} & ··· & x_{2d} & 1 \\\\
	... & ... & ... & ... & ...\\\\
	x_{m1} & x_{m2} & ··· & x_{md} & 1 \\\\
\end{pmatrix} = 
\begin{pmatrix}
	\boldsymbol{x}_1^T & 1 \\\\
	\boldsymbol{x}_2^T & 1 \\\\
	... & ...\\\\
	\boldsymbol{x}_m^T & 1 \\\\
\end{pmatrix}
$$
</div>

求解方程：
    
<div>
$$
\hat{\boldsymbol{w}}^* = \underset{\hat{\boldsymbol{w}}}{\arg \min}(\boldsymbol{y}-\mathbf{X}\hat{\boldsymbol{w}})^T(\boldsymbol{y}-\mathbf{X}\hat{\boldsymbol{w}})
$$
</div>

> 求解 $\hat{\boldsymbol{w}}$ 使 $E_(\hat{\boldsymbol{w}})=(\boldsymbol{y}-\mathbf{X}\hat{\boldsymbol{w}})^T(\boldsymbol{y}-\mathbf{X}\hat{\boldsymbol{w}})$ 最小化的过程

求解方法：

> **矩阵求导**

求解过程

- 对 $\hat{\boldsymbol{w}}$ 求导

    - $$\frac{\partial{E_{\hat{\boldsymbol{w}}}}}{\partial{\hat{\boldsymbol{w}}}} = 2 \mathbf{X}^T(\mathbf{X}\hat{\boldsymbol{w}}-\boldsymbol{y})$$

- 令其为0
    - 得到 $\hat{\boldsymbol{w}}$ 的最优解的闭式解

结论：(两种情况)

- $\mathbf{X}$ 为满秩矩阵或正定矩阵

    即：$$\hat{\boldsymbol{w}}^*=(\mathbf{X^TX})^{-1}\mathbf{X^T}\boldsymbol{y}$$

    令 $\hat{\boldsymbol{x}}_i=(\boldsymbol{x}_i;1)$，则最终学得的多元线性回归模型为 $$f(\hat{\boldsymbol{x}}_i) = \hat{\boldsymbol{x}}_i^{\mathbf{T}}(\mathbf{X^TX})^{-1}\mathbf{X}^{\mathbf{T}}\boldsymbol{y}$$

- $\mathbf{X}$ 不为满秩矩阵
    
    此时可解出多个 $\hat{\boldsymbol{w}}$，它们都能使均方误差最小化
    
    选择哪一个解是依据学习算法的归纳偏好决定的，最常见的做法就是引入正则化（regularization）

#### 对数线性回归

**使得模型的预测值对应于真实值的<u>某个衍生物</u>**

![对数线性回归](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/10/30/f42071431685770ffa9a1f0dbb7f0260.webp)

- 例如：$\ln{y} = \boldsymbol{w}^{\mathbf{T}}\boldsymbol{x}+b$

#### 广义线性模型

**模型的预测值与真实值之间可以通过“联系函数”（link function）对应**

- 例如：$y=g^{-1}(\boldsymbol{w}^{\mathbf{T}}\boldsymbol{x}+b)$

- 广义线性模型的参数估计常通过加权最小二乘法或极大似然法进行

## 3.3 对数几率回归

### 通过线性回归的模型来进行分类任务

- 利用广义线性模型
- 只需找一个单调可微函数将分类任务的真实标记与线性回归模型的预测值联系起来

### 二分类任务

![二分类任务](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/10/30/a6df2baed2aab1bd5f8e17a0aff86d2d.webp)

期望结果输出 $y \in \left\\{{0, 1}\right\\}$

### 联系函数

#### 单位阶跃函数（unit-step function）

$y = 
\begin{cases} 
0, & z<0; \\\\ 
0.5, & z=0; \\\\ 
1, & z>0; 
\end{cases}
$

- 若预测值大于零判为正例
- 若预测值小于零判为反例
- 若预测值等于零则可任意判别

> 缺点是不连续性

#### 对数几率函数（logistic function）

一定程度上近似单位阶跃函数的“替代函数”（surrogate function），并希望它单调可微

- Sigmoid函数(最为重要的代表)

	- 形似 S 的函数，它将 $z$ 值转化为一个接近 0 或 1 的 $y$ 值，并且其输出值在 $z=0$ 附近变化很陡
	- $y = \frac{1}{1+e^{-z}}$

### 对数几率回归（logistics regression）

将 $y$ 视为样本 $\boldsymbol{x}$ 作为正例的可能性,则 $1-y$ 是其反例的可能性，两者的比值 $\frac{y}{1-y}$ 称为“几率”(odds)，$\ln \frac{y}{1-y}$，则称为对数几率 (log odds, 亦称logit)

如果将Sigmoid函数带入到广义线性模型的表达式中：

$$y = \frac{1}{1+e^{-(\boldsymbol{w}^{\text{T}}\boldsymbol{x}+b)}}$$

可得：

$$\ln \frac{y}{1-y} = \boldsymbol{w}^{\text{T}}\boldsymbol{x} + b$$

> **实际上在用线性回归模型的预测结果去逼近真实标记的对数几率**

### 特点

虽然名字中包含回归，但是实质却是一种分类学习方法

优点
    - 它是直接对分类可能性进行建模，无需事先假设数据分布
    - 它不是仅预测出“类别”，而是可得到近似概率预测，这对许多需利用概率辅助决策的任务很有用
    - 对率函数是任意阶可导的凸函数，有很好的的数学性质，现有的许多数值优化算法都可直接用于求解最优解

### 问题求解

> *如何确定 $\boldsymbol{w}$ 和 $b$ ？*

将 $y$ 视为类后验概率估计 $p(y=1|\boldsymbol{x})$

将 $\ln \frac{y}{1-y} = \boldsymbol{w}^{\text{T}}\boldsymbol{x} + b$ 转化为 $$\ln \frac{p(y = 1|\boldsymbol{x})}{p(y = 0|\boldsymbol{x})} = \boldsymbol{w}^{\text{T}}\boldsymbol{x} + b$$

- $p(y=1|\boldsymbol{x})=\frac{e^{\boldsymbol{w^{\text{T}}\boldsymbol{x} + b}}}{1+e^{\boldsymbol{w^{\text{T}}\boldsymbol{x} + b}}}$
	
- $p(y=0|\boldsymbol{x})=\frac{1}{1+e^{\boldsymbol{w^{\text{T}}\boldsymbol{x} + b}}}$

通过“极大似然法”(maximum likelihood method)来估计 $\boldsymbol{w}$ 和 $b$

<div>
$$
\ell(\boldsymbol{w},b)=\sum_{i=1}^m\ln{p(y_i|\boldsymbol{x}_i;\boldsymbol{w},b)}
$$
</div>
	
对率回归模型最大化“对数似然”（loglikelihood），即令每个样本属于其真实标记的概率越大越好

## 3.4 线性判别分析

### 概念

> 线性判别分析（Linear Discriminant Analysis，简称 LDA）的核心思想

- 给定训练样例集，设法将样例投影到一条直线上，使得同类样例的投影点尽可能接近、异类样例的投影点尽可能远离

> 也称 “Fisher判别法”，但严格来说 LDA 要求更严格，各类样本的协方差矩阵相同且满秩

![线性判别分析](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/10/30/e31ec2e03803d1fd74633c7ff6faa943.webp)

**LDA可以从贝叶斯决策理论的角度来阐释，并可证明，当两类数据同先验、满足高斯分布且协方差相等时，LDA可达到最优分类**

### 二分类任务

#### 广义瑞利商（generalized Rayleigh quotient）

##### 参数定义

- 数据集 <div>$D=\left\{ (\boldsymbol{x}\_i,y_i) \right\}_{i=1}^m,y_i \in \left\{0,1\right\}$</div>
- $X_i$ 表示第 <span>$i \in \left\{0,1\right\}$</span> 类示例的集合
- $\boldsymbol{\mu}_i$ 表示第 <span>$i \in \left\{0,1\right\}$</span> 类示例的均值向量
- $\boldsymbol{\Sigma}_i$ 表示第 <span>$i \in \left\{0,1\right\}$</span> 类示例的协方差矩阵

##### 实现方式

将数据投影到直线 $y=\boldsymbol{w}^{\text{T}}\boldsymbol{x}$ 上

- 两类样本的中心在直线上的投影分别为 $\boldsymbol{w}^{\text{T}}\boldsymbol{\mu}_0$ 和 $\boldsymbol{w}^{\text{T}}\boldsymbol{\mu}_1$
- 两类样本的协方差分别为 $\boldsymbol{w}^{\text{T}}\boldsymbol{\Sigma_{0}}\boldsymbol{w}$ 和 $\boldsymbol{w}^{\text{T}}\boldsymbol{\Sigma_{1}}\boldsymbol{w}$

> 由于直线是一维空间，因此这四个值均为实数

目标

- 欲使<u>同类样例</u>的投影点尽可能接近，可以让同类样例投影点的协方差尽可能小，即 $$\boldsymbol{w}^{\text{T}}\boldsymbol{\Sigma_{0}}\boldsymbol{w} + \boldsymbol{w}^{\text{T}}\boldsymbol{\Sigma_{1}}\boldsymbol{w}$$ 尽可能小
- 欲使<u>异类样例</u>的投影点尽可能远离，可以让类中心点之间的距离尽可能大，即 $$||\boldsymbol{w}^{\text{T}}\boldsymbol{\mu}_0 + \boldsymbol{w}^{\text{T}}\boldsymbol{\mu}_1||_2^2$$ 尽可能大

公式

- 类内散度矩阵（within-class scatter matrix）

<div>
$$
\begin{align}
\boldsymbol{S}_w & = \boldsymbol{\Sigma}_0 + \boldsymbol{\Sigma}_1 \\
& = \sum_{\boldsymbol{x} \in X_0}(\boldsymbol{x}-\boldsymbol{\mu}_0)(\boldsymbol{x}-\boldsymbol{\mu}_0)^{\text{T}} + \sum_{\boldsymbol{x} \in X_1}(\boldsymbol{x}-\boldsymbol{\mu}_1)(\boldsymbol{x}-\boldsymbol{\mu}_1)^{\text{T}}
\end{align}
$$
</div>

- 类间散度矩阵（between-class scatter matrix）

<div>
$$
\boldsymbol{S}_b = (\boldsymbol{\mu_{0}}-\boldsymbol{\mu_{1}})(\boldsymbol{\mu_{0}}-\boldsymbol{\mu_{1}})^{\text{T}}
$$
</div>

- 同时考虑两者，则可得到与最大化的目标 

<div>
$$
\begin{align}
J & = \frac{||\boldsymbol{w}^{\text{T}}\boldsymbol{\mu}_0 + \boldsymbol{w}^{\text{T}}\boldsymbol{\mu}_1||_2^2}{\boldsymbol{w}^{\text{T}}\boldsymbol{\Sigma_{0}}\boldsymbol{w} + \boldsymbol{w}^{\text{T}}\boldsymbol{\Sigma_{1}}\boldsymbol{w}} \\
& = \frac{\boldsymbol{w}^{\text{T}}(\boldsymbol{\mu_{0}}-\boldsymbol{\mu_{1}})(\boldsymbol{\mu_{0}}-\boldsymbol{\mu_{1}})^{\text{T}}\boldsymbol{w}}{\boldsymbol{w}^{\text{T}}(\boldsymbol{\Sigma_{0}}+\boldsymbol{\Sigma_{1}})\boldsymbol{w}} \\
& = \frac{\boldsymbol{w}^{\text{T}}\boldsymbol{S}_b\boldsymbol{w}}{\boldsymbol{w}^{\text{T}}\boldsymbol{S}_w\boldsymbol{w}}
\end{align}
$$
</div> 

##### 如何求解 $\boldsymbol{w}$ ?

推导过程

- 分子分母都是关于 $\boldsymbol{w}$ 的二次项，因此其解与 $\boldsymbol{x}$ 长度和方向有关
- 因此可假设，$\boldsymbol{w}^{\text{T}}\boldsymbol{S}_w\boldsymbol{w} = 1$，则原广义瑞利商变为最大化 $\boldsymbol{w}^{\text{T}}\boldsymbol{S}_b\boldsymbol{w}$
- 则其对偶问题变为: $$\begin{align}
\underset{\boldsymbol{w}}{\min} - \boldsymbol{w}^{\text{T}}\boldsymbol{S}_b\boldsymbol{w} \\
\text{s.t.} \quad \boldsymbol{w}^{\text{T}}\boldsymbol{S}_w\boldsymbol{w} = 1
\end{align}$$
  
求解过程
    
- 利用拉格朗日乘子法得：$\boldsymbol{S}_b\boldsymbol{w} = \lambda\boldsymbol{S}_w\boldsymbol{w}$
- $\lambda$ 是拉格朗日乘数是个实数，则 $\boldsymbol{S}_b\boldsymbol{w}$ 的方向只与$(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$ 有关
- 设 $\boldsymbol{S}_b\boldsymbol{w} = \lambda(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$
	
带入解得 $\boldsymbol{w} = \boldsymbol{S}_w^{-1}(\boldsymbol{\mu}_0 - \boldsymbol{\mu}_1)$

- 同样该问题的最终解也只与其方向有关，因此只需要对 $\boldsymbol{S}_w$ 进行奇异值分解即可
- $\boldsymbol{S}_w = \boldsymbol{U}\Sigma\boldsymbol{V}^{\text{T}} 则 \boldsymbol{S}_w^{-1} = \boldsymbol{V}\Sigma^{-1}\boldsymbol{U}^{\text{T}}$
  
## 3.5 多分类学习

### 拆解法

#### 基本思路

- 即将多分类任务拆分为若干个二分类任务求解
- 具体来说，先对问题进行拆分，然后为拆出的每个二分类任务训练一个分类器；在测试时，对这些分类器的预测结果进行集成以获得最终的多分类结果

#### 拆分策略

- “一对一”（One vs. One，简称 OvO）
- “一对其余”（One vs. Rest，简称 OvR）
- “多对多”（Many vs. Many，简称 MvM）

#### 示例

##### OvO && OvR

![OvO && OvR](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/10/30/e79f1e61fcd4fd2c4089b634e56adb70.webp)

OvO

- 将这N个类别两两配对，从而产生N(N-1)/2个二分类任务
- 在测试阶段，新样本将同时提交给所有分类器，欲使我们将得到N(N-1)/2个分类结果，最终结果通过投票产生

OvR

- 每次将一个类的样例作为正例、所有其他类的样例作为反例来训练N个分类器
- 在测试阶段，若仅有一个分类器预测为正类，则对应的类别标记作为最终结果分类

对比

- 在测试时，OvO的存储开销和测试时间开销通常比OvR更大
- 在训练时，OvR的每个分类器均使用全部的训练样例，而OvO得每个分类器仅用到两个类的样例，训练开销比较小

##### MvM

![编码矩阵](https://cdn.jsdelivr.net/gh/Ryuchen/ImageBed@develop/2020/10/30/387d33fb4f6b686495e8eac6560638ab.webp)

每次将若干个类作为正类，若干个其他类作为反类

纠错输出码（Error Correcting Output Codes，简称 ECOC）

工作过程

- 编码

  - 对N个类别做M次划分，每次划分将一部分类别划为正类，一部分划为反类，从而形成一个二分类训练集
  - 这样一共产生M个训练集，可训练出M个分类器

- 解码

  - M个分类器分别对测试样本进行预测，这些预测标记形成一个编码
  - 将这个预测编码与每个类别各自的编码进行比较，返回其中距离最小的类别作为最终预测结果

编码矩阵 （coding matrix）

- 常见形式

  - 二元码: (正类\反类)

  - 三元码:(正类\反类\停用类)

为什么称“纠错输出码”？

- 在测试阶段，ECOC编码对分类器的错误有一定的容忍和修正能力
- 一般来说，对同一个学习任务，ECOC编码越长，纠错能力越强

  - 编码越长，付出的代价是所需训练的分类器越多，计算、存储开销都会增大
  - 对有限类别数，可能的组合数目是有限的，码长超过一定范围后就失去了意义

- 对同等长度的编码，理论上来说，任意两个类别之间的编码距离越远，纠错能力越强

  - 码长较小时可根据这个原则计算出理论最优编码，然而码长稍大一些就难以有效地确定最优编码
  - 并不是编码的理论性质越好，分类性能就越好

## 3.6 类别不平衡问题

### 类别不平横（class-imbalance）

- 指分类任务中不同类别的训练样例数目差别很大的情况

#### 理想情况下

- 不同类别的训练样例数目相当
- 若 $\frac{y}{1-y}>1$，则预测为正例

#### 真实情况下

- 正类与反类样例数目可能出现较大差异

	- 本生训练集中不同类别的数目就是有差距的
	- 经过拆分策略之后导致出现类别不平衡现象

- 若 $\frac{y}{1-y}>\frac{m^+}{m^-}$，则预测为正例

### 再缩放（rescaling）

只需对预测值进行平衡性调整即可

$$\frac{y'}{1-y'}=\frac{y}{1-y} \times \frac{m^-}{m^+}$$

> 思想简单，实际操作困难

- 主要原因“训练集是真实样本总体的无偏采样”这个假设往往并不成立
- 即可能无法从训练集观测几率来推断出真实几率

#### 解决方法

##### 欠采样（undersampling）或 下采样（downsampling）

- 直接对训练集里的反类样例进行欠采样，即去除一些反例，使得正、反例数目接近，然后再进行学习
- 特点

	- 分类器训练集远小于初始训练集
	- 若采用随机采样，可能会导致一些重要信息丢失

- 代表算法：EasyEnsemble

	- 利用集成学习机制，将反例划分为若干个集合供不同学习其使用

##### 过采样（oversampling）或 上采样（upsampling）

- 对训练集里的正类样例进行过采样，即增加一些正例使得正、反例数目接近，然后再进行学习
- 特点

	- 分类器训练集远大于初始训练集
	- 不可进行重复采样，否则会导致严重的过拟合

- 代表算法：SMOTE

	- 对训练集里的正例进行插值来产生额外的正例

- 阈值移动（threshold-moving）

    - 直接基于原始训练集进行学习，但在用训练好的分类器进行预测时，引入比例进行调整

---

> 由于Github公式显式限制，可以去我的博客 [https://ryuchen.club](https://ryuchen.club)进行查看，同时可以领取以下内容

- 本章脑图
- 本章图片
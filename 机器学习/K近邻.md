# K近邻

1.  **k近邻法**（k-Nearest Neighbor，简称kNN）是一种基本的分类与回归方法。

   - 分类问题：对新的样本，根据其k个最近邻的训练样本的类别，通过多数表决等方式进行预测。
   - 回归问题：对新的样本，根据其k个最近邻的训练样本标签值的均值作为预测值。

2. k近邻法不具有显式的学习过程，它是直接预测。它是**惰性学习**(lazy learning)的著名代表。

   - 它实际上利用训练数据集对特征向量空间进行划分，并且作为其分类的"模型"。

   - 这类学习技术在训练阶段仅仅将样本保存起来，训练时间开销为零，等到收到测试样本后再进行处理。

     那些在训练阶段就对样本进行学习处理的方法称**作急切学习**(eager learning)。

3. k近邻法是个非参数学习算法，它没有任何参数（k是超参数，而不是需要学习的参数）。

   - k近邻模型具有非常高的容量，这使得它在训练样本数量较大时能获得较高的精度。

   - 它的缺点有：

     - 计算成本很高。因为需要构建一个$N\times N$的距离矩阵，其计算量为$O(N^2)$，其中N为训练样本的数量。

       当数据集是几十亿个样本时，计算量是不可接受的。

     - 在训练集较小时，泛化能力很差，非常容易陷入过拟合。

     - 无法判断特征的重要性。

4. k近邻法的三要素：

   -  k值选择。
   - 距离度量。
   - 决策规则。

## 1. k值选择

1. 当$k=1$时的k近邻算法称为最近邻算法，此时将训练集中与$\mathbf x$最近的点的类别作为$\mathbf x$的分类。

2. k值的选择会对k近邻法的结果产生重大影响。

   - 若k值较小，则相当于用较小的邻域中的训练样本进行预测，"学习"的偏差减小。

     只有与输入样本较近的训练样本才会对预测起作用，预测结果会对近邻的样本点非常敏感。

     若近邻的训练样本点刚好是噪声，则预测会出错。即：k值的减小意味着模型整体变复杂，易发生过拟合。

     - 优点：减少"学习"的偏差。
     - 缺点：增大"学习"的方差（即波动较大）。

   - 若k值较大，则相当于用较大的邻域中的训练样本进行预测。

     这时输入样本较远的训练样本也会对预测起作用，使预测偏离预期的结果。

     即： k值增大意味着模型整体变简单。

     - 优点：减少"学习"的方差（即波动较小）。
     - 缺点：增大"学习"的偏差。

3. 应用中，k值一般取一个较小的数值。通常采用交叉验证法来选取最优的k值。

## 2. 距离度量

1. 特征空间中两个样本点的距离是两个样本点的相似程度的反映。

    近邻模型的特征空间一般是n维实数向量空间$\mathbb R^n$，其距离一般为欧氏距离，也可以是一般的$L_p$距离：
   $$
   L_p(\mathbf x_i,\mathbf x_j)=(\sum_{l=1}^{N}|\mathbf x_{i,l}-\mathbf x_{j,l}|^p)^{1/p},\ p\geqslant1\\
   \mathbf x_i,\mathbf x_j\in \mathcal X=\mathbb R^n;\mathbf x_i=(x_{i,1},x_{i,2},...,x_{i,n})^T\tag{1}
   $$

   - 当$p=2$时，为欧氏距离： $L_2(\mathbf x_i,\mathbf x_j)=(\sum_{l=1}^{N}|\mathbf x_{i,l}-\mathbf x_{j,l}|^2)^{1/2}$
   - 当$p=1$时，为曼哈顿距离：$L(\mathbf x_i,\mathbf x_j)=\sum_{l=1}^{N}|\mathbf x_{i,l}-\mathbf x_{j,l}|$
   - 当$p=\infty$时，为各维度距离中的最大值：$L_{\infty}(\mathbf x_i,\mathbf x_j)=max_l\ |\mathbf x_{i,l}-\mathbf x_{j,l}|$

2. 不同的距离度量所确定的最近邻点是不同的。

## 3. 决策规则

1. 分类决策通常采用多数表决，也可以基于距离的远近进行加权投票：距离越近的样本权重越大。

2. 回归决策通常采用均值回归，也可以基于距离的远近进行加权投票：距离越近的样本权重越大。

转自：

- [AI算法工程师手册](<http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/chapters/1_linear.html>)
# K-Means

1. **在无监督学习**(unsupervised learning)中，训练样本的标记信息是未知的。

2. 无监督学习的目标：通过对无标记训练样本的学习来揭露数据的内在性质以及规律。

3. 一个经典的无监督学习任务：寻找数据的**最佳表达**(representation)。常见的有：

   - 低维表达：试图将数据（位于高维空间）中的信息尽可能压缩在一个较低维空间中。
   - 稀疏表达：将数据嵌入到大多数项为零的一个表达中。该策略通常需要进行维度扩张。
   - 独立表达：使数据的各个特征相互独立。

4. 无监督学习应用最广的是聚类(clustering) 。

   - 假定样本集$D = \left \{\mathbf x_1,\mathbf x_2,…,\mathbf x_m  \right \}$包含$m$个无标记样本，每个样本$\mathbf x_i = (x_{i1};x_{i2};…;x_{in})$是一个n维特征向量，则聚类算法将样本集$D$划分为$k$个不相交的簇$\left \{C_l\ |\ l=1,2,…,k  \right \}$，其中$C_{l^{'}}\bigcap_{l^{'}\neq l}C_l = \phi$且$D=\bigcup_{l=1}^kC_l$。
   - 通过这样的划分，每个簇可能对应于一个潜在的概念。这些概念对于聚类算法而言，事先可能是未知的。
   - 聚类过程仅仅能自动形成簇结构，簇所对应的概念语义需要由使用者来提供。

5. 相应的，用$\lambda_j\in \left \{1,2,…,k  \right \}$表示样本$\mathbf x_j$的**簇标记**(cluster label)，即$\mathbf x_j \in C_{\lambda j}$。于是，聚类的结果可用包含$m$个元素的簇标记向量$\mathbf \lambda=(\lambda_1;\lambda_2;…;\lambda_m)$表示。

6. 聚类的作用：

   - 可以作为一个单独的过程，用于寻找数据内在的分布结构。
   - 也可以作为其他学习任务的前驱过程。如对数据先进行聚类，然后对每个簇单独训练模型。

7. 聚类问题本身是病态的。即：没有某个标准来衡量聚类的效果。

   - 可以简单的度量聚类的性质，如每个聚类的元素到该类中心点的平均距离。

     但是实际上不知道这个平均距离对应于真实世界的物理意义。

   - 可能很多不同的聚类都很好地对应了现实世界的某些属性，它们都是合理的。

     如：在图片识别中包含的图片有：红色卡车、红色汽车、灰色卡车、灰色汽车。可以聚类成：红色一类、灰色一类；也可以聚类成：卡车一类、汽车一类。

     > 解决该问题的一个做法是：利用深度学习来进行分布式表达，可以对每个车辆赋予两个属性：一个表示颜色、一个表示型号。

## 一、性能度量

1. 聚类的性能度量也称作聚类的**有效性指标**(validity index)。
2. 直观上看，希望同一簇的样本尽可能彼此相似，不同簇的样本之间尽可能不同。即：**簇内相似度**(intra-cluster similarity)高，且**簇间相似度**(inter-cluster similarity)低。
3. 聚类的性能度量分两类：
   - 聚类结果与某个**参考模型**(reference model)进行比较，称作**外部指标**(external index)。
   - 直接考察聚类结果而不利用任何参考模型，称作**内部指标**(internal index) 。

### 1.1 外部指标

1. 对数据集$D=\left \{\mathbf x_1,\mathbf x_2,…,\mathbf x_m  \right \}$，假定通过聚类给出的簇划分为$C=\left \{C_1,C_2,…,C_k  \right \}$，参考模型给定的簇划分为$C^*=\left \{C_1^*,C_2^*,…,C_s^*  \right \}$。相应地，令$\mathbf \lambda$与$\mathbf \lambda^*$分别表示与$C$和$C^*$对应的簇标记向量，将样本两两配对考虑，定义：
   $$
   a=|SS|,\ SS=\left \{(\mathbf x_i,\mathbf x_j)|\lambda_i=\lambda_j,\lambda_i^*=\lambda_j^*,i<j  \right \}\tag{1}
   $$

   $$
   b=|SD|,\ SD=\left \{(\mathbf x_i,\mathbf x_j)|\lambda_i=\lambda_j,\lambda_i^*\neq \lambda_j^*,i<j  \right \}\tag{2}
   $$

   $$
   c=|DS|,\ SD=\left \{(\mathbf x_i,\mathbf x_j)|\lambda_i\neq\lambda_j,\lambda_i^*=\lambda_j^*,i<j  \right \}\tag{3}
   $$

   $$
   d=|DD|,\ SD=\left \{(\mathbf x_i,\mathbf x_j)|\lambda_i\neq\lambda_j,\lambda_i^*\neq\lambda_j^*,i<j  \right \}\tag{4}
   $$

   $|·|$表示集合的元素的个数。各集合的意义为：

   - $SS$：包含了同时隶属于$C,C^*$的样本对。
   - $SD$：包含了隶属于$C$，但是不隶属与$C^*$的样本对。
   - $DS$：包含了不隶属于$C$，但是隶属与$C^*$的样本对。
   - $DD$：包含了既不隶属于$C$，又不隶属与$C^*$的样本对。

   由于每个样本对$(\mathbf x_i,\mathbf x_j)\ (i<j)$仅能出现在一个集合中，因此有$a+b+c+d=m(m-1)/2$成立。

2. 基于式(1)~(4)可导出下面这些常用的聚类性能外部指标：

   - Jaccard系数(Jaccard Coefficient，简称JC)
     $$
     JC=\frac{a}{a+b+c}\tag{5}
     $$

   - FM指数(Fowlkes and Mallows Index，简称FMI)
     $$
     FMI=\sqrt{\frac{a}{a+b}·\frac{a}{a+c}}\tag{6}
     $$

   - Rand指数(Rand Index，简称RI)
     $$
     RI=\frac{2(a+d)}{m(m-1)}\tag{7}
     $$
     

   上述性能度量的结果值均在$[0,1]$区间，值越大越好。

## 1.2 内部指标

1. 对聚类结果的簇划分$C=\left \{C_1,C_2,…,C_k  \right \}$，定义：
   $$
   avg(C)=\frac{2}{|C|(|C|-1)}\sum_{1\leqslant i<j\leqslant |C|}^{}dist(\mathbf x_i,\mathbf x_j)\tag{8}
   $$

   $$
   diam(C)=max_{1\leqslant i<j\leqslant |C|}dist(\mathbf x_i,\mathbf x_j)\tag{9}
   $$

   $$
   d_{min}(C_i,Cj)=min_{\mathbf x_i\in C_i,x_j\in C_j}dist(\mathbf x_i,\mathbf x_j)\tag{10}
   $$

   $$
   d_{cen}(C_i,C_j)=dist(\mathbf \mu_i,\mathbf \mu_j)\tag{11}
   $$

   其中$dist(·,·)$用于计算两个样本之间的距离；$\mathbf \mu$代表簇$C$的中心点$\mathbf \mu=\frac{1}{C}\sum_{1\leqslant i\leqslant|C|}\mathbf x_i$。

   上述定义的意义为：

   - $avg(C)$：簇$C$中每对样本之间的平均距离。
   - $diam(C)$：簇$C$中距离最远的两个样本的距离。
   - $d_{min}(C_i,Cj)$：簇$C_i$与簇$C_j$中最近样本间的距离。
   - $d_{cen}(C_i,C_j)$：簇$C_i$与簇$C_j$中心点间的距离。

2. 基于式(8)~(11)可导出下面这些常用的聚类性能内部指标：

   - DB指数(Davies-Bouldin Index，简称DBI)：
     $$
     DBI=\frac{1}{k}\sum_{i=1}^{k}\underset{j\neq i}{max}\left (\frac{avg(C_i)+avg(C_j)}{dcen(C_i,C_j)}  \right )\tag{12}
     $$

   - Dunn指数(Dunn Index，简称DI)：
     $$
     DI=\underset{1\leqslant i\leqslant k}{min}\left \{\underset{j\neq i}{min} \left (\frac{d_{min}(C_i,C_j)}{max_{1\leqslant l\leqslant k }diam(C_l)}  \right )\right \}\tag{13}
     $$

   显然，DBI的值越小越好，DI的值越大越好。

## 1.3 距离计算

1. 对函数$dist(·,·)$，若它是一个**距离度量**(distance measure)，则需满足一些基本性质：
   $$
   非负性：dist(\mathbf x_i,\mathbf x_j)\geqslant0\tag{14}
   $$

   $$
   同一性：dist(\mathbf x_i,\mathbf x_j)=0\ 当且仅当\mathbf x_i=\mathbf x_j\tag{15}
   $$

   $$
   对称性:dist(\mathbf x_i,\mathbf x_j)=dist(\mathbf x_j,\mathbf x_i)\tag{16}
   $$

   $$
   直递性：对称性:dist(\mathbf x_i,\mathbf x_j)\leqslant dist(\mathbf x_i,\mathbf x_k)\leqslant dist(\mathbf x_k,\mathbf x_j)\tag{17}
   $$

2. 当定样本$\mathbf x_i = (x_{i1};x_{i2};…;x_{in})$与$\mathbf x_j = (x_{j1};x_{j2};…;x_{jn})$，常用的有以下距离：

   - **闵可夫斯基距离**(Minkowski distance)
     $$
     dist_{mk}=\left (\sum_{u=1}^{n}|x_{iu}-x_{ju}|^{\frac{1}{p}}  \right )\tag{18}
     $$
     当$p=2$时，闵可夫斯基距离就是**欧式距离**(Euclidean distance)：
     $$
     dist_{ed}=||\mathbf x_{i}-\mathbf x_{j}||_2=\sqrt{\sum_{u=1}^{n}|x_{iu}-x_{ju}|}\tag{19}
     $$
     当$p=1$时，闵可夫斯基距离就是**曼哈顿距离**(Manhattan distance)，亦称为**街区距离**(city block distance)：
     $$
     dist_{man}=||\mathbf x_{i}-\mathbf x_{j}||_1=\sum_{u=1}^{n}|x_{iu}-x_{ju}|\tag{20}
     $$
     当$p→\infty$时则得到切比雪夫距离。

3. 我们常将属性划分为**连续属性**(continuous attribute)和**离散属性**(categorical attribute)，前者在定义域上有无穷多个取值，后者在定义域上是有限个取值。然而，在讨论距离计算时，属性上是否定义了"序"关系更为重要。例如定义域为$\left \{1,2,3  \right \}$的离散属性与连续属性的性质更接近一些，能直接在属性上计算距离："1"与"2"比较接近、与"3"比较远，这样的属性称为**有序属性**(ordinal attribute)；而定义域为{飞机，火车，轮船}这样的离散属性则不能直接在属性上计算距离，称为**无序属性**(non-ordinal attribute)。闵可夫斯基距离可用于有序属性。

4. 对无序属性可采用**VDM**(Value Difference Metric)。令$m_{u,a}$表示在属性$u$上取值为$a$的样本数，$m_{u,a,i}$表示在第$i$个样本簇中在属性$u$上取值为$a$的样本数，$k$为样本簇数，则属性$u$上两个离散值$a$与$b$之间的VDM距离为：
   $$
   VDM_p(a,b)=\sum_{i=1}^{k}|\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}|^p\tag{21}
   $$
   该距离刻画的是：属性取值在各簇上的频率分布之间的差异。

5. 将闵可夫斯基距离和VDM结合即可处理混合属性。假定有$n_c$个有序属性，$n-n_c$个无序属性，不失一般性，令有序属性排列在无序属性之前，则：
   $$
   MinkovDM_p(\mathbf x_i,\mathbf x_j)=\left (\sum_{u=1}^{n_c}|x_{iu}-x_{ju}|^p+\sum_{n_{c+1}}^{n}VDM(x_{iu}-x_{ju})  \right )^{\frac{1}{p}}\tag{22}
   $$
   
6. 当样本空间中不同属性的重要性不同时，可使用**加权距离**(weighted distance)。以加权闵可夫斯基距离为例：
   $$
   dist_{wmk}=(\omega_1·|x_{i1}-x_{j1}|^{\frac{1}{p}}+...+\omega_n·|x_{in}-x_{jn}|^{\frac{1}{p}})\tag{23}
   $$
   

   其中权重$\omega_i \geqslant 0\ (i=1,2,…,n)$表征不同属性的重要性，通常$\sum_{i=1}^{m}\omega_i=1$。

7. 需注意的是，通常我们是基于某种形式的距离来定义**相似度度量**(similarity measure)，距离越大，相似度越小。然而，用于相似度度量的距离未必一定要满足距离度量的所有性质，尤其是式(17)。例如在某些任务中我们可能希望有这样的相似度度量："人" "马"分别与"人马"相似，但"人"与"马"很不相似；要达到这个目的，可以领"人" "马"与"人马"之间的距离都比较小，但"人"与"马"之间的距离很大，如下图所示，此时距离不再满足直递性；这样的距离称为**非度量距离**(non-metric distance)。此外，这里介绍的距离计算式都是事先定义好的，但在不少现实任务中，有必要基于数据样本来确定合适的距离计算式，这可通过**距离度量学习**(distance metric learning)来事先。

   <img src="/Users/wangyutian/文档/markdown/pic/K-Means/pic1.jpg" width = 500 height = 400 div align=center />

   

   > 这个例子中，从数学上看，令$d_3=3$即可满足直递性；但从语义上看，$d_3$应远大于$d_1$与$d_2$。

## 二、K-Means算法

1. 给定样本集$D=\left \{ \mathbf x_1,\mathbf x_2,…,\mathbf x_m\right \}$,**k均值**(k-means)算法针对聚类所得划分$C=\left \{C_1,C_2,…,C_k  \right \}$最小化平方误差：
   $$
   E=\sum_{i=1}^{k}\sum_{\mathbf x \in C_i}||\mathbf x-\mathbf \mu_i||_2^2\tag{24}
   $$
   其中$\mathbf \mu_i=\frac{1}{|C_i|}\sum_{x\in C_i}\mathbf x$是簇$C_i$的均值向量

   - E刻画了簇类样本围绕簇均值向量的紧密程度，其值越小，则簇内样本相似度越高。
   - k-means 算法的优化目标为：最小化E。即：$min_c \sum_{i=1}^{k}\sum_{\mathbf x \in C_i}||\mathbf x-\mathbf \mu_i||_2^2$

2. k-means的优化目标需要考察  的所有可能的划分，这是一个NP难的问题。实际上k-means 采用贪心策略，通过迭代优化来近似求解。

   - 首先假设一组均值向量。

   - 然后根据假设的均值向量给出了$D$的一个划分。

   - 再根据这个划分来计算真实的均值向量：

     - 如果真实的均值向量等于假设的均值向量，则说明假设正确。根据假设均值向量给出的D的一个划分确实是原问题的解。
     - 如果真实的均值向量不等于假设的均值向量，则可以将真实的均值向量作为新的假设均值向量，继续迭代求解。

   - 这里的一个关键就是：给定一组假设的均值向量，如何计算出$D$的一个簇划分？

     k均值算法的策略是：样本离哪个簇的均值向量最近，则该样本就划归到那个簇。

3. 算法步骤：

   输入：样本集$D=\left \{ \mathbf x_1,\mathbf x_2,…,\mathbf x_m\right \}$;

   ​		    聚类簇数$k$

   过程：

     1：从$D$中随机选择$k$个样本作为初始均值向量$\left \{ \mathbf \mu_1,\mathbf \mu_2,…,\mathbf \mu_k\right \}$

     2：**repeat**

     3：	令$C_i=\phi \ (1\leqslant i\leqslant k)$

     4：	**for** j = 1,2,…,m **do**

     5：		计算样本$\mathbf x_j$与各均值向量$\mathbf \mu_i \ (1\leqslant i\leqslant k)$的距离$d_{ji}=||\mathbf x_i-\mathbf x_j||_2$

     6：		根据距离最近的均值向量确定$\mathbf x_j$的簇标记：$\lambda_j=argmin_{x\in \left \{1,2,…,k  \right \}}d_{ji}$

     7：		将样本$\mathbf x_j$划入相应的簇：$C_{\lambda_{j}}\bigcup\left \{\mathbf x_j  \right \}$

     8：	**end for**

     9：	**for** i = 1,2,…,k **do**

   10：		计算新均值向量：$\mathbf \mu_i^{'}=\frac{1}{|C_i|}\sum_{x\in C_i}\mathbf x$

   11：		**if** $\mathbf \mu_i^{'}\neq \mathbf \mu_i$ **then**

   12：			将当前均值向量$\mathbf \mu_i$更新为$\mathbf \mu_i^{'}$

   13：		**else**

   14：			保持当前均值向量不变

   15：		**end if**

   16：	**end for**

   17：**until** 当前均值向量均未更新

   **输出：**簇划分$C=\left \{C_1,C_2,…,C_k  \right \}$

4. 前面的k-means中簇的数目k是一个用户预先定义的参数，在k值未知的情况下可使用手肘法求得最佳k值

   - 核心指标SSE(sum of the squared errors，误差平方和)
     $$
     SSE=\sum_{i=1}^{k}\sum_{\mathbf x\in C_i}|\mathbf x-\mu_i|^2\tag{25}
     $$

   - 手肘法核心思想：

     - 随着聚类数k的增大，样本划分会更加精细，每个簇的聚合程度会逐渐提高，那么误差平方和SSE自然会逐渐变小。
     - 当k小于真实聚类数时，由于k的增大会大幅增加每个簇的聚合程度，故SSE的下降幅度会很大，而当k到达真实聚类数时，再增加k所得到的聚合程度回报会迅速变小，所以SSE的下降幅度会骤减，然后随着k值的继续增大而趋于平缓，也就是说SSE和k的关系图是一个手肘的形状，而这个肘部对应的k值就是数据的真实聚类数。

   <img src="/Users/wangyutian/文档/markdown/pic/K-Means/pic2.png" width = 500 height = 400 div align=center />

5. k-means优点：

   - 计算复杂度低，为O(N×K×q)，其中q为迭代次数。

     通常K和q要远远小于N，此时复杂度相当于O(N)。

   - 思想简单，容易实现。

6. k-means缺点：

   - 需要首先确定聚类的数量K。

   - 分类结果严重依赖于分类中心的初始化。

     通常进行多次k-means，然后选择最优的那次作为最终聚类结果。

   - 结果不一定是全局最优的，只能保证局部最优。

   - 对噪声敏感。因为簇的中心是取平均，因此聚类簇很远地方的噪音会导致簇的中心点偏移。

   - 无法解决不规则形状的聚类。

   - 无法处理离散特征，如：国籍、性别等。

7. k-means性质：

   - k-means实际上假设数据是呈现球形分布，实际任务中很少有这种情况。

     与之相比，GMM使用更加一般的数据表示，即高斯分布。

   - k-means 假设各个簇的先验概率相同，但是各个簇的数据量可能不均匀。

   - k-means 使用欧式距离来衡量样本与各个簇的相似度。这种距离实际上假设数据的各个维度对于相似度的作用是相同的。

   - k-means 中，各个样本点只属于与其相似度最高的那个簇，这实际上是硬分簇。

   - k-means 算法的迭代过程实际上等价于EM 算法。

参考

- 《机器学习》
- [AI算法工程师手册](<http://www.huaxiaozhuan.com/%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B9%A0/chapters/11_cluster.html>)
- [kmeans最优k值的确定方法-手肘法和轮廓系数法](<https://www.jianshu.com/p/335b376174d4>)
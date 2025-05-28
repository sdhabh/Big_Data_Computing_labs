   <div style="text-align: center;">
    <img src="E:\Typora\photo\NK.jpg" alt="南开大学校徽" style="border: none; box-shadow: none;">
</div>
<center style="font-size: 1.2rem; line-height: 1.8;">
  <br> <div style="
      font-size: 1.8rem;
      letter-spacing: 0.2em;
      margin-bottom: 1.2rem;
  ">
  计 算 机 学 院
  </div>
  <div style="
      font-size: 1.3rem;
      letter-spacing: 0.3em;
      margin-bottom: 2rem;
  ">
    大数据计算及应用实验报告
  </div>
  <!-- 上方横线 -->
  <hr style="
      width: 60%;
      border: none;
      height: 1px;
      background: #ccc;
      margin: 1.5rem auto 0.5rem;
  ">
  <!-- 标题 -->
  <h1 style="
      font-size: 2rem;
      letter-spacing: 0.12em;
      margin: 0.5rem 0 0.5rem;
      text-align: center;
      line-height: 1.3;
      font-weight: 480;
      color: #34495e;
  ">
期末作业 RecommendationSystems
  </h1>
  <!-- 下方横线 -->
  <hr style="
      width: 60%;
      border: none;
      height: 1px;
      background: #ccc;
      margin: 0.5rem auto 2rem;
  ">
  姓名：胡进喆 原敬闰 张明昆<br>
  学号：2213045 2211771 2211585<br>
  专业：计算机科学与技术 物联网工程<br>
  指导老师：杨征路 <br><br><br><br><br><br><br>
  日期：2025 年 6 月 5 日<br><br><br><br>
</center>




# 摘要





[TOC]









## 实验背景



## 实验原理

**协同过滤推荐**（Collaborative Filtering recommendation）是在信息过滤和信息系统中正迅速成为一项很受欢迎的技术。与传统的基于内容过滤直接分析内容进行 推荐不同，协同过滤分析用户兴趣，在用户群中找到指定用户的相似（兴趣）用户，**综合这些相似用户对某一信息的评价，形成系统对该指定用户对此信息的喜好程度预测。协同过滤算法主要分为基于启发式和基于模型式两种。 其中，**基于启发式的协同过滤算法，又可以分为基于用户的协同过滤算法 （User-Based）和基于项目的协同过滤算法（Item-Based）。



## 数据集说明

我们编写`data_analysis.py`对数据集进行统计分析，得到如下结果：

### train.txt

数据集概览为：

|     属性     | 数值   | 属性             | 数值     |
| :----------: | ------ | ---------------- | -------- |
|   评分数量   | 90,854 |  用户数量   | 598|
| 用户平均评分数量 | 151.93   |物品数量   | 9,077|
| 物品平均评分数量 | 10.01    |用户ID范围       | 1~610    |
|   评分均值   | 69.88  | 数据集稀疏度 | 98.33% |
|评分标准差       | 20.78    |物品ID范围       | 1~193609 |

从用户 ID 范围来看，最小用户 ID 为 1，最大用户 ID 为 610，表明**用户 ID 是连续且紧密分布的**。物品 ID 范围从 1 到 193609，呈现出较大的跨度，与物品数量（9077）对比，**说明物品 ID 存在明显不连续的情况**。每个用户的平均评分数量为 151.93 条，表明数据集中用户参与评分的活跃度较高。反观物品维度，平均每个物品仅获得约 10.01 个评分。这揭示了物品的受欢迎程度分布不均，多数物品可能仅获得少数用户关注，长尾效应显著，部分冷门物品的评分稀缺可能影响推荐算法对这些物品的准确评估。

评分的均值为 69.88，标准差为 20.78，表明评分数据整体上呈现出较为集中的趋势，但也有一定程度的波动。从分位数来看，中位数为 70，25% 分位数为 60，75% 分位数为 80。这暗示着评分分布**可能呈现出一定的正偏态**，即多数评分集中在中高分段，低分段的评分数量相对较少。

评分稀疏度高达 0.9833，即数据矩阵中约 98.33% 的位置是空缺的，**这在推荐系统中属于非常高的稀疏度水平**，给模型训练带来了极大的挑战，需要采用有效的矩阵填充或降维方法。

下表展示了各评分值的出现频率：

| 评分值 | 出现次数 | 评分值 | 出现次数 |
| ------ | -------- | ------ | -------- |
| 10     | 1221     | 60     | 18119    |
| 20     | 2552     | 70     | 11996    |
| 30     | 1603     | 80     | 24177    |
| 40     | 6852     | 90     | 7742     |
| 50     | 5067     | 100    | 11525    |

从上表可以看出，**高分值（如 60、80、100）的出现频率明显高于低分值。**评分 80 出现次数最多，达到 24177 次，占总评分数的约 26.6%，而评分 10 的出现次数最少，仅占约 1.34%。这种评分分布表明用户在评分时更倾向于给出中高分评价，可能反映出数据集整体质量较高，或者存在用户评分偏乐观的倾向。

### test.txt

同样，对测试集进行分析得到如下结果：

|       属性       | 数值   | 属性         | 数值     |
| :--------------: | ------ | ------------ | -------- |
|     评分数量     | 9982 | 用户ID范围 | 1~610 |
|  用户数量 | 610 | 物品ID范围 | 1~193565 |
| 物品数量 | 3618 |  |     |

从数据规模上来看，test集覆盖了完整的用户ID范围（1-610），但包含3618个物品，仅占完整物品ID范围（1-193565）的约1.87%，这表明测试集在物品维度上具有较高的稀疏性，物品覆盖率较低，**这反映了推荐系统在实际应用中面临的冷启动问题**。

因此我们对训练集和测试集进行冷处理分析，得到如下图表：

![物品分布情况](images/distribution_pie_charts.png)

 上述饼图直观展示了训练集与测试集在物品和用户分布上的差异：左图显示，约62.8%的物品仅出现在训练集，30.6%的物品为训练集和测试集共有，仅有6.7%的物品为测试集独有，说明测试集存在比例为17.88%的冷启动物品；右图则表明，绝大多数用户（98.03%）在训练集和测试集中均有出现，只有极少数用户为测试集独有，用户冷启动问题并不突出。整体来看，**推荐系统在物品维度面临更为显著的冷启动挑战，而用户维度的数据分布则较为充分。**



## User-Based CF

User-Based协同过滤（User-Based Collaborative Filtering，简称User-Based CF）是一种基于群体智慧的核心推荐方法，其思想源于"**相似用户可能对未交互项目具有相近的偏好**"。该方法通过分析用户历史行为数据，挖掘用户间的相似性关系，并利用相似用户对目标项目的评分来预测目标用户的潜在兴趣。

### 算法原理

User-Based CF的输入依赖于用户对项目的显式或隐式反馈数据。显式反馈（主动评分）包括用户直接对商品、电影等项目的评分或评价 ，而隐式反馈（被动评分）则通过用户行为（如购买记录、浏览时长、点击率）间接反映兴趣强度。例如，电子商务场景中，用户的购买行为天然构成隐式评分矩阵，其中购买频次或金额可量化为评分值。这些数据被组织为**用户-项目评分矩阵**，矩阵中的**每个元素*Ru,i*表示用户𝑢对项目𝑖的评分**，未评分的项目则作为待预测目标。

#### 用户相似度计算

核心假设是相似用户对同一项目的评分具有一致性。为此，需定义用户间相似性度量方法，常见算法包括：

==**皮尔逊相关系数**==  衡量两位用户评分趋势的线性相关性，通过消除用户评分尺度偏差（如部分用户习惯性打高分或低分）提升相似度准确性。公式为：
$$
\sin(u,v)=\frac{\sum_{i\in I_{uv}}(R_{u,i}-\bar{R}_u)(R_{v,i}-\bar{R}_v)}{\sqrt{\sum_{i\in I_{uv}}(R_{u,i}-\bar{R}_u)^2}\sqrt{\sum_{i\in I_{uv}}(R_{v,i}-\bar{R}_v)^2}}
$$

- 其中，𝐼𝑢𝑣为用户𝑢与𝑣 共同评分的项目集合，𝑅_𝑢 、𝑅_𝑣 为各自的平均评分。皮尔逊系数范围在[-1,1]，值越大表示用户偏好越相似。

 ==**余弦相似度**==  将用户评分视作向量，计算其夹角的余弦值以衡量方向相似性，适合处理稀疏数据。公式为：
$$
\sin(u,v)=\frac{\sum_{i\in I_{uv}}R_{u,i}\cdot R_{v,i}}{\sqrt{\sum_{i\in I_{uv}}R_{u,i}^2}\cdot\sqrt{\sum_{i\in I_{uv}}R_{v,i}^2}}
$$

- 调整后的余弦相似度进一步考虑项目平均评分，消除热门项目的高评分偏差，公式中每个评分减去对应项目的平均分𝑅ˉ𝑖*R*ˉ*i*。

选择相似度方法需结合数据特性：若用户评分存在明显尺度差异（如严格型与宽容型用户），皮尔逊系数更优；若需快速处理高维稀疏数据，余弦法更为高效。

#### 评分预测

确定目标用户𝑢的最近邻集合𝑁(𝑢)（即相似度最高的𝑘个用户）后，预测其对未评分项目𝑖的兴趣分值。预测公式为加权平均：
$$
\hat{R}_{u,i}=\bar{R}_u+\frac{\sum_{v\in N(u)}\sin(u,v)\cdot(R_{v,i}-\bar{R}_v)}{\sum_{v\in N(u)}|\sin(u,v)|}
$$

- 此公式通过引入用户平均评分𝑅_𝑢 、𝑅_𝑣 消除个体评分偏差，并利用相似度作为权重，强调高相似用户的评分影响。最终，系统按预测分值降序推荐Top-N项目给用户。

User-Based CF的优势在于直观性强，能够发现长尾项目，但面临计算复杂度高（用户数远大于项目数）、冷启动（新用户数据稀疏）等挑战。其适用于用户规模相对稳定、用户行为数据丰富的场景（如电商、社交平台），尤其在隐式反馈场景中，通过行为日志构建评分矩阵，可有效捕捉用户偏好动态变化。



### 代码实现

#### 数据预处理与矩阵构建

User-Based协同过滤的数据预处理与矩阵构建过程是推荐系统实现的核心基础。其核心目标是将原始评分数据转化为结构化矩阵表示，为后续的相似度计算和预测提供高效的数据支撑。整个处理流程可分为数据加载、特征工程和矩阵化转换三个阶段：

数据加载阶段采用双层字典结构进行原始数据存储。通过遍历训练文件，以用户行为为中心建立双向索引：**当读取到"用户|..."格式行时，记录当前用户上下文**；后续**非空行解析为<物品ID 评分>键值对，分别存入user_ratings和item_ratings两个defaultdict构成的嵌套字典**。这种设计既保留了用户维度的完整评分记录，又建立了物品维度的反向索引，为后续矩阵构建提供双向查询能力。典型实现如：

```python
with open(train_file) as f:
    for line in f:
        if '|' in line:  # 用户上下文切换
            current_user = int(line.split('|')[0])
        else:  # 评分记录解析
            item, score = map(int, line.split())
            self.user_ratings[current_user][item] = score
            self.item_ratings[item][current_user] = score
```

特征工程阶段着重于数据标准化处理。针对每个用户的评分记录计算均值评分，该均值将用于后续相似度计算时的评分中心化处理（如皮尔逊相关系数需要扣除用户评分偏差）。此处的均值计算采用numpy向量化操作，避免循环带来的性能损耗：

```python
self.mean_ratings = {
    user: np.mean(list(ratings.values())) 
    for user, ratings in self.user_ratings.items()
}
```

矩阵化转换阶段构建用户-物品的二维评分矩阵。首先建立双向索引映射：将原始用户ID和物品ID分别映射为矩阵的行列索引，通过字典结构实现O(1)复杂度的ID查找。随后初始化零值矩阵，遍历用户评分字典填充有效评分。这种稀疏矩阵表示方法既压缩了存储空间，又保持了矩阵运算的便利性：

```python
# 建立索引映射
self.user_to_idx = {user: idx for idx, user in enumerate(users)}
self.item_to_idx = {item: idx for idx, item in enumerate(items)}

# 矩阵填充
for user, ratings in self.user_ratings.items():
    user_idx = self.user_to_idx[user]
    for item, rating in ratings.items():
        if item in self.item_to_idx:  # 过滤未出现在物品索引中的条目
            item_idx = self.item_to_idx[item]
            self.user_item_matrix[user_idx, item_idx] = rating
```

最终形成的用户-物品评分矩阵作为特征空间的基础表示，使得相似度计算可转化为矩阵运算问题。例如余弦相似度通过标准化后的矩阵点积实现，而皮尔逊相关系数则需先进行均值中心化处理。这种矩阵化表示不仅提高了计算效率，更重要的是将推荐问题转化为可量化的向量空间中的邻近度计算问题，为后续的邻居选择和评分预测奠定了数学基础。



#### 相似度计算

User-Based协同过滤的相似度计算本质上是将用户行为数据映射到向量空间，通过量化向量间几何关系来建立用户相似性度量。其数学核心在于构建用户评分向量并定义合适的距离函数，其中余弦相似度与皮尔逊相关系数是两种最具代表性的空间映射方法。

余弦相似度直接采用原始评分向量计算空间夹角余弦值，适用于显式评分数据且不考虑用户评分尺度差异的场景。代码实现借助sklearn的优化矩阵运算：

```python
def cosine_similarity(matrix):
    # 使用sklearn的cosine_similarity函数
    return cosine_similarity(matrix)
```

皮尔逊相关系数则通过统计学视角改进相似度度量，其计算分为三个关键步骤：首先进行均值中心化消除用户评分偏差；随后通过标准差标准化将向量缩放到可比尺度；最终执行协方差计算。该方法的代码实现显式展现其数学本质：

```python
def pearson_similarity(matrix):
    mean = np.mean(matrix, axis=1, keepdims=True)
    centered = matrix - mean
    std = np.sqrt(np.sum(centered ** 2, axis=1, keepdims=True))
    std[std == 0] = 1
    normalized = centered / std
    return np.dot(normalized, normalized.T)
```

两种方法均通过矩阵运算实现高效计算，**其中皮尔逊方法通过中心化处理能更准确捕捉用户间相对评分模式，而余弦相似度保留原始评分绝对值差异。**在工程实现中，需特别注意零方差用户的处理——当用户所有评分相同时，皮尔逊分母为零会导致计算异常，代码通过标准差掩码替换巧妙规避了该问题。相似度矩阵的生成本质上构建了用户关系的拓扑图，其质量直接决定后续近邻选择的准确性，是影响推荐效果最关键的数学变换。



#### 冷启动处理

协同过滤系统在处理冷启动问题时，主要面临两种情况：新用户冷启动和新物品冷启动。在我们的实现中，通过引入全局平均评分（global mean rating）来优雅地处理这两种情况。首先，在模型训练阶段，我们计算并存储全局平均评分：

```python
# 在fit方法中计算全局平均分
all_ratings = []  # 存储所有评分用于计算全局平均分
for user in self.user_ratings:
    for item, score in self.user_ratings[user].items():
        all_ratings.append(score)
self.global_mean_rating = np.mean(all_ratings) if all_ratings else 0
```

在预测阶段，当遇到新用户或新物品时，系统会进行冷启动检查。这个检查过程非常关键，它需要同时验证用户和物品是否在训练集中：

```python
def predict(self, user_id, item_id):
    # 冷启动处理：如果用户或物品不在训练集中，返回全局平均分
    if item_id not in self.item_to_idx or user_id not in self.user_to_idx:
        return self.global_mean_rating
```

这种处理方式的优势在于：全局平均评分能够反映整个评分系统的整体水平，它比单独使用用户平均分或物品平均分更加合理。在预测过程中，如果找不到足够的相似用户或相似物品，系统也会使用全局平均分作为默认值：

```python
# 当没有找到足够的相似用户/物品时
if not similar_users:  # 或 if not similar_items:
    return self.global_mean_rating

# 当相似度加权和为0时
if denominator == 0:
    return self.global_mean_rating
```



#### 模型预测与评估

User-Based协同过滤的预测评分计算是基于近邻用户的加权偏差调整过程，其数学本质是通过相似用户的评分模式对目标用户的潜在偏好进行线性估计。当预测用户u对物品i的评分时，算法首先在用户相似度矩阵中定位目标用户的近邻集合，该集合需满足双重约束：**相似度超过预设阈值且对目标物品有历史评分记录。**通过相似度排序截取Top-N邻居后，**执行偏差修正的加权平均计算——每个邻居的贡献由其与目标用户的相似度加权**，同时扣除该邻居的平均评分偏差以消除个体评分尺度差异。具体实现如代码所示：

```python
# 偏差调整计算核心逻辑
numerator = 0
denominator = 0
for other_user_id, similarity in similar_users:
    rating = self.user_ratings[other_user_id][item_id]
    numerator += similarity * (rating - self.mean_ratings[other_user_id])  # 偏差调整项
    denominator += abs(similarity)  # 相似度正则化

predicted_rating = self.mean_ratings[user_id] + numerator / denominator
```



评估 User-Based CF 模型的性能通常使用验证集来进行。主要的评估指标包括`平均绝对误差`（Mean Absolute Error, MAE）和`均方根误差`（Root Mean Square Error, RMSE）。这些指标能够量化模型预测评分与真实评分之间的差异。

在评估过程中，模型读取验证集文件，逐行处理用户和物品的评分记录。对于每个用户物品对 (u, i)，模型预测用户 u 对物品 i 的评分，并与真实评分进行比较。通过计算预测评分与真实评分的绝对误差及其平方误差，累积所有误差后计算 MAE 和 RMSE。

```python
 with open(validation_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:  # 用户行
                    current_user, _ = line.split('|')
                    current_user = int(current_user)
                else:  # 评分行
                    if line:
                        item_id, true_rating = line.split()
                        item_id = int(item_id)
                        true_rating = int(true_rating)
                        predicted_rating = self.predict(current_user, item_id)
                        error = abs(predicted_rating - true_rating)
                        mae += error
                        rmse += error ** 2
                        count += 1
        if count > 0:
            mae /= count
            rmse = np.sqrt(rmse / count)
```





## Item-Based CF

Item-Based协同过滤（Item-Based CF）的核心假设是**相似物品可能获得同一用户的相近评分**。与User-Based CF的用户相似性驱动不同，其以物品为分析主体，通过挖掘物品间的共现评分模式构建推荐模型。输入数据同样基于用户-物品评分矩阵，但建模焦点转向物品维度，构建**物品-用户评分矩阵**，矩阵元素*R_{i,u}*表示用户𝑢对物品𝑖的评分，未评分项作为预测目标。

### 算法原理

物品相似性度量聚焦于共同评分用户的偏好一致性，主要方法和user_CF相同，分为**余弦相似度**，直接计算物品评分向量的夹角余弦；**皮尔逊相关系数**，消除用户评分偏差，将评分中心化处理。其公式分别为：


$$
\sin(i,j)=\frac{\sum_{u\in U_{ij}}R_{i,u}\cdot R_{j,u}}{\sqrt{\sum R_{i,u}^2}\cdot\sqrt{\sum R_{j,u}^2}}\quad\sin(i,j)=\frac{\sum_{u\in U_{ij}}(R_{i,u}-\bar{R}_u)\cdot(R_{j,u}-\bar{R}_u)}{\sqrt{\sum(R_{i,u}-\bar{R}_u)^2}\cdot\sqrt{\sum(R_{j,u}-\bar{R}_u)^2}}
$$

其中，𝑈𝑖𝑗为同时对物品𝑖和𝑗评分的用户集合，𝑅_𝑢为用户𝑢的平均评分。相似度计算后形成物品相似度矩阵，用于后续邻居选择。预测用户𝑢对目标物品𝑖的评分时，执行两个步骤：1.**相似物品筛选**：从物品相似度矩阵中提取与𝑖最相似的𝑘个物品（阈值过滤后），构成邻居集合𝑁(𝑖)；2.**加权评分聚合**：基于用户𝑢对𝑁(𝑖)中物品的历史评分，计算加权预测值：
$$
   \hat{R}_{u,i}=\bar{R}_i+\frac{\sum_{j\in N(i)}\sin(i,j)\cdot(R_{u,j}-\bar{R}_j)}{\sum|\sin(i,j)|}
$$

其中，𝑅_𝑖为物品𝑖的平均评分，通过引入物品平均分消除热门物品的评分偏差。最终评分通过截断处理约束在合理范围（如0-100）。**冷启动处理**策略与User-Based CF对称：若目标物品或用户未出现在训练集中，直接返回全局平均分；若用户未对任何相似物品评分，则依赖物品平均分或全局均值填补。

Item-Based CF适用于**物品数量相对稳定、用户行为动态性强**的场景（如新闻推荐、短视频流），其优势包括：

- **计算效率高**：物品数通常远小于用户数，相似度矩阵规模更小；
- **实时性优**：物品相似度可离线预计算，在线预测仅需局部加权；
- **隐式反馈适配**：通过用户行为频次构建物品关联，增强可解释性。

与User-Based CF形成互补，二者可根据业务特性选择或融合，以平衡推荐多样性、时效性与计算开销。



### 代码实现

> 注意：ItemCF在代码实现上和UserCF大抵相似，鉴于篇幅，下述只列举部分关键代码

在数据加载与预处理层面，ItemCF首先从训练文件中加载用户-物品评分数据，构建物品-用户评分矩阵。这一步与UserCF类似，但ItemCF更关注物品之间的相似性,构建的是***物品-用户评分矩阵***。

```python
        # 填充评分矩阵
for item in self.item_ratings:
    item_idx = self.item_to_idx[item]
    for user, rating in self.item_ratings[item].items():
        if user in self.user_to_idx:
            user_idx = self.user_to_idx[user]
            self.item_user_matrix[item_idx, user_idx] = rating
```



ItemCF的核心是计算物品之间的相似度。与UserCF不同，ItemCF使用**物品-用户评分矩阵**来计算物品相似度。

```python
# 计算物品相似度矩阵
if self.similarity_method == 'cosine':
    self.item_similarity = cosine_similarity(self.item_user_matrix)
elif self.similarity_method == 'pearson':
    self.item_similarity = pearson_similarity(self.item_user_matrix)
else:
    raise ValueError(f"不支持的相似度计算方法: {self.similarity_method}")
```



ItemCF的预测逻辑与UserCF不同。ItemCF通过用户已评分的物品来预测未评分物品的评分。具体来说，对于目标物品，找到与之最相似的物品，并根据用户对这些相似物品的评分来预测目标物品的评分。

```python
def predict(self, user_id, item_id):
    # 冷启动处理：如果用户或物品不在训练集中，返回全局平均分
    if item_id not in self.item_to_idx or user_id not in self.user_to_idx:
        return self.global_mean_rating

    item_idx = self.item_to_idx[item_id]
    user_idx = self.user_to_idx[user_id]

    # 获取物品的相似物品
    similar_items = []
    for other_item_idx, similarity in enumerate(self.item_similarity[item_idx]):
        if other_item_idx != item_idx and similarity > self.min_similarity:
            other_item_id = self.idx_to_item[other_item_idx]
            if other_item_id in self.user_ratings[user_id]:
                similar_items.append((other_item_id, similarity))

    # 按相似度排序并选择top-N个邻居
    similar_items.sort(key=lambda x: x[1], reverse=True)
    similar_items = similar_items[:self.n_neighbors]

    # 计算预测评分
    numerator = 0
    denominator = 0
    for other_item_id, similarity in similar_items:
        rating = self.user_ratings[user_id][other_item_id]
        numerator += similarity * (rating - self.mean_ratings[other_item_id])
        denominator += abs(similarity)

    predicted_rating = self.mean_ratings[item_id] + numerator / denominator
    return max(0, min(100, predicted_rating))  # 确保评分在0-100之间
```



**总结：**UserCF和ItemCF均基于协同过滤框架，分别通过用户-物品矩阵与物品-用户矩阵建模。UserCF计算用户间余弦/皮尔逊相似度，筛选高相似邻居用户，结合其评分偏差加权预测目标用户对物品的偏好，通过用户平均评分与全局均值处理冷启动；ItemCF则构建物品相似度网络，基于用户历史评分与相似物品的评分偏差聚合预测值，利用物品平均分进行校准。二者均采用MAE/RMSE评估预测误差，核心差异在于UserCF侧重用户行为关联性，而ItemCF聚焦物品共现模式，分别通过矩阵转置实现双向推荐逻辑。

 

## Graph CF

基于图的协同过滤将推荐问题转化为图上的相似度传播问题，**通过随机游走计算用户和物品之间的相似度**，并利用相似度和统计信息进行评分预测。这种方法能够捕获用户-物品交互的复杂结构，从而提供准确的推荐。

### 算法原理

算法将**用户-物品交互关系建模为二部图（bipartite graph）**，通过图上的相似性传播实现推荐。其核心思想是将用户和物品视为图节点，评分行为视为边，利用图结构捕捉高阶相似性。以下是算法的核心原理：

图构建  首先，我们构建一个二部图 `𝐺=(𝑈∪𝐼,𝐸)` ，其中 𝑈 是用户节点集合，𝐼 是物品节点集合，𝐸 是连接用户和物品的边的集合。每条边 (𝑢,𝑖)  表示用户 𝑢 对物品 𝑖 有过评分，边的权重通常与评分值相关。在此次实现中，权重被设计为归一化后的评分的绝对值，并乘以一个与用户和物品评分标准差相关的因子（以考虑评分分布的差异）。具体而言，权重计算如下：
$$
\mathrm{weight}(u,i)=|r_{ui}|\times(1+0.1\times(\sigma_u+\sigma_i))
$$

- 其中 R𝑢𝑖 是归一化后的评分 ，𝜎𝑢 和 𝜎𝑖 分别是用户 𝑢 和物品 𝑖 的评分标准差。这样设计权重可以使得评分分布较广的用户或物品对相似度传播有更大的影响。

转移概率矩阵 构建一个转移概率矩阵 𝑃 来描述图中节点间的转移概率。矩阵 𝑃 的大小为 `(∣𝑈∣+∣𝐼∣)×(∣𝑈∣+∣𝐼∣)` 。对于每个用户节点 𝑢，其指向物品节点 𝑖 的转移概率为：
$$
P_{u\to i}=\frac{\mathrm{weight}(u,i)}{\sum_{j\in N(u)}\mathrm{weight}(u,j)}
$$

- 其中 𝑁(𝑢)  是用户 𝑢 评分过的物品集合。同样，从物品节点 𝑖 指向用户节点 𝑢 的转移概率也类似，此处省略。

随机游走 随机游走过程用于计算节点之间的相似度。这里采用带重启的随机游走（Random Walk with Restart, RWR），其思想是：从一个节点出发，以概率 𝛼 按照转移概率矩阵进行游走，以概率 1−𝛼 跳回起始节点（重启）。通过多次迭代，可以计算一个节点到其他节点的访问概率，这些概率可以解释为节点间的相似度。迭代公式为：
$$
S^{(t+1)}=\alpha\cdot S^{(t)}\cdot P+(1-\alpha)\cdot I
$$

- 其中 𝑆(𝑡) 是第 𝑡 步的相似度矩阵，𝐼 是单位矩阵（表示重启时回到自身）。初始时 𝑆(0)=𝐼 。经过 𝑛 次迭代后，得到的 𝑆(𝑛) 即为最终的相似度矩阵。 这个相似度矩阵 scores 记录了任意两个节点之间的相似度。

评分预测 对于给定的用户 𝑢  和物品 𝑖 ，我们利用随机游走得到的相似度分数 score 进行预测。预测公式综合考虑了用户平均评分、物品平均评分以及相似度分数：
$$
\hat{r}_{ui}=\mu_u+\beta\cdot\mathrm{score}_{u,i}\cdot\sigma_u+\gamma\cdot(\mu_i-\mu_\mathrm{global})
$$

- 其中 μ𝑢 是用户 𝑢  的平均评分，σ𝑢 是用户 𝑢 的评分标准差， μ𝑖 是物品 𝑖  的平均评分， μ*global*  是全局平均评分。在代码中，系数 𝛽  取0.7，𝛾取0.3。最后，将预测评分限制在评分范围内（如0到100）。

基于图的协同过滤能够捕捉**用户和物品之间的高阶关系（通过多步游走）**，而传统的协同过滤（如基于邻域的方法）通常只考虑直接邻居。该方法对稀疏数据有较好的鲁棒性，因为随机游走能够通过多次跳转探索更广泛的连接。需要注意的是，该算法的计算复杂度主要在于转移概率矩阵的构建和随机游走的迭代，当用户和物品数量很大时，矩阵可能非常大，**迭代计算会消耗较多内存和时间**。



### 代码实现

基于图的协同过滤算法的代码实现主要细节如下：

首先，在 `_normalize_ratings` 中，算法先汇集所有评分计算全局均值，然后针对每个用户与每件物品分别计算均值与标准差，保证评分分布信息被保留又能消除量纲影响。这样用 Z-score 归一化后，不会因为某个用户或物品评分方差过小而导致数值爆炸。

```python
all_ratings = [r for ur in self.user_ratings.values() for r in ur.values()]
self.global_mean = np.mean(all_ratings) if all_ratings else 0
for user_id, ratings in self.user_ratings.items():
    self.user_mean_ratings[user_id] = np.mean(ratings.values())
    self.user_std[user_id] = max(np.std(ratings.values()), 1.0)
# 归一化
normalized = (rating - user_mean) / user_std
```



接着在 `_build_graph` 中，构造一个无向二部图，用户和物品分别作为两类节点，通过编号映射 `user_map`、`item_map` 与反向映射保证能在后续矩阵中定位。添加边时，以归一化评分绝对值加权，再略微放大用户与物品评分波动对权重的影响,这种设计既体现了评分强度，又兼顾了评分的离散程度:

```python
weight = abs(rating) * (1 + 0.1 * (self.user_std[user_id] + self.item_std[item_id]))
self.G.add_edge(f"u_{user_id}", f"i_{item_id}", weight=weight)
```



随后在 `_build_transition_matrix` 中，算法在大小为 `(n_users + n_items)` 的方阵中填充转移概率。对每个用户到其评分物品的边，按权重占比归一化，并对称地将概率赋给物品到用户：

```python
total = sum(weights)
self.P[u_idx, i_idx] = weight / total
self.P[i_idx, u_idx] = weight / total
```



核心的随机游走在 `_random_walk` 中迭代完成，融合了重启机制与传播机制，在每次迭代更新中既保留了原始意义的身份矩阵，又不断吸纳网络中新的评分关联：

```python
scores = np.eye(n)
for _ in range(self.n_iter):
    scores = self.alpha * scores.dot(self.P) + (1 - self.alpha) * np.eye(n)
```

经过 `n_iter` 次迭代后，`scores[u, i]` 即衡量了用户与物品间的关联强度。



最后在 `predict` 方法中，当要预测某用户对某物品的评分时，先从 `scores` 中读取二者的随机游走得分，再结合用户自身评分分布与物品相对全局的偏移做线性融合，并裁剪到 `[min_rating, max_rating]` 范围：

```python
score = self.scores[u_idx, i_idx]
pred = user_mean + score * user_std * 0.7 + (item_mean - self.global_mean) * 0.3
pred = max(self.min_rating, min(self.max_rating, pred))
```

这种融合充分利用了全局、用户和物品三个层面的统计信息，既有图结构挖掘的相似度，又兼顾了用户与物品的固有评分倾向。

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
  日期：2025 年 5 月 21 日<br><br><br><br>
</center>



# 摘要





[TOC]









## 实验背景



## 实验原理

**协同过滤推荐**（Collaborative Filtering recommendation）是在信息过滤和信息系统中正迅速成为一项很受欢迎的技术。与传统的基于内容过滤直接分析内容进行 推荐不同，协同过滤分析用户兴趣，在用户群中找到指定用户的相似（兴趣）用 户，**综合这些相似用户对某一信息的评价，形成系统对该指定用户对此信息的喜好程度预测。协同过滤算法主要分为基于启发式和基于模型式两种。 其中，**基于启发式的协同过滤算法，又可以分为基于用户的协同过滤算法 （User-Based）和基于项目的协同过滤算法（Item-Based）。



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







## User-Based CF

> 此部分为胡进喆单独完成

### 算法原理

User-Based协同过滤（User-Based Collaborative Filtering，简称User-Based CF）是一种基于群体智慧的核心推荐方法，其思想源于“**相似用户可能对未交互项目具有相近的偏好**”。该方法通过分析用户历史行为数据，挖掘用户间的相似性关系，并利用相似用户对目标项目的评分来预测目标用户的潜在兴趣。具体原理可分为以下三个关键环节：

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



#### 预测评分计算

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



#### 模型性能评估

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








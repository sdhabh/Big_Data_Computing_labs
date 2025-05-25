import pandas as pd
from algorithm import Dataset, Reader
from algorithm import SVD, SVDpp, NMF, SlopeOne, BaselineOnly
from algorithm.model_selection import train_test_split
from algorithm.utils import accuracy

# 1. 读数据
records = []
with open('data/train.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        line = line.strip()
        if '|' in line:
            user_id, cnt = line.split('|')
            cnt = int(cnt)
            for _ in range(cnt):
                item_id, score = f.readline().strip().split()
                records.append((user_id, item_id, float(score)))

df = pd.DataFrame(records, columns=['user', 'item', 'rating'])

# 2. 构造 Dataset
reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(df[['user','item','rating']], reader)

# 3. 算法列表
algos = [
    ("BaselineOnly",       BaselineOnly()),
    ("SVD",                SVD()),
    ("SVDpp cache=False",  SVDpp(cache_ratings=False)),
    ("SVDpp cache=True",   SVDpp(cache_ratings=True)),
    ("NMF",                NMF()),
    ("SlopeOne",           SlopeOne()),
]

# 4. train_test_split 测试
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

for name, algo in algos:
    print(f"算法: {name}")
    # 训练
    algo.fit(trainset)
    # 预测
    predictions = algo.test(testset)
    # 评估
    rmse = accuracy.rmse(predictions, verbose=False)
    mae  = accuracy.mae(predictions,  verbose=False)
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE : {mae:.4f}\n")

import pandas as pd
import numpy as np
import os
import time
import psutil
import gc
from algorithm import Dataset, Reader
from algorithm import SVD, SVDpp, NMF, SlopeOne, BaselineOnly
from algorithm.model_selection import train_test_split
from algorithm.utils import accuracy

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # 转换为MB

# 创建predictions目录
if not os.path.exists('predictions'):
    os.makedirs('predictions')

# 1. 读训练数据
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
    # ("SVDpp cache=False",  SVDpp(cache_ratings=False)),
    # ("SVDpp cache=True",   SVDpp(cache_ratings=True)),
    ("NMF_Optimized",      NMF(
        n_factors=20,          # 增加因子数量
        n_epochs=100,          # 增加训练轮数
        biased=True,           # 启用偏置项
        reg_pu=0.08,           # 增加用户因子正则化
        reg_qi=0.08,           # 增加物品因子正则化
        reg_bu=0.03,           # 增加用户偏置正则化
        reg_bi=0.03,           # 增加物品偏置正则化
        lr_bu=0.007,           # 调整用户偏置学习率
        lr_bi=0.007,           # 调整物品偏置学习率
        init_low=0.1,          # 调整初始化下界
        init_high=0.9,         # 调整初始化上界
        random_state=42,       # 设置随机种子
        verbose=True           # 启用详细输出
    )),
    ("SlopeOne",           SlopeOne()),
]

# 4. 训练所有模型
trained_models = {}
for name, algo in algos:
    print(f"\n训练算法: {name}")
    
    # 记录训练开始时的内存使用
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    # 训练模型
    algo.fit(data.build_full_trainset())
    
    # 计算训练时间和内存消耗
    training_time = time.time() - start_time
    final_memory = get_memory_usage()
    memory_usage = final_memory - initial_memory
    
    print(f"训练时间: {training_time:.2f} 秒")
    print(f"内存消耗: {memory_usage:.2f} MB")
    
    trained_models[name] = algo
    
    # 清理内存
    gc.collect()

# 5. 处理测试数据并预测
test_records = []
with open('data/test.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        line = line.strip()
        if '|' in line:
            user_id, cnt = line.split('|')
            cnt = int(cnt)
            for _ in range(cnt):
                item_id = f.readline().strip()
                test_records.append((user_id, item_id))

# 6. 对每个模型进行预测并保存结果
for name, model in trained_models.items():
    print(f"\n使用 {name} 进行预测...")
    
    # 记录预测开始时的内存使用
    initial_memory = get_memory_usage()
    start_time = time.time()
    
    predictions = []
    for user_id, item_id in test_records:
        try:
            pred = model.predict(user_id, item_id)
            predictions.append((user_id, item_id, pred.est))
        except:
            predictions.append((user_id, item_id, 0))  # 如果预测失败，使用0作为默认值
    
    # 计算预测时间和内存消耗
    prediction_time = time.time() - start_time
    final_memory = get_memory_usage()
    memory_usage = final_memory - initial_memory
    
    print(f"预测时间: {prediction_time:.2f} 秒")
    print(f"预测内存消耗: {memory_usage:.2f} MB")
    
    # 保存预测结果
    output_file = os.path.join('predictions', f'results_{name.replace(" ", "_")}.txt')
    with open(output_file, 'w') as f:
        current_user = None
        user_items = []
        
        for user_id, item_id, pred in predictions:
            if current_user != user_id:
                if current_user is not None:
                    f.write(f"{current_user}|{len(user_items)}\n")
                    for item_id, score in user_items:
                        f.write(f"{item_id}  {score:.0f}\n")
                current_user = user_id
                user_items = []
            user_items.append((item_id, pred))
        
        # 写入最后一个用户的数据
        if current_user is not None:
            f.write(f"{current_user}|{len(user_items)}\n")
            for item_id, score in user_items:
                f.write(f"{item_id}  {score:.0f}\n")
    
    print(f"预测结果已保存到 {output_file}")
    
    # 清理内存
    gc.collect()

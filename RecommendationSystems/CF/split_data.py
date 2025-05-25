import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_data(file_path):
    users_data = {}
    current_user = None
    current_items = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '|' in line:  # 这是用户行
                if current_user is not None:
                    users_data[current_user] = current_items
                user_id, num_ratings = line.split('|')
                current_user = user_id
                current_items = []
            else:  # 这是物品评分行
                if line:  # 确保不是空行
                    item_id, score = line.split()
                    current_items.append((item_id, score))
    
    # 添加最后一个用户的数据
    if current_user is not None:
        users_data[current_user] = current_items
    
    return users_data

def write_data(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for user_id, items in data.items():
            # 写入用户行
            f.write(f"{user_id}|{len(items)}\n")
            # 写入物品评分行
            for item_id, score in items:
                f.write(f"{item_id}   {score}\n")

import random

# 设置随机种子以确保可重复性
random.seed(42)

print("正在读取数据...")
users_data = read_data('data/train.txt')

print("正在分割数据...")
train_data = {}
val_data = {}

# 对每个用户的数据进行分割
for user_id, items in users_data.items():
    # 随机打乱物品列表
    random.shuffle(items)
    # 计算分割点
    split_point = int(len(items) * 0.8)
    # 分割数据
    train_data[user_id] = items[:split_point]
    val_data[user_id] = items[split_point:]

print("正在保存数据...")
write_data(train_data, 'data/train_split.txt')
write_data(val_data, 'data/val_split.txt')

# 统计信息
total_train_items = sum(len(items) for items in train_data.values())
total_val_items = sum(len(items) for items in val_data.values())

print(f"数据分割完成！")
print(f"用户数量: {len(users_data)}")
print(f"训练集物品数量: {total_train_items}")
print(f"验证集物品数量: {total_val_items}") 
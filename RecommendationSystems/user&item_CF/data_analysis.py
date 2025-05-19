import pandas as pd
import numpy as np
import os

def analyze_dataset(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return
    
    try:
        # 读取数据集
        print(f"正在读取文件：{file_path}")
        
        # 初始化列表存储数据
        data = []
        current_user = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:  # 用户行
                    current_user, num_ratings = line.split('|')
                    current_user = int(current_user)
                else:  # 评分行
                    if line:  # 确保不是空行
                        item_id, score = line.split()
                        data.append({
                            'user_id': current_user,
                            'item_id': int(item_id),
                            'rating': int(score)
                        })
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 打印数据的前几行，用于调试
        print("\n数据预览（前5行）：")
        print(df.head())
        
        # 检查数据是否为空
        if df.empty:
            print("错误：数据集为空")
            return
        
        # 计算基本统计信息
        num_users = df['user_id'].nunique()
        num_items = df['item_id'].nunique()
        num_ratings = len(df)
        
        # 计算用户和物品的ID范围
        min_user_id = df['user_id'].min()
        max_user_id = df['user_id'].max()
        min_item_id = df['item_id'].min()
        max_item_id = df['item_id'].max()
        
        # 检查是否有有效数据
        if num_users == 0 or num_items == 0:
            print("错误：未找到有效的用户或物品数据")
            return
        
        # 计算评分的统计信息
        rating_stats = df['rating'].describe()
        
        # 计算每个用户的平均评分数量
        ratings_per_user = df.groupby('user_id').size()
        avg_ratings_per_user = ratings_per_user.mean()
        
        # 计算每个物品的平均评分数量
        ratings_per_item = df.groupby('item_id').size()
        avg_ratings_per_item = ratings_per_item.mean()
        
        # 计算数据集的稀疏度
        total_possible_ratings = num_users * num_items
        if total_possible_ratings > 0:
            sparsity = 1 - (num_ratings / total_possible_ratings)
        else:
            sparsity = 1.0
        
        print("\n数据集基本信息：")
        print(f"用户数量: {num_users}")
        print(f"物品数量: {num_items}")
        print(f"评分数量: {num_ratings}")
        print(f"\n用户ID范围：")
        print(f"最小用户ID: {min_user_id}")
        print(f"最大用户ID: {max_user_id}")
        print(f"\n物品ID范围：")
        print(f"最小物品ID: {min_item_id}")
        print(f"最大物品ID: {max_item_id}")
        print(f"\n评分统计信息：")
        print(rating_stats)
        print(f"\n每个用户的平均评分数量: {avg_ratings_per_user:.2f}")
        print(f"每个物品的平均评分数量: {avg_ratings_per_item:.2f}")
        print(f"数据集稀疏度: {sparsity:.4f}")
        
        # 额外统计信息
        print("\n评分分布：")
        rating_distribution = df['rating'].value_counts().sort_index()
        print(rating_distribution)
        
    except Exception as e:
        print(f"处理数据时发生错误：{str(e)}")

if __name__ == "__main__":
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(os.path.dirname(current_dir), "user&item_CF", "data", "train.txt")
    analyze_dataset(file_path) 
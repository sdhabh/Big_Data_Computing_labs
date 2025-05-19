import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class UserCF:
    def __init__(self, n_neighbors=20, min_similarity=0):
        self.n_neighbors = n_neighbors  # 邻居数量
        self.min_similarity = min_similarity  # 最小相似度阈值
        self.user_item_matrix = None  # 用户-物品评分矩阵
        self.user_similarity = None  # 用户相似度矩阵
        self.user_ratings = None  # 用户评分字典
        self.item_ratings = None  # 物品评分字典
        self.mean_ratings = None  # 用户平均评分

    def fit(self, train_file):
        """训练模型"""
        # 读取训练数据
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        current_user = None

        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:  # 用户行
                    current_user, _ = line.split('|')
                    current_user = int(current_user)
                else:  # 评分行
                    if line:
                        item_id, score = line.split()
                        item_id = int(item_id)
                        score = int(score)
                        self.user_ratings[current_user][item_id] = score
                        self.item_ratings[item_id][current_user] = score

        # 计算用户平均评分
        self.mean_ratings = {}
        for user in self.user_ratings:
            ratings = list(self.user_ratings[user].values())
            self.mean_ratings[user] = np.mean(ratings)

        # 构建用户-物品评分矩阵
        users = list(self.user_ratings.keys())
        items = list(self.item_ratings.keys())
        self.user_item_matrix = np.zeros((len(users), len(items)))
        
        # 创建用户和物品的索引映射
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        # 填充评分矩阵
        for user in self.user_ratings:
            user_idx = self.user_to_idx[user]
            for item, rating in self.user_ratings[user].items():
                if item in self.item_to_idx:
                    item_idx = self.item_to_idx[item]
                    self.user_item_matrix[user_idx, item_idx] = rating

        # 计算用户相似度矩阵
        self.user_similarity = cosine_similarity(self.user_item_matrix)

    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if item_id not in self.item_to_idx:
            print(f"物品 {item_id} 不在训练集中")
            return self.mean_ratings.get(user_id, 0)  # 如果物品不在训练集中，返回用户平均分
        
        if user_id not in self.user_to_idx:
            print(f"用户 {user_id} 不在训练集中")
            return self.mean_ratings.get(user_id, 0)  # 如果用户不在训练集中，返回用户平均分

        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[item_id]

        # 获取用户的相似用户
        similar_users = []
        for other_user_idx, similarity in enumerate(self.user_similarity[user_idx]):
            if other_user_idx != user_idx and similarity > self.min_similarity:
                other_user_id = self.idx_to_user[other_user_idx]
                if item_id in self.user_ratings[other_user_id]:
                    similar_users.append((other_user_id, similarity))

        # 按相似度排序并选择top-N个邻居
        similar_users.sort(key=lambda x: x[1], reverse=True)
        similar_users = similar_users[:self.n_neighbors]

        if not similar_users:
            return self.mean_ratings.get(user_id, 0)

        # 计算预测评分
        numerator = 0
        denominator = 0
        for other_user_id, similarity in similar_users:
            rating = self.user_ratings[other_user_id][item_id]
            numerator += similarity * (rating - self.mean_ratings[other_user_id])
            denominator += abs(similarity)

        if denominator == 0:
            return self.mean_ratings.get(user_id, 0)

        predicted_rating = self.mean_ratings[user_id] + numerator / denominator
        return max(0, min(100, predicted_rating))  # 确保评分在0-100之间

    def evaluate(self, test_file):
        """评估模型性能"""
        test_data = []
        current_user = None

        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:  # 用户行
                    current_user, _ = line.split('|')
                    current_user = int(current_user)
                else:  # 评分行
                    if line:
                        item_id, score = line.split()
                        test_data.append((current_user, int(item_id), int(score)))

        # 计算预测误差
        mae = 0
        rmse = 0
        count = 0

        for user_id, item_id, true_rating in test_data:
            predicted_rating = self.predict(user_id, item_id)
            error = abs(predicted_rating - true_rating)
            mae += error
            rmse += error ** 2
            count += 1

        mae /= count
        rmse = np.sqrt(rmse / count)

        return mae, rmse 
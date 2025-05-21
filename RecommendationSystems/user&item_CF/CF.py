import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import random

def pearson_similarity(matrix):
    """计算皮尔逊相关系数相似度矩阵"""
    # 计算每个向量的均值
    mean = np.mean(matrix, axis=1, keepdims=True)
    # 中心化
    centered = matrix - mean
    # 计算标准差
    std = np.sqrt(np.sum(centered ** 2, axis=1, keepdims=True))
    # 避免除以0
    std[std == 0] = 1
    # 标准化
    normalized = centered / std
    # 计算相关系数
    return np.dot(normalized, normalized.T)

class UserCF:
    def __init__(self, n_neighbors=20, min_similarity=0, similarity_method='cosine'):
        self.n_neighbors = n_neighbors  # 邻居数量
        self.min_similarity = min_similarity  # 最小相似度阈值
        self.similarity_method = similarity_method  # 相似度计算方法
        self.user_item_matrix = None  # 用户-物品评分矩阵
        self.user_similarity = None  # 用户相似度矩阵
        self.user_ratings = None  # 用户评分字典
        self.item_ratings = None  # 物品评分字典
        self.mean_ratings = None  # 用户平均评分
        self.global_mean_rating = 0  # 全局平均评分

    def fit(self, train_file):
        """训练模型"""
        # 读取训练数据
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        current_user = None
        all_ratings = []  # 存储所有评分用于计算全局平均分

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
                        all_ratings.append(score)

        # 计算全局平均分
        self.global_mean_rating = np.mean(all_ratings) if all_ratings else 0

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
        if self.similarity_method == 'cosine':
            self.user_similarity = cosine_similarity(self.user_item_matrix)
        elif self.similarity_method == 'pearson':
            self.user_similarity = pearson_similarity(self.user_item_matrix)
        else:
            raise ValueError(f"不支持的相似度计算方法: {self.similarity_method}")

    def evaluate(self, validation_file):
        """评估模型性能"""
        mae = 0
        rmse = 0
        count = 0
        current_user = None

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

        return mae, rmse

    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        # 冷启动处理：如果用户或物品不在训练集中，返回全局平均分
        if item_id not in self.item_to_idx or user_id not in self.user_to_idx:
            return self.global_mean_rating

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
            return self.global_mean_rating

        # 计算预测评分
        numerator = 0
        denominator = 0
        for other_user_id, similarity in similar_users:
            rating = self.user_ratings[other_user_id][item_id]
            numerator += similarity * (rating - self.mean_ratings[other_user_id])
            denominator += abs(similarity)

        if denominator == 0:
            return self.global_mean_rating

        predicted_rating = self.mean_ratings[user_id] + numerator / denominator
        return max(0, min(100, predicted_rating))  # 确保评分在0-100之间

class ItemCF:
    def __init__(self, n_neighbors=20, min_similarity=0, similarity_method='cosine'):
        self.n_neighbors = n_neighbors  # 邻居数量
        self.min_similarity = min_similarity  # 最小相似度阈值
        self.similarity_method = similarity_method  # 相似度计算方法
        self.item_user_matrix = None  # 物品-用户评分矩阵
        self.item_similarity = None  # 物品相似度矩阵
        self.user_ratings = None  # 用户评分字典
        self.item_ratings = None  # 物品评分字典
        self.mean_ratings = None  # 物品平均评分
        self.global_mean_rating = 0  # 全局平均评分

    def fit(self, train_file):
        """训练模型"""
        # 读取训练数据
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        current_user = None
        all_ratings = []  # 存储所有评分用于计算全局平均分

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
                        all_ratings.append(score)

        # 计算全局平均分
        self.global_mean_rating = np.mean(all_ratings) if all_ratings else 0

        # 计算物品平均评分
        self.mean_ratings = {}
        for item in self.item_ratings:
            ratings = list(self.item_ratings[item].values())
            self.mean_ratings[item] = np.mean(ratings)

        # 构建物品-用户评分矩阵
        items = list(self.item_ratings.keys())
        users = list(self.user_ratings.keys())
        self.item_user_matrix = np.zeros((len(items), len(users)))
        
        # 创建物品和用户的索引映射
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}
        self.user_to_idx = {user: idx for idx, user in enumerate(users)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}

        # 填充评分矩阵
        for item in self.item_ratings:
            item_idx = self.item_to_idx[item]
            for user, rating in self.item_ratings[item].items():
                if user in self.user_to_idx:
                    user_idx = self.user_to_idx[user]
                    self.item_user_matrix[item_idx, user_idx] = rating

        # 计算物品相似度矩阵
        if self.similarity_method == 'cosine':
            self.item_similarity = cosine_similarity(self.item_user_matrix)
        elif self.similarity_method == 'pearson':
            self.item_similarity = pearson_similarity(self.item_user_matrix)
        else:
            raise ValueError(f"不支持的相似度计算方法: {self.similarity_method}")

    def evaluate(self, validation_file):
        """评估模型性能"""
        mae = 0
        rmse = 0
        count = 0
        current_user = None

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

        return mae, rmse

    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
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

        if not similar_items:
            return self.global_mean_rating

        # 计算预测评分
        numerator = 0
        denominator = 0
        for other_item_id, similarity in similar_items:
            rating = self.user_ratings[user_id][other_item_id]
            numerator += similarity * (rating - self.mean_ratings[other_item_id])
            denominator += abs(similarity)

        if denominator == 0:
            return self.global_mean_rating

        predicted_rating = self.mean_ratings[item_id] + numerator / denominator
        return max(0, min(100, predicted_rating))  # 确保评分在0-100之间 
import numpy as np
from collections import defaultdict

class SlopeOne:
    def __init__(self):
        self.user_ratings = defaultdict(dict)  # 用户评分数据
        self.item_ratings = defaultdict(dict)  # 物品评分数据
        self.diff_matrix = defaultdict(dict)   # 评分差异矩阵
        self.freq_matrix = defaultdict(dict)   # 频率矩阵
        self.global_mean = 0                   # 全局平均评分
        
    def fit(self, train_file):
        """训练模型"""
        # 读取训练数据
        current_user = None
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if '|' in line:  # 用户行
                    current_user, _ = line.split('|')
                    current_user = int(current_user)
                else:  # 评分行
                    parts = line.split()
                    if len(parts) >= 2:
                        item_id = int(parts[0])
                        rating = float(parts[1])
                        self.user_ratings[current_user][item_id] = rating
                        self.item_ratings[item_id][current_user] = rating
        
        # 计算全局平均评分
        all_ratings = [rating for user_ratings in self.user_ratings.values() 
                      for rating in user_ratings.values()]
        self.global_mean = np.mean(all_ratings) if all_ratings else 0
        
        # 优化：只计算有共同用户的物品对之间的差异
        # 1. 首先构建物品-用户倒排表
        item_users = defaultdict(set)
        for user, ratings in self.user_ratings.items():
            for item in ratings:
                item_users[item].add(user)
        
        # 2. 只计算有共同用户的物品对
        for item1, users1 in item_users.items():
            for item2, users2 in item_users.items():
                if item1 < item2:  # 只计算上三角矩阵，避免重复计算
                    common_users = users1 & users2
                    if common_users:  # 只处理有共同用户的物品对
                        # 计算评分差异
                        diffs = [self.item_ratings[item1][u] - self.item_ratings[item2][u] 
                                for u in common_users]
                        mean_diff = np.mean(diffs)
                        freq = len(common_users)
                        
                        # 存储差异和频率
                        self.diff_matrix[item1][item2] = mean_diff
                        self.diff_matrix[item2][item1] = -mean_diff  # 利用对称性
                        self.freq_matrix[item1][item2] = freq
                        self.freq_matrix[item2][item1] = freq  # 频率矩阵也是对称的
    
    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if user_id not in self.user_ratings:
            return self.global_mean
            
        user_ratings = self.user_ratings[user_id]
        if not user_ratings:
            return self.global_mean
            
        # 收集所有可用的预测
        predictions = []
        weights = []
        
        for rated_item, rating in user_ratings.items():
            if rated_item in self.diff_matrix and item_id in self.diff_matrix[rated_item]:
                diff = self.diff_matrix[rated_item][item_id]
                freq = self.freq_matrix[rated_item][item_id]
                predictions.append(rating - diff)
                weights.append(freq)
        
        if not predictions:
            return self.global_mean
            
        # 加权平均
        return np.average(predictions, weights=weights)
    
    def evaluate(self, val_file):
        """评估模型性能"""
        mae_sum = 0
        rmse_sum = 0
        count = 0
        
        current_user = None
        with open(val_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if '|' in line:  # 用户行
                    current_user, _ = line.split('|')
                    current_user = int(current_user)
                else:  # 评分行
                    parts = line.split()
                    if len(parts) >= 2:
                        item_id = int(parts[0])
                        true_rating = float(parts[1])
                        pred_rating = self.predict(current_user, item_id)
                        
                        # 计算误差
                        error = abs(true_rating - pred_rating)
                        mae_sum += error
                        rmse_sum += error ** 2
                        count += 1
        
        if count == 0:
            return 0, 0
            
        mae = mae_sum / count
        rmse = np.sqrt(rmse_sum / count)
        
        return mae, rmse 
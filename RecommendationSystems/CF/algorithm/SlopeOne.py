import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class SlopeOne:
    def __init__(self):
        self.user_ratings = defaultdict(dict)  # 用户评分数据
        self.item_ratings = defaultdict(dict)  # 物品评分数据
        self.dev = None  # 评分差异矩阵
        self.freq = None  # 频率矩阵
        self.user_mean = None  # 用户平均评分
        self.item_to_idx = {}  # 物品ID到索引的映射
        self.idx_to_item = {}  # 索引到物品ID的映射
        self.global_mean = 0  # 全局平均评分

    def _build_item_mapping(self):
        """构建物品ID到索引的映射"""
        items = sorted(self.item_ratings.keys())
        self.item_to_idx = {item: idx for idx, item in enumerate(items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        return len(items)

    def _compute_matrices(self, n_items):
        """计算差异矩阵和频率矩阵"""
        # 初始化矩阵
        freq = np.zeros((n_items, n_items), dtype=np.int_)
        dev = np.zeros((n_items, n_items), dtype=np.float64)
        
        # 计算频率和差异
        for user, ratings in self.user_ratings.items():
            rated_items = list(ratings.items())
            for i, (item1, r1) in enumerate(rated_items):
                if item1 not in self.item_to_idx:
                    continue
                idx1 = self.item_to_idx[item1]
                for item2, r2 in rated_items[i+1:]:
                    if item2 not in self.item_to_idx:
                        continue
                    idx2 = self.item_to_idx[item2]
                    freq[idx1, idx2] += 1
                    freq[idx2, idx1] += 1
                    dev[idx1, idx2] += r1 - r2
                    dev[idx2, idx1] += r2 - r1

        # 计算平均差异
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if freq[i, j] > 0:
                    dev[i, j] /= freq[i, j]
                    dev[j, i] = -dev[i, j]

        self.freq = freq
        self.dev = dev

    def _compute_user_means(self):
        """计算用户平均评分"""
        self.user_mean = {}
        for user, ratings in self.user_ratings.items():
            if ratings:
                self.user_mean[user] = np.mean(list(ratings.values()))
            else:
                self.user_mean[user] = self.global_mean

    def fit(self, train_file):
        """训练模型"""
        # 读取训练数据
        current_user = None
        all_ratings = []
        
        with open(train_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "|" in line:  # 用户行
                    current_user, _ = line.split("|")
                    current_user = int(current_user)
                else:  # 评分行
                    parts = line.split()
                    if len(parts) >= 2:
                        item_id = int(parts[0])
                        rating = float(parts[1])
                        self.user_ratings[current_user][item_id] = rating
                        self.item_ratings[item_id][current_user] = rating
                        all_ratings.append(rating)

        # 计算全局平均评分
        self.global_mean = np.mean(all_ratings) if all_ratings else 0

        # 构建物品映射并计算矩阵
        n_items = self._build_item_mapping()
        self._compute_matrices(n_items)
        self._compute_user_means()

    def fit_from_dict(self, train_dict):
        """训练模型（字典输入）"""
        self.user_ratings = defaultdict(dict)
        self.item_ratings = defaultdict(dict)
        all_ratings = []
        
        for user_id, items in train_dict.items():
            for item_id, rating in items.items():
                self.user_ratings[user_id][item_id] = rating
                self.item_ratings[item_id][user_id] = rating
                all_ratings.append(rating)

        # 计算全局平均评分
        self.global_mean = np.mean(all_ratings) if all_ratings else 0

        # 构建物品映射并计算矩阵
        n_items = self._build_item_mapping()
        self._compute_matrices(n_items)
        self._compute_user_means()

    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if user_id not in self.user_ratings or item_id not in self.item_to_idx:
            return self.global_mean

        user_ratings = self.user_ratings[user_id]
        if not user_ratings:
            return self.global_mean

        item_idx = self.item_to_idx[item_id]
        predictions = []
        weights = []

        # 使用numpy数组进行预测
        for rated_item, rating in user_ratings.items():
            if rated_item in self.item_to_idx:
                rated_idx = self.item_to_idx[rated_item]
                if self.freq[item_idx, rated_idx] > 0:
                    predictions.append(rating + self.dev[item_idx, rated_idx])
                    weights.append(self.freq[item_idx, rated_idx])

        if not predictions:
            return self.user_mean.get(user_id, self.global_mean)

        # 使用numpy进行加权平均计算
        return np.average(predictions, weights=weights)

    def evaluate(self, val_file):
        """评估模型性能"""
        mae_sum = 0
        rmse_sum = 0
        count = 0

        current_user = None
        with open(val_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "|" in line:  # 用户行
                    current_user, _ = line.split("|")
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
                        rmse_sum += error**2
                        count += 1

        if count == 0:
            return 0, 0

        mae = mae_sum / count
        rmse = np.sqrt(rmse_sum / count)

        return mae, rmse

    def evaluate_from_dict(self, val_dict):
        """评估模型性能（字典输入）"""
        errors = []
        for user_id, items in val_dict.items():
            for item_id, true_rating in items.items():
                pred_rating = self.predict(user_id, item_id)
                errors.append(abs(true_rating - pred_rating))
        
        if not errors:
            return 0, 0
            
        errors = np.array(errors)
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(errors**2))
        
        return mae, rmse

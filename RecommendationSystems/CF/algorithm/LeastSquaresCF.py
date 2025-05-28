import numpy as np
from collections import defaultdict, Counter


class LeastSquaresCF:
    def __init__(self, top_n=5):
        # item_weights[item_id] = (other_item_ids, weights)
        self.item_weights = dict()
        self.user_ratings = defaultdict(dict)
        self.global_mean = 0
        self.top_n = top_n

    def fit_from_dict(self, train_dict):
        """
        用最小二乘法拟合每个item的线性组合权重（与base.py一致，只对评分最多的那组other_items做拟合）
        """
        self.user_ratings = defaultdict(dict)
        all_ratings = []
        for user, items in train_dict.items():
            for item, rating in items.items():
                self.user_ratings[user][item] = rating
                all_ratings.append(rating)
        self.global_mean = np.mean(all_ratings) if all_ratings else 0

        # 构建item->user评分表
        item_user = defaultdict(dict)
        for user, items in self.user_ratings.items():
            for item, score in items.items():
                item_user[item][user] = score
        items = list(item_user.keys())

        self.item_weights = dict()
        for target_item in items:
            users = list(item_user[target_item].keys())
            # 统计所有用户评分过的other_items
            other_items_counter = Counter()
            for user in users:
                other_items = [item for item in self.user_ratings[user] if item != target_item]
                other_items_counter.update(other_items)
            # 选出现频率最高的N个other_items
            most_common_items = [item for item, _ in other_items_counter.most_common(self.top_n)]
            if not most_common_items:
                continue
            # 拟合时允许部分缺失，未评分的用全局均值填充
            X = []
            y = []
            for user in users:
                user_ratings = self.user_ratings[user]
                if target_item in user_ratings:
                    x = [user_ratings.get(item, self.global_mean) for item in most_common_items]
                    X.append(x)
                    y.append(user_ratings[target_item])
            if not X:
                continue
            X = np.array(X)
            y = np.array(y)
            try:
                w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                self.item_weights[target_item] = (most_common_items, w)
            except Exception:
                continue

    def predict(self, user_id, item_id):
        """
        用拟合权重预测user_id对item_id的评分（与base.py一致，要求用户评分过所有other_items，否则回退到均值）
        """
        if item_id not in self.item_weights or user_id not in self.user_ratings:
            user_ratings = self.user_ratings.get(user_id, {})
            return (
                np.mean(list(user_ratings.values()))
                if user_ratings
                else self.global_mean
            )
        other_items, w = self.item_weights[item_id]
        user_ratings = self.user_ratings[user_id]
        # 允许部分缺失，未评分的用全局均值填充
        x = np.array([user_ratings.get(i, self.global_mean) for i in other_items])
        pred = float(np.dot(w, x))
        return max(0, min(100, pred))

    def evaluate_from_dict(self, val_dict):
        """
        评估MAE和RMSE
        """
        mae = 0
        rmse = 0
        count = 0
        for user, items in val_dict.items():
            for item, true_rating in items.items():
                pred = self.predict(user, item)
                mae += abs(pred - true_rating)
                rmse += (pred - true_rating) ** 2
                count += 1
        if count == 0:
            return 0, 0
        mae /= count
        rmse = np.sqrt(rmse / count)
        return mae, rmse

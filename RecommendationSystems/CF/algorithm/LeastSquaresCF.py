import numpy as np
from collections import defaultdict, Counter


class LeastSquaresCF:
    def __init__(self):
        # item_weights[item_id] = (other_item_ids, weights)
        self.item_weights = dict()
        self.user_ratings = defaultdict(dict)
        self.global_mean = 0

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
            # 统计每个用户评分过的other_items集合
            itemset_counter = Counter(
                tuple(
                    sorted(
                        [
                            item
                            for item in self.user_ratings[user]
                            if item != target_item
                        ]
                    )
                )
                for user in users
            )
            if not itemset_counter:
                continue
            most_common_items, _ = itemset_counter.most_common(1)[0]
            if not most_common_items:
                continue
            # 只对评分过most_common_items的用户做拟合
            X = []
            y = []
            for user in users:
                user_ratings = self.user_ratings[user]
                if (
                    all(item in user_ratings for item in most_common_items)
                    and target_item in user_ratings
                ):
                    x = [user_ratings[item] for item in most_common_items]
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
        if all(i in user_ratings for i in other_items):
            x = np.array([user_ratings[i] for i in other_items])
            pred = float(np.dot(w, x))
            return max(0, min(100, pred))
        else:
            return (
                np.mean(list(user_ratings.values()))
                if user_ratings
                else self.global_mean
            )

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

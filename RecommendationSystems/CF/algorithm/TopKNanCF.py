import numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor


class TopKNanCF:
    def __init__(self, topk=10):
        self.topk = topk
        self.models = dict()
        self.topk_dict = dict()
        self.user_ratings = defaultdict(dict)
        self.global_mean = 0

    def fit_from_dict(self, train_dict):
        """
        用于训练Top-K+GBDT模型，允许特征缺失
        """
        self.user_ratings = defaultdict(dict)
        all_ratings = []
        for user, items in train_dict.items():
            for item, rating in items.items():
                self.user_ratings[user][item] = rating
                all_ratings.append(rating)
        self.global_mean = np.mean(all_ratings) if all_ratings else 0

        # 计算物品两两之间的余弦相似度
        item_user = defaultdict(dict)
        for user, items in self.user_ratings.items():
            for item, score in items.items():
                item_user[item][user] = score
        items = list(item_user.keys())
        sim_matrix = dict()
        for i in range(len(items)):
            sim_matrix[items[i]] = dict()
            for j in range(len(items)):
                if i == j:
                    continue
                users_i = item_user[items[i]]
                users_j = item_user[items[j]]
                common_users = set(users_i.keys()) & set(users_j.keys())
                if not common_users:
                    continue
                vi = np.array([users_i[u] for u in common_users])
                vj = np.array([users_j[u] for u in common_users])
                num = np.dot(vi, vj)
                denom = np.linalg.norm(vi) * np.linalg.norm(vj)
                sim = num / denom if denom != 0 else 0
                sim_matrix[items[i]][items[j]] = sim

        # 训练每个item的GBDT模型
        self.models = dict()
        self.topk_dict = dict()
        for target_item in items:
            sim_items = sorted(
                sim_matrix.get(target_item, {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )
            topk_items = [item for item, _ in sim_items[: self.topk]]
            self.topk_dict[target_item] = topk_items
            users = list(item_user[target_item].keys())
            X = []
            y = []
            for user in users:
                user_ratings = self.user_ratings[user]
                x = []
                for item in topk_items:
                    x.append(user_ratings[item] if item in user_ratings else np.nan)
                X.append(x)
                y.append(user_ratings[target_item])
            if not X:
                continue
            X = np.array(X)
            y = np.array(y)
            try:
                model = HistGradientBoostingRegressor(max_iter=100, random_state=42)
                model.fit(X, y)
                self.models[target_item] = model
            except Exception:
                continue

    def predict(self, user_id, item_id):
        """
        用训练好的GBDT模型预测用户对item的评分
        """
        if item_id not in self.models or user_id not in self.user_ratings:
            user_ratings = self.user_ratings.get(user_id, {})
            return (
                np.mean(list(user_ratings.values()))
                if user_ratings
                else self.global_mean
            )
        user_ratings = self.user_ratings[user_id]
        topk_items = self.topk_dict[item_id]
        x = []
        for i in topk_items:
            x.append(user_ratings[i] if i in user_ratings else np.nan)
        x = np.array(x).reshape(1, -1)
        try:
            pred = self.models[item_id].predict(x)[0]
        except Exception:
            pred = (
                np.mean(list(user_ratings.values()))
                if user_ratings
                else self.global_mean
            )
        return pred

    def evaluate_from_dict(self, val_dict):
        """
        评估MAE和RMSE
        """
        preds = []
        reals = []
        for user, items in val_dict.items():
            for item, real_score in items.items():
                pred = self.predict(user, item)
                preds.append(pred)
                reals.append(real_score)
        preds = np.array(preds)
        reals = np.array(reals)
        if len(preds) == 0:
            return 0, 0
        mae = np.mean(np.abs(preds - reals))
        rmse = np.sqrt(np.mean((preds - reals) ** 2))
        return mae, rmse

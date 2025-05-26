import numpy as np
from collections import defaultdict


class GDLinearCF:
    def __init__(self, topk=10, lr=0.0005, epochs=500, reg_lambda=0.01):
        self.topk = topk
        self.lr = lr
        self.epochs = epochs
        self.reg_lambda = reg_lambda
        # 以下为训练后保存的参数
        self.w = None
        self.b = None
        self.topk_dict = None
        self.X_mean = None
        self.X_std = None
        self.y_mean = None
        self.y_std = None
        self.item_mean = None
        self.global_mean = None
        self.user_ratings = defaultdict(dict)

    def fit_from_dict(self, train_dict):
        self.user_ratings = defaultdict(dict)
        all_ratings = []
        for user, items in train_dict.items():
            for item, rating in items.items():
                self.user_ratings[user][item] = rating
                all_ratings.append(rating)
        self.global_mean = np.mean(all_ratings) if all_ratings else 0

        # 构建item_user表
        item_user = defaultdict(dict)
        for user, items in self.user_ratings.items():
            for item, score in items.items():
                item_user[item][user] = score

        # 计算相似度矩阵和topk
        sim_matrix = defaultdict(dict)
        items = list(item_user.keys())
        for i, item_i in enumerate(items):
            for j in range(i + 1, len(items)):
                item_j = items[j]
                common_users = set(item_user[item_i].keys()) & set(
                    item_user[item_j].keys()
                )
                if not common_users:
                    continue
                v1 = np.array([item_user[item_i][u] for u in common_users])
                v2 = np.array([item_user[item_j][u] for u in common_users])
                num = np.dot(v1, v2)
                denom = np.linalg.norm(v1) * np.linalg.norm(v2)
                sim = num / denom if denom != 0 else 0
                if sim > 0:
                    sim_matrix[item_i][item_j] = sim
                    sim_matrix[item_j][item_i] = sim
        self.topk_dict = {}
        for item, sims in sim_matrix.items():
            sorted_items = sorted(sims.items(), key=lambda x: x[1], reverse=True)
            self.topk_dict[item] = [i for i, _ in sorted_items[: self.topk]]

        # 计算item均值
        self.item_mean = {
            item: np.mean(list(users.values())) for item, users in item_user.items()
        }

        # 构建训练样本
        X = []
        y = []
        for user, items in self.user_ratings.items():
            user_mean = np.mean(list(items.values())) if items else self.global_mean
            for target_item, target_score in items.items():
                topk_items = self.topk_dict.get(target_item, [])
                features = []
                for sim_item in topk_items:
                    features.append(
                        items.get(
                            sim_item, self.item_mean.get(sim_item, self.global_mean)
                        )
                    )
                if len(features) < self.topk:
                    features += [self.global_mean] * (self.topk - len(features))
                features.append(user_mean)
                features.append(self.item_mean.get(target_item, self.global_mean))
                features.append(self.global_mean)
                X.append(features)
                y.append(target_score)
        X = np.array(X)
        y = np.array(y)

        # 归一化
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.X_std[self.X_std == 0] = 1
        X = (X - self.X_mean) / self.X_std
        self.y_mean = y.mean()
        self.y_std = y.std() + 1e-8
        y = (y - self.y_mean) / self.y_std

        # 梯度下降训练
        n_samples, n_features = X.shape
        w = np.random.randn(n_features) * 0.01
        b = 0
        max_grad = 1.0
        for epoch in range(self.epochs):
            current_lr = self.lr * (0.95**epoch)
            y_pred = X @ w + b
            error = y_pred - y
            grad_w = np.clip((X.T @ error) / n_samples, -max_grad, max_grad)
            grad_b = np.clip(np.mean(error), -max_grad, max_grad)
            grad_w += self.reg_lambda * w
            w -= current_lr * grad_w
            b -= current_lr * grad_b
            # 可选：打印loss
            # if epoch % 50 == 0:
            #     loss = np.mean(error**2) + 0.5 * self.reg_lambda * np.sum(w**2)
            #     print(f"Epoch {epoch}, Loss: {loss:.4f}")
        self.w = w
        self.b = b

    def predict(self, user_id, item_id):
        if self.w is None or self.b is None or self.topk_dict is None:
            return self.global_mean
        items = self.user_ratings.get(user_id, {})
        user_mean = np.mean(list(items.values())) if items else self.global_mean
        topk_items = self.topk_dict.get(item_id, [])
        features = []
        for sim_item in topk_items:
            features.append(
                items.get(sim_item, self.item_mean.get(sim_item, self.global_mean))
            )
        if len(features) < self.topk:
            features += [self.global_mean] * (self.topk - len(features))
        features.append(user_mean)
        features.append(self.item_mean.get(item_id, self.global_mean))
        features.append(self.global_mean)
        features = np.array(features)
        features = (features - self.X_mean) / self.X_std
        pred = float(features @ self.w + self.b)
        pred = pred * self.y_std + self.y_mean
        return max(0, min(100, pred))

    def evaluate_from_dict(self, val_dict):
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

import numpy as np
from collections import defaultdict
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import sys

class GraphCF:
    def __init__(self, alpha=0.85, n_iter=20, min_rating=0, max_rating=100):
        """
        基于图的协同过滤算法
        
        参数:
            alpha: 随机游走的重启概率，控制随机游走的范围
            n_iter: 迭代次数，控制随机游走的深度
            min_rating: 最小评分值（默认0）
            max_rating: 最大评分值（默认100）
        """
        self.alpha = alpha
        self.n_iter = n_iter
        self.min_rating = min_rating
        self.max_rating = max_rating
        self.user_ratings = defaultdict(dict)  # 用户评分数据
        self.item_ratings = defaultdict(dict)  # 物品评分数据
        self.user_map = {}  # 用户ID到索引的映射
        self.item_map = {}  # 物品ID到索引的映射
        self.reverse_user_map = {}  # 索引到用户ID的映射
        self.reverse_item_map = {}  # 物品ID到索引的映射
        self.G = nx.Graph()  # 二部图
        self.P = None  # 转移概率矩阵
        self.scores = None  # 预测分数矩阵
        self.user_mean_ratings = {}  # 用户平均评分
        self.item_mean_ratings = {}  # 物品平均评分
        self.global_mean = 0  # 全局平均评分
        self.user_std = {}  # 用户评分标准差
        self.item_std = {}  # 物品评分标准差
        
    def _normalize_ratings(self):
        """归一化评分数据，保持评分的相对差异"""
        print("正在归一化评分数据...")
        # 计算全局平均评分
        all_ratings = []
        for user_ratings in self.user_ratings.values():
            all_ratings.extend(user_ratings.values())
        self.global_mean = np.mean(all_ratings) if all_ratings else 0
        
        # 计算用户统计信息
        for user_id, ratings in self.user_ratings.items():
            ratings_list = list(ratings.values())
            self.user_mean_ratings[user_id] = np.mean(ratings_list)
            # 处理标准差为0的情况
            std = np.std(ratings_list)
            self.user_std[user_id] = std if std > 0 else 1.0
        
        # 计算物品统计信息
        for item_id, ratings in self.item_ratings.items():
            ratings_list = list(ratings.values())
            self.item_mean_ratings[item_id] = np.mean(ratings_list)
            # 处理标准差为0的情况
            std = np.std(ratings_list)
            self.item_std[item_id] = std if std > 0 else 1.0
        
        # 归一化评分，保持相对差异
        normalized_user_ratings = defaultdict(dict)
        normalized_item_ratings = defaultdict(dict)
        
        for user_id, ratings in self.user_ratings.items():
            user_mean = self.user_mean_ratings[user_id]
            user_std = self.user_std[user_id]
            
            for item_id, rating in ratings.items():
                # 使用Z-score归一化，保持评分的相对差异
                # 当标准差为0时，使用简单的中心化
                if user_std > 0:
                    normalized_rating = (rating - user_mean) / user_std
                else:
                    normalized_rating = rating - user_mean
                normalized_user_ratings[user_id][item_id] = normalized_rating
                normalized_item_ratings[item_id][user_id] = normalized_rating
        
        self.user_ratings = normalized_user_ratings
        self.item_ratings = normalized_item_ratings
        
    def _build_graph(self):
        """构建用户-物品二部图"""
        print("正在构建用户-物品二部图...")
        # 添加用户节点
        for user_id in self.user_ratings:
            if user_id not in self.user_map:
                idx = len(self.user_map)
                self.user_map[user_id] = idx
                self.reverse_user_map[idx] = user_id
                self.G.add_node(f'u_{user_id}', bipartite=0)
        
        # 添加物品节点
        for item_id in self.item_ratings:
            if item_id not in self.item_map:
                idx = len(self.item_map)
                self.item_map[item_id] = idx
                self.reverse_item_map[idx] = item_id
                self.G.add_node(f'i_{item_id}', bipartite=1)
        
        # 添加边（评分关系）
        for user_id, ratings in self.user_ratings.items():
            for item_id, rating in ratings.items():
                # 使用评分的绝对值作为边的权重，并考虑用户和物品的评分分布
                weight = abs(rating) * (1 + 0.1 * (self.user_std[user_id] + self.item_std[item_id]))
                self.G.add_edge(f'u_{user_id}', f'i_{item_id}', weight=weight)
        
    def _build_transition_matrix(self):
        """构建转移概率矩阵"""
        print("正在构建转移概率矩阵...")
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        n = n_users + n_items
        
        # 初始化转移概率矩阵
        self.P = np.zeros((n, n))
        
        # 计算转移概率
        for user_id, ratings in self.user_ratings.items():
            u_idx = self.user_map[user_id]
            weights = []
            items = []
            
            for item_id, rating in ratings.items():
                i_idx = self.item_map[item_id] + n_users
                # 考虑用户和物品的评分分布
                weight = abs(rating) * (1 + 0.1 * (self.user_std[user_id] + self.item_std[item_id]))
                weights.append(weight)
                items.append(i_idx)
            
            total_weight = sum(weights)
            if total_weight > 0:
                for i, item_idx in enumerate(items):
                    weight = weights[i] / total_weight
                    # 用户到物品的转移概率
                    self.P[u_idx, item_idx] = weight
                    # 物品到用户的转移概率
                    self.P[item_idx, u_idx] = weight
        
    def _random_walk(self):
        """执行随机游走"""
        print(f"开始随机游走迭代（共 {self.n_iter} 次）...")
        n = len(self.P)
        scores = np.eye(n)  # 初始状态
        
        for i in range(self.n_iter):
            # 随机游走更新，增加分数传播的多样性
            scores = self.alpha * np.dot(scores, self.P) + (1 - self.alpha) * np.eye(n)
            
            # 显示进度
            progress = (i + 1) / self.n_iter * 100
            sys.stdout.write(f"\r迭代进度: {progress:.1f}% ({i + 1}/{self.n_iter})")
            sys.stdout.flush()
        
        print("\n随机游走完成！")
        self.scores = scores
    
    def fit(self, train_file):
        """训练模型"""
        print("开始读取训练数据...")
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
        
 
        
        # 归一化评分
        self._normalize_ratings()
        
        # 构建图
        self._build_graph()
        
        # 构建转移概率矩阵
        self._build_transition_matrix()
        
        # 执行随机游走
        self._random_walk()
    
    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if user_id not in self.user_map or item_id not in self.item_map:
            return self.global_mean
            
        u_idx = self.user_map[user_id]
        i_idx = self.item_map[item_id] + len(self.user_map)
        
        # 获取用户-物品的相似度分数
        score = self.scores[u_idx, i_idx]
        
        # 获取用户和物品的统计信息
        user_mean = self.user_mean_ratings.get(user_id, self.global_mean)
        user_std = self.user_std.get(user_id, 1.0)
        item_mean = self.item_mean_ratings.get(item_id, self.global_mean)
        item_std = self.item_std.get(item_id, 1.0)
        
        # 综合考虑用户和物品的评分分布
        predicted_rating = user_mean + score * user_std * 0.7 + (item_mean - self.global_mean) * 0.3
        
        # 限制预测评分在有效范围内
        predicted_rating = max(self.min_rating, min(self.max_rating, predicted_rating))
        
        return predicted_rating
    
    def evaluate(self, val_file):
        """评估模型性能"""
        print("\n开始评估模型性能...")
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
        
        print(f"评估完成！共处理 {count} 个评分")
        return mae, rmse 
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithm'))
from GraphCF import GraphCF

# 数据文件路径
train_file = os.path.join(os.path.dirname(__file__), 'data', 'train_split.txt')
val_file = os.path.join(os.path.dirname(__file__), 'data', 'val_split.txt')

# 超参数范围
alpha_list = [0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
score_weight_list = [0.1, 0.3, 0.5, 0.7, 0.9]

mae_matrix = np.zeros((len(alpha_list), len(score_weight_list)))
rmse_matrix = np.zeros((len(alpha_list), len(score_weight_list)))

print('开始GraphCF alpha和score_weight自动调参实验...')
for i, alpha in enumerate(alpha_list):
    for j, score_weight in enumerate(score_weight_list):
        print(f'GraphCF alpha={alpha}, score_weight={score_weight}...')
        # 修改GraphCF的预测融合方式
        class GraphCFCustom(GraphCF):
            def predict(self, user_id, item_id):
                if user_id not in self.user_map or item_id not in self.item_map:
                    return self.global_mean
                u_idx = self.user_map[user_id]
                i_idx = self.item_map[item_id] + len(self.user_map)
                score = self.scores[u_idx, i_idx]
                user_mean = self.user_mean_ratings.get(user_id, self.global_mean)
                user_std = self.user_std.get(user_id, 1.0)
                item_mean = self.item_mean_ratings.get(item_id, self.global_mean)
                # 融合权重
                pred = user_mean + score * user_std * score_weight + (item_mean - self.global_mean) * (1-score_weight)
                pred = max(self.min_rating, min(self.max_rating, pred))
                return pred
        model = GraphCFCustom(alpha=alpha, n_iter=20)
        model.fit(train_file)
        # 评估
        from collections import defaultdict
        def read_data_to_dict(file_path):
            user_item_score = defaultdict(dict)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line or "|" not in line:
                    i += 1
                    continue
                user_id, num = line.split("|")
                user_id = int(user_id)
                num = int(num)
                for j in range(i + 1, i + 1 + num):
                    item_id, score = lines[j].strip().split()
                    user_item_score[user_id][int(item_id)] = int(score)
                i += num + 1
            return user_item_score
        val_dict = read_data_to_dict(val_file)
        mae, rmse, count = 0, 0, 0
        for user_id, items in val_dict.items():
            for item_id, true_rating in items.items():
                pred = model.predict(user_id, item_id)
                error = abs(pred - true_rating)
                mae += error
                rmse += error ** 2
                count += 1
        if count > 0:
            mae /= count
            rmse = np.sqrt(rmse / count)
        mae_matrix[i, j] = mae
        rmse_matrix[i, j] = rmse
        print(f'  MAE={mae:.4f}, RMSE={rmse:.4f}')

# 输出最优结果
min_mae = np.min(mae_matrix)
min_rmse = np.min(rmse_matrix)
mae_pos = np.unravel_index(np.argmin(mae_matrix), mae_matrix.shape)
rmse_pos = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
print(f'最优MAE: {min_mae:.4f} (alpha={alpha_list[mae_pos[0]]}, score_weight={score_weight_list[mae_pos[1]]})')
print(f'最优RMSE: {min_rmse:.4f} (alpha={alpha_list[rmse_pos[0]]}, score_weight={score_weight_list[rmse_pos[1]]})')

# 绘制热力图
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
im1 = ax[0].imshow(mae_matrix, cmap='YlGnBu', aspect='auto', origin='lower')
ax[0].set_xticks(np.arange(len(score_weight_list)))
ax[0].set_yticks(np.arange(len(alpha_list)))
ax[0].set_xticklabels([str(x) for x in score_weight_list])
ax[0].set_yticklabels([str(x) for x in alpha_list])
ax[0].set_xlabel('score_weight')
ax[0].set_ylabel('alpha')
ax[0].set_title('MAE热力图')
for i in range(len(alpha_list)):
    for j in range(len(score_weight_list)):
        ax[0].text(j, i, f'{mae_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=9)
fig.colorbar(im1, ax=ax[0])

im2 = ax[1].imshow(rmse_matrix, cmap='YlOrRd', aspect='auto', origin='lower')
ax[1].set_xticks(np.arange(len(score_weight_list)))
ax[1].set_yticks(np.arange(len(alpha_list)))
ax[1].set_xticklabels([str(x) for x in score_weight_list])
ax[1].set_yticklabels([str(x) for x in alpha_list])
ax[1].set_xlabel('score_weight')
ax[1].set_ylabel('alpha')
ax[1].set_title('RMSE热力图')
for i in range(len(alpha_list)):
    for j in range(len(score_weight_list)):
        ax[1].text(j, i, f'{rmse_matrix[i, j]:.2f}', ha='center', va='center', color='black', fontsize=9)
fig.colorbar(im2, ax=ax[1])

plt.suptitle('GraphCF不同alpha和score_weight下的MAE与RMSE热力图', fontsize=15, color='#34495E')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('graphcf_param_heatmap.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('实验完成，热力图已保存为 graphcf_param_heatmap.png') 
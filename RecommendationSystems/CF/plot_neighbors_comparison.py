import matplotlib.pyplot as plt
import numpy as np
import sys
import os
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithm'))
from UserCF import UserCF
from ItemCF import ItemCF

# 数据文件路径
train_file = os.path.join(os.path.dirname(__file__), 'data', 'train_split.txt')
val_file = os.path.join(os.path.dirname(__file__), 'data', 'val_split.txt')

n_neighbors_list = [5, 10, 15, 20, 30, 40, 50]
usercf_mae, usercf_rmse = [], []
itemcf_mae, itemcf_rmse = [], []

print('开始批量实验...')
for n in n_neighbors_list:
    print(f'UserCF n_neighbors={n}...')
    usercf = UserCF(n_neighbors=n, similarity_method="cosine")
    usercf.fit(train_file)
    mae, rmse = usercf.evaluate(val_file)
    usercf_mae.append(mae)
    usercf_rmse.append(rmse)
    print(f'  MAE={mae:.4f}, RMSE={rmse:.4f}')

    print(f'ItemCF n_neighbors={n}...')
    itemcf = ItemCF(n_neighbors=n, similarity_method="cosine")
    itemcf.fit(train_file)
    mae, rmse = itemcf.evaluate(val_file)
    itemcf_mae.append(mae)
    itemcf_rmse.append(rmse)
    print(f'  MAE={mae:.4f}, RMSE={rmse:.4f}')

plt.figure(figsize=(10, 6))

# 绘制MAE和RMSE在同一张图
plt.plot(n_neighbors_list, usercf_mae, marker='o', label='UserCF MAE', color='#5DADE2', linewidth=2)
plt.plot(n_neighbors_list, usercf_rmse, marker='o', label='UserCF RMSE', color='#2874A6', linestyle='--', linewidth=2)
plt.plot(n_neighbors_list, itemcf_mae, marker='s', label='ItemCF MAE', color='#F5B041', linewidth=2)
plt.plot(n_neighbors_list, itemcf_rmse, marker='s', label='ItemCF RMSE', color='#B9770E', linestyle='--', linewidth=2)

# 标注数据点
for x, y in zip(n_neighbors_list, usercf_mae):
    plt.text(x, y+0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=9, color='#34495E')
for x, y in zip(n_neighbors_list, usercf_rmse):
    plt.text(x, y+0.05, f'{y:.2f}', ha='center', va='bottom', fontsize=9, color='#34495E')
for x, y in zip(n_neighbors_list, itemcf_mae):
    plt.text(x, y-0.15, f'{y:.2f}', ha='center', va='top', fontsize=9, color='#7D6608')
for x, y in zip(n_neighbors_list, itemcf_rmse):
    plt.text(x, y-0.15, f'{y:.2f}', ha='center', va='top', fontsize=9, color='#7D6608')

plt.xlabel('邻居数量 n_neighbors', fontsize=12)
plt.ylabel('误差值', fontsize=12)
plt.title('UserCF与ItemCF在不同邻居数量下的MAE与RMSE对比', fontsize=15, pad=15, color='#34495E')
plt.legend(fontsize=11, frameon=True, facecolor='white', edgecolor='none')
plt.grid(True, linestyle='--', alpha=0.25)
plt.tight_layout()
plt.savefig('neighbors_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print('实验完成，图表已保存为 neighbors_comparison.png') 
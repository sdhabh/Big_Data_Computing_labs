import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 设置matplotlib样式
plt.style.use('bmh')

# 数据准备
models = ['UserCF', 'ItemCF', 'SlopeOne', 'GraphCF', 'TopKNanCF', 'GDLinearCF', 'LeastSquaresCF']
mae_values = [13.50, 13.08, 13.41, 13.94, 14.79, 16.46, 16.65]
rmse_values = [17.81, 17.17, 17.59, 18.06, 19.15, 20.70, 22.00]

# 创建图形
plt.figure(figsize=(12, 7))

# 设置柱状图的位置
x = np.arange(len(models))
width = 0.35

# 创建柱状图 - 使用柔和的配色
plt.bar(x - width/2, mae_values, width, label='MAE', color='#7FB3D5', alpha=0.85)  # 淡蓝色
plt.bar(x + width/2, rmse_values, width, label='RMSE', color='#F5B041', alpha=0.85)  # 淡橙色

# 添加数据标签
for i, v in enumerate(mae_values):
    plt.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=10, color='#34495E')
for i, v in enumerate(rmse_values):
    plt.text(i + width/2, v + 0.1, f'{v:.2f}', ha='center', va='bottom', fontsize=10, color='#34495E')

# 设置图表属性
plt.title('不同推荐模型性能对比', fontsize=16, pad=20, color='#34495E')
plt.xlabel('模型', fontsize=12, color='#34495E', labelpad=10)
plt.ylabel('误差值', fontsize=12, color='#34495E', labelpad=10)
plt.xticks(x, models, rotation=45, color='#34495E')
plt.yticks(color='#34495E')

# 设置图例
plt.legend(fontsize=10, frameon=True, facecolor='white', edgecolor='none')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.2)

# 设置背景色
plt.gca().set_facecolor('#F8F9F9')  # 更淡的背景色
plt.gcf().set_facecolor('white')

# 调整布局
plt.tight_layout()

# 保存图表
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close() 
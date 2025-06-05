import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager
import os

# 设置中文字体
try:
    # 尝试使用微软雅黑
    font_path = 'C:/Windows/Fonts/msyh.ttc'
    font_prop = font_manager.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except:
    # 如果找不到微软雅黑，尝试使用系统默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
models = ['UserCF', 'ItemCF', 'SlopeOne', 'GraphCF', 'TopKNanCF', 'GDLinearCF', 'LeastSquaresCF']
time_used = [0.29, 0.90, 31.02, 284.04, 516.64, 117.79, 5.05]  # 训练时间（秒）
memory_used = [61.27, 628.15, 1060.57, 593.70, 713.55, 17.39, 7.00]  # 内存消耗（MB）

# 创建图形和轴对象
fig, ax1 = plt.subplots(figsize=(12, 6))

# 设置样式
sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

# 设置背景色
ax1.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

# 绘制柱状图（内存使用）
bars = ax1.bar(models, memory_used, color='#4e79a7', alpha=0.7, label='内存消耗 (MB)')
ax1.set_ylabel('内存消耗 (MB)', fontsize=12, color='#4e79a7', fontproperties=font_prop)
ax1.tick_params(axis='y', labelcolor='#4e79a7')

# 在柱状图上添加数值标签
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}',
             ha='center', va='bottom', color='#4e79a7', fontsize=9)

# 创建第二个y轴（训练时间）
ax2 = ax1.twinx()
line = ax2.plot(models, time_used, color='#e15759', marker='o', linewidth=2, markersize=8, label='训练时间 (秒)')
ax2.set_ylabel('训练时间 (秒)', fontsize=12, color='#e15759', fontproperties=font_prop)
ax2.tick_params(axis='y', labelcolor='#e15759')

# 在折线图上添加数值标签
for i, v in enumerate(time_used):
    ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom', color='#e15759', fontsize=9)

# 设置标题和标签
plt.title('不同推荐模型的内存消耗和训练时间对比', fontsize=14, pad=20, fontproperties=font_prop)

# 调整x轴标签
plt.xticks(rotation=45, ha='right', fontproperties=font_prop)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', prop=font_prop)

# 调整布局
plt.tight_layout()

# 确保输出目录存在
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, 'output')
os.makedirs(output_dir, exist_ok=True)

# 保存图表
output_path = os.path.join(output_dir, 'model_comparison.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

print(f"图表已保存到: {output_path}") 
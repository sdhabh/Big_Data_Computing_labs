import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from algorithm.prediction_algorithms.matrix_factorization import SVD
from algorithm import Dataset, Reader
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.font_manager import FontProperties

# 设置全局样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']  # 更美观的中文字体
plt.rcParams['axes.unicode_minus'] = False

# 自定义颜色方案
COLORS = {
    'primary': '#2c7bb6',
    'secondary': '#d7191c',
    'accent': '#fdae61',
    'background': '#f8f9fa',
    'text': '#343a40'
}

def calculate_loss(model, trainset):
    """计算当前模型在训练集上的损失"""
    total_loss = 0
    n_samples = 0
    
    for u, i, r in trainset.all_ratings():
        pred = model.estimate(u, i)
        err = r - pred
        total_loss += err * err
        n_samples += 1
    
    return total_loss / n_samples

def load_data(data_path):
    """加载数据"""
    records = []
    with open(data_path, 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            line = line.strip()
            if '|' in line:
                user_id, cnt = line.split('|')
                cnt = int(cnt)
                for _ in range(cnt):
                    item_id, score = f.readline().strip().split()
                    records.append((user_id, item_id, float(score)))

    df = pd.DataFrame(records, columns=['user', 'item', 'rating'])
    reader = Reader(rating_scale=(0, 100))
    data = Dataset.load_from_df(df[['user','item','rating']], reader)
    return data.build_full_trainset()

def analyze_n_factors(trainset, n_factors_list, n_epochs=20):
    """分析不同潜在因子数量的影响"""
    results = {
        'n_factors': [],
        'final_loss': [],
        'training_time': []
    }
    
    for n_factors in n_factors_list:
        print(f"\n分析 n_factors = {n_factors}")
        
        # 初始化模型
        model = SVD(n_factors=n_factors, n_epochs=n_epochs)
        
        # 记录训练过程中的损失
        losses = []
        start_time = time.time()
        
        # 训练模型
        for epoch in range(n_epochs):
            model.fit(trainset)
            current_loss = calculate_loss(model, trainset)
            losses.append(current_loss)
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {current_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # 记录结果
        results['n_factors'].append(n_factors)
        results['final_loss'].append(losses[-1])
        results['training_time'].append(training_time)
    
    return results

def plot_analysis_results(results):
    """绘制美观的分析结果图表"""
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})
    fig.patch.set_facecolor(COLORS['background'])
    
    # ========== 左图：双轴折线图 ==========
    # 绘制损失曲线（左y轴）
    line1 = ax1.plot(results['n_factors'], results['final_loss'], 
                     color=COLORS['primary'], marker='o', markersize=10,
                     linewidth=3, label='最终损失', zorder=5)
    
    ax1.set_xlabel('潜在因子数量', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax1.set_ylabel('最终损失 (MSE)', fontsize=14, fontweight='bold', color=COLORS['primary'])
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 设置左轴刻度格式
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # 创建右y轴
    ax1b = ax1.twinx()
    line2 = ax1b.plot(results['n_factors'], results['training_time'], 
                      color=COLORS['secondary'], marker='s', markersize=10,
                      linewidth=3, label='训练时间', zorder=5)
    
    ax1b.set_ylabel('训练时间 (秒)', fontsize=14, fontweight='bold', color=COLORS['secondary'])
    ax1b.tick_params(axis='y', labelcolor=COLORS['secondary'])
    
    # 添加数据标签
    for i, (n, loss) in enumerate(zip(results['n_factors'], results['final_loss'])):
        ax1.annotate(f'{loss:.1f}', 
                    (n, loss), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=10,
                    color=COLORS['primary'])
    
    for i, (n, time_val) in enumerate(zip(results['n_factors'], results['training_time'])):
        ax1b.annotate(f'{time_val:.1f}s', 
                     (n, time_val), 
                     textcoords="offset points", 
                     xytext=(0,-15), 
                     ha='center',
                     fontsize=10,
                     color=COLORS['secondary'])
    
    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), 
              ncol=2, fontsize=12, frameon=True)
    
    # 设置标题
    ax1.set_title('SVD模型：潜在因子数量对性能的影响', 
                 fontsize=16, fontweight='bold', pad=20, color=COLORS['text'])
    
    # ========== 右图：散点图（损失-时间权衡） ==========
    scatter = ax2.scatter(results['training_time'], results['final_loss'],
                         c=results['n_factors'], cmap='viridis',
                         s=350, alpha=0.8, edgecolors='w', linewidth=2, zorder=5)
    
    ax2.set_xlabel('训练时间 (秒)', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax2.set_ylabel('最终损失 (MSE)', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # 添加数据点标签
    for i, (time_val, loss, n) in enumerate(zip(results['training_time'], 
                                               results['final_loss'], 
                                               results['n_factors'])):
        ax2.annotate(f'n={n}', 
                    (time_val, loss), 
                    textcoords="offset points", 
                    xytext=(10,5), 
                    ha='left',
                    fontsize=11,
                    fontweight='bold',
                    color='white')
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('潜在因子数量', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # 设置标题
    ax2.set_title('损失与时间的权衡分析', 
                 fontsize=16, fontweight='bold', pad=20, color=COLORS['text'])
    
    # 添加整体标题
    fig.suptitle('SVD模型参数优化分析', 
                fontsize=20, fontweight='bold', 
                y=0.98, color=COLORS['text'])
    
    # 添加说明文字
    caption = "图1: 评估不同潜在因子数量对SVD模型性能的影响\n左图展示因子数量与损失/训练时间的关系，右图揭示损失与时间的权衡关系"
    fig.text(0.5, 0.01, caption, ha='center', fontsize=12, color=COLORS['text'], alpha=0.8)
    
    # 调整布局
    plt.subplots_adjust(wspace=0.3, bottom=0.15, top=0.9)
    
    # 保存高分辨率图片
    plt.savefig('svd_parameter_analysis.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.savefig('svd_parameter_analysis.pdf', bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.show()

if __name__ == "__main__":
    # 设置数据路径
    data_path = "data/train.txt"
    
    # 加载数据
    print("加载数据...")
    trainset = load_data(data_path)
    
    # 设置要分析的潜在因子数量列表
    n_factors_list = [10, 30, 50, 70, 90, 110, 130, 150, 170, 200]
    
    # 分析不同潜在因子数量的影响
    print("\n开始分析不同潜在因子数量的影响...")
    results = analyze_n_factors(trainset, n_factors_list)
    
    # 绘制分析结果
    print("\n绘制分析结果...")
    plot_analysis_results(results)
    
    print("\n分析完成！结果已保存为 svd_parameter_analysis.png 和 svd_parameter_analysis.pdf")
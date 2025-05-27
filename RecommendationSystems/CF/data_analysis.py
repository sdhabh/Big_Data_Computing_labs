import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class Tee:
    """同时将输出写入到文件和控制台"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def plot_cold_start_analysis(train_df, test_df, output_dir):
    """可视化冷启动分析结果"""
    # 创建图表目录
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 获取用户和物品集合
    train_users = set(train_df['user_id'].unique())
    train_items = set(train_df['item_id'].unique())
    test_users = set(test_df['user_id'].unique())
    test_items = set(test_df['item_id'].unique())
    
    # 计算分布数据
    only_in_train_items = train_items - test_items
    only_in_test_items = test_items - train_items
    in_both_items = train_items & test_items
    
    only_in_train_users = train_users - test_users
    only_in_test_users = test_users - train_users
    in_both_users = train_users & test_users
    
    # 1. 物品分布饼图
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    item_sizes = [len(only_in_train_items), len(only_in_test_items), len(in_both_items)]
    item_labels = ['仅在训练集', '仅在测试集', '共同物品']
    plt.pie(item_sizes, labels=item_labels, autopct='%1.1f%%', colors=sns.color_palette("Set3"))
    plt.title('物品分布情况')
    
    # 2. 用户分布饼图
    plt.subplot(1, 2, 2)
    user_sizes = [len(only_in_train_users), len(only_in_test_users), len(in_both_users)]
    user_labels = ['仅在训练集', '仅在测试集', '共同用户']
    plt.pie(user_sizes, labels=user_labels, autopct='%1.1f%%', colors=sns.color_palette("Set2"))
    plt.title('用户分布情况')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distribution_pie_charts.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 物品分布条形图
    plt.figure(figsize=(10, 6))
    item_data = pd.DataFrame({
        '类别': ['训练集物品', '测试集物品', '共同物品', '仅在训练集', '仅在测试集'],
        '数量': [len(train_items), len(test_items), len(in_both_items), 
                len(only_in_train_items), len(only_in_test_items)]
    })
    
    sns.barplot(x='类别', y='数量', data=item_data, palette='viridis')
    plt.title('物品分布统计')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'item_distribution_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 用户分布条形图
    plt.figure(figsize=(10, 6))
    user_data = pd.DataFrame({
        '类别': ['训练集用户', '测试集用户', '共同用户', '仅在训练集', '仅在测试集'],
        '数量': [len(train_users), len(test_users), len(in_both_users), 
                len(only_in_train_users), len(only_in_test_users)]
    })
    
    sns.barplot(x='类别', y='数量', data=user_data, palette='magma')
    plt.title('用户分布统计')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'user_distribution_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 冷启动比例热力图
    plt.figure(figsize=(8, 6))
    cold_start_data = np.array([
        [len(in_both_items)/len(test_items), len(only_in_test_items)/len(test_items)],
        [len(in_both_users)/len(test_users), len(only_in_test_users)/len(test_users)]
    ])
    
    sns.heatmap(cold_start_data, 
                annot=True, 
                fmt='.2%',
                cmap='YlOrRd',
                xticklabels=['共同比例', '冷启动比例'],
                yticklabels=['物品', '用户'])
    plt.title('冷启动比例热力图')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cold_start_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化结果已保存到: {plots_dir}")

def analyze_train_dataset(file_path, output_file):
    """分析训练集数据（包含用户评分信息）"""
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return None
    
    try:
        # 读取数据集
        print(f"正在读取训练集文件：{file_path}")
        
        # 初始化列表存储数据
        data = []
        current_user = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:  # 用户行
                    current_user, num_ratings = line.split('|')
                    current_user = int(current_user)
                else:  # 评分行
                    if line:  # 确保不是空行
                        item_id, score = line.split()
                        data.append({
                            'user_id': current_user,
                            'item_id': int(item_id),
                            'rating': int(score)
                        })
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 打印数据的前几行，用于调试
        print("\n数据预览（前5行）：")
        print(df.head())
        
        # 检查数据是否为空
        if df.empty:
            print("错误：数据集为空")
            return None
        
        # 计算基本统计信息
        num_users = df['user_id'].nunique()
        num_items = df['item_id'].nunique()
        num_ratings = len(df)
        
        # 计算用户和物品的ID范围
        min_user_id = df['user_id'].min()
        max_user_id = df['user_id'].max()
        min_item_id = df['item_id'].min()
        max_item_id = df['item_id'].max()
        
        # 检查是否有有效数据
        if num_users == 0 or num_items == 0:
            print("错误：未找到有效的用户或物品数据")
            return None
        
        # 计算评分的统计信息
        rating_stats = df['rating'].describe()
        
        # 计算每个用户的平均评分数量
        ratings_per_user = df.groupby('user_id').size()
        avg_ratings_per_user = ratings_per_user.mean()
        
        # 计算每个物品的平均评分数量
        ratings_per_item = df.groupby('item_id').size()
        avg_ratings_per_item = ratings_per_item.mean()
        
        # 计算数据集的稀疏度
        total_possible_ratings = num_users * num_items
        if total_possible_ratings > 0:
            sparsity = 1 - (num_ratings / total_possible_ratings)
        else:
            sparsity = 1.0
        
        print("\n训练集基本信息：")
        print(f"用户数量: {num_users}")
        print(f"物品数量: {num_items}")
        print(f"评分数量: {num_ratings}")
        print(f"\n用户ID范围：")
        print(f"最小用户ID: {min_user_id}")
        print(f"最大用户ID: {max_user_id}")
        print(f"\n物品ID范围：")
        print(f"最小物品ID: {min_item_id}")
        print(f"最大物品ID: {max_item_id}")
        print(f"\n评分统计信息：")
        print(rating_stats)
        print(f"\n每个用户的平均评分数量: {avg_ratings_per_user:.2f}")
        print(f"每个物品的平均评分数量: {avg_ratings_per_item:.2f}")
        print(f"数据集稀疏度: {sparsity:.4f}")
        
        # 额外统计信息
        print("\n评分分布：")
        rating_distribution = df['rating'].value_counts().sort_index()
        print(rating_distribution)
        
        return df
        
    except Exception as e:
        print(f"处理训练集数据时发生错误：{str(e)}")
        return None

def analyze_test_dataset(file_path, train_df=None, output_file=None):
    """分析测试集数据（仅包含用户-物品对，无评分信息）"""
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在")
        return
    
    try:
        # 读取数据集
        print(f"正在读取测试集文件：{file_path}")
        
        # 初始化列表存储数据
        data = []
        current_user = None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '|' in line:  # 用户行
                    current_user, num_items = line.split('|')
                    current_user = int(current_user)
                else:  # 物品行
                    if line:  # 确保不是空行
                        item_id = int(line)
                        data.append({
                            'user_id': current_user,
                            'item_id': item_id
                        })
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 打印数据的前几行，用于调试
        print("\n数据预览（前5行）：")
        print(df.head())
        
        # 检查数据是否为空
        if df.empty:
            print("错误：数据集为空")
            return
        
        # 计算基本统计信息
        num_users = df['user_id'].nunique()
        num_items = df['item_id'].nunique()
        num_predictions = len(df)
        
        # 计算用户和物品的ID范围
        min_user_id = df['user_id'].min()
        max_user_id = df['user_id'].max()
        min_item_id = df['item_id'].min()
        max_item_id = df['item_id'].max()
        
        # 检查是否有有效数据
        if num_users == 0 or num_items == 0:
            print("错误：未找到有效的用户或物品数据")
            return
        
        # 计算每个用户需要预测的物品数量
        items_per_user = df.groupby('user_id').size()
        avg_items_per_user = items_per_user.mean()
        min_items_per_user = items_per_user.min()
        max_items_per_user = items_per_user.max()
        
        # 计算每个物品被预测的次数
        predictions_per_item = df.groupby('item_id').size()
        avg_predictions_per_item = predictions_per_item.mean()
        min_predictions_per_item = predictions_per_item.min()
        max_predictions_per_item = predictions_per_item.max()
        
        print("\n测试集基本信息：")
        print(f"用户数量: {num_users}")
        print(f"物品数量: {num_items}")
        print(f"需要预测的评分数量: {num_predictions}")
        print(f"\n用户ID范围：")
        print(f"最小用户ID: {min_user_id}")
        print(f"最大用户ID: {max_user_id}")
        print(f"\n物品ID范围：")
        print(f"最小物品ID: {min_item_id}")
        print(f"最大物品ID: {max_item_id}")
        
        print(f"\n每个用户的预测物品数量统计：")
        print(f"平均数量: {avg_items_per_user:.2f}")
        print(f"最小数量: {min_items_per_user}")
        print(f"最大数量: {max_items_per_user}")
        
        print(f"\n每个物品被预测的次数统计：")
        print(f"平均次数: {avg_predictions_per_item:.2f}")
        print(f"最小次数: {min_predictions_per_item}")
        print(f"最大次数: {max_predictions_per_item}")
        
        # 冷启动分析
        if train_df is not None:
            # 获取训练集中的用户和物品集合
            train_users = set(train_df['user_id'].unique())
            train_items = set(train_df['item_id'].unique())
            
            # 获取测试集中的用户和物品集合
            test_users = set(df['user_id'].unique())
            test_items = set(df['item_id'].unique())
            
            # 计算新用户和新物品
            new_users = test_users - train_users
            new_items = test_items - train_items
            
            # 计算冷启动比例
            new_user_ratio = len(new_users) / len(test_users)
            new_item_ratio = len(new_items) / len(test_items)
            
            # 计算需要预测的新用户-物品对数量
            new_user_item_pairs = df[df['user_id'].isin(new_users) | df['item_id'].isin(new_items)]
            new_pairs_ratio = len(new_user_item_pairs) / len(df)
            
            # 计算物品分布情况
            only_in_train = train_items - test_items
            only_in_test = test_items - train_items
            in_both = train_items & test_items
            
            print("\n冷启动分析：")
            print(f"新用户数量: {len(new_users)}")
            print(f"新用户比例: {new_user_ratio:.2%}")
            print(f"新物品数量: {len(new_items)}")
            print(f"新物品比例: {new_item_ratio:.2%}")
            print(f"包含新用户或新物品的预测对数量: {len(new_user_item_pairs)}")
            print(f"包含新用户或新物品的预测对比例: {new_pairs_ratio:.2%}")
            
            print("\n物品分布详细分析：")
            print(f"仅在训练集中出现的物品数量: {len(only_in_train)}")
            print(f"仅在测试集中出现的物品数量: {len(only_in_test)}")
            print(f"在训练集和测试集中都出现的物品数量: {len(in_both)}")
            print(f"训练集中物品总数: {len(train_items)}")
            print(f"测试集中物品总数: {len(test_items)}")
            print(f"仅在训练集中出现的物品比例: {len(only_in_train)/len(train_items):.2%}")
            print(f"仅在测试集中出现的物品比例: {len(only_in_test)/len(test_items):.2%}")
            print(f"共同物品比例: {len(in_both)/len(test_items):.2%}")
            
            # 计算用户分布情况
            only_in_train_users = train_users - test_users
            only_in_test_users = test_users - train_users
            in_both_users = train_users & test_users
            
            print("\n用户分布详细分析：")
            print(f"仅在训练集中出现的用户数量: {len(only_in_train_users)}")
            print(f"仅在测试集中出现的用户数量: {len(only_in_test_users)}")
            print(f"在训练集和测试集中都出现的用户数量: {len(in_both_users)}")
            print(f"训练集中用户总数: {len(train_users)}")
            print(f"测试集中用户总数: {len(test_users)}")
            print(f"仅在训练集中出现的用户比例: {len(only_in_train_users)/len(train_users):.2%}")
            print(f"仅在测试集中出现的用户比例: {len(only_in_test_users)/len(test_users):.2%}")
            print(f"共同用户比例: {len(in_both_users)/len(test_users):.2%}")
            
            # 在冷启动分析完成后添加可视化
            plot_cold_start_analysis(train_df, df, os.path.dirname(output_file))
        
    except Exception as e:
        print(f"处理测试集数据时发生错误：{str(e)}")

if __name__ == "__main__":
    # 使用绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用固定的输出文件名
    output_file = os.path.join(output_dir, "data_analysis.txt")
    
    # 重定向输出到文件
    original_stdout = sys.stdout
    tee = Tee(output_file)
    sys.stdout = tee
    
    try:
        print(f"数据分析开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*50)
        
        # 分析训练集
        train_file = os.path.join(data_dir, "train.txt")
        print("\n" + "="*50)
        train_df = analyze_train_dataset(train_file, output_file)
        
        # 分析测试集
        test_file = os.path.join(data_dir, "test.txt")
        print("\n" + "="*50)
        analyze_test_dataset(test_file, train_df, output_file)
        
        print("\n" + "="*50)
        print(f"数据分析结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"分析结果已保存到: {output_file}")
        
    finally:
        # 恢复标准输出
        sys.stdout = original_stdout
        tee.close() 
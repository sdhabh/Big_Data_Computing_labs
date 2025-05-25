import os
import time
import psutil
from RecommendationSystems.CF.algorithm.UserCF import UserCF
from RecommendationSystems.CF.algorithm.ItemCF import ItemCF

def get_display_width(text):
    """计算字符串的显示宽度，中文字符计为2个宽度"""
    width = 0
    for char in text:
        if ord(char) > 127:  # 中文字符
            width += 2
        else:  # 英文字符
            width += 1
    return width

def print_separator(char='=', length=50):
    """打印分隔线"""
    print(char * length)

def print_section_title(title):
    """打印带格式的标题，支持中文字符居中"""
    width = 50
    title_width = get_display_width(title)
    left_padding = (width - title_width) // 2
    right_padding = width - title_width - left_padding
    
    print_separator()
    print(' ' * left_padding + title + ' ' * right_padding)
    print_separator()

def run_model(model, model_name, train_file, val_file, test_file, output_dir):
    print_section_title(f"{model_name} 模型训练与评估")
    
    # 训练模型
    print(f"[1/3] 开始训练{model_name}模型...")
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss if process else 0
    model.fit(train_file)
    time_used = time.time() - start_time
    end_memory = process.memory_info().rss if process else 0
    memory_used = (end_memory - start_memory) / (1024 ** 2)  # 转换为MB
    print(f"✓ {model_name}模型训练完成！")
    print(f"├─ 耗时: {time_used:.2f} 秒")
    print(f"└─ 内存消耗: {memory_used:.2f} MB")
    
    # 在验证集上评估模型
    print(f"\n[2/3] 在验证集上评估{model_name}模型性能...")
    eval_start_time = time.time()
    eval_start_memory = process.memory_info().rss if process else 0
    mae, rmse = model.evaluate(val_file)
    eval_time = time.time() - eval_start_time
    eval_end_memory = process.memory_info().rss if process else 0
    eval_memory = (eval_end_memory - eval_start_memory) / (1024 ** 2)
    print("验证集评估结果：")
    print(f"├─ MAE (平均绝对误差): {mae:.2f}")
    print(f"├─ RMSE (均方根误差): {rmse:.2f}")
    print(f"├─ 评估耗时: {eval_time:.2f} 秒")
    print(f"└─ 评估内存消耗: {eval_memory:.2f} MB")
    
    # 对测试集中的每个用户-物品对进行预测
    print(f"\n[3/3] 开始使用{model_name}预测测试集中的评分...")
    predict_start_time = time.time()
    predict_start_memory = process.memory_info().rss if process else 0
    current_user = None
    predictions = []
    
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                if '|' in line:  # 用户行
                    current_user, _ = line.split('|')
                    current_user = int(current_user)
                else:  # 评分行
                    parts = line.split()
                    if len(parts) >= 1:  # 确保至少有一个值
                        item_id = int(parts[0])
                        predicted_rating = model.predict(current_user, item_id)
                        predictions.append((current_user, item_id, predicted_rating))
    
        # 将预测结果保存到文件
        output_file = os.path.join(output_dir, f"{model_name.lower()}_output.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for user_id, item_id, pred_rating in predictions:
                f.write(f"{user_id}\t{item_id}\t{pred_rating:.2f}\n")
        
        predict_time = time.time() - predict_start_time
        predict_end_memory = process.memory_info().rss if process else 0
        predict_memory = (predict_end_memory - predict_start_memory) / (1024 ** 2)
        total_memory = (predict_end_memory - start_memory) / (1024 ** 2)
        
        print(f"✓ {model_name}预测完成！")
        print(f"├─ 预测耗时: {predict_time:.2f} 秒")
        print(f"├─ 预测内存消耗: {predict_memory:.2f} MB")
        print(f"├─ 总内存消耗: {total_memory:.2f} MB")
        print(f"└─ 预测结果已保存到：{output_file}\n")
        
    except Exception as e:
        print(f"\n❌ 处理文件时发生错误：{str(e)}")
        print(f"├─ 当前处理的行：{line}")
        print(f"└─ 当前用户：{current_user}")

def main():
    print_section_title("协同过滤推荐系统")
    n_neighbors = 20
    min_similarity = 0
    similarity_method = 'cosine'  # pearson cosine
    print("系统配置：")
    print(f"├─ 邻居数量 (n_neighbors): {n_neighbors}")
    print(f"├─ 最小相似度阈值 (min_similarity): {min_similarity}")
    print(f"└─ 相似度计算方法 (similarity_method): {similarity_method}\n")
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建训练集、验证集和测试集的文件路径
    train_file = os.path.join(current_dir, "data", "train_split.txt")
    val_file = os.path.join(current_dir, "data", "val_split.txt")
    test_file = os.path.join(current_dir, "data", "test.txt")
    
    # 创建输出目录
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行UserCF模型 
    user_cf_model = UserCF(n_neighbors, min_similarity, similarity_method)
    run_model(user_cf_model, "UserCF", train_file, val_file, test_file, output_dir)
    
    # 运行ItemCF模型 
    item_cf_model = ItemCF(n_neighbors, min_similarity, similarity_method)
    run_model(item_cf_model, "ItemCF", train_file, val_file, test_file, output_dir)
    
    print_section_title("运行完成")
    print(" 所有模型运行完成！预测结果已保存在output目录下。")

if __name__ == "__main__":
    main() 
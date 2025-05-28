import os
import time
import random
import argparse
from collections import defaultdict
import psutil
from algorithm.UserCF import UserCF
from algorithm.ItemCF import ItemCF
from algorithm.GraphCF import GraphCF
from algorithm.SlopeOne import SlopeOne
from algorithm.LeastSquaresCF import LeastSquaresCF
from algorithm.TopKNanCF import TopKNanCF
from algorithm.GDLinearCF import GDLinearCF
import sys


def get_display_width(text):
    width = 0
    for char in text:
        if ord(char) > 127:
            width += 2
        else:
            width += 1
    return width


def print_separator(char="=", length=50):
    print(char * length)


def print_section_title(title):
    width = 50
    title_width = get_display_width(title)
    left_padding = (width - title_width) // 2
    right_padding = width - title_width - left_padding
    print_separator()
    print(" " * left_padding + title + " " * right_padding)
    print_separator()


def read_data_to_dict(file_path):
    """读取数据文件并转换为字典格式"""
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


def train_test_split_per_user(user_item_score, test_size=0.2, seed=42):
    train = defaultdict(dict)
    test = defaultdict(dict)
    random.seed(seed)
    for user, items in user_item_score.items():
        item_list = list(items.items())
        if len(item_list) < 2:
            train[user] = dict(item_list)
            continue
        random.shuffle(item_list)
        split = int(len(item_list) * (1 - test_size))
        train_items = item_list[:split]
        test_items = item_list[split:]
        train[user] = dict(train_items)
        test[user] = dict(test_items)
    return train, test


def run_itemcf_predict(model, test_file, output_file):
    current_user = None
    predictions = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "|" in line:
                current_user, _ = line.split("|")
                current_user = int(current_user)
            else:
                parts = line.split()
                if len(parts) >= 1:
                    item_id = int(parts[0])
                    predicted_rating = model.predict(current_user, item_id)
                    predictions.append((current_user, item_id, predicted_rating))
    with open(output_file, "w", encoding="utf-8") as f:
        for user_id, item_id, pred_rating in predictions:
            f.write(f"{user_id}\t{item_id}\t{pred_rating:.2f}\n")


def cross_validate_and_predict(
    model_class,
    model_name,
    user_item_score,
    test_file,
    output_dir,
    n_neighbors=20,
    min_similarity=0,
    similarity_method="cosine",
    n_splits=5,
    test_size=0.2,
):
    print_section_title(f"{model_name} {n_splits}次随机划分交叉验证")
    mae_list = []
    rmse_list = []
    for i in range(n_splits):
        print(f"\n第 {i+1} 次划分与评估：")
        train_dict, val_dict = train_test_split_per_user(
            user_item_score, test_size=test_size, seed=int(time.time()) + i
        )
        if model_name.lower() == "itemcf":
            model = model_class(n_neighbors, min_similarity, similarity_method)
        elif model_name.lower() == "usercf":
            model = model_class(n_neighbors, min_similarity, similarity_method)
        elif model_name.lower() == "graphcf":
            model = model_class()
        elif model_name.lower() == "slopeone":
            model = model_class()
        else:
            model = model_class()
        model.fit_from_dict(train_dict)
        mae, rmse = model.evaluate_from_dict(val_dict)
        print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        mae_list.append(mae)
        rmse_list.append(rmse)
    print(f"\n{model_name}交叉验证结果：")
    print(f"MAE均值: {sum(mae_list)/len(mae_list):.4f}")
    print(f"RMSE均值: {sum(rmse_list)/len(rmse_list):.4f}")

    # 用全部数据训练并预测
    print_section_title(f"{model_name} 用全部训练数据训练并预测测试集")
    if model_name.lower() == "itemcf":
        model = model_class(n_neighbors, min_similarity, similarity_method)
    elif model_name.lower() == "usercf":
        model = model_class(n_neighbors, min_similarity, similarity_method)
    elif model_name.lower() == "graphcf":
        model = model_class()
    elif model_name.lower() == "slopeone":
        model = model_class()
    else:
        model = model_class()
    model.fit_from_dict(user_item_score)
    output_file = os.path.join(output_dir, f"{model_name.lower()}_output.txt")
    run_itemcf_predict(model, test_file, output_file)
    print(f"测试集预测结果已保存到：{output_file}")


def run_model(model, model_name, train_file, val_file, test_file, output_dir):
    print_section_title(f"{model_name} 模型训练与评估")
    print(f"[1/3] 开始训练{model_name}模型...")
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss if process else 0
    
    # 读取训练数据
    train_dict = read_data_to_dict(train_file)
    model.fit_from_dict(train_dict)
    
    time_used = time.time() - start_time
    end_memory = process.memory_info().rss if process else 0
    memory_used = (end_memory - start_memory) / (1024**2)
    print(f"✓ {model_name}模型训练完成！")
    print(f"├─ 耗时: {time_used:.2f} 秒")
    print(f"└─ 内存消耗: {memory_used:.2f} MB")

    print(f"\n[2/3] 在验证集上评估{model_name}模型性能...")
    eval_start_time = time.time()
    eval_start_memory = process.memory_info().rss if process else 0
    
    # 读取验证数据
    val_dict = read_data_to_dict(val_file)
    mae, rmse = model.evaluate_from_dict(val_dict)
    
    eval_time = time.time() - eval_start_time
    eval_end_memory = process.memory_info().rss if process else 0
    eval_memory = (eval_end_memory - eval_start_memory) / (1024**2)
    print("验证集评估结果：")
    print(f"├─ MAE (平均绝对误差): {mae:.2f}")
    print(f"├─ RMSE (均方根误差): {rmse:.2f}")
    print(f"├─ 评估耗时: {eval_time:.2f} 秒")
    print(f"└─ 评估内存消耗: {eval_memory:.2f} MB")

    print(f"\n[3/3] 开始使用{model_name}预测测试集中的评分...")
    predict_start_time = time.time()
    predict_start_memory = process.memory_info().rss if process else 0
    predictions = []

    try:
        with open(test_file, "r", encoding="utf-8") as f:
            current_user = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "|" in line:
                    current_user, _ = line.split("|")
                    current_user = int(current_user)
                else:
                    parts = line.split()
                    if len(parts) >= 1:
                        item_id = int(parts[0])
                        predicted_rating = model.predict(current_user, item_id)
                        predictions.append((current_user, item_id, predicted_rating))

        output_file = os.path.join(output_dir, f"{model_name.lower()}_output.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for user_id, item_id, pred_rating in predictions:
                f.write(f"{user_id}\t{item_id}\t{pred_rating:.2f}\n")

        predict_time = time.time() - predict_start_time
        predict_end_memory = process.memory_info().rss if process else 0
        predict_memory = (predict_end_memory - predict_start_memory) / (1024**2)
        total_memory = (predict_end_memory - start_memory) / (1024**2)

        print(f"✓ {model_name}预测完成！")
        print(f"├─ 预测耗时: {predict_time:.2f} 秒")
        print(f"├─ 预测内存消耗: {predict_memory:.2f} MB")
        print(f"├─ 总内存消耗: {total_memory:.2f} MB")
        print(f"└─ 预测结果已保存到：{output_file}\n")

    except Exception as e:
        print(f"\n❌ 处理文件时发生错误：{str(e)}")
        print(f"├─ 当前处理的行：{line}")
        print(f"└─ 当前用户：{current_user}")


def run_basic_models():
    """运行基础模型（不使用交叉验证）"""
    print_section_title("协同过滤推荐系统 - 基础模式")
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

    # 运行GraphCF模型
    # graph_cf_model = GraphCF()
    # run_model(graph_cf_model, "GraphCF", train_file, val_file, test_file, output_dir)
    
    # 运行SlopeOne模型
    slope_one_model = SlopeOne()
    run_model(slope_one_model, "SlopeOne", train_file, val_file, test_file, output_dir)
    
    # 运行LeastSquaresCF模型
    least_squares_model = LeastSquaresCF()
    run_model(least_squares_model, "LeastSquaresCF", train_file, val_file, test_file, output_dir)
    
    # 运行TopKNanCF模型
    # topk_nan_model = TopKNanCF()
    # run_model(topk_nan_model, "TopKNanCF", train_file, val_file, test_file, output_dir)
    
    # 运行GDLinearCF模型
    gd_linear_model = GDLinearCF()
    run_model(gd_linear_model, "GDLinearCF", train_file, val_file, test_file, output_dir)
    
    print_section_title("基础模式运行完成")
    print(" 所有模型运行完成！预测结果已保存在output目录下。")


def run_cross_validation_models():
    """运行交叉验证模式"""
    print_section_title("协同过滤推荐系统 - 交叉验证模式")
    n_neighbors = 20
    min_similarity = 0
    similarity_method = "cosine"
    print("系统配置：")
    print(f"├─ 邻居数量 (n_neighbors): {n_neighbors}")
    print(f"├─ 最小相似度阈值 (min_similarity): {min_similarity}")
    print(f"└─ 相似度计算方法 (similarity_method): {similarity_method}\n")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    train_txt_path = os.path.join(data_dir, "train.txt")
    test_file = os.path.join(data_dir, "test.txt")

    # 读取完整训练数据
    user_item_score = read_data_to_dict(train_txt_path)

    # 对所有模型进行交叉验证和最终预测
    cross_validate_and_predict(
        ItemCF,
        "ItemCF",
        user_item_score,
        test_file,
        output_dir,
        n_neighbors,
        min_similarity,
        similarity_method,
    )
    cross_validate_and_predict(
        UserCF,
        "UserCF",
        user_item_score,
        test_file,
        output_dir,
        n_neighbors,
        min_similarity,
        similarity_method,
    )
    cross_validate_and_predict(
        GraphCF, "GraphCF", user_item_score, test_file, output_dir
    )
    cross_validate_and_predict(
        SlopeOne, "SlopeOne", user_item_score, test_file, output_dir
    )
    cross_validate_and_predict(
        LeastSquaresCF, "LeastSquaresCF", user_item_score, test_file, output_dir
    )
    cross_validate_and_predict(
        TopKNanCF,
        "TopKNanCF",
        user_item_score,
        test_file,
        output_dir,
    )
    cross_validate_and_predict(
        GDLinearCF,
        "GDLinearCF",
        user_item_score,
        output_dir,
    )
    print_section_title("交叉验证模式运行完成")
    print(" 所有模型运行完成！预测结果已保存在output目录下。")


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


def main():
    parser = argparse.ArgumentParser(description='协同过滤推荐系统')
    parser.add_argument('--mode', type=str, choices=['basic', 'cross'], default='basic',
                      help='运行模式: basic (基础模式) 或 cross (交叉验证模式)')
    args = parser.parse_args()

    # 日志重定向到不同的log文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    if args.mode == 'basic':
        log_file = os.path.join(output_dir, "basic1_log.txt")
    else:
        log_file = os.path.join(output_dir, "cross_log.txt")
    original_stdout = sys.stdout
    tee = Tee(log_file)
    sys.stdout = tee
    try:
        if args.mode == 'basic':
            run_basic_models()
        else:
            run_cross_validation_models()
    finally:
        sys.stdout = original_stdout
        tee.close()


if __name__ == "__main__":
    main()

import os
import time
from collections import defaultdict
from algorithm.LeastSquaresCF import LeastSquaresCF
from algorithm.GDLinearCF import GDLinearCF
from algorithm.TopKNanCF import TopKNanCF


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


def train_test_split_per_user(user_item_score, test_size=0.2, seed=42):
    import random

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


def evaluate_model(model_class, param_dict, train_dict, test_dict):
    model = model_class(**param_dict)
    model.fit_from_dict(train_dict)
    mae, rmse = model.evaluate_from_dict(test_dict)
    return mae, rmse


def main():
    # 数据路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "data")
    train_file = os.path.join(data_dir, "train.txt")

    # 读取数据并划分
    user_item_score = read_data_to_dict(train_file)
    train_dict, test_dict = train_test_split_per_user(
        user_item_score, test_size=0.2, seed=42
    )

    print("GDLinearCF 参数分析：")
    for topk in [2, 5, 10, 15, 30, 50]:
        for lr in [0.001, 0.00075, 0.0005, 0.00025, 0.0001]:
            param = {"topk": topk, "lr": lr, "epochs": 150}
            mae, rmse = evaluate_model(GDLinearCF, param, train_dict, test_dict)
            print(f"topk={topk}, lr={lr}, epochs=150 -> MAE={mae:.4f}, RMSE={rmse:.4f}")
    print()


if __name__ == "__main__":
    main()

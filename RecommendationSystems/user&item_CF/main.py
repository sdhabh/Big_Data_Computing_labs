import os
from user_cf import UserCF

def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 构建训练集和测试集的文件路径
    train_file = os.path.join(os.path.dirname(current_dir), "user&item_CF", "data", "train.txt")
    test_file = os.path.join(os.path.dirname(current_dir), "user&item_CF", "data", "test.txt")
    
    # 创建UserCF模型实例
    model = UserCF(n_neighbors=20, min_similarity=0)
    
    print("开始训练模型...")
    # 训练模型并获取验证集
    validation_data = model.fit(train_file, validation_ratio=0.2)
    print("模型训练完成！")
    
    # 在验证集上评估模型
    print("\n在验证集上评估模型性能...")
    mae, rmse = model.evaluate(validation_data)
    print(f"验证集评估结果：")
    print(f"MAE (平均绝对误差): {mae:.2f}")
    print(f"RMSE (均方根误差): {rmse:.2f}")
    
    # 对测试集中的每个用户-物品对进行预测
    print("\n开始预测测试集中的评分...")
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
        output_file = os.path.join(current_dir, "output.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            for user_id, item_id, pred_rating in predictions:
                f.write(f"{user_id}\t{item_id}\t{pred_rating:.2f}\n")
        
        print(f"\n预测完成！预测结果已保存到：{output_file}")
        
    except Exception as e:
        print(f"处理文件时发生错误：{str(e)}")
        print(f"当前处理的行：{line}")
        print(f"当前用户：{current_user}")

if __name__ == "__main__":
    main() 
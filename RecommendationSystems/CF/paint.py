import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 参数范围
topk_list = [2, 5, 10, 15, 30, 50]
lr_list = [0.001, 0.00075, 0.0005, 0.00025, 0.0001]

# MAE实验结果，行对应lr，列对应topk
mae_matrix = np.array(
    [
        [16.2879, 16.2408, 16.2753, 16.2533, 16.3292, 16.4570],
        [16.2430, 16.4894, 16.3701, 16.3440, 16.2770, 16.2310],
        [16.3021, 16.3797, 16.4400, 16.4352, 16.3376, 16.6207],
        [16.2245, 16.3355, 16.4938, 16.5543, 16.3369, 16.5600],
        [16.4233, 16.5779, 16.4635, 16.4276, 16.6228, 16.5885],
    ]
)

plt.figure(figsize=(8, 6))
sns.heatmap(
    mae_matrix,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    xticklabels=topk_list,
    yticklabels=lr_list,
    cbar_kws={"label": "MAE"},
)
plt.xlabel("topk")
plt.ylabel("lr")
plt.title("GDLinearCF")
plt.tight_layout()
plt.show()

import networkx as nx
from collections import defaultdict

# 参数设置
DAMPING = 0.85
EPS = 1e-8
MAX_ITER = 100
TOP_K = 100

# 读取边列表并去重
edges = set()
nodes = set()
with open("Data.txt", "r") as fin:
    for line in fin:
        u, v = map(int, line.strip().split())
        if (u, v) not in edges:
            edges.add((u, v))
            nodes.update([u, v])

# 创建有向图
G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# 处理死胡同节点（networkx的pagerank已自动处理）
# 直接调用pagerank算法
pageranks = nx.pagerank(
    G, 
    alpha=DAMPING,  # 阻尼因子对应(1 - DAMPING)的随机跳转
    tol=EPS,        # 收敛阈值
    max_iter=MAX_ITER
)

# 归一化处理（networkx结果已归一化，此处可省略）
# 按PageRank值排序
sorted_pr = sorted(pageranks.items(), key=lambda x: -x[1])

# 输出结果
print("nodeID\tPageRank")
for node, rank in sorted_pr[:TOP_K]:
    print(f"{node}\t{rank:.18f}")
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

# 调用pagerank算法
pageranks = nx.pagerank(
    G, 
    alpha=DAMPING,
    tol=EPS,
    max_iter=MAX_ITER
)

# 排序结果
sorted_pr = sorted(pageranks.items(), key=lambda x: -x[1])

# 输出到文件
with open("Res.txt", "w") as f:
    # 写入表头
    f.write("nodeID\tPageRank\n")
    
    # 写入前TOP_K个结果
    for node, rank in sorted_pr[:TOP_K]:
        f.write(f"{node}\t{rank:.18f}\n")
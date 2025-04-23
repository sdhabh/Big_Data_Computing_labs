#ifndef PAGERANK_H
#define PAGERANK_H

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

class PageRank {
public:
    // 构造函数
    PageRank(double damping = 0.85, double epsilon = 1e-8, 
            int max_iter = 100, int top_k = 100);

    // 读取边列表并构建图结构
    void readEdges(const std::string& filename);
    
    // 处理死胡同节点
    void handleDeadEnds();
    
    // 执行PageRank计算
    void calculate();
    
    // 输出结果
    void printTopK() const;

private:
    // 内部数据结构
    struct hash_pair {
        template <class T1, class T2>
        size_t operator()(const std::pair<T1, T2>& p) const {
            return std::hash<T1>{}(p.first) ^ std::hash<T2>{}(p.second);
        }
    };

    // 参数配置
    double damping_;
    double epsilon_;
    int max_iter_;
    int top_k_;

    // 图结构存储
    std::unordered_map<int, std::vector<int>> in_links_;
    std::unordered_map<int, int> out_degree_;
    std::unordered_set<int> nodes_;
    std::unordered_set<std::pair<int, int>, hash_pair> edges_;

    // PageRank值存储
    std::unordered_map<int, double> pr_;

    // 初始化PageRank值
    void initializeRank();
    
    // 归一化处理
    void normalize();
};

#endif // PAGERANK_H
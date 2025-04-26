// pagerank.h
#ifndef PAGERANK_H
#define PAGERANK_H

//#define BASIC_VERSION
//#define ENABLE_EDGE_DEDUP
//#define OPENMP_ENABLED

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <iostream>
#ifdef OPENMP_ENABLED
#include <omp.h>
#endif

class PageRank_basic {
    public:
        // 构造函数
        PageRank_basic(double damping = 0.85, double epsilon = 1e-8, 
                int max_iter = 100, int top_k = 100);
        void readEdges(const std::string& filename);
        // 处理死胡同节点
        void handleDeadEnds();
        void calculate();
        void printTopK() const;
    
    private:
        // 内部数据结构
        #ifdef ENABLE_EDGE_DEDUP
        struct hash_pair {
            template <class T1, class T2>
            size_t operator()(const std::pair<T1, T2>& p) const {
                return std::hash<T1>{}(p.first) ^ std::hash<T2>{}(p.second);
            }
        };
        #endif
    
        // 参数配置
        double DAMPING;
        double EPS;
        int MAX_ITER;
        int TOP_K;
    
        // 图结构存储
        std::unordered_map<int, std::vector<int>> in_links_;
        std::unordered_map<int, int> out_degree_;
        std::unordered_set<int> nodes_;
        #ifdef ENABLE_EDGE_DEDUP
        std::unordered_set<std::pair<int, int>, hash_pair> edges_;
        #endif
        // PageRank值存储
        std::unordered_map<int, double> pr_;
    
        // 初始化PageRank值
        void initializeRank();
        
        // 归一化处理
        void normalize();
    };

class PageRank_block {
public:
    PageRank_block(double damping = 0.85, double epsilon = 1e-8, 
            int max_iter = 100, int top_k = 100,
            int num_blocks = 100, int block_size = 100000);
    
    void calculate(const std::string& input_file);
    void printTopK() const;

private:
    struct BlockData {
        std::vector<int> src_list;
        std::vector<int> dst_list;
    };

    // 配置参数
    const double DAMPING;
    const double EPS;
    const int MAX_ITER;
    const int TOP_K;
    const int NUM_BLOCKS;
    const int BLOCK_SIZE;

    // 图数据
    std::unordered_map<int, int> id_map;
    std::vector<int> original_ids;
    std::vector<int> out_degree;
    int node_count;

    // PageRank值
    std::vector<double> pr;

    // 预处理方法
    void preprocessData(const std::string& input_file);
    BlockData loadBlock(int block_id) const;
    void buildGraphStructure();
    void initializePRValues();
    
    // 核心计算
    double computeIteration();
    void updatePRValues(double S_dead, double base);
};

#endif // PAGERANK_H
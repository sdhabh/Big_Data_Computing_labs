// pagerank.cpp
#include "pagerank.h"

#include <cstdlib>
#include <sys/stat.h>

PageRank_basic::PageRank_basic(double damping, double epsilon, int max_iter, int top_k)
    : DAMPING(damping), EPS(epsilon), 
      MAX_ITER(max_iter), TOP_K(top_k) {}

// 修改后的readEdges实现
void PageRank_basic::readEdges(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) throw std::runtime_error("Failed to open: " + filename);

    int u, v;
    while (fin >> u >> v) {
#ifdef ENABLE_EDGE_DEDUP
        auto edge = std::make_pair(u, v);
        if (edges_.find(edge) != edges_.end()) continue;
        edges_.insert(edge);
#endif
        // 关键数据结构更新保持不变
        in_links_[v].push_back(u);
        out_degree_[u]++;
        nodes_.insert(u);
        nodes_.insert(v);
    }
    fin.close();
}

void PageRank_basic::handleDeadEnds() {
    const int N = nodes_.size();
    for (const auto& node : nodes_) {
        if (out_degree_[node] == 0) {
            out_degree_[node] = N;
            for (const auto& neighbor : nodes_) {
                in_links_[neighbor].push_back(node);
            }
        }
    }
}

void PageRank_basic::initializeRank() {
    const double init_rank = 1.0 / nodes_.size();
    for (const auto& node : nodes_) {
        pr_[node] = init_rank;
    }
}

void PageRank_basic::normalize() {
    double sum = 0.0;
    for (const auto& kv : pr_) {
        sum += kv.second;
    }
    for (auto& kv : pr_) {
        kv.second /= sum;
    }
}

void PageRank_basic::calculate() {
    initializeRank();
    std::unordered_map<int, double> pr_new;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        double diff = 0.0;
        pr_new.clear();

        // 计算新的PageRank值
        for (const auto& node : nodes_) {
            double sum_in = 0.0;
            for (const auto& src : in_links_.at(node)) {
                sum_in += pr_[src] / out_degree_.at(src);
            }
            pr_new[node] = (1.0 - DAMPING) / nodes_.size() + DAMPING * sum_in;
        }

        // 计算差值并更新
        for (const auto& node : nodes_) {
            diff += std::fabs(pr_new[node] - pr_[node]);
            pr_[node] = pr_new[node];
        }

        // 收敛检查
        if (diff < EPS) {
            // std::cerr << "Converged at iteration: " << iter + 1 
            //          << " (diff=" << diff << ")\n";
            break;
        }
    }
    normalize();
}

void PageRank_basic::printTopK() const {
    std::vector<std::pair<int, double>> results;
    results.reserve(pr_.size());
    
    for (const auto& kv : pr_) {
        results.emplace_back(kv.first, kv.second);
    }

    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // 将结果输出到 result.txt 文件
    std::ofstream outFile("Res.txt"); // 创建文件输出流
    if (!outFile) {
        // std::cerr << "Failed to open result.txt for writing." << std::endl;
        return;
    }

    outFile << "/******pagerank_basic_result******/ \n";

    outFile << "nodeID\tPageRank\n"; // 写入表头
    const int output_size = std::min(TOP_K, static_cast<int>(results.size()));
    for (int i = 0; i < output_size; ++i) {
        outFile << results[i].first << "\t" // 将结果写入文件
                << std::fixed << std::setprecision(18)
                << results[i].second << "\n";
    }
    outFile.close(); // 关闭文件流
}



PageRank_block::PageRank_block(double damping, double epsilon, int max_iter, 
                 int top_k, int num_blocks, int block_size)
    : DAMPING(damping), EPS(epsilon), MAX_ITER(max_iter),
      TOP_K(top_k), NUM_BLOCKS(num_blocks), BLOCK_SIZE(block_size),
      node_count(0) {}

void PageRank_block::preprocessData(const std::string& input_file) {
    struct stat info;
    if(stat("blocks", &info) != 0) {
        system("mkdir -p blocks");
        std::vector<std::ofstream> block_files(NUM_BLOCKS);
        for(int i = 0; i < NUM_BLOCKS; ++i) {
            std::string filename = "blocks/block_" + std::to_string(i) + ".dat";
            block_files[i].open(filename, std::ios::binary);
        }
        std::ifstream fin(input_file);
        int u, v;
        while(fin >> u >> v) {
            int block_id = (v % NUM_BLOCKS + NUM_BLOCKS) % NUM_BLOCKS;
            block_files[block_id].write(reinterpret_cast<char*>(&u), sizeof(int));
            block_files[block_id].write(reinterpret_cast<char*>(&v), sizeof(int));
        }
    }
}

PageRank_block::BlockData PageRank_block::loadBlock(int block_id) const {
    BlockData block;
    std::string filename = "blocks/block_" + std::to_string(block_id) + ".dat";
    std::ifstream fin(filename, std::ios::binary);
    int u, v;
    while(fin.read(reinterpret_cast<char*>(&u), sizeof(int))) {
        fin.read(reinterpret_cast<char*>(&v), sizeof(int));
        block.src_list.push_back(u);
        block.dst_list.push_back(v);
    }
    return block;
}

void PageRank_block::buildGraphStructure() {
    // 建立ID映射
    for(int blk = 0; blk < NUM_BLOCKS; ++blk) {
        BlockData block = loadBlock(blk);
        for(size_t i = 0; i < block.src_list.size(); ++i) {
            int src = block.src_list[i], dst = block.dst_list[i];
            if(!id_map.count(src)) id_map[src] = node_count++;
            if(!id_map.count(dst)) id_map[dst] = node_count++;
        }
    }
    
    // 初始化数据结构
    original_ids.resize(node_count);
    out_degree.assign(node_count, 0);
    for(auto &kv : id_map) original_ids[kv.second] = kv.first;

    // 统计出度
    for(int blk = 0; blk < NUM_BLOCKS; ++blk) {
        BlockData block = loadBlock(blk);
        for(int src : block.src_list) {
            int mapped_src = id_map[src];
            ++out_degree[mapped_src];
        }
    }
}

void PageRank_block::initializePRValues() {
    const int N = node_count;
    pr.assign(N, 1.0 / N);
}

double PageRank_block::computeIteration() {
    const int N = node_count;
    std::vector<double> pr_new(N);
    double S_dead = 0.0;

    // 死节点贡献计算
    #ifdef OPENMP_ENABLED
    #pragma omp parallel for reduction(+:S_dead)
    #endif
    for(int i = 0; i < N; ++i) {
        if(out_degree[i] == 0) S_dead += pr[i];
    }

    const double base = (1.0 - DAMPING) / N + DAMPING * S_dead / N;
    
    // 基础值初始化
    #ifdef OPENMP_ENABLED
    #pragma omp parallel for
    #endif
    for(int i = 0; i < N; ++i) pr_new[i] = base;

    // 分块矩阵计算
    #ifdef OPENMP_ENABLED
    #pragma omp parallel for schedule(dynamic)
    #endif
    for(int blk = 0; blk < NUM_BLOCKS; ++blk) {
        BlockData block = loadBlock(blk);
        for(size_t i = 0; i < block.dst_list.size(); ++i) {
            int src = block.src_list[i];
            int dst = block.dst_list[i];
            const int mapped_src = id_map[src];
            const int mapped_dst = id_map[dst];
            
            if(out_degree[mapped_src] > 0) {
                const double contrib = DAMPING * pr[mapped_src] / out_degree[mapped_src];
                #ifdef OPENMP_ENABLED
                #pragma omp atomic
                #endif
                pr_new[mapped_dst] += contrib;
            }
        }
    }

    // 收敛性检查
    double diff = 0.0;
    #ifdef OPENMP_ENABLED
    #pragma omp parallel for reduction(+:diff)
    #endif
    for(int i = 0; i < N; ++i) diff += fabs(pr_new[i] - pr[i]);
    
    pr.swap(pr_new);
    return diff;
}

void PageRank_block::calculate(const std::string& input_file) {
    preprocessData(input_file);
    buildGraphStructure();
    initializePRValues();

    //double start_time = omp_get_wtime();
    for(int iter = 0; iter < MAX_ITER; ++iter) {
        double diff = computeIteration();
        if(diff < EPS) {
            // std::cerr << "Converged at iteration " << iter + 1 << '\n';
            break;
        }
    }
    //std::cerr << "Elapsed " << omp_get_wtime() - start_time << " s\n";
}

void PageRank_block::printTopK() const {
    std::ofstream output_file("Res.txt");
    if (!output_file) {
        throw std::runtime_error("Failed to open Res.txt for writing");
    }

    // 写入标识头
    output_file << "/******pagerank_block_result******/ \n";
    
    std::vector<std::pair<double, int>> results;
    for(int i = 0; i < node_count; ++i)
        results.emplace_back(pr[i], original_ids[i]);
    
    std::partial_sort(results.begin(), results.begin() + std::min(TOP_K, node_count),
                     results.end(), [](auto &a, auto &b) { return a.first > b.first; });

    // 输出表头和数据
    output_file << "nodeID\tPageRank\n";
    for(int i = 0; i < std::min(TOP_K, node_count); ++i) {
        output_file << results[i].second << '\t' 
                   << std::fixed << std::setprecision(18) << results[i].first<< '\n';
    }

    // 添加结果统计
 
}


int main() {

    const double DAMPING = 0.85;
    const double EPS = 1e-8;
    const int MAX_ITER = 100;
    const int TOP_K = 100;
    const int NUM_BLOCKS = 100;
    const int BLOCK_SIZE = 100000;


    try {
#ifdef BASIC_VERSION
        // 基本算法版本
        PageRank_basic pr(DAMPING, EPS, MAX_ITER, TOP_K);
        pr.readEdges("Data.txt");
        // pr.handleDeadEnds(); // 可选死节点处理
        pr.calculate();
        pr.printTopK();
#else
        // 分块矩阵优化版本
        PageRank_block pr(DAMPING, EPS, MAX_ITER, TOP_K,NUM_BLOCKS,BLOCK_SIZE);
        pr.calculate("Data.txt");
        pr.printTopK();
#endif
    } catch (const std::exception& e) {
        // std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
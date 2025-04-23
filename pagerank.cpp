#include "pagerank.h"


PageRank::PageRank(double damping, double epsilon, int max_iter, int top_k)
    : damping_(damping), epsilon_(epsilon), 
      max_iter_(max_iter), top_k_(top_k) {}

void PageRank::readEdges(const std::string& filename) {
    std::ifstream fin(filename);
    if (!fin) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    int u, v;
    while (fin >> u >> v) {
        auto edge = std::make_pair(u, v);
        if (edges_.find(edge) == edges_.end()) {
            edges_.insert(edge);
            in_links_[v].push_back(u);
            out_degree_[u]++;
            nodes_.insert(u);
            nodes_.insert(v);
        }
    }
    fin.close();
}

void PageRank::handleDeadEnds() {
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

void PageRank::initializeRank() {
    const double init_rank = 1.0 / nodes_.size();
    for (const auto& node : nodes_) {
        pr_[node] = init_rank;
    }
}

void PageRank::normalize() {
    double sum = 0.0;
    for (const auto& kv : pr_) {
        sum += kv.second;
    }
    for (auto& kv : pr_) {
        kv.second /= sum;
    }
}

void PageRank::calculate() {
    initializeRank();
    std::unordered_map<int, double> pr_new;

    for (int iter = 0; iter < max_iter_; ++iter) {
        double diff = 0.0;
        pr_new.clear();

        // 计算新的PageRank值
        for (const auto& node : nodes_) {
            double sum_in = 0.0;
            for (const auto& src : in_links_.at(node)) {
                sum_in += pr_[src] / out_degree_.at(src);
            }
            pr_new[node] = (1.0 - damping_) / nodes_.size() + damping_ * sum_in;
        }

        // 计算差值并更新
        for (const auto& node : nodes_) {
            diff += std::fabs(pr_new[node] - pr_[node]);
            pr_[node] = pr_new[node];
        }

        // 收敛检查
        if (diff < epsilon_) {
            std::cerr << "Converged at iteration: " << iter + 1 
                     << " (diff=" << diff << ")\n";
            break;
        }
    }

    normalize();
}

void PageRank::printTopK() const {
    std::vector<std::pair<int, double>> results;
    results.reserve(pr_.size());
    
    for (const auto& kv : pr_) {
        results.emplace_back(kv.first, kv.second);
    }

    std::sort(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    // 将结果输出到 result.txt 文件
    std::ofstream outFile("result.txt"); // 创建文件输出流
    if (!outFile) {
        std::cerr << "Failed to open result.txt for writing." << std::endl;
        return;
    }

    outFile << "nodeID\tPageRank\n"; // 写入表头
    const int output_size = std::min(top_k_, static_cast<int>(results.size()));
    for (int i = 0; i < output_size; ++i) {
        outFile << results[i].first << "\t" // 将结果写入文件
                << std::fixed << std::setprecision(18)
                << results[i].second << "\n";
    }
    outFile.close(); // 关闭文件流
}

int main() {
    try {
        PageRank pr(0.85, 1e-8, 100, 100);
        pr.readEdges("Data.txt");
        //pr.handleDeadEnds();  // 单独处理死节点 但是没有必要  调用会增加很多内存和时间开销
        pr.calculate();
        pr.printTopK();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
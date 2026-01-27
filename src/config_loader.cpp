#include "config_types.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

// 简单的字符串辅助函数
std::string clean_str(std::string s) {
    s.erase(remove(s.begin(), s.end(), '\"'), s.end());
    s.erase(remove(s.begin(), s.end(), ','), s.end());
    s.erase(remove(s.begin(), s.end(), ':'), s.end());
    s.erase(remove(s.begin(), s.end(), '}'), s.end());
    s.erase(remove(s.begin(), s.end(), '{'), s.end());
    return s;
}

// 极其简易的解析器：逐行查找关键词
// 注意：这只是为了演示无依赖环境下的方案。生产环境请使用 nlohmann/json。
SimulationConfig load_config(const std::string& filename) {
    SimulationConfig cfg;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open config file " << filename << std::endl;
        exit(1);
    }

    std::string line, key;
    while (file >> key) {
        key = clean_str(key);
        
        // --- General ---
        if (key == "case_name") file >> cfg.case_name;
        
        // --- Source ---
        else if (key == "source_type") file >> cfg.source.type;
        else if (key == "frequency") file >> cfg.source.frequency;
        else if (key == "src_pos") {
            file >> cfg.source.position[0] >> cfg.source.position[1] >> cfg.source.position[2];
        }
        else if (key == "src_dir") {
            file >> cfg.source.direction[0] >> cfg.source.direction[1] >> cfg.source.direction[2];
        }
        else if (key == "amplitude") file >> cfg.source.amplitude;

        // --- Model ---
        else if (key == "model_type") file >> cfg.model.type;
        else if (key == "bg_rho") file >> cfg.model.bg_rho;
        else if (key == "pml_layers") file >> cfg.model.pml_layers;

        // --- Grid ---
        else if (key == "nx") file >> cfg.grid.nx;
        else if (key == "ny") file >> cfg.grid.ny;
        else if (key == "nz") file >> cfg.grid.nz;
        else if (key == "dx_min") file >> cfg.grid.dx_min;
        else if (key == "stretch_ratio") file >> cfg.grid.stretch_ratio;

        // --- Receiver ---
        else if (key == "receiver_type") file >> cfg.receiver.type;
        else if (key == "output_file") file >> cfg.receiver.output_file;
        else if (key == "receiver_points") {
             // 简易处理：假设只有两个点用于测试
             double x, y, z;
             // Read until "]" or specific count. Simplified for demo:
             // 实际上这需要复杂的 parser。这里我们先留空，在 main 中硬编码测试列表
        }
        
        // --- Validation ---
        else if (key == "validation_enable") {
            std::string val; file >> val;
            cfg.validation.enabled = (val == "true");
        }
    }
    
    // 清理字符串残留
    cfg.case_name = clean_str(cfg.case_name);
    cfg.source.type = clean_str(cfg.source.type);
    cfg.model.type = clean_str(cfg.model.type);
    cfg.receiver.type = clean_str(cfg.receiver.type);
    cfg.receiver.output_file = clean_str(cfg.receiver.output_file);
    
    return cfg;
}
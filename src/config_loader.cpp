#include "config_types.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 使用 nlohmann/json 解析 JSON 配置
SimulationConfig load_config(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + filename);
    }
    
    json j;
    file >> j;
    file.close();
    
    SimulationConfig cfg;
    
    // 解析网格参数
    if (j.contains("grid")) {
        auto& g = j["grid"];
        cfg.grid.nx = g.value("nx", 64);
        cfg.grid.ny = g.value("ny", 64);
        cfg.grid.nz = g.value("nz", 64);
        cfg.grid.dx_min = g.value("dx_min", 0.5);
        cfg.grid.stretch_ratio = g.value("stretch_ratio", 1.1);
    }
    
    // 解析源项参数
    if (j.contains("source")) {
        auto& s = j["source"];
        cfg.source.type = s.value("type", "magnetic_dipole");
        cfg.source.frequency = s.value("frequency", 500.0);
        cfg.source.amplitude = s.value("amplitude", 1.0);
        if (s.contains("position")) cfg.source.position = s["position"].get<std::vector<double>>();
        if (s.contains("direction")) cfg.source.direction = s["direction"].get<std::vector<double>>();
    }
    
    // 解析模型参数
    if (j.contains("model")) {
        auto& m = j["model"];
        cfg.model.bg_rho = m.value("bg_rho", 1.0);
        cfg.model.pml_layers = m.value("pml_layers", 8);
    }
    
    // 解析接收器参数
    if (j.contains("receiver")) {
        auto& r = j["receiver"];
        cfg.receiver.type = r.value("type", "full_grid");
        cfg.receiver.output_file = r.value("output_file", "output");
    }
    
    // 解析验证参数
    if (j.contains("validation")) {
        auto& v = j["validation"];
        cfg.validation.enabled = v.value("enabled", false);
        cfg.validation.mode = v.value("mode", "analytical");
    }
    
    return cfg;
}
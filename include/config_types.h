#pragma once
#include <string>
#include <vector>
#include <array>

// 1. 源配置
struct SourceConfig {
    std::string type; // "electric_dipole" or "magnetic_dipole"
    double frequency;
    std::array<double, 3> position; // [x, y, z] (m)
    std::array<double, 3> direction; // [nx, ny, nz] (Normalized)
    double amplitude;
};

// 2. 接收器配置
struct ReceiverConfig {
    std::string type; // "full_grid" or "point_list"
    std::vector<std::array<double, 3>> points; // Only used if "point_list"
    std::string output_file;
};

// 3. 模型配置
struct ModelConfig {
    std::string type; // "homogeneous" or "voxel"
    // For homogeneous
    double bg_rho; 
    double anisotropy; // rho_v / rho_h
    // For voxel (future expansion)
    std::string voxel_file;
    int pml_layers;
};

// 4. 网格配置
struct GridConfig {
    int nx, ny, nz;
    double dx_min; // Center spacing
    double stretch_ratio; // For non-uniform
};

// 5. 验证配置
struct ValidationConfig {
    bool enabled;
    std::string mode; // "analytical" or "file"
    std::string compare_file;
};

// 总配置
struct SimulationConfig {
    std::string case_name;
    SourceConfig source;
    ReceiverConfig receiver;
    ModelConfig model;
    GridConfig grid;
    ValidationConfig validation;
};
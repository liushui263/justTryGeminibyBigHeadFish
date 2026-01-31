#pragma once
#include <vector>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

struct SourceConfig {
    std::string type;
    double frequency = 0.0;
    std::vector<double> position = {0.0, 0.0, 0.0};
    std::vector<double> direction = {0.0, 0.0, 1.0};
    double amplitude = 1.0;
};

struct GridConfig {
    int nx = 64, ny = 64, nz = 64;
    double dx_min = 0.5;
    double stretch_ratio = 1.1;
};

struct ModelConfig {
    double bg_rho = 1.0;
    int pml_layers = 8;
};

struct ReceiverConfig {
    std::string type = "full_grid";
    std::string output_file = "output";
};

struct ValidationConfig {
    bool enabled = false;
    std::string mode = "analytical";
};

struct SimulationConfig {
    GridConfig grid;
    SourceConfig source;
    ModelConfig model;
    ReceiverConfig receiver;
    ValidationConfig validation;
};
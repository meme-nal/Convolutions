#include "conv.hpp"
#include "net.hpp"
#include "lin.hpp"
#include <fstream>

int main() {
  Tensor X = { // 3x5x5
    {{50.f, 50.f, 50.f, 50.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 50.f, 50.f, 50.f, 50.f}},

    {{50.f, 50.f, 50.f, 50.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 50.f, 50.f, 50.f, 50.f}},

    {{50.f, 50.f, 50.f, 50.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 200.f, 200.f, 200.f, 50.f},
     {50.f, 50.f, 50.f, 50.f, 50.f}}};

  Tensor Y = {
    {{3.f, 0.f},
     {0.f, 3.f}}};

  /// READING CONFIG ///
  const std::string netConfigStr = "../net_cnf.json";

  std::ifstream netConfigFile(netConfigStr);
  auto netConfig = json::parse(netConfigFile);

  netConfigFile.close();

  /// CREATING MODEL ///
  net* model {new net(netConfig)};

  /// TRAINING MODEL ///
  const size_t epochs {1};
  std::vector<double> losses;
  for (size_t epoch {0}; epoch < epochs; ++epoch) {
    // FORWARD PASS //
    Tensor prediction = model->forward(X);
    double loss = MSE(prediction, Y);
    losses.push_back(loss);
  }
  
  for (const auto& loss : losses) {
    std::cout << loss << '\n';
  }

  delete model;

  return 0;
}
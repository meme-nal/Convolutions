#include "conv.hpp"

int main() {
  Tensor X = {
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

  Tensor W = {
    {{0.f, -1.f, 0.f},
     {-1.f, 5.f, -1.f},
     {0.f, -1.f, 0.f}},
     
    {{0.f, -1.f, 0.f},
     {-1.f, 5.f, -1.f},
     {0.f, -1.f, 0.f}},

    {{0.f, -1.f, 0.f},
     {-1.f, 5.f, -1.f},
     {0.f, -1.f, 0.f}}};

  std::cout << "X shape: " << X.size() << '\t' << X[0].size() << '\t' << X[0][0].size() << '\n';
  std::cout << "W shape: " << W.size() << '\t' << W[0].size() << '\t' << W[0][0].size() << '\n';

  Tensor Z = Conv3d(X, W);

  std::cout << "Z shape: " << Z.size() << '\t' << Z[0].size() << '\t' << Z[0][0].size() << '\n';

  for (size_t i {0}; i < Z.size(); ++i) {
    for (size_t j {0}; j < Z[0].size(); ++j) {
      for (size_t k {0}; k < Z[0][0].size(); ++k) {
        std::cout << Z[i][j][k] << '\t';
      }
      std::cout << '\n';
    }
    std::cout << '\n';
  }

  return 0;
}
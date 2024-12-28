#include "conv.hpp"

int main() {
  Matrix X = {{50.f, 50.f, 50.f, 50.f},
              {50.f, 150.f, 150.f, 50.f},
              {50.f, 150.f, 150.f, 50.f},
              {50.f, 50.f, 50.f, 50.f}};

  Matrix W = {{0.f, 0.f, 0.f},
              {0.f, 1.f, 0.f},
              {0.f, 0.f, 0.f}};

  Matrix Z = Conv2d(X, W);

  for (size_t i {0}; i < Z.size(); ++i) {
    for (size_t j {0}; j < Z[0].size(); ++j) {
      std::cout << Z[i][j] << '\t'; 
    }
    std::cout << '\n';
  }

  return 0;
}
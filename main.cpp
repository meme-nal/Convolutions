#include "conv.hpp"

int main() {
  Matrix X = {{1.f}, {2.f}, {3.f}, {4.f}, {5.f}};
  Matrix W = {{1.f}, {0.f}, {1.f}};

  Matrix Z = Conv1d(X, W);

  for (size_t i {0}; i < Z.size(); ++i) {
    for (size_t j {0}; j < Z[0].size(); ++j) {
      std::cout << Z[i][j] << '\t'; 
    }
    std::cout << '\n';
  }

  return 0;
}
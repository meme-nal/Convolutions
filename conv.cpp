#include "conv.hpp"

Matrix Conv1d(const Matrix& X, const Matrix& F) {
  Matrix res = Matrix(X.size() - F.size() + 1, std::vector<float>(X[0].size()));

  for (size_t i {0}; i < res.size(); ++i) {
    float tmp {0.f};
    for (size_t n {0}; n < F.size(); ++n) {
      tmp += X[i+n][0] * F[n][0];
    }
    res[i][0] = tmp;
  }

  return res;
}
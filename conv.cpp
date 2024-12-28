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

Matrix Conv2d(const Matrix& X, const Matrix& F) {
  Matrix res = Matrix(X.size() - F.size() + 1, std::vector<float>(X[0].size() - F[0].size() + 1));

  for (size_t i {0}; i < res.size(); ++i) {
    for (size_t j {0}; j < res[0].size(); ++j) {
      float tmp {0.f};

      for (size_t n {0}; n < F.size(); ++n) {
        for (size_t m {0}; m < F[0].size(); ++m) {
          tmp += X[i+n][j+m] * F[n][m];
        }
      }
      res[i][j] = tmp;
    } 
  }

  return res;
}
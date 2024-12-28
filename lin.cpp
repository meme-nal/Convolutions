#include "lin.hpp"

Matrix add(const Matrix& A, const Matrix& B) {
  // check if matrices are the same size
  if (A.size() != B.size() && A[0].size() != B[0].size()) {
    throw std::runtime_error("Matrices are not the same size");
  }

  Matrix res(A.size(), std::vector<float>(A[0].size()));

  for (size_t i {0}; i < res.size(); ++i) {
    for (size_t j {0}; j < res[0].size(); ++j) {
      res[i][j] = A[i][j] + B[i][j];
    }
  }
  return res;  
}

Tensor add(const Tensor& A, const Tensor& B) {
  // check if matrices are the same size
  if (    A.size() != B.size() && 
       A[0].size() != B[0].size() &&
    A[0][0].size() != B[0][0].size()) {
    throw std::runtime_error("Tensors are not the same size");
  }

  Tensor res(A.size(), Matrix(A[0].size(), std::vector<float>(A[0][0].size())));

  for (size_t i {0}; i < res.size(); ++i) {
    for (size_t j {0}; j < res[0].size(); ++j) {
      for (size_t k {0}; k < res[0][0].size(); ++k) {
        res[i][j][k] = A[i][j][k] + B[i][j][k];
      }
    }
  }
  return res;
}

Matrix sub(const Matrix& A, const Matrix& B) {
  // check if matrices are the same size
  if (A.size() != B.size() && A[0].size() != B[0].size()) {
    throw std::runtime_error("Matrices are not the same size");
  }

  Matrix res(A.size(), std::vector<float>(A[0].size()));

  for (size_t i {0}; i < res.size(); ++i) {
    for (size_t j {0}; j < res[0].size(); ++j) {
      res[i][j] = A[i][j] - B[i][j];
    }
  }
  return res;  
}

Tensor sub(const Tensor& A, const Tensor& B) {
  // check if matrices are the same size
  if (    A.size() != B.size() && 
       A[0].size() != B[0].size() &&
    A[0][0].size() != B[0][0].size()) {
    throw std::runtime_error("Tensors are not the same size");
  }

  Tensor res(A.size(), Matrix(A[0].size(), std::vector<float>(A[0][0].size())));

  for (size_t i {0}; i < res.size(); ++i) {
    for (size_t j {0}; j < res[0].size(); ++j) {
      for (size_t k {0}; k < res[0][0].size(); ++k) {
        res[i][j][k] = A[i][j][k] - B[i][j][k];
      }
    }
  }
  return res;
}

std::ostream& operator << (std::ostream& out, Matrix& m) {
  out << '[';
  for (size_t i {0}; i < m.size(); ++i) {
    out << "[ ";
    for (size_t j {0}; j < m[0].size(); ++j) { 
      if (j != m[0].size() - 1) {
        out << m[i][j] << ", ";
      } else {
        out << m[i][j] << " ";
      }
      
    }
    if (i != m.size() - 1) {
      out << "]\n";
    } else {
      out << ']';
    }
  }
  out << ']';
  return out;
}
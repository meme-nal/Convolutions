#ifndef LIN_HPP
#define LIN_HPP

#include "conv.hpp"

Matrix add(const Matrix& A, const Matrix& B);
Tensor add(const Tensor& A, const Tensor& B);

Matrix sub(const Matrix& A, const Matrix& B);
Tensor sub(const Tensor& A, const Tensor& B);

std::ostream& operator << (std::ostream& out, Matrix& m);

#endif // LIN_HPP
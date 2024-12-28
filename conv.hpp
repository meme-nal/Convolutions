#ifndef CONV_HPP
#define CONV_HPP

#include <iostream>
#include <vector>
#include <memory>

using Scalar = float;
using Matrix = std::vector<std::vector<float>>;
using Tensor = std::vector<Matrix>;

Matrix Conv1d(const Matrix& X, const Matrix& F);
Matrix Conv2d(const Matrix& X, const Matrix& F);
Tensor Conv3d(const Tensor& X, const Tensor& F);

#endif // CONV_HPP
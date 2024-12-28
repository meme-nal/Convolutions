#ifndef CONV_HPP
#define CONV_HPP

#include <iostream>
#include <vector>
#include <memory>

using Matrix = std::vector<std::vector<float>>;
using Tensor = std::vector<std::vector<std::vector<float>>>;

Matrix Conv1d(const Matrix& X, const Matrix& F);
//Matrix& Conv2d(const Matrix& X, const Matrix& F);
//Matrix& Conv3d(const Matrix& X, const Matrix& F);

#endif // CONV_HPP
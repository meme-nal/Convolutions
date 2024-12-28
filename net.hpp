#ifndef NET_HPP
#define NET_HPP

#include "conv.hpp"
#include "lin.hpp"
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

/// ACTIVATION FUNCTIONS ///
Matrix ReLU(const Matrix& Z);
Tensor ReLU(const Tensor& Z);
Matrix Linear(const Matrix& Z);
Tensor Linear(const Tensor& Z);

/// LOSS FUNCTIONS ///
float MSE(const Matrix& pr, const Matrix& gt);
float MSE(const Tensor& pr, const Tensor& gt);

class Layer {
public:
  Layer() = default;
  Layer(const Layer& layer) = default;
  virtual ~Layer() {};

public:
  virtual Tensor forward(Tensor) = 0;
  virtual void backprop() = 0;
};

class Conv : public Layer {
private:
  Tensor _W;
  Tensor _B;
  Tensor _X;
  std::string _act;
  std::string _winit;

public:
  Conv(const json& layer_node);
  Conv(const Conv& conv) = default;

private:
  void winit(const std::string type);

public:
  Tensor forward(Tensor) override;
  void backprop() override;
};

class net {
private:
  std::vector<std::shared_ptr<Layer>> _layers;
  std::string _loss;
  float _lr;
  size_t _miniBatchSize;
  std::string _optimizer;

  Tensor _prediction;
  Tensor _gt;

public:
  net(const json& net_cnf);
  net(const net& nnet) = default;

public:
  Tensor forward(Tensor);
  void backprop();
};

#endif // NET_HPP
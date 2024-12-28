#ifndef NET_HPP
#define NET_HPP

#include "conv.hpp"
#include <nlohmann/json.hpp>
#include <string>
using json = nlohmann::json;

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
  Scalar __lr;
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
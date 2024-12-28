#include "net.hpp"

Tensor Conv::forward(Tensor X) {
  Tensor Z = add(Conv3d(X, _W), _B);
  Tensor a;

  if (this->_act == "ReLU") {
    a = ReLU(Z);
  } else if (this->_act == "Linear") {
    a = Linear(Z);
  } else {
    throw std::runtime_error("Incorrect type of activation function");
  }

  return a; 
}

Tensor net::forward(Tensor X) {
  for (const auto& layer : this->_layers) {
    X = layer->forward(X);
  }

  return X;
}

void Conv::backprop() {
  std::cout << "BACKPROPAGATION\n";
}

void net::backprop() {
  for (const auto& layer : this->_layers) {
    layer->backprop();
  }
}

std::shared_ptr<Layer> getLayerImpl(const json& arch_node, const std::string& layerName) {
  const std::string type = arch_node[layerName]["type"].get<std::string>();
  if (type == "Conv") {
    return std::make_shared<Conv>(arch_node[layerName]);
  } else {
    throw std::runtime_error("Incorrect type of layer");
  }
}

net::net(const json& net_cnf) {
  const size_t layersNum {net_cnf["net"]["arch"]["to_use"].size()};

  _lr = net_cnf["net"]["optim"]["lr"].get<float>();
  _loss = net_cnf["net"]["optim"]["loss"].get<std::string>();
  _optimizer = net_cnf["net"]["optim"]["optimizer"].get<std::string>();
  _miniBatchSize = net_cnf["net"]["optim"]["miniBatchSize"].get<size_t>();

  const auto& layers = net_cnf["net"]["arch"]["to_use"];

  for (size_t i {0}; i < layersNum; ++i) {
    _layers.emplace_back(getLayerImpl(net_cnf["net"]["arch"], layers[i]));
  }
}

Conv::Conv(const json& layer_node) {
  _act = layer_node["act"].get<std::string>();
  _winit = layer_node["winit"].get<std::string>();

  _W = Tensor(layer_node["W_shape"][0].get<size_t>(), Matrix(layer_node["W_shape"][1].get<size_t>(), std::vector<float>(layer_node["W_shape"][2].get<size_t>())));
  _B = Tensor(layer_node["B_shape"][0].get<size_t>(), Matrix(layer_node["B_shape"][1].get<size_t>(), std::vector<float>(layer_node["B_shape"][2].get<size_t>())));

  winit(_winit);
}

void Conv::winit(const std::string type) {
  if (type == "one") {
    for (size_t i {0}; i < _W.size(); ++i) {
      for (size_t j {0}; j < _W[0].size(); ++j) {
        for (size_t k {0}; k < _W[0][0].size(); ++k) {
          _W[i][j][k] = 1.f;
        }
      }
    }
    for (size_t i {0}; i < _B.size(); ++i) {
      for (size_t j {0}; j < _B[0].size(); ++j) {
        for (size_t k {0}; k < _B[0][0].size(); ++k) {
          _B[i][j][k] = 1.f;
        }
      }
    }
  } else if (type == "zero") {
    return; // weights are equal zero by default
  } else {
    throw std::runtime_error("Incorrect type of weights initialization");
  }
}

/// ACTIVATION FUNCTION ///
Matrix ReLU(const Matrix& Z) {
  Matrix res(Z.size(), std::vector<float>(Z[0].size()));

  for (size_t i {0}; i < res.size(); ++i) {
    for (size_t j {0}; j < res[0].size(); ++j) {
      res[i][j] = (Z[i][j] < 0.0 ? 0.0 : Z[i][j]);
    }
  }

  return res;
}

Tensor ReLU(const Tensor& Z) {
  Tensor res(Z.size(), Matrix(Z[0].size(), std::vector<float>(Z[0][0].size())));

  for (size_t i {0}; i < res.size(); ++i) {
    for (size_t j {0}; j < res[0].size(); ++j) {
      for (size_t k {0}; k < res[0][0].size(); ++k) {
        res[i][j][k] = (Z[i][j][k] < 0.f ? 0.f : Z[i][j][k]);
      }
    }
  }

  return res;
}

Matrix Linear(const Matrix& Z) {
  return Z;
}

Tensor Linear(const Tensor& Z) {
  return Z;
}

/// LOSS FUNCTIONS ///
float MSE(const Matrix& pr, const Matrix& gt) {
  float total_loss {0.f};
  for (size_t i {0}; i < pr.size(); ++i) {
    float loss {0.f};
    for (size_t j {0}; j < pr[0].size(); ++j) {
      loss += (pr[i][j] - gt[i][j]) * (pr[i][j] - gt[i][j]);
    }
    total_loss += loss / pr[0].size();
  }

  return total_loss / pr.size();
}

float MSE(const Tensor& pr, const Tensor& gt) {
  float total_loss {0.f};
  for (size_t i {0}; i < pr.size(); ++i) {
    total_loss += MSE(pr[i], gt[i]);
  }
  return total_loss / pr.size();
}
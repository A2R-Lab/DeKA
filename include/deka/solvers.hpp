#pragma once

#include <Eigen/Dense>

namespace deka {
namespace solvers {

class RecursiveKaczmarz {
 public:
  RecursiveKaczmarz(int max_iter);

  RecursiveKaczmarz(
    int max_iter,
    double tolerance,
    double alpha
  );

  ~RecursiveKaczmarz();

  Eigen::VectorXd iterate(
    const Eigen::MatrixXd& A,
    const Eigen::VectorXd& b,
    const Eigen::VectorXd& x0
  ) const;

 private:
  int max_iter;
  double tolerance;
  double alpha;
};

}
}

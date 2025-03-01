#include <deka/solvers.hpp>
#include <deka/util.hpp>

#include <Eigen/Dense>

namespace deka {
namespace solvers {

RecursiveKaczmarz::RecursiveKaczmarz(
  int max_iter
) : max_iter(max_iter), tolerance(0.01), alpha(0.99) {}

RecursiveKaczmarz::RecursiveKaczmarz(
  int max_iter,
  double tolerance,
  double alpha
) : max_iter(max_iter), tolerance(tolerance), alpha(alpha) {}

RecursiveKaczmarz::~RecursiveKaczmarz() = default;

Eigen::VectorXd RecursiveKaczmarz::iterate(
  const Eigen::MatrixXd& A,
  const Eigen::VectorXd& b,
  const Eigen::VectorXd& x0
) const {
  assert(x0.rows() == A.cols() && b.rows() == A.rows());

  Eigen::VectorXd x = x0;
  int m = x.size();

  for (int iter = 0; iter < max_iter; iter++) {
    Eigen::VectorXd residual = A * x - b;

    // avg. per-value abs. tolerance
    if (residual.cwiseAbs().sum() / m < this->tolerance) break;

    // Mask rows and choose select row using weighted norm
    Eigen::ArrayXd row_norms = A.rowwise().squaredNorm();
    Eigen::ArrayXd mask = (row_norms.array() > 1e-6).cast<double>();
    Eigen::ArrayXd weights = mask * (this->alpha * row_norms).exp();
    weights = weights / weights.sum();

    auto pmf = std::vector<double>(weights.data(), weights.data() + weights.size());    
    int i = deka::util::choice(pmf);
    Eigen::VectorXd A_i = A.row(i);
    double b_i = b(i);
    
    x += ((b_i - A_i.dot(x)) / A_i.squaredNorm()) * A_i;
  }

  return x;
}

}
}

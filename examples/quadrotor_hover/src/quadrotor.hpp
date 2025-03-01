#pragma once

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

using namespace autodiff;

namespace constants {

constexpr double mass = 0.035;
constexpr double g = 9.81;
constexpr double thrust_to_torque = 0.0008;
constexpr double arm_len = 0.046 / 1.414213562;
constexpr double scale = 65535;
constexpr double thrust_coeff = 2.245365e-6 * scale;
constexpr double moment_coeff = thrust_coeff * thrust_to_torque;
constexpr int n_states = 13;
constexpr int n_controls = 4;

const static Eigen::Matrix3d J (
  (Eigen::Matrix3d() <<
    1.66e-5, 0.83e-6, 0.72e-6,
    0.83e-6, 1.66e-5, 1.80e-6,
    0.72e-6, 1.80e-6, 2.93e-5
  ).finished()
);

}

class Quadrotor {
 public:
  Quadrotor();

  ~Quadrotor();

  VectorXreal quad_dynamics(
    const VectorXreal& x,
    const VectorXreal& u,
    const real& mass,
    const real& g
  );

  VectorXreal quad_dynamics_rk4(
    const VectorXreal& x,
    const VectorXreal& u,
    const real& mass,
    const real& g
  );

 private:
  Eigen::Matrix3d hat(const Eigen::Vector3d& v);

  Eigen::MatrixXd L(const Eigen::VectorXd& q);

  Eigen::MatrixXd qtoQ(const Eigen::VectorXd& q);

  Eigen::MatrixXd G(const Eigen::VectorXd& q);

  Eigen::MatrixXd rptoq(const Eigen::VectorXd& phi);

  // Eigen::Matrix
};

